"""Fine-tune TinyCLIP with in-batch hard-negative SCT triplets.

This entry point follows Xuan et al. (ECCV 2020): every batch contains two
images from each randomly selected leaf class. Each image is an anchor, the
other image from its class is the positive, and the most similar image from a
different class in the current batch is the hard negative. The image encoder
is optimized with TinyCLIP ClipLoss plus Selectively Contrastive Triplet (SCT)
loss. No offline triplet table or Fourier augmentation is used.
"""

import argparse
import functools
import hashlib
import json
import math
import random
from pathlib import Path

import torch
import torch.nn.functional as functional
from torch.utils.data import DataLoader, Dataset, Sampler
from tqdm import tqdm

import train_n2_fourier_hard_negative as base
from train_n2_fourier_projected_hypersphere import (
    index_images,
    load_manifest_samples,
    load_rgb_image,
    split_samples_by_image_id,
)


N2_DIR = (
    Path(__file__).resolve().parents[2]
    / "data"
    / "OpenImage"
    / "meta"
    / "Hierarchy"
    / "n2"
)
DEFAULT_MANIFEST = (
    N2_DIR
    / "Image IDs_with_last_two_layers_multi_members_n2_one_label_per_parent.csv"
)
DEFAULT_IMAGES_ROOT = N2_DIR / "images_all"
DEFAULT_OUTPUT_DIR = (
    Path(__file__).resolve().parent
    / "output"
    / "n2_in_batch_hard_negative_triplets_tinyclip"
)
CHECKPOINT_OBJECTIVE = "clip_plus_in_batch_sct_v1"


class LabeledImageDataset(Dataset):
    """Return one manifest-associated image, caption, class, and ImageID."""

    def __init__(self, samples, transform):
        self.samples = samples
        self.transform = transform
        self.labels = [sample.leaf_label for sample in samples]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        sample = self.samples[index]
        image = load_rgb_image(sample.image_path)
        return (
            self.transform(image),
            sample.caption,
            sample.leaf_label,
            sample.image_id,
        )


class UniqueImageTwoPerClassBatchSampler(Sampler):
    """Randomly choose classes and two distinct ImageIDs from every class."""

    def __init__(
        self,
        labels,
        image_ids,
        classes_per_batch,
        epoch,
        seed,
        steps=None,
    ):
        if len(labels) != len(image_ids):
            raise ValueError("labels and image_ids must have the same length")
        self.classes_per_batch = classes_per_batch
        self.epoch = epoch
        self.seed = seed
        indices_by_label = {}
        for index, (label, image_id) in enumerate(zip(labels, image_ids)):
            indices_by_label.setdefault(label, {}).setdefault(image_id, []).append(index)
        self.indices_by_label_and_image = {
            label: indices_by_image
            for label, indices_by_image in indices_by_label.items()
            if len(indices_by_image) >= 2
        }
        if len(self.indices_by_label_and_image) < classes_per_batch:
            raise ValueError(
                "Balanced sampling needs at least {} leaf classes with two "
                "distinct images, but only {} are available".format(
                    classes_per_batch, len(self.indices_by_label_and_image)
                )
            )
        eligible_samples = sum(
            len(indices_by_image)
            for indices_by_image in self.indices_by_label_and_image.values()
        )
        automatic_steps = max(
            1, math.ceil(eligible_samples / (2 * classes_per_batch))
        )
        self.steps = automatic_steps if steps is None else steps

    def __len__(self):
        return self.steps

    def __iter__(self):
        generator = random.Random(self.seed + self.epoch * 100003)
        labels = sorted(self.indices_by_label_and_image)
        for _ in range(self.steps):
            for _attempt in range(100):
                candidate_labels = list(labels)
                generator.shuffle(candidate_labels)
                used_image_ids = set()
                batch = []
                selected_class_count = 0
                for label in candidate_labels:
                    indices_by_image = self.indices_by_label_and_image[label]
                    available_image_ids = [
                        image_id
                        for image_id in indices_by_image
                        if image_id not in used_image_ids
                    ]
                    if len(available_image_ids) < 2:
                        continue
                    selected_image_ids = generator.sample(available_image_ids, 2)
                    for image_id in selected_image_ids:
                        batch.append(generator.choice(indices_by_image[image_id]))
                        used_image_ids.add(image_id)
                    selected_class_count += 1
                    if selected_class_count == self.classes_per_batch:
                        break
                if selected_class_count == self.classes_per_batch:
                    generator.shuffle(batch)
                    yield batch
                    break
            else:
                raise ValueError(
                    "Unable to build a batch with {} classes x 2 unique "
                    "ImageIDs after 100 attempts; reduce --batch-size".format(
                        self.classes_per_batch
                    )
                )


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST, help="Open Images CSV with ImageID, caption, LabelName, and ParentLabelName. Default: %(default)s")
    parser.add_argument("--images-root", type=Path, default=DEFAULT_IMAGES_ROOT, help="Image root recursively organized like n2/images_all. Filenames must contain the ImageID. Default: %(default)s")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR, help="Checkpoint, metric, and run metadata directory. Default: %(default)s")
    parser.add_argument("--model", default="TinyCLIP-ViT-40M-32-Text-19M", help="TinyCLIP model configuration. Default: %(default)s")
    parser.add_argument("--pretrained", default="LAION400M", help="Registered pretrained tag or local checkpoint. Default: %(default)s")
    parser.add_argument("--cache-dir", type=Path, default=None, help="Optional pretrained checkpoint cache. Default: open_clip cache")
    parser.add_argument("--fine-tune-mode", choices=("full", "lora"), default="lora", help="Train all model parameters or LoRA adapters. Default: %(default)s")
    parser.add_argument("--lora-rank", type=int, default=16, help="LoRA rank. Default: %(default)s")
    parser.add_argument("--lora-alpha", type=float, default=32.0, help="LoRA scaling numerator. Default: %(default)s")
    parser.add_argument("--lora-dropout", type=float, default=0.05, help="LoRA dropout in [0, 1). Default: %(default)s")
    parser.add_argument("--epochs", type=int, default=50, help="Total training epochs. Default: %(default)s")
    parser.add_argument("--batch-size", type=int, default=512, help="Images per batch. Must be even; each selected class contributes exactly two images. Default: %(default)s")
    parser.add_argument("--steps-per-epoch", type=int, default=None, help="Balanced batches per epoch. Default: ceil(eligible samples / batch size)")
    parser.add_argument("--learning-rate", type=float, default=None, help="AdamW learning rate. Default: 1e-5 for full tuning and 1e-4 for LoRA")
    parser.add_argument("--weight-decay", type=float, default=0.2, help="AdamW matrix-parameter weight decay. Default: %(default)s")
    parser.add_argument("--beta1", type=float, default=0.9, help="AdamW beta1 in (0, 1). Default: %(default)s")
    parser.add_argument("--beta2", type=float, default=0.98, help="AdamW beta2 in (0, 1). Default: %(default)s")
    parser.add_argument("--eps", type=float, default=1e-6, help="Positive AdamW epsilon. Default: %(default)s")
    parser.add_argument("--warmup-steps", type=int, default=200, help="Linear warmup optimizer steps. Default: %(default)s")
    parser.add_argument("--clip-loss-weight", type=float, default=1.0, help="Nonnegative weight for TinyCLIP ClipLoss. Default: %(default)s")
    parser.add_argument("--sct-loss-weight", type=float, default=1.0, help="Nonnegative weight for in-batch SCT loss. Default: %(default)s")
    parser.add_argument("--sct-lambda", type=float, default=1.0, help="Lambda multiplying San for hard triplets in Eq. (13). Default: %(default)s")
    parser.add_argument("--temperature", type=float, default=1.0, help="Positive NCA logit temperature for non-hard triplets. The paper's Eq. (1) corresponds to 1.0. Default: %(default)s")
    parser.add_argument("--validation-fraction", type=float, default=0.1, help="Fraction of unique ImageIDs reserved for validation, in [0, 1). Default: %(default)s")
    parser.add_argument("--validate-every", type=int, default=1, help="Validate every N epochs; 0 disables validation. Default: %(default)s")
    parser.add_argument("--num-workers", type=int, default=4, help="DataLoader workers; use 0 for in-process loading. Default: %(default)s")
    parser.add_argument("--device", default="cuda", help="Training device, such as cuda or cpu. Default: %(default)s")
    parser.add_argument("--gpu", type=int, default=0, help="Nonnegative CUDA device index used when --device is cuda. Default: %(default)s")
    parser.add_argument("--precision", choices=("amp", "amp_bfloat16", "fp32"), default="amp", help="Training precision. Choices: amp, amp_bfloat16, fp32. Default: %(default)s")
    parser.add_argument("--grad-clip-norm", type=float, default=1.0, help="Maximum gradient norm; 0 disables clipping. Default: %(default)s")
    parser.add_argument("--checkpoint-steps", type=int, default=500, help="Persist last.pt every N optimizer steps. Default: %(default)s")
    parser.add_argument("--save-every", type=int, default=1, help="Save an epoch checkpoint every N epochs. Default: %(default)s")
    parser.add_argument("--log-every", type=int, default=20, help="Write step metrics every N optimizer steps. Default: %(default)s")
    parser.add_argument("--tensorboard-dir", type=Path, default=None, help="TensorBoard directory. Default: <output-dir>/tensorboard")
    parser.add_argument("--progress-refresh-seconds", type=float, default=0.1, help="Positive minimum tqdm refresh interval. Default: %(default)s")
    parser.add_argument("--seed", type=int, default=42, help="Random seed. Default: %(default)s")
    parser.add_argument("--resume", type=Path, default=None, help="Checkpoint from this script; restores model, optimizer, scheduler, scaler, RNG, epoch, and batch. Default: disabled")
    args = parser.parse_args()

    positive_integer_names = (
        "lora_rank",
        "epochs",
        "batch_size",
        "warmup_steps",
        "checkpoint_steps",
        "save_every",
        "log_every",
    )
    for name in positive_integer_names:
        if getattr(args, name) <= 0:
            parser.error("--{} must be positive".format(name.replace("_", "-")))
    if args.batch_size < 4 or args.batch_size % 2:
        parser.error("--batch-size must be even and at least 4")
    if args.steps_per_epoch is not None and args.steps_per_epoch <= 0:
        parser.error("--steps-per-epoch must be positive")
    if args.num_workers < 0 or args.validate_every < 0 or args.gpu < 0:
        parser.error("--num-workers, --validate-every, and --gpu must be nonnegative")
    if args.learning_rate is None:
        args.learning_rate = 1e-5 if args.fine_tune_mode == "full" else 1e-4
    if args.learning_rate <= 0.0 or args.weight_decay < 0.0:
        parser.error("--learning-rate must be positive and --weight-decay nonnegative")
    if not 0.0 < args.beta1 < 1.0 or not 0.0 < args.beta2 < 1.0:
        parser.error("--beta1 and --beta2 must be in (0, 1)")
    if args.eps <= 0.0 or args.lora_alpha <= 0.0:
        parser.error("--eps and --lora-alpha must be positive")
    if not 0.0 <= args.lora_dropout < 1.0:
        parser.error("--lora-dropout must be in [0, 1)")
    if args.clip_loss_weight < 0.0 or args.sct_loss_weight < 0.0:
        parser.error("--clip-loss-weight and --sct-loss-weight must be nonnegative")
    if args.sct_lambda < 0.0 or args.temperature <= 0.0:
        parser.error("--sct-lambda must be nonnegative and --temperature positive")
    if not 0.0 <= args.validation_fraction < 1.0:
        parser.error("--validation-fraction must be in [0, 1)")
    if args.grad_clip_norm < 0.0 or args.progress_refresh_seconds <= 0.0:
        parser.error("--grad-clip-norm must be nonnegative and progress refresh positive")
    return args


def sample_signature(samples):
    payload = [
        {
            "image_id": sample.image_id,
            "image_path": str(sample.image_path),
            "caption": sample.caption,
            "leaf_label": sample.leaf_label,
            "parent_label": sample.parent_label,
        }
        for sample in samples
    ]
    encoded = json.dumps(
        payload, ensure_ascii=True, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def make_loader(samples, transform, args, epoch, validation=False):
    dataset = LabeledImageDataset(samples, transform)
    image_ids = [sample.image_id for sample in samples]
    unique_images_by_label = {}
    for label, image_id in zip(dataset.labels, image_ids):
        unique_images_by_label.setdefault(label, set()).add(image_id)
    eligible_class_count = sum(
        len(unique_image_ids) >= 2
        for unique_image_ids in unique_images_by_label.values()
    )
    if eligible_class_count < 2:
        if validation:
            return None
        raise ValueError(
            "In-batch mining requires at least two leaf classes with two images "
            "each; only {} are available".format(eligible_class_count)
        )

    classes_per_batch = args.batch_size // 2
    if validation:
        classes_per_batch = min(classes_per_batch, eligible_class_count)
    sampler = UniqueImageTwoPerClassBatchSampler(
        dataset.labels,
        image_ids,
        classes_per_batch=classes_per_batch,
        epoch=epoch,
        seed=args.seed + (1 if validation else 0),
        steps=args.steps_per_epoch if not validation else None,
    )
    return DataLoader(
        dataset,
        batch_sampler=sampler,
        num_workers=args.num_workers,
        pin_memory=True,
        persistent_workers=False,
        worker_init_fn=(
            functools.partial(
                base.initialize_data_worker,
                base_seed=args.seed,
                epoch=epoch,
            )
            if args.num_workers
            else None
        ),
    )


def label_ids_from_strings(labels, device):
    label_to_id = {
        label: index for index, label in enumerate(dict.fromkeys(labels))
    }
    return torch.tensor(
        [label_to_id[label] for label in labels],
        device=device,
        dtype=torch.long,
    )


def mine_in_batch_triplets(image_features, labels):
    """Mine one positive and the cosine-hardest negative for every anchor."""
    if image_features.ndim != 2 or image_features.shape[0] != len(labels):
        raise ValueError("image_features and labels must describe the same batch")
    if len(labels) < 4:
        raise ValueError("in-batch triplet mining requires at least four images")

    features = functional.normalize(image_features.float(), dim=-1)
    similarities = features @ features.t()
    label_ids = label_ids_from_strings(labels, similarities.device)
    identity = torch.eye(
        len(labels), device=similarities.device, dtype=torch.bool
    )
    same_class = label_ids[:, None].eq(label_ids[None, :])
    positive_mask = same_class & ~identity
    positive_counts = positive_mask.sum(dim=1)
    if not torch.all(positive_counts == 1):
        raise ValueError(
            "Each batch class must contain exactly two images; positive counts: {}"
            .format(positive_counts.tolist())
        )
    negative_mask = ~same_class
    if not negative_mask.any(dim=1).all():
        raise ValueError("Every anchor needs at least one different-class negative")

    positive_indices = positive_mask.to(torch.int64).argmax(dim=1)
    negative_indices = similarities.masked_fill(
        ~negative_mask, float("-inf")
    ).argmax(dim=1)
    anchor_indices = torch.arange(len(labels), device=similarities.device)
    positive_similarities = similarities[anchor_indices, positive_indices]
    negative_similarities = similarities[anchor_indices, negative_indices]
    return (
        positive_similarities,
        negative_similarities,
        positive_indices,
        negative_indices,
    )


def selectively_contrastive_triplet_loss(
    image_features,
    labels,
    sct_lambda,
    temperature,
):
    """Compute Eq. (13) after mining the hardest negatives in this batch."""
    (
        positive_similarities,
        negative_similarities,
        positive_indices,
        negative_indices,
    ) = mine_in_batch_triplets(image_features, labels)
    hard_mask = negative_similarities > positive_similarities
    nca_logits = torch.stack(
        (positive_similarities, negative_similarities), dim=1
    ) / temperature
    nca_loss = -functional.log_softmax(nca_logits, dim=1)[:, 0]
    per_anchor_loss = torch.where(
        hard_mask,
        sct_lambda * negative_similarities,
        nca_loss,
    )
    metrics = {
        "positive_similarity": positive_similarities.detach().mean().item(),
        "hardest_negative_similarity": negative_similarities.detach().mean().item(),
        "hard_triplet_fraction": hard_mask.detach().float().mean().item(),
        "triplet_accuracy": (
            positive_similarities.detach() > negative_similarities.detach()
        ).float().mean().item(),
        "sample_count": len(labels),
        "positive_indices": positive_indices.detach(),
        "negative_indices": negative_indices.detach(),
    }
    return per_anchor_loss.mean(), metrics


def batch_losses(model, clip_loss_fn, tokenizer, batch, args, device, autocast):
    images, captions, labels, image_ids = batch
    images = images.to(device, non_blocking=True)
    tokens = tokenizer(list(captions)).to(device, non_blocking=True)
    labels = list(labels)
    with autocast:
        image_features, text_features, logit_scale = model(
            images, tokens, normalized=True
        )
        clip_loss = clip_loss_fn(image_features, text_features, logit_scale)
        sct_loss, metrics = selectively_contrastive_triplet_loss(
            image_features,
            labels,
            args.sct_lambda,
            args.temperature,
        )
    metrics["unique_image_fraction"] = len(set(image_ids)) / len(image_ids)
    return clip_loss, sct_loss, metrics


def empty_totals():
    return {
        "samples": 0,
        "loss": 0.0,
        "clip_loss": 0.0,
        "sct_loss": 0.0,
        "positive_similarity": 0.0,
        "hardest_negative_similarity": 0.0,
        "hard_triplet_fraction": 0.0,
        "triplet_accuracy": 0.0,
        "unique_image_fraction": 0.0,
    }


def aggregate_metrics(totals, clip_loss, sct_loss, metrics, args):
    count = metrics["sample_count"]
    total_loss = (
        args.clip_loss_weight * clip_loss
        + args.sct_loss_weight * sct_loss
    )
    totals["samples"] += count
    totals["loss"] += total_loss.item() * count
    totals["clip_loss"] += clip_loss.item() * count
    totals["sct_loss"] += sct_loss.item() * count
    for name in (
        "positive_similarity",
        "hardest_negative_similarity",
        "hard_triplet_fraction",
        "triplet_accuracy",
        "unique_image_fraction",
    ):
        totals[name] += metrics[name] * count
    return total_loss


def finish_metrics(totals):
    samples = totals["samples"]
    if not samples:
        raise ValueError("No samples were processed")
    return {
        name: value / samples
        for name, value in totals.items()
        if name != "samples"
    }


def objective_config(args):
    return {
        "objective": CHECKPOINT_OBJECTIVE,
        "batch_size": args.batch_size,
        "clip_loss_weight": args.clip_loss_weight,
        "sct_loss_weight": args.sct_loss_weight,
        "sct_lambda": args.sct_lambda,
        "temperature": args.temperature,
    }


def checkpoint_payload(
    model,
    optimizer,
    scheduler,
    scaler,
    epoch,
    next_batch,
    global_step,
    data_hash,
    args,
):
    payload = base.checkpoint_payload(
        model,
        optimizer,
        scheduler,
        scaler,
        epoch,
        next_batch,
        global_step,
        data_hash,
        args,
    )
    payload["data_signature"] = data_hash
    payload["objective_config"] = objective_config(args)
    return payload


def restore_checkpoint(
    path,
    model,
    optimizer,
    scheduler,
    scaler,
    data_hash,
    args,
    device,
):
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    if checkpoint.get("objective_config") != objective_config(args):
        raise ValueError(
            "Checkpoint objective configuration does not match current arguments"
        )
    if checkpoint.get("data_signature") != data_hash:
        raise ValueError(
            "Checkpoint data signature does not match current manifest and images"
        )
    del checkpoint
    return base.restore_checkpoint(
        path,
        model,
        optimizer,
        scheduler,
        scaler,
        data_hash,
        args,
        device,
    )


@torch.no_grad()
def run_validation(
    model,
    samples,
    transform,
    tokenizer,
    clip_loss_fn,
    args,
    device,
    autocast,
):
    if not samples:
        return None
    loader = make_loader(
        samples, transform, args, epoch=0, validation=True
    )
    if loader is None:
        return None
    totals = empty_totals()
    model.eval()
    with tqdm(
        total=len(loader),
        desc="Validation",
        unit="batch",
        mininterval=args.progress_refresh_seconds,
    ) as progress:
        for batch in loader:
            clip_loss, sct_loss, metrics = batch_losses(
                model,
                clip_loss_fn,
                tokenizer,
                batch,
                args,
                device,
                autocast,
            )
            aggregate_metrics(totals, clip_loss, sct_loss, metrics, args)
            progress.update(1)
    return finish_metrics(totals)


def main():
    args = parse_args()
    manifest = args.manifest.expanduser()
    images_root = args.images_root.expanduser()
    output_dir = args.output_dir.expanduser()
    tensorboard_dir = (
        args.tensorboard_dir.expanduser()
        if args.tensorboard_dir
        else output_dir / "tensorboard"
    )
    if not manifest.is_file():
        raise FileNotFoundError("Manifest CSV does not exist: {}".format(manifest))
    if not images_root.is_dir():
        raise FileNotFoundError(
            "Image root directory does not exist: {}".format(images_root)
        )
    if args.resume is not None and not args.resume.expanduser().is_file():
        raise FileNotFoundError(
            "Resume checkpoint does not exist: {}".format(args.resume)
        )

    device = base.ensure_device(args.device, args.gpu)
    print("Training device: {}".format(device), flush=True)
    base.seed_everything(args.seed)
    image_paths, duplicate_image_ids = index_images(
        images_root, args.progress_refresh_seconds
    )
    samples, missing_image_ids, skipped_incomplete_rows = load_manifest_samples(
        manifest, image_paths, args.progress_refresh_seconds
    )
    train_samples, validation_samples = split_samples_by_image_id(
        samples, args.validation_fraction, args.seed
    )
    if not train_samples:
        raise ValueError("No samples were assigned to training")

    signature_payload = "{}:{}".format(
        sample_signature(train_samples),
        sample_signature(validation_samples),
    ).encode("ascii")
    data_hash = hashlib.sha256(signature_payload).hexdigest()
    output_dir.mkdir(parents=True, exist_ok=True)
    run_metadata = {
        "arguments": vars(args),
        "objective": CHECKPOINT_OBJECTIVE,
        "data_signature": data_hash,
        "indexed_unique_images": len(image_paths),
        "duplicate_image_files": duplicate_image_ids,
        "manifest_samples_with_images": len(samples),
        "missing_unique_image_ids": len(missing_image_ids),
        "skipped_incomplete_manifest_rows": skipped_incomplete_rows,
        "train_samples": len(train_samples),
        "validation_samples": len(validation_samples),
        "manifest": str(manifest.resolve()),
        "images_root": str(images_root.resolve()),
    }
    with (output_dir / "run_config.json").open("w", encoding="utf-8") as handle:
        json.dump(run_metadata, handle, ensure_ascii=True, indent=2, default=str)
    writer = base.create_tensorboard_writer(
        tensorboard_dir, args, run_metadata
    )

    cache_dir = str(args.cache_dir.expanduser()) if args.cache_dir else None
    model, train_transform, validation_transform = (
        base.create_model_and_transforms(
            args.model,
            pretrained=args.pretrained,
            precision="fp32",
            device=device,
            cache_dir=cache_dir,
        )
    )
    adapter_summary = base.configure_trainable_parameters(model, args)
    optimizer = base.make_optimizer(model, args)
    initial_loader = make_loader(
        train_samples, train_transform, args, epoch=0, validation=False
    )
    total_steps = max(1, len(initial_loader) * args.epochs)
    scheduler = base.make_scheduler(
        optimizer, total_steps, args.warmup_steps
    )
    scaler = torch.amp.GradScaler(
        "cuda", enabled=device.type == "cuda" and args.precision == "amp"
    )
    tokenizer = base.get_tokenizer(args.model)
    clip_loss_fn = base.ClipLoss(cache_labels=True)
    autocast = base.autocast_context(args.precision, device)
    metrics_path = output_dir / "metrics.jsonl"
    start_epoch = start_batch = global_step = 0
    if args.resume is not None:
        start_epoch, start_batch, global_step = restore_checkpoint(
            args.resume.expanduser(),
            model,
            optimizer,
            scheduler,
            scaler,
            data_hash,
            args,
            device,
        )
        print(
            "Resumed from {} at epoch {}, next batch {}, global step {}".format(
                args.resume.expanduser(),
                start_epoch + 1,
                start_batch + 1,
                global_step,
            ),
            flush=True,
        )

    trainable_parameters = sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )
    all_parameters = sum(parameter.numel() for parameter in model.parameters())
    print("Indexed unique images: {}".format(len(image_paths)), flush=True)
    print("Duplicate image files: {}".format(duplicate_image_ids), flush=True)
    print("Manifest samples with images: {}".format(len(samples)), flush=True)
    print("Missing unique ImageIDs: {}".format(len(missing_image_ids)), flush=True)
    print(
        "Train samples: {}; validation samples: {}".format(
            len(train_samples), len(validation_samples)
        ),
        flush=True,
    )
    print(
        "Balanced batch: {} classes x 2 images".format(args.batch_size // 2),
        flush=True,
    )
    print(
        "SCT lambda: {}; NCA temperature: {}".format(
            args.sct_lambda, args.temperature
        ),
        flush=True,
    )
    print(
        "Fine-tune mode: {}; adapter summary: {}".format(
            args.fine_tune_mode, adapter_summary
        ),
        flush=True,
    )
    print(
        "Trainable parameters: {} / {}".format(
            trainable_parameters, all_parameters
        ),
        flush=True,
    )

    best_validation_triplet_accuracy = -1.0
    last_checkpoint = output_dir / "last.pt"
    try:
        for epoch in range(start_epoch, args.epochs):
            loader = make_loader(
                train_samples,
                train_transform,
                args,
                epoch=epoch,
                validation=False,
            )
            model.train()
            totals = empty_totals()
            skipped_batches = start_batch if epoch == start_epoch else 0
            with tqdm(
                total=len(loader),
                desc="Epoch {}/{}".format(epoch + 1, args.epochs),
                unit="batch",
                mininterval=args.progress_refresh_seconds,
            ) as progress:
                if skipped_batches:
                    progress.update(skipped_batches)
                for batch_index, batch in enumerate(loader):
                    if batch_index < skipped_batches:
                        continue
                    optimizer.zero_grad(set_to_none=True)
                    clip_loss, sct_loss, batch_metrics = batch_losses(
                        model,
                        clip_loss_fn,
                        tokenizer,
                        batch,
                        args,
                        device,
                        autocast,
                    )
                    total_loss = aggregate_metrics(
                        totals, clip_loss, sct_loss, batch_metrics, args
                    )
                    scaler.scale(total_loss).backward()
                    if args.grad_clip_norm > 0.0:
                        scaler.unscale_(optimizer)
                        torch.nn.utils.clip_grad_norm_(
                            model.parameters(), args.grad_clip_norm
                        )
                    scaler.step(optimizer)
                    scaler.update()
                    scheduler.step()
                    global_step += 1
                    progress.update(1)

                    if (
                        global_step == 1
                        or global_step % args.log_every == 0
                        or batch_index + 1 == len(loader)
                    ):
                        step_metrics = {
                            "loss": total_loss.item(),
                            "clip_loss": clip_loss.item(),
                            "sct_loss": sct_loss.item(),
                            "positive_similarity": batch_metrics["positive_similarity"],
                            "hardest_negative_similarity": batch_metrics["hardest_negative_similarity"],
                            "hard_triplet_fraction": batch_metrics["hard_triplet_fraction"],
                            "triplet_accuracy": batch_metrics["triplet_accuracy"],
                            "unique_image_fraction": batch_metrics["unique_image_fraction"],
                            "learning_rate": optimizer.param_groups[0]["lr"],
                        }
                        progress.set_postfix(
                            loss="{:.4f}".format(step_metrics["loss"]),
                            sct="{:.4f}".format(step_metrics["sct_loss"]),
                            hard="{:.3f}".format(step_metrics["hard_triplet_fraction"]),
                            acc="{:.3f}".format(step_metrics["triplet_accuracy"]),
                        )
                        base.append_metric(
                            metrics_path,
                            {
                                "split": "train_step",
                                "epoch": epoch + 1,
                                "global_step": global_step,
                                **step_metrics,
                            },
                        )
                        base.write_tensorboard_scalars(
                            writer, "train_step", step_metrics, global_step
                        )
                    if global_step % args.checkpoint_steps == 0:
                        base.save_checkpoint(
                            last_checkpoint,
                            checkpoint_payload(
                                model,
                                optimizer,
                                scheduler,
                                scaler,
                                epoch,
                                batch_index + 1,
                                global_step,
                                data_hash,
                                args,
                            ),
                        )

            start_batch = 0
            train_metrics = finish_metrics(totals)
            base.append_metric(
                metrics_path,
                {
                    "split": "train_epoch",
                    "epoch": epoch + 1,
                    "global_step": global_step,
                    **train_metrics,
                },
            )
            base.write_tensorboard_scalars(
                writer, "train_epoch", train_metrics, epoch + 1
            )
            validation_metrics = None
            if (
                validation_samples
                and args.validate_every
                and (epoch + 1) % args.validate_every == 0
            ):
                validation_metrics = run_validation(
                    model,
                    validation_samples,
                    validation_transform,
                    tokenizer,
                    clip_loss_fn,
                    args,
                    device,
                    autocast,
                )
                if validation_metrics is not None:
                    base.append_metric(
                        metrics_path,
                        {
                            "split": "validation",
                            "epoch": epoch + 1,
                            "global_step": global_step,
                            **validation_metrics,
                        },
                    )
                    base.write_tensorboard_scalars(
                        writer,
                        "validation",
                        validation_metrics,
                        epoch + 1,
                    )
                    print(
                        "Epoch {} validation: {}".format(
                            epoch + 1, validation_metrics
                        ),
                        flush=True,
                    )
                    if (
                        validation_metrics["triplet_accuracy"]
                        > best_validation_triplet_accuracy
                    ):
                        best_validation_triplet_accuracy = validation_metrics[
                            "triplet_accuracy"
                        ]
                        base.save_checkpoint(
                            output_dir / "best.pt",
                            checkpoint_payload(
                                model,
                                optimizer,
                                scheduler,
                                scaler,
                                epoch + 1,
                                0,
                                global_step,
                                data_hash,
                                args,
                            ),
                        )
            if (epoch + 1) % args.save_every == 0:
                base.save_checkpoint(
                    output_dir
                    / "checkpoints"
                    / "epoch_{:03d}.pt".format(epoch + 1),
                    checkpoint_payload(
                        model,
                        optimizer,
                        scheduler,
                        scaler,
                        epoch + 1,
                        0,
                        global_step,
                        data_hash,
                        args,
                    ),
                )
            base.save_checkpoint(
                last_checkpoint,
                checkpoint_payload(
                    model,
                    optimizer,
                    scheduler,
                    scaler,
                    epoch + 1,
                    0,
                    global_step,
                    data_hash,
                    args,
                ),
            )
            print(
                "Epoch {} train: {}".format(epoch + 1, train_metrics),
                flush=True,
            )
    finally:
        writer.close()

    print(
        "Training complete. Last checkpoint: {}".format(
            last_checkpoint.resolve()
        )
    )
    print("TensorBoard logs: {}".format(tensorboard_dir.resolve()))
    if validation_samples:
        print(
            "Best validation checkpoint: {}".format(
                (output_dir / "best.pt").resolve()
            )
        )
    print("Training manifest: {}".format(manifest.resolve()))
    print("Training image root: {}".format(images_root.resolve()))


if __name__ == "__main__":
    main()
