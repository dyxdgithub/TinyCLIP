"""Fine-tune TinyCLIP with Fourier samples and projected-hypersphere SupCon.

The prepared pair table and Fourier preprocessing follow
``train_n2_fourier_hard_negative.py``. Each pair is expanded into two labeled
image-text samples: the original anchor and the Fourier-augmented hard
negative. Training batches contain exactly two images per selected leaf class,
as in DPHM. The objective is TinyCLIP ClipLoss plus a weighted supervised
contrastive loss whose logits are negative projected-hypersphere distances.
"""

import argparse
import functools
import json
import math
import random
from contextlib import nullcontext
from pathlib import Path

import torch
import torch.nn.functional as functional
from PIL import Image
from torch import nn
from torch.utils.data import DataLoader, Dataset, Sampler
from tqdm import tqdm

import train_n2_fourier_hard_negative as base


DEFAULT_OUTPUT_DIR = (
    Path(__file__).resolve().parent
    / "output"
    / "fourier_projected_hypersphere_tinyclip_vit_40m_32_text_19m"
)
CHECKPOINT_OBJECTIVE = "clip_plus_projected_hypersphere_supcon_v1"


class FourierLabeledSampleDataset(Dataset):
    """Expose both sides of each pair as independent labeled samples."""

    def __init__(self, pairs, transform, phi, size_policy):
        self.pairs = pairs
        self.transform = transform
        self.phi = phi
        self.size_policy = size_policy
        self._augmentor = None
        self.labels = []
        for pair in pairs:
            self.labels.extend((pair.anchor_leaf, pair.negative_leaf))

    def __len__(self):
        return 2 * len(self.pairs)

    def __getstate__(self):
        state = self.__dict__.copy()
        state["_augmentor"] = None
        return state

    def _get_augmentor(self):
        if self._augmentor is None:
            self._augmentor = base.FourierAugmentor(
                phi=self.phi,
                size_policy=self.size_policy,
            )
        return self._augmentor

    def __getitem__(self, index):
        pair = self.pairs[index // 2]
        if index % 2 == 0:
            with Image.open(pair.anchor_path) as image_file:
                image = image_file.convert("RGB")
            return self.transform(image), pair.anchor_caption, pair.anchor_leaf

        with Image.open(pair.anchor_path) as anchor_file:
            anchor = anchor_file.convert("RGB")
        with Image.open(pair.negative_path) as negative_file:
            negative = negative_file.convert("RGB")
        augmented_negative = self._get_augmentor()(anchor, negative)
        return (
            self.transform(augmented_negative),
            pair.negative_caption,
            pair.negative_leaf,
        )


class TwoPerClassBatchSampler(Sampler):
    """Build deterministic epoch batches with two samples from every class."""

    def __init__(self, labels, classes_per_batch, epoch, seed, steps=None):
        self.classes_per_batch = classes_per_batch
        self.epoch = epoch
        self.seed = seed
        indices_by_label = {}
        for index, label in enumerate(labels):
            indices_by_label.setdefault(label, []).append(index)
        self.indices_by_label = {
            label: indices
            for label, indices in indices_by_label.items()
            if len(indices) >= 2
        }
        if len(self.indices_by_label) < classes_per_batch:
            raise ValueError(
                "Balanced sampling needs at least {} leaf classes with two "
                "samples, but only {} are available".format(
                    classes_per_batch, len(self.indices_by_label)
                )
            )
        eligible_samples = sum(len(indices) for indices in self.indices_by_label.values())
        automatic_steps = max(
            1, math.ceil(eligible_samples / (2 * classes_per_batch))
        )
        self.steps = automatic_steps if steps is None else steps

    def __len__(self):
        return self.steps

    def __iter__(self):
        generator = random.Random(self.seed + self.epoch * 100003)
        labels = sorted(self.indices_by_label)
        queues = {}
        positions = {}

        def refill(label):
            queue = list(self.indices_by_label[label])
            generator.shuffle(queue)
            queues[label] = queue
            positions[label] = 0

        def take_two(label):
            if label not in queues or positions[label] + 2 > len(queues[label]):
                refill(label)
            start = positions[label]
            positions[label] += 2
            return queues[label][start : start + 2]

        for _ in range(self.steps):
            selected_labels = generator.sample(labels, self.classes_per_batch)
            batch = []
            for label in selected_labels:
                batch.extend(take_two(label))
            generator.shuffle(batch)
            yield batch


def infer_clip_embedding_dimension(model):
    projection = getattr(model, "text_projection", None)
    if projection is not None and hasattr(projection, "shape"):
        return int(projection.shape[-1])
    visual = getattr(model, "visual", None)
    output_dim = getattr(visual, "output_dim", None)
    if output_dim is not None:
        return int(output_dim)
    raise ValueError("Unable to infer TinyCLIP embedding dimension")


class ProjectedHypersphereModel(nn.Module):
    """TinyCLIP plus the paper's trainable Euclidean projection head."""

    def __init__(self, clip_model, input_dim, projection_dim):
        super().__init__()
        self.clip_model = clip_model
        self.metric_projection = nn.Linear(input_dim, projection_dim)

    def forward(self, images, tokens):
        image_features, text_features, logit_scale = self.clip_model(
            images, tokens, normalized=False
        )
        metric_features = self.metric_projection(image_features)
        return (
            functional.normalize(image_features, dim=-1),
            functional.normalize(text_features, dim=-1),
            logit_scale,
            metric_features,
        )


def exponential_map_at_origin(vectors, curvature, epsilon):
    """Map Euclidean vectors into projected-hypersphere coordinates."""
    vectors = vectors.float()
    sqrt_curvature = math.sqrt(curvature)
    norms = vectors.norm(dim=-1, keepdim=True).clamp_min(epsilon)
    maximum_norm = (math.pi / 2.0 - epsilon) / sqrt_curvature
    safe_norms = norms.clamp_max(maximum_norm)
    scale = torch.tan(sqrt_curvature * safe_norms) / (
        sqrt_curvature * safe_norms
    )
    return vectors * (safe_norms / norms) * scale


def projected_hypersphere_distances(points, curvature, epsilon):
    """Return all-pairs geodesic distances from Eq. (5) of DPHM."""
    points = points.float()
    squared_norms = points.square().sum(dim=-1)
    squared_differences = (
        points[:, None, :] - points[None, :, :]
    ).square().sum(dim=-1)
    denominator = (
        (1.0 + curvature * squared_norms[:, None])
        * (1.0 + curvature * squared_norms[None, :])
    ).clamp_min(epsilon)
    cosine_argument = 1.0 - 2.0 * curvature * squared_differences / denominator
    cosine_argument = cosine_argument.clamp(-1.0 + epsilon, 1.0 - epsilon)
    distances = torch.acos(cosine_argument) / math.sqrt(curvature)
    identity = torch.eye(
        points.shape[0], device=points.device, dtype=torch.bool
    )
    return distances.masked_fill(identity, 0.0)


def projected_hypersphere_supcon_loss(
    euclidean_features,
    labels,
    curvature,
    temperature,
    epsilon,
):
    """Compute the paper's symmetric supervised contrastive objective."""
    projected = exponential_map_at_origin(
        euclidean_features, curvature, epsilon
    )
    distances = projected_hypersphere_distances(projected, curvature, epsilon)
    label_to_id = {label: index for index, label in enumerate(dict.fromkeys(labels))}
    label_ids = torch.tensor(
        [label_to_id[label] for label in labels],
        device=distances.device,
        dtype=torch.long,
    )
    identity = torch.eye(len(labels), device=distances.device, dtype=torch.bool)
    positive_mask = label_ids[:, None].eq(label_ids[None, :]) & ~identity
    valid_anchors = positive_mask.any(dim=1)
    if not valid_anchors.any():
        return distances.sum() * 0.0, distances.detach(), 0

    logits = -distances / temperature
    logits = logits.masked_fill(identity, float("-inf"))
    log_probabilities = logits - torch.logsumexp(logits, dim=1, keepdim=True)
    positive_counts = positive_mask.sum(dim=1).clamp_min(1)
    per_anchor = -(
        log_probabilities.masked_fill(~positive_mask, 0.0).sum(dim=1)
        / positive_counts
    )
    loss = per_anchor[valid_anchors].mean()
    return loss, distances.detach(), int(valid_anchors.sum().item())


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pairs-csv", type=Path, default=base.DEFAULT_PAIRS_CSV, help="Prepared globally ImageID-unique pair CSV. Default: %(default)s")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR, help="Directory for logs and checkpoints. Default: %(default)s")
    parser.add_argument("--model", default="TinyCLIP-ViT-40M-32-Text-19M", help="TinyCLIP model configuration name. Default: %(default)s")
    parser.add_argument("--pretrained", default="LAION400M", help="Registered pretrained tag or local checkpoint. Default: %(default)s")
    parser.add_argument("--cache-dir", type=Path, default=None, help="Optional pretrained checkpoint cache. Default: open_clip cache")
    parser.add_argument("--fine-tune-mode", choices=("full", "lora"), default="lora", help="Train both towers fully or with LoRA. The metric projection is always trained. Default: %(default)s")
    parser.add_argument("--lora-rank", type=int, default=16, help="LoRA rank. Default: %(default)s")
    parser.add_argument("--lora-alpha", type=float, default=32.0, help="LoRA scaling numerator. Default: %(default)s")
    parser.add_argument("--lora-dropout", type=float, default=0.05, help="LoRA dropout in [0, 1). Default: %(default)s")
    parser.add_argument("--epochs", type=int, default=10, help="Total epochs. Default: %(default)s")
    parser.add_argument("--batch-size", type=int, default=32, help="Image count per batch; must be even because each selected class contributes two images. Default: %(default)s")
    parser.add_argument("--steps-per-epoch", type=int, default=None, help="Balanced batches per epoch. Default: ceil(eligible samples / batch size)")
    parser.add_argument("--learning-rate", type=float, default=None, help="AdamW learning rate. Default: 1e-5 for full and 1e-4 for LoRA")
    parser.add_argument("--weight-decay", type=float, default=0.2, help="AdamW matrix-parameter decay. Default: %(default)s")
    parser.add_argument("--beta1", type=float, default=0.9, help="AdamW beta1. Default: %(default)s")
    parser.add_argument("--beta2", type=float, default=0.98, help="AdamW beta2. Default: %(default)s")
    parser.add_argument("--eps", type=float, default=1e-6, help="AdamW epsilon. Default: %(default)s")
    parser.add_argument("--warmup-steps", type=int, default=200, help="Linear warmup steps. Default: %(default)s")
    parser.add_argument("--supcon-weight", type=float, default=1.0, help="Weight lambda in ClipLoss + lambda * projected-hypersphere SupCon. Default: %(default)s")
    parser.add_argument("--curvature", type=float, default=0.1, help="Positive projected-hypersphere curvature K. The paper's common setting is 0.1. Default: %(default)s")
    parser.add_argument("--temperature", type=float, default=0.2, help="Projected-hypersphere supervised contrastive temperature. The paper uses 0.2. Default: %(default)s")
    parser.add_argument("--projection-dim", type=int, default=128, help="FC projection dimension before exponential mapping. Default: %(default)s")
    parser.add_argument("--geometry-epsilon", type=float, default=1e-6, help="Numerical clamp used by tan and acos operations. Default: %(default)s")
    parser.add_argument("--fourier-phi", type=float, default=0.1, help="Centered low-frequency side-length proportion. Range: (0, 1]. Default: %(default)s")
    parser.add_argument("--fourier-size-policy", choices=("anchor-to-negative", "error"), default="anchor-to-negative", help="Fourier behavior for unequal image sizes. Default: %(default)s")
    parser.add_argument("--validation-fraction", type=float, default=0.1, help="Fraction of pairs reserved for validation. Range: [0, 1). Default: %(default)s")
    parser.add_argument("--seed", type=int, default=42, help="Random seed. Default: %(default)s")
    parser.add_argument("--num-workers", type=int, default=4, help="DataLoader workers; use 0 for in-process loading. Default: %(default)s")
    parser.add_argument("--device", default="cuda", help="Training device. Default: %(default)s")
    parser.add_argument("--gpu", type=int, choices=(0, 1, 2, 3), default=0, help="CUDA GPU index. Choices: 0, 1, 2, 3. Default: %(default)s")
    parser.add_argument("--precision", choices=("amp", "amp_bfloat16", "fp32"), default="amp", help="CUDA precision. Default: %(default)s")
    parser.add_argument("--grad-clip-norm", type=float, default=1.0, help="Maximum gradient norm; 0 disables clipping. Default: %(default)s")
    parser.add_argument("--checkpoint-steps", type=int, default=500, help="Save last.pt every N optimizer steps. Default: %(default)s")
    parser.add_argument("--save-every", type=int, default=1, help="Save an epoch checkpoint every N epochs. Default: %(default)s")
    parser.add_argument("--validate-every", type=int, default=1, help="Validate every N epochs; 0 disables validation. Default: %(default)s")
    parser.add_argument("--log-every", type=int, default=20, help="Log every N optimizer steps. Default: %(default)s")
    parser.add_argument("--tensorboard-dir", type=Path, default=None, help="TensorBoard directory. Default: <output-dir>/tensorboard")
    parser.add_argument("--progress-refresh-seconds", type=float, default=0.1, help="Minimum tqdm refresh interval. Default: %(default)s")
    parser.add_argument("--resume", type=Path, default=None, help="Checkpoint from this script; restores model, optimizer, scheduler, scaler, RNG, epoch, and batch. Default: disabled")
    args = parser.parse_args()

    positive_integer_names = (
        "lora_rank", "epochs", "batch_size", "warmup_steps",
        "checkpoint_steps", "save_every", "log_every", "projection_dim",
    )
    for name in positive_integer_names:
        if getattr(args, name) <= 0:
            parser.error("--{} must be positive".format(name.replace("_", "-")))
    if args.num_workers < 0:
        parser.error("--num-workers must be zero or positive")
    if args.batch_size % 2:
        parser.error("--batch-size must be even")
    if args.steps_per_epoch is not None and args.steps_per_epoch <= 0:
        parser.error("--steps-per-epoch must be positive")
    if args.validate_every < 0:
        parser.error("--validate-every must be zero or positive")
    if args.learning_rate is None:
        args.learning_rate = 1e-5 if args.fine_tune_mode == "full" else 1e-4
    if args.learning_rate <= 0.0 or args.weight_decay < 0.0:
        parser.error("--learning-rate must be positive and --weight-decay nonnegative")
    if not 0.0 <= args.lora_dropout < 1.0:
        parser.error("--lora-dropout must be in [0, 1)")
    if args.lora_alpha <= 0.0 or args.supcon_weight < 0.0:
        parser.error("--lora-alpha must be positive and --supcon-weight nonnegative")
    if args.curvature <= 0.0 or args.temperature <= 0.0:
        parser.error("--curvature and --temperature must be positive")
    if not 0.0 < args.geometry_epsilon < 0.1:
        parser.error("--geometry-epsilon must be in (0, 0.1)")
    if not 0.0 < args.fourier_phi <= 1.0:
        parser.error("--fourier-phi must be in (0, 1]")
    if not 0.0 <= args.validation_fraction < 1.0:
        parser.error("--validation-fraction must be in [0, 1)")
    if args.grad_clip_norm < 0.0 or args.progress_refresh_seconds <= 0.0:
        parser.error("--grad-clip-norm must be nonnegative and progress refresh positive")
    return args


def make_loader(pairs, transform, args, epoch, validation=False):
    dataset = FourierLabeledSampleDataset(
        pairs, transform, args.fourier_phi, args.fourier_size_policy
    )
    eligible_class_count = len(
        {
            label
            for label in dataset.labels
            if dataset.labels.count(label) >= 2
        }
    )
    if validation and eligible_class_count == 0:
        return None
    classes_per_batch = args.batch_size // 2
    if validation:
        classes_per_batch = min(classes_per_batch, eligible_class_count)
    sampler = TwoPerClassBatchSampler(
        dataset.labels,
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


def batch_losses(model, clip_loss_fn, tokenizer, batch, args, device, autocast):
    images, captions, labels = batch
    images = images.to(device, non_blocking=True)
    tokens = tokenizer(list(captions)).to(device, non_blocking=True)
    labels = list(labels)
    with autocast:
        image_features, text_features, logit_scale, metric_features = model(
            images, tokens
        )
        clip_loss = clip_loss_fn(image_features, text_features, logit_scale)
        supcon_loss, distances, valid_anchors = projected_hypersphere_supcon_loss(
            metric_features,
            labels,
            args.curvature,
            args.temperature,
            args.geometry_epsilon,
        )
    identity = torch.eye(len(labels), device=distances.device, dtype=torch.bool)
    label_equal = torch.tensor(
        [[left == right for right in labels] for left in labels],
        device=distances.device,
        dtype=torch.bool,
    )
    positive_distances = distances[label_equal & ~identity]
    negative_distances = distances[~label_equal]
    metrics = {
        "positive_distance": positive_distances.mean().item(),
        "negative_distance": negative_distances.mean().item(),
        "valid_supcon_anchors": valid_anchors,
        "sample_count": len(labels),
    }
    return clip_loss, supcon_loss, metrics


def geometry_config(args):
    return {
        "objective": CHECKPOINT_OBJECTIVE,
        "curvature": args.curvature,
        "temperature": args.temperature,
        "projection_dim": args.projection_dim,
        "fourier_phi": args.fourier_phi,
        "fourier_size_policy": args.fourier_size_policy,
    }


def checkpoint_payload(model, optimizer, scheduler, scaler, epoch, next_batch, global_step, pairs_hash, args):
    payload = base.checkpoint_payload(
        model, optimizer, scheduler, scaler, epoch, next_batch,
        global_step, pairs_hash, args
    )
    payload["geometry_config"] = geometry_config(args)
    return payload


def restore_checkpoint(path, model, optimizer, scheduler, scaler, pairs_hash, args, device):
    checkpoint = torch.load(path, map_location=device)
    if checkpoint.get("geometry_config") != geometry_config(args):
        raise ValueError("Checkpoint geometry configuration does not match current arguments")
    return base.restore_checkpoint(
        path, model, optimizer, scheduler, scaler, pairs_hash, args, device
    )


def aggregate_metrics(total, clip_loss, supcon_loss, metrics, weight):
    count = metrics["sample_count"]
    total["samples"] += count
    total["loss"] += (clip_loss.item() + weight * supcon_loss.item()) * count
    total["clip_loss"] += clip_loss.item() * count
    total["supcon_loss"] += supcon_loss.item() * count
    total["positive_distance"] += metrics["positive_distance"] * count
    total["negative_distance"] += metrics["negative_distance"] * count
    total["valid_supcon_anchors"] += metrics["valid_supcon_anchors"]


def finish_metrics(total):
    samples = total["samples"]
    return {
        "loss": total["loss"] / samples,
        "clip_loss": total["clip_loss"] / samples,
        "supcon_loss": total["supcon_loss"] / samples,
        "positive_distance": total["positive_distance"] / samples,
        "negative_distance": total["negative_distance"] / samples,
        "valid_supcon_anchor_fraction": total["valid_supcon_anchors"] / samples,
    }


def empty_totals():
    return {
        "samples": 0,
        "loss": 0.0,
        "clip_loss": 0.0,
        "supcon_loss": 0.0,
        "positive_distance": 0.0,
        "negative_distance": 0.0,
        "valid_supcon_anchors": 0,
    }


@torch.no_grad()
def run_validation(model, pairs, transform, tokenizer, clip_loss_fn, args, device, autocast):
    if not pairs:
        return None
    loader = make_loader(pairs, transform, args, epoch=0, validation=True)
    if loader is None:
        return None
    totals = empty_totals()
    model.eval()
    with tqdm(total=len(loader), desc="Validation", unit="batch", mininterval=args.progress_refresh_seconds) as progress:
        for batch in loader:
            clip_loss, supcon_loss, metrics = batch_losses(
                model, clip_loss_fn, tokenizer, batch, args, device, autocast
            )
            aggregate_metrics(totals, clip_loss, supcon_loss, metrics, args.supcon_weight)
            progress.update(1)
    return finish_metrics(totals)


def main():
    args = parse_args()
    pairs_csv = args.pairs_csv.expanduser()
    output_dir = args.output_dir.expanduser()
    tensorboard_dir = (
        args.tensorboard_dir.expanduser()
        if args.tensorboard_dir
        else output_dir / "tensorboard"
    )
    if not pairs_csv.is_file():
        raise FileNotFoundError("Prepared pair CSV does not exist: {}".format(pairs_csv))
    if args.resume is not None and not args.resume.expanduser().is_file():
        raise FileNotFoundError("Resume checkpoint does not exist: {}".format(args.resume))

    device = base.ensure_device(args.device, args.gpu)
    print("Training device: {}".format(device), flush=True)
    base.seed_everything(args.seed)
    pairs = base.load_prepared_pairs(pairs_csv, args)
    pairs_hash = base.pair_signature(pairs)
    train_pairs, validation_pairs = base.split_pairs(
        pairs, args.validation_fraction, args.seed
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    run_metadata = {
        "arguments": vars(args),
        "objective": CHECKPOINT_OBJECTIVE,
        "pair_signature": pairs_hash,
        "selected_pairs": len(pairs),
        "train_pairs": len(train_pairs),
        "validation_pairs": len(validation_pairs),
        "pairs_csv": str(pairs_csv.resolve()),
    }
    with (output_dir / "run_config.json").open("w", encoding="utf-8") as handle:
        json.dump(run_metadata, handle, ensure_ascii=True, indent=2, default=str)
    writer = base.create_tensorboard_writer(tensorboard_dir, args, run_metadata)

    cache_dir = str(args.cache_dir.expanduser()) if args.cache_dir else None
    clip_model, train_transform, validation_transform = base.create_model_and_transforms(
        args.model,
        pretrained=args.pretrained,
        precision="fp32",
        device=device,
        cache_dir=cache_dir,
    )
    lora_summary = base.configure_trainable_parameters(clip_model, args)
    embedding_dim = infer_clip_embedding_dimension(clip_model)
    model = ProjectedHypersphereModel(
        clip_model, embedding_dim, args.projection_dim
    ).to(device)
    optimizer = base.make_optimizer(model, args)
    initial_loader = make_loader(
        train_pairs, train_transform, args, epoch=0, validation=False
    )
    total_steps = max(1, len(initial_loader) * args.epochs)
    scheduler = base.make_scheduler(optimizer, total_steps, args.warmup_steps)
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
            args.resume.expanduser(), model, optimizer, scheduler, scaler,
            pairs_hash, args, device
        )
        print(
            "Resumed from {} at epoch {}, next batch {}, global step {}".format(
                args.resume.expanduser(), start_epoch + 1,
                start_batch + 1, global_step
            ),
            flush=True,
        )

    trainable_parameters = sum(
        parameter.numel() for parameter in model.parameters()
        if parameter.requires_grad
    )
    all_parameters = sum(parameter.numel() for parameter in model.parameters())
    print("Prepared pairs: {}".format(len(pairs)), flush=True)
    print("Train pairs: {}; validation pairs: {}".format(len(train_pairs), len(validation_pairs)), flush=True)
    print("Balanced batch: {} classes x 2 images".format(args.batch_size // 2), flush=True)
    print("Curvature: {}; temperature: {}; projection dim: {}".format(args.curvature, args.temperature, args.projection_dim), flush=True)
    print("Fine-tune mode: {}; adapter summary: {}".format(args.fine_tune_mode, lora_summary), flush=True)
    print("Trainable parameters: {} / {}".format(trainable_parameters, all_parameters), flush=True)

    best_validation_loss = float("inf")
    last_checkpoint = output_dir / "last.pt"
    try:
        for epoch in range(start_epoch, args.epochs):
            loader = make_loader(
                train_pairs, train_transform, args, epoch=epoch,
                validation=False
            )
            model.train()
            totals = empty_totals()
            skipped_batches = start_batch if epoch == start_epoch else 0
            with tqdm(total=len(loader), desc="Epoch {}/{}".format(epoch + 1, args.epochs), unit="batch", mininterval=args.progress_refresh_seconds) as progress:
                if skipped_batches:
                    progress.update(skipped_batches)
                for batch_index, batch in enumerate(loader):
                    if batch_index < skipped_batches:
                        continue
                    optimizer.zero_grad(set_to_none=True)
                    clip_loss, supcon_loss, batch_metrics = batch_losses(
                        model, clip_loss_fn, tokenizer, batch, args,
                        device, autocast
                    )
                    total_loss = clip_loss + args.supcon_weight * supcon_loss
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
                    aggregate_metrics(
                        totals, clip_loss, supcon_loss, batch_metrics,
                        args.supcon_weight
                    )
                    progress.update(1)

                    if global_step % args.log_every == 0:
                        step_metrics = {
                            "loss": total_loss.item(),
                            "clip_loss": clip_loss.item(),
                            "supcon_loss": supcon_loss.item(),
                            "positive_distance": batch_metrics["positive_distance"],
                            "negative_distance": batch_metrics["negative_distance"],
                            "valid_supcon_anchor_fraction": batch_metrics["valid_supcon_anchors"] / batch_metrics["sample_count"],
                            "learning_rate": optimizer.param_groups[0]["lr"],
                        }
                        progress.set_postfix(
                            loss="{:.4f}".format(step_metrics["loss"]),
                            clip="{:.4f}".format(step_metrics["clip_loss"]),
                            supcon="{:.4f}".format(step_metrics["supcon_loss"]),
                            pos_d="{:.3f}".format(step_metrics["positive_distance"]),
                            neg_d="{:.3f}".format(step_metrics["negative_distance"]),
                        )
                        base.append_metric(metrics_path, {
                            "split": "train_step",
                            "epoch": epoch + 1,
                            "global_step": global_step,
                            **step_metrics,
                        })
                        base.write_tensorboard_scalars(
                            writer, "train_step", step_metrics, global_step
                        )
                    if global_step % args.checkpoint_steps == 0:
                        base.save_checkpoint(
                            last_checkpoint,
                            checkpoint_payload(
                                model, optimizer, scheduler, scaler, epoch,
                                batch_index + 1, global_step, pairs_hash, args
                            ),
                        )

            start_batch = 0
            train_metrics = finish_metrics(totals)
            base.append_metric(metrics_path, {
                "split": "train_epoch", "epoch": epoch + 1,
                "global_step": global_step, **train_metrics,
            })
            base.write_tensorboard_scalars(
                writer, "train_epoch", train_metrics, epoch + 1
            )
            validation_metrics = None
            if validation_pairs and args.validate_every and (epoch + 1) % args.validate_every == 0:
                validation_metrics = run_validation(
                    model, validation_pairs, validation_transform, tokenizer,
                    clip_loss_fn, args, device, autocast
                )
                if validation_metrics is not None:
                    base.append_metric(metrics_path, {
                        "split": "validation", "epoch": epoch + 1,
                        "global_step": global_step, **validation_metrics,
                    })
                    base.write_tensorboard_scalars(
                        writer, "validation", validation_metrics, epoch + 1
                    )
                    print("Epoch {} validation: {}".format(epoch + 1, validation_metrics), flush=True)
                    if validation_metrics["loss"] < best_validation_loss:
                        best_validation_loss = validation_metrics["loss"]
                        base.save_checkpoint(
                            output_dir / "best.pt",
                            checkpoint_payload(
                                model, optimizer, scheduler, scaler, epoch + 1,
                                0, global_step, pairs_hash, args
                            ),
                        )
            if (epoch + 1) % args.save_every == 0:
                base.save_checkpoint(
                    output_dir / "checkpoints" / "epoch_{:03d}.pt".format(epoch + 1),
                    checkpoint_payload(
                        model, optimizer, scheduler, scaler, epoch + 1,
                        0, global_step, pairs_hash, args
                    ),
                )
            base.save_checkpoint(
                last_checkpoint,
                checkpoint_payload(
                    model, optimizer, scheduler, scaler, epoch + 1,
                    0, global_step, pairs_hash, args
                ),
            )
            print("Epoch {} train: {}".format(epoch + 1, train_metrics), flush=True)
    finally:
        writer.close()

    print("Training complete. Last checkpoint: {}".format(last_checkpoint.resolve()))
    print("TensorBoard logs: {}".format(tensorboard_dir.resolve()))
    if validation_pairs:
        print("Best validation checkpoint: {}".format((output_dir / "best.pt").resolve()))
    print("Training pair table: {}".format(pairs_csv.resolve()))


if __name__ == "__main__":
    main()
