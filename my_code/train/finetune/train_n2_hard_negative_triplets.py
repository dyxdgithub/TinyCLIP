"""Train TinyCLIP from offline DINOv2 hard-negative triplets.

This entry point is intentionally separate from train_n2_fourier_hard_negative.py.
It reads hard_negative_triplets_dinov2_vitb14.csv, obtains captions and all
classification labels from the Open Images n2 manifest, applies Fourier
augmentation to each negative image, and optimizes TinyCLIP ClipLoss plus the
paper's Selectively Contrastive Triplet (SCT) loss.
"""

import argparse
import csv
import functools
import importlib.util
import json
import math
import random
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as functional
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
SRC_DIR = REPOSITORY_ROOT / "src"
FOURIER_AUGMENTOR_PATH = REPOSITORY_ROOT / "my_code" / "module" / "fourier_augmentor.py"
META_DIR = REPOSITORY_ROOT / "my_code" / "data" / "OpenImage" / "meta"
N2_DIR = META_DIR / "Hierarchy" / "n2"
DEFAULT_TRIPLETS = N2_DIR / "hard_negative_triplets" / "hard_negative_triplets_dinov2_vitb14.csv"
DEFAULT_MANIFEST = N2_DIR / "Image IDs_with_last_two_layers_multi_members_n2_one_label_per_parent.csv"
DEFAULT_OUTPUT = Path(__file__).resolve().parent / "output" / "n2_hard_negative_triplets_tinyclip"
for path in (SRC_DIR, Path(__file__).resolve().parent):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from open_clip import ClipLoss, create_model_and_transforms, get_tokenizer
from train_n2_fourier_hard_negative import (
    LoRALinear,
    LoRAMultiheadAttention,
    autocast_context,
    configure_trainable_parameters,
    create_tensorboard_writer,
    ensure_device,
    make_optimizer,
    make_scheduler,
    replace_submodule,
    seed_everything,
)


def load_augmentor():
    spec = importlib.util.spec_from_file_location("fourier_augmentor_for_triplets", FOURIER_AUGMENTOR_PATH)
    if spec is None or spec.loader is None:
        raise ImportError("Unable to load FourierAugmentor from {}".format(FOURIER_AUGMENTOR_PATH))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.FourierAugmentor


FourierAugmentor = load_augmentor()


@dataclass(frozen=True)
class Triplet:
    anchor_id: str
    anchor_path: Path
    positive_id: str
    positive_path: Path
    negative_id: str
    negative_path: Path
    similarity: float


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--triplets-csv", type=Path, default=DEFAULT_TRIPLETS, help="Offline hard-negative triplet CSV. Default: %(default)s")
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST, help="Open Images n2 manifest providing captions and labels. Default: %(default)s")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT, help="Training output and TensorBoard directory. Default: %(default)s")
    parser.add_argument("--model", default="TinyCLIP-ViT-40M-32-Text-19M", help="TinyCLIP model name. Default: %(default)s")
    parser.add_argument("--pretrained", default="LAION400M", help="Registered pretrained tag or local checkpoint. Default: %(default)s")
    parser.add_argument("--cache-dir", type=Path, default=None, help="Optional model cache directory. Default: open_clip default")
    parser.add_argument("--fine-tune-mode", choices=("full", "lora"), default="lora", help="Train all parameters or LoRA adapters. Default: lora")
    parser.add_argument("--lora-rank", type=int, default=16, help="LoRA rank. Default: 16")
    parser.add_argument("--lora-alpha", type=float, default=32.0, help="LoRA alpha. Default: 32")
    parser.add_argument("--lora-dropout", type=float, default=0.05, help="LoRA dropout. Default: 0.05")
    parser.add_argument("--epochs", type=int, default=10, help="Training epochs. Default: 10")
    parser.add_argument("--batch-size", type=int, default=16, help="Triplets per batch; three image-text items per triplet. Default: 16")
    parser.add_argument("--learning-rate", type=float, default=None, help="AdamW learning rate. Default: 1e-5 full, 1e-4 lora")
    parser.add_argument("--weight-decay", type=float, default=0.2, help="AdamW weight decay. Default: 0.2")
    parser.add_argument("--beta1", type=float, default=0.9, help="AdamW beta1. Default: 0.9")
    parser.add_argument("--beta2", type=float, default=0.98, help="AdamW beta2. Default: 0.98")
    parser.add_argument("--eps", type=float, default=1e-6, help="AdamW epsilon. Default: 1e-6")
    parser.add_argument("--warmup-steps", type=int, default=200, help="Linear warmup steps. Default: 200")
    parser.add_argument("--clip-loss-weight", type=float, default=1.0, help="Weight of TinyCLIP ClipLoss. Default: 1")
    parser.add_argument("--sct-loss-weight", type=float, default=1.0, help="Weight of paper SCT loss. Default: 1")
    parser.add_argument("--sct-lambda", type=float, default=1.0, help="SCT hard-negative weight lambda. Default: 1")
    parser.add_argument("--temperature", type=float, default=0.1, help="SCT NCA temperature for non-hard triplets. Default: 0.1")
    parser.add_argument("--fourier-phi", type=float, default=0.1, help="Fourier low-frequency ratio. Default: 0.1")
    parser.add_argument("--fourier-size-policy", choices=("anchor-to-negative", "error"), default="anchor-to-negative", help="Fourier size policy. Default: anchor-to-negative")
    parser.add_argument("--validation-fraction", type=float, default=0.1, help="Validation fraction of triplets. Default: 0.1")
    parser.add_argument("--validate-every", type=int, default=1, help="Validate every N epochs. Default: 1")
    parser.add_argument("--num-workers", type=int, default=4, help="DataLoader workers; use 0 if worker spawning fails. Default: 4")
    parser.add_argument("--device", default="cuda", help="Device. Default: cuda")
    parser.add_argument("--gpu", type=int, default=0, help="CUDA device index. Default: 0")
    parser.add_argument("--precision", choices=("amp", "amp_bfloat16", "fp32"), default="amp", help="Autocast precision. Default: amp")
    parser.add_argument("--seed", type=int, default=42, help="Random seed. Default: 42")
    parser.add_argument("--resume", type=Path, default=None, help="Checkpoint to resume. Default: disabled")
    parser.add_argument("--tensorboard-dir", type=Path, default=None, help="TensorBoard directory. Default: <output-dir>/tensorboard")
    parser.add_argument("--progress-refresh-seconds", type=float, default=0.1, help="tqdm refresh interval. Default: 0.1")
    args = parser.parse_args()
    if args.learning_rate is None:
        args.learning_rate = 1e-5 if args.fine_tune_mode == "full" else 1e-4
    if args.batch_size <= 0 or args.epochs <= 0 or args.num_workers < 0 or args.validate_every < 0:
        parser.error("batch-size and epochs must be positive; num-workers and validate-every must be nonnegative")
    if args.sct_lambda < 0 or args.sct_loss_weight < 0 or args.clip_loss_weight < 0 or args.temperature <= 0:
        parser.error("loss weights must be nonnegative and temperature must be positive")
    if not 0.0 < args.beta1 < 1.0 or not 0.0 < args.beta2 < 1.0 or args.eps <= 0.0:
        parser.error("beta1 and beta2 must be in (0, 1), and eps must be positive")
    return args


def longest_caption_manifest(path):
    values = {}
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        required = {"ImageID", "caption", "LabelName"}
        missing = required.difference(reader.fieldnames or [])
        if missing:
            raise ValueError("Manifest lacks columns: {}".format(", ".join(sorted(missing))))
        with tqdm(desc="Reading captions and labels", unit="row") as progress:
            for row in reader:
                image_id = (row.get("ImageID") or "").strip()
                if not image_id:
                    progress.update(1)
                    continue
                item = values.setdefault(image_id, {"caption": "", "labels": set()})
                caption = (row.get("caption") or "").strip()
                if len(caption) > len(item["caption"]):
                    item["caption"] = caption
                label = (row.get("LabelName") or "").strip()
                if label:
                    item["labels"].add(label)
                progress.update(1)
    return values


def load_triplets(path, manifest_values):
    required = {"Status", "AnchorImageID", "AnchorImagePath", "PositiveImageID", "PositiveImagePath", "HardNegativeImageID", "HardNegativeImagePath", "HardNegativeSimilarity"}
    rows = []
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        missing = required.difference(reader.fieldnames or [])
        if missing:
            raise ValueError("Triplet CSV lacks columns: {}".format(", ".join(sorted(missing))))
        with tqdm(desc="Reading triplets", unit="triplet") as progress:
            for row_number, row in enumerate(reader, start=2):
                if (row.get("Status") or "").strip() != "ok":
                    progress.update(1)
                    continue
                ids = [(row.get(name) or "").strip() for name in ("AnchorImageID", "PositiveImageID", "HardNegativeImageID")]
                paths = [Path((row.get(name) or "").strip()) for name in ("AnchorImagePath", "PositiveImagePath", "HardNegativeImagePath")]
                if any(not value for value in ids) or len(set(ids)) != 3:
                    raise ValueError("Invalid triplet IDs at row {}".format(row_number))
                if any(not path.is_file() for path in paths):
                    raise FileNotFoundError("Missing triplet image at row {}: {}".format(row_number, paths))
                if any(image_id not in manifest_values for image_id in ids):
                    raise ValueError("Triplet ImageID missing from manifest at row {}".format(row_number))
                if any(not manifest_values[image_id]["caption"] for image_id in ids):
                    raise ValueError("Triplet ImageID has no caption at row {}".format(row_number))
                rows.append(Triplet(ids[0], paths[0].resolve(), ids[1], paths[1].resolve(), ids[2], paths[2].resolve(), float(row["HardNegativeSimilarity"])))
                progress.update(1)
    if not rows:
        raise ValueError("No usable triplets found in {}".format(path))
    return rows


class TripletDataset(Dataset):
    def __init__(self, triplets, manifest_values, transform, phi, size_policy):
        self.triplets = triplets
        self.manifest_values = manifest_values
        self.transform = transform
        self.phi = phi
        self.size_policy = size_policy
        self._augmentor = None

    def __len__(self):
        return len(self.triplets)

    def __getstate__(self):
        state = self.__dict__.copy()
        state["_augmentor"] = None
        return state

    def _get_augmentor(self):
        if self._augmentor is None:
            self._augmentor = FourierAugmentor(phi=self.phi, size_policy=self.size_policy)
        return self._augmentor

    def __getitem__(self, index):
        triplet = self.triplets[index]
        images = []
        for path in (triplet.anchor_path, triplet.positive_path, triplet.negative_path):
            with Image.open(path) as handle:
                images.append(handle.convert("RGB"))
        augmented_negative = self._get_augmentor()(images[0], images[2])
        ids = (triplet.anchor_id, triplet.positive_id, triplet.negative_id)
        tensors = (self.transform(images[0]), self.transform(images[1]), self.transform(augmented_negative))
        captions = tuple(self.manifest_values[image_id]["caption"] for image_id in ids)
        labels = tuple(frozenset(self.manifest_values[image_id]["labels"]) for image_id in ids)
        return tensors + captions + labels + ids


def split_triplets(rows, fraction, seed):
    indices = list(range(len(rows)))
    random.Random(seed).shuffle(indices)
    count = 0 if fraction == 0 else max(1, min(len(rows) - 1, round(len(rows) * fraction)))
    validation = set(indices[:count])
    return [row for i, row in enumerate(rows) if i not in validation], [row for i, row in enumerate(rows) if i in validation]


def make_loader(rows, manifest_values, transform, args, shuffle):
    dataset = TripletDataset(rows, manifest_values, transform, args.fourier_phi, args.fourier_size_policy)
    return DataLoader(dataset, batch_size=args.batch_size, shuffle=shuffle, num_workers=args.num_workers, pin_memory=True, drop_last=False, persistent_workers=False, worker_init_fn=functools.partial(seed_worker, args.seed) if args.num_workers else None)


def seed_worker(base_seed, worker_id):
    seed = base_seed + worker_id
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def compute_losses(model, clip_loss, tokenizer, batch, device, autocast, args):
    anchor, positive, negative, anchor_caption, positive_caption, negative_caption, anchor_labels, positive_labels, negative_labels, anchor_id, positive_id, negative_id = batch
    images = torch.cat((anchor, positive, negative), dim=0).to(device, non_blocking=True)
    captions = list(anchor_caption) + list(positive_caption) + list(negative_caption)
    tokens = tokenizer(captions).to(device, non_blocking=True)
    with autocast:
        image_features, text_features, logit_scale = model(images, tokens, normalized=True)
        clip_value = clip_loss(image_features, text_features, logit_scale)
        anchor_features, positive_features, negative_features = image_features.chunk(3, dim=0)
        positive_similarity = (anchor_features * positive_features).sum(dim=1)
        negative_similarity = (anchor_features * negative_features).sum(dim=1)
        hard = negative_similarity > positive_similarity
        nca = -functional.log_softmax(torch.stack((positive_similarity, negative_similarity), dim=1) / args.temperature, dim=1)[:, 0]
        sct_value = torch.where(hard, args.sct_lambda * negative_similarity, nca).mean()
        total = args.clip_loss_weight * clip_value + args.sct_loss_weight * sct_value
        triplet_accuracy = (positive_similarity > negative_similarity).float().mean()
        negative_separation = (negative_similarity < positive_similarity).float().mean()
    return total, clip_value, sct_value, triplet_accuracy, negative_separation, image_features.detach(), captions, (anchor_labels, positive_labels, negative_labels), (anchor_id, positive_id, negative_id)


@torch.no_grad()
def validation(model, rows, manifest_values, transform, tokenizer, clip_loss, args, device, autocast):
    if not rows:
        return {}
    loader = make_loader(rows, manifest_values, transform, args, False)
    losses = []
    triplet_acc = []
    all_features = {}
    all_labels = {}
    model.eval()
    with tqdm(total=len(loader), desc="Validation", unit="batch") as progress:
        for batch in loader:
            total, clip_value, sct_value, accuracy, separation, features, _, labels, ids = compute_losses(model, clip_loss, tokenizer, batch, device, autocast, args)
            losses.append((total.item(), clip_value.item(), sct_value.item()))
            triplet_acc.append((accuracy.item(), separation.item()))
            batch_ids = sum(ids, ())
            batch_labels = sum(labels, ())
            for position, image_id in enumerate(batch_ids):
                if image_id not in all_features:
                    all_features[image_id] = features[position].float().cpu()
                    all_labels[image_id] = set(batch_labels[position])
            progress.update(1)
    feature_ids = list(all_features)
    feature_matrix = functional.normalize(torch.stack([all_features[i] for i in feature_ids]), dim=1)
    label_sets = [all_labels[i] for i in feature_ids]
    similarities = feature_matrix @ feature_matrix.t()
    similarities.fill_diagonal_(-torch.inf)
    nearest = similarities.argmax(dim=1).tolist()
    label_accuracy = sum(bool(label_sets[i].intersection(label_sets[nearest[i]])) for i in range(len(nearest))) / max(1, len(nearest))
    return {
        "loss": sum(item[0] for item in losses) / len(losses),
        "clip_loss": sum(item[1] for item in losses) / len(losses),
        "sct_loss": sum(item[2] for item in losses) / len(losses),
        "triplet_accuracy": sum(item[0] for item in triplet_acc) / len(triplet_acc),
        "negative_separation_accuracy": sum(item[1] for item in triplet_acc) / len(triplet_acc),
        "label_retrieval_accuracy": label_accuracy,
    }


def main():
    args = parse_args()
    device = ensure_device(args.device, args.gpu)
    seed_everything(args.seed)
    manifest_values = longest_caption_manifest(args.manifest.expanduser())
    triplets = load_triplets(args.triplets_csv.expanduser(), manifest_values)
    train_rows, validation_rows = split_triplets(triplets, args.validation_fraction, args.seed)
    output_dir = args.output_dir.expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)
    tensorboard_dir = args.tensorboard_dir.expanduser() if args.tensorboard_dir else output_dir / "tensorboard"
    metadata = {"arguments": vars(args), "triplets": len(triplets), "train_triplets": len(train_rows), "validation_triplets": len(validation_rows)}
    with (output_dir / "run_config.json").open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2, default=str)
    writer = create_tensorboard_writer(tensorboard_dir, args, metadata)
    cache_dir = str(args.cache_dir.expanduser()) if args.cache_dir else None
    model, train_transform, validation_transform = create_model_and_transforms(args.model, pretrained=args.pretrained, precision="fp32", device=device, cache_dir=cache_dir)
    adapter_summary = configure_trainable_parameters(model, args)
    optimizer = make_optimizer(model, args)
    train_loader = make_loader(train_rows, manifest_values, train_transform, args, True)
    scheduler = make_scheduler(optimizer, max(1, len(train_loader) * args.epochs), args.warmup_steps)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda" and args.precision == "amp")
    tokenizer = get_tokenizer(args.model)
    clip_loss = ClipLoss(cache_labels=True)
    autocast = autocast_context(args.precision, device)
    best_accuracy = -1.0
    for epoch in range(args.epochs):
        model.train()
        totals = np.zeros(5, dtype=np.float64)
        with tqdm(total=len(train_loader), desc="Epoch {}/{}".format(epoch + 1, args.epochs), unit="batch") as progress:
            for batch in train_loader:
                optimizer.zero_grad(set_to_none=True)
                total, clip_value, sct_value, accuracy, separation, _, _, _, _ = compute_losses(model, clip_loss, tokenizer, batch, device, autocast, args)
                scaler.scale(total).backward()
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
                values = [total.item(), clip_value.item(), sct_value.item(), accuracy.item(), separation.item()]
                totals += values
                progress.set_postfix(loss="{:.4f}".format(values[0]), triplet_acc="{:.3f}".format(values[3]))
                progress.update(1)
        train_metrics = {"loss": totals[0] / len(train_loader), "clip_loss": totals[1] / len(train_loader), "sct_loss": totals[2] / len(train_loader), "triplet_accuracy": totals[3] / len(train_loader), "negative_separation_accuracy": totals[4] / len(train_loader)}
        for name, value in train_metrics.items():
            writer.add_scalar("train/{}".format(name), value, epoch + 1)
        if validation_rows and args.validate_every and (epoch + 1) % args.validate_every == 0:
            validation_metrics = validation(model, validation_rows, manifest_values, validation_transform, tokenizer, clip_loss, args, device, autocast)
            for name, value in validation_metrics.items():
                writer.add_scalar("validation/{}".format(name), value, epoch + 1)
            writer.flush()
            print("Epoch {} validation: {}".format(epoch + 1, validation_metrics), flush=True)
            if validation_metrics["label_retrieval_accuracy"] > best_accuracy:
                best_accuracy = validation_metrics["label_retrieval_accuracy"]
                torch.save({"model": model.state_dict(), "epoch": epoch + 1, "validation": validation_metrics, "args": vars(args)}, output_dir / "best.pt")
        torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(), "scaler": scaler.state_dict(), "epoch": epoch + 1, "args": vars(args)}, output_dir / "last.pt")
    writer.close()
    print("Training complete. Output: {}".format(output_dir.resolve()))


if __name__ == "__main__":
    main()
