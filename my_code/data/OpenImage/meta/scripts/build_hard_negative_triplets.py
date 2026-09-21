"""Build one offline hard-negative triplet for every unique Open Images image.

The input manifest may contain several rows for the same ImageID because an
image can have multiple n2 leaf labels. This script groups those rows first, so
each unique ImageID is emitted exactly once as an anchor. A positive shares at
least one leaf LabelName with its anchor; a hard negative shares no leaf
LabelName and has the greatest cosine similarity under the selected backbone.

Embeddings are cached as a resumable NumPy memmap and triplets are appended to
an in-progress CSV, so a long run can resume without recomputing completed
image embeddings or anchor rows.
"""

import argparse
import csv
import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as functional
from PIL import Image, ImageOps
from tqdm import tqdm
from torchvision import models


META_DIR = Path(__file__).resolve().parent.parent
N2_DIR = META_DIR / "Hierarchy" / "n2"
DEFAULT_MANIFEST = N2_DIR / "Image IDs_with_last_two_layers_multi_members_n2_one_label_per_parent.csv"
DEFAULT_OUTPUT_DIR = N2_DIR / "hard_negative_triplets"
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".ppm"}
CACHE_VERSION = 1
RESULT_FIELDS = [
    "AnchorIndex",
    "Status",
    "Backbone",
    "NegativeScope",
    "AnchorImageID",
    "AnchorImagePath",
    "AnchorLabelNames",
    "AnchorDisplayNames",
    "AnchorParentLabelNames",
    "AnchorParentDisplayNames",
    "PositiveImageID",
    "PositiveImagePath",
    "PositiveSharedLabelNames",
    "PositiveLabelNames",
    "PositiveDisplayNames",
    "PositiveSimilarity",
    "HardNegativeImageID",
    "HardNegativeImagePath",
    "HardNegativeLabelNames",
    "HardNegativeDisplayNames",
    "HardNegativeSimilarity",
]


@dataclass(frozen=True)
class ImageRecord:
    image_id: str
    path: Path
    label_names: tuple
    display_names: tuple
    parent_label_names: tuple
    parent_display_names: tuple


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST,
                        help="n2 manifest containing ImageID, LabelName, and ParentLabelName. Default: %(default)s")
    parser.add_argument("--images-root", type=Path, required=True,
                        help="Local root that contains downloaded images in any nested layout. Image files are indexed recursively by filename stem (ImageID).")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR,
                        help="Directory for the embedding cache and default output CSV. Default: %(default)s")
    parser.add_argument("--output", type=Path, default=None,
                        help="Triplet CSV path. Default: <output-dir>/hard_negative_triplets_<backbone>.csv")
    parser.add_argument("--backbone", choices=("resnet50", "dinov2"), default="dinov2",
                        help="Embedding backbone. resnet50 uses the paper's ImageNet-pretrained ResNet-50 pooled feature; dinov2 uses --dinov2-model. Default: %(default)s")
    parser.add_argument("--dinov2-model", default="dinov2_vitb14",
                        help="DINOv2 Torch Hub model when --backbone dinov2. Default: %(default)s")
    parser.add_argument("--torch-hub-repo", default="facebookresearch/dinov2",
                        help="Torch Hub repository for DINOv2. Default: %(default)s")
    parser.add_argument("--gpu", type=int, default=0,
                        help="CUDA device index used for embeddings and similarity. Default: %(default)s")
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda",
                        help="Execution device. GPU is the default; CPU is only for intentional debugging. Default: %(default)s")
    parser.add_argument("--batch-size", type=int, default=64,
                        help="Images embedded per batch. Lower this if GPU memory is exhausted. Default: %(default)s")
    parser.add_argument("--similarity-batch-size", type=int, default=128,
                        help="Anchor embeddings ranked per GPU similarity batch. Lower this if GPU memory is exhausted. Default: %(default)s")
    parser.add_argument("--image-size", type=int, default=224,
                        help="Square image preprocessing size. Default: %(default)s")
    parser.add_argument("--negative-scope", choices=("all", "same-parent"), default="all",
                        help="all mines negatives from every different-label image. same-parent restricts negatives to images sharing a parent label. Default: %(default)s")
    parser.add_argument("--image-id-column", default="ImageID",
                        help="Manifest ImageID column. Default: %(default)s")
    parser.add_argument("--label-column", default="LabelName",
                        help="Manifest leaf-label ID column used to define positives. Default: %(default)s")
    parser.add_argument("--display-name-column", default="DisplayName",
                        help="Manifest human-readable leaf-label column. Default: %(default)s")
    parser.add_argument("--parent-label-column", default="ParentLabelName",
                        help="Manifest parent-label ID column. Default: %(default)s")
    parser.add_argument("--parent-display-name-column", default="ParentDisplayName",
                        help="Manifest human-readable parent-label column. Default: %(default)s")
    parser.add_argument("--resume", action="store_true",
                        help="Resume matching embedding and triplet checkpoints. Default: disabled.")
    parser.add_argument("--overwrite", action="store_true",
                        help="Discard this backbone's cache and rewrite the selected output. Default: disabled.")
    parser.add_argument("--progress-refresh-seconds", type=float, default=0.1,
                        help="Minimum tqdm refresh interval. Default: %(default)s")
    args = parser.parse_args()
    if args.resume and args.overwrite:
        parser.error("Use either --resume or --overwrite, not both")
    for name in ("gpu", "batch_size", "similarity_batch_size", "image_size"):
        if getattr(args, name) < (0 if name == "gpu" else 1):
            parser.error("--{} must be {}".format(name.replace("_", "-"), "nonnegative" if name == "gpu" else "positive"))
    if args.progress_refresh_seconds <= 0:
        parser.error("--progress-refresh-seconds must be positive")
    return args


def ensure_device(args):
    if args.device == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested, but no CUDA device is available")
        torch.cuda.set_device(args.gpu)
        return torch.device("cuda", args.gpu)
    return torch.device("cpu")


def safe_file_component(value):
    return "".join("_" if character in '<>:"/\\|?*' else character for character in value).strip().rstrip(".") or "unnamed"


def read_manifest(path, args):
    if not path.is_file():
        raise FileNotFoundError("Manifest does not exist: {}".format(path))
    required = {
        args.image_id_column,
        args.label_column,
        args.display_name_column,
        args.parent_label_column,
        args.parent_display_name_column,
    }
    grouped = {}
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        missing = required.difference(reader.fieldnames or [])
        if missing:
            raise ValueError("Manifest is missing required columns: {}".format(", ".join(sorted(missing))))
        with tqdm(desc="Reading manifest", unit="row", mininterval=args.progress_refresh_seconds) as progress:
            for row_number, row in enumerate(reader, start=2):
                image_id = (row.get(args.image_id_column) or "").strip()
                label = (row.get(args.label_column) or "").strip()
                parent_label = (row.get(args.parent_label_column) or "").strip()
                if not image_id or not label:
                    raise ValueError("Manifest row {} has an empty {} or {}".format(
                        row_number, args.image_id_column, args.label_column))
                values = grouped.setdefault(image_id, {
                    "labels": set(), "display_names": set(), "parent_labels": set(), "parent_display_names": set(),
                })
                values["labels"].add(label)
                display_name = (row.get(args.display_name_column) or "").strip()
                if display_name:
                    values["display_names"].add(display_name)
                if parent_label:
                    values["parent_labels"].add(parent_label)
                parent_display_name = (row.get(args.parent_display_name_column) or "").strip()
                if parent_display_name:
                    values["parent_display_names"].add(parent_display_name)
                progress.update(1)
    return grouped


def index_images(images_root, args):
    if not images_root.is_dir():
        raise FileNotFoundError("Image root does not exist: {}".format(images_root))
    image_paths = {}
    duplicate_ids = 0
    with tqdm(desc="Indexing images", unit="file", mininterval=args.progress_refresh_seconds) as progress:
        for path in images_root.rglob("*"):
            if not path.is_file() or path.suffix.lower() not in IMAGE_SUFFIXES:
                continue
            image_id = path.stem
            existing = image_paths.get(image_id)
            if existing is None or str(path).lower() < str(existing).lower():
                if existing is not None:
                    duplicate_ids += 1
                image_paths[image_id] = path.resolve()
            else:
                duplicate_ids += 1
            progress.update(1)
    if not image_paths:
        raise ValueError("No supported images were found below: {}".format(images_root))
    return image_paths, duplicate_ids


def build_records(manifest_rows, image_paths):
    records = []
    missing_ids = []
    for image_id in sorted(manifest_rows):
        values = manifest_rows[image_id]
        path = image_paths.get(image_id)
        if path is None:
            missing_ids.append(image_id)
            continue
        records.append(ImageRecord(
            image_id=image_id,
            path=path,
            label_names=tuple(sorted(values["labels"])),
            display_names=tuple(sorted(values["display_names"])),
            parent_label_names=tuple(sorted(values["parent_labels"])),
            parent_display_names=tuple(sorted(values["parent_display_names"])),
        ))
    if not records:
        raise ValueError("None of the manifest ImageIDs could be resolved under --images-root")
    return records, missing_ids


def records_signature(records, args):
    digest = hashlib.sha256()
    digest.update(str(CACHE_VERSION).encode("ascii"))
    digest.update(args.backbone.encode("utf-8"))
    digest.update(args.dinov2_model.encode("utf-8"))
    digest.update(str(args.image_size).encode("ascii"))
    for record in records:
        stat = record.path.stat()
        digest.update(json.dumps((
            record.image_id, str(record.path), stat.st_size, stat.st_mtime_ns,
            record.label_names, record.parent_label_names,
        ), ensure_ascii=True).encode("utf-8"))
    return digest.hexdigest()


def cache_paths(output_dir, args):
    cache_name = safe_file_component(args.backbone if args.backbone == "resnet50" else args.dinov2_model)
    base = output_dir / "cache" / cache_name
    return base.with_suffix(".npy"), base.with_suffix(".json")


def remove_cache(embedding_path, metadata_path):
    embedding_path.unlink(missing_ok=True)
    metadata_path.unlink(missing_ok=True)


def load_embedding_state(embedding_path, metadata_path, signature, record_count):
    if not embedding_path.is_file() or not metadata_path.is_file():
        return None
    try:
        with metadata_path.open("r", encoding="utf-8") as handle:
            metadata = json.load(handle)
        if metadata.get("cache_version") != CACHE_VERSION:
            return None
        if metadata.get("signature") != signature or metadata.get("record_count") != record_count:
            return None
        embedding_dim = int(metadata["embedding_dim"])
        completed = int(metadata.get("completed", 0))
        if embedding_dim <= 0 or completed < 0 or completed > record_count:
            return None
        embeddings = np.lib.format.open_memmap(embedding_path, mode="r+", dtype=np.float32, shape=(record_count, embedding_dim))
        return embeddings, metadata, completed
    except (OSError, ValueError, TypeError, KeyError, json.JSONDecodeError):
        return None


def save_embedding_state(metadata_path, metadata):
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = metadata_path.with_name(metadata_path.name + ".inprogress")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2, sort_keys=True)
    temporary.replace(metadata_path)


def preprocess_image(path, image_size):
    with Image.open(path) as image:
        image = ImageOps.fit(image.convert("RGB"), (image_size, image_size), method=Image.Resampling.BICUBIC)
    values = torch.from_numpy(np.asarray(image).copy()).permute(2, 0, 1).float().div_(255.0)
    mean = torch.tensor((0.485, 0.456, 0.406)).view(3, 1, 1)
    std = torch.tensor((0.229, 0.224, 0.225)).view(3, 1, 1)
    return values.sub_(mean).div_(std)


def load_embedding_model(args, device):
    if args.backbone == "resnet50":
        model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
        model.fc = nn.Identity()
    else:
        model = torch.hub.load(args.torch_hub_repo, args.dinov2_model)
    return model.eval().to(device)


def embed_records(records, args, device, embedding_path, metadata_path, signature):
    state = load_embedding_state(embedding_path, metadata_path, signature, len(records)) if args.resume else None
    if state is None:
        model = load_embedding_model(args, device)
        first_image = preprocess_image(records[0].path, args.image_size).unsqueeze(0).to(device)
        with torch.inference_mode():
            first_embedding = model(first_image)
        if not isinstance(first_embedding, torch.Tensor) or first_embedding.ndim != 2:
            raise TypeError("Selected backbone must return a 2D embedding tensor")
        embedding_dim = int(first_embedding.shape[1])
        embedding_path.parent.mkdir(parents=True, exist_ok=True)
        embedding_path.unlink(missing_ok=True)
        embeddings = np.lib.format.open_memmap(embedding_path, mode="w+", dtype=np.float32, shape=(len(records), embedding_dim))
        embeddings[0] = functional.normalize(first_embedding.float(), dim=1).cpu().numpy()[0]
        metadata = {
            "cache_version": CACHE_VERSION,
            "signature": signature,
            "record_count": len(records),
            "embedding_dim": embedding_dim,
            "completed": 1,
        }
        embeddings.flush()
        save_embedding_state(metadata_path, metadata)
        start = 1
    else:
        embeddings, metadata, start = state
        if start == len(records):
            print("Reusing complete embedding cache: {}".format(embedding_path), flush=True)
            return embeddings
        model = load_embedding_model(args, device)
        print("Resuming embeddings at image {}/{}".format(start, len(records)), flush=True)
    with torch.inference_mode():
        with tqdm(total=len(records), initial=start, desc="Embedding {}".format(args.backbone), unit="image", mininterval=args.progress_refresh_seconds) as progress:
            for batch_start in range(start, len(records), args.batch_size):
                batch_records = records[batch_start:batch_start + args.batch_size]
                images = torch.stack([preprocess_image(record.path, args.image_size) for record in batch_records]).to(device, non_blocking=True)
                vectors = model(images)
                if not isinstance(vectors, torch.Tensor) or vectors.ndim != 2:
                    raise TypeError("Selected backbone must return a 2D embedding tensor")
                embeddings[batch_start:batch_start + len(batch_records)] = functional.normalize(vectors.float(), dim=1).cpu().numpy()
                metadata["completed"] = batch_start + len(batch_records)
                embeddings.flush()
                save_embedding_state(metadata_path, metadata)
                progress.update(len(batch_records))
    return embeddings


def membership_indices(records, attribute):
    grouped = {}
    for index, record in enumerate(records):
        for value in getattr(record, attribute):
            grouped.setdefault(value, []).append(index)
    return {key: torch.tensor(value, dtype=torch.long) for key, value in grouped.items()}


def joined(values):
    return ";".join(values)


def output_row(anchor_index, status, anchor, positive, negative, positive_shared, positive_similarity, negative_similarity, args):
    row = {
        "AnchorIndex": anchor_index,
        "Status": status,
        "Backbone": args.backbone if args.backbone == "resnet50" else args.dinov2_model,
        "NegativeScope": args.negative_scope,
        "AnchorImageID": anchor.image_id,
        "AnchorImagePath": str(anchor.path),
        "AnchorLabelNames": joined(anchor.label_names),
        "AnchorDisplayNames": joined(anchor.display_names),
        "AnchorParentLabelNames": joined(anchor.parent_label_names),
        "AnchorParentDisplayNames": joined(anchor.parent_display_names),
        "PositiveImageID": "",
        "PositiveImagePath": "",
        "PositiveSharedLabelNames": joined(positive_shared),
        "PositiveLabelNames": "",
        "PositiveDisplayNames": "",
        "PositiveSimilarity": "",
        "HardNegativeImageID": "",
        "HardNegativeImagePath": "",
        "HardNegativeLabelNames": "",
        "HardNegativeDisplayNames": "",
        "HardNegativeSimilarity": "",
    }
    if positive is not None:
        row.update({
            "PositiveImageID": positive.image_id,
            "PositiveImagePath": str(positive.path),
            "PositiveLabelNames": joined(positive.label_names),
            "PositiveDisplayNames": joined(positive.display_names),
            "PositiveSimilarity": "{:.8f}".format(positive_similarity),
        })
    if negative is not None:
        row.update({
            "HardNegativeImageID": negative.image_id,
            "HardNegativeImagePath": str(negative.path),
            "HardNegativeLabelNames": joined(negative.label_names),
            "HardNegativeDisplayNames": joined(negative.display_names),
            "HardNegativeSimilarity": "{:.8f}".format(negative_similarity),
        })
    return row


def read_triplet_checkpoint(path, record_count):
    if not path.is_file():
        return 0
    expected = 0
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames != RESULT_FIELDS:
            raise ValueError("Triplet checkpoint has unexpected columns: {}".format(path))
        for row in reader:
            try:
                anchor_index = int(row["AnchorIndex"])
            except (TypeError, ValueError) as error:
                raise ValueError("Triplet checkpoint has an invalid AnchorIndex") from error
            if anchor_index != expected:
                raise ValueError("Triplet checkpoint is not contiguous at anchor index {}".format(expected))
            expected += 1
    if expected > record_count:
        raise ValueError("Triplet checkpoint has more rows than current records")
    return expected


def rank_triplets(records, embeddings, args, device, checkpoint_path):
    start = read_triplet_checkpoint(checkpoint_path, len(records)) if args.resume else 0
    if start == len(records):
        print("Reusing complete triplet checkpoint: {}".format(checkpoint_path), flush=True)
        return
    label_to_indices = membership_indices(records, "label_names")
    parent_to_indices = membership_indices(records, "parent_label_names")
    all_embeddings = functional.normalize(torch.from_numpy(np.asarray(embeddings)).to(device), dim=1)
    mode = "a" if start else "w"
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    with checkpoint_path.open(mode, encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=RESULT_FIELDS)
        if start == 0:
            writer.writeheader()
        with tqdm(total=len(records), initial=start, desc="Mining triplets", unit="anchor", mininterval=args.progress_refresh_seconds) as progress:
            for batch_start in range(start, len(records), args.similarity_batch_size):
                batch_end = min(batch_start + args.similarity_batch_size, len(records))
                scores = all_embeddings[batch_start:batch_end] @ all_embeddings.T
                positive_mask = torch.zeros((batch_end - batch_start, len(records)), dtype=torch.bool, device=device)
                allowed_negative_mask = torch.ones_like(positive_mask)
                for offset, anchor_index in enumerate(range(batch_start, batch_end)):
                    anchor = records[anchor_index]
                    for label in anchor.label_names:
                        positive_mask[offset, label_to_indices[label].to(device)] = True
                    positive_mask[offset, anchor_index] = False
                    if args.negative_scope == "same-parent":
                        allowed_negative_mask[offset].fill_(False)
                        for parent in anchor.parent_label_names:
                            allowed_negative_mask[offset, parent_to_indices[parent].to(device)] = True
                positive_scores = scores.masked_fill(~positive_mask, -torch.inf)
                positive_values, positive_indices = positive_scores.max(dim=1)
                negative_mask = allowed_negative_mask & ~positive_mask
                for offset, anchor_index in enumerate(range(batch_start, batch_end)):
                    negative_mask[offset, anchor_index] = False
                negative_scores = scores.masked_fill(~negative_mask, -torch.inf)
                negative_values, negative_indices = negative_scores.max(dim=1)
                for offset, anchor_index in enumerate(range(batch_start, batch_end)):
                    anchor = records[anchor_index]
                    positive = records[int(positive_indices[offset])] if torch.isfinite(positive_values[offset]) else None
                    negative = records[int(negative_indices[offset])] if torch.isfinite(negative_values[offset]) else None
                    if positive is None:
                        status = "no_positive"
                    elif negative is None:
                        status = "no_hard_negative"
                    else:
                        status = "ok"
                    shared = tuple(sorted(set(anchor.label_names).intersection(positive.label_names))) if positive else ()
                    writer.writerow(output_row(
                        anchor_index, status, anchor, positive, negative, shared,
                        float(positive_values[offset]) if positive else None,
                        float(negative_values[offset]) if negative else None,
                        args,
                    ))
                handle.flush()
                os.fsync(handle.fileno())
                progress.update(batch_end - batch_start)


def write_missing_report(path, missing_ids):
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["ImageID"])
        writer.writerows((image_id,) for image_id in missing_ids)


def main():
    args = parse_args()
    manifest = args.manifest.expanduser()
    images_root = args.images_root.expanduser()
    output_dir = args.output_dir.expanduser()
    output = args.output.expanduser() if args.output else output_dir / "hard_negative_triplets_{}.csv".format(safe_file_component(args.backbone if args.backbone == "resnet50" else args.dinov2_model))
    checkpoint = output.with_name(output.stem + ".inprogress" + output.suffix)
    if args.overwrite:
        output.unlink(missing_ok=True)
        checkpoint.unlink(missing_ok=True)
    if output.is_file() and args.resume:
        print("Complete output already exists: {}".format(output.resolve()))
        return
    if output.is_file() and not args.overwrite:
        raise FileExistsError("Output already exists. Use --resume or --overwrite: {}".format(output))
    device = ensure_device(args)
    manifest_rows = read_manifest(manifest, args)
    image_paths, duplicate_file_ids = index_images(images_root, args)
    records, missing_ids = build_records(manifest_rows, image_paths)
    output_dir.mkdir(parents=True, exist_ok=True)
    missing_report = output_dir / "missing_manifest_image_ids.csv"
    write_missing_report(missing_report, missing_ids)
    print("Manifest rows grouped into unique ImageIDs: {}".format(len(manifest_rows)))
    print("Resolved anchor images: {}; missing: {}; duplicate file stems: {}".format(
        len(records), len(missing_ids), duplicate_file_ids
    ))
    print("Missing ImageID report: {}".format(missing_report.resolve()))
    signature = records_signature(records, args)
    embedding_path, metadata_path = cache_paths(output_dir, args)
    if args.overwrite:
        remove_cache(embedding_path, metadata_path)
    embeddings = embed_records(records, args, device, embedding_path, metadata_path, signature)
    rank_triplets(records, embeddings, args, device, checkpoint)
    checkpoint.replace(output)
    print("Triplet rows: {}".format(len(records)))
    print("Output: {}".format(output.resolve()))


if __name__ == "__main__":
    main()
