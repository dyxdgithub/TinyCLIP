"""Rank cross-leaf DINOv2 image similarities inside each n2 parent category.

Images are read from ``parent/leaf/image`` folders. For every image, the
script ranks only images from other leaf folders within the same parent folder,
then writes its top-N cosine-similar matches to one combined CSV. Completed
parent embeddings and tables are cached so --resume avoids repeated GPU work.
"""

import argparse
import csv
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as functional
from PIL import Image, ImageOps
from tqdm import tqdm


META_DIR = Path(__file__).resolve().parent.parent
SAMPLED_DIR = META_DIR / "Hierarchy" / "n2" / "300"
DEFAULT_INPUT_DIR = SAMPLED_DIR / "images_fiftyone_sampled_300_per_class"
DEFAULT_OUTPUT_DIR = SAMPLED_DIR / "dinov2_similarity_fiftyone_sampled_300_per_class"
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
RESULT_FIELDS = [
    "ParentCategory",
    "SourceLeafCategory",
    "SourceImageID",
    "SourceImagePath",
    "CandidateCount",
    "Rank",
    "MatchedLeafCategory",
    "MatchedImageID",
    "MatchedImagePath",
    "CosineSimilarity",
]
CACHE_VERSION = 1


@dataclass(frozen=True)
class ImageRecord:
    parent: str
    leaf: str
    image_id: str
    path: Path
    size: int
    modified_time_ns: int


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT_DIR, help="Image root with parent/leaf/image hierarchy. Default: %(default)s")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR, help="Output directory for cached embeddings, per-parent tables, and combined CSV. Default: %(default)s")
    parser.add_argument("--output-csv", type=Path, default=None, help="Combined top-N similarity CSV. Default: <output-dir>/top_n_cross_leaf_dinov2_similarity.csv")
    parser.add_argument("--model", default="dinov2_vitb14", help="DINOv2 model name passed to torch.hub. Default: %(default)s")
    parser.add_argument("--torch-hub-repo", default="facebookresearch/dinov2", help="Torch Hub repository containing --model. Default: %(default)s")
    parser.add_argument("--device", default="cuda", help="Torch device for embedding and similarity computation. Default: %(default)s. Use cpu only when GPU execution is intentionally unavailable.")
    parser.add_argument("--batch-size", type=int, default=64, help="Images embedded per GPU batch. Lower this when CUDA runs out of memory. Default: %(default)s")
    parser.add_argument("--similarity-batch-size", type=int, default=1024, help="Source embeddings processed per cosine-similarity batch. Default: %(default)s")
    parser.add_argument("--top-n", type=int, default=10, help="Cross-leaf nearest images retained per source image. Default: %(default)s")
    parser.add_argument("--image-size", type=int, default=224, help="Square DINOv2 preprocessing resolution in pixels. Default: %(default)s")
    parser.add_argument("--exclude-same-image-id", action=argparse.BooleanOptionalAction, default=True, help="Exclude candidates with the same ImageID even when copied into another leaf category. Default: enabled.")
    parser.add_argument("--parent", action="append", default=None, help="Process only this parent directory name. Repeat to select multiple parents. Default: all parents.")
    parser.add_argument("--resume", action="store_true", help="Reuse valid cached parent embeddings and completed parent result tables.")
    parser.add_argument("--overwrite", action="store_true", help="Recompute all selected parents and replace their cache/table files. Does not delete unrelated output files.")
    parser.add_argument("--progress-refresh-seconds", type=float, default=0.1, help="Minimum seconds between tqdm refreshes. Default: %(default)s")
    args = parser.parse_args()
    for name in ("batch_size", "similarity_batch_size", "top_n", "image_size"):
        if getattr(args, name) <= 0:
            parser.error("--{} must be positive".format(name.replace("_", "-")))
    if args.progress_refresh_seconds <= 0:
        parser.error("--progress-refresh-seconds must be positive")
    return args


def ensure_device(device_name):
    device = torch.device(device_name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("--device {} was requested, but CUDA is unavailable".format(device_name))
    return device


def safe_file_component(value):
    return "".join("_" if character in '<>:"/\\|?*' else character for character in value).strip().rstrip(".") or "unnamed"


def discover_records(input_dir, selected_parents):
    records_by_parent = {}
    allowed_parents = set(selected_parents) if selected_parents else None
    for parent_dir in sorted(path for path in input_dir.iterdir() if path.is_dir()):
        if allowed_parents is not None and parent_dir.name not in allowed_parents:
            continue
        records = []
        for leaf_dir in sorted(path for path in parent_dir.iterdir() if path.is_dir()):
            for image_path in sorted(leaf_dir.rglob("*")):
                if not image_path.is_file() or image_path.suffix.lower() not in IMAGE_SUFFIXES:
                    continue
                info = image_path.stat()
                records.append(
                    ImageRecord(
                        parent=parent_dir.name,
                        leaf=leaf_dir.name,
                        image_id=image_path.stem,
                        path=image_path.resolve(),
                        size=info.st_size,
                        modified_time_ns=info.st_mtime_ns,
                    )
                )
        if records:
            records_by_parent[parent_dir.name] = records
    return records_by_parent


def record_signature(records):
    payload = [
        (record.leaf, record.image_id, str(record.path), record.size, record.modified_time_ns)
        for record in records
    ]
    return hashlib.sha256(json.dumps(payload, ensure_ascii=True).encode("utf-8")).hexdigest()


def cache_path(output_dir, parent):
    return output_dir / "embeddings" / (safe_file_component(parent) + ".pt")


def table_path(output_dir, parent):
    return output_dir / "tables" / (safe_file_component(parent) + ".csv")


def load_cached_embeddings(path, records, args):
    if not path.is_file():
        return None
    try:
        payload = torch.load(path, map_location="cpu")
    except (RuntimeError, ValueError, TypeError, OSError):
        return None
    if payload.get("cache_version") != CACHE_VERSION:
        return None
    if payload.get("model") != args.model or payload.get("image_size") != args.image_size:
        return None
    if payload.get("record_signature") != record_signature(records):
        return None
    embeddings = payload.get("embeddings")
    if not isinstance(embeddings, torch.Tensor) or embeddings.ndim != 2:
        return None
    if embeddings.shape[0] != len(records):
        return None
    return embeddings.float()


def save_cached_embeddings(path, records, embeddings, args):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".inprogress")
    temporary.unlink(missing_ok=True)
    torch.save(
        {
            "cache_version": CACHE_VERSION,
            "model": args.model,
            "image_size": args.image_size,
            "record_signature": record_signature(records),
            "embeddings": embeddings.cpu(),
        },
        temporary,
    )
    temporary.replace(path)


def preprocess_image(path, image_size):
    with Image.open(path) as image:
        image = ImageOps.fit(
            image.convert("RGB"),
            (image_size, image_size),
            method=Image.Resampling.BICUBIC,
        )
    values = torch.from_numpy(np.asarray(image).copy()).permute(2, 0, 1).float()
    values = values.div_(255.0)
    mean = torch.tensor((0.485, 0.456, 0.406)).view(3, 1, 1)
    std = torch.tensor((0.229, 0.224, 0.225)).view(3, 1, 1)
    return values.sub_(mean).div_(std)


def load_dinov2_model(args, device):
    model = torch.hub.load(args.torch_hub_repo, args.model)
    model.eval().to(device)
    return model


def embed_records(model, records, args, device):
    embeddings = []
    with torch.inference_mode():
        with tqdm(total=len(records), desc="Embedding {}".format(records[0].parent), unit="image", mininterval=args.progress_refresh_seconds) as progress:
            for start in range(0, len(records), args.batch_size):
                batch_records = records[start:start + args.batch_size]
                images = torch.stack([preprocess_image(record.path, args.image_size) for record in batch_records]).to(device, non_blocking=True)
                batch_embeddings = model(images)
                if not isinstance(batch_embeddings, torch.Tensor) or batch_embeddings.ndim != 2:
                    raise TypeError("DINOv2 model must return a 2D embedding tensor; got {}".format(type(batch_embeddings).__name__))
                embeddings.append(functional.normalize(batch_embeddings.float(), dim=1).cpu())
                progress.update(len(batch_records))
    return torch.cat(embeddings, dim=0)


def rank_parent(records, embeddings, args, device):
    embeddings = functional.normalize(embeddings.to(device), dim=1)
    leaf_to_code = {
        leaf: index
        for index, leaf in enumerate(dict.fromkeys(record.leaf for record in records))
    }
    image_id_to_code = {
        image_id: index
        for index, image_id in enumerate(
            dict.fromkeys(record.image_id for record in records)
        )
    }
    leaf_codes = torch.tensor(
        [leaf_to_code[record.leaf] for record in records],
        device=device,
    )
    image_id_codes = torch.tensor(
        [image_id_to_code[record.image_id] for record in records],
        device=device,
    )
    rows = []
    with torch.inference_mode():
        with tqdm(total=len(records), desc="Ranking {}".format(records[0].parent), unit="image", mininterval=args.progress_refresh_seconds) as progress:
            for start in range(0, len(records), args.similarity_batch_size):
                end = min(start + args.similarity_batch_size, len(records))
                scores = embeddings[start:end] @ embeddings.T
                for offset, source_index in enumerate(range(start, end)):
                    valid = leaf_codes.ne(leaf_codes[source_index])
                    if args.exclude_same_image_id:
                        valid &= image_id_codes.ne(image_id_codes[source_index])
                    candidate_count = int(valid.sum().item())
                    source = records[source_index]
                    if candidate_count == 0:
                        rows.append(make_empty_row(source, candidate_count))
                        continue
                    candidate_indices = torch.nonzero(valid, as_tuple=False).flatten()
                    candidate_scores = scores[offset, candidate_indices]
                    count = min(args.top_n, candidate_count)
                    values, positions = torch.topk(candidate_scores, count, largest=True, sorted=True)
                    for rank, (value, position) in enumerate(zip(values.tolist(), positions.tolist()), start=1):
                        matched = records[int(candidate_indices[position].item())]
                        rows.append(make_result_row(source, matched, candidate_count, rank, value))
                progress.update(end - start)
    return rows


def make_empty_row(source, candidate_count):
    return {
        "ParentCategory": source.parent,
        "SourceLeafCategory": source.leaf,
        "SourceImageID": source.image_id,
        "SourceImagePath": str(source.path),
        "CandidateCount": candidate_count,
        "Rank": 0,
        "MatchedLeafCategory": "",
        "MatchedImageID": "",
        "MatchedImagePath": "",
        "CosineSimilarity": "",
    }


def make_result_row(source, matched, candidate_count, rank, similarity):
    return {
        "ParentCategory": source.parent,
        "SourceLeafCategory": source.leaf,
        "SourceImageID": source.image_id,
        "SourceImagePath": str(source.path),
        "CandidateCount": candidate_count,
        "Rank": rank,
        "MatchedLeafCategory": matched.leaf,
        "MatchedImageID": matched.image_id,
        "MatchedImagePath": str(matched.path),
        "CosineSimilarity": "{:.8f}".format(similarity),
    }


def write_table(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".inprogress")
    temporary.unlink(missing_ok=True)
    with temporary.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=RESULT_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def table_is_complete(path, records):
    if not path.is_file():
        return False
    expected_sources = {
        (record.leaf, record.image_id, str(record.path)) for record in records
    }
    actual_sources = set()
    try:
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle)
            if reader.fieldnames != RESULT_FIELDS:
                return False
            for row in reader:
                if row.get("ParentCategory") != records[0].parent:
                    return False
                actual_sources.add(
                    (
                        row.get("SourceLeafCategory", ""),
                        row.get("SourceImageID", ""),
                        row.get("SourceImagePath", ""),
                    )
                )
            return expected_sources == actual_sources
    except (OSError, csv.Error):
        return False


def write_combined_output(output_csv, parent_tables):
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_csv.with_name(output_csv.name + ".inprogress")
    temporary.unlink(missing_ok=True)
    with temporary.open("w", encoding="utf-8-sig", newline="") as destination:
        writer = csv.DictWriter(destination, fieldnames=RESULT_FIELDS)
        writer.writeheader()
        for path in parent_tables:
            with path.open("r", encoding="utf-8-sig", newline="") as source:
                writer.writerows(csv.DictReader(source))
    temporary.replace(output_csv)


def main():
    args = parse_args()
    input_dir = args.input_dir.expanduser()
    output_dir = args.output_dir.expanduser()
    output_csv = args.output_csv.expanduser() if args.output_csv else output_dir / "top_n_cross_leaf_dinov2_similarity.csv"
    if not input_dir.is_dir():
        raise FileNotFoundError("Input image directory does not exist: {}".format(input_dir))
    if args.overwrite and args.resume:
        raise ValueError("Use either --resume or --overwrite, not both")
    device = ensure_device(args.device)
    records_by_parent = discover_records(input_dir, args.parent)
    if not records_by_parent:
        raise ValueError("No supported images found under parent/leaf directories: {}".format(input_dir))
    if args.parent:
        missing = set(args.parent).difference(records_by_parent)
        if missing:
            raise ValueError("Requested parent directories have no images: {}".format(", ".join(sorted(missing))))

    output_dir.mkdir(parents=True, exist_ok=True)
    model = None
    parent_tables = []
    for parent, records in records_by_parent.items():
        parent_table = table_path(output_dir, parent)
        parent_tables.append(parent_table)
        if args.resume and not args.overwrite and table_is_complete(parent_table, records):
            print("Reusing completed table for {}".format(parent), flush=True)
            continue
        embeddings = None if args.overwrite else load_cached_embeddings(cache_path(output_dir, parent), records, args)
        if embeddings is None:
            if model is None:
                model = load_dinov2_model(args, device)
            embeddings = embed_records(model, records, args, device)
            save_cached_embeddings(cache_path(output_dir, parent), records, embeddings, args)
        else:
            print("Reusing cached embeddings for {}".format(parent), flush=True)
        rows = rank_parent(records, embeddings, args, device)
        write_table(parent_table, rows)
    write_combined_output(output_csv, parent_tables)
    print("Parents processed: {}".format(len(records_by_parent)))
    print("Combined table: {}".format(output_csv.resolve()))


if __name__ == "__main__":
    main()
