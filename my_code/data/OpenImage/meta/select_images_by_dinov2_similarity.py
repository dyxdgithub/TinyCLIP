"""Select each leaf class's images most similar to other images in its parent class."""

import argparse
import csv
import hashlib
import shutil
from dataclasses import dataclass
from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from tqdm import tqdm


META_DIR = Path(__file__).resolve().parent
DEFAULT_IMAGES_DIR = META_DIR / "Hierarchy" / "n2" / "audio_image" / "images"
DEFAULT_OUTPUT_DIR = META_DIR / "Hierarchy" / "n2" / "audio_image" / "dinov2_similarity"
IMAGE_EXTENSIONS = {".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff", ".webp"}
MODEL_NAMES = {
    "dinov2_vits14",
    "dinov2_vitb14",
    "dinov2_vitl14",
    "dinov2_vitg14",
    "dinov2_vits14_reg",
    "dinov2_vitb14_reg",
    "dinov2_vitl14_reg",
    "dinov2_vitg14_reg",
}
TABLE_FIELDS = [
    "ParentClass",
    "SmallClass",
    "Rank",
    "ImagePath",
    "MostSimilarImagePath",
    "Similarity",
    "SmallClassImageCount",
    "CrossClassCandidateCount",
]


@dataclass(frozen=True)
class ImageRecord:
    path: Path
    parent_class: str
    small_class: str


class ImageDataset(Dataset):
    def __init__(self, records, transform):
        self.records = records
        self.transform = transform

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        record = self.records[index]
        try:
            with Image.open(record.path) as image:
                return self.transform(image.convert("RGB")), record
        except Exception as error:
            print("Skipping unreadable image {}: {}".format(record.path, error))
            return None


def collate_images(batch):
    valid_items = [item for item in batch if item is not None]
    if not valid_items:
        return None
    images, records = zip(*valid_items)
    return torch.stack(images), list(records)


def build_transform():
    return transforms.Compose(
        [
            transforms.Resize(256, interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=(0.485, 0.456, 0.406),
                std=(0.229, 0.224, 0.225),
            ),
        ]
    )


def collect_parent_records(images_dir):
    """Read a strict two-level parent-class/small-class directory structure."""
    images_dir = Path(images_dir)
    if not images_dir.is_dir():
        raise FileNotFoundError("Images directory does not exist: {}".format(images_dir))
    records_by_parent = {}
    for parent_dir in sorted(path for path in images_dir.iterdir() if path.is_dir()):
        records = []
        for small_dir in sorted(path for path in parent_dir.iterdir() if path.is_dir()):
            for image_path in sorted(
                path
                for path in small_dir.rglob("*")
                if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
            ):
                records.append(
                    ImageRecord(image_path, parent_dir.name, small_dir.name)
                )
        if records:
            records_by_parent[parent_dir.name] = records
    return records_by_parent


def load_model(model_name, dinov2_repo, checkpoint, device):
    """Load DINOv2 from torch.hub or explicitly supplied local artifacts."""
    if model_name not in MODEL_NAMES:
        raise ValueError("Unsupported DINOv2 model: {}".format(model_name))
    if checkpoint is None:
        if dinov2_repo is None:
            model = torch.hub.load("facebookresearch/dinov2", model_name, trust_repo=True)
        else:
            repo_dir = Path(dinov2_repo).expanduser().resolve()
            if not (repo_dir / "hubconf.py").is_file():
                raise FileNotFoundError(
                    "--dinov2-repo must contain hubconf.py: {}".format(repo_dir)
                )
            model = torch.hub.load(
                str(repo_dir), model_name, source="local", trust_repo=True
            )
        return model.eval().to(device)

    if dinov2_repo is None:
        raise ValueError("--dinov2-repo is required when --checkpoint is provided")
    repo_dir = Path(dinov2_repo).expanduser().resolve()
    checkpoint_path = Path(checkpoint).expanduser().resolve()
    if not (repo_dir / "hubconf.py").is_file():
        raise FileNotFoundError(
            "--dinov2-repo must contain hubconf.py: {}".format(repo_dir)
        )
    if not checkpoint_path.is_file():
        raise FileNotFoundError("--checkpoint does not exist: {}".format(checkpoint_path))
    model = torch.hub.load(
        str(repo_dir), model_name, source="local", trust_repo=True, pretrained=False
    )
    try:
        checkpoint_data = torch.load(
            checkpoint_path, map_location="cpu", weights_only=False
        )
    except TypeError:
        checkpoint_data = torch.load(checkpoint_path, map_location="cpu")
    state_dict = extract_state_dict(checkpoint_data)
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing or unexpected:
        raise RuntimeError(
            "Checkpoint does not match {}. Missing keys: {}; unexpected keys: {}"
            .format(model_name, len(missing), len(unexpected))
        )
    return model.eval().to(device)


def extract_state_dict(checkpoint_data):
    if not isinstance(checkpoint_data, dict):
        raise ValueError("DINOv2 checkpoint must be a state-dict dictionary")
    for key in ("model", "state_dict", "teacher"):
        candidate = checkpoint_data.get(key)
        if isinstance(candidate, dict):
            checkpoint_data = candidate
            break
    state_dict = {}
    for key, value in checkpoint_data.items():
        if not isinstance(value, torch.Tensor):
            continue
        for prefix in ("module.", "backbone."):
            if key.startswith(prefix):
                key = key[len(prefix):]
        state_dict[key] = value
    if not state_dict:
        raise ValueError("No tensor weights found in DINOv2 checkpoint")
    return state_dict


def extract_features(model_output):
    if isinstance(model_output, dict):
        for key in ("x_norm_clstoken", "x_prenorm", "x_norm_patchtokens"):
            if key in model_output:
                features = model_output[key]
                return features[:, 0] if features.ndim == 3 else features
        raise RuntimeError("DINOv2 output does not contain a supported feature key")
    if isinstance(model_output, (tuple, list)):
        model_output = model_output[0]
    return model_output[:, 0] if model_output.ndim == 3 else model_output


def embed_records(model, records, device, batch_size, num_workers):
    data_loader = DataLoader(
        ImageDataset(records, build_transform()),
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
        collate_fn=collate_images,
    )
    embeddings = []
    embedded_records = []
    for batch in tqdm(data_loader, desc="Embedding images", unit="batch", leave=False):
        if batch is None:
            continue
        images, batch_records = batch
        images = images.to(device, non_blocking=device.type == "cuda")
        with torch.inference_mode():
            features = extract_features(model(images))
        embeddings.append(torch.nn.functional.normalize(features, dim=1).cpu())
        embedded_records.extend(batch_records)
    if not embeddings:
        return torch.empty((0, 0)), []
    return torch.cat(embeddings), embedded_records


def records_signature(records):
    digest = hashlib.sha256()
    for record in records:
        stat = record.path.stat()
        digest.update(
            "{}|{}|{}|{}\n".format(
                record.path.resolve(), record.small_class, stat.st_size, stat.st_mtime_ns
            ).encode("utf-8")
        )
    return digest.hexdigest()


def load_cached_embeddings(cache_path, signature):
    if not cache_path.is_file():
        return None
    try:
        data = torch.load(cache_path, map_location="cpu", weights_only=False)
    except TypeError:
        data = torch.load(cache_path, map_location="cpu")
    if data.get("signature") != signature:
        return None
    paths = [Path(path) for path in data.get("paths", [])]
    small_classes = data.get("small_classes", [])
    embeddings = data.get("embeddings")
    if len(paths) != len(small_classes) or len(paths) != len(embeddings):
        return None
    parent_class = data.get("parent_class", "")
    records = [
        ImageRecord(path, parent_class, small_class)
        for path, small_class in zip(paths, small_classes)
    ]
    return embeddings, records


def save_cached_embeddings(cache_path, signature, embeddings, records):
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "signature": signature,
            "parent_class": records[0].parent_class if records else "",
            "paths": [str(record.path.resolve()) for record in records],
            "small_classes": [record.small_class for record in records],
            "embeddings": embeddings,
        },
        cache_path,
    )


def find_cross_class_neighbors(embeddings, records, device, similarity_batch_size):
    """Find every image's highest cosine-similarity neighbor from another small class."""
    if len(records) != len(embeddings):
        raise ValueError("Record and embedding counts must match")
    class_names = sorted({record.small_class for record in records})
    class_to_index = {name: index for index, name in enumerate(class_names)}
    class_indices = torch.tensor(
        [class_to_index[record.small_class] for record in records], dtype=torch.long
    )
    if len(class_names) < 2:
        return [None] * len(records)

    vectors = embeddings.to(device)
    reference_classes = class_indices.to(device)
    neighbors = []
    for start in range(0, len(records), similarity_batch_size):
        end = min(start + similarity_batch_size, len(records))
        query_vectors = vectors[start:end]
        scores = query_vectors @ vectors.T
        same_class = class_indices[start:end].to(device).unsqueeze(1).eq(reference_classes)
        scores.masked_fill_(same_class, float("-inf"))
        best_scores, best_indices = scores.max(dim=1)
        neighbors.extend(
            (float(score), int(index))
            for score, index in zip(best_scores.cpu(), best_indices.cpu())
        )
    return neighbors


def select_top_n(records, neighbors, top_n):
    grouped = {}
    for record, neighbor in zip(records, neighbors):
        if neighbor is None:
            continue
        score, neighbor_index = neighbor
        grouped.setdefault(record.small_class, []).append(
            (score, str(record.path.resolve()), neighbor_index, record)
        )

    class_counts = {}
    for record in records:
        class_counts[record.small_class] = class_counts.get(record.small_class, 0) + 1
    total_count = len(records)
    selected = []
    for small_class, candidates in sorted(grouped.items()):
        candidates.sort(key=lambda item: (-item[0], item[1]))
        for rank, (score, _, neighbor_index, record) in enumerate(candidates[:top_n], start=1):
            selected.append(
                {
                    "ParentClass": record.parent_class,
                    "SmallClass": small_class,
                    "Rank": rank,
                    "ImagePath": str(record.path.resolve()),
                    "MostSimilarImagePath": str(records[neighbor_index].path.resolve()),
                    "Similarity": "{:.8f}".format(score),
                    "SmallClassImageCount": class_counts[small_class],
                    "CrossClassCandidateCount": total_count - class_counts[small_class],
                }
            )
    return selected


def safe_file_name(value):
    return "".join(character if character.isalnum() or character in " ._-" else "_" for character in value)


def write_parent_table(output_dir, parent_class, rows):
    table_path = Path(output_dir) / "tables" / (safe_file_name(parent_class) + ".csv")
    table_path.parent.mkdir(parents=True, exist_ok=True)
    with table_path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=TABLE_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    return table_path


def copy_retained_images(rows, retained_dir):
    for row in rows:
        destination = Path(retained_dir) / row["ParentClass"] / row["SmallClass"] / Path(row["ImagePath"]).name
        destination.parent.mkdir(parents=True, exist_ok=True)
        if not destination.is_file():
            shutil.copy2(row["ImagePath"], destination)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--images-dir", type=Path, default=DEFAULT_IMAGES_DIR,
        help="Two-level parent-class/small-class image directory. Default: %(default)s",
    )
    parser.add_argument(
        "--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR,
        help="Directory for cached embeddings and per-parent Top-N CSV tables. Default: %(default)s",
    )
    parser.add_argument(
        "--dinov2-repo", type=Path, default=None,
        help="Optional local DINOv2 repository containing hubconf.py. Without it, torch.hub downloads the official repository when this script runs.",
    )
    parser.add_argument(
        "--checkpoint", type=Path, default=None,
        help="Optional local DINOv2 checkpoint matching --model. Requires --dinov2-repo and avoids downloading weights.",
    )
    parser.add_argument("--model", choices=sorted(MODEL_NAMES), default="dinov2_vitb14",
                        help="DINOv2 model architecture matching --checkpoint. Default: %(default)s")
    parser.add_argument("--device", default="cuda",
                        help="Torch device. GPU is the default; use cpu only when GPU is unavailable. Default: %(default)s")
    parser.add_argument("--top-n", type=int, default=20,
                        help="Retain the N highest-scoring images in each small class. Default: %(default)s")
    parser.add_argument("--batch-size", type=int, default=32,
                        help="DINOv2 inference batch size; reduce it for less GPU memory use. Default: %(default)s")
    parser.add_argument("--similarity-batch-size", type=int, default=1024,
                        help="Query rows per cosine-similarity matrix chunk. Default: %(default)s")
    parser.add_argument("--num-workers", type=int, default=4,
                        help="DataLoader workers. Set 0 when multiprocessing is unsuitable. Default: %(default)s")
    parser.add_argument("--copy-retained", action="store_true",
                        help="Copy Top-N images into --output-dir/retained_images. Original images are never moved or deleted.")
    parser.add_argument("--force-reembed", action="store_true",
                        help="Ignore matching per-parent embedding caches and run DINOv2 again.")
    parser.add_argument("--dry-run", action="store_true",
                        help="Report image counts by parent class without loading DINOv2 or writing output files.")
    return parser.parse_args()


def main():
    args = parse_args()
    if args.top_n < 1 or args.batch_size < 1 or args.similarity_batch_size < 1:
        raise ValueError("--top-n, --batch-size, and --similarity-batch-size must be positive")
    if args.num_workers < 0:
        raise ValueError("--num-workers cannot be negative")
    records_by_parent = collect_parent_records(args.images_dir)
    total_images = sum(len(records) for records in records_by_parent.values())
    print("Parent classes with images: {}".format(len(records_by_parent)))
    print("Images discovered: {}".format(total_images))
    for parent_class, records in records_by_parent.items():
        small_classes = len({record.small_class for record in records})
        print("  {}: {} images in {} small classes".format(parent_class, len(records), small_classes))
    if args.dry_run:
        return
    if not records_by_parent:
        raise ValueError("No supported images found below {}".format(args.images_dir))

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable. Use --device cpu only when GPU execution is unavailable.")

    model = load_model(args.model, args.dinov2_repo, args.checkpoint, device)
    all_rows = []
    for parent_class, records in tqdm(records_by_parent.items(), desc="Parent classes", unit="parent"):
        cache_path = Path(args.output_dir) / "embeddings" / (safe_file_name(parent_class) + ".pt")
        signature = records_signature(records)
        cached = None if args.force_reembed else load_cached_embeddings(cache_path, signature)
        if cached is None:
            embeddings, embedded_records = embed_records(
                model, records, device, args.batch_size, args.num_workers
            )
            if not embedded_records:
                continue
            save_cached_embeddings(cache_path, signature, embeddings, embedded_records)
        else:
            embeddings, embedded_records = cached
            print("Using embedding cache for {}".format(parent_class))
        neighbors = find_cross_class_neighbors(
            embeddings, embedded_records, device, args.similarity_batch_size
        )
        rows = select_top_n(embedded_records, neighbors, args.top_n)
        table_path = write_parent_table(args.output_dir, parent_class, rows)
        if args.copy_retained:
            copy_retained_images(rows, Path(args.output_dir) / "retained_images")
        all_rows.extend(rows)
        print("{}: wrote {} selected images to {}".format(parent_class, len(rows), table_path))

    index_path = Path(args.output_dir) / "top_n_similarity_all_classes.csv"
    index_path.parent.mkdir(parents=True, exist_ok=True)
    with index_path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=TABLE_FIELDS)
        writer.writeheader()
        writer.writerows(all_rows)
    print("Selected image rows: {}".format(len(all_rows)))
    print("Combined Top-N table: {}".format(index_path.resolve()))


if __name__ == "__main__":
    main()
