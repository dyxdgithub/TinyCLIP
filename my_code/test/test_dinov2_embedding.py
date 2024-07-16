"""Embed images with DINOv2 and report inference timing."""

import argparse
import json
import time
from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from tqdm import tqdm


IMAGE_EXTENSIONS = {
    ".bmp",
    ".jpeg",
    ".jpg",
    ".png",
    ".tif",
    ".tiff",
    ".webp",
}
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


class ImageDataset(Dataset):
    def __init__(self, image_paths, transform):
        self.image_paths = image_paths
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, index):
        path = self.image_paths[index]
        try:
            with Image.open(path) as image:
                image = image.convert("RGB")
                return self.transform(image), str(path)
        except Exception as error:
            print("Skipping {}: {}".format(path, error))
            return None


def collate_images(batch):
    valid_items = [item for item in batch if item is not None]
    if not valid_items:
        return None
    images, paths = zip(*valid_items)
    return torch.stack(images), list(paths)


def collect_image_paths(image_path, recursive):
    raw_path = str(image_path)
    if "\x07" in raw_path:
        corrected_path = raw_path.replace("\x07", "a")
        if Path(corrected_path).exists():
            print("Corrected BEL character in image path: {}".format(corrected_path))
            image_path = corrected_path
        else:
            raise ValueError(
                "Image path contains a BEL character from an escaped Windows path. "
                "Use forward slashes or a raw string, for example: {}".format(
                    corrected_path
                )
            )
    image_path = Path(image_path)
    if image_path.is_file():
        if image_path.suffix.lower() not in IMAGE_EXTENSIONS:
            raise ValueError("Unsupported image extension: {}".format(image_path))
        return [image_path]
    if not image_path.is_dir():
        raise FileNotFoundError("Image path does not exist: {}".format(image_path))
    iterator = image_path.rglob("*") if recursive else image_path.glob("*")
    return sorted(
        path for path in iterator if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )


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


def load_model(model_name, device, dinov2_repo=None):
    if model_name not in MODEL_NAMES:
        raise ValueError(
            "Unsupported model '{}'. Choose from: {}".format(
                model_name, ", ".join(sorted(MODEL_NAMES))
            )
        )
    if dinov2_repo:
        repo_dir = Path(dinov2_repo).expanduser().resolve()
        if not (repo_dir / "hubconf.py").is_file():
            raise FileNotFoundError(
                "Local DINOv2 repository must contain hubconf.py: {}".format(repo_dir)
            )
        model = torch.hub.load(
            str(repo_dir), model_name, source="local", trust_repo=True
        )
    else:
        try:
            model = torch.hub.load(
                "facebookresearch/dinov2", model_name, trust_repo=True
            )
        except Exception as error:
            raise RuntimeError(
                "Unable to download DINOv2 from GitHub. Check network/proxy access, "
                "or download the DINOv2 repository locally and pass "
                "--dinov2-repo C:/path/to/dinov2. Original error: {}".format(error)
            ) from error
    return model.eval().to(device)


def extract_features(model_output):
    if isinstance(model_output, dict):
        for key in ("x_norm_clstoken", "x_prenorm", "x_norm_patchtokens"):
            if key in model_output:
                features = model_output[key]
                if features.ndim == 3:
                    return features[:, 0]
                return features
        raise RuntimeError("DINOv2 output does not contain a supported feature key")
    if isinstance(model_output, (tuple, list)):
        model_output = model_output[0]
    if model_output.ndim == 3:
        return model_output[:, 0]
    return model_output


def synchronize(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def embed_images(
    model,
    data_loader,
    device,
    normalize,
):
    embeddings = []
    image_paths = []
    processed_images = 0

    for batch in tqdm(data_loader, desc="Embedding images", unit="batch"):
        if batch is None:
            continue
        images, paths = batch
        images = images.to(device, non_blocking=device.type == "cuda")
        with torch.inference_mode():
            features = extract_features(model(images))
        if normalize:
            features = torch.nn.functional.normalize(features, dim=1)
        embeddings.append(features.cpu())
        image_paths.extend(paths)
        processed_images += len(paths)

    if not embeddings:
        return torch.empty((0, 0)), image_paths, processed_images
    return torch.cat(embeddings, dim=0), image_paths, processed_images


def warmup_model(model, data_loader, device, warmup_batches):
    if warmup_batches == 0:
        return 0
    warmup_count = 0
    for batch in data_loader:
        if batch is None:
            continue
        images, _ = batch
        images = images.to(device, non_blocking=device.type == "cuda")
        with torch.inference_mode():
            extract_features(model(images))
        warmup_count += 1
        if warmup_count >= warmup_batches:
            break
    synchronize(device)
    return warmup_count


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image_path", type=Path, help="Image file or image directory",default="C:/Users/ASUS/Desktop/大模型代码/TinyCLIP/my_code/data/OpenImage/meta/Hierarchy/n2/audio_image/images/Clock/Wall clock")
    parser.add_argument("--model", default="dinov2_vitb14", choices=sorted(MODEL_NAMES))
    parser.add_argument(
        "--dinov2-repo",
        type=Path,
        default=None,
        help="Local DINOv2 repository containing hubconf.py; avoids GitHub download.",
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--warmup-batches", type=int, default=1)
    parser.add_argument("--recursive", action="store_true")
    parser.add_argument("--normalize", action="store_true")
    parser.add_argument("--output", type=Path, default=None)
    return parser.parse_args()


def main():
    args = parse_args()
    if args.batch_size <= 0 or args.num_workers < 0 or args.warmup_batches < 0:
        raise ValueError("batch-size must be positive; other counts cannot be negative")
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable. Install a CUDA-enabled PyTorch build or use --device cpu.")

    image_paths = collect_image_paths(args.image_path, args.recursive)
    if not image_paths:
        raise ValueError("No supported images found under {}".format(args.image_path))

    print("Loading model: {}".format(args.model))
    model_load_start = time.perf_counter()
    model = load_model(args.model, device, args.dinov2_repo)
    synchronize(device)
    model_load_seconds = time.perf_counter() - model_load_start

    dataset = ImageDataset(image_paths, build_transform())
    data_loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        collate_fn=collate_images,
    )

    actual_warmup_batches = warmup_model(
        model, data_loader, device, args.warmup_batches
    )
    synchronize(device)
    inference_start = time.perf_counter()
    embeddings, embedded_paths, embedded_count = embed_images(
        model,
        data_loader,
        device,
        args.normalize,
    )
    synchronize(device)
    inference_seconds = time.perf_counter() - inference_start

    result = {
        "model": args.model,
        "device": str(device),
        "input_images": len(image_paths),
        "embedded_images": embedded_count,
        "embedding_shape": list(embeddings.shape),
        "model_load_seconds": model_load_seconds,
        "inference_seconds": inference_seconds,
        "average_ms_per_image": (
            inference_seconds * 1000 / embedded_count if embedded_count else None
        ),
        "images_per_second": (
            embedded_count / inference_seconds if inference_seconds > 0 else None
        ),
        "warmup_batches": actual_warmup_batches,
        "normalized": args.normalize,
    }
    print(json.dumps(result, ensure_ascii=False, indent=2))

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "image_paths": embedded_paths,
                "embeddings": embeddings,
                "metrics": result,
            },
            args.output,
        )
        print("Saved embeddings: {}".format(args.output.resolve()))


if __name__ == "__main__":
    main()
