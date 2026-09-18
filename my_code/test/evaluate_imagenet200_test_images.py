"""Evaluate TinyCLIP zero-shot classification on ImageNet-200 validation.

By default, images are read from the 200 WNID subdirectories under
``ImageNet-200/val``. The WNID parent directory supplies every image's
ground-truth label, so the script writes Top-K predictions and reports Top-1
and Top-K accuracy. An optional ``--labels-csv`` can override these labels.
"""

import argparse
import csv
import json
import math
import sys
from contextlib import nullcontext
from pathlib import Path

import torch
import torch.nn.functional as functional
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = REPOSITORY_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from open_clip import create_model_and_transforms, get_tokenizer


IMAGENET200_DIR = REPOSITORY_ROOT / "my_code" / "data" / "ImageNet-200"
DEFAULT_IMAGES_DIR = IMAGENET200_DIR / "val"
DEFAULT_CLASS_MAP = IMAGENET200_DIR / "map" / "imagenet200_label_map_sorted.csv"
DEFAULT_OUTPUT_DIR = REPOSITORY_ROOT / "my_code" / "test" / "output"
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


class FlatImageDataset(Dataset):
    def __init__(self, image_paths, transform):
        self.image_paths = image_paths
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, index):
        path = self.image_paths[index]
        with Image.open(path) as image:
            image = image.convert("RGB")
        return self.transform(image), path.name, str(path)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--images-dir", type=Path, default=DEFAULT_IMAGES_DIR, help="ImageNet-200 validation root with WNID subdirectories. Default: %(default)s")
    parser.add_argument("--class-map", type=Path, default=DEFAULT_CLASS_MAP, help="ImageNet-200 class-map CSV containing label, wnid, and class_name columns. Default: %(default)s")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR, help="Directory for prediction CSV, summary JSON, and optional accuracy report. Default: %(default)s")
    parser.add_argument("--output-csv", type=Path, default=None, help="Prediction CSV path. Default: <output-dir>/predictions.csv")
    parser.add_argument("--summary-json", type=Path, default=None, help="Evaluation summary JSON path. Default: <output-dir>/summary.json")
    parser.add_argument("--model", default="TinyCLIP-ViT-40M-32-Text-19M", help="TinyCLIP model configuration name. Default: %(default)s")
    parser.add_argument("--pretrained", default="LAION400M", help="Registered TinyCLIP pretrained tag or local base checkpoint loaded before --checkpoint. Default: %(default)s")
    parser.add_argument("--checkpoint", type=Path, default=None, help="Optional full-finetuning checkpoint from train_n2_fourier_hard_negative.py. It overrides matching model parameters after --pretrained is loaded. Default: none")
    parser.add_argument("--cache-dir", type=Path, default=None, help="Optional cache directory when --pretrained is a registered tag. Default: open_clip cache")
    parser.add_argument("--batch-size", type=int, default=128, help="Images processed per GPU inference batch. Lower this if CUDA runs out of memory. Default: %(default)s")
    parser.add_argument("--num-workers", type=int, default=4, help="DataLoader workers. Use 0 to debug local image-read errors. Default: %(default)s")
    parser.add_argument("--device", default="cuda", help="Inference device. Default: %(default)s. Use cpu only when GPU execution is intentionally unavailable.")
    parser.add_argument("--precision", choices=("amp", "amp_bfloat16", "fp32"), default="amp", help="Inference precision. Default: %(default)s")
    parser.add_argument("--top-k", type=int, default=5, help="Number of highest-scoring classes written per image. Must be in [1, number of classes]. Default: %(default)s")
    parser.add_argument("--template", action="append", default=None, help="Prompt template containing one '{}' placeholder for a class name. Repeat to ensemble templates. Default: 'a photo of a {}.'")
    parser.add_argument("--labels-csv", type=Path, default=None, help="Optional ground-truth CSV overriding labels inferred from each image's WNID parent directory. Default: infer WNID labels from --images-dir")
    parser.add_argument("--label-image-column", default="ImageName", help="Image-name column in --labels-csv. Values are matched to test filenames. Default: %(default)s")
    parser.add_argument("--label-column", default="Label", help="Ground-truth label column in --labels-csv. Values may be class indices, WNIDs, or class names. Default: %(default)s")
    parser.add_argument("--label-format", choices=("auto", "index", "wnid", "class_name"), default="auto", help="Interpretation of --label-column values. auto tries index, then WNID, then class name. Default: %(default)s")
    parser.add_argument("--progress-refresh-seconds", type=float, default=0.1, help="Minimum seconds between real-time tqdm refreshes. Default: %(default)s")
    args = parser.parse_args()
    if args.batch_size <= 0 or args.num_workers < 0:
        parser.error("--batch-size must be positive and --num-workers must be nonnegative")
    if args.top_k <= 0:
        parser.error("--top-k must be positive")
    if args.progress_refresh_seconds <= 0.0:
        parser.error("--progress-refresh-seconds must be positive")
    return args


def ensure_device(device_name):
    device = torch.device(device_name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("--device {} was requested, but CUDA is unavailable".format(device_name))
    return device


def autocast_context(precision, device):
    if device.type != "cuda" or precision == "fp32":
        return nullcontext()
    if precision == "amp_bfloat16":
        return torch.amp.autocast("cuda", dtype=torch.bfloat16)
    return torch.amp.autocast("cuda")


def load_class_map(path):
    required = {"label", "wnid", "class_name"}
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        missing = required.difference(reader.fieldnames or [])
        if missing:
            raise ValueError("Class map {} lacks columns: {}".format(path, ", ".join(sorted(missing))))
        classes = []
        for row_number, row in enumerate(reader, start=2):
            try:
                label = int((row.get("label") or "").strip())
            except ValueError as error:
                raise ValueError("Invalid class label at {}:{}".format(path, row_number)) from error
            wnid = (row.get("wnid") or "").strip()
            class_name = (row.get("class_name") or "").strip()
            if not wnid or not class_name:
                raise ValueError("Empty WNID or class name at {}:{}".format(path, row_number))
            classes.append({"label": label, "wnid": wnid, "class_name": class_name})
    classes.sort(key=lambda item: item["label"])
    if [item["label"] for item in classes] != list(range(len(classes))):
        raise ValueError("Class-map labels must be unique contiguous values beginning at 0")
    return classes


def discover_images(images_dir):
    paths = sorted(
        path for path in images_dir.rglob("*")
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
    )
    if not paths:
        raise ValueError("No supported images found under {}".format(images_dir))
    return paths


def load_checkpoint(model, checkpoint_path, device):
    # This checkpoint is expected to be a trusted local training artifact. It
    # stores optimizer and RNG state in addition to model tensors, so PyTorch
    # 2.6+ cannot load it with its default weights_only=True setting.
    payload = torch.load(checkpoint_path, map_location=device, weights_only=False)
    if isinstance(payload, dict) and "model" in payload:
        state_dict = payload["model"]
    elif isinstance(payload, dict) and "state_dict" in payload:
        state_dict = payload["state_dict"]
    else:
        state_dict = payload
    if not isinstance(state_dict, dict):
        raise ValueError("Checkpoint has no model state dictionary: {}".format(checkpoint_path))
    if any(key.startswith("module.") for key in state_dict):
        state_dict = {
            key[7:] if key.startswith("module.") else key: value
            for key, value in state_dict.items()
        }
    incompatible = model.load_state_dict(state_dict, strict=False)
    missing = [key for key in incompatible.missing_keys if not key.endswith("num_batches_tracked")]
    unexpected = [key for key in incompatible.unexpected_keys if not key.endswith("num_batches_tracked")]
    if missing or unexpected:
        raise ValueError(
            "Checkpoint is incompatible with --model {}. Missing keys: {}; unexpected keys: {}"
            .format(model, missing[:10], unexpected[:10])
        )


def encode_zero_shot_classifier(model, tokenizer, classes, templates, device, autocast):
    columns = []
    model.eval()
    with torch.inference_mode():
        with tqdm(total=len(classes), desc="Encoding class prompts", unit="class") as progress:
            for item in classes:
                prompts = [template.format(item["class_name"]) for template in templates]
                tokens = tokenizer(prompts).to(device)
                with autocast:
                    text_features = model.encode_text(tokens, normalized=True)
                class_feature = functional.normalize(text_features.float().mean(dim=0), dim=0)
                columns.append(class_feature)
                progress.update(1)
    return torch.stack(columns, dim=1)


def load_ground_truth(path, classes, args):
    class_by_index = {str(item["label"]): item["label"] for item in classes}
    class_by_wnid = {item["wnid"]: item["label"] for item in classes}
    class_by_name = {item["class_name"].casefold(): item["label"] for item in classes}
    labels = {}
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        required = {args.label_image_column, args.label_column}
        missing = required.difference(reader.fieldnames or [])
        if missing:
            raise ValueError("Ground-truth CSV {} lacks columns: {}".format(path, ", ".join(sorted(missing))))
        for row_number, row in enumerate(reader, start=2):
            image_name = (row.get(args.label_image_column) or "").strip()
            raw_label = (row.get(args.label_column) or "").strip()
            if not image_name or not raw_label:
                continue
            formats = (args.label_format,) if args.label_format != "auto" else ("index", "wnid", "class_name")
            parsed = None
            for label_format in formats:
                if label_format == "index":
                    parsed = class_by_index.get(raw_label)
                elif label_format == "wnid":
                    parsed = class_by_wnid.get(raw_label)
                else:
                    parsed = class_by_name.get(raw_label.casefold())
                if parsed is not None:
                    break
            if parsed is None:
                raise ValueError("Unknown label {!r} at {}:{}".format(raw_label, path, row_number))
            labels[image_name] = parsed
    return labels


def infer_ground_truth_from_parent_dirs(image_paths, classes, images_dir):
    """Map validation images to labels from their immediate WNID parent folder."""
    label_by_wnid = {item["wnid"]: item["label"] for item in classes}
    labels = {}
    invalid_paths = []
    for image_path in image_paths:
        try:
            relative_path = image_path.relative_to(images_dir)
        except ValueError:
            invalid_paths.append(str(image_path))
            continue
        if len(relative_path.parts) < 2:
            invalid_paths.append(str(image_path))
            continue
        wnid = relative_path.parts[0]
        label = label_by_wnid.get(wnid)
        if label is None:
            invalid_paths.append(str(image_path))
            continue
        if image_path.name in labels:
            raise ValueError(
                "Duplicate image filename prevents ground-truth matching: {}"
                .format(image_path.name)
            )
        labels[image_path.name] = label
    if invalid_paths:
        raise ValueError(
            "Cannot infer WNID labels for {} images under {}. Expected "
            "<images-dir>/<wnid>/<image>. Examples: {}"
            .format(len(invalid_paths), images_dir, invalid_paths[:3])
        )
    return labels


def prediction_fields(top_k):
    fields = ["ImageName", "ImagePath"]
    for rank in range(1, top_k + 1):
        fields.extend([
            "Top{}_Label".format(rank),
            "Top{}_WNID".format(rank),
            "Top{}_ClassName".format(rank),
            "Top{}_Probability".format(rank),
        ])
    fields.extend(["GroundTruthLabel", "GroundTruthWNID", "GroundTruthClassName", "Top1Correct"])
    return fields


def main():
    args = parse_args()
    images_dir = args.images_dir.expanduser()
    class_map_path = args.class_map.expanduser()
    output_dir = args.output_dir.expanduser()
    output_csv = args.output_csv.expanduser() if args.output_csv else output_dir / "predictions.csv"
    summary_json = args.summary_json.expanduser() if args.summary_json else output_dir / "summary.json"
    checkpoint = args.checkpoint.expanduser() if args.checkpoint else None
    labels_csv = args.labels_csv.expanduser() if args.labels_csv else None
    if not images_dir.is_dir():
        raise FileNotFoundError("Image directory does not exist: {}".format(images_dir))
    if not class_map_path.is_file():
        raise FileNotFoundError("Class-map CSV does not exist: {}".format(class_map_path))
    if checkpoint is not None and not checkpoint.is_file():
        raise FileNotFoundError("Checkpoint does not exist: {}".format(checkpoint))
    if labels_csv is not None and not labels_csv.is_file():
        raise FileNotFoundError("Ground-truth CSV does not exist: {}".format(labels_csv))

    device = ensure_device(args.device)
    classes = load_class_map(class_map_path)
    if args.top_k > len(classes):
        raise ValueError("--top-k {} exceeds the {} classes in {}".format(args.top_k, len(classes), class_map_path))
    image_paths = discover_images(images_dir)
    templates = args.template or ["a photo of a {}."]
    if any(template.count("{}") != 1 for template in templates):
        raise ValueError("Every --template must contain exactly one '{}' placeholder")
    ground_truth = (
        load_ground_truth(labels_csv, classes, args)
        if labels_csv
        else infer_ground_truth_from_parent_dirs(image_paths, classes, images_dir)
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = str(args.cache_dir.expanduser()) if args.cache_dir else None
    model, _, preprocess = create_model_and_transforms(
        args.model,
        pretrained=args.pretrained or "",
        precision="fp32",
        device=device,
        cache_dir=cache_dir,
    )
    if checkpoint is not None:
        load_checkpoint(model, checkpoint, device)
    model.eval()
    tokenizer = get_tokenizer(args.model)
    autocast = autocast_context(args.precision, device)
    classifier = encode_zero_shot_classifier(model, tokenizer, classes, templates, device, autocast)
    dataset = FlatImageDataset(image_paths, preprocess)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )

    fields = prediction_fields(args.top_k)
    temporary = output_csv.with_name(output_csv.name + ".inprogress")
    temporary.unlink(missing_ok=True)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    evaluated_with_labels = 0
    top1_correct = 0
    topk_correct = 0
    with temporary.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        with torch.inference_mode():
            with tqdm(total=len(dataset), desc="Evaluating ImageNet-200 validation", unit="image", mininterval=args.progress_refresh_seconds) as progress:
                for images, image_names, image_paths_batch in loader:
                    images = images.to(device, non_blocking=True)
                    with autocast:
                        image_features = model.encode_image(images, normalized=True)
                    logits = 100.0 * image_features.float() @ classifier
                    probabilities = logits.softmax(dim=1)
                    values, indices = probabilities.topk(args.top_k, dim=1)
                    for image_name, image_path, row_values, row_indices in zip(image_names, image_paths_batch, values.cpu(), indices.cpu()):
                        row = {"ImageName": image_name, "ImagePath": image_path, "GroundTruthLabel": "", "GroundTruthWNID": "", "GroundTruthClassName": "", "Top1Correct": ""}
                        for rank, (probability, index) in enumerate(zip(row_values.tolist(), row_indices.tolist()), start=1):
                            item = classes[index]
                            row.update({
                                "Top{}_Label".format(rank): item["label"],
                                "Top{}_WNID".format(rank): item["wnid"],
                                "Top{}_ClassName".format(rank): item["class_name"],
                                "Top{}_Probability".format(rank): "{:.8f}".format(probability),
                            })
                        if ground_truth is not None and image_name in ground_truth:
                            truth = ground_truth[image_name]
                            truth_item = classes[truth]
                            predicted = row_indices.tolist()
                            top1 = int(predicted[0] == truth)
                            row.update({
                                "GroundTruthLabel": truth,
                                "GroundTruthWNID": truth_item["wnid"],
                                "GroundTruthClassName": truth_item["class_name"],
                                "Top1Correct": top1,
                            })
                            evaluated_with_labels += 1
                            top1_correct += top1
                            topk_correct += int(truth in predicted)
                        writer.writerow(row)
                    progress.update(len(image_names))
    temporary.replace(output_csv)

    summary = {
        "arguments": vars(args),
        "num_images": len(dataset),
        "num_classes": len(classes),
        "prompt_templates": templates,
        "checkpoint": str(checkpoint) if checkpoint else None,
        "predictions_csv": str(output_csv.resolve()),
        "labeled_images": evaluated_with_labels,
        "top1_accuracy": top1_correct / evaluated_with_labels if evaluated_with_labels else None,
        "top{}_accuracy".format(args.top_k): topk_correct / evaluated_with_labels if evaluated_with_labels else None,
    }
    summary_json.parent.mkdir(parents=True, exist_ok=True)
    with summary_json.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=True, indent=2, default=str)
    print("Images evaluated: {}".format(len(dataset)))
    if evaluated_with_labels != len(dataset):
        raise RuntimeError(
            "Accuracy requires one label per image, but only {} of {} images "
            "were labeled".format(evaluated_with_labels, len(dataset))
        )
    print("Validation images: {}; Top-1: {:.4%}; Top-{}: {:.4%}".format(evaluated_with_labels, summary["top1_accuracy"], args.top_k, summary["top{}_accuracy".format(args.top_k)]))
    print("Predictions: {}".format(output_csv.resolve()))
    print("Summary: {}".format(summary_json.resolve()))


if __name__ == "__main__":
    main()
