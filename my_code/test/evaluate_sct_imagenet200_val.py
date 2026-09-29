"""Evaluate an SCT-trained TinyCLIP checkpoint on ImageNet-200 validation.

The script performs zero-shot ImageNet-200 classification, reports Top-1 and
Top-5 accuracy, and projects image embeddings to two dimensions for a cluster
scatter plot. It supports both full and LoRA checkpoints produced by
``train_n2_hard_negative_triplets.py``.
"""

import argparse
import csv
import json
import random
import sys
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = REPOSITORY_ROOT / "src"
TEST_DIR = Path(__file__).resolve().parent
for import_path in (SRC_DIR, TEST_DIR):
    if str(import_path) not in sys.path:
        sys.path.insert(0, str(import_path))

from open_clip import create_model_and_transforms, get_tokenizer
import evaluate_imagenet200_test_images as imagenet_eval


IMAGENET200_DIR = REPOSITORY_ROOT / "my_code" / "data" / "ImageNet-200"
DEFAULT_IMAGES_DIR = IMAGENET200_DIR / "val"
DEFAULT_CLASS_MAP = (
    IMAGENET200_DIR / "map" / "imagenet200_label_map_sorted.csv"
)
DEFAULT_OUTPUT_ROOT = TEST_DIR / "output"


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True, help="Trusted best.pt or last.pt produced by train_n2_hard_negative_triplets.py. Required.")
    parser.add_argument("--images-dir", type=Path, default=DEFAULT_IMAGES_DIR, help="ImageNet-200 validation root containing WNID folders. Default: %(default)s")
    parser.add_argument("--class-map", type=Path, default=DEFAULT_CLASS_MAP, help="CSV containing label, wnid, and class_name. Default: %(default)s")
    parser.add_argument("--output-dir", type=Path, default=None, help="Output directory. Default: <script-dir>/output/sct_<checkpoint-name>")
    parser.add_argument("--model", default=None, help="TinyCLIP model name. Default: value stored in the checkpoint, otherwise TinyCLIP-ViT-40M-32-Text-19M")
    parser.add_argument("--pretrained", default=None, help="Base pretrained tag or local checkpoint loaded before SCT weights. Default: checkpoint value, otherwise LAION400M")
    parser.add_argument("--cache-dir", type=Path, default=None, help="Optional cache for registered pretrained weights. Default: open_clip cache")
    parser.add_argument("--fine-tune-mode", choices=("auto", "full", "lora"), default="auto", help="SCT checkpoint architecture. auto reads checkpoint arguments. Default: %(default)s")
    parser.add_argument("--lora-rank", type=int, default=None, help="LoRA rank override. Default: checkpoint value")
    parser.add_argument("--lora-alpha", type=float, default=None, help="LoRA alpha override. Default: checkpoint value")
    parser.add_argument("--lora-dropout", type=float, default=None, help="LoRA dropout override. Default: checkpoint value")
    parser.add_argument("--batch-size", type=int, default=128, help="Images per inference batch. Default: %(default)s")
    parser.add_argument("--num-workers", type=int, default=4, help="DataLoader workers; 0 loads images in-process. Default: %(default)s")
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda", help="Inference device type. Default: %(default)s")
    parser.add_argument("--gpu", type=int, choices=(0, 1, 2, 3), default=0, help="CUDA GPU index when --device cuda. Choices: 0, 1, 2, 3. Default: %(default)s")
    parser.add_argument("--precision", choices=("amp", "amp_bfloat16", "fp32"), default="amp", help="Inference precision. Default: %(default)s")
    parser.add_argument("--top-k", type=int, default=5, help="Top-K accuracy and predictions. Range: [1, number of classes]. Default: %(default)s")
    parser.add_argument("--template", action="append", default=None, help="Prompt template containing one '{}' placeholder. Repeat to ensemble prompts. Default: 'a photo of a {}.'")
    parser.add_argument("--plot-method", choices=("tsne", "pca"), default="tsne", help="Two-dimensional embedding method. tsne first applies PCA to at most 50 dimensions. Default: %(default)s")
    parser.add_argument("--plot-max-samples", type=int, default=4000, help="Maximum class-balanced samples in the scatter plot. Accuracy still uses every validation image. Default: %(default)s")
    parser.add_argument("--tsne-perplexity", type=float, default=30.0, help="t-SNE perplexity; it must be smaller than the plotted sample count. Default: %(default)s")
    parser.add_argument("--plot-dpi", type=int, default=220, help="Scatter plot PNG resolution. Default: %(default)s")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for plot sampling and dimensionality reduction. Default: %(default)s")
    parser.add_argument("--progress-refresh-seconds", type=float, default=0.1, help="Minimum tqdm refresh interval in seconds. Default: %(default)s")
    args = parser.parse_args()
    if args.batch_size <= 0 or args.num_workers < 0:
        parser.error("--batch-size must be positive and --num-workers nonnegative")
    if args.top_k <= 0 or args.plot_max_samples <= 1 or args.plot_dpi <= 0:
        parser.error("--top-k and --plot-dpi must be positive; --plot-max-samples must exceed 1")
    if args.tsne_perplexity <= 0.0 or args.progress_refresh_seconds <= 0.0:
        parser.error("--tsne-perplexity and --progress-refresh-seconds must be positive")
    if args.lora_rank is not None and args.lora_rank <= 0:
        parser.error("--lora-rank must be positive")
    if args.lora_alpha is not None and args.lora_alpha <= 0.0:
        parser.error("--lora-alpha must be positive")
    if args.lora_dropout is not None and not 0.0 <= args.lora_dropout < 1.0:
        parser.error("--lora-dropout must be in [0, 1)")
    return args


def ensure_device(device_name, gpu_index):
    if device_name == "cpu":
        return torch.device("cpu")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested, but no CUDA device is available")
    if gpu_index >= torch.cuda.device_count():
        raise ValueError(
            "--gpu {} is unavailable; {} CUDA device(s) are visible".format(
                gpu_index, torch.cuda.device_count()
            )
        )
    device = torch.device("cuda", gpu_index)
    torch.cuda.set_device(device)
    return device


def autocast_context(precision, device):
    if device.type != "cuda" or precision == "fp32":
        return nullcontext()
    if precision == "amp_bfloat16":
        return torch.amp.autocast("cuda", dtype=torch.bfloat16)
    return torch.amp.autocast("cuda")


def checkpoint_arguments(payload):
    values = payload.get("args", {}) if isinstance(payload, dict) else {}
    return values if isinstance(values, dict) else {}


def resolve_checkpoint_configuration(args, payload):
    saved_args = checkpoint_arguments(payload)
    saved_mode = payload.get("fine_tune_mode") or saved_args.get("fine_tune_mode")
    if saved_mode not in (None, "full", "lora"):
        raise ValueError("Unsupported checkpoint fine_tune_mode: {!r}".format(saved_mode))
    mode = saved_mode or "full" if args.fine_tune_mode == "auto" else args.fine_tune_mode
    if saved_mode is not None and mode != saved_mode:
        raise ValueError(
            "--fine-tune-mode {} does not match checkpoint mode {}".format(
                mode, saved_mode
            )
        )
    model_name = args.model or saved_args.get("model") or "TinyCLIP-ViT-40M-32-Text-19M"
    pretrained = args.pretrained or saved_args.get("pretrained") or "LAION400M"
    lora_config = None
    if mode == "lora":
        values = {}
        for name in ("lora_rank", "lora_alpha", "lora_dropout"):
            requested = getattr(args, name)
            saved = saved_args.get(name)
            if requested is not None and saved is not None and requested != saved:
                raise ValueError(
                    "--{} {} does not match checkpoint value {}".format(
                        name.replace("_", "-"), requested, saved
                    )
                )
            values[name] = requested if requested is not None else saved
            if values[name] is None:
                raise ValueError(
                    "LoRA checkpoint does not contain {}; pass --{}".format(
                        name, name.replace("_", "-")
                    )
                )
        lora_config = (
            int(values["lora_rank"]),
            float(values["lora_alpha"]),
            float(values["lora_dropout"]),
        )
    return mode, model_name, pretrained, lora_config


def output_directory(args, mode, checkpoint):
    if args.output_dir is not None:
        return args.output_dir.expanduser()
    safe_name = "".join(
        character if character.isalnum() or character in ".-_" else "_"
        for character in checkpoint.stem
    )
    return DEFAULT_OUTPUT_ROOT / "sct_{}_{}".format(mode, safe_name)


def stratified_plot_indices(labels, maximum, seed):
    if len(labels) <= maximum:
        return np.arange(len(labels), dtype=np.int64)
    rng = random.Random(seed)
    by_label = {}
    for index, label in enumerate(labels):
        by_label.setdefault(int(label), []).append(index)
    for indices in by_label.values():
        rng.shuffle(indices)
    selected = []
    depth = 0
    ordered_labels = sorted(by_label)
    while len(selected) < maximum:
        added = False
        for label in ordered_labels:
            indices = by_label[label]
            if depth < len(indices):
                selected.append(indices[depth])
                added = True
                if len(selected) == maximum:
                    break
        if not added:
            break
        depth += 1
    return np.asarray(selected, dtype=np.int64)


def reduce_features(features, method, perplexity, seed):
    try:
        from sklearn.decomposition import PCA
    except ImportError as error:
        raise RuntimeError(
            "Cluster plotting requires scikit-learn from requirements-test.txt"
        ) from error
    component_count = min(50, features.shape[1], features.shape[0] - 1)
    if component_count < 2:
        raise ValueError("At least three plotted samples are required")
    print(
        "Reducing {} samples from {} to {} PCA dimensions...".format(
            features.shape[0], features.shape[1], component_count
        ),
        flush=True,
    )
    pca_features = PCA(
        n_components=component_count, random_state=seed
    ).fit_transform(features)
    if method == "pca":
        return pca_features[:, :2]
    if perplexity >= len(pca_features):
        raise ValueError(
            "--tsne-perplexity {} must be smaller than {} plotted samples".format(
                perplexity, len(pca_features)
            )
        )
    try:
        from sklearn.manifold import TSNE
    except ImportError as error:
        raise RuntimeError(
            "t-SNE plotting requires scikit-learn from requirements-test.txt"
        ) from error
    print("Running t-SNE; iterative progress follows...", flush=True)
    return TSNE(
        n_components=2,
        perplexity=perplexity,
        init="pca",
        learning_rate="auto",
        random_state=seed,
        verbose=1,
    ).fit_transform(pca_features)


def write_cluster_coordinates(path, coordinates, records, classes):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".inprogress")
    temporary.unlink(missing_ok=True)
    fields = [
        "ImageName", "ImagePath", "GroundTruthLabel", "GroundTruthWNID",
        "GroundTruthClassName", "PredictedLabel", "PredictedWNID",
        "PredictedClassName", "Top1Correct", "X", "Y",
    ]
    with temporary.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for coordinate, record in zip(coordinates, records):
            truth = classes[record["truth"]]
            predicted = classes[record["prediction"]]
            writer.writerow({
                "ImageName": record["name"],
                "ImagePath": record["path"],
                "GroundTruthLabel": truth["label"],
                "GroundTruthWNID": truth["wnid"],
                "GroundTruthClassName": truth["class_name"],
                "PredictedLabel": predicted["label"],
                "PredictedWNID": predicted["wnid"],
                "PredictedClassName": predicted["class_name"],
                "Top1Correct": int(record["truth"] == record["prediction"]),
                "X": "{:.8f}".format(float(coordinate[0])),
                "Y": "{:.8f}".format(float(coordinate[1])),
            })
    temporary.replace(path)


def plot_clusters(path, coordinates, records, class_count, method, dpi):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as error:
        raise RuntimeError(
            "Cluster plotting requires matplotlib from requirements-test.txt"
        ) from error
    labels = np.asarray([record["truth"] for record in records])
    correct = np.asarray([
        record["truth"] == record["prediction"] for record in records
    ])
    figure, axis = plt.subplots(figsize=(13, 10), constrained_layout=True)
    scatter = axis.scatter(
        coordinates[correct, 0],
        coordinates[correct, 1],
        c=labels[correct],
        cmap="turbo",
        vmin=0,
        vmax=max(1, class_count - 1),
        s=10,
        alpha=0.72,
        marker="o",
        linewidths=0,
        label="Top-1 correct",
    )
    if (~correct).any():
        axis.scatter(
            coordinates[~correct, 0],
            coordinates[~correct, 1],
            c=labels[~correct],
            cmap="turbo",
            vmin=0,
            vmax=max(1, class_count - 1),
            s=18,
            alpha=0.85,
            marker="x",
            linewidths=0.6,
            label="Top-1 incorrect",
        )
    colorbar = figure.colorbar(scatter, ax=axis, pad=0.01)
    colorbar.set_label("ImageNet-200 class index")
    axis.set_title("SCT TinyCLIP ImageNet-200 validation embeddings ({})".format(method.upper()))
    axis.set_xlabel("Component 1")
    axis.set_ylabel("Component 2")
    axis.grid(alpha=0.15, linewidth=0.5)
    axis.legend(loc="best", frameon=True)
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(figure)


def prediction_fields(top_k):
    fields = ["ImageName", "ImagePath"]
    for rank in range(1, top_k + 1):
        fields.extend([
            "Top{}_Label".format(rank),
            "Top{}_WNID".format(rank),
            "Top{}_ClassName".format(rank),
            "Top{}_Probability".format(rank),
        ])
    fields.extend([
        "GroundTruthLabel", "GroundTruthWNID", "GroundTruthClassName",
        "Top1Correct",
    ])
    return fields


def main():
    args = parse_args()
    checkpoint = args.checkpoint.expanduser()
    images_dir = args.images_dir.expanduser()
    class_map_path = args.class_map.expanduser()
    if not checkpoint.is_file():
        raise FileNotFoundError("SCT checkpoint does not exist: {}".format(checkpoint))
    if not images_dir.is_dir():
        raise FileNotFoundError("ImageNet-200 validation directory does not exist: {}".format(images_dir))
    if not class_map_path.is_file():
        raise FileNotFoundError("Class-map CSV does not exist: {}".format(class_map_path))

    device = ensure_device(args.device, args.gpu)
    print("Evaluation device: {}".format(device), flush=True)
    payload = imagenet_eval.read_checkpoint(checkpoint, device)
    mode, model_name, pretrained, lora_config = resolve_checkpoint_configuration(
        args, payload
    )
    output_dir = output_directory(args, mode, checkpoint)
    output_dir.mkdir(parents=True, exist_ok=True)
    predictions_path = output_dir / "predictions.csv"
    summary_path = output_dir / "summary.json"
    coordinates_path = output_dir / "cluster_coordinates.csv"
    plot_path = output_dir / "cluster_scatter.png"

    classes = imagenet_eval.load_class_map(class_map_path)
    if args.top_k > len(classes):
        raise ValueError("--top-k exceeds the {} available classes".format(len(classes)))
    image_paths = imagenet_eval.discover_images(images_dir)
    ground_truth = imagenet_eval.infer_ground_truth_from_parent_dirs(
        image_paths, classes, images_dir
    )
    templates = args.template or ["a photo of a {}."]
    if any(template.count("{}") != 1 for template in templates):
        raise ValueError("Every --template must contain exactly one '{}' placeholder")

    cache_dir = str(args.cache_dir.expanduser()) if args.cache_dir else None
    model, _, preprocess = create_model_and_transforms(
        model_name,
        pretrained=pretrained,
        precision="fp32",
        device=device,
        cache_dir=cache_dir,
    )
    if lora_config is not None:
        rank, alpha, dropout = lora_config
        replacements = imagenet_eval.add_lora_adapters(
            model, rank, alpha, dropout
        )
        print(
            "Rebuilt SCT LoRA adapters: rank={}, alpha={}, dropout={}; {}".format(
                rank, alpha, dropout, replacements
            ),
            flush=True,
        )
    imagenet_eval.load_checkpoint(
        model, payload, checkpoint, strict=True
    )
    model.eval()
    tokenizer = get_tokenizer(model_name)
    autocast = autocast_context(args.precision, device)
    classifier = imagenet_eval.encode_zero_shot_classifier(
        model, tokenizer, classes, templates, device, autocast
    )
    dataset = imagenet_eval.FlatImageDataset(image_paths, preprocess)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        persistent_workers=False,
    )

    temporary_predictions = predictions_path.with_name(
        predictions_path.name + ".inprogress"
    )
    temporary_predictions.unlink(missing_ok=True)
    all_features = []
    plot_records = []
    top1_correct = 0
    topk_correct = 0
    fields = prediction_fields(args.top_k)
    with temporary_predictions.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        with torch.inference_mode():
            with tqdm(total=len(dataset), desc="Evaluating SCT on ImageNet-200 val", unit="image", mininterval=args.progress_refresh_seconds) as progress:
                for images, image_names, image_paths_batch in loader:
                    images = images.to(device, non_blocking=True)
                    with autocast:
                        image_features = model.encode_image(images, normalized=True)
                    float_features = image_features.float()
                    probabilities = (100.0 * float_features @ classifier).softmax(dim=1)
                    values, indices = probabilities.topk(args.top_k, dim=1)
                    all_features.append(float_features.cpu())
                    for name, path, row_values, row_indices in zip(
                        image_names, image_paths_batch, values.cpu(), indices.cpu()
                    ):
                        truth = ground_truth[name]
                        predicted_indices = row_indices.tolist()
                        prediction = predicted_indices[0]
                        top1 = int(prediction == truth)
                        top1_correct += top1
                        topk_correct += int(truth in predicted_indices)
                        truth_item = classes[truth]
                        row = {
                            "ImageName": name,
                            "ImagePath": path,
                            "GroundTruthLabel": truth_item["label"],
                            "GroundTruthWNID": truth_item["wnid"],
                            "GroundTruthClassName": truth_item["class_name"],
                            "Top1Correct": top1,
                        }
                        for rank_value, (probability, class_index) in enumerate(
                            zip(row_values.tolist(), predicted_indices), start=1
                        ):
                            item = classes[class_index]
                            row.update({
                                "Top{}_Label".format(rank_value): item["label"],
                                "Top{}_WNID".format(rank_value): item["wnid"],
                                "Top{}_ClassName".format(rank_value): item["class_name"],
                                "Top{}_Probability".format(rank_value): "{:.8f}".format(probability),
                            })
                        writer.writerow(row)
                        plot_records.append({
                            "name": name,
                            "path": path,
                            "truth": truth,
                            "prediction": prediction,
                        })
                    progress.update(len(image_names))
    temporary_predictions.replace(predictions_path)

    feature_matrix = torch.cat(all_features, dim=0).numpy()
    labels = np.asarray([record["truth"] for record in plot_records])
    selected_indices = stratified_plot_indices(
        labels, args.plot_max_samples, args.seed
    )
    selected_features = feature_matrix[selected_indices]
    selected_records = [plot_records[index] for index in selected_indices]
    coordinates = reduce_features(
        selected_features,
        args.plot_method,
        args.tsne_perplexity,
        args.seed,
    )
    write_cluster_coordinates(
        coordinates_path, coordinates, selected_records, classes
    )
    plot_clusters(
        plot_path,
        coordinates,
        selected_records,
        len(classes),
        args.plot_method,
        args.plot_dpi,
    )

    sample_count = len(dataset)
    summary = {
        "arguments": vars(args),
        "checkpoint": str(checkpoint.resolve()),
        "checkpoint_epoch": payload.get("epoch") if isinstance(payload, dict) else None,
        "fine_tune_mode": mode,
        "model": model_name,
        "pretrained_base": pretrained,
        "lora_config": (
            {"rank": lora_config[0], "alpha": lora_config[1], "dropout": lora_config[2]}
            if lora_config is not None else None
        ),
        "num_images": sample_count,
        "num_classes": len(classes),
        "top1_accuracy": top1_correct / sample_count,
        "top{}_accuracy".format(args.top_k): topk_correct / sample_count,
        "plot_method": args.plot_method,
        "plot_samples": len(selected_indices),
        "predictions_csv": str(predictions_path.resolve()),
        "cluster_coordinates_csv": str(coordinates_path.resolve()),
        "cluster_scatter_png": str(plot_path.resolve()),
    }
    with summary_path.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=True, indent=2, default=str)

    print(
        "ImageNet-200 val: {} images; Top-1 ACC: {:.4%}; Top-{} ACC: {:.4%}".format(
            sample_count,
            summary["top1_accuracy"],
            args.top_k,
            summary["top{}_accuracy".format(args.top_k)],
        ),
        flush=True,
    )
    print("Predictions: {}".format(predictions_path.resolve()))
    print("Cluster coordinates: {}".format(coordinates_path.resolve()))
    print("Cluster scatter plot: {}".format(plot_path.resolve()))
    print("Summary: {}".format(summary_path.resolve()))


if __name__ == "__main__":
    main()
