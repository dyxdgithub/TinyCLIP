"""Evaluate projected-hypersphere TinyCLIP on ImageNet-200 validation.

The script reports TinyCLIP zero-shot Top-1/Top-K classification accuracy and
projected-hypersphere nearest-neighbor retrieval accuracy. It also writes a
class-balanced PCA or t-SNE cluster plot from the learned hypersphere
embeddings. Full and LoRA checkpoints from
``train_n2_fourier_projected_hypersphere.py`` are supported.
"""

import argparse
import csv
import json
import random
import subprocess
import sys
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as functional
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

try:
    from torch.utils.tensorboard import SummaryWriter
except ImportError:
    SummaryWriter = None


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
EXPECTED_OBJECTIVE = "clip_plus_projected_hypersphere_supcon_v1"
CLUSTER_RENDERER = TEST_DIR / "render_embedding_cluster_scatter.py"


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
    """Evaluation architecture matching the training checkpoint."""

    def __init__(self, clip_model, input_dim, projection_dim):
        super().__init__()
        self.clip_model = clip_model
        self.metric_projection = nn.Linear(input_dim, projection_dim)


def exponential_map_at_origin(vectors, curvature, epsilon):
    vectors = vectors.float()
    sqrt_curvature = np.sqrt(curvature)
    norms = vectors.norm(dim=-1, keepdim=True).clamp_min(epsilon)
    maximum_norm = (np.pi / 2.0 - epsilon) / sqrt_curvature
    safe_norms = norms.clamp_max(maximum_norm)
    scale = torch.tan(sqrt_curvature * safe_norms) / (
        sqrt_curvature * safe_norms
    )
    return vectors * (safe_norms / norms) * scale


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True, help="Trusted best.pt or last.pt from train_n2_fourier_projected_hypersphere.py. Required.")
    parser.add_argument("--images-dir", type=Path, default=DEFAULT_IMAGES_DIR, help="ImageNet-200 validation root containing WNID folders. Default: %(default)s")
    parser.add_argument("--class-map", type=Path, default=DEFAULT_CLASS_MAP, help="CSV containing label, wnid, and class_name. Default: %(default)s")
    parser.add_argument("--output-dir", type=Path, default=None, help="Output directory. Default: <script-dir>/output/projected_hypersphere_<mode>_<checkpoint-name>")
    parser.add_argument("--tensorboard-dir", type=Path, default=None, help="TensorBoard event directory. Default: <output-dir>/tensorboard")
    parser.add_argument("--model", default=None, help="TinyCLIP model name override. Default: checkpoint value")
    parser.add_argument("--pretrained", default=None, help="Base pretrained tag or local checkpoint loaded before fine-tuned weights. Default: checkpoint value")
    parser.add_argument("--cache-dir", type=Path, default=None, help="Optional cache for registered pretrained weights. Default: open_clip cache")
    parser.add_argument("--fine-tune-mode", choices=("auto", "full", "lora"), default="auto", help="Checkpoint architecture. auto reads checkpoint metadata. Default: %(default)s")
    parser.add_argument("--lora-rank", type=int, default=None, help="LoRA rank override. Default: checkpoint value")
    parser.add_argument("--lora-alpha", type=float, default=None, help="LoRA alpha override. Default: checkpoint value")
    parser.add_argument("--lora-dropout", type=float, default=None, help="LoRA dropout override. Default: checkpoint value")
    parser.add_argument("--batch-size", type=int, default=128, help="Images per feature-extraction batch. Default: %(default)s")
    parser.add_argument("--retrieval-query-batch-size", type=int, default=512, help="Queries per projected-hypersphere distance block. Lower this if CUDA runs out of memory. Default: %(default)s")
    parser.add_argument("--num-workers", type=int, default=4, help="DataLoader workers; use 0 for in-process image loading. Default: %(default)s")
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda", help="Evaluation device type. Default: %(default)s")
    parser.add_argument("--gpu", type=int, choices=(0, 1, 2, 3), default=0, help="CUDA GPU index. Choices: 0, 1, 2, 3. Default: %(default)s")
    parser.add_argument("--precision", choices=("amp", "amp_bfloat16", "fp32"), default="amp", help="Feature-extraction precision. Geometry calculations always use float32. Default: %(default)s")
    parser.add_argument("--top-k", type=int, default=5, help="Zero-shot Top-K accuracy and predictions. Default: %(default)s")
    parser.add_argument("--retrieval-k", type=int, default=5, help="Projected-hypersphere Recall@K; 1-NN ACC is always calculated. Default: %(default)s")
    parser.add_argument("--template", action="append", default=None, help="Prompt template containing one '{}' placeholder. Repeat to ensemble prompts. Default: 'a photo of a {}.'")
    parser.add_argument("--plot-method", choices=("tsne", "pca"), default="tsne", help="Two-dimensional plot method. t-SNE first applies PCA to at most 50 dimensions. Default: %(default)s")
    parser.add_argument("--plot-max-samples", type=int, default=4000, help="Maximum class-balanced samples used in the plot. Metrics use all validation images. Default: %(default)s")
    parser.add_argument("--tsne-perplexity", type=float, default=30.0, help="t-SNE perplexity; must be below the plotted sample count. Default: %(default)s")
    parser.add_argument("--plot-dpi", type=int, default=220, help="Scatter plot PNG resolution. Default: %(default)s")
    parser.add_argument("--seed", type=int, default=42, help="Plot sampling and dimensionality-reduction seed. Default: %(default)s")
    parser.add_argument("--progress-refresh-seconds", type=float, default=0.1, help="Minimum tqdm refresh interval. Default: %(default)s")
    args = parser.parse_args()
    positive_names = (
        "batch_size", "retrieval_query_batch_size", "top_k", "retrieval_k",
        "plot_max_samples", "plot_dpi",
    )
    for name in positive_names:
        if getattr(args, name) <= 0:
            parser.error("--{} must be positive".format(name.replace("_", "-")))
    if args.num_workers < 0:
        parser.error("--num-workers must be nonnegative")
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


def resolve_value(name, requested, saved, required=True):
    if requested is not None and saved is not None and requested != saved:
        raise ValueError(
            "--{} {} does not match checkpoint value {}".format(
                name.replace("_", "-"), requested, saved
            )
        )
    value = requested if requested is not None else saved
    if required and value is None:
        raise ValueError(
            "Checkpoint lacks {}; pass --{} explicitly".format(
                name, name.replace("_", "-")
            )
        )
    return value


def resolve_checkpoint_configuration(args, payload):
    if not isinstance(payload, dict) or "model" not in payload:
        raise ValueError("Checkpoint must contain a model state dictionary")
    saved_args = payload.get("args", {})
    if not isinstance(saved_args, dict):
        saved_args = {}
    geometry = payload.get("geometry_config", {})
    if not isinstance(geometry, dict):
        geometry = {}
    objective = geometry.get("objective")
    if objective not in (None, EXPECTED_OBJECTIVE):
        raise ValueError("Unsupported checkpoint objective: {!r}".format(objective))

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

    model_name = resolve_value(
        "model", args.model, payload.get("model_name") or saved_args.get("model")
    )
    pretrained = resolve_value(
        "pretrained", args.pretrained, saved_args.get("pretrained")
    )
    projection_dim = int(resolve_value(
        "projection_dim", None,
        geometry.get("projection_dim") or saved_args.get("projection_dim")
    ))
    curvature = float(resolve_value(
        "curvature", None, geometry.get("curvature") or saved_args.get("curvature")
    ))
    epsilon = float(saved_args.get("geometry_epsilon", 1e-6))
    if projection_dim <= 0 or curvature <= 0.0 or not 0.0 < epsilon < 0.1:
        raise ValueError("Invalid projected-hypersphere configuration in checkpoint")

    lora_config = None
    if mode == "lora":
        rank = int(resolve_value("lora_rank", args.lora_rank, saved_args.get("lora_rank")))
        alpha = float(resolve_value("lora_alpha", args.lora_alpha, saved_args.get("lora_alpha")))
        dropout = float(resolve_value("lora_dropout", args.lora_dropout, saved_args.get("lora_dropout")))
        if rank <= 0 or alpha <= 0.0 or not 0.0 <= dropout < 1.0:
            raise ValueError("Invalid LoRA configuration in checkpoint")
        lora_config = (rank, alpha, dropout)
    return {
        "mode": mode,
        "model": model_name,
        "pretrained": pretrained,
        "projection_dim": projection_dim,
        "curvature": curvature,
        "epsilon": epsilon,
        "lora_config": lora_config,
    }


def default_output_dir(mode, checkpoint):
    safe_name = "".join(
        character if character.isalnum() or character in ".-_" else "_"
        for character in checkpoint.stem
    )
    return DEFAULT_OUTPUT_ROOT / "projected_hypersphere_{}_{}".format(
        mode, safe_name
    )


def stratified_plot_indices(labels, maximum, seed):
    if len(labels) <= maximum:
        return np.arange(len(labels), dtype=np.int64)
    generator = random.Random(seed)
    indices_by_label = {}
    for index, label in enumerate(labels):
        indices_by_label.setdefault(int(label), []).append(index)
    for indices in indices_by_label.values():
        generator.shuffle(indices)
    selected = []
    depth = 0
    ordered_labels = sorted(indices_by_label)
    while len(selected) < maximum:
        added = False
        for label in ordered_labels:
            candidates = indices_by_label[label]
            if depth < len(candidates):
                selected.append(candidates[depth])
                added = True
                if len(selected) == maximum:
                    break
        if not added:
            break
        depth += 1
    return np.asarray(selected, dtype=np.int64)


def cross_projected_hypersphere_distances(
    queries, references, curvature, epsilon
):
    query_squared_norms = queries.square().sum(dim=1, keepdim=True)
    reference_squared_norms = references.square().sum(dim=1).unsqueeze(0)
    squared_differences = (
        query_squared_norms
        + reference_squared_norms
        - 2.0 * queries @ references.t()
    ).clamp_min(0.0)
    denominator = (
        (1.0 + curvature * query_squared_norms)
        * (1.0 + curvature * reference_squared_norms)
    ).clamp_min(epsilon)
    cosine_argument = 1.0 - (
        2.0 * curvature * squared_differences / denominator
    )
    cosine_argument = cosine_argument.clamp(-1.0 + epsilon, 1.0 - epsilon)
    return torch.acos(cosine_argument) / np.sqrt(curvature)


@torch.no_grad()
def projected_hypersphere_retrieval(
    features, labels, curvature, epsilon, retrieval_k, query_batch_size,
    device, refresh_seconds
):
    if len(features) < 2:
        raise ValueError("At least two validation images are required for retrieval")
    maximum_k = min(retrieval_k, len(features) - 1)
    references = features.to(device=device, dtype=torch.float32)
    label_tensor = labels.to(device=device, dtype=torch.long)
    nearest_indices = torch.empty(len(features), dtype=torch.long)
    top1_correct = 0
    recall_correct = 0
    with tqdm(
        total=len(features),
        desc="Projected-hypersphere retrieval",
        unit="query",
        mininterval=refresh_seconds,
    ) as progress:
        for start in range(0, len(features), query_batch_size):
            end = min(start + query_batch_size, len(features))
            distances = cross_projected_hypersphere_distances(
                references[start:end], references, curvature, epsilon
            )
            local_rows = torch.arange(end - start, device=device)
            global_rows = torch.arange(start, end, device=device)
            distances[local_rows, global_rows] = float("inf")
            indices = distances.topk(maximum_k, largest=False, dim=1).indices
            query_labels = label_tensor[start:end]
            retrieved_labels = label_tensor[indices]
            matches = retrieved_labels.eq(query_labels.unsqueeze(1))
            top1_correct += int(matches[:, 0].sum().item())
            recall_correct += int(matches.any(dim=1).sum().item())
            nearest_indices[start:end] = indices[:, 0].cpu()
            progress.update(end - start)
    return {
        "projected_hypersphere_1nn_accuracy": top1_correct / len(features),
        "projected_hypersphere_recall_at_{}".format(maximum_k): recall_correct / len(features),
        "effective_retrieval_k": maximum_k,
        "nearest_indices": nearest_indices,
    }


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
        "ZeroShotTop1Correct", "HypersphereNearestImageName",
        "HypersphereNearestImagePath", "HypersphereNearestLabel",
        "HypersphereNearestWNID", "HypersphereNearestClassName",
        "Hypersphere1NNCorrect",
    ])
    return fields


def write_predictions(path, records, classes, top_k):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".inprogress")
    temporary.unlink(missing_ok=True)
    with temporary.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=prediction_fields(top_k))
        writer.writeheader()
        for record in records:
            truth = classes[record["truth"]]
            nearest = classes[record["nearest_truth"]]
            row = {
                "ImageName": record["name"],
                "ImagePath": record["path"],
                "GroundTruthLabel": truth["label"],
                "GroundTruthWNID": truth["wnid"],
                "GroundTruthClassName": truth["class_name"],
                "ZeroShotTop1Correct": int(record["prediction"] == record["truth"]),
                "HypersphereNearestImageName": record["nearest_name"],
                "HypersphereNearestImagePath": record["nearest_path"],
                "HypersphereNearestLabel": nearest["label"],
                "HypersphereNearestWNID": nearest["wnid"],
                "HypersphereNearestClassName": nearest["class_name"],
                "Hypersphere1NNCorrect": int(record["nearest_truth"] == record["truth"]),
            }
            for rank, (probability, class_index) in enumerate(
                zip(record["top_probabilities"], record["top_indices"]), start=1
            ):
                item = classes[class_index]
                row.update({
                    "Top{}_Label".format(rank): item["label"],
                    "Top{}_WNID".format(rank): item["wnid"],
                    "Top{}_ClassName".format(rank): item["class_name"],
                    "Top{}_Probability".format(rank): "{:.8f}".format(probability),
                })
            writer.writerow(row)
    temporary.replace(path)


def render_cluster_artifacts(
    output_dir, features, records, classes, coordinates_path, plot_path, args
):
    if not CLUSTER_RENDERER.is_file():
        raise FileNotFoundError(
            "Cluster renderer does not exist: {}".format(CLUSTER_RENDERER)
        )
    features_path = output_dir / ".cluster_features.npy"
    records_path = output_dir / ".cluster_records.csv"
    np.save(features_path, features)
    with records_path.open("w", encoding="utf-8-sig", newline="") as handle:
        fields = [
            "ImageName", "ImagePath", "GroundTruthLabel", "GroundTruthWNID",
            "GroundTruthClassName", "HypersphereNearestLabel",
            "Hypersphere1NNCorrect",
        ]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for record in records:
            truth = classes[record["truth"]]
            writer.writerow({
                "ImageName": record["name"],
                "ImagePath": record["path"],
                "GroundTruthLabel": truth["label"],
                "GroundTruthWNID": truth["wnid"],
                "GroundTruthClassName": truth["class_name"],
                "HypersphereNearestLabel": record["nearest_truth"],
                "Hypersphere1NNCorrect": int(
                    record["nearest_truth"] == record["truth"]
                ),
            })
    command = [
        sys.executable,
        str(CLUSTER_RENDERER),
        "--features", str(features_path),
        "--records", str(records_path),
        "--coordinates-output", str(coordinates_path),
        "--plot-output", str(plot_path),
        "--method", args.plot_method,
        "--perplexity", str(args.tsne_perplexity),
        "--seed", str(args.seed),
        "--dpi", str(args.plot_dpi),
        "--class-count", str(len(classes)),
        "--title", "Projected-hypersphere ImageNet-200 validation embeddings",
        "--correct-label", "Hypersphere 1-NN correct",
        "--incorrect-label", "Hypersphere 1-NN incorrect",
    ]
    try:
        subprocess.run(command, check=True)
    finally:
        features_path.unlink(missing_ok=True)
        records_path.unlink(missing_ok=True)


def write_tensorboard(path, metrics, payload):
    if SummaryWriter is None:
        raise RuntimeError("TensorBoard output requires the tensorboard package")
    writer = SummaryWriter(log_dir=str(path), flush_secs=1)
    step = int(payload.get("epoch", 0)) if isinstance(payload, dict) else 0
    for name, value in metrics.items():
        if isinstance(value, (int, float)) and name != "effective_retrieval_k":
            writer.add_scalar("validation/{}".format(name), float(value), step)
    writer.flush()
    writer.close()


def main():
    args = parse_args()
    checkpoint = args.checkpoint.expanduser()
    images_dir = args.images_dir.expanduser()
    class_map_path = args.class_map.expanduser()
    if not checkpoint.is_file():
        raise FileNotFoundError("Checkpoint does not exist: {}".format(checkpoint))
    if not images_dir.is_dir():
        raise FileNotFoundError("ImageNet-200 validation directory does not exist: {}".format(images_dir))
    if not class_map_path.is_file():
        raise FileNotFoundError("Class-map CSV does not exist: {}".format(class_map_path))

    device = ensure_device(args.device, args.gpu)
    print("Evaluation device: {}".format(device), flush=True)
    payload = imagenet_eval.read_checkpoint(checkpoint, device)
    config = resolve_checkpoint_configuration(args, payload)
    output_dir = (
        args.output_dir.expanduser()
        if args.output_dir is not None
        else default_output_dir(config["mode"], checkpoint)
    )
    tensorboard_dir = (
        args.tensorboard_dir.expanduser()
        if args.tensorboard_dir is not None
        else output_dir / "tensorboard"
    )
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
    clip_model, _, preprocess = create_model_and_transforms(
        config["model"], pretrained=config["pretrained"], precision="fp32",
        device=device, cache_dir=cache_dir,
    )
    if config["lora_config"] is not None:
        rank, alpha, dropout = config["lora_config"]
        replacements = imagenet_eval.add_lora_adapters(
            clip_model, rank, alpha, dropout
        )
        print("Rebuilt LoRA adapters: {}".format(replacements), flush=True)
    embedding_dim = infer_clip_embedding_dimension(clip_model)
    model = ProjectedHypersphereModel(
        clip_model, embedding_dim, config["projection_dim"]
    ).to(device)
    imagenet_eval.load_checkpoint(model, payload, checkpoint, strict=True)
    model.eval()

    tokenizer = get_tokenizer(config["model"])
    autocast = autocast_context(args.precision, device)
    classifier = imagenet_eval.encode_zero_shot_classifier(
        model.clip_model, tokenizer, classes, templates, device, autocast
    )
    dataset = imagenet_eval.FlatImageDataset(image_paths, preprocess)
    loader = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=device.type == "cuda",
        persistent_workers=False,
    )

    records = []
    projected_batches = []
    label_values = []
    zero_shot_top1_correct = 0
    zero_shot_topk_correct = 0
    with torch.inference_mode():
        with tqdm(
            total=len(dataset), desc="Evaluating projected-hypersphere model",
            unit="image", mininterval=args.progress_refresh_seconds,
        ) as progress:
            for images, image_names, image_paths_batch in loader:
                images = images.to(device, non_blocking=True)
                with autocast:
                    raw_features = model.clip_model.encode_image(
                        images, normalized=False
                    )
                    clip_features = functional.normalize(raw_features, dim=-1)
                    metric_features = model.metric_projection(raw_features)
                projected_features = exponential_map_at_origin(
                    metric_features, config["curvature"], config["epsilon"]
                )
                probabilities = (
                    100.0 * clip_features.float() @ classifier
                ).softmax(dim=1)
                values, indices = probabilities.topk(args.top_k, dim=1)
                projected_batches.append(projected_features.cpu())
                for name, path, row_values, row_indices in zip(
                    image_names, image_paths_batch, values.cpu(), indices.cpu()
                ):
                    truth = ground_truth[name]
                    top_indices = row_indices.tolist()
                    prediction = top_indices[0]
                    zero_shot_top1_correct += int(prediction == truth)
                    zero_shot_topk_correct += int(truth in top_indices)
                    label_values.append(truth)
                    records.append({
                        "name": name, "path": path, "truth": truth,
                        "prediction": prediction,
                        "top_indices": top_indices,
                        "top_probabilities": row_values.tolist(),
                    })
                progress.update(len(image_names))

    projected_features = torch.cat(projected_batches, dim=0).float()
    labels = torch.tensor(label_values, dtype=torch.long)
    retrieval = projected_hypersphere_retrieval(
        projected_features, labels, config["curvature"], config["epsilon"],
        args.retrieval_k, args.retrieval_query_batch_size, device,
        args.progress_refresh_seconds,
    )
    nearest_indices = retrieval.pop("nearest_indices")
    for index, nearest_index in enumerate(nearest_indices.tolist()):
        nearest_record = records[nearest_index]
        records[index].update({
            "nearest_name": nearest_record["name"],
            "nearest_path": nearest_record["path"],
            "nearest_truth": nearest_record["truth"],
        })
    write_predictions(predictions_path, records, classes, args.top_k)

    labels_numpy = labels.numpy()
    selected_indices = stratified_plot_indices(
        labels_numpy, args.plot_max_samples, args.seed
    )
    selected_features = projected_features[selected_indices].numpy()
    selected_records = [records[index] for index in selected_indices]
    render_cluster_artifacts(
        output_dir, selected_features, selected_records, classes,
        coordinates_path, plot_path, args
    )

    sample_count = len(dataset)
    metrics = {
        "zero_shot_top1_accuracy": zero_shot_top1_correct / sample_count,
        "zero_shot_top{}_accuracy".format(args.top_k): zero_shot_topk_correct / sample_count,
        **retrieval,
    }
    summary = {
        "arguments": vars(args),
        "checkpoint": str(checkpoint.resolve()),
        "checkpoint_epoch": payload.get("epoch"),
        "configuration": config,
        "num_images": sample_count,
        "num_classes": len(classes),
        "metrics": metrics,
        "plot_method": args.plot_method,
        "plot_samples": len(selected_indices),
        "predictions_csv": str(predictions_path.resolve()),
        "cluster_coordinates_csv": str(coordinates_path.resolve()),
        "cluster_scatter_png": str(plot_path.resolve()),
        "tensorboard_dir": str(tensorboard_dir.resolve()),
    }
    with summary_path.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=True, indent=2, default=str)
    write_tensorboard(tensorboard_dir, metrics, payload)

    print("ImageNet-200 val images: {}".format(sample_count))
    print("Zero-shot Top-1 ACC: {:.4%}".format(metrics["zero_shot_top1_accuracy"]))
    print("Zero-shot Top-{} ACC: {:.4%}".format(args.top_k, metrics["zero_shot_top{}_accuracy".format(args.top_k)]))
    print("Projected-hypersphere 1-NN ACC: {:.4%}".format(metrics["projected_hypersphere_1nn_accuracy"]))
    print("Projected-hypersphere Recall@{}: {:.4%}".format(metrics["effective_retrieval_k"], metrics["projected_hypersphere_recall_at_{}".format(metrics["effective_retrieval_k"])]))
    print("Predictions: {}".format(predictions_path.resolve()))
    print("Cluster plot: {}".format(plot_path.resolve()))
    print("Summary: {}".format(summary_path.resolve()))
    print("TensorBoard: {}".format(tensorboard_dir.resolve()))


if __name__ == "__main__":
    main()
