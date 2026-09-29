"""Render a two-dimensional embedding scatter plot in an isolated process."""

import argparse
import csv
from pathlib import Path

import numpy as np


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features", type=Path, required=True, help="Input NumPy feature matrix. Required.")
    parser.add_argument("--records", type=Path, required=True, help="Input CSV with one metadata row per feature. Required.")
    parser.add_argument("--coordinates-output", type=Path, required=True, help="Output metadata and two-dimensional coordinates CSV. Required.")
    parser.add_argument("--plot-output", type=Path, required=True, help="Output scatter PNG. Required.")
    parser.add_argument("--method", choices=("tsne", "pca"), required=True, help="Dimensionality-reduction method. Required.")
    parser.add_argument("--perplexity", type=float, required=True, help="t-SNE perplexity; ignored by PCA. Required.")
    parser.add_argument("--seed", type=int, required=True, help="Dimensionality-reduction seed. Required.")
    parser.add_argument("--dpi", type=int, required=True, help="Output PNG resolution. Required.")
    parser.add_argument("--class-count", type=int, required=True, help="Number of class indices represented by the color scale. Required.")
    parser.add_argument("--title", required=True, help="Scatter plot title prefix. Required.")
    parser.add_argument("--correct-label", required=True, help="Legend label for correct points. Required.")
    parser.add_argument("--incorrect-label", required=True, help="Legend label for incorrect points. Required.")
    args = parser.parse_args()
    if args.perplexity <= 0.0 or args.dpi <= 0 or args.class_count <= 0:
        parser.error("--perplexity, --dpi, and --class-count must be positive")
    return args


def load_records(path):
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def reduce_features(features, method, perplexity, seed):
    try:
        from sklearn.decomposition import PCA
    except ImportError as error:
        raise RuntimeError("Cluster plotting requires scikit-learn") from error
    components = min(50, features.shape[1], features.shape[0] - 1)
    if components < 2:
        raise ValueError("At least three feature rows are required")
    print(
        "Reducing {} samples from {} to {} PCA dimensions...".format(
            features.shape[0], features.shape[1], components
        ),
        flush=True,
    )
    reduced = PCA(
        n_components=components, random_state=seed
    ).fit_transform(features)
    if method == "pca":
        return reduced[:, :2]
    if perplexity >= len(reduced):
        raise ValueError(
            "t-SNE perplexity {} must be below sample count {}".format(
                perplexity, len(reduced)
            )
        )
    try:
        from sklearn.manifold import TSNE
    except ImportError as error:
        raise RuntimeError("t-SNE requires scikit-learn") from error
    print("Running t-SNE; iterative progress follows...", flush=True)
    return TSNE(
        n_components=2,
        perplexity=perplexity,
        init="pca",
        learning_rate="auto",
        random_state=seed,
        verbose=1,
    ).fit_transform(reduced)


def write_coordinates(path, records, coordinates):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".inprogress")
    temporary.unlink(missing_ok=True)
    fields = list(records[0]) + ["X", "Y"]
    with temporary.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for record, coordinate in zip(records, coordinates):
            writer.writerow({
                **record,
                "X": "{:.8f}".format(float(coordinate[0])),
                "Y": "{:.8f}".format(float(coordinate[1])),
            })
    temporary.replace(path)


def render_plot(path, records, coordinates, args):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as error:
        raise RuntimeError("Cluster plotting requires matplotlib") from error
    labels = np.asarray(
        [int(record["GroundTruthLabel"]) for record in records], dtype=np.int64
    )
    correct = np.asarray(
        [bool(int(record["Hypersphere1NNCorrect"])) for record in records]
    )
    figure, axis = plt.subplots(figsize=(13, 10), constrained_layout=True)
    scatter = axis.scatter(
        coordinates[correct, 0], coordinates[correct, 1],
        c=labels[correct], cmap="turbo", vmin=0,
        vmax=max(1, args.class_count - 1), s=10, alpha=0.72,
        marker="o", linewidths=0, label=args.correct_label,
    )
    if (~correct).any():
        axis.scatter(
            coordinates[~correct, 0], coordinates[~correct, 1],
            c=labels[~correct], cmap="turbo", vmin=0,
            vmax=max(1, args.class_count - 1), s=18, alpha=0.85,
            marker="x", linewidths=0.6, label=args.incorrect_label,
        )
    colorbar = figure.colorbar(scatter, ax=axis, pad=0.01)
    colorbar.set_label("ImageNet-200 class index")
    axis.set_title("{} ({})".format(args.title, args.method.upper()))
    axis.set_xlabel("Component 1")
    axis.set_ylabel("Component 2")
    axis.grid(alpha=0.15, linewidth=0.5)
    axis.legend(loc="best", frameon=True)
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=args.dpi, bbox_inches="tight")
    plt.close(figure)


def main():
    args = parse_args()
    features = np.load(args.features)
    records = load_records(args.records)
    if features.ndim != 2 or len(features) != len(records):
        raise ValueError(
            "Feature matrix and metadata rows do not align: {} / {}".format(
                features.shape, len(records)
            )
        )
    if not records:
        raise ValueError("Metadata CSV contains no records")
    coordinates = reduce_features(
        features, args.method, args.perplexity, args.seed
    )
    write_coordinates(args.coordinates_output, records, coordinates)
    render_plot(args.plot_output, records, coordinates, args)
    print("Cluster coordinates: {}".format(args.coordinates_output.resolve()))
    print("Cluster plot: {}".format(args.plot_output.resolve()))


if __name__ == "__main__":
    main()
