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


def render_plot(path, records, coordinates, args, selection="all"):
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
    if selection not in ("all", "correct", "incorrect"):
        raise ValueError("Unknown cluster selection: {}".format(selection))
    figure, axis = plt.subplots(figsize=(13, 10), constrained_layout=True)
    for kind, mask, label, marker, size, alpha, linewidth in (
        ("correct", correct, args.correct_label, "o", 10, 0.72, 0),
        ("incorrect", ~correct, args.incorrect_label, "x", 18, 0.85, 0.6),
    ):
        if selection != "all" and selection != kind:
            continue
        if mask.any():
            axis.scatter(
                coordinates[mask, 0], coordinates[mask, 1],
                c=labels[mask], cmap="turbo", vmin=0,
                vmax=max(1, args.class_count - 1), s=size, alpha=alpha,
                marker=marker, linewidths=linewidth, label=label,
            )
    if selection != "all" and not (correct if selection == "correct" else ~correct).any():
        axis.text(0.5, 0.5, "No {} samples".format(selection),
                  transform=axis.transAxes, ha="center", va="center")
    color_scale = plt.cm.ScalarMappable(
        norm=plt.Normalize(vmin=0, vmax=max(1, args.class_count - 1)),
        cmap="turbo",
    )
    colorbar = figure.colorbar(color_scale, ax=axis, pad=0.01)
    colorbar.set_label("ImageNet-200 class index")
    axis.set_title("{} ({}, {})".format(args.title, args.method.upper(), selection))
    for set_limits, values in ((axis.set_xlim, coordinates[:, 0]),
                               (axis.set_ylim, coordinates[:, 1])):
        low, high = float(values.min()), float(values.max())
        padding = (high - low) * 0.05 or 1.0
        set_limits(low - padding, high + padding)
    axis.set_xlabel("Component 1")
    axis.set_ylabel("Component 2")
    axis.grid(alpha=0.15, linewidth=0.5)
    if axis.get_legend_handles_labels()[0]:
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
    correct_plot = args.plot_output.with_name(
        args.plot_output.stem + "_correct" + args.plot_output.suffix
    )
    incorrect_plot = args.plot_output.with_name(
        args.plot_output.stem + "_incorrect" + args.plot_output.suffix
    )
    render_plot(correct_plot, records, coordinates, args, "correct")
    render_plot(incorrect_plot, records, coordinates, args, "incorrect")
    print("Cluster coordinates: {}".format(args.coordinates_output.resolve()))
    print("Cluster plot: {}".format(args.plot_output.resolve()))
    print("Correct cluster plot: {}".format(correct_plot.resolve()))
    print("Incorrect cluster plot: {}".format(incorrect_plot.resolve()))


if __name__ == "__main__":
    main()
