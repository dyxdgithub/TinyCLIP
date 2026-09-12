"""Compute class-name similarity separately inside each top-level category."""

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

try:
    from class_name_similarity import compute_metric_matrix, load_embeddings
except ModuleNotFoundError:
    sys.path.insert(0, str(Path(__file__).parent))
    from class_name_similarity import compute_metric_matrix, load_embeddings


DEFAULT_INPUT = Path(__file__).with_name("Hierarchy") / "n2" / (
    "Last_Two_Layers_Multi_Members_n2.json"
)


def load_categories(path):
    with Path(path).open("r", encoding="utf-8") as handle:
        root = json.load(handle)
    categories = root.get("Subcategory")
    if not isinstance(categories, list):
        raise ValueError("JSON root must contain a Subcategory list")
    result = []
    for category in categories:
        children = category.get("Subcategory", [])
        if not isinstance(children, list) or len(children) < 2:
            continue
        result.append({
            "LabelName": category["LabelName"],
            "TextName": category.get("TextName", category["LabelName"]),
            "children": [{
                "LabelName": child["LabelName"],
                "TextName": child.get("TextName", child["LabelName"]),
            } for child in children],
        })
    return result


def rank_group(category, embeddings, top_k, metric, threshold):
    children = category["children"]
    names = [child["TextName"] for child in children]
    indices = [embeddings[name] for name in names]
    matrix = compute_metric_matrix(np.asarray(indices), metric)
    rows = []
    for row_index, child in enumerate(children):
        candidates = [(float(matrix[row_index, col]), col)
                      for col in range(len(children))
                      if col != row_index and
                      (threshold is None or
                       (metric == "cosine" and matrix[row_index, col] > threshold) or
                       (metric != "cosine" and matrix[row_index, col] < threshold))]
        candidates.sort(key=lambda item: ((-item[0] if metric == "cosine" else item[0]), item[1]))
        output = {
            "CategoryLabelName": category["LabelName"],
            "CategoryTextName": category["TextName"],
            "LabelName": child["LabelName"],
            "TextName": child["TextName"],
        }
        for rank in range(top_k):
            if rank < len(candidates):
                score, col = candidates[rank]
                output["Top{}Name".format(rank + 1)] = children[col]["TextName"]
                output["Top{}Value".format(rank + 1)] = score
            else:
                output["Top{}Name".format(rank + 1)] = ""
                output["Top{}Value".format(rank + 1)] = ""
        rows.append(output)
    return rows, matrix


def write_neighbors(path, rows, top_k):
    fields = ["CategoryLabelName", "CategoryTextName", "LabelName", "TextName"]
    fields += [item for rank in range(1, top_k + 1) for item in
               ("Top{}Name".format(rank), "Top{}Value".format(rank))]
    with Path(path).open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def write_pairs(path, categories, matrices):
    with Path(path).open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=[
            "CategoryLabelName", "CategoryTextName", "LabelName1", "TextName1",
            "LabelName2", "TextName2", "MetricValue",
        ])
        writer.writeheader()
        for category, matrix in zip(categories, matrices):
            children = category["children"]
            for first in range(len(children)):
                for second in range(first + 1, len(children)):
                    writer.writerow({
                        "CategoryLabelName": category["LabelName"],
                        "CategoryTextName": category["TextName"],
                        "LabelName1": children[first]["LabelName"],
                        "TextName1": children[first]["TextName"],
                        "LabelName2": children[second]["LabelName"],
                        "TextName2": children[second]["TextName"],
                        "MetricValue": float(matrix[first, second]),
                    })


def write_matrices(directory, categories, matrices):
    directory.mkdir(parents=True, exist_ok=True)
    for category, matrix in zip(categories, matrices):
        keys = ["{:06d}:{}".format(index, child["TextName"])
                for index, child in enumerate(category["children"])]
        safe_name = "{}_{}".format(category["LabelName"].replace("/", "_"),
                                    category["TextName"].replace("/", "_"))
        with (directory / (safe_name + ".csv")).open(
                "w", encoding="utf-8-sig", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(["row_id"] + keys)
            for key, values in zip(keys, matrix):
                writer.writerow([key] + ["{:.8f}".format(float(value))
                                         for value in values])


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=Path,
                        default=Path("Hierarchy/category_name_similarity"))
    parser.add_argument("--model", default="BAAI/bge-m3")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--metric", choices=("cosine", "euclidean", "manhattan"),
                        default="cosine")
    parser.add_argument("--threshold", type=float, default=None,
                        help="Cosine: keep values above; distances: keep values below")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--max-seq-length", type=int, default=None)
    parser.add_argument("--cache", type=Path,
                        default=Path("Hierarchy/category_name_similarity/bge-m3_embeddings.npy"))
    return parser.parse_args()


def main():
    args = parse_args()
    if args.batch_size < 1 or args.top_k < 1:
        raise ValueError("--batch-size and --top-k must be positive")
    if args.threshold is not None and args.metric == "cosine" and not -1.0 <= args.threshold <= 1.0:
        raise ValueError("Cosine --threshold must be between -1 and 1")
    if args.threshold is not None and args.metric != "cosine" and args.threshold < 0:
        raise ValueError("Distance --threshold must be non-negative")
    categories = load_categories(args.input)
    names = list(dict.fromkeys(child["TextName"] for category in categories
                               for child in category["children"]))
    device = args.device
    if device == "auto":
        device = "cuda" if _cuda_available() else "cpu"
    vectors = load_embeddings(args.model, names, args.batch_size, device,
                              args.max_seq_length, args.cache)
    embeddings = dict(zip(names, vectors))
    all_neighbors = []
    matrices = []
    for category in categories:
        neighbors, matrix = rank_group(category, embeddings, args.top_k,
                                       args.metric, args.threshold)
        all_neighbors.extend(neighbors)
        matrices.append(matrix)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_neighbors(args.output_dir / "category_nearest_neighbors.csv",
                    all_neighbors, args.top_k)
    write_pairs(args.output_dir / "category_pairwise_similarity.csv",
                categories, matrices)
    write_matrices(args.output_dir / "category_matrices", categories, matrices)
    print("Processed {} categories and {} unique child classes".format(
        len(categories), len(names)))


def _cuda_available():
    try:
        import torch
        return torch.cuda.is_available()
    except ImportError:
        return False


if __name__ == "__main__":
    main()
