"""Remove rows whose Human_LabelName is a top-level n2 category."""

import argparse
import csv
import json
from pathlib import Path

from tqdm import tqdm


META_DIR = Path(__file__).resolve().parent
DEFAULT_INPUT = (
    META_DIR / "Hierarchy" / "n2" / "audio_filename_annotations_n2.csv"
)
DEFAULT_JSON = (
    META_DIR / "Hierarchy" / "n2" / "Last_Two_Layers_Multi_Members_n2.json"
)
DEFAULT_OUTPUT = (
    META_DIR / "Hierarchy" / "n2" / "audio_filename_annotations_n2_leaf_only.csv"
)


def load_top_level_labels(json_path):
    with Path(json_path).open("r", encoding="utf-8") as handle:
        root = json.load(handle)
    children = root.get("Subcategory", [])
    if not isinstance(children, list):
        raise ValueError("Root Subcategory must be a list")
    labels = set()
    for child in children:
        if not isinstance(child, dict) or not child.get("LabelName"):
            raise ValueError("Every top-level category must contain LabelName")
        labels.add(str(child["LabelName"]).strip())
    return labels


def remove_top_level_rows(input_path, json_path, output_path):
    top_level_labels = load_top_level_labels(json_path)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    total_rows = 0
    removed_rows = 0
    kept_rows = 0
    removed_labels = set()
    with Path(input_path).open("r", encoding="utf-8-sig", newline="") as source:
        reader = csv.DictReader(source)
        fields = reader.fieldnames or []
        if "Human_LabelName" not in fields:
            raise ValueError("Input CSV must contain Human_LabelName")
        with output_path.open("w", encoding="utf-8-sig", newline="") as target:
            writer = csv.DictWriter(target, fieldnames=fields)
            writer.writeheader()
            for row in tqdm(reader, desc="Filtering top-level categories", unit="row"):
                total_rows += 1
                label = str(row.get("Human_LabelName", "")).strip()
                if label in top_level_labels:
                    removed_rows += 1
                    removed_labels.add(label)
                    continue
                writer.writerow(row)
                kept_rows += 1
    print("Top-level labels in JSON: {}".format(len(top_level_labels)))
    print("Rows read: {}".format(total_rows))
    print("Rows removed: {}".format(removed_rows))
    print("Rows kept: {}".format(kept_rows))
    print("Removed labels found: {}".format(len(removed_labels)))
    print("Output: {}".format(output_path.resolve()))
    return kept_rows


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--json", type=Path, default=DEFAULT_JSON)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def main():
    args = parse_args()
    remove_top_level_rows(args.input, args.json, args.output)


if __name__ == "__main__":
    main()
