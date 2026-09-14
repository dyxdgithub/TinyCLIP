"""Keep image rows and labels that occur in an n2 hierarchy JSON class tree.

The script recursively collects Open Images LabelName values from the JSON
tree. It retains rows with at least one matching label and rewrites aligned
LabelNames and DisplayNames JSON arrays to remove nonmatching labels.
"""

import argparse
import csv
import json
import shutil
from itertools import islice
from pathlib import Path

from tqdm import tqdm


META_DIR = Path(__file__).resolve().parent.parent
HIERARCHY_DIR = META_DIR / "Hierarchy"
DEFAULT_INPUT = HIERARCHY_DIR / "all" / "Image IDs_with_hierarchy_labels_all.csv"
DEFAULT_CLASS_TREE = HIERARCHY_DIR / "n2" / "Last_Two_Layers_Multi_Members_n2.json"
DEFAULT_OUTPUT = HIERARCHY_DIR / "n2" / "Image IDs_with_last_two_layers_multi_members_n2.csv"


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT, help="Input image CSV. Default: %(default)s")
    parser.add_argument("--class-tree", type=Path, default=DEFAULT_CLASS_TREE, help="n2 class-tree JSON. Default: %(default)s")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT, help="Filtered output CSV. Default: %(default)s")
    parser.add_argument("--labels-column", default="LabelNames", help="JSON label-ID array column. Default: %(default)s")
    parser.add_argument("--display-names-column", default="DisplayNames", help="JSON display-name array aligned with labels. Default: %(default)s")
    parser.add_argument("--label-key", default="LabelName", help="Class-label key in the JSON tree. Default: %(default)s")
    parser.add_argument("--children-key", default="Subcategory", help="Child-node key in the JSON tree. Default: %(default)s")
    parser.add_argument("--rows-per-part", type=int, default=100000, help="Input rows in each resumable part. Default: %(default)s")
    parser.add_argument("--progress-refresh-seconds", type=float, default=0.1, help="Minimum seconds between tqdm refreshes. Default: %(default)s")
    parser.add_argument("--parts-dir", type=Path, default=None, help="Directory for resumable parts. Default: hidden directory beside --output.")
    parser.add_argument("--resume", action="store_true", help="Reuse completed parts after interruption.")
    parser.add_argument("--overwrite", action="store_true", help="Replace an existing output and recovery directory.")
    parser.add_argument("--cleanup-parts", action="store_true", help="Delete recovery parts after successful output creation.")
    args = parser.parse_args()
    if args.rows_per_part <= 0:
        parser.error("--rows-per-part must be positive")
    if args.progress_refresh_seconds <= 0:
        parser.error("--progress-refresh-seconds must be positive")
    return args


def read_header(path):
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return csv.DictReader(handle).fieldnames or []


def collect_tree_labels(node, label_key, children_key, labels):
    if isinstance(node, dict):
        label = node.get(label_key)
        if isinstance(label, str) and label.startswith("/m/"):
            labels.add(label)
        children = node.get(children_key, [])
        if not isinstance(children, list):
            raise ValueError("JSON key '{}' must contain a list when present".format(children_key))
        for child in children:
            collect_tree_labels(child, label_key, children_key, labels)
    elif isinstance(node, list):
        for child in node:
            collect_tree_labels(child, label_key, children_key, labels)
    else:
        raise ValueError("Class-tree root must be a JSON object or list")


def load_tree_labels(class_tree_path, label_key, children_key):
    with class_tree_path.open("r", encoding="utf-8-sig") as handle:
        tree = json.load(handle)
    labels = set()
    collect_tree_labels(tree, label_key, children_key, labels)
    if not labels:
        raise ValueError("No Open Images labels found in {}".format(class_tree_path))
    return labels


def parse_json_array(raw_value, column, input_path, row_number):
    raw_value = (raw_value or "").strip()
    if not raw_value:
        return []
    try:
        values = json.loads(raw_value)
    except json.JSONDecodeError as error:
        raise ValueError(
            "Invalid JSON in column '{}' at {} row {}: {}".format(
                column, input_path, row_number, error
            )
        ) from error
    if not isinstance(values, list):
        raise ValueError("Expected a JSON array in column '{}' at {} row {}".format(column, input_path, row_number))
    return values


def filter_row_labels(row, labels_column, display_names_column, tree_labels, input_path, row_number):
    label_names = parse_json_array(row.get(labels_column), labels_column, input_path, row_number)
    display_names = parse_json_array(row.get(display_names_column), display_names_column, input_path, row_number)
    if len(label_names) != len(display_names):
        raise ValueError(
            "Mismatched {} and {} lengths at {} row {}".format(
                labels_column, display_names_column, input_path, row_number
            )
        )
    kept_pairs = [
        (str(label_name).strip(), display_name)
        for label_name, display_name in zip(label_names, display_names)
        if str(label_name).strip() in tree_labels
    ]
    if not kept_pairs:
        return False
    row[labels_column] = json.dumps([label for label, _ in kept_pairs], ensure_ascii=False, separators=(",", ":"))
    row[display_names_column] = json.dumps([name for _, name in kept_pairs], ensure_ascii=False, separators=(",", ":"))
    return True


def row_batches(reader, size):
    while True:
        rows = list(islice(reader, size))
        if not rows:
            return
        yield rows


def part_path(parts_dir, number):
    return parts_dir / "part_{:06d}.csv".format(number)


def temporary_path(path):
    return path.with_name(path.name + ".inprogress")


def write_part(rows, fields, args, tree_labels, input_path, final_part, first_row_number):
    temporary_part = temporary_path(final_part)
    temporary_part.unlink(missing_ok=True)
    kept_rows = 0
    with temporary_part.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for offset, row in enumerate(rows):
            if filter_row_labels(
                row,
                args.labels_column,
                args.display_names_column,
                tree_labels,
                input_path,
                first_row_number + offset,
            ):
                writer.writerow({field: row.get(field, "") for field in fields})
                kept_rows += 1
    temporary_part.replace(final_part)
    return kept_rows


def build_parts(input_path, fields, args, tree_labels, parts_dir, progress):
    parts = []
    completed_parts = 0
    input_rows = 0
    newly_processed_rows = 0
    newly_retained_rows = 0
    with input_path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        for part_number, rows in enumerate(row_batches(reader, args.rows_per_part), start=1):
            final_part = part_path(parts_dir, part_number)
            parts.append(final_part)
            first_row_number = input_rows + 2
            input_rows += len(rows)
            if final_part.is_file():
                completed_parts += 1
                progress.update(len(rows))
                continue
            newly_retained_rows += write_part(
                rows, fields, args, tree_labels, input_path, final_part, first_row_number
            )
            newly_processed_rows += len(rows)
            progress.update(len(rows))
    return parts, completed_parts, input_rows, newly_processed_rows, newly_retained_rows


def combine_parts(parts, output_path, fields, refresh_seconds):
    temporary_output = temporary_path(output_path)
    temporary_output.unlink(missing_ok=True)
    with temporary_output.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for part in tqdm(parts, desc="Combining parts", unit="part", mininterval=refresh_seconds):
            with part.open("r", encoding="utf-8-sig", newline="") as part_handle:
                for row in csv.DictReader(part_handle):
                    writer.writerow(row)
    temporary_output.replace(output_path)


def validate_state(output_path, parts_dir, overwrite, resume):
    if overwrite:
        output_path.unlink(missing_ok=True)
        if parts_dir.exists():
            shutil.rmtree(parts_dir)
        return
    if output_path.exists():
        raise FileExistsError("Output already exists: {}. Use --overwrite to replace it.".format(output_path))
    if parts_dir.exists() and any(parts_dir.iterdir()) and not resume:
        raise FileExistsError("Recovery data exists in {}. Use --resume or --overwrite.".format(parts_dir))


def count_rows(input_path, refresh_seconds):
    row_count = 0
    with input_path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.reader(handle)
        next(reader, None)
        with tqdm(desc="Counting input rows", unit="row", mininterval=refresh_seconds) as progress:
            for row_count, _ in enumerate(reader, start=1):
                progress.update(1)
    return row_count


def main():
    args = parse_args()
    input_path = args.input.expanduser()
    class_tree_path = args.class_tree.expanduser()
    output_path = args.output.expanduser()
    parts_dir = args.parts_dir.expanduser() if args.parts_dir else output_path.parent / ".{}_parts".format(output_path.stem)
    if not input_path.is_file():
        raise FileNotFoundError("Input CSV does not exist: {}".format(input_path))
    if not class_tree_path.is_file():
        raise FileNotFoundError("Class-tree JSON does not exist: {}".format(class_tree_path))
    fields = read_header(input_path)
    missing_columns = [
        column
        for column in (args.labels_column, args.display_names_column)
        if column not in fields
    ]
    if missing_columns:
        raise ValueError("Input CSV is missing required columns: {}".format(", ".join(missing_columns)))
    validate_state(output_path, parts_dir, args.overwrite, args.resume)
    tree_labels = load_tree_labels(class_tree_path, args.label_key, args.children_key)
    parts_dir.mkdir(parents=True, exist_ok=True)
    total_rows = count_rows(input_path, args.progress_refresh_seconds)
    with tqdm(total=total_rows, desc="Filtering n2 labels", unit="row", mininterval=args.progress_refresh_seconds) as progress:
        parts, completed_parts, input_rows, newly_processed_rows, newly_retained_rows = build_parts(
            input_path, fields, args, tree_labels, parts_dir, progress
        )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    combine_parts(parts, output_path, fields, args.progress_refresh_seconds)
    print("n2 JSON labels: {}".format(len(tree_labels)))
    print("Input rows: {}".format(input_rows))
    print("Reused completed parts: {}".format(completed_parts))
    print("Newly processed rows: {}".format(newly_processed_rows))
    print("Newly retained rows: {}".format(newly_retained_rows))
    print("Output CSV: {}".format(output_path.resolve()))
    if args.cleanup_parts:
        shutil.rmtree(parts_dir)
        print("Removed recovery directory: {}".format(parts_dir.resolve()))
    else:
        print("Recovery directory: {}".format(parts_dir.resolve()))


if __name__ == "__main__":
    main()
