"""Remove n2 hierarchy parent labels while preserving every image row.

The source CSV stores LabelNames and DisplayNames as aligned JSON arrays. A
label is treated as a parent when its matching Open Images JSON tree node has
at least one entry under Subcategory. Parent entries are removed from both
arrays, while leaf labels and rows with no remaining labels are retained.
"""

import argparse
import csv
import json
import shutil
from itertools import islice
from pathlib import Path

from tqdm import tqdm


META_DIR = Path(__file__).resolve().parent.parent
N2_DIR = META_DIR / "Hierarchy" / "n2"
DEFAULT_INPUT = N2_DIR / "Image IDs_with_last_two_layers_multi_members_n2.csv"
DEFAULT_CLASS_TREE = N2_DIR / "Last_Two_Layers_Multi_Members_n2.json"
DEFAULT_OUTPUT = N2_DIR / "Image IDs_with_last_two_layers_multi_members_n2_leaf_labels_only.csv"


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT, help="Input image CSV. Default: %(default)s")
    parser.add_argument("--class-tree", type=Path, default=DEFAULT_CLASS_TREE, help="n2 class-tree JSON. Default: %(default)s")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT, help="Output CSV with n2 parent labels removed. Default: %(default)s")
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


def collect_parent_labels(node, label_key, children_key, parent_labels):
    if isinstance(node, list):
        for child in node:
            collect_parent_labels(child, label_key, children_key, parent_labels)
        return
    if not isinstance(node, dict):
        raise ValueError("Class-tree nodes must be JSON objects or lists")

    children = node.get(children_key, [])
    if not isinstance(children, list):
        raise ValueError("JSON key '{}' must contain a list when present".format(children_key))
    label = node.get(label_key)
    if isinstance(label, str) and label.startswith("/m/") and children:
        parent_labels.add(label)
    for child in children:
        collect_parent_labels(child, label_key, children_key, parent_labels)


def load_parent_labels(class_tree_path, label_key, children_key):
    with class_tree_path.open("r", encoding="utf-8-sig") as handle:
        tree = json.load(handle)
    parent_labels = set()
    collect_parent_labels(tree, label_key, children_key, parent_labels)
    if not parent_labels:
        raise ValueError("No Open Images parent labels found in {}".format(class_tree_path))
    return parent_labels


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
        raise ValueError(
            "Expected a JSON array in column '{}' at {} row {}".format(
                column, input_path, row_number
            )
        )
    return values


def remove_parent_labels(row, labels_column, display_names_column, parent_labels, input_path, row_number):
    label_names = parse_json_array(row.get(labels_column), labels_column, input_path, row_number)
    display_names = parse_json_array(
        row.get(display_names_column), display_names_column, input_path, row_number
    )
    if len(label_names) != len(display_names):
        raise ValueError(
            "Mismatched {} and {} lengths at {} row {}".format(
                labels_column, display_names_column, input_path, row_number
            )
        )

    retained_pairs = [
        (str(label_name).strip(), display_name)
        for label_name, display_name in zip(label_names, display_names)
        if str(label_name).strip() not in parent_labels
    ]
    row[labels_column] = json.dumps(
        [label_name for label_name, _ in retained_pairs],
        ensure_ascii=False,
        separators=(",", ":"),
    )
    row[display_names_column] = json.dumps(
        [display_name for _, display_name in retained_pairs],
        ensure_ascii=False,
        separators=(",", ":"),
    )
    return len(label_names) - len(retained_pairs)


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


def write_part(rows, fields, args, parent_labels, input_path, final_part, first_row_number):
    temporary_part = temporary_path(final_part)
    temporary_part.unlink(missing_ok=True)
    removed_labels = 0
    with temporary_part.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for offset, row in enumerate(rows):
            removed_labels += remove_parent_labels(
                row,
                args.labels_column,
                args.display_names_column,
                parent_labels,
                input_path,
                first_row_number + offset,
            )
            writer.writerow({field: row.get(field, "") for field in fields})
    temporary_part.replace(final_part)
    return removed_labels


def build_parts(input_path, fields, args, parent_labels, parts_dir, progress):
    parts = []
    completed_parts = 0
    input_rows = 0
    newly_processed_rows = 0
    newly_removed_labels = 0
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
            newly_removed_labels += write_part(
                rows, fields, args, parent_labels, input_path, final_part, first_row_number
            )
            newly_processed_rows += len(rows)
            progress.update(len(rows))
    return parts, completed_parts, input_rows, newly_processed_rows, newly_removed_labels


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
    parent_labels = load_parent_labels(class_tree_path, args.label_key, args.children_key)
    parts_dir.mkdir(parents=True, exist_ok=True)
    total_rows = count_rows(input_path, args.progress_refresh_seconds)
    with tqdm(total=total_rows, desc="Removing n2 parent labels", unit="row", mininterval=args.progress_refresh_seconds) as progress:
        parts, completed_parts, input_rows, newly_processed_rows, newly_removed_labels = build_parts(
            input_path, fields, args, parent_labels, parts_dir, progress
        )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    combine_parts(parts, output_path, fields, args.progress_refresh_seconds)
    print("n2 parent labels: {}".format(len(parent_labels)))
    print("Input rows: {}".format(input_rows))
    print("Reused completed parts: {}".format(completed_parts))
    print("Newly processed rows: {}".format(newly_processed_rows))
    print("Newly removed parent-label entries: {}".format(newly_removed_labels))
    print("Output CSV: {}".format(output_path.resolve()))
    if args.cleanup_parts:
        shutil.rmtree(parts_dir)
        print("Removed recovery directory: {}".format(parts_dir.resolve()))
    else:
        print("Recovery directory: {}".format(parts_dir.resolve()))


if __name__ == "__main__":
    main()
