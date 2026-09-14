"""Split n2 leaf-label rows into one image-label row per top-level class.

For each input sample, labels that belong to the same top-level class under
the n2 JSON root are reduced to one randomly selected label. Labels from
different top-level classes are each retained, so the output has one label per
row while preserving at most one label from every top-level class per sample.
When a label occurs beneath multiple top-level classes, it is assigned to the
first class in JSON order by default so each image-label pair has one parent.
"""

import argparse
import csv
import json
import random
import shutil
from itertools import islice
from pathlib import Path

from tqdm import tqdm


META_DIR = Path(__file__).resolve().parent.parent
N2_DIR = META_DIR / "Hierarchy" / "n2"
DEFAULT_INPUT = N2_DIR / "Image IDs_with_last_two_layers_multi_members_n2_leaf_labels_only.csv"
DEFAULT_CLASS_TREE = N2_DIR / "Last_Two_Layers_Multi_Members_n2.json"
DEFAULT_OUTPUT = N2_DIR / "Image IDs_with_last_two_layers_multi_members_n2_one_label_per_parent.csv"


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT, help="Input image CSV. Default: %(default)s")
    parser.add_argument("--class-tree", type=Path, default=DEFAULT_CLASS_TREE, help="n2 class-tree JSON. Default: %(default)s")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT, help="One-label-per-row output CSV. Default: %(default)s")
    parser.add_argument("--labels-column", default="LabelNames", help="JSON label-ID array column. Default: %(default)s")
    parser.add_argument("--display-names-column", default="DisplayNames", help="JSON display-name array aligned with labels. Default: %(default)s")
    parser.add_argument("--image-id-column", default="ImageID", help="Stable image-ID column used for deterministic random selection. Default: %(default)s")
    parser.add_argument("--label-key", default="LabelName", help="Class-label key in the JSON tree. Default: %(default)s")
    parser.add_argument("--text-key", default="TextName", help="Display-name key in the JSON tree. Default: %(default)s")
    parser.add_argument("--children-key", default="Subcategory", help="Child-node key in the JSON tree. Default: %(default)s")
    parser.add_argument("--multi-parent-label-policy", choices=("first", "error"), default="first", help="Handling for labels under multiple top-level classes: 'first' uses the first JSON occurrence, 'error' stops with an error. Default: %(default)s")
    parser.add_argument("--output-label-column", default="LabelName", help="Single label-ID column written to the output. Default: %(default)s")
    parser.add_argument("--output-display-name-column", default="DisplayName", help="Single display-name column written to the output. Default: %(default)s")
    parser.add_argument("--parent-label-column", default="ParentLabelName", help="Top-level class label-ID column written to the output. Default: %(default)s")
    parser.add_argument("--parent-display-name-column", default="ParentDisplayName", help="Top-level class display-name column written to the output. Default: %(default)s")
    parser.add_argument("--random-seed", type=int, default=42, help="Seed for deterministic per-sample label selection. Default: %(default)s")
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
    output_columns = [
        args.output_label_column,
        args.output_display_name_column,
        args.parent_label_column,
        args.parent_display_name_column,
    ]
    if len(set(output_columns)) != len(output_columns):
        parser.error("Output label column names must be unique")
    return args


def read_header(path):
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return csv.DictReader(handle).fieldnames or []


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


def collect_top_level_labels(
    node,
    top_label,
    top_name,
    args,
    label_to_parent,
    multi_parent_assignments,
):
    if not isinstance(node, dict):
        raise ValueError("Class-tree nodes must be JSON objects")
    label = node.get(args.label_key)
    if isinstance(label, str) and label.startswith("/m/"):
        previous_parent = label_to_parent.get(label)
        parent = (top_label, top_name)
        if previous_parent and previous_parent != parent:
            multi_parent_assignments.add((label, previous_parent[0], top_label))
            if args.multi_parent_label_policy == "error":
                raise ValueError(
                    "Label '{}' belongs to multiple top-level classes: '{}' and '{}'".format(
                        label, previous_parent[0], top_label
                    )
                )
        elif not previous_parent:
            label_to_parent[label] = parent
    children = node.get(args.children_key, [])
    if not isinstance(children, list):
        raise ValueError("JSON key '{}' must contain a list when present".format(args.children_key))
    for child in children:
        collect_top_level_labels(
            child,
            top_label,
            top_name,
            args,
            label_to_parent,
            multi_parent_assignments,
        )


def load_label_to_parent(class_tree_path, args):
    with class_tree_path.open("r", encoding="utf-8-sig") as handle:
        root = json.load(handle)
    if not isinstance(root, dict):
        raise ValueError("Class-tree root must be a JSON object")
    top_level_nodes = root.get(args.children_key, [])
    if not isinstance(top_level_nodes, list) or not top_level_nodes:
        raise ValueError("Class-tree root key '{}' must contain a nonempty list".format(args.children_key))

    label_to_parent = {}
    multi_parent_assignments = set()
    for node in top_level_nodes:
        if not isinstance(node, dict):
            raise ValueError("Top-level class-tree entries must be JSON objects")
        top_label = node.get(args.label_key)
        if not isinstance(top_label, str) or not top_label.startswith("/m/"):
            raise ValueError("Every top-level class must have an Open Images label ID")
        top_name = node.get(args.text_key)
        if not isinstance(top_name, str):
            top_name = ""
        collect_top_level_labels(
            node,
            top_label,
            top_name,
            args,
            label_to_parent,
            multi_parent_assignments,
        )
    if not label_to_parent:
        raise ValueError("No Open Images labels found in {}".format(class_tree_path))
    return label_to_parent, multi_parent_assignments


def output_fields(input_fields, args):
    removed_columns = {args.labels_column, args.display_names_column}
    retained_fields = [field for field in input_fields if field not in removed_columns]
    added_fields = [
        args.output_label_column,
        args.output_display_name_column,
        args.parent_label_column,
        args.parent_display_name_column,
    ]
    conflicts = set(retained_fields).intersection(added_fields)
    if conflicts:
        raise ValueError(
            "Input columns conflict with requested output columns: {}".format(", ".join(sorted(conflicts)))
        )
    return retained_fields + added_fields


def choose_rows(row, args, label_to_parent, input_path, row_number):
    label_names = parse_json_array(row.get(args.labels_column), args.labels_column, input_path, row_number)
    display_names = parse_json_array(
        row.get(args.display_names_column), args.display_names_column, input_path, row_number
    )
    if len(label_names) != len(display_names):
        raise ValueError(
            "Mismatched {} and {} lengths at {} row {}".format(
                args.labels_column, args.display_names_column, input_path, row_number
            )
        )
    pairs_by_parent = {}
    for label_name, display_name in zip(label_names, display_names):
        label_name = str(label_name).strip()
        if label_name not in label_to_parent:
            raise ValueError(
                "Label '{}' at {} row {} is absent from the n2 class tree".format(
                    label_name, input_path, row_number
                )
            )
        parent = label_to_parent[label_name]
        pairs_by_parent.setdefault(parent, []).append((label_name, display_name))

    image_id = row.get(args.image_id_column, "")
    chosen_rows = []
    for parent in sorted(pairs_by_parent):
        candidates = pairs_by_parent[parent]
        selection_key = "{}:{}:{}".format(args.random_seed, image_id, parent[0])
        label_name, display_name = random.Random(selection_key).choice(candidates)
        output_row = {
            field: value
            for field, value in row.items()
            if field not in {args.labels_column, args.display_names_column}
        }
        output_row[args.output_label_column] = label_name
        output_row[args.output_display_name_column] = display_name
        output_row[args.parent_label_column] = parent[0]
        output_row[args.parent_display_name_column] = parent[1]
        chosen_rows.append(output_row)
    return chosen_rows


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


def write_part(rows, fields, args, label_to_parent, input_path, final_part, first_row_number):
    temporary_part = temporary_path(final_part)
    temporary_part.unlink(missing_ok=True)
    output_rows = 0
    dropped_rows = 0
    with temporary_part.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for offset, row in enumerate(rows):
            chosen_rows = choose_rows(row, args, label_to_parent, input_path, first_row_number + offset)
            if not chosen_rows:
                dropped_rows += 1
                continue
            writer.writerows(chosen_rows)
            output_rows += len(chosen_rows)
    temporary_part.replace(final_part)
    return output_rows, dropped_rows


def build_parts(input_path, fields, args, label_to_parent, parts_dir, progress):
    parts = []
    completed_parts = 0
    input_rows = 0
    newly_processed_rows = 0
    newly_output_rows = 0
    newly_dropped_rows = 0
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
            output_rows, dropped_rows = write_part(
                rows, fields, args, label_to_parent, input_path, final_part, first_row_number
            )
            newly_processed_rows += len(rows)
            newly_output_rows += output_rows
            newly_dropped_rows += dropped_rows
            progress.update(len(rows))
    return (
        parts,
        completed_parts,
        input_rows,
        newly_processed_rows,
        newly_output_rows,
        newly_dropped_rows,
    )


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
    input_fields = read_header(input_path)
    required_columns = [args.labels_column, args.display_names_column, args.image_id_column]
    missing_columns = [column for column in required_columns if column not in input_fields]
    if missing_columns:
        raise ValueError("Input CSV is missing required columns: {}".format(", ".join(missing_columns)))
    fields = output_fields(input_fields, args)
    validate_state(output_path, parts_dir, args.overwrite, args.resume)
    label_to_parent, multi_parent_assignments = load_label_to_parent(class_tree_path, args)
    parts_dir.mkdir(parents=True, exist_ok=True)
    total_rows = count_rows(input_path, args.progress_refresh_seconds)
    with tqdm(total=total_rows, desc="Splitting labels", unit="row", mininterval=args.progress_refresh_seconds) as progress:
        (
            parts,
            completed_parts,
            input_rows,
            newly_processed_rows,
            newly_output_rows,
            newly_dropped_rows,
        ) = build_parts(input_path, fields, args, label_to_parent, parts_dir, progress)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    combine_parts(parts, output_path, fields, args.progress_refresh_seconds)
    print("n2 labels mapped to top-level classes: {}".format(len(label_to_parent)))
    print(
        "Labels with multiple top-level classes: {} (policy: {})".format(
            len({label for label, _, _ in multi_parent_assignments}),
            args.multi_parent_label_policy,
        )
    )
    print("Input rows: {}".format(input_rows))
    print("Reused completed parts: {}".format(completed_parts))
    print("Newly processed rows: {}".format(newly_processed_rows))
    print("Newly output rows: {}".format(newly_output_rows))
    print("Newly dropped rows without labels: {}".format(newly_dropped_rows))
    print("Output CSV: {}".format(output_path.resolve()))
    if args.cleanup_parts:
        shutil.rmtree(parts_dir)
        print("Removed recovery directory: {}".format(parts_dir.resolve()))
    else:
        print("Recovery directory: {}".format(parts_dir.resolve()))


if __name__ == "__main__":
    main()
