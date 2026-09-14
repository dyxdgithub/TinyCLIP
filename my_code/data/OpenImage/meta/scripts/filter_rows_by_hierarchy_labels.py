"""Keep hierarchy-mapped labels in image rows selected by a class-map CSV.

The image table stores aligned label IDs and names as JSON arrays. A row is
retained when at least one label ID occurs in the hierarchy map, and both
arrays are rewritten to remove labels that do not occur in that map.
"""

import argparse
import csv
import json
import shutil
from itertools import islice
from pathlib import Path

from tqdm import tqdm


META_DIR = Path(__file__).resolve().parent.parent
ALL_SET_DIR = META_DIR / "AllSet"
DEFAULT_INPUT = ALL_SET_DIR / "Image IDs_with_captions_and_labels_captioned_only_url_valid_only.csv"
DEFAULT_CLASS_MAP = META_DIR / "Hierarchy" / "all" / "class_map_hierachy_all.csv"
DEFAULT_OUTPUT = ALL_SET_DIR / "Image IDs_with_hierarchy_labels.csv"


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT, help="Input image CSV. Default: %(default)s")
    parser.add_argument("--class-map", type=Path, default=DEFAULT_CLASS_MAP, help="Hierarchy class-map CSV. Default: %(default)s")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT, help="Filtered output CSV. Default: %(default)s")
    parser.add_argument("--labels-column", default="LabelNames", help="JSON label-ID array column in input. Default: %(default)s")
    parser.add_argument("--display-names-column", default="DisplayNames", help="JSON display-name array aligned with --labels-column. Default: %(default)s")
    parser.add_argument("--class-map-label-column", default="LabelName", help="Label ID column in class map. Default: %(default)s")
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


def load_hierarchy_labels(class_map_path, label_column):
    fields = set(read_header(class_map_path))
    if label_column not in fields:
        raise ValueError("Class map is missing label column '{}': {}".format(label_column, class_map_path))
    labels = set()
    with class_map_path.open("r", encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            label = (row.get(label_column) or "").strip()
            if label:
                labels.add(label)
    if not labels:
        raise ValueError("No hierarchy labels found in {}".format(class_map_path))
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
        raise ValueError(
            "Expected a JSON array in column '{}' at {} row {}".format(
                column, input_path, row_number
            )
        )
    return values


def filter_row_labels(row, labels_column, display_names_column, hierarchy_labels, input_path, row_number):
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
        if str(label_name).strip() in hierarchy_labels
    ]
    if not retained_pairs:
        return False

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
    return True


def part_path(parts_dir, number):
    return parts_dir / "part_{:06d}.csv".format(number)


def temporary_path(path):
    return path.with_name(path.name + ".inprogress")


def write_part(
    rows,
    fields,
    labels_column,
    display_names_column,
    hierarchy_labels,
    input_path,
    final_path,
    first_row_number,
):
    temporary_part = temporary_path(final_path)
    temporary_part.unlink(missing_ok=True)
    kept = 0
    with temporary_part.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for offset, row in enumerate(rows):
            if filter_row_labels(
                row,
                labels_column,
                display_names_column,
                hierarchy_labels,
                input_path,
                first_row_number + offset,
            ):
                writer.writerow({field: row.get(field, "") for field in fields})
                kept += 1
    temporary_part.replace(final_path)
    return kept


def build_parts(
    input_path,
    fields,
    labels_column,
    display_names_column,
    hierarchy_labels,
    parts_dir,
    rows_per_part,
    progress,
):
    parts = []
    completed_parts = 0
    input_rows = 0
    newly_processed = 0
    newly_kept = 0
    with input_path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        for part_number, rows in enumerate(row_batches(reader, rows_per_part), start=1):
            final_part = part_path(parts_dir, part_number)
            parts.append(final_part)
            first_row_number = input_rows + 2
            input_rows += len(rows)
            if final_part.is_file():
                completed_parts += 1
                progress.update(len(rows))
                continue
            newly_kept += write_part(
                rows,
                fields,
                labels_column,
                display_names_column,
                hierarchy_labels,
                input_path,
                final_part,
                first_row_number,
            )
            newly_processed += len(rows)
            progress.update(len(rows))
    return parts, completed_parts, input_rows, newly_processed, newly_kept


def row_batches(reader, size):
    while True:
        rows = list(islice(reader, size))
        if not rows:
            return
        yield rows


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


def main():
    args = parse_args()
    input_path = args.input.expanduser()
    class_map_path = args.class_map.expanduser()
    output_path = args.output.expanduser()
    parts_dir = args.parts_dir.expanduser() if args.parts_dir else output_path.parent / ".{}_parts".format(output_path.stem)
    if not input_path.is_file():
        raise FileNotFoundError("Input CSV does not exist: {}".format(input_path))
    if not class_map_path.is_file():
        raise FileNotFoundError("Class-map CSV does not exist: {}".format(class_map_path))
    fields = read_header(input_path)
    missing_columns = [
        column
        for column in (args.labels_column, args.display_names_column)
        if column not in fields
    ]
    if missing_columns:
        raise ValueError(
            "Input CSV is missing required columns: {}".format(", ".join(missing_columns))
        )
    validate_state(output_path, parts_dir, args.overwrite, args.resume)
    hierarchy_labels = load_hierarchy_labels(class_map_path, args.class_map_label_column)
    parts_dir.mkdir(parents=True, exist_ok=True)
    total_rows = 0
    with input_path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.reader(handle)
        next(reader, None)
        with tqdm(desc="Counting input rows", unit="row", mininterval=args.progress_refresh_seconds) as progress:
            for total_rows, _ in enumerate(reader, start=1):
                progress.update(1)
    with tqdm(total=total_rows, desc="Filtering hierarchy labels", unit="row", mininterval=args.progress_refresh_seconds) as progress:
        parts, completed_parts, processed_rows, newly_processed, newly_kept = build_parts(
            input_path,
            fields,
            args.labels_column,
            args.display_names_column,
            hierarchy_labels,
            parts_dir,
            args.rows_per_part,
            progress,
        )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    combine_parts(parts, output_path, fields, args.progress_refresh_seconds)
    print("Hierarchy labels: {}".format(len(hierarchy_labels)))
    print("Input rows: {}".format(processed_rows))
    print("Reused completed parts: {}".format(completed_parts))
    print("Newly processed rows: {}".format(newly_processed))
    print("Newly retained rows: {}".format(newly_kept))
    print("Output CSV: {}".format(output_path.resolve()))
    if args.cleanup_parts:
        shutil.rmtree(parts_dir)
        print("Removed recovery directory: {}".format(parts_dir.resolve()))
    else:
        print("Recovery directory: {}".format(parts_dir.resolve()))


if __name__ == "__main__":
    main()
