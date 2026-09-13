"""Append Open Images display names to image-label rows by LabelName.

The source label CSV is processed in resumable chunks. Completed chunks remain
in a parts directory until the final CSV has been assembled successfully.
"""

import argparse
import csv
import shutil
from pathlib import Path


META_DIR = Path(__file__).resolve().parent.parent
ALL_SET_DIR = META_DIR / "AllSet"
DEFAULT_CLASS_NAMES = ALL_SET_DIR / "Class names(all sets).csv"
DEFAULT_IMAGE_LABELS = ALL_SET_DIR / "Image labels_confidence_1.csv"
DEFAULT_OUTPUT = ALL_SET_DIR / "Image labels_confidence_1_with_display_names.csv"


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--class-names",
        type=Path,
        default=DEFAULT_CLASS_NAMES,
        help="CSV containing label identifiers and display names. Default: %(default)s",
    )
    parser.add_argument(
        "--image-labels",
        type=Path,
        default=DEFAULT_IMAGE_LABELS,
        help="CSV containing image-label rows to enrich. Default: %(default)s",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="Combined enriched CSV path. Default: %(default)s",
    )
    parser.add_argument(
        "--label-column",
        default="LabelName",
        help="Shared label identifier column in both CSV files. Default: %(default)s",
    )
    parser.add_argument(
        "--display-name-column",
        default="DisplayName",
        help="Display-name column in --class-names and output column name. Default: %(default)s",
    )
    parser.add_argument(
        "--rows-per-part",
        type=int,
        default=100000,
        help="Number of input rows in each resumable CSV part. Default: %(default)s",
    )
    parser.add_argument(
        "--parts-dir",
        type=Path,
        default=None,
        help=(
            "Directory for completed CSV parts. Default: a hidden directory beside "
            "--output."
        ),
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Reuse completed parts and process only unfinished chunks.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Discard an existing output and parts directory before processing.",
    )
    parser.add_argument(
        "--cleanup-parts",
        action="store_true",
        help="Delete completed parts after the final output is created successfully.",
    )
    args = parser.parse_args()
    if args.rows_per_part <= 0:
        parser.error("--rows-per-part must be a positive integer")
    return args


def read_csv_header(csv_path):
    with csv_path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        return reader.fieldnames or []


def require_columns(csv_path, columns):
    fields = set(read_csv_header(csv_path))
    missing = [column for column in columns if column not in fields]
    if missing:
        raise ValueError(
            "CSV {} is missing required columns: {}".format(csv_path, ", ".join(missing))
        )


def load_display_names(class_names_path, label_column, display_name_column):
    require_columns(class_names_path, (label_column, display_name_column))
    display_names = {}
    with class_names_path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        for line_number, row in enumerate(reader, start=2):
            label_name = (row.get(label_column) or "").strip()
            if not label_name:
                raise ValueError(
                    "Empty {} at {}:{}".format(label_column, class_names_path, line_number)
                )
            display_name = row.get(display_name_column) or ""
            previous = display_names.get(label_name)
            if previous is not None and previous != display_name:
                raise ValueError(
                    "Conflicting display names for {} in {}".format(label_name, class_names_path)
                )
            display_names[label_name] = display_name
    return display_names


def part_path(parts_dir, part_number):
    return parts_dir / "part_{:06d}.csv".format(part_number)


def part_temporary_path(final_path):
    return final_path.with_name(final_path.name + ".inprogress")


def finalize_part(handle, temporary_path, final_path):
    if handle is None:
        return
    handle.close()
    temporary_path.replace(final_path)


def build_parts(
    image_labels_path,
    display_names,
    label_column,
    display_name_column,
    output_fields,
    parts_dir,
    rows_per_part,
):
    part_paths = []
    completed_parts = 0
    written_rows = 0
    part_handle = None
    temporary_path = None
    final_path = None
    writer = None
    completed_successfully = False

    try:
        with image_labels_path.open("r", encoding="utf-8-sig", newline="") as input_handle:
            reader = csv.DictReader(input_handle)
            for row_number, row in enumerate(reader, start=1):
                part_number = (row_number - 1) // rows_per_part + 1
                expected_final_path = part_path(parts_dir, part_number)

                if final_path != expected_final_path:
                    finalize_part(part_handle, temporary_path, final_path)
                    part_handle = None
                    writer = None
                    final_path = expected_final_path
                    temporary_path = part_temporary_path(final_path)
                    part_paths.append(final_path)

                    if final_path.is_file():
                        completed_parts += 1
                    else:
                        temporary_path.unlink(missing_ok=True)
                        part_handle = temporary_path.open(
                            "w", encoding="utf-8-sig", newline=""
                        )
                        writer = csv.DictWriter(part_handle, fieldnames=output_fields)
                        writer.writeheader()

                if writer is None:
                    continue

                label_name = (row.get(label_column) or "").strip()
                if label_name not in display_names:
                    raise ValueError(
                        "No display name found for {} '{}' at {} row {}".format(
                            label_column, label_name, image_labels_path, row_number + 1
                        )
                    )
                row[display_name_column] = display_names[label_name]
                writer.writerow({field: row.get(field, "") for field in output_fields})
                written_rows += 1
        completed_successfully = True
    finally:
        if part_handle is not None:
            if completed_successfully:
                finalize_part(part_handle, temporary_path, final_path)
            else:
                part_handle.close()

    return part_paths, completed_parts, written_rows


def combine_parts(part_paths, output_path, output_fields):
    temporary_path = part_temporary_path(output_path)
    temporary_path.unlink(missing_ok=True)
    with temporary_path.open("w", encoding="utf-8-sig", newline="") as output_handle:
        writer = csv.DictWriter(output_handle, fieldnames=output_fields)
        writer.writeheader()
        for path in part_paths:
            with path.open("r", encoding="utf-8-sig", newline="") as part_handle:
                reader = csv.DictReader(part_handle)
                for row in reader:
                    writer.writerow(row)
    temporary_path.replace(output_path)


def prepare_paths(args):
    class_names_path = args.class_names.expanduser()
    image_labels_path = args.image_labels.expanduser()
    output_path = args.output.expanduser()
    parts_dir = (
        args.parts_dir.expanduser()
        if args.parts_dir
        else output_path.parent / ".{}_parts".format(output_path.stem)
    )
    for path, label in (
        (class_names_path, "Class-names CSV"),
        (image_labels_path, "Image-label CSV"),
    ):
        if not path.is_file():
            raise FileNotFoundError("{} does not exist: {}".format(label, path))
    return class_names_path, image_labels_path, output_path, parts_dir


def validate_run_state(output_path, parts_dir, overwrite, resume):
    if overwrite:
        output_path.unlink(missing_ok=True)
        if parts_dir.exists():
            shutil.rmtree(parts_dir)
        return
    if output_path.exists():
        raise FileExistsError(
            "Output already exists: {}. Use --overwrite to replace it.".format(output_path)
        )
    if parts_dir.exists() and any(parts_dir.iterdir()) and not resume:
        raise FileExistsError(
            "Completed parts exist in {}. Use --resume to continue, or --overwrite "
            "to start again.".format(parts_dir)
        )


def main():
    args = parse_args()
    class_names_path, image_labels_path, output_path, parts_dir = prepare_paths(args)
    require_columns(image_labels_path, (args.label_column,))
    input_fields = read_csv_header(image_labels_path)
    if args.display_name_column in input_fields:
        raise ValueError(
            "Output column '{}' already exists in {}. Choose a different "
            "--display-name-column.".format(args.display_name_column, image_labels_path)
        )
    validate_run_state(output_path, parts_dir, args.overwrite, args.resume)
    display_names = load_display_names(
        class_names_path, args.label_column, args.display_name_column
    )
    output_fields = input_fields + [args.display_name_column]
    parts_dir.mkdir(parents=True, exist_ok=True)

    part_paths, completed_parts, written_rows = build_parts(
        image_labels_path,
        display_names,
        args.label_column,
        args.display_name_column,
        output_fields,
        parts_dir,
        args.rows_per_part,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    combine_parts(part_paths, output_path, output_fields)

    print("Loaded display-name mappings: {}".format(len(display_names)))
    print("Reused completed parts: {}".format(completed_parts))
    print("Newly written rows: {}".format(written_rows))
    print("Output CSV: {}".format(output_path.resolve()))
    if args.cleanup_parts:
        shutil.rmtree(parts_dir)
        print("Removed parts directory: {}".format(parts_dir.resolve()))
    else:
        print("Resume parts: {}".format(parts_dir.resolve()))


if __name__ == "__main__":
    main()
