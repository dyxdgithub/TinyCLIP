"""Append aggregated Open Images labels to rows in Image IDs_with_captions.csv.

Each output row preserves one caption record. Labels for the matching ImageID
are stored in LabelNames and DisplayNames as aligned JSON arrays, avoiding a
many-to-many row expansion when images have multiple captions and labels.
"""

import argparse
import csv
import json
import shutil
import sqlite3
from pathlib import Path


META_DIR = Path(__file__).resolve().parent.parent
ALL_SET_DIR = META_DIR / "AllSet"
DEFAULT_LABELS = ALL_SET_DIR / "Image labels_confidence_1_with_display_names.csv"
DEFAULT_IMAGE_CAPTIONS = ALL_SET_DIR / "Image IDs_with_captions.csv"
DEFAULT_OUTPUT = ALL_SET_DIR / "Image IDs_with_captions_and_labels.csv"
LABEL_NAMES_COLUMN = "LabelNames"
DISPLAY_NAMES_COLUMN = "DisplayNames"


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--labels",
        type=Path,
        default=DEFAULT_LABELS,
        help="CSV containing ImageID, LabelName, and DisplayName. Default: %(default)s",
    )
    parser.add_argument(
        "--image-captions",
        type=Path,
        default=DEFAULT_IMAGE_CAPTIONS,
        help="Primary CSV containing image metadata and captions. Default: %(default)s",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="Joined CSV path. Default: %(default)s",
    )
    parser.add_argument(
        "--image-id-column",
        default="ImageID",
        help="Shared image identifier column. Default: %(default)s",
    )
    parser.add_argument(
        "--label-name-column",
        default="LabelName",
        help="Label identifier column in --labels. Default: %(default)s",
    )
    parser.add_argument(
        "--display-name-column",
        default="DisplayName",
        help="Human-readable label column in --labels. Default: %(default)s",
    )
    parser.add_argument(
        "--rows-per-part",
        type=int,
        default=100000,
        help="Number of primary-table rows in each resumable CSV part. Default: %(default)s",
    )
    parser.add_argument(
        "--parts-dir",
        type=Path,
        default=None,
        help=(
            "Directory for the persistent label index and completed CSV parts. "
            "Default: a hidden directory beside --output."
        ),
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Reuse the completed label index and CSV parts after an interruption.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Discard an existing output and parts directory before processing.",
    )
    parser.add_argument(
        "--cleanup-parts",
        action="store_true",
        help="Delete the label index and completed parts after output creation succeeds.",
    )
    args = parser.parse_args()
    if args.rows_per_part <= 0:
        parser.error("--rows-per-part must be a positive integer")
    return args


def read_csv_header(csv_path):
    with csv_path.open("r", encoding="utf-8-sig", newline="") as handle:
        return csv.DictReader(handle).fieldnames or []


def require_columns(csv_path, columns):
    fields = set(read_csv_header(csv_path))
    missing = [column for column in columns if column not in fields]
    if missing:
        raise ValueError(
            "CSV {} is missing required columns: {}".format(csv_path, ", ".join(missing))
        )


def index_path(parts_dir):
    return parts_dir / "labels.sqlite3"


def in_progress_path(final_path):
    return final_path.with_name(final_path.name + ".inprogress")


def build_label_index(labels_path, database_path, image_id_column, label_name_column, display_name_column):
    temporary_database = in_progress_path(database_path)
    temporary_database.unlink(missing_ok=True)
    label_rows = 0
    unique_labels = 0
    connection = sqlite3.connect(temporary_database)
    try:
        connection.execute(
            "CREATE TABLE labels ("
            "image_id TEXT NOT NULL, "
            "label_name TEXT NOT NULL, "
            "display_name TEXT NOT NULL, "
            "PRIMARY KEY (image_id, label_name, display_name)"
            ") WITHOUT ROWID"
        )
        with labels_path.open("r", encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle)
            with connection:
                for line_number, row in enumerate(reader, start=2):
                    image_id = (row.get(image_id_column) or "").strip()
                    label_name = row.get(label_name_column) or ""
                    display_name = row.get(display_name_column) or ""
                    if not image_id or not label_name:
                        raise ValueError(
                            "Empty image ID or label name at {}:{}".format(
                                labels_path, line_number
                            )
                        )
                    result = connection.execute(
                        "INSERT OR IGNORE INTO labels (image_id, label_name, display_name) "
                        "VALUES (?, ?, ?)",
                        (image_id, label_name, display_name),
                    )
                    label_rows += 1
                    unique_labels += result.rowcount
    finally:
        connection.close()
    temporary_database.replace(database_path)
    return label_rows, unique_labels


def get_label_index_stats(database_path):
    with sqlite3.connect(database_path) as connection:
        label_rows = connection.execute("SELECT COUNT(*) FROM labels").fetchone()[0]
        image_count = connection.execute("SELECT COUNT(DISTINCT image_id) FROM labels").fetchone()[0]
    return label_rows, image_count


def ensure_label_index(labels_path, parts_dir, image_id_column, label_name_column, display_name_column):
    database_path = index_path(parts_dir)
    if database_path.is_file():
        label_rows, image_count = get_label_index_stats(database_path)
        print("Reusing label index: {} unique labels across {} images".format(label_rows, image_count))
        return database_path

    print("Building label index from {}".format(labels_path.name))
    input_rows, unique_labels = build_label_index(
        labels_path,
        database_path,
        image_id_column,
        label_name_column,
        display_name_column,
    )
    print("Indexed {} label rows ({} unique image-label pairs)".format(input_rows, unique_labels))
    return database_path


def part_path(parts_dir, part_number):
    return parts_dir / "part_{:06d}.csv".format(part_number)


def finalize_part(handle, temporary_part, final_part):
    if handle is None:
        return
    handle.close()
    temporary_part.replace(final_part)


def build_parts(image_captions_path, database_path, image_id_column, output_fields, parts_dir, rows_per_part):
    part_paths = []
    completed_parts = 0
    primary_rows = 0
    rows_with_labels = 0
    newly_written_rows = 0
    part_handle = None
    temporary_part = None
    final_part = None
    writer = None
    completed_successfully = False
    cached_image_id = None
    cached_label_names = "[]"
    cached_display_names = "[]"

    connection = sqlite3.connect(database_path)
    try:
        cursor = connection.cursor()
        with image_captions_path.open("r", encoding="utf-8-sig", newline="") as input_handle:
            reader = csv.DictReader(input_handle)
            for row_number, row in enumerate(reader, start=1):
                primary_rows += 1
                part_number = (row_number - 1) // rows_per_part + 1
                expected_final_part = part_path(parts_dir, part_number)
                if final_part != expected_final_part:
                    finalize_part(part_handle, temporary_part, final_part)
                    part_handle = None
                    writer = None
                    final_part = expected_final_part
                    temporary_part = in_progress_path(final_part)
                    part_paths.append(final_part)
                    if final_part.is_file():
                        completed_parts += 1
                    else:
                        temporary_part.unlink(missing_ok=True)
                        part_handle = temporary_part.open("w", encoding="utf-8-sig", newline="")
                        writer = csv.DictWriter(part_handle, fieldnames=output_fields)
                        writer.writeheader()

                image_id = (row.get(image_id_column) or "").strip()
                if image_id != cached_image_id:
                    records = cursor.execute(
                        "SELECT label_name, display_name FROM labels "
                        "WHERE image_id = ? ORDER BY label_name, display_name",
                        (image_id,),
                    ).fetchall()
                    cached_image_id = image_id
                    cached_label_names = json.dumps(
                        [label_name for label_name, _ in records],
                        ensure_ascii=False,
                        separators=(",", ":"),
                    )
                    cached_display_names = json.dumps(
                        [display_name for _, display_name in records],
                        ensure_ascii=False,
                        separators=(",", ":"),
                    )
                if cached_label_names != "[]":
                    rows_with_labels += 1
                if writer is None:
                    continue

                row[LABEL_NAMES_COLUMN] = cached_label_names
                row[DISPLAY_NAMES_COLUMN] = cached_display_names
                writer.writerow({field: row.get(field, "") for field in output_fields})
                newly_written_rows += 1
        completed_successfully = True
    finally:
        connection.close()
        if part_handle is not None:
            if completed_successfully:
                finalize_part(part_handle, temporary_part, final_part)
            else:
                part_handle.close()

    return part_paths, completed_parts, primary_rows, rows_with_labels, newly_written_rows


def combine_parts(part_paths, output_path, output_fields):
    temporary_output = in_progress_path(output_path)
    temporary_output.unlink(missing_ok=True)
    with temporary_output.open("w", encoding="utf-8-sig", newline="") as output_handle:
        writer = csv.DictWriter(output_handle, fieldnames=output_fields)
        writer.writeheader()
        for current_part_path in part_paths:
            with current_part_path.open("r", encoding="utf-8-sig", newline="") as part_handle:
                reader = csv.DictReader(part_handle)
                for row in reader:
                    writer.writerow(row)
    temporary_output.replace(output_path)


def prepare_paths(args):
    labels_path = args.labels.expanduser()
    image_captions_path = args.image_captions.expanduser()
    output_path = args.output.expanduser()
    parts_dir = (
        args.parts_dir.expanduser()
        if args.parts_dir
        else output_path.parent / ".{}_parts".format(output_path.stem)
    )
    for path, label in ((labels_path, "Labels CSV"), (image_captions_path, "Image-captions CSV")):
        if not path.is_file():
            raise FileNotFoundError("{} does not exist: {}".format(label, path))
    return labels_path, image_captions_path, output_path, parts_dir


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
            "Recovery data exists in {}. Use --resume to continue, or --overwrite "
            "to start again.".format(parts_dir)
        )


def main():
    args = parse_args()
    labels_path, image_captions_path, output_path, parts_dir = prepare_paths(args)
    require_columns(
        labels_path,
        (args.image_id_column, args.label_name_column, args.display_name_column),
    )
    require_columns(image_captions_path, (args.image_id_column,))
    input_fields = read_csv_header(image_captions_path)
    conflicts = set(input_fields) & {LABEL_NAMES_COLUMN, DISPLAY_NAMES_COLUMN}
    if conflicts:
        raise ValueError(
            "Output columns already exist in {}: {}".format(
                image_captions_path, ", ".join(sorted(conflicts))
            )
        )
    validate_run_state(output_path, parts_dir, args.overwrite, args.resume)
    parts_dir.mkdir(parents=True, exist_ok=True)
    database_path = ensure_label_index(
        labels_path,
        parts_dir,
        args.image_id_column,
        args.label_name_column,
        args.display_name_column,
    )
    output_fields = input_fields + [LABEL_NAMES_COLUMN, DISPLAY_NAMES_COLUMN]
    (
        part_paths,
        completed_parts,
        primary_rows,
        rows_with_labels,
        newly_written_rows,
    ) = build_parts(
        image_captions_path,
        database_path,
        args.image_id_column,
        output_fields,
        parts_dir,
        args.rows_per_part,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    combine_parts(part_paths, output_path, output_fields)

    print("Primary rows: {}".format(primary_rows))
    print("Rows with one or more labels: {}".format(rows_with_labels))
    print("Reused completed parts: {}".format(completed_parts))
    print("Newly written rows: {}".format(newly_written_rows))
    print("Output CSV: {}".format(output_path.resolve()))
    if args.cleanup_parts:
        shutil.rmtree(parts_dir)
        print("Removed recovery directory: {}".format(parts_dir.resolve()))
    else:
        print("Recovery directory: {}".format(parts_dir.resolve()))


if __name__ == "__main__":
    main()
