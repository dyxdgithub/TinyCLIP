"""Keep only rows marked valid in a URL-validation CSV without rechecking URLs.

By default, rows with OriginalURL_Valid equal to "1" are retained. Supply
multiple --valid-columns values to require every selected URL validation label
to be valid. The input is processed in resumable CSV parts.
"""

import argparse
import csv
import shutil
from itertools import islice
from pathlib import Path

from tqdm import tqdm


META_DIR = Path(__file__).resolve().parent.parent
ALL_SET_DIR = META_DIR / "AllSet"
DEFAULT_INPUT = ALL_SET_DIR / "Image IDs_with_captions_and_labels_captioned_only_url_validation.csv"
DEFAULT_OUTPUT = ALL_SET_DIR / "Image IDs_with_captions_and_labels_captioned_only_url_valid_only.csv"


def parse_column_list(value):
    columns = []
    for column in value.split(","):
        column = column.strip()
        if column and column not in columns:
            columns.append(column)
    if not columns:
        raise argparse.ArgumentTypeError("At least one validation column is required")
    return columns


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_INPUT,
        help="CSV with URL-validation label columns. Default: %(default)s",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="CSV containing only rows valid for all selected URL columns. Default: %(default)s",
    )
    parser.add_argument(
        "--valid-columns",
        type=parse_column_list,
        default=["OriginalURL_Valid"],
        help=(
            "Comma-separated validity-label columns that must equal 1. Default: "
            "OriginalURL_Valid. Example: OriginalURL_Valid,Thumbnail300KURL_Valid"
        ),
    )
    parser.add_argument(
        "--rows-per-part",
        type=int,
        default=100000,
        help="Input rows in each resumable CSV part. Default: %(default)s",
    )
    parser.add_argument(
        "--progress-refresh-seconds",
        type=float,
        default=0.1,
        help="Minimum seconds between tqdm progress refreshes. Default: %(default)s",
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
        help="Reuse completed parts and process only unfinished input chunks.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Discard an existing output and parts directory before filtering.",
    )
    parser.add_argument(
        "--cleanup-parts",
        action="store_true",
        help="Delete completed parts after final output creation succeeds.",
    )
    args = parser.parse_args()
    if args.rows_per_part <= 0:
        parser.error("--rows-per-part must be positive")
    if args.progress_refresh_seconds <= 0:
        parser.error("--progress-refresh-seconds must be positive")
    return args


def read_header(input_path):
    with input_path.open("r", encoding="utf-8-sig", newline="") as handle:
        return csv.DictReader(handle).fieldnames or []


def count_rows(input_path, refresh_seconds):
    row_count = 0
    with input_path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.reader(handle)
        next(reader, None)
        with tqdm(
            desc="Counting input rows",
            unit="row",
            miniters=1,
            mininterval=refresh_seconds,
        ) as progress:
            for row_count, _ in enumerate(reader, start=1):
                progress.update(1)
    return row_count


def row_parts(reader, rows_per_part):
    while True:
        rows = list(islice(reader, rows_per_part))
        if not rows:
            return
        yield rows


def part_path(parts_dir, part_number):
    return parts_dir / "part_{:06d}.csv".format(part_number)


def temporary_path(final_path):
    return final_path.with_name(final_path.name + ".inprogress")


def write_part(rows, fields, valid_columns, final_part):
    temporary_part = temporary_path(final_part)
    temporary_part.unlink(missing_ok=True)
    kept_rows = 0
    with temporary_part.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            if all((row.get(column) or "").strip() == "1" for column in valid_columns):
                writer.writerow({field: row.get(field, "") for field in fields})
                kept_rows += 1
    temporary_part.replace(final_part)
    return kept_rows


def build_parts(input_path, fields, valid_columns, parts_dir, rows_per_part, progress):
    part_paths = []
    completed_parts = 0
    processed_rows = 0
    newly_processed_rows = 0
    newly_kept_rows = 0
    with input_path.open("r", encoding="utf-8-sig", newline="") as input_handle:
        reader = csv.DictReader(input_handle)
        for part_number, rows in enumerate(row_parts(reader, rows_per_part), start=1):
            final_part = part_path(parts_dir, part_number)
            part_paths.append(final_part)
            processed_rows += len(rows)
            if final_part.is_file():
                completed_parts += 1
                progress.update(len(rows))
                continue
            newly_kept_rows += write_part(rows, fields, valid_columns, final_part)
            newly_processed_rows += len(rows)
            progress.update(len(rows))
    return part_paths, completed_parts, processed_rows, newly_processed_rows, newly_kept_rows


def combine_parts(part_paths, output_path, fields):
    temporary_output = temporary_path(output_path)
    temporary_output.unlink(missing_ok=True)
    with temporary_output.open("w", encoding="utf-8-sig", newline="") as output_handle:
        writer = csv.DictWriter(output_handle, fieldnames=fields)
        writer.writeheader()
        for current_part in tqdm(part_paths, desc="Combining parts", unit="part"):
            with current_part.open("r", encoding="utf-8-sig", newline="") as part_handle:
                reader = csv.DictReader(part_handle)
                for row in reader:
                    writer.writerow(row)
    temporary_output.replace(output_path)


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
    input_path = args.input.expanduser()
    output_path = args.output.expanduser()
    parts_dir = (
        args.parts_dir.expanduser()
        if args.parts_dir
        else output_path.parent / ".{}_parts".format(output_path.stem)
    )
    if not input_path.is_file():
        raise FileNotFoundError("Input CSV does not exist: {}".format(input_path))

    fields = read_header(input_path)
    missing_columns = [column for column in args.valid_columns if column not in fields]
    if missing_columns:
        raise ValueError(
            "Input CSV is missing validity-label columns: {}".format(
                ", ".join(missing_columns)
            )
        )
    validate_run_state(output_path, parts_dir, args.overwrite, args.resume)
    parts_dir.mkdir(parents=True, exist_ok=True)
    total_rows = count_rows(input_path, args.progress_refresh_seconds)
    with tqdm(
        total=total_rows,
        desc="Filtering URL-valid rows",
        unit="row",
        miniters=1,
        mininterval=args.progress_refresh_seconds,
    ) as progress:
        (
            part_paths,
            completed_parts,
            processed_rows,
            newly_processed_rows,
            newly_kept_rows,
        ) = build_parts(
            input_path,
            fields,
            args.valid_columns,
            parts_dir,
            args.rows_per_part,
            progress,
        )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    combine_parts(part_paths, output_path, fields)

    print("Input rows: {}".format(processed_rows))
    print("Reused completed parts: {}".format(completed_parts))
    print("Newly processed rows: {}".format(newly_processed_rows))
    print("Newly retained URL-valid rows: {}".format(newly_kept_rows))
    print("Output CSV: {}".format(output_path.resolve()))
    if args.cleanup_parts:
        shutil.rmtree(parts_dir)
        print("Removed recovery directory: {}".format(parts_dir.resolve()))
    else:
        print("Recovery directory: {}".format(parts_dir.resolve()))


if __name__ == "__main__":
    main()
