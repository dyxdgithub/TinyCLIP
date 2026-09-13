"""Keep rows with a non-empty caption from Image IDs_with_captions_and_labels.csv.

The input is processed in resumable CSV parts so a large final-data run can
continue after interruption without recreating completed output chunks.
"""

import argparse
import csv
import shutil
from pathlib import Path


META_DIR = Path(__file__).resolve().parent.parent
ALL_SET_DIR = META_DIR / "AllSet"
DEFAULT_INPUT = ALL_SET_DIR / "Image IDs_with_captions_and_labels.csv"
DEFAULT_OUTPUT = ALL_SET_DIR / "Image IDs_with_captions_and_labels_captioned_only.csv"


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_INPUT,
        help="CSV to filter. Default: %(default)s",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="CSV containing only rows with a non-empty caption. Default: %(default)s",
    )
    parser.add_argument(
        "--caption-column",
        default="caption",
        help="Caption column used to decide whether a row is retained. Default: %(default)s",
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
        help="Reuse completed parts and process only unfinished input chunks.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Discard an existing output and parts directory before processing.",
    )
    parser.add_argument(
        "--cleanup-parts",
        action="store_true",
        help="Delete completed parts after the final output is written successfully.",
    )
    args = parser.parse_args()
    if args.rows_per_part <= 0:
        parser.error("--rows-per-part must be a positive integer")
    return args


def read_header(input_path):
    with input_path.open("r", encoding="utf-8-sig", newline="") as handle:
        return csv.DictReader(handle).fieldnames or []


def part_path(parts_dir, part_number):
    return parts_dir / "part_{:06d}.csv".format(part_number)


def in_progress_path(final_path):
    return final_path.with_name(final_path.name + ".inprogress")


def finalize_part(handle, temporary_part, final_part):
    if handle is None:
        return
    handle.close()
    temporary_part.replace(final_part)


def build_parts(input_path, output_fields, caption_column, parts_dir, rows_per_part):
    part_paths = []
    completed_parts = 0
    input_rows = 0
    captioned_rows = 0
    newly_written_rows = 0
    part_handle = None
    temporary_part = None
    final_part = None
    writer = None
    completed_successfully = False

    try:
        with input_path.open("r", encoding="utf-8-sig", newline="") as input_handle:
            reader = csv.DictReader(input_handle)
            for row_number, row in enumerate(reader, start=1):
                input_rows += 1
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

                if not (row.get(caption_column) or "").strip():
                    continue
                captioned_rows += 1
                if writer is None:
                    continue
                writer.writerow({field: row.get(field, "") for field in output_fields})
                newly_written_rows += 1
        completed_successfully = True
    finally:
        if part_handle is not None:
            if completed_successfully:
                finalize_part(part_handle, temporary_part, final_part)
            else:
                part_handle.close()

    return part_paths, completed_parts, input_rows, captioned_rows, newly_written_rows


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

    output_fields = read_header(input_path)
    if args.caption_column not in output_fields:
        raise ValueError(
            "Input CSV is missing caption column '{}': {}".format(
                args.caption_column, input_path
            )
        )
    validate_run_state(output_path, parts_dir, args.overwrite, args.resume)
    parts_dir.mkdir(parents=True, exist_ok=True)
    (
        part_paths,
        completed_parts,
        input_rows,
        captioned_rows,
        newly_written_rows,
    ) = build_parts(
        input_path,
        output_fields,
        args.caption_column,
        parts_dir,
        args.rows_per_part,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    combine_parts(part_paths, output_path, output_fields)

    print("Input rows: {}".format(input_rows))
    print("Rows with non-empty captions: {}".format(captioned_rows))
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
