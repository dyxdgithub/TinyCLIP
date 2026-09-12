"""Extract selected metadata fields from Open Images Localized Narratives shards.

The annotation shards are JSON Lines files: each non-empty line is one JSON
object.  Completed shard outputs are kept in a parts directory so an
interrupted run can continue with --resume without processing finished shards.
"""

import argparse
import csv
import json
import shutil
from pathlib import Path


DEFAULT_INPUT_DIR = Path(
    r"E:\数据集\Open Image\train\Download dense annotations over 1.9M images\Localized narratives"
)
OUTPUT_FIELDS = ("dataset_id", "image_id", "caption", "voice_recording")
JSON_SUFFIXES = {".json", ".jsonl", ".ndjson"}


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=DEFAULT_INPUT_DIR,
        help=(
            "Directory containing Localized Narratives JSON Lines shards. "
            "Default: %(default)s"
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help=(
            "Combined CSV path. Default: localized_narratives_metadata.csv "
            "inside --input-dir."
        ),
    )
    parser.add_argument(
        "--parts-dir",
        type=Path,
        default=None,
        help=(
            "Directory for completed per-shard CSV parts used by --resume. "
            "Default: a hidden directory beside --output."
        ),
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Reuse completed shard parts and process only unfinished input files.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Discard an existing output and parts directory before processing.",
    )
    parser.add_argument(
        "--cleanup-parts",
        action="store_true",
        help="Delete the parts directory after the combined CSV is written successfully.",
    )
    return parser.parse_args()


def find_input_files(input_dir):
    files = sorted(
        path
        for path in input_dir.iterdir()
        if path.is_file() and path.suffix.lower() in JSON_SUFFIXES
    )
    if not files:
        accepted = ", ".join(sorted(JSON_SUFFIXES))
        raise FileNotFoundError(
            "No annotation files with suffixes {} found in {}".format(
                accepted, input_dir
            )
        )
    return files


def iter_annotations(input_path):
    with input_path.open("r", encoding="utf-8-sig") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                annotation = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(
                    "Invalid JSON at {}:{}: {}".format(input_path, line_number, error)
                ) from error
            if not isinstance(annotation, dict):
                raise ValueError(
                    "Expected a JSON object at {}:{}, got {}".format(
                        input_path, line_number, type(annotation).__name__
                    )
                )
            yield annotation


def selected_row(annotation):
    return {
        field: "" if annotation.get(field) is None else annotation.get(field, "")
        for field in OUTPUT_FIELDS
    }


def part_path(parts_dir, index, input_path):
    return parts_dir / "{:02d}_{}.csv".format(index, input_path.stem)


def write_part(input_path, output_path):
    temporary_path = output_path.with_name(output_path.name + ".inprogress")
    temporary_path.unlink(missing_ok=True)
    row_count = 0
    with temporary_path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=OUTPUT_FIELDS)
        writer.writeheader()
        for annotation in iter_annotations(input_path):
            writer.writerow(selected_row(annotation))
            row_count += 1
    temporary_path.replace(output_path)
    return row_count


def count_part_rows(part_path):
    with part_path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.reader(handle)
        next(reader, None)
        return sum(1 for _ in reader)


def combine_parts(part_paths, output_path):
    temporary_path = output_path.with_name(output_path.name + ".inprogress")
    temporary_path.unlink(missing_ok=True)
    with temporary_path.open("w", encoding="utf-8-sig", newline="") as output_handle:
        writer = csv.writer(output_handle)
        writer.writerow(OUTPUT_FIELDS)
        for part_path in part_paths:
            with part_path.open("r", encoding="utf-8-sig", newline="") as part_handle:
                next(part_handle, None)
                shutil.copyfileobj(part_handle, output_handle)
    temporary_path.replace(output_path)


def prepare_paths(args):
    input_dir = args.input_dir.expanduser()
    if not input_dir.is_dir():
        raise FileNotFoundError("Input directory does not exist: {}".format(input_dir))

    output_path = args.output.expanduser() if args.output else input_dir / "localized_narratives_metadata.csv"
    parts_dir = (
        args.parts_dir.expanduser()
        if args.parts_dir
        else output_path.parent / ".{}_parts".format(output_path.stem)
    )
    return input_dir, output_path, parts_dir


def validate_run_state(output_path, parts_dir, args):
    if args.overwrite:
        output_path.unlink(missing_ok=True)
        if parts_dir.exists():
            shutil.rmtree(parts_dir)
        return
    if output_path.exists():
        raise FileExistsError(
            "Output already exists: {}. Use --overwrite to replace it.".format(output_path)
        )
    if parts_dir.exists() and any(parts_dir.iterdir()) and not args.resume:
        raise FileExistsError(
            "Completed parts exist in {}. Use --resume to continue, or --overwrite "
            "to start again.".format(parts_dir)
        )


def main():
    args = parse_args()
    input_dir, output_path, parts_dir = prepare_paths(args)
    validate_run_state(output_path, parts_dir, args)
    input_files = find_input_files(input_dir)
    parts_dir.mkdir(parents=True, exist_ok=True)

    part_paths = []
    total_rows = 0
    for index, input_path in enumerate(input_files, start=1):
        output_part = part_path(parts_dir, index, input_path)
        part_paths.append(output_part)
        if output_part.is_file():
            row_count = count_part_rows(output_part)
            print("Reusing completed shard: {} ({} rows)".format(input_path.name, row_count))
        else:
            print("Extracting shard: {}".format(input_path.name))
            row_count = write_part(input_path, output_part)
            print("Completed shard: {} ({} rows)".format(input_path.name, row_count))
        total_rows += row_count

    output_path.parent.mkdir(parents=True, exist_ok=True)
    combine_parts(part_paths, output_path)
    print("Input shards: {}".format(len(input_files)))
    print("Output rows: {}".format(total_rows))
    print("Output CSV: {}".format(output_path.resolve()))

    if args.cleanup_parts:
        shutil.rmtree(parts_dir)
        print("Removed parts directory: {}".format(parts_dir.resolve()))
    else:
        print("Resume parts: {}".format(parts_dir.resolve()))


if __name__ == "__main__":
    main()
