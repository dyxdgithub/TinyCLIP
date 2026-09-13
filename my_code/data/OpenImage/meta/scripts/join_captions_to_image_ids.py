"""Left-join Localized Narratives captions onto the Open Images image-ID table.

The output preserves every Image IDs.csv row. Images with multiple captions
are expanded into one output row per caption and voice recording. Processing
uses resumable CSV parts for large input files.
"""

import argparse
import csv
import shutil
from pathlib import Path


META_DIR = Path(__file__).resolve().parent.parent
ALL_SET_DIR = META_DIR / "AllSet"
DEFAULT_CAPTIONS = ALL_SET_DIR / "Captions.csv"
DEFAULT_IMAGE_IDS = ALL_SET_DIR / "Image IDs.csv"
DEFAULT_OUTPUT = ALL_SET_DIR / "Image IDs_with_captions.csv"


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--captions",
        type=Path,
        default=DEFAULT_CAPTIONS,
        help="Caption CSV containing image_id, caption, and voice_recording. Default: %(default)s",
    )
    parser.add_argument(
        "--image-ids",
        type=Path,
        default=DEFAULT_IMAGE_IDS,
        help="Primary Image IDs CSV to preserve in the output. Default: %(default)s",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="Joined CSV path. Default: %(default)s",
    )
    parser.add_argument(
        "--caption-id-column",
        default="image_id",
        help="Image identifier column in --captions. Default: %(default)s",
    )
    parser.add_argument(
        "--image-id-column",
        default="ImageID",
        help="Image identifier column in --image-ids. Default: %(default)s",
    )
    parser.add_argument(
        "--caption-column",
        default="caption",
        help="Caption text column in --captions and appended output column. Default: %(default)s",
    )
    parser.add_argument(
        "--voice-column",
        default="voice_recording",
        help="Voice-recording column in --captions and appended output column. Default: %(default)s",
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
            "Directory for completed CSV parts. Default: a hidden directory beside "
            "--output."
        ),
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Reuse completed parts and process only unfinished output chunks.",
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


def load_captions(caption_path, image_id_column, caption_column, voice_column):
    require_columns(caption_path, (image_id_column, caption_column, voice_column))
    captions = {}
    caption_rows = 0
    with caption_path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        for line_number, row in enumerate(reader, start=2):
            image_id = (row.get(image_id_column) or "").strip()
            if not image_id:
                raise ValueError(
                    "Empty {} at {}:{}".format(image_id_column, caption_path, line_number)
                )
            caption_data = (row.get(caption_column) or "", row.get(voice_column) or "")
            captions.setdefault(image_id, []).append(caption_data)
            caption_rows += 1
    return captions, caption_rows


def part_path(parts_dir, part_number):
    return parts_dir / "part_{:06d}.csv".format(part_number)


def temporary_path(final_path):
    return final_path.with_name(final_path.name + ".inprogress")


def finalize_part(handle, in_progress_path, final_path):
    if handle is None:
        return
    handle.close()
    in_progress_path.replace(final_path)


def build_parts(
    image_ids_path,
    captions,
    image_id_column,
    caption_column,
    voice_column,
    output_fields,
    parts_dir,
    rows_per_part,
):
    part_paths = []
    completed_parts = 0
    total_rows = 0
    matched_rows = 0
    newly_written_rows = 0
    part_handle = None
    in_progress_path = None
    final_path = None
    writer = None
    completed_successfully = False

    try:
        with image_ids_path.open("r", encoding="utf-8-sig", newline="") as input_handle:
            reader = csv.DictReader(input_handle)
            for row_number, row in enumerate(reader, start=1):
                total_rows += 1
                part_number = (row_number - 1) // rows_per_part + 1
                expected_final_path = part_path(parts_dir, part_number)

                if final_path != expected_final_path:
                    finalize_part(part_handle, in_progress_path, final_path)
                    part_handle = None
                    writer = None
                    final_path = expected_final_path
                    in_progress_path = temporary_path(final_path)
                    part_paths.append(final_path)
                    if final_path.is_file():
                        completed_parts += 1
                    else:
                        in_progress_path.unlink(missing_ok=True)
                        part_handle = in_progress_path.open(
                            "w", encoding="utf-8-sig", newline=""
                        )
                        writer = csv.DictWriter(part_handle, fieldnames=output_fields)
                        writer.writeheader()

                image_id = (row.get(image_id_column) or "").strip()
                caption_records = captions.get(image_id, [])
                if caption_records:
                    matched_rows += 1
                if writer is None:
                    continue

                if not caption_records:
                    caption_records = [("", "")]
                for caption, voice_recording in caption_records:
                    row[caption_column] = caption
                    row[voice_column] = voice_recording
                    writer.writerow({field: row.get(field, "") for field in output_fields})
                    newly_written_rows += 1
        completed_successfully = True
    finally:
        if part_handle is not None:
            if completed_successfully:
                finalize_part(part_handle, in_progress_path, final_path)
            else:
                part_handle.close()

    return part_paths, completed_parts, total_rows, matched_rows, newly_written_rows


def combine_parts(part_paths, output_path, output_fields):
    in_progress_path = temporary_path(output_path)
    in_progress_path.unlink(missing_ok=True)
    with in_progress_path.open("w", encoding="utf-8-sig", newline="") as output_handle:
        writer = csv.DictWriter(output_handle, fieldnames=output_fields)
        writer.writeheader()
        for part_path_value in part_paths:
            with part_path_value.open("r", encoding="utf-8-sig", newline="") as part_handle:
                reader = csv.DictReader(part_handle)
                for row in reader:
                    writer.writerow(row)
    in_progress_path.replace(output_path)


def prepare_paths(args):
    caption_path = args.captions.expanduser()
    image_ids_path = args.image_ids.expanduser()
    output_path = args.output.expanduser()
    parts_dir = (
        args.parts_dir.expanduser()
        if args.parts_dir
        else output_path.parent / ".{}_parts".format(output_path.stem)
    )
    for path, label in ((caption_path, "Captions CSV"), (image_ids_path, "Image IDs CSV")):
        if not path.is_file():
            raise FileNotFoundError("{} does not exist: {}".format(label, path))
    return caption_path, image_ids_path, output_path, parts_dir


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
    caption_path, image_ids_path, output_path, parts_dir = prepare_paths(args)
    require_columns(image_ids_path, (args.image_id_column,))
    input_fields = read_csv_header(image_ids_path)
    conflicts = set(input_fields) & {args.caption_column, args.voice_column}
    if conflicts:
        raise ValueError(
            "Output columns already exist in {}: {}".format(
                image_ids_path, ", ".join(sorted(conflicts))
            )
        )
    validate_run_state(output_path, parts_dir, args.overwrite, args.resume)
    captions, caption_rows = load_captions(
        caption_path,
        args.caption_id_column,
        args.caption_column,
        args.voice_column,
    )
    output_fields = input_fields + [args.caption_column, args.voice_column]
    parts_dir.mkdir(parents=True, exist_ok=True)

    (
        part_paths,
        completed_parts,
        total_rows,
        matched_rows,
        newly_written_rows,
    ) = build_parts(
        image_ids_path,
        captions,
        args.image_id_column,
        args.caption_column,
        args.voice_column,
        output_fields,
        parts_dir,
        args.rows_per_part,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    combine_parts(part_paths, output_path, output_fields)

    print("Caption rows loaded: {}".format(caption_rows))
    print("Image IDs with one or more captions: {}".format(len(captions)))
    print("Primary-table rows: {}".format(total_rows))
    print("Rows with captions: {}".format(matched_rows))
    print("Reused completed parts: {}".format(completed_parts))
    print("Newly written rows: {}".format(newly_written_rows))
    print("Output CSV: {}".format(output_path.resolve()))
    if args.cleanup_parts:
        shutil.rmtree(parts_dir)
        print("Removed parts directory: {}".format(parts_dir.resolve()))
    else:
        print("Resume parts: {}".format(parts_dir.resolve()))


if __name__ == "__main__":
    main()
