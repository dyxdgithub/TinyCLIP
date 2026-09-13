"""Validate HTTP(S) URL columns and append validity labels to every image row.

Each configured URL column receives four appended result columns: Valid,
HTTPStatus, FinalURL, and Error. The large input is processed in resumable
parts, so --resume continues without rechecking completed chunks. By default,
both valid and invalid URL rows are retained for review.
"""

import argparse
import csv
import shutil
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from functools import partial
from itertools import islice
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.parse import urlparse
from urllib.request import Request, urlopen

from tqdm import tqdm


META_DIR = Path(__file__).resolve().parent.parent
ALL_SET_DIR = META_DIR / "AllSet"
DEFAULT_INPUT = ALL_SET_DIR / "Image IDs_with_captions_and_labels_captioned_only.csv"
DEFAULT_OUTPUT = ALL_SET_DIR / "Image IDs_with_captions_and_labels_captioned_only_url_validation.csv"
DEFAULT_USER_AGENT = "TinyCLIP-URL-Validator/1.0"
HEAD_FALLBACK_STATUSES = {403, 405, 501}


def parse_url_columns(value):
    columns = []
    for column in value.split(","):
        column = column.strip()
        if column and column not in columns:
            columns.append(column)
    if not columns:
        raise argparse.ArgumentTypeError("At least one URL column is required")
    return columns


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_INPUT,
        help="CSV containing rows and URL columns to validate. Default: %(default)s",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="CSV containing all rows with appended URL-validation columns. Default: %(default)s",
    )
    parser.add_argument(
        "--url-columns",
        type=parse_url_columns,
        default=["OriginalURL"],
        help=(
            "Comma-separated HTTP(S) URL columns to validate. Default: OriginalURL. "
            "Example: OriginalURL,Thumbnail300KURL"
        ),
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=15.0,
        help="Timeout in seconds for one HTTP request. Default: %(default)s",
    )
    parser.add_argument(
        "--retries",
        type=int,
        default=2,
        help="Retry count after the first failed network request. Default: %(default)s",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=32,
        help=(
            "Maximum URL validations performed concurrently. Increase this for "
            "more parallel checks, subject to network and remote-server limits. "
            "Default: %(default)s"
        ),
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=200,
        help=(
            "Rows queued per concurrent-validation batch. Set this no lower than "
            "--workers to use the requested concurrency. Default: %(default)s"
        ),
    )
    parser.add_argument(
        "--progress-refresh-seconds",
        "--progress-interval",
        dest="progress_refresh_seconds",
        type=float,
        default=0.1,
        help=(
            "Minimum seconds between real-time tqdm progress refreshes. "
            "Default: %(default)s"
        ),
    )
    parser.add_argument(
        "--rows-per-part",
        type=int,
        default=5000,
        help="Input rows in each resumable CSV part. Default: %(default)s",
    )
    parser.add_argument(
        "--user-agent",
        default=DEFAULT_USER_AGENT,
        help="HTTP User-Agent header sent with requests. Default: %(default)s",
    )
    parser.add_argument(
        "--parts-dir",
        type=Path,
        default=None,
        help=(
            "Directory for completed validation parts. Default: a hidden directory "
            "beside --output."
        ),
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Reuse completed parts and validate only unfinished input chunks.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Discard an existing output and parts directory before validating.",
    )
    parser.add_argument(
        "--cleanup-parts",
        action="store_true",
        help="Delete completed parts after final output creation succeeds.",
    )
    parser.add_argument(
        "--valid-only",
        action="store_true",
        help="Optional: write only rows valid for every selected URL column.",
    )
    args = parser.parse_args()
    if args.timeout <= 0:
        parser.error("--timeout must be positive")
    if args.retries < 0:
        parser.error("--retries must be zero or greater")
    if args.workers <= 0:
        parser.error("--workers must be positive")
    if args.batch_size <= 0:
        parser.error("--batch-size must be positive")
    if args.batch_size < args.workers:
        parser.error("--batch-size must be greater than or equal to --workers")
    if args.progress_refresh_seconds <= 0:
        parser.error("--progress-refresh-seconds must be positive")
    if args.rows_per_part <= 0:
        parser.error("--rows-per-part must be positive")
    return args


def read_header(input_path):
    with input_path.open("r", encoding="utf-8-sig", newline="") as handle:
        return csv.DictReader(handle).fieldnames or []


def count_input_rows(input_path, progress_refresh_seconds):
    row_count = 0
    with input_path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.reader(handle)
        next(reader, None)
        with tqdm(
            desc="Counting input rows",
            unit="row",
            miniters=1,
            mininterval=progress_refresh_seconds,
        ) as progress:
            for row_count, _ in enumerate(reader, start=1):
                progress.update(1)
    return row_count


def result_columns(url_column):
    return [
        "{}_Valid".format(url_column),
        "{}_HTTPStatus".format(url_column),
        "{}_FinalURL".format(url_column),
        "{}_Error".format(url_column),
    ]


def request_url(url, method, timeout, user_agent):
    headers = {"User-Agent": user_agent}
    if method == "GET":
        headers["Range"] = "bytes=0-0"
    request = Request(url, headers=headers, method=method)
    with urlopen(request, timeout=timeout) as response:
        return response.getcode(), response.geturl()


def validate_url(url, timeout, retries, user_agent):
    url = (url or "").strip()
    parsed = urlparse(url)
    if not url:
        return "0", "", "", "empty URL"
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        return "0", "", "", "not an HTTP(S) URL"

    last_error = ""
    for attempt in range(retries + 1):
        try:
            status, final_url = request_url(url, "HEAD", timeout, user_agent)
            return ("1" if 200 <= status < 400 else "0"), str(status), final_url, ""
        except HTTPError as error:
            if error.code not in HEAD_FALLBACK_STATUSES:
                return "0", str(error.code), error.geturl() or "", "HTTP {}".format(error.code)
            try:
                status, final_url = request_url(url, "GET", timeout, user_agent)
                return ("1" if 200 <= status < 400 else "0"), str(status), final_url, ""
            except HTTPError as get_error:
                return (
                    "0",
                    str(get_error.code),
                    get_error.geturl() or "",
                    "HTTP {}".format(get_error.code),
                )
            except (URLError, OSError, TimeoutError) as get_error:
                last_error = str(get_error.reason if isinstance(get_error, URLError) else get_error)
        except (URLError, OSError, TimeoutError) as error:
            last_error = str(error.reason if isinstance(error, URLError) else error)

        if attempt < retries:
            time.sleep(min(2**attempt, 4))
    return "0", "", "", last_error or "request failed"


def validate_row(row, url_columns, timeout, retries, user_agent):
    result = dict(row)
    for url_column in url_columns:
        valid, status, final_url, error = validate_url(
            row.get(url_column, ""), timeout, retries, user_agent
        )
        valid_column, status_column, final_url_column, error_column = result_columns(url_column)
        result[valid_column] = valid
        result[status_column] = status
        result[final_url_column] = final_url
        result[error_column] = error
    return result


def row_has_valid_urls(row, url_columns):
    return all(row["{}_Valid".format(column)] == "1" for column in url_columns)


def part_path(parts_dir, part_number):
    return parts_dir / "part_{:06d}.csv".format(part_number)


def in_progress_path(final_path):
    return final_path.with_name(final_path.name + ".inprogress")


def row_batches(reader, batch_size):
    while True:
        batch = list(islice(reader, batch_size))
        if not batch:
            return
        yield batch


def finalize_part(handle, temporary_part, final_part):
    if handle is None:
        return
    handle.close()
    temporary_part.replace(final_part)


def build_parts(input_path, output_fields, args, parts_dir, progress):
    part_paths = []
    completed_parts = 0
    input_rows = 0
    newly_checked_rows = 0
    newly_valid_rows = 0
    newly_filtered_rows = 0
    newly_written_rows = 0
    part_number = 0
    with input_path.open("r", encoding="utf-8-sig", newline="") as input_handle:
        reader = csv.DictReader(input_handle)
        validator = partial(
            validate_row,
            url_columns=args.url_columns,
            timeout=args.timeout,
            retries=args.retries,
            user_agent=args.user_agent,
        )
        with ThreadPoolExecutor(max_workers=args.workers) as executor:
            for part_rows in row_batches(reader, args.rows_per_part):
                part_number += 1
                final_part = part_path(parts_dir, part_number)
                part_paths.append(final_part)
                if final_part.is_file():
                    completed_parts += 1
                    input_rows += len(part_rows)
                    progress.update(len(part_rows))
                    continue

                temporary_part = in_progress_path(final_part)
                temporary_part.unlink(missing_ok=True)
                part_handle = temporary_part.open("w", encoding="utf-8-sig", newline="")
                try:
                    writer = csv.DictWriter(part_handle, fieldnames=output_fields)
                    writer.writeheader()
                    for batch in (
                        part_rows[index : index + args.batch_size]
                        for index in range(0, len(part_rows), args.batch_size)
                    ):
                        futures = {
                            executor.submit(validator, row): index
                            for index, row in enumerate(batch)
                        }
                        results = [None] * len(batch)
                        for future in as_completed(futures):
                            result_index = futures[future]
                            result = future.result()
                            results[result_index] = result
                            newly_checked_rows += 1
                            is_valid = row_has_valid_urls(result, args.url_columns)
                            if is_valid:
                                newly_valid_rows += 1
                            elif args.valid_only:
                                newly_filtered_rows += 1
                            progress.update(1)
                        for result in results:
                            is_valid = row_has_valid_urls(result, args.url_columns)
                            if not is_valid and args.valid_only:
                                continue
                            writer.writerow({field: result.get(field, "") for field in output_fields})
                            newly_written_rows += 1
                        input_rows += len(batch)
                except Exception:
                    part_handle.close()
                    raise
                else:
                    finalize_part(part_handle, temporary_part, final_part)
    return (
        part_paths,
        completed_parts,
        input_rows,
        newly_checked_rows,
        newly_valid_rows,
        newly_filtered_rows,
        newly_written_rows,
    )


def combine_parts(part_paths, output_path, output_fields):
    temporary_output = in_progress_path(output_path)
    temporary_output.unlink(missing_ok=True)
    total_parts = len(part_paths)
    with temporary_output.open("w", encoding="utf-8-sig", newline="") as output_handle:
        writer = csv.DictWriter(output_handle, fieldnames=output_fields)
        writer.writeheader()
        for current_part in tqdm(
            part_paths,
            desc="Combining parts",
            total=total_parts,
            unit="part",
        ):
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

    input_fields = read_header(input_path)
    missing_columns = [column for column in args.url_columns if column not in input_fields]
    if missing_columns:
        raise ValueError(
            "Input CSV is missing URL columns: {}".format(", ".join(missing_columns))
        )
    appended_fields = [field for column in args.url_columns for field in result_columns(column)]
    conflicts = set(input_fields) & set(appended_fields)
    if conflicts:
        raise ValueError(
            "Validation columns already exist in {}: {}".format(
                input_path, ", ".join(sorted(conflicts))
            )
        )
    validate_run_state(output_path, parts_dir, args.overwrite, args.resume)
    parts_dir.mkdir(parents=True, exist_ok=True)
    total_input_rows = count_input_rows(input_path, args.progress_refresh_seconds)
    output_fields = input_fields + appended_fields
    with tqdm(
        total=total_input_rows,
        desc="Validating URLs",
        unit="row",
        miniters=1,
        mininterval=args.progress_refresh_seconds,
    ) as validation_progress:
        (
            part_paths,
            completed_parts,
            input_rows,
            newly_checked_rows,
            newly_valid_rows,
            newly_filtered_rows,
            newly_written_rows,
        ) = build_parts(
            input_path, output_fields, args, parts_dir, validation_progress
        )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    combine_parts(part_paths, output_path, output_fields)

    print("Input rows: {}".format(input_rows))
    print("Reused completed parts: {}".format(completed_parts))
    print("Newly checked rows: {}".format(newly_checked_rows))
    print("Newly valid rows: {}".format(newly_valid_rows))
    print("Newly filtered invalid rows: {}".format(newly_filtered_rows))
    print("Newly written rows: {}".format(newly_written_rows))
    print("Output CSV: {}".format(output_path.resolve()))
    if args.cleanup_parts:
        shutil.rmtree(parts_dir)
        print("Removed recovery directory: {}".format(parts_dir.resolve()))
    else:
        print("Recovery directory: {}".format(parts_dir.resolve()))


if __name__ == "__main__":
    main()
