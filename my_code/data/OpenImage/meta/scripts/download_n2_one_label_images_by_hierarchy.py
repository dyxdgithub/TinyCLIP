"""Download one-label n2 image rows into folders defined by the hierarchy JSON.

Each input row is downloaded to the path of its ParentLabelName and LabelName
in the n2 hierarchy. The final status CSV keeps every input row and appends
download fields. A SQLite checkpoint records each completed row immediately,
so --resume avoids repeating successful downloads after an interruption.
"""

import argparse
import csv
import json
import shutil
import sqlite3
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from itertools import islice
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.parse import urlparse
from urllib.request import Request, urlopen

from tqdm import tqdm


META_DIR = Path(__file__).resolve().parent.parent
N2_DIR = META_DIR / "Hierarchy" / "n2"
DEFAULT_INPUT = N2_DIR / "Image IDs_with_last_two_layers_multi_members_n2_one_label_per_parent.csv"
DEFAULT_CLASS_TREE = N2_DIR / "Last_Two_Layers_Multi_Members_n2.json"
DEFAULT_OUTPUT_DIR = N2_DIR / "images"
DEFAULT_STATUS_OUTPUT = N2_DIR / "Image IDs_with_last_two_layers_multi_members_n2_download_status.csv"
IMAGE_SUFFIXES = (".jpg", ".jpeg", ".png", ".webp", ".bmp")
INVALID_PATH_CHARS = '<>:"/\\|?*'
STATUS_FIELDS = [
    "InputRowNumber",
    "DownloadStatus",
    "DownloadHTTPStatus",
    "DownloadFinalURL",
    "DownloadError",
    "DownloadedImagePath",
]


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT, help="One-label-per-row source CSV. Default: %(default)s")
    parser.add_argument("--class-tree", type=Path, default=DEFAULT_CLASS_TREE, help="n2 class-tree JSON. Default: %(default)s")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR, help="Root directory for hierarchy-organized images. Default: %(default)s")
    parser.add_argument("--status-output", type=Path, default=DEFAULT_STATUS_OUTPUT, help="Final CSV with one status record per input row. Default: %(default)s")
    parser.add_argument("--checkpoint-db", type=Path, default=None, help="SQLite checkpoint database for resume. Default: hidden database beside --status-output.")
    parser.add_argument("--url-columns", default="OriginalURL_FinalURL,OriginalURL", help="Comma-separated URL columns tried in order. Default: %(default)s")
    parser.add_argument("--image-id-column", default="ImageID", help="Image-ID column used for file names. Default: %(default)s")
    parser.add_argument("--label-column", default="LabelName", help="Single leaf-label ID column. Default: %(default)s")
    parser.add_argument("--parent-label-column", default="ParentLabelName", help="Top-level parent-label ID column. Default: %(default)s")
    parser.add_argument("--label-key", default="LabelName", help="Class-label key in the JSON tree. Default: %(default)s")
    parser.add_argument("--text-key", default="TextName", help="Class display-name key in the JSON tree. Default: %(default)s")
    parser.add_argument("--children-key", default="Subcategory", help="Child-node key in the JSON tree. Default: %(default)s")
    parser.add_argument("--workers", type=int, default=32, help="Maximum concurrent download workers. Default: %(default)s")
    parser.add_argument("--timeout-seconds", type=float, default=30.0, help="HTTP request timeout in seconds. Default: %(default)s")
    parser.add_argument("--retries", type=int, default=2, help="Additional attempts after a failed download. Default: %(default)s")
    parser.add_argument("--retry-delay-seconds", type=float, default=1.0, help="Delay before each retry in seconds. Default: %(default)s")
    parser.add_argument("--rows-per-batch", type=int, default=1000, help="Input rows queued at one time. Default: %(default)s")
    parser.add_argument("--progress-refresh-seconds", type=float, default=0.1, help="Minimum seconds between tqdm refreshes. Default: %(default)s")
    parser.add_argument("--resume", action="store_true", help="Reuse successful/skipped checkpoint records after interruption.")
    parser.add_argument("--retry-errors", action="store_true", help="With --resume, retry records whose prior status is error.")
    parser.add_argument("--overwrite", action="store_true", help="Replace existing status output and checkpoint database. Existing images are retained and reported as skipped.")
    parser.add_argument("--cleanup-checkpoint", action="store_true", help="Delete the checkpoint database after successfully writing --status-output.")
    args = parser.parse_args()
    args.url_columns = [column.strip() for column in args.url_columns.split(",") if column.strip()]
    if not args.url_columns:
        parser.error("--url-columns must include at least one column name")
    if args.workers <= 0:
        parser.error("--workers must be positive")
    if args.timeout_seconds <= 0:
        parser.error("--timeout-seconds must be positive")
    if args.retries < 0:
        parser.error("--retries must be zero or positive")
    if args.retry_delay_seconds < 0:
        parser.error("--retry-delay-seconds must be zero or positive")
    if args.rows_per_batch <= 0:
        parser.error("--rows-per-batch must be positive")
    if args.progress_refresh_seconds <= 0:
        parser.error("--progress-refresh-seconds must be positive")
    return args


def safe_component(value):
    value = "".join("_" if character in INVALID_PATH_CHARS else character for character in str(value))
    value = value.strip().rstrip(".")
    return value or "unnamed"


def read_header(path):
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return csv.DictReader(handle).fieldnames or []


def output_fields(input_fields):
    conflicts = set(input_fields).intersection(STATUS_FIELDS)
    if conflicts:
        raise ValueError("Input CSV conflicts with download status columns: {}".format(", ".join(sorted(conflicts))))
    return input_fields + STATUS_FIELDS


def load_destination_paths(class_tree_path, args):
    with class_tree_path.open("r", encoding="utf-8-sig") as handle:
        root = json.load(handle)
    if not isinstance(root, dict):
        raise ValueError("Class-tree root must be a JSON object")
    top_level_nodes = root.get(args.children_key, [])
    if not isinstance(top_level_nodes, list) or not top_level_nodes:
        raise ValueError("Class-tree root key '{}' must contain a nonempty list".format(args.children_key))

    paths = {}

    def visit(node, top_label, components):
        if not isinstance(node, dict):
            raise ValueError("Class-tree nodes must be JSON objects")
        label = node.get(args.label_key)
        if not isinstance(label, str) or not label.startswith("/m/"):
            raise ValueError("Every non-root class-tree node must have an Open Images label ID")
        text_name = node.get(args.text_key)
        component = safe_component(text_name if isinstance(text_name, str) and text_name else label)
        current_components = components + (component,)
        path_key = (top_label, label)
        previous_path = paths.get(path_key)
        if previous_path and previous_path != current_components:
            raise ValueError(
                "Label '{}' appears multiple times under top-level class '{}'".format(label, top_label)
            )
        paths[path_key] = current_components
        children = node.get(args.children_key, [])
        if not isinstance(children, list):
            raise ValueError("JSON key '{}' must contain a list when present".format(args.children_key))
        for child in children:
            visit(child, top_label, current_components)

    for node in top_level_nodes:
        if not isinstance(node, dict):
            raise ValueError("Top-level class-tree entries must be JSON objects")
        top_label = node.get(args.label_key)
        if not isinstance(top_label, str) or not top_label.startswith("/m/"):
            raise ValueError("Every top-level class must have an Open Images label ID")
        visit(node, top_label, ())
    return paths


def count_rows(input_path, refresh_seconds):
    count = 0
    with input_path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.reader(handle)
        next(reader, None)
        with tqdm(desc="Counting input rows", unit="row", mininterval=refresh_seconds) as progress:
            for count, _ in enumerate(reader, start=1):
                progress.update(1)
    return count


def row_batches(reader, size):
    while True:
        rows = list(islice(reader, size))
        if not rows:
            return
        yield rows


def record_key(input_row_number):
    return "row-{:012d}".format(input_row_number)


def find_url(row, url_columns):
    for column in url_columns:
        value = (row.get(column) or "").strip()
        if value.startswith(("http://", "https://")):
            return value
    return ""


def extension_from_url(url):
    suffix = Path(urlparse(url).path).suffix.lower()
    return suffix if suffix in IMAGE_SUFFIXES else ".jpg"


def find_existing_image(destination_dir, image_id):
    for suffix in IMAGE_SUFFIXES:
        candidate = destination_dir / (image_id + suffix)
        if candidate.is_file() and candidate.stat().st_size > 0:
            return candidate
    return None


def download_record(row, input_row_number, destination_paths, args):
    image_id = safe_component(row.get(args.image_id_column, ""))
    label = (row.get(args.label_column) or "").strip()
    parent_label = (row.get(args.parent_label_column) or "").strip()
    hierarchy_path = destination_paths.get((parent_label, label))
    if not hierarchy_path:
        return {
            "DownloadStatus": "error",
            "DownloadHTTPStatus": "",
            "DownloadFinalURL": "",
            "DownloadError": "No JSON hierarchy path for parent '{}' and label '{}'".format(parent_label, label),
            "DownloadedImagePath": "",
        }
    if image_id == "unnamed":
        return {
            "DownloadStatus": "error",
            "DownloadHTTPStatus": "",
            "DownloadFinalURL": "",
            "DownloadError": "Missing image ID",
            "DownloadedImagePath": "",
        }
    destination_dir = args.output_dir.joinpath(*hierarchy_path)
    existing_image = find_existing_image(destination_dir, image_id)
    if existing_image:
        return {
            "DownloadStatus": "skipped",
            "DownloadHTTPStatus": "",
            "DownloadFinalURL": "",
            "DownloadError": "",
            "DownloadedImagePath": str(existing_image.resolve()),
        }
    url = find_url(row, args.url_columns)
    if not url:
        return {
            "DownloadStatus": "error",
            "DownloadHTTPStatus": "",
            "DownloadFinalURL": "",
            "DownloadError": "No valid HTTP(S) URL in {}".format(", ".join(args.url_columns)),
            "DownloadedImagePath": "",
        }

    destination_dir.mkdir(parents=True, exist_ok=True)
    destination = destination_dir / (image_id + extension_from_url(url))
    temporary = destination.with_name(
        destination.name + ".{}.inprogress".format(record_key(input_row_number))
    )
    last_error = ""
    last_http_status = ""
    last_final_url = ""
    for attempt in range(args.retries + 1):
        try:
            request = Request(url, headers={"User-Agent": "TinyCLIP OpenImage downloader"})
            with urlopen(request, timeout=args.timeout_seconds) as response:
                http_status = response.getcode() or 200
                final_url = response.geturl()
                if not 200 <= http_status < 300:
                    last_http_status = str(http_status)
                    last_final_url = final_url
                    last_error = "HTTP {}".format(http_status)
                else:
                    with temporary.open("wb") as handle:
                        shutil.copyfileobj(response, handle)
                    if temporary.stat().st_size == 0:
                        temporary.unlink(missing_ok=True)
                        last_http_status = str(http_status)
                        last_final_url = final_url
                        last_error = "Empty response body"
                    else:
                        temporary.replace(destination)
                        return {
                            "DownloadStatus": "downloaded",
                            "DownloadHTTPStatus": str(http_status),
                            "DownloadFinalURL": final_url,
                            "DownloadError": "",
                            "DownloadedImagePath": str(destination.resolve()),
                        }
        except HTTPError as error:
            last_http_status = str(error.code)
            last_final_url = error.geturl()
            last_error = "HTTP {}: {}".format(error.code, error.reason)
        except (URLError, TimeoutError, OSError) as error:
            last_error = str(error)
        finally:
            temporary.unlink(missing_ok=True)
        if attempt < args.retries and args.retry_delay_seconds:
            time.sleep(args.retry_delay_seconds)
    return {
        "DownloadStatus": "error",
        "DownloadHTTPStatus": last_http_status,
        "DownloadFinalURL": last_final_url,
        "DownloadError": last_error or "Download failed",
        "DownloadedImagePath": "",
    }


class CheckpointStore:
    def __init__(self, path):
        path.parent.mkdir(parents=True, exist_ok=True)
        self.connection = sqlite3.connect(path)
        self.connection.execute("PRAGMA journal_mode=WAL")
        self.connection.execute(
            "CREATE TABLE IF NOT EXISTS download_status ("
            "record_key TEXT PRIMARY KEY, "
            "download_status TEXT NOT NULL, "
            "http_status TEXT NOT NULL, "
            "final_url TEXT NOT NULL, "
            "error_text TEXT NOT NULL, "
            "image_path TEXT NOT NULL)"
        )
        self.connection.commit()

    def get(self, key):
        row = self.connection.execute(
            "SELECT download_status, http_status, final_url, error_text, image_path "
            "FROM download_status WHERE record_key = ?",
            (key,),
        ).fetchone()
        if row is None:
            return None
        return {
            "DownloadStatus": row[0],
            "DownloadHTTPStatus": row[1],
            "DownloadFinalURL": row[2],
            "DownloadError": row[3],
            "DownloadedImagePath": row[4],
        }

    def save(self, key, status):
        self.connection.execute(
            "INSERT INTO download_status (record_key, download_status, http_status, final_url, error_text, image_path) "
            "VALUES (?, ?, ?, ?, ?, ?) "
            "ON CONFLICT(record_key) DO UPDATE SET "
            "download_status = excluded.download_status, "
            "http_status = excluded.http_status, "
            "final_url = excluded.final_url, "
            "error_text = excluded.error_text, "
            "image_path = excluded.image_path",
            (
                key,
                status["DownloadStatus"],
                status["DownloadHTTPStatus"],
                status["DownloadFinalURL"],
                status["DownloadError"],
                status["DownloadedImagePath"],
            ),
        )
        self.connection.commit()

    def close(self):
        self.connection.close()


def process_downloads(input_path, destination_paths, checkpoint, args, total_rows):
    counts = {"downloaded": 0, "skipped": 0, "error": 0, "resumed": 0}
    completed_statuses = {"downloaded", "skipped"}
    if not args.retry_errors:
        completed_statuses.add("error")
    with input_path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        with ThreadPoolExecutor(max_workers=args.workers) as executor:
            with tqdm(
                total=total_rows,
                desc="Downloading images",
                unit="image",
                mininterval=args.progress_refresh_seconds,
            ) as progress:
                for batch in row_batches(enumerate(reader, start=2), args.rows_per_batch):
                    futures = {}
                    for input_row_number, row in batch:
                        key = record_key(input_row_number)
                        previous_status = checkpoint.get(key)
                        if previous_status and previous_status["DownloadStatus"] in completed_statuses:
                            counts["resumed"] += 1
                            progress.update(1)
                            continue
                        future = executor.submit(
                            download_record,
                            row,
                            input_row_number,
                            destination_paths,
                            args,
                        )
                        futures[future] = (key, input_row_number)
                    for future in as_completed(futures):
                        key, _ = futures[future]
                        try:
                            status = future.result()
                        except Exception as error:
                            status = {
                                "DownloadStatus": "error",
                                "DownloadHTTPStatus": "",
                                "DownloadFinalURL": "",
                                "DownloadError": "Unhandled downloader error: {}".format(error),
                                "DownloadedImagePath": "",
                            }
                        checkpoint.save(key, status)
                        counts[status["DownloadStatus"]] += 1
                        progress.update(1)
                        progress.set_postfix(**counts)
    return counts


def write_status_output(input_path, status_output, checkpoint, fields, total_rows, refresh_seconds):
    temporary_output = status_output.with_name(status_output.name + ".inprogress")
    temporary_output.unlink(missing_ok=True)
    with input_path.open("r", encoding="utf-8-sig", newline="") as input_handle:
        reader = csv.DictReader(input_handle)
        with temporary_output.open("w", encoding="utf-8-sig", newline="") as output_handle:
            writer = csv.DictWriter(output_handle, fieldnames=fields)
            writer.writeheader()
            with tqdm(
                total=total_rows,
                desc="Writing status table",
                unit="row",
                mininterval=refresh_seconds,
            ) as progress:
                for input_row_number, row in enumerate(reader, start=2):
                    status = checkpoint.get(record_key(input_row_number))
                    if status is None:
                        raise RuntimeError(
                            "Missing checkpoint status for input row {}. Use --resume to continue processing.".format(
                                input_row_number
                            )
                        )
                    output_row = dict(row)
                    output_row["InputRowNumber"] = input_row_number
                    output_row.update(status)
                    writer.writerow({field: output_row.get(field, "") for field in fields})
                    progress.update(1)
    temporary_output.replace(status_output)


def validate_state(status_output, checkpoint_db, overwrite, resume):
    if overwrite:
        status_output.unlink(missing_ok=True)
        checkpoint_db.unlink(missing_ok=True)
        checkpoint_db.with_name(checkpoint_db.name + "-wal").unlink(missing_ok=True)
        checkpoint_db.with_name(checkpoint_db.name + "-shm").unlink(missing_ok=True)
        return
    if status_output.exists() and not resume:
        raise FileExistsError("Status output already exists: {}. Use --resume or --overwrite.".format(status_output))
    if checkpoint_db.exists() and not resume:
        raise FileExistsError("Checkpoint database exists: {}. Use --resume or --overwrite.".format(checkpoint_db))


def main():
    args = parse_args()
    input_path = args.input.expanduser()
    class_tree_path = args.class_tree.expanduser()
    args.output_dir = args.output_dir.expanduser()
    status_output = args.status_output.expanduser()
    checkpoint_db = (
        args.checkpoint_db.expanduser()
        if args.checkpoint_db
        else status_output.parent / ".{}_checkpoint.sqlite3".format(status_output.stem)
    )
    if not input_path.is_file():
        raise FileNotFoundError("Input CSV does not exist: {}".format(input_path))
    if not class_tree_path.is_file():
        raise FileNotFoundError("Class-tree JSON does not exist: {}".format(class_tree_path))
    input_fields = read_header(input_path)
    required_columns = [args.image_id_column, args.label_column, args.parent_label_column]
    missing_columns = [column for column in required_columns if column not in input_fields]
    if missing_columns:
        raise ValueError("Input CSV is missing required columns: {}".format(", ".join(missing_columns)))
    if not any(column in input_fields for column in args.url_columns):
        raise ValueError("Input CSV has none of the URL columns: {}".format(", ".join(args.url_columns)))
    fields = output_fields(input_fields)
    validate_state(status_output, checkpoint_db, args.overwrite, args.resume)
    destination_paths = load_destination_paths(class_tree_path, args)
    total_rows = count_rows(input_path, args.progress_refresh_seconds)
    checkpoint = CheckpointStore(checkpoint_db)
    try:
        counts = process_downloads(input_path, destination_paths, checkpoint, args, total_rows)
        status_output.parent.mkdir(parents=True, exist_ok=True)
        write_status_output(
            input_path,
            status_output,
            checkpoint,
            fields,
            total_rows,
            args.progress_refresh_seconds,
        )
    finally:
        checkpoint.close()
    print("JSON hierarchy paths: {}".format(len(destination_paths)))
    print("Input rows: {}".format(total_rows))
    print("Download results: {}".format(counts))
    print("Status CSV: {}".format(status_output.resolve()))
    if args.cleanup_checkpoint:
        checkpoint_db.unlink(missing_ok=True)
        checkpoint_db.with_name(checkpoint_db.name + "-wal").unlink(missing_ok=True)
        checkpoint_db.with_name(checkpoint_db.name + "-shm").unlink(missing_ok=True)
        print("Removed checkpoint database: {}".format(checkpoint_db.resolve()))
    else:
        print("Checkpoint database: {}".format(checkpoint_db.resolve()))


if __name__ == "__main__":
    main()
