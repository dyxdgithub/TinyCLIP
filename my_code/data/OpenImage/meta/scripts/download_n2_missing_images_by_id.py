"""Download selected missing Open Images files directly by ImageID.

Only input rows whose FiftyOne status is ``not_found`` or ``error`` are
eligible by default. The downloader tries the public Open Images S3 object for
each selected ImageID, then any OriginalURL and Thumbnail300KURL values present
in the input CSV. Images are copied into the existing n2 parent/leaf hierarchy.

Unlike TFDS, this script never prepares or scans the complete dataset. An
SQLite checkpoint records every input row so interrupted runs can be resumed.
"""

import argparse
import csv
import json
import shutil
import sqlite3
import time
import urllib.error
import urllib.parse
import urllib.request
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from pathlib import Path

from PIL import Image
from tqdm import tqdm


META_DIR = Path(__file__).resolve().parent.parent
N2_DIR = META_DIR / "Hierarchy" / "n2"
DEFAULT_INPUT = N2_DIR / "Image_IDs_n2_all_fiftyone_download_status.csv"
DEFAULT_CLASS_TREE = N2_DIR / "Last_Two_Layers_Multi_Members_n2.json"
DEFAULT_OUTPUT_DIR = N2_DIR / "images_fiftyone_all"
DEFAULT_STATUS_OUTPUT = N2_DIR / "Image_IDs_n2_all_direct_download_status.csv"
DEFAULT_S3_TEMPLATE = "https://open-images-dataset.s3.amazonaws.com/{split}/{image_id}.jpg"
INVALID_PATH_CHARS = '<>:"/\\|?*'
IMAGE_SUFFIXES = (".jpg", ".jpeg", ".png", ".webp", ".bmp")
STATUS_FIELDS = [
    "DirectDownloadStatus",
    "DirectDownloadSourceURL",
    "DirectDownloadedImagePath",
    "DirectDownloadError",
]


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT,
                        help="FiftyOne status CSV. Default: %(default)s")
    parser.add_argument("--class-tree", type=Path, default=DEFAULT_CLASS_TREE,
                        help="n2 parent/leaf class-tree JSON. Default: %(default)s")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR,
                        help="Parent/leaf image hierarchy destination. Default: %(default)s")
    parser.add_argument("--status-output", type=Path, default=DEFAULT_STATUS_OUTPUT,
                        help="Combined input plus direct-download status fields. Default: %(default)s")
    parser.add_argument("--checkpoint-db", type=Path, default=None,
                        help="SQLite resume checkpoint. Default: hidden database beside --status-output.")
    parser.add_argument("--split", choices=("train", "validation", "test"), default="train",
                        help="Open Images S3 split used in source URLs. Accepted: train, validation, test. Default: %(default)s")
    parser.add_argument("--workers", type=int, default=16,
                        help="Maximum concurrent image downloads. Default: %(default)s")
    parser.add_argument("--source-url-template", action="append", default=None,
                        help="Primary URL template; repeatable. Fields: {split}, {image_id}. Default: public Open Images S3 URL.")
    parser.add_argument("--disable-metadata-urls", action="store_true",
                        help="Do not try OriginalURL or Thumbnail300KURL after the primary URL templates. Default: disabled.")
    parser.add_argument("--original-url-column", default="OriginalURL",
                        help="Original image URL column used as fallback when present. Default: %(default)s")
    parser.add_argument("--thumbnail-url-column", default="Thumbnail300KURL",
                        help="Thumbnail URL column used as fallback when present. Default: %(default)s")
    parser.add_argument("--url-timeout", type=float, default=30.0,
                        help="Timeout in seconds for each HTTP request. Default: %(default)s")
    parser.add_argument("--url-retries", type=int, default=3,
                        help="Attempts per candidate URL, including the first attempt. Default: %(default)s")
    parser.add_argument("--url-backoff", type=float, default=1.0,
                        help="Base exponential retry delay in seconds. Default: %(default)s")
    parser.add_argument("--eligible-fiftyone-status", action="append", default=None,
                        help="Eligible FiftyOne status; repeatable. Default: not_found and error.")
    parser.add_argument("--fiftyone-status-column", default="FiftyOneDownloadStatus",
                        help="FiftyOne status column in --input. Default: %(default)s")
    parser.add_argument("--image-id-column", default="ImageID",
                        help="Open Images ImageID column. Default: %(default)s")
    parser.add_argument("--label-column", default="LabelName",
                        help="n2 leaf-label column. Default: %(default)s")
    parser.add_argument("--parent-label-column", default="ParentLabelName",
                        help="n2 parent-label column. Default: %(default)s")
    parser.add_argument("--label-key", default="LabelName",
                        help="Class-tree label key. Default: %(default)s")
    parser.add_argument("--text-key", default="TextName",
                        help="Class-tree display-name key. Default: %(default)s")
    parser.add_argument("--children-key", default="Subcategory",
                        help="Class-tree child key. Default: %(default)s")
    parser.add_argument("--max-image-ids", type=int, default=None,
                        help="Download only the first N pending unique IDs for a connectivity test. Default: all pending IDs.")
    parser.add_argument("--progress-refresh-seconds", type=float, default=0.1,
                        help="Minimum tqdm refresh interval in seconds. Default: %(default)s")
    parser.add_argument("--resume", action="store_true",
                        help="Reuse the checkpoint and retry prior error/not_found rows. Default: disabled.")
    parser.add_argument("--overwrite", action="store_true",
                        help="Discard this script's checkpoint and status CSV; keep downloaded images. Default: disabled.")
    parser.add_argument("--cleanup-checkpoint", action="store_true",
                        help="Delete the SQLite checkpoint after writing the status CSV. Default: disabled.")
    args = parser.parse_args()
    if args.workers <= 0:
        parser.error("--workers must be positive")
    if args.url_timeout <= 0 or args.url_retries <= 0 or args.url_backoff < 0:
        parser.error("--url-timeout and --url-retries must be positive; --url-backoff must be nonnegative")
    if args.max_image_ids is not None and args.max_image_ids <= 0:
        parser.error("--max-image-ids must be positive")
    if args.progress_refresh_seconds <= 0:
        parser.error("--progress-refresh-seconds must be positive")
    if args.resume and args.overwrite:
        parser.error("Use either --resume or --overwrite, not both")
    args.eligible_fiftyone_status = tuple(args.eligible_fiftyone_status or ("not_found", "error"))
    args.source_url_template = tuple(args.source_url_template or (DEFAULT_S3_TEMPLATE,))
    for template in args.source_url_template:
        try:
            template.format(split="train", image_id="example")
        except (KeyError, ValueError) as error:
            parser.error("Invalid --source-url-template {!r}: {}".format(template, error))
    return args


def safe_component(value):
    value = "".join("_" if character in INVALID_PATH_CHARS else character for character in str(value))
    return value.strip().rstrip(".") or "unnamed"


def read_header(path):
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return csv.DictReader(handle).fieldnames or []


def load_leaf_paths(class_tree_path, args):
    with class_tree_path.open("r", encoding="utf-8-sig") as handle:
        root = json.load(handle)
    nodes = root.get(args.children_key) if isinstance(root, dict) else None
    if not isinstance(nodes, list) or not nodes:
        raise ValueError("Class-tree root key {!r} must contain a nonempty list".format(args.children_key))
    paths = {}

    def visit(node, parent_label=None, parent_name=None):
        if not isinstance(node, dict):
            raise ValueError("Class-tree nodes must be JSON objects")
        label = node.get(args.label_key)
        if not isinstance(label, str) or not label:
            raise ValueError("Class-tree node lacks {!r}".format(args.label_key))
        name = node.get(args.text_key)
        name = name if isinstance(name, str) and name else label
        children = node.get(args.children_key, [])
        if children:
            if not isinstance(children, list):
                raise ValueError("Class-tree child value must be a list at {!r}".format(label))
            for child in children:
                visit(child, label, name)
            return
        if parent_label is None:
            raise ValueError("Class-tree top-level label {!r} cannot be a leaf".format(label))
        paths.setdefault((parent_label, label), (safe_component(parent_name), safe_component(name)))

    for node in nodes:
        visit(node)
    return paths


def find_existing_image(destination_dir, image_id):
    for suffix in IMAGE_SUFFIXES:
        candidate = destination_dir / (image_id + suffix)
        if candidate.is_file() and candidate.stat().st_size > 0:
            return candidate
    return None


def copy_image(source_path, destination):
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(destination.stem + ".inprogress" + destination.suffix)
    temporary.unlink(missing_ok=True)
    try:
        shutil.copyfile(source_path, temporary)
        temporary.replace(destination)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise


def detect_image_suffix(path):
    with Image.open(path) as image:
        image.verify()
        return {
            "JPEG": ".jpg",
            "PNG": ".png",
            "WEBP": ".webp",
            "BMP": ".bmp",
        }.get(image.format, ".jpg")


def candidate_urls(image_id, metadata_urls, args):
    quoted_id = urllib.parse.quote(image_id, safe="")
    quoted_split = urllib.parse.quote(args.split, safe="")
    candidates = []
    for template in args.source_url_template:
        candidates.append((template.format(split=quoted_split, image_id=quoted_id), "s3"))
    if not args.disable_metadata_urls:
        candidates.extend((url, "metadata") for url in metadata_urls)
    unique = []
    seen = set()
    for candidate in candidates:
        if candidate[0] and candidate[0] not in seen:
            unique.append(candidate)
            seen.add(candidate[0])
    return unique


def download_image(image_id, candidates, temp_dir, args):
    temp_dir.mkdir(parents=True, exist_ok=True)
    destination = temp_dir / (safe_component(image_id) + ".download")
    errors = []
    all_not_found = bool(candidates)
    for url, source_kind in candidates:
        for attempt in range(args.url_retries):
            destination.unlink(missing_ok=True)
            request = urllib.request.Request(url, headers={"User-Agent": "TinyCLIP-OpenImages-downloader/1.0"})
            try:
                with urllib.request.urlopen(request, timeout=args.url_timeout) as response:
                    with destination.open("wb") as handle:
                        shutil.copyfileobj(response, handle)
                if destination.stat().st_size == 0:
                    raise ValueError("URL returned an empty response")
                suffix = detect_image_suffix(destination)
                return destination, suffix, url, source_kind, ""
            except urllib.error.HTTPError as error:
                destination.unlink(missing_ok=True)
                errors.append("{} (HTTP {}): {}".format(url, error.code, error.reason))
                if error.code not in (403, 404):
                    all_not_found = False
                if error.code in (403, 404):
                    break
            except (OSError, ValueError, urllib.error.URLError) as error:
                destination.unlink(missing_ok=True)
                all_not_found = False
                errors.append("{} (attempt {}/{}): {}".format(url, attempt + 1, args.url_retries, error))
            if attempt + 1 < args.url_retries and args.url_backoff:
                time.sleep(args.url_backoff * (2 ** attempt))
    message = " | ".join(errors) if errors else "No candidate URL was available"
    return None, None, "", "not_found" if all_not_found else "error", message[:4000]


def load_metadata_urls(input_path, requested_ids, args):
    if args.disable_metadata_urls:
        return {}
    header = set(read_header(input_path))
    url_fields = [field for field in (args.original_url_column, args.thumbnail_url_column) if field in header]
    if not url_fields:
        return {}
    image_urls = {}
    with input_path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            image_id = (row.get(args.image_id_column) or "").strip()
            if image_id not in requested_ids:
                continue
            urls = image_urls.setdefault(image_id, [])
            for field in url_fields:
                url = (row.get(field) or "").strip()
                if url and url not in urls:
                    urls.append(url)
    return image_urls


class DownloadStore:
    def __init__(self, path):
        path.parent.mkdir(parents=True, exist_ok=True)
        self.connection = sqlite3.connect(path)
        self.connection.execute("PRAGMA journal_mode=WAL")
        self.connection.execute(
            "CREATE TABLE IF NOT EXISTS rows "
            "(input_row_number INTEGER PRIMARY KEY, image_id TEXT NOT NULL, parent_label TEXT NOT NULL, "
            "leaf_label TEXT NOT NULL, fiftyone_status TEXT NOT NULL, direct_status TEXT NOT NULL, "
            "source_url TEXT NOT NULL, image_path TEXT NOT NULL, error_text TEXT NOT NULL)"
        )
        self.connection.commit()

    def seed_rows(self, input_path, args):
        eligible = set(args.eligible_fiftyone_status)
        batch = []
        count = 0
        with input_path.open("r", encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle)
            for number, row in enumerate(reader, start=2):
                count += 1
                image_id = (row.get(args.image_id_column) or "").strip()
                parent = (row.get(args.parent_label_column) or "").strip()
                leaf = (row.get(args.label_column) or "").strip()
                fiftyone_status = (row.get(args.fiftyone_status_column) or "").strip()
                if not image_id or not parent or not leaf:
                    raise ValueError("Input row {} lacks ImageID, parent label, or leaf label".format(number))
                initial_status = "pending" if fiftyone_status in eligible else "skipped_fiftyone_success"
                batch.append((number, image_id, parent, leaf, fiftyone_status, initial_status, "", "", ""))
                if len(batch) >= 5000:
                    self._insert_batch(batch)
                    batch.clear()
        if batch:
            self._insert_batch(batch)
        self.connection.commit()
        return count

    def _insert_batch(self, rows):
        self.connection.executemany(
            "INSERT OR IGNORE INTO rows "
            "(input_row_number, image_id, parent_label, leaf_label, fiftyone_status, direct_status, source_url, image_path, error_text) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
            rows,
        )

    def pending_by_image_id(self):
        rows = self.connection.execute(
            "SELECT input_row_number, image_id, parent_label, leaf_label FROM rows "
            "WHERE direct_status IN ('pending', 'error', 'not_found') ORDER BY input_row_number"
        ).fetchall()
        targets = {}
        for row in rows:
            targets.setdefault(row[1], []).append(row)
        return targets

    def record_results(self, results):
        self.connection.executemany(
            "UPDATE rows SET direct_status = ?, source_url = ?, image_path = ?, error_text = ? "
            "WHERE input_row_number = ?",
            ((status, source_url, image_path, error_text, row_number)
             for row_number, status, source_url, image_path, error_text in results),
        )
        self.connection.commit()

    def ordered_statuses(self):
        return self.connection.execute(
            "SELECT input_row_number, direct_status, source_url, image_path, error_text "
            "FROM rows ORDER BY input_row_number"
        )

    def summary(self):
        return dict(self.connection.execute(
            "SELECT direct_status, COUNT(*) FROM rows GROUP BY direct_status"
        ).fetchall())

    def close(self):
        self.connection.close()


def process_image(image_id, rows, metadata_urls, hierarchy_paths, output_dir, temp_dir, args):
    results = []
    missing = []
    for row_number, _, parent_label, leaf_label in rows:
        hierarchy_path = hierarchy_paths.get((parent_label, leaf_label))
        if hierarchy_path is None:
            results.append((
                row_number, "error", "", "",
                "Input parent/leaf pair ({!r}, {!r}) is absent from the class tree".format(parent_label, leaf_label),
            ))
            continue
        parent_name, leaf_name = hierarchy_path
        destination_dir = output_dir / parent_name / leaf_name
        existing = find_existing_image(destination_dir, image_id)
        if existing is not None:
            results.append((row_number, "existing", "", str(existing), ""))
        else:
            missing.append((row_number, destination_dir))
    if not missing:
        return results

    download = download_image(image_id, candidate_urls(image_id, metadata_urls, args), temp_dir, args)
    source_path, suffix, source_url, source_kind, error_text = download
    if source_path is None:
        results.extend((row_number, source_kind, "", "", error_text) for row_number, _ in missing)
        return results
    try:
        status = "downloaded_s3" if source_kind == "s3" else "downloaded_url"
        for row_number, destination_dir in missing:
            destination = destination_dir / (image_id + suffix)
            try:
                copy_image(source_path, destination)
                results.append((row_number, status, source_url, str(destination), ""))
            except Exception as error:
                results.append((row_number, "error", source_url, "", str(error)))
    finally:
        source_path.unlink(missing_ok=True)
    return results


def bounded_results(image_ids, targets, metadata_urls, hierarchy_paths, output_dir, temp_dir, args):
    iterator = iter(image_ids)
    pending = set()
    limit = max(args.workers * 2, 1)
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        while len(pending) < limit:
            try:
                image_id = next(iterator)
            except StopIteration:
                break
            pending.add(executor.submit(
                process_image, image_id, targets[image_id], metadata_urls.get(image_id, []),
                hierarchy_paths, output_dir, temp_dir, args,
            ))
        while pending:
            done, pending = wait(pending, return_when=FIRST_COMPLETED)
            for future in done:
                yield future.result()
                try:
                    image_id = next(iterator)
                except StopIteration:
                    continue
                pending.add(executor.submit(
                    process_image, image_id, targets[image_id], metadata_urls.get(image_id, []),
                    hierarchy_paths, output_dir, temp_dir, args,
                ))


def write_status_output(input_path, output_path, store):
    fields = read_header(input_path)
    conflicts = set(fields).intersection(STATUS_FIELDS)
    if conflicts:
        raise ValueError("Input CSV conflicts with direct status columns: {}".format(", ".join(sorted(conflicts))))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_name(output_path.name + ".inprogress")
    temporary.unlink(missing_ok=True)
    statuses = iter(store.ordered_statuses())
    with input_path.open("r", encoding="utf-8-sig", newline="") as source, \
            temporary.open("w", encoding="utf-8-sig", newline="") as destination:
        reader = csv.DictReader(source)
        writer = csv.DictWriter(destination, fieldnames=fields + STATUS_FIELDS)
        writer.writeheader()
        for number, row in enumerate(reader, start=2):
            stored_number, status, source_url, image_path, error_text = next(statuses)
            if stored_number != number:
                raise RuntimeError("Checkpoint row numbering does not match the input CSV")
            row.update({
                "DirectDownloadStatus": status,
                "DirectDownloadSourceURL": source_url,
                "DirectDownloadedImagePath": image_path,
                "DirectDownloadError": error_text,
            })
            writer.writerow(row)
    temporary.replace(output_path)


def remove_checkpoint(path):
    path.unlink(missing_ok=True)
    path.with_name(path.name + "-wal").unlink(missing_ok=True)
    path.with_name(path.name + "-shm").unlink(missing_ok=True)


def main():
    args = parse_args()
    input_path = args.input.expanduser()
    class_tree_path = args.class_tree.expanduser()
    output_dir = args.output_dir.expanduser()
    status_output = args.status_output.expanduser()
    checkpoint_db = (
        args.checkpoint_db.expanduser() if args.checkpoint_db
        else status_output.with_name("." + status_output.stem + "_checkpoint.sqlite3")
    )
    if not input_path.is_file():
        raise FileNotFoundError("FiftyOne status CSV does not exist: {}".format(input_path))
    if not class_tree_path.is_file():
        raise FileNotFoundError("Class-tree JSON does not exist: {}".format(class_tree_path))
    header = read_header(input_path)
    required = {
        args.image_id_column, args.parent_label_column, args.label_column,
        args.fiftyone_status_column,
    }
    missing = required.difference(header)
    if missing:
        raise ValueError("Input CSV is missing required columns: {}".format(", ".join(sorted(missing))))
    if args.overwrite:
        remove_checkpoint(checkpoint_db)
        status_output.unlink(missing_ok=True)
    elif (checkpoint_db.exists() or status_output.exists()) and not args.resume:
        raise FileExistsError("Output or checkpoint exists. Use --resume or --overwrite. Checkpoint: {}".format(checkpoint_db))

    hierarchy_paths = load_leaf_paths(class_tree_path, args)
    store = DownloadStore(checkpoint_db)
    try:
        input_count = store.seed_rows(input_path, args)
        targets = store.pending_by_image_id()
        if not targets:
            write_status_output(input_path, status_output, store)
            print("No rows need direct recovery. Status CSV: {}".format(status_output.resolve()))
            return
        requested_ids = sorted(targets)
        if args.max_image_ids is not None:
            requested_ids = requested_ids[:args.max_image_ids]
        requested_set = set(requested_ids)
        metadata_urls = load_metadata_urls(input_path, requested_set, args)
        output_dir.mkdir(parents=True, exist_ok=True)
        temp_dir = output_dir / ".direct_download_cache"
        print("Directly downloading {} unique Open Images IDs...".format(len(requested_ids)), flush=True)
        counts = {}
        total_rows = sum(len(targets[image_id]) for image_id in requested_ids)
        with tqdm(total=total_rows, desc="Downloading selected images", unit="row",
                  mininterval=args.progress_refresh_seconds) as progress:
            for results in bounded_results(
                    requested_ids, targets, metadata_urls, hierarchy_paths,
                    output_dir, temp_dir, args):
                store.record_results(results)
                for _, status, _, _, _ in results:
                    counts[status] = counts.get(status, 0) + 1
                progress.update(len(results))
                progress.set_postfix(**counts)
        try:
            temp_dir.rmdir()
        except OSError:
            pass
        if args.max_image_ids is not None:
            print("Stopped at --max-image-ids; unselected rows remain pending.", flush=True)
        write_status_output(input_path, status_output, store)
        print("Input rows: {}".format(input_count))
        print("Direct-download status counts: {}".format(store.summary()))
        print("Status CSV: {}".format(status_output.resolve()))
        print("Checkpoint database: {}".format(checkpoint_db.resolve()))
    finally:
        store.close()
    if args.cleanup_checkpoint:
        remove_checkpoint(checkpoint_db)


if __name__ == "__main__":
    main()
