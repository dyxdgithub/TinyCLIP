"""Download sampled n2 Open Images through the FiftyOne Open Images V7 Zoo.

The script requests only ImageIDs present in the sampled n2 manifest, copies
the downloaded files into the n2 parent/leaf hierarchy, and records one status
per manifest row. If FiftyOne does not return an image, the manifest's
OriginalURL and Thumbnail300KURL columns are tried as a direct-download
fallback. FiftyOne keeps its own Zoo cache; a SQLite checkpoint keeps the
hierarchy-copy step resumable.
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
from pathlib import Path

from PIL import Image
from tqdm import tqdm


META_DIR = Path(__file__).resolve().parent.parent
N2_DIR = META_DIR / "Hierarchy" / "n2"
SAMPLED_DIR = N2_DIR / "300"
DEFAULT_INPUT = SAMPLED_DIR / "Image IDs_with_last_two_layers_multi_members_n2_one_label_per_parent_sampled_300_per_class.csv"
DEFAULT_CLASS_TREE = N2_DIR / "Last_Two_Layers_Multi_Members_n2.json"
DEFAULT_OUTPUT_DIR = SAMPLED_DIR / "images_fiftyone_sampled_300_per_class"
DEFAULT_STATUS_OUTPUT = SAMPLED_DIR / "Image IDs_with_last_two_layers_multi_members_n2_one_label_per_parent_sampled_300_per_class_fiftyone_download_status.csv"
DEFAULT_FIFTYONE_DATASET_NAME = "tinyclip_n2_sampled_open_images_v7"
INVALID_PATH_CHARS = '<>:"/\\|?*'
IMAGE_SUFFIXES = (".jpg", ".jpeg", ".png", ".webp", ".bmp")
STATUS_FIELDS = [
    "FiftyOneDownloadStatus",
    "FiftyOneSourceFilePath",
    "FiftyOneDownloadedImagePath",
    "FiftyOneDownloadError",
]


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT, help="Sampled one-label-per-parent input CSV. Default: %(default)s")
    parser.add_argument("--class-tree", type=Path, default=DEFAULT_CLASS_TREE, help="n2 class-tree JSON. Default: %(default)s")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR, help="Root output directory for parent/leaf image folders. Default: %(default)s")
    parser.add_argument("--status-output", type=Path, default=DEFAULT_STATUS_OUTPUT, help="Output status CSV. Default: %(default)s")
    parser.add_argument("--checkpoint-db", type=Path, default=None, help="SQLite resume checkpoint. Default: hidden database beside --status-output.")
    parser.add_argument("--zoo-dataset", default="open-images-v7", help="FiftyOne Zoo dataset name. Default: %(default)s")
    parser.add_argument("--split", default="train", help="Open Images split requested from FiftyOne. Default: %(default)s")
    parser.add_argument("--fiftyone-dataset-name", default=DEFAULT_FIFTYONE_DATASET_NAME, help="Persistent local FiftyOne dataset name. Default: %(default)s")
    parser.add_argument("--dataset-dir", type=Path, default=None, help="Optional FiftyOne Zoo download/cache directory. Default: FiftyOne's configured cache directory.")
    parser.add_argument("--workers", type=int, default=32, help="Maximum concurrent FiftyOne image-download workers. Default: %(default)s")
    parser.add_argument("--image-id-column", default="ImageID", help="Target ImageID column in --input. Default: %(default)s")
    parser.add_argument("--label-column", default="LabelName", help="Target leaf-label column in --input. Default: %(default)s")
    parser.add_argument("--parent-label-column", default="ParentLabelName", help="Target parent-label column in --input. Default: %(default)s")
    parser.add_argument("--original-url-column", default="OriginalURL", help="Full-size image URL column used when FiftyOne has no image. Default: %(default)s")
    parser.add_argument("--thumbnail-url-column", default="Thumbnail300KURL", help="Fallback thumbnail URL column used when OriginalURL fails. Default: %(default)s")
    parser.add_argument("--url-timeout", type=float, default=30.0, help="Timeout in seconds for each direct URL request. Default: %(default)s")
    parser.add_argument("--url-retries", type=int, default=3, help="Maximum attempts per candidate URL, including the first attempt. Default: %(default)s")
    parser.add_argument("--url-backoff", type=float, default=1.0, help="Base exponential delay in seconds between URL retries. Default: %(default)s")
    parser.add_argument("--label-key", default="LabelName", help="Class-label key in --class-tree. Default: %(default)s")
    parser.add_argument("--text-key", default="TextName", help="Class display-name key in --class-tree. Default: %(default)s")
    parser.add_argument("--children-key", default="Subcategory", help="Child-node key in --class-tree. Default: %(default)s")
    parser.add_argument("--max-image-ids", type=int, default=None, help="Request only the first N pending unique ImageIDs for a FiftyOne connectivity test. Default: all pending IDs.")
    parser.add_argument("--progress-refresh-seconds", type=float, default=0.1, help="Minimum seconds between tqdm progress refreshes. Default: %(default)s")
    parser.add_argument("--resume", action="store_true", help="Reuse prior checkpoint rows and the persistent FiftyOne dataset; retry prior error/not-found rows.")
    parser.add_argument("--overwrite", action="store_true", help="Remove prior output and checkpoint state. The FiftyOne Zoo dataset and cached images are retained.")
    parser.add_argument("--cleanup-checkpoint", action="store_true", help="Delete the SQLite checkpoint after writing the status CSV.")
    args = parser.parse_args()
    if args.workers <= 0:
        parser.error("--workers must be positive")
    if args.url_timeout <= 0 or args.url_retries <= 0 or args.url_backoff < 0:
        parser.error("--url-timeout and --url-retries must be positive; --url-backoff must be nonnegative")
    if args.max_image_ids is not None and args.max_image_ids <= 0:
        parser.error("--max-image-ids must be positive")
    if args.progress_refresh_seconds <= 0:
        parser.error("--progress-refresh-seconds must be positive")
    return args


def load_fiftyone_runtime():
    try:
        import fiftyone.zoo as foz
    except ImportError as error:
        raise RuntimeError("This script requires FiftyOne. Install requirements-training.txt first.") from error
    return foz


def safe_component(value):
    value = "".join("_" if character in INVALID_PATH_CHARS else character for character in str(value))
    return value.strip().rstrip(".") or "unnamed"


def read_header(path):
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return csv.DictReader(handle).fieldnames or []


def count_rows(path, refresh_seconds):
    count = 0
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.reader(handle)
        next(reader, None)
        with tqdm(desc="Counting input rows", unit="row", mininterval=refresh_seconds) as progress:
            for count, _ in enumerate(reader, start=1):
                progress.update(1)
    return count


def load_leaf_paths(class_tree_path, args):
    with class_tree_path.open("r", encoding="utf-8-sig") as handle:
        root = json.load(handle)
    nodes = root.get(args.children_key) if isinstance(root, dict) else None
    if not isinstance(nodes, list) or not nodes:
        raise ValueError("Class-tree root key '{}' must contain a nonempty list".format(args.children_key))
    paths = {}

    def visit(node, parent_label=None, parent_name=None):
        if not isinstance(node, dict):
            raise ValueError("Class-tree nodes must be JSON objects")
        label = node.get(args.label_key)
        if not isinstance(label, str) or not label:
            raise ValueError("Class-tree node lacks '{}'".format(args.label_key))
        name = node.get(args.text_key)
        name = name if isinstance(name, str) and name else label
        children = node.get(args.children_key, [])
        if children:
            if not isinstance(children, list):
                raise ValueError("Class-tree child value must be a list at '{}'".format(label))
            for child in children:
                visit(child, label, name)
            return
        if parent_label is None:
            raise ValueError("Class-tree top-level label '{}' cannot be a leaf".format(label))
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
    if not source_path.is_file() or source_path.stat().st_size == 0:
        raise FileNotFoundError("FiftyOne source image does not exist or is empty: {}".format(source_path))
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(destination.stem + ".inprogress" + destination.suffix)
    temporary.unlink(missing_ok=True)
    try:
        shutil.copyfile(source_path, temporary)
        temporary.replace(destination)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise


def load_image_urls(input_path, args):
    """Index original and thumbnail URLs by ImageID for the fallback path."""
    available_fields = set(read_header(input_path))
    url_fields = [
        field for field in (args.original_url_column, args.thumbnail_url_column)
        if field in available_fields
    ]
    if not url_fields:
        raise ValueError(
            "Input CSV must contain at least one configured URL column: {} or {}".format(
                args.original_url_column, args.thumbnail_url_column
            )
        )
    image_urls = {}
    with input_path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        with tqdm(desc="Indexing fallback URLs", unit="row", mininterval=args.progress_refresh_seconds) as progress:
            for row in reader:
                image_id = (row.get(args.image_id_column) or "").strip()
                if image_id:
                    urls = image_urls.setdefault(image_id, [])
                    for field in url_fields:
                        url = (row.get(field) or "").strip()
                        if url and url not in urls:
                            urls.append(url)
                progress.update(1)
    return image_urls


def download_url_fallback(image_id, urls, temp_dir, args):
    """Download and validate one image, trying each available URL with retries."""
    Path(temp_dir).mkdir(parents=True, exist_ok=True)
    errors = []
    for url in urls:
        parsed = urllib.parse.urlparse(url)
        suffix = Path(parsed.path).suffix.lower()
        if suffix not in IMAGE_SUFFIXES:
            suffix = ".jpg"
        destination = Path(temp_dir) / (safe_component(image_id) + suffix)
        for attempt in range(args.url_retries):
            temporary = destination.with_name(destination.name + ".inprogress")
            temporary.unlink(missing_ok=True)
            request = urllib.request.Request(
                url,
                headers={"User-Agent": "TinyCLIP-OpenImages-downloader/1.0"},
            )
            try:
                with urllib.request.urlopen(request, timeout=args.url_timeout) as response:
                    content_type = response.headers.get_content_type()
                    if content_type and not content_type.startswith("image/"):
                        raise ValueError("URL returned non-image content type: {}".format(content_type))
                    with temporary.open("wb") as handle:
                        shutil.copyfileobj(response, handle)
                if temporary.stat().st_size == 0:
                    raise ValueError("URL returned an empty response")
                with Image.open(temporary) as image:
                    image.verify()
                    detected_suffix = {
                        "JPEG": ".jpg",
                        "PNG": ".png",
                        "WEBP": ".webp",
                        "BMP": ".bmp",
                    }.get(image.format)
                if detected_suffix in IMAGE_SUFFIXES and suffix == ".jpg":
                    destination = destination.with_suffix(detected_suffix)
                temporary.replace(destination)
                return destination, url
            except (OSError, ValueError, urllib.error.URLError, urllib.error.HTTPError) as error:
                temporary.unlink(missing_ok=True)
                errors.append("{} (attempt {}/{}): {}".format(url, attempt + 1, args.url_retries, error))
                if attempt + 1 < args.url_retries and args.url_backoff:
                    time.sleep(args.url_backoff * (2 ** attempt))
        destination.unlink(missing_ok=True)
    raise RuntimeError("All fallback URLs failed: {}".format(" | ".join(errors) if errors else "no URL was present"))


class DownloadStore:
    def __init__(self, path):
        path.parent.mkdir(parents=True, exist_ok=True)
        self.connection = sqlite3.connect(path)
        self.connection.execute("PRAGMA journal_mode=WAL")
        self.connection.execute("CREATE TABLE IF NOT EXISTS rows (input_row_number INTEGER PRIMARY KEY, image_id TEXT NOT NULL, parent_label TEXT NOT NULL, leaf_label TEXT NOT NULL, status TEXT NOT NULL, source_path TEXT NOT NULL, image_path TEXT NOT NULL, error_text TEXT NOT NULL)")
        self.connection.commit()

    def seed_rows(self, input_path, args):
        rows = []
        with input_path.open("r", encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle)
            for number, row in enumerate(reader, start=2):
                image_id = (row.get(args.image_id_column) or "").strip()
                parent = (row.get(args.parent_label_column) or "").strip()
                leaf = (row.get(args.label_column) or "").strip()
                if not image_id or not parent or not leaf:
                    raise ValueError("Input row {} lacks ImageID, parent label, or leaf label".format(number))
                rows.append((number, image_id, parent, leaf, "pending", "", "", ""))
        self.connection.executemany("INSERT OR IGNORE INTO rows (input_row_number, image_id, parent_label, leaf_label, status, source_path, image_path, error_text) VALUES (?, ?, ?, ?, ?, ?, ?, ?)", rows)
        self.connection.commit()

    def pending_by_image_id(self):
        rows = self.connection.execute("SELECT input_row_number, image_id, parent_label, leaf_label FROM rows WHERE status NOT IN ('downloaded', 'downloaded_url', 'existing') ORDER BY input_row_number").fetchall()
        targets = {}
        for row in rows:
            targets.setdefault(row[1], []).append(row)
        return targets

    def record_result(self, row_number, status, source_path="", image_path="", error_text=""):
        self.connection.execute("UPDATE rows SET status = ?, source_path = ?, image_path = ?, error_text = ? WHERE input_row_number = ?", (status, source_path, image_path, error_text, row_number))
        self.connection.commit()

    def mark_not_found(self, image_ids):
        self.connection.executemany("UPDATE rows SET status = 'not_found', source_path = '', image_path = '', error_text = 'ImageID was not found in the FiftyOne Open Images dataset' WHERE image_id = ? AND status NOT IN ('downloaded', 'downloaded_url', 'existing')", ((image_id,) for image_id in image_ids))
        self.connection.commit()

    def status_for_row(self, row_number):
        return self.connection.execute("SELECT status, source_path, image_path, error_text FROM rows WHERE input_row_number = ?", (row_number,)).fetchone()

    def summary(self):
        return dict(self.connection.execute("SELECT status, COUNT(*) FROM rows GROUP BY status").fetchall())

    def close(self):
        self.connection.close()


def write_status_output(input_path, output_path, store):
    fields = read_header(input_path)
    conflicts = set(fields).intersection(STATUS_FIELDS)
    if conflicts:
        raise ValueError("Input CSV conflicts with status columns: {}".format(", ".join(sorted(conflicts))))
    temporary = output_path.with_name(output_path.name + ".inprogress")
    temporary.unlink(missing_ok=True)
    with input_path.open("r", encoding="utf-8-sig", newline="") as source, temporary.open("w", encoding="utf-8-sig", newline="") as destination:
        reader = csv.DictReader(source)
        writer = csv.DictWriter(destination, fieldnames=fields + STATUS_FIELDS)
        writer.writeheader()
        for number, row in enumerate(reader, start=2):
            status, source_path, image_path, error_text = store.status_for_row(number)
            row.update({"FiftyOneDownloadStatus": status, "FiftyOneSourceFilePath": source_path, "FiftyOneDownloadedImagePath": image_path, "FiftyOneDownloadError": error_text})
            writer.writerow(row)
    temporary.replace(output_path)


def remove_checkpoint(path):
    path.unlink(missing_ok=True)
    path.with_name(path.name + "-wal").unlink(missing_ok=True)
    path.with_name(path.name + "-shm").unlink(missing_ok=True)


def remove_state(checkpoint_db, status_output):
    remove_checkpoint(checkpoint_db)
    status_output.unlink(missing_ok=True)


def sample_open_images_id(sample):
    if "open_images_id" in sample.field_names:
        image_id = sample.get_field("open_images_id")
        if image_id:
            return str(image_id).strip()
    return Path(sample.filepath).stem


def load_requested_dataset(foz, image_ids, args):
    kwargs = {
        "split": args.split,
        "label_types": [],
        "image_ids": image_ids,
        "num_workers": args.workers,
        "dataset_name": args.fiftyone_dataset_name,
    }
    if args.dataset_dir is not None:
        kwargs["dataset_dir"] = str(args.dataset_dir.expanduser())
    return foz.load_zoo_dataset(args.zoo_dataset, **kwargs)


def main():
    args = parse_args()
    input_path = args.input.expanduser()
    class_tree_path = args.class_tree.expanduser()
    output_dir = args.output_dir.expanduser()
    status_output = args.status_output.expanduser()
    checkpoint_db = args.checkpoint_db.expanduser() if args.checkpoint_db else status_output.with_name("." + status_output.stem + "_checkpoint.sqlite3")
    if not input_path.is_file():
        raise FileNotFoundError("Input CSV does not exist: {}".format(input_path))
    if not class_tree_path.is_file():
        raise FileNotFoundError("Class-tree JSON does not exist: {}".format(class_tree_path))
    header = read_header(input_path)
    for field in (args.image_id_column, args.parent_label_column, args.label_column):
        if field not in header:
            raise ValueError("Input CSV is missing required column '{}': {}".format(field, input_path))
    url_fields = {args.original_url_column, args.thumbnail_url_column}
    if not url_fields.intersection(header):
        raise ValueError("Input CSV needs at least one fallback URL column: {} or {}".format(args.original_url_column, args.thumbnail_url_column))
    if args.overwrite:
        remove_state(checkpoint_db, status_output)
    elif (checkpoint_db.exists() or status_output.exists()) and not args.resume:
        raise FileExistsError("Output or checkpoint exists. Use --resume or --overwrite. Checkpoint: {}".format(checkpoint_db))

    paths = load_leaf_paths(class_tree_path, args)
    input_count = count_rows(input_path, args.progress_refresh_seconds)
    image_urls = load_image_urls(input_path, args)
    store = DownloadStore(checkpoint_db)
    try:
        store.seed_rows(input_path, args)
        targets = store.pending_by_image_id()
        if not targets:
            status_output.parent.mkdir(parents=True, exist_ok=True)
            write_status_output(input_path, status_output, store)
            print("All input rows are already completed. Status CSV: {}".format(status_output.resolve()))
            return
        requested_ids = sorted(targets)
        if args.max_image_ids is not None:
            requested_ids = requested_ids[:args.max_image_ids]
        print("Requesting {} unique Open Images IDs through FiftyOne...".format(len(requested_ids)), flush=True)
        foz = load_fiftyone_runtime()
        dataset = load_requested_dataset(foz, requested_ids, args)
        source_paths = {}
        with tqdm(total=len(dataset), desc="Indexing FiftyOne samples", unit="sample", mininterval=args.progress_refresh_seconds) as progress:
            for sample in dataset:
                image_id = sample_open_images_id(sample)
                filepath = Path(sample.filepath)
                if image_id in targets and filepath.is_file():
                    source_paths.setdefault(image_id, filepath)
                progress.update(1)
        selected_targets = {image_id: targets[image_id] for image_id in requested_ids}
        counts = {"downloaded": 0, "downloaded_url": 0, "existing": 0, "error": 0, "not_found": 0}
        temp_dir = output_dir / ".url_fallback_cache"
        temp_dir.mkdir(parents=True, exist_ok=True)
        total_rows = sum(len(rows) for rows in selected_targets.values())
        with tqdm(total=total_rows, desc="Writing n2 hierarchy images", unit="row", mininterval=args.progress_refresh_seconds) as progress:
            for image_id, rows in selected_targets.items():
                source_path = source_paths.get(image_id)
                source_url = ""
                if source_path is None:
                    try:
                        source_path, source_url = download_url_fallback(
                            image_id,
                            image_urls.get(image_id, []),
                            temp_dir,
                            args,
                        )
                    except Exception as error:
                        for row_number, _, _, _ in rows:
                            store.record_result(
                                row_number,
                                "error",
                                error_text="ImageID was not returned by FiftyOne; URL fallback failed: {}".format(error),
                            )
                            counts["error"] += 1
                            progress.update(1)
                        progress.set_postfix(**counts)
                        continue
                try:
                    for row_number, _, parent_label, leaf_label in rows:
                        hierarchy_path = paths.get((parent_label, leaf_label))
                        if hierarchy_path is None:
                            store.record_result(row_number, "error", str(source_url or source_path), error_text="Input parent/leaf pair ({!r}, {!r}) is absent from the class tree".format(parent_label, leaf_label))
                            counts["error"] += 1
                            progress.update(1)
                            continue
                        parent_name, leaf_name = hierarchy_path
                        destination_dir = output_dir / parent_name / leaf_name
                        existing = find_existing_image(destination_dir, image_id)
                        if existing:
                            store.record_result(row_number, "existing", str(source_url or source_path), str(existing))
                            counts["existing"] += 1
                            progress.update(1)
                            continue
                        suffix = source_path.suffix.lower()
                        suffix = suffix if suffix in IMAGE_SUFFIXES else ".jpg"
                        destination = destination_dir / (image_id + suffix)
                        try:
                            copy_image(source_path, destination)
                            status = "downloaded_url" if source_url else "downloaded"
                            store.record_result(row_number, status, str(source_url or source_path), str(destination))
                            counts[status] += 1
                        except Exception as error:
                            store.record_result(row_number, "error", str(source_url or source_path), error_text=str(error))
                            counts["error"] += 1
                        progress.update(1)
                        progress.set_postfix(**counts)
                finally:
                    if source_url and source_path is not None:
                        source_path.unlink(missing_ok=True)
        try:
            temp_dir.rmdir()
        except OSError:
            pass
        if args.max_image_ids is None:
            store.mark_not_found(set(targets).difference(requested_ids))
        else:
            print("FiftyOne request stopped at --max-image-ids; unselected rows remain pending.", flush=True)
        status_output.parent.mkdir(parents=True, exist_ok=True)
        write_status_output(input_path, status_output, store)
        print("Input rows: {}".format(input_count))
        print("Status counts: {}".format(store.summary()))
        print("Status CSV: {}".format(status_output.resolve()))
        print("Checkpoint database: {}".format(checkpoint_db.resolve()))
    finally:
        store.close()
    if args.cleanup_checkpoint:
        remove_checkpoint(checkpoint_db)


if __name__ == "__main__":
    main()
