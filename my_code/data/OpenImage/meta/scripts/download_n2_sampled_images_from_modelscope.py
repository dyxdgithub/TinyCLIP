"""Download sampled n2 images by matching ImageIDs in a ModelScope dataset.

The source dataset is scanned in streaming mode. Each record whose configured
source ImageID matches an input row is written beneath the n2 parent/leaf
directory from the class-tree JSON. A SQLite checkpoint persists completed
rows immediately; use --resume after an interrupted or failed run.
"""

import argparse
import base64
import binascii
import csv
import json
import shutil
import sqlite3
from itertools import islice
from pathlib import Path
from urllib.request import Request, urlopen

from tqdm import tqdm


META_DIR = Path(__file__).resolve().parent.parent
N2_DIR = META_DIR / "Hierarchy" / "n2"
SAMPLED_DIR = N2_DIR / "300"
DEFAULT_INPUT = SAMPLED_DIR / "Image IDs_with_last_two_layers_multi_members_n2_one_label_per_parent_sampled_300_per_class.csv"
DEFAULT_CLASS_TREE = N2_DIR / "Last_Two_Layers_Multi_Members_n2.json"
DEFAULT_OUTPUT_DIR = SAMPLED_DIR / "images_modelscope_sampled_300_per_class"
DEFAULT_STATUS_OUTPUT = SAMPLED_DIR / "Image IDs_with_last_two_layers_multi_members_n2_one_label_per_parent_sampled_300_per_class_modelscope_download_status.csv"
IMAGE_SUFFIXES = (".jpg", ".jpeg", ".png", ".webp", ".bmp")
INVALID_PATH_CHARS = '<>:"/\\|?*'
STATUS_FIELDS = [
    "ModelScopeDownloadStatus",
    "ModelScopeSourceRecordNumber",
    "ModelScopeDownloadedImagePath",
    "ModelScopeDownloadError",
]


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT, help="Sampled one-label-per-row input CSV. Default: %(default)s")
    parser.add_argument("--class-tree", type=Path, default=DEFAULT_CLASS_TREE, help="n2 class-tree JSON. Default: %(default)s")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR, help="Root directory for parent/leaf hierarchy folders. Default: %(default)s")
    parser.add_argument("--status-output", type=Path, default=DEFAULT_STATUS_OUTPUT, help="Final CSV with one ModelScope status per input row. Default: %(default)s")
    parser.add_argument("--checkpoint-db", type=Path, default=None, help="SQLite checkpoint database for resuming. Default: hidden database beside --status-output.")
    parser.add_argument("--dataset-id", default="yiqunchen/openimages", help="ModelScope dataset repository ID in owner/name form. The default requires ModelScope login and must expose the original Open Images IDs. Default: %(default)s")
    parser.add_argument("--subset-name", default="default", help="ModelScope dataset subset/configuration name. Default: %(default)s")
    parser.add_argument("--split", default="train", help="ModelScope dataset split to scan. Default: %(default)s")
    parser.add_argument("--streaming", action=argparse.BooleanOptionalAction, default=True, help="Use ModelScope streaming mode so source records are read without a complete local dataset download. Default: enabled.")
    parser.add_argument("--source-image-id-column", default="__key__", help="Original Open Images ID field in each ModelScope source record. For the default WebDataset source this is __key__. Default: %(default)s")
    parser.add_argument("--source-image-column", default="jpg", help="Image field in each ModelScope source record. For the default WebDataset source this is jpg. Default: %(default)s")
    parser.add_argument("--image-id-column", default="ImageID", help="Target ImageID column in --input. Default: %(default)s")
    parser.add_argument("--label-column", default="LabelName", help="Target leaf-label ID column in --input. Default: %(default)s")
    parser.add_argument("--parent-label-column", default="ParentLabelName", help="Target parent-label ID column in --input. Default: %(default)s")
    parser.add_argument("--label-key", default="LabelName", help="Class-label key in --class-tree. Default: %(default)s")
    parser.add_argument("--text-key", default="TextName", help="Class display-name key in --class-tree. Default: %(default)s")
    parser.add_argument("--children-key", default="Subcategory", help="Child-node key in --class-tree. Default: %(default)s")
    parser.add_argument("--image-extension", default=".jpg", choices=IMAGE_SUFFIXES, help="Output suffix used when a source image has no usable suffix. Choices: %(choices)s. Default: %(default)s")
    parser.add_argument("--source-url-timeout-seconds", type=float, default=30.0, help="Timeout for a source-image URL supplied by ModelScope, in seconds. Default: %(default)s")
    parser.add_argument("--max-source-records", type=int, default=None, help="Scan only the first N ModelScope records for a schema/connectivity test. Default: all records.")
    parser.add_argument("--progress-refresh-seconds", type=float, default=0.1, help="Minimum seconds between real-time tqdm updates. Default: %(default)s")
    parser.add_argument("--resume", action="store_true", help="Reuse rows already downloaded or found locally; re-scan for prior errors and not-found rows.")
    parser.add_argument("--overwrite", action="store_true", help="Remove prior status output and checkpoint before scanning. Existing images are retained and reported as existing.")
    parser.add_argument("--cleanup-checkpoint", action="store_true", help="Remove the checkpoint database after writing a complete status output.")
    args = parser.parse_args()
    if args.source_url_timeout_seconds <= 0:
        parser.error("--source-url-timeout-seconds must be positive")
    if args.max_source_records is not None and args.max_source_records <= 0:
        parser.error("--max-source-records must be positive")
    if args.progress_refresh_seconds <= 0:
        parser.error("--progress-refresh-seconds must be positive")
    return args


def load_runtime_dependencies():
    try:
        from modelscope.msdatasets import MsDataset
    except ImportError as error:
        raise RuntimeError(
            "This script requires ModelScope. Install requirements-training.txt before running it."
        ) from error
    return MsDataset


def safe_component(value):
    value = "".join("_" if character in INVALID_PATH_CHARS else character for character in str(value))
    value = value.strip().rstrip(".")
    return value or "unnamed"


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
    if not isinstance(root, dict):
        raise ValueError("Class-tree root must be a JSON object")
    top_level_nodes = root.get(args.children_key)
    if not isinstance(top_level_nodes, list) or not top_level_nodes:
        raise ValueError("Class-tree root key '{}' must contain a nonempty list".format(args.children_key))

    paths = {}

    def visit(node, parent_label, parent_name):
        if not isinstance(node, dict):
            raise ValueError("Class-tree nodes must be JSON objects")
        label = node.get(args.label_key)
        name = node.get(args.text_key)
        if not isinstance(label, str) or not label:
            raise ValueError("Every non-root class-tree node needs a nonempty '{}'".format(args.label_key))
        if not isinstance(name, str) or not name:
            name = label
        children = node.get(args.children_key, [])
        if children:
            if not isinstance(children, list):
                raise ValueError("Class-tree '{}' must be a list at label '{}'".format(args.children_key, label))
            for child in children:
                visit(child, label, name)
            return
        if parent_label is None:
            raise ValueError("Class-tree top-level label '{}' cannot be a leaf".format(label))
        key = (parent_label, label)
        paths.setdefault(key, (safe_component(parent_name), safe_component(name)))

    for top_level_node in top_level_nodes:
        visit(top_level_node, None, None)
    return paths


def find_existing_image(destination_dir, image_id):
    for suffix in IMAGE_SUFFIXES:
        candidate = destination_dir / (image_id + suffix)
        if candidate.is_file() and candidate.stat().st_size > 0:
            return candidate
    return None


def image_suffix(image_value, fallback):
    if isinstance(image_value, dict):
        path = image_value.get("path") or image_value.get("url") or image_value.get("src")
        if isinstance(path, str):
            suffix = Path(path.split("?", 1)[0]).suffix.lower()
            if suffix in IMAGE_SUFFIXES:
                return suffix
    if isinstance(image_value, str):
        suffix = Path(image_value.split("?", 1)[0]).suffix.lower()
        if suffix in IMAGE_SUFFIXES:
            return suffix
    return fallback


def decode_image_bytes(value):
    if isinstance(value, bytes):
        return value
    if isinstance(value, str):
        try:
            return base64.b64decode(value, validate=True)
        except (ValueError, binascii.Error):
            return None
    return None


def normalize_source_image_id(value):
    """Convert a WebDataset key or filename to a bare Open Images ImageID."""
    image_id = str(value).strip().replace("\\", "/").rsplit("/", 1)[-1]
    suffix = Path(image_id).suffix.lower()
    return image_id[: -len(suffix)] if suffix in IMAGE_SUFFIXES else image_id


def write_image_bytes(image_bytes, destination):
    temporary = destination.with_name(destination.stem + ".inprogress" + destination.suffix)
    temporary.unlink(missing_ok=True)
    try:
        temporary.write_bytes(image_bytes)
        if temporary.stat().st_size == 0:
            raise ValueError("Saved image is empty")
        temporary.replace(destination)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise


def copy_source_path(source_path, destination):
    source = Path(source_path)
    if not source.is_file():
        raise FileNotFoundError("ModelScope image path does not exist: {}".format(source))
    temporary = destination.with_name(destination.stem + ".inprogress" + destination.suffix)
    temporary.unlink(missing_ok=True)
    shutil.copyfile(source, temporary)
    temporary.replace(destination)


def download_source_url(url, destination, timeout_seconds):
    request = Request(url, headers={"User-Agent": "TinyCLIP-ModelScope-Downloader/1.0"})
    temporary = destination.with_name(destination.stem + ".inprogress" + destination.suffix)
    temporary.unlink(missing_ok=True)
    try:
        with urlopen(request, timeout=timeout_seconds) as response, temporary.open("wb") as handle:
            shutil.copyfileobj(response, handle)
        if temporary.stat().st_size == 0:
            raise ValueError("Downloaded image is empty")
        temporary.replace(destination)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise


def save_image(image_value, destination, args):
    destination.parent.mkdir(parents=True, exist_ok=True)
    raw_bytes = decode_image_bytes(image_value)
    if raw_bytes is not None:
        write_image_bytes(raw_bytes, destination)
        return
    if isinstance(image_value, dict):
        raw_bytes = decode_image_bytes(image_value.get("bytes"))
        if raw_bytes is not None:
            write_image_bytes(raw_bytes, destination)
            return
        source_path = image_value.get("path")
        if isinstance(source_path, str) and source_path:
            copy_source_path(source_path, destination)
            return
        source_url = image_value.get("url") or image_value.get("src")
        if isinstance(source_url, str) and source_url.startswith(("http://", "https://")):
            download_source_url(source_url, destination, args.source_url_timeout_seconds)
            return
    if isinstance(image_value, str):
        if image_value.startswith(("http://", "https://")):
            download_source_url(image_value, destination, args.source_url_timeout_seconds)
            return
        copy_source_path(image_value, destination)
        return
    if hasattr(image_value, "save"):
        temporary = destination.with_name(destination.stem + ".inprogress" + destination.suffix)
        temporary.unlink(missing_ok=True)
        try:
            image_to_save = image_value.convert("RGB") if destination.suffix.lower() in {".jpg", ".jpeg"} and hasattr(image_value, "convert") else image_value
            image_to_save.save(temporary)
            if temporary.stat().st_size == 0:
                raise ValueError("Saved image is empty")
            temporary.replace(destination)
        except Exception:
            temporary.unlink(missing_ok=True)
            raise
        return
    raise TypeError("Unsupported ModelScope image value type: {}".format(type(image_value).__name__))


class DownloadStore:
    def __init__(self, path):
        path.parent.mkdir(parents=True, exist_ok=True)
        self.connection = sqlite3.connect(path)
        self.connection.execute("PRAGMA journal_mode=WAL")
        self.connection.execute(
            "CREATE TABLE IF NOT EXISTS rows ("
            "input_row_number INTEGER PRIMARY KEY, image_id TEXT NOT NULL, parent_label TEXT NOT NULL, "
            "leaf_label TEXT NOT NULL, status TEXT NOT NULL, source_record_number INTEGER NOT NULL, "
            "image_path TEXT NOT NULL, error_text TEXT NOT NULL)"
        )
        self.connection.commit()

    def seed_rows(self, input_path, args):
        rows = []
        with input_path.open("r", encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle)
            for input_row_number, row in enumerate(reader, start=2):
                image_id = (row.get(args.image_id_column) or "").strip()
                parent_label = (row.get(args.parent_label_column) or "").strip()
                leaf_label = (row.get(args.label_column) or "").strip()
                if not image_id or not parent_label or not leaf_label:
                    raise ValueError("Input row {} is missing ImageID, parent label, or leaf label".format(input_row_number))
                rows.append((input_row_number, image_id, parent_label, leaf_label, "pending", 0, "", ""))
        self.connection.executemany(
            "INSERT OR IGNORE INTO rows (input_row_number, image_id, parent_label, leaf_label, status, source_record_number, image_path, error_text) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            rows,
        )
        self.connection.commit()

    def target_rows_by_image(self):
        rows = self.connection.execute(
            "SELECT input_row_number, image_id, parent_label, leaf_label FROM rows "
            "WHERE status NOT IN ('downloaded', 'existing') ORDER BY input_row_number"
        ).fetchall()
        targets = {}
        for row in rows:
            targets.setdefault(row[1], []).append(row)
        return targets

    def record_result(self, input_row_number, status, source_record_number=0, image_path="", error_text=""):
        self.connection.execute(
            "UPDATE rows SET status = ?, source_record_number = ?, image_path = ?, error_text = ? "
            "WHERE input_row_number = ?",
            (status, source_record_number, image_path, error_text, input_row_number),
        )
        self.connection.commit()

    def mark_not_found(self, pending_image_ids):
        for image_id in pending_image_ids:
            self.connection.execute(
                "UPDATE rows SET status = 'not_found', source_record_number = 0, image_path = '', "
                "error_text = 'ImageID was not found while scanning the configured ModelScope dataset' "
                "WHERE image_id = ? AND status NOT IN ('downloaded', 'existing')",
                (image_id,),
            )
        self.connection.commit()

    def status_for_row(self, input_row_number):
        return self.connection.execute(
            "SELECT status, source_record_number, image_path, error_text FROM rows WHERE input_row_number = ?",
            (input_row_number,),
        ).fetchone()

    def summary(self):
        return dict(
            self.connection.execute("SELECT status, COUNT(*) FROM rows GROUP BY status").fetchall()
        )

    def close(self):
        self.connection.close()


def write_status_output(input_path, output_path, store):
    fields = read_header(input_path)
    conflicts = set(fields).intersection(STATUS_FIELDS)
    if conflicts:
        raise ValueError("Input CSV conflicts with status fields: {}".format(", ".join(sorted(conflicts))))
    temporary = output_path.with_name(output_path.name + ".inprogress")
    temporary.unlink(missing_ok=True)
    with input_path.open("r", encoding="utf-8-sig", newline="") as source, temporary.open("w", encoding="utf-8-sig", newline="") as destination:
        reader = csv.DictReader(source)
        writer = csv.DictWriter(destination, fieldnames=fields + STATUS_FIELDS)
        writer.writeheader()
        for input_row_number, row in enumerate(reader, start=2):
            status, source_record, image_path, error_text = store.status_for_row(input_row_number)
            row.update(
                {
                    "ModelScopeDownloadStatus": status,
                    "ModelScopeSourceRecordNumber": source_record or "",
                    "ModelScopeDownloadedImagePath": image_path,
                    "ModelScopeDownloadError": error_text,
                }
            )
            writer.writerow(row)
    temporary.replace(output_path)


def remove_state(checkpoint_db, status_output):
    checkpoint_db.unlink(missing_ok=True)
    checkpoint_db.with_name(checkpoint_db.name + "-wal").unlink(missing_ok=True)
    checkpoint_db.with_name(checkpoint_db.name + "-shm").unlink(missing_ok=True)
    status_output.unlink(missing_ok=True)


def source_records(dataset, max_records):
    iterator = iter(dataset)
    return islice(iterator, max_records) if max_records is not None else iterator


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
    fields = read_header(input_path)
    for field in (args.image_id_column, args.label_column, args.parent_label_column):
        if field not in fields:
            raise ValueError("Input CSV is missing required column '{}': {}".format(field, input_path))
    if args.overwrite:
        remove_state(checkpoint_db, status_output)
    elif (checkpoint_db.exists() or status_output.exists()) and not args.resume:
        raise FileExistsError("Output or checkpoint already exists. Use --resume or --overwrite. Checkpoint: {}".format(checkpoint_db))

    leaf_paths = load_leaf_paths(class_tree_path, args)
    input_count = count_rows(input_path, args.progress_refresh_seconds)
    store = DownloadStore(checkpoint_db)
    try:
        store.seed_rows(input_path, args)
        targets = store.target_rows_by_image()
        if not targets:
            status_output.parent.mkdir(parents=True, exist_ok=True)
            write_status_output(input_path, status_output, store)
            print("All input rows are already downloaded or present locally. Status CSV: {}".format(status_output.resolve()))
            return

        MsDataset = load_runtime_dependencies()
        load_kwargs = {
            "subset_name": args.subset_name,
            "split": args.split,
            "use_streaming": args.streaming,
        }
        dataset = MsDataset.load(args.dataset_id, **load_kwargs)
        discovered_source_id_column = False
        found_images = set()
        counts = {"matched": 0, "downloaded": 0, "existing": 0, "error": 0}
        with tqdm(desc="Scanning ModelScope records", total=None, unit="record", mininterval=args.progress_refresh_seconds) as progress:
            for source_record_number, record in enumerate(source_records(dataset, args.max_source_records), start=1):
                progress.update(1)
                if not isinstance(record, dict):
                    raise TypeError("ModelScope record {} is {}, expected a mapping".format(source_record_number, type(record).__name__))
                if args.source_image_id_column not in record:
                    available = ", ".join(sorted(map(str, record.keys())))
                    raise ValueError(
                        "ModelScope record {} has no source ImageID column '{}'. Available fields: {}. "
                        "Choose a ModelScope dataset that preserves original Open Images IDs, or pass "
                        "--source-image-id-column with its actual field name.".format(
                            source_record_number, args.source_image_id_column, available
                        )
                    )
                discovered_source_id_column = True
                source_image_id = normalize_source_image_id(record[args.source_image_id_column])
                matching_rows = targets.get(source_image_id)
                if not matching_rows:
                    continue
                if args.source_image_column not in record:
                    available = ", ".join(sorted(map(str, record.keys())))
                    raise ValueError("Matched ModelScope record {} has no image column '{}'. Available fields: {}".format(source_record_number, args.source_image_column, available))
                found_images.add(source_image_id)
                image_value = record[args.source_image_column]
                for input_row_number, image_id, parent_label, leaf_label in matching_rows:
                    hierarchy_path = leaf_paths.get((parent_label, leaf_label))
                    if hierarchy_path is None:
                        store.record_result(
                            input_row_number,
                            "error",
                            source_record_number,
                            error_text="Input parent/leaf pair ({!r}, {!r}) is absent from the class tree".format(
                                parent_label, leaf_label
                            ),
                        )
                        counts["error"] += 1
                        continue
                    parent_name, leaf_name = hierarchy_path
                    destination_dir = output_dir / parent_name / leaf_name
                    existing = find_existing_image(destination_dir, image_id)
                    if existing:
                        store.record_result(input_row_number, "existing", source_record_number, str(existing))
                        counts["existing"] += 1
                        continue
                    suffix = image_suffix(image_value, args.image_extension)
                    destination = destination_dir / (image_id + suffix)
                    try:
                        save_image(image_value, destination, args)
                        store.record_result(input_row_number, "downloaded", source_record_number, str(destination))
                        counts["downloaded"] += 1
                    except Exception as error:
                        store.record_result(input_row_number, "error", source_record_number, error_text=str(error))
                        counts["error"] += 1
                    counts["matched"] += 1
                progress.set_postfix(**counts, remaining_image_ids=max(0, len(targets) - len(found_images)))
        if not discovered_source_id_column:
            raise ValueError("The configured ModelScope dataset yielded no records; verify --dataset-id, --subset-name, and --split")
        if args.max_source_records is None:
            store.mark_not_found(set(targets).difference(found_images))
        else:
            print(
                "Source scan stopped at --max-source-records; unmatched target rows remain pending "
                "rather than being marked not_found.",
                flush=True,
            )
        status_output.parent.mkdir(parents=True, exist_ok=True)
        write_status_output(input_path, status_output, store)
        print("Input rows: {}".format(input_count))
        print("Status counts: {}".format(store.summary()))
        print("Status CSV: {}".format(status_output.resolve()))
        print("Checkpoint database: {}".format(checkpoint_db.resolve()))
    finally:
        store.close()
    if args.cleanup_checkpoint:
        checkpoint_db.unlink(missing_ok=True)
        checkpoint_db.with_name(checkpoint_db.name + "-wal").unlink(missing_ok=True)
        checkpoint_db.with_name(checkpoint_db.name + "-shm").unlink(missing_ok=True)


if __name__ == "__main__":
    main()
