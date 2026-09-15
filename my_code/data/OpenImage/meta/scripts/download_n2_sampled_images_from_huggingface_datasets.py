"""Download sampled n2 images with Hugging Face datasets.load_dataset.

The script streams Fhrozen/openimages-narratives-v2, matches its ImageId field
to the sampled n2 manifest's ImageID column, and saves matched RGB images in
the n2 parent/leaf directory structure. A SQLite checkpoint enables --resume.
"""

import argparse
import base64
import binascii
import csv
import json
import os
import shutil
import sqlite3
from itertools import islice
from pathlib import Path

from PIL import Image
from tqdm import tqdm


META_DIR = Path(__file__).resolve().parent.parent
N2_DIR = META_DIR / "Hierarchy" / "n2"
SAMPLED_DIR = N2_DIR / "300"
DEFAULT_INPUT = SAMPLED_DIR / "Image IDs_with_last_two_layers_multi_members_n2_one_label_per_parent_sampled_300_per_class.csv"
DEFAULT_CLASS_TREE = N2_DIR / "Last_Two_Layers_Multi_Members_n2.json"
DEFAULT_OUTPUT_DIR = SAMPLED_DIR / "images_huggingface_datasets_v2_sampled_300_per_class"
DEFAULT_STATUS_OUTPUT = SAMPLED_DIR / "Image IDs_with_last_two_layers_multi_members_n2_one_label_per_parent_sampled_300_per_class_huggingface_datasets_v2_download_status.csv"
INVALID_PATH_CHARS = '<>:"/\\|?*'
IMAGE_SUFFIXES = (".jpg", ".jpeg", ".png", ".webp", ".bmp")
STATUS_FIELDS = [
    "HuggingFaceDatasetsDownloadStatus",
    "HuggingFaceDatasetsSourceRecordNumber",
    "HuggingFaceDatasetsDownloadedImagePath",
    "HuggingFaceDatasetsDownloadError",
]


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT, help="Sampled one-label-per-parent input CSV. Default: %(default)s")
    parser.add_argument("--class-tree", type=Path, default=DEFAULT_CLASS_TREE, help="n2 class-tree JSON. Default: %(default)s")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR, help="Root output directory for parent/leaf image folders. Default: %(default)s")
    parser.add_argument("--status-output", type=Path, default=DEFAULT_STATUS_OUTPUT, help="Output status CSV. Default: %(default)s")
    parser.add_argument("--checkpoint-db", type=Path, default=None, help="SQLite resume checkpoint. Default: hidden database beside --status-output.")
    parser.add_argument("--dataset-id", default="Fhrozen/openimages-narratives-v2", help="Hugging Face dataset ID passed to datasets.load_dataset. Default: %(default)s")
    parser.add_argument("--config-name", default=None, help="Optional Hugging Face dataset configuration name. Default: none.")
    parser.add_argument("--split", default="train", help="Dataset split passed to datasets.load_dataset. Default: %(default)s")
    parser.add_argument("--revision", default=None, help="Optional Hugging Face dataset revision. Default: latest revision.")
    parser.add_argument("--streaming", action=argparse.BooleanOptionalAction, default=True, help="Use streaming=True with datasets.load_dataset to avoid a complete source-dataset download. Default: enabled.")
    parser.add_argument("--hf-token-env", default="HF_TOKEN", help="Environment variable containing an optional Hugging Face token. Default: %(default)s")
    parser.add_argument("--source-image-id-column", default="ImageId", help="Image ID field in the Hugging Face dataset record. Default: %(default)s")
    parser.add_argument("--source-image-column", default="image", help="Image field in the Hugging Face dataset record. Default: %(default)s")
    parser.add_argument("--image-id-column", default="ImageID", help="Target ImageID column in --input. Default: %(default)s")
    parser.add_argument("--label-column", default="LabelName", help="Target leaf-label column in --input. Default: %(default)s")
    parser.add_argument("--parent-label-column", default="ParentLabelName", help="Target parent-label column in --input. Default: %(default)s")
    parser.add_argument("--label-key", default="LabelName", help="Class-label key in --class-tree. Default: %(default)s")
    parser.add_argument("--text-key", default="TextName", help="Class display-name key in --class-tree. Default: %(default)s")
    parser.add_argument("--children-key", default="Subcategory", help="Child-node key in --class-tree. Default: %(default)s")
    parser.add_argument("--image-extension", choices=IMAGE_SUFFIXES, default=".jpg", help="Output image extension. Choices: %(choices)s. Default: %(default)s")
    parser.add_argument("--max-source-records", type=int, default=None, help="Scan only the first N dataset records for connectivity/schema testing. Default: all records.")
    parser.add_argument("--progress-refresh-seconds", type=float, default=0.1, help="Minimum seconds between tqdm progress refreshes. Default: %(default)s")
    parser.add_argument("--resume", action="store_true", help="Reuse completed rows and retry error/not-found rows from a previous run.")
    parser.add_argument("--overwrite", action="store_true", help="Delete prior status and checkpoint state before running. Existing image files are retained and reported as existing.")
    parser.add_argument("--cleanup-checkpoint", action="store_true", help="Delete the checkpoint after the status CSV is written.")
    args = parser.parse_args()
    if args.max_source_records is not None and args.max_source_records <= 0:
        parser.error("--max-source-records must be positive")
    if args.progress_refresh_seconds <= 0:
        parser.error("--progress-refresh-seconds must be positive")
    return args


def load_dataset_runtime():
    try:
        from datasets import load_dataset
    except ImportError as error:
        raise RuntimeError("This script requires the 'datasets' package. Install requirements-training.txt first.") from error
    return load_dataset


def safe_component(value):
    value = "".join("_" if character in INVALID_PATH_CHARS else character for character in str(value))
    return value.strip().rstrip(".") or "unnamed"


def read_header(path):
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return csv.DictReader(handle).fieldnames or []


def count_rows(path, refresh_seconds):
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.reader(handle)
        next(reader, None)
        count = 0
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


def write_bytes_atomically(image_bytes, destination):
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


def save_source_image(image_value, destination):
    destination.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(image_value, Image.Image):
        temporary = destination.with_name(destination.stem + ".inprogress" + destination.suffix)
        temporary.unlink(missing_ok=True)
        try:
            image_value.convert("RGB").save(temporary)
            if temporary.stat().st_size == 0:
                raise ValueError("Saved image is empty")
            temporary.replace(destination)
        except Exception:
            temporary.unlink(missing_ok=True)
            raise
        return
    if isinstance(image_value, dict):
        raw_bytes = image_value.get("bytes")
        if isinstance(raw_bytes, str):
            try:
                raw_bytes = base64.b64decode(raw_bytes, validate=True)
            except (ValueError, binascii.Error):
                raw_bytes = None
        if isinstance(raw_bytes, bytes):
            write_bytes_atomically(raw_bytes, destination)
            return
        source_path = image_value.get("path")
        if isinstance(source_path, str) and Path(source_path).is_file():
            temporary = destination.with_name(destination.stem + ".inprogress" + destination.suffix)
            temporary.unlink(missing_ok=True)
            try:
                shutil.copyfile(source_path, temporary)
                temporary.replace(destination)
            except Exception:
                temporary.unlink(missing_ok=True)
                raise
            return
    raise TypeError("Unsupported Hugging Face image value type: {}".format(type(image_value).__name__))


class DownloadStore:
    def __init__(self, path):
        path.parent.mkdir(parents=True, exist_ok=True)
        self.connection = sqlite3.connect(path)
        self.connection.execute("PRAGMA journal_mode=WAL")
        self.connection.execute("CREATE TABLE IF NOT EXISTS rows (input_row_number INTEGER PRIMARY KEY, image_id TEXT NOT NULL, parent_label TEXT NOT NULL, leaf_label TEXT NOT NULL, status TEXT NOT NULL, source_record_number INTEGER NOT NULL, image_path TEXT NOT NULL, error_text TEXT NOT NULL)")
        self.connection.commit()

    def seed_rows(self, input_path, args):
        values = []
        with input_path.open("r", encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle)
            for number, row in enumerate(reader, start=2):
                image_id = (row.get(args.image_id_column) or "").strip()
                parent = (row.get(args.parent_label_column) or "").strip()
                leaf = (row.get(args.label_column) or "").strip()
                if not image_id or not parent or not leaf:
                    raise ValueError("Input row {} lacks ImageID, parent label, or leaf label".format(number))
                values.append((number, image_id, parent, leaf, "pending", 0, "", ""))
        self.connection.executemany("INSERT OR IGNORE INTO rows (input_row_number, image_id, parent_label, leaf_label, status, source_record_number, image_path, error_text) VALUES (?, ?, ?, ?, ?, ?, ?, ?)", values)
        self.connection.commit()

    def pending_by_image_id(self):
        rows = self.connection.execute("SELECT input_row_number, image_id, parent_label, leaf_label FROM rows WHERE status NOT IN ('downloaded', 'existing') ORDER BY input_row_number").fetchall()
        targets = {}
        for row in rows:
            targets.setdefault(row[1], []).append(row)
        return targets

    def record_result(self, row_number, status, source_record_number=0, image_path="", error_text=""):
        self.connection.execute("UPDATE rows SET status = ?, source_record_number = ?, image_path = ?, error_text = ? WHERE input_row_number = ?", (status, source_record_number, image_path, error_text, row_number))
        self.connection.commit()

    def mark_not_found(self, image_ids):
        self.connection.executemany("UPDATE rows SET status = 'not_found', source_record_number = 0, image_path = '', error_text = 'ImageID was not found while scanning the Hugging Face dataset' WHERE image_id = ? AND status NOT IN ('downloaded', 'existing')", ((image_id,) for image_id in image_ids))
        self.connection.commit()

    def status_for_row(self, row_number):
        return self.connection.execute("SELECT status, source_record_number, image_path, error_text FROM rows WHERE input_row_number = ?", (row_number,)).fetchone()

    def summary(self):
        return dict(self.connection.execute("SELECT status, COUNT(*) FROM rows GROUP BY status").fetchall())

    def close(self):
        self.connection.close()


def write_status_output(input_path, output_path, store):
    fields = read_header(input_path)
    conflicting = set(fields).intersection(STATUS_FIELDS)
    if conflicting:
        raise ValueError("Input CSV conflicts with output status columns: {}".format(", ".join(sorted(conflicting))))
    temporary = output_path.with_name(output_path.name + ".inprogress")
    temporary.unlink(missing_ok=True)
    with input_path.open("r", encoding="utf-8-sig", newline="") as source, temporary.open("w", encoding="utf-8-sig", newline="") as destination:
        reader = csv.DictReader(source)
        writer = csv.DictWriter(destination, fieldnames=fields + STATUS_FIELDS)
        writer.writeheader()
        for number, row in enumerate(reader, start=2):
            status, source_record, image_path, error_text = store.status_for_row(number)
            row.update({"HuggingFaceDatasetsDownloadStatus": status, "HuggingFaceDatasetsSourceRecordNumber": source_record or "", "HuggingFaceDatasetsDownloadedImagePath": image_path, "HuggingFaceDatasetsDownloadError": error_text})
            writer.writerow(row)
    temporary.replace(output_path)


def remove_state(checkpoint_db, status_output):
    remove_checkpoint(checkpoint_db)
    status_output.unlink(missing_ok=True)


def remove_checkpoint(checkpoint_db):
    checkpoint_db.unlink(missing_ok=True)
    checkpoint_db.with_name(checkpoint_db.name + "-wal").unlink(missing_ok=True)
    checkpoint_db.with_name(checkpoint_db.name + "-shm").unlink(missing_ok=True)


def load_source_dataset(args):
    load_dataset = load_dataset_runtime()
    kwargs = {"split": args.split, "streaming": args.streaming}
    if args.config_name:
        kwargs["name"] = args.config_name
    if args.revision:
        kwargs["revision"] = args.revision
    token = os.getenv(args.hf_token_env)
    if token:
        kwargs["token"] = token
    return load_dataset(args.dataset_id, **kwargs)


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
    for field in (args.image_id_column, args.parent_label_column, args.label_column):
        if field not in read_header(input_path):
            raise ValueError("Input CSV is missing required column '{}': {}".format(field, input_path))
    if args.overwrite:
        remove_state(checkpoint_db, status_output)
    elif (checkpoint_db.exists() or status_output.exists()) and not args.resume:
        raise FileExistsError("Output or checkpoint exists. Use --resume or --overwrite. Checkpoint: {}".format(checkpoint_db))

    paths = load_leaf_paths(class_tree_path, args)
    input_count = count_rows(input_path, args.progress_refresh_seconds)
    store = DownloadStore(checkpoint_db)
    try:
        store.seed_rows(input_path, args)
        targets = store.pending_by_image_id()
        if not targets:
            status_output.parent.mkdir(parents=True, exist_ok=True)
            write_status_output(input_path, status_output, store)
            print("All input rows are already completed. Status CSV: {}".format(status_output.resolve()))
            return
        source = load_source_dataset(args)
        counts = {"matched": 0, "downloaded": 0, "existing": 0, "error": 0}
        source_fields_checked = False
        records = islice(iter(source), args.max_source_records) if args.max_source_records else iter(source)
        with tqdm(desc="Scanning Hugging Face dataset (total unknown)", unit="record", mininterval=args.progress_refresh_seconds) as progress:
            for source_number, record in enumerate(records, start=1):
                progress.update(1)
                if not isinstance(record, dict):
                    raise TypeError("Dataset record {} is {}, expected a mapping".format(source_number, type(record).__name__))
                if not source_fields_checked:
                    missing = [field for field in (args.source_image_id_column, args.source_image_column) if field not in record]
                    if missing:
                        raise ValueError("Dataset record 1 lacks required field(s) {}. Available fields: {}".format(", ".join(missing), ", ".join(sorted(map(str, record.keys())))))
                    source_fields_checked = True
                image_id = str(record[args.source_image_id_column]).strip()
                matching_rows = targets.pop(image_id, None)
                if matching_rows is None:
                    continue
                image_value = record[args.source_image_column]
                for row_number, _, parent_label, leaf_label in matching_rows:
                    hierarchy_path = paths.get((parent_label, leaf_label))
                    if hierarchy_path is None:
                        store.record_result(row_number, "error", source_number, error_text="Input parent/leaf pair ({!r}, {!r}) is absent from the class tree".format(parent_label, leaf_label))
                        counts["error"] += 1
                        continue
                    parent_name, leaf_name = hierarchy_path
                    destination_dir = output_dir / parent_name / leaf_name
                    existing = find_existing_image(destination_dir, image_id)
                    if existing:
                        store.record_result(row_number, "existing", source_number, str(existing))
                        counts["existing"] += 1
                        continue
                    destination = destination_dir / (image_id + args.image_extension)
                    try:
                        save_source_image(image_value, destination)
                        store.record_result(row_number, "downloaded", source_number, str(destination))
                        counts["downloaded"] += 1
                    except Exception as error:
                        store.record_result(row_number, "error", source_number, error_text=str(error))
                        counts["error"] += 1
                    counts["matched"] += 1
                progress.set_postfix(**counts, remaining_image_ids=len(targets))
        if not source_fields_checked:
            raise ValueError("The configured dataset yielded no records; verify --dataset-id, --config-name, and --split")
        if args.max_source_records is None:
            store.mark_not_found(targets)
        else:
            print("Source scan stopped at --max-source-records; unmatched target rows remain pending.", flush=True)
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
