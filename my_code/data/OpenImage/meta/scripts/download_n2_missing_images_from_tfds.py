"""Supplement FiftyOne downloads from the TFDS Open Images V7 builder.

Only ImageIDs whose FiftyOne status is ``not_found`` or ``error`` are scanned.
Matches are copied into the existing n2 parent/leaf hierarchy without replacing
FiftyOne-downloaded images. The script keeps an SQLite status checkpoint so a
long TFDS scan can be resumed.

TFDS iterates a split rather than exposing a remote ImageID lookup API. The
default uses ``tfds.load("open_images/v7", split="train")``. The script
requires ``--download-tfds-data`` before it lets TFDS prepare missing data.
"""

import argparse
import csv
import json
import os
import shutil
import sqlite3
from pathlib import Path

from PIL import Image
from tqdm import tqdm


META_DIR = Path(__file__).resolve().parent.parent
N2_DIR = META_DIR / "Hierarchy" / "n2"
DEFAULT_INPUT = N2_DIR / "Image_IDs_n2_all_fiftyone_download_status.csv"
DEFAULT_CLASS_TREE = N2_DIR / "Last_Two_Layers_Multi_Members_n2.json"
DEFAULT_OUTPUT_DIR = N2_DIR / "images_fiftyone_all"
DEFAULT_STATUS_OUTPUT = N2_DIR / "Image_IDs_n2_all_fiftyone_tfds_download_status.csv"
DEFAULT_TFDS_DATA_DIR = N2_DIR / "tfds_open_images_v7"
INVALID_PATH_CHARS = '<>:"/\\|?*'
STATUS_FIELDS = [
    "TFDSDownloadStatus",
    "TFDSSourceFilename",
    "TFDSDownloadedImagePath",
    "TFDSDownloadError",
]


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT,
                        help="FiftyOne status CSV. Only rows whose FiftyOne status is eligible are considered. Default: %(default)s")
    parser.add_argument("--class-tree", type=Path, default=DEFAULT_CLASS_TREE,
                        help="n2 parent/leaf class-tree JSON. Default: %(default)s")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR,
                        help="Existing parent/leaf image hierarchy. Found TFDS images are copied here without overwriting existing files. Default: %(default)s")
    parser.add_argument("--status-output", type=Path, default=DEFAULT_STATUS_OUTPUT,
                        help="Combined input plus TFDS status fields. Default: %(default)s")
    parser.add_argument("--checkpoint-db", type=Path, default=None,
                        help="SQLite resume checkpoint. Default: hidden database beside --status-output.")
    parser.add_argument("--tfds-name", default="open_images/v7",
                        help="TFDS builder/configuration. Default: %(default)s")
    parser.add_argument("--split", default="train",
                        help="TFDS split to scan. Default: %(default)s")
    parser.add_argument("--tfds-data-dir", type=Path, default=DEFAULT_TFDS_DATA_DIR,
                        help="TFDS cache/prepared-dataset directory. Default: %(default)s")
    parser.add_argument("--download-tfds-data", action="store_true",
                        help="Allow TFDS to download and prepare a missing dataset. The Open Images train split is large. Default: disabled.")
    parser.add_argument("--tfds-image-id-key", default="auto",
                        help="Feature key containing the Open Images ID. auto tries image_id, image/filename, image_filename, and id; a filename value is converted to its stem. Default: %(default)s")
    parser.add_argument("--eligible-fiftyone-status", action="append", default=None,
                        help="FiftyOne status eligible for TFDS recovery. Repeatable. Default: not_found and error.")
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
                        help="Scan for only the first N pending unique IDs as a connectivity test. Unselected rows remain pending. Default: all pending IDs.")
    parser.add_argument("--progress-refresh-seconds", type=float, default=0.1,
                        help="Minimum tqdm refresh interval. Default: %(default)s")
    parser.add_argument("--resume", action="store_true",
                        help="Reuse the SQLite checkpoint and retry prior TFDS not_found/error rows. Default: disabled.")
    parser.add_argument("--overwrite", action="store_true",
                        help="Discard only this script's checkpoint and status output. Existing images and the TFDS cache are retained. Default: disabled.")
    parser.add_argument("--cleanup-checkpoint", action="store_true",
                        help="Delete this script's SQLite checkpoint after writing the status CSV. Default: disabled.")
    args = parser.parse_args()
    if args.max_image_ids is not None and args.max_image_ids <= 0:
        parser.error("--max-image-ids must be positive")
    if args.progress_refresh_seconds <= 0:
        parser.error("--progress-refresh-seconds must be positive")
    if args.resume and args.overwrite:
        parser.error("Use either --resume or --overwrite, not both")
    args.eligible_fiftyone_status = tuple(args.eligible_fiftyone_status or ("not_found", "error"))
    return args


def load_tfds_runtime():
    try:
        import tensorflow as tf  # noqa: F401
        import tensorflow_datasets as tfds
    except ImportError as error:
        raise RuntimeError(
            "This script requires TensorFlow Datasets. Install a TensorFlow build compatible with the server, "
            "then install tensorflow-datasets."
        ) from error
    return tfds


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
    for suffix in (".jpg", ".jpeg", ".png", ".webp", ".bmp"):
        candidate = destination_dir / (image_id + suffix)
        if candidate.is_file() and candidate.stat().st_size > 0:
            return candidate
    return None


def save_image(image_array, destination):
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(destination.stem + ".inprogress" + destination.suffix)
    temporary.unlink(missing_ok=True)
    try:
        image = Image.fromarray(image_array)
        image.save(temporary, format="JPEG", quality=95)
        temporary.replace(destination)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise


def decode_text(value):
    if isinstance(value, bytes):
        return value.decode("utf-8")
    if hasattr(value, "item"):
        value = value.item()
        if isinstance(value, bytes):
            return value.decode("utf-8")
    return str(value)


def image_id_from_tfds_example(example, image_id_key):
    if image_id_key == "auto":
        candidates = ("image_id", "image/filename", "image_filename", "id")
        key = next((candidate for candidate in candidates if candidate in example), None)
        if key is None:
            raise KeyError(
                "TFDS example does not expose a recognized ImageID feature. Available keys: {}. "
                "Pass --tfds-image-id-key with the correct key.".format(
                    ", ".join(sorted(example.keys()))
                )
            )
    else:
        key = image_id_key
        if key not in example:
            raise KeyError(
                "TFDS example has no --tfds-image-id-key {!r}. Available keys: {}".format(
                    key, ", ".join(sorted(example.keys()))
                )
            )
    raw_value = decode_text(example[key])
    return Path(raw_value).stem, raw_value


class DownloadStore:
    def __init__(self, path):
        path.parent.mkdir(parents=True, exist_ok=True)
        self.connection = sqlite3.connect(path)
        self.connection.execute("PRAGMA journal_mode=WAL")
        self.connection.execute(
            "CREATE TABLE IF NOT EXISTS rows "
            "(input_row_number INTEGER PRIMARY KEY, image_id TEXT NOT NULL, parent_label TEXT NOT NULL, "
            "leaf_label TEXT NOT NULL, fiftyone_status TEXT NOT NULL, tfds_status TEXT NOT NULL, "
            "source_filename TEXT NOT NULL, image_path TEXT NOT NULL, error_text TEXT NOT NULL)"
        )
        self.connection.commit()

    def seed_rows(self, input_path, args):
        rows = []
        eligible = set(args.eligible_fiftyone_status)
        with input_path.open("r", encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle)
            for number, row in enumerate(reader, start=2):
                image_id = (row.get(args.image_id_column) or "").strip()
                parent = (row.get(args.parent_label_column) or "").strip()
                leaf = (row.get(args.label_column) or "").strip()
                fiftyone_status = (row.get(args.fiftyone_status_column) or "").strip()
                if not image_id or not parent or not leaf:
                    raise ValueError("Input row {} lacks ImageID, parent label, or leaf label".format(number))
                initial_status = "pending" if fiftyone_status in eligible else "skipped_fiftyone_success"
                rows.append((number, image_id, parent, leaf, fiftyone_status, initial_status, "", "", ""))
        self.connection.executemany(
            "INSERT OR IGNORE INTO rows "
            "(input_row_number, image_id, parent_label, leaf_label, fiftyone_status, tfds_status, source_filename, image_path, error_text) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
            rows,
        )
        self.connection.commit()

    def pending_by_image_id(self):
        rows = self.connection.execute(
            "SELECT input_row_number, image_id, parent_label, leaf_label FROM rows "
            "WHERE tfds_status IN ('pending', 'error', 'not_found') ORDER BY input_row_number"
        ).fetchall()
        targets = {}
        for row in rows:
            targets.setdefault(row[1], []).append(row)
        return targets

    def record_result(self, row_number, status, source_filename="", image_path="", error_text=""):
        self.connection.execute(
            "UPDATE rows SET tfds_status = ?, source_filename = ?, image_path = ?, error_text = ? "
            "WHERE input_row_number = ?",
            (status, source_filename, image_path, error_text, row_number),
        )
        self.connection.commit()

    def mark_not_found(self, image_ids):
        self.connection.executemany(
            "UPDATE rows SET tfds_status = 'not_found', source_filename = '', image_path = '', "
            "error_text = 'ImageID was not found in the scanned TFDS split' "
            "WHERE image_id = ? AND tfds_status IN ('pending', 'error', 'not_found')",
            ((image_id,) for image_id in image_ids),
        )
        self.connection.commit()

    def status_for_row(self, row_number):
        return self.connection.execute(
            "SELECT tfds_status, source_filename, image_path, error_text FROM rows WHERE input_row_number = ?",
            (row_number,),
        ).fetchone()

    def summary(self):
        return dict(self.connection.execute("SELECT tfds_status, COUNT(*) FROM rows GROUP BY tfds_status").fetchall())

    def close(self):
        self.connection.close()


def write_status_output(input_path, output_path, store):
    fields = read_header(input_path)
    conflicts = set(fields).intersection(STATUS_FIELDS)
    if conflicts:
        raise ValueError("Input CSV conflicts with TFDS status columns: {}".format(", ".join(sorted(conflicts))))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_name(output_path.name + ".inprogress")
    temporary.unlink(missing_ok=True)
    with input_path.open("r", encoding="utf-8-sig", newline="") as source, temporary.open("w", encoding="utf-8-sig", newline="") as destination:
        reader = csv.DictReader(source)
        writer = csv.DictWriter(destination, fieldnames=fields + STATUS_FIELDS)
        writer.writeheader()
        for number, row in enumerate(reader, start=2):
            status, filename, image_path, error_text = store.status_for_row(number)
            row.update({
                "TFDSDownloadStatus": status,
                "TFDSSourceFilename": filename,
                "TFDSDownloadedImagePath": image_path,
                "TFDSDownloadError": error_text,
            })
            writer.writerow(row)
    temporary.replace(output_path)


def remove_checkpoint(path):
    path.unlink(missing_ok=True)
    path.with_name(path.name + "-wal").unlink(missing_ok=True)
    path.with_name(path.name + "-shm").unlink(missing_ok=True)


def load_tfds_split(tfds, args):
    try:
        dataset, info = tfds.load(
            args.tfds_name,
            split=args.split,
            data_dir=str(args.tfds_data_dir),
            download=args.download_tfds_data,
            shuffle_files=False,
            with_info=True,
        )
    except Exception as error:
        if not args.download_tfds_data:
            raise RuntimeError(
                "Could not open prepared TFDS data at {}. Run again with --download-tfds-data only after "
                "confirming sufficient disk space for {}. Original error: {}".format(
                    args.tfds_data_dir, args.tfds_name, error
                )
            ) from error
        raise
    if args.split not in info.splits:
        raise ValueError("TFDS dataset {!r} has no split {!r}".format(args.tfds_name, args.split))
    return dataset, int(info.splits[args.split].num_examples)


def main():
    args = parse_args()
    input_path = args.input.expanduser()
    class_tree_path = args.class_tree.expanduser()
    output_dir = args.output_dir.expanduser()
    status_output = args.status_output.expanduser()
    checkpoint_db = args.checkpoint_db.expanduser() if args.checkpoint_db else status_output.with_name("." + status_output.stem + "_checkpoint.sqlite3")
    args.tfds_data_dir = args.tfds_data_dir.expanduser()
    if not input_path.is_file():
        raise FileNotFoundError("FiftyOne status CSV does not exist: {}".format(input_path))
    if not class_tree_path.is_file():
        raise FileNotFoundError("Class-tree JSON does not exist: {}".format(class_tree_path))
    header = read_header(input_path)
    required = {args.image_id_column, args.parent_label_column, args.label_column, args.fiftyone_status_column}
    missing = required.difference(header)
    if missing:
        raise ValueError("Input CSV is missing required columns: {}".format(", ".join(sorted(missing))))
    if args.overwrite:
        remove_checkpoint(checkpoint_db)
        status_output.unlink(missing_ok=True)
    elif (checkpoint_db.exists() or status_output.exists()) and not args.resume:
        raise FileExistsError("Output or checkpoint exists. Use --resume or --overwrite. Checkpoint: {}".format(checkpoint_db))

    hierarchy_paths = load_leaf_paths(class_tree_path, args)
    input_count = count_rows(input_path, args.progress_refresh_seconds)
    store = DownloadStore(checkpoint_db)
    try:
        store.seed_rows(input_path, args)
        targets = store.pending_by_image_id()
        if not targets:
            write_status_output(input_path, status_output, store)
            print("No rows need TFDS recovery. Status CSV: {}".format(status_output.resolve()))
            return
        requested_ids = sorted(targets)
        if args.max_image_ids is not None:
            requested_ids = requested_ids[:args.max_image_ids]
        requested_set = set(requested_ids)
        print("Scanning TFDS for {} unique ImageIDs...".format(len(requested_ids)), flush=True)
        tfds = load_tfds_runtime()
        dataset, total_examples = load_tfds_split(tfds, args)
        found_ids = set()
        counts = {"downloaded": 0, "existing": 0, "error": 0}
        with tqdm(total=total_examples, desc="Scanning TFDS {}".format(args.split), unit="example", mininterval=args.progress_refresh_seconds) as progress:
            for example in tfds.as_numpy(dataset):
                image_id, filename = image_id_from_tfds_example(example, args.tfds_image_id_key)
                rows = targets.get(image_id)
                if rows is not None and image_id in requested_set:
                    found_ids.add(image_id)
                    image_array = example["image"]
                    for row_number, _, parent_label, leaf_label in rows:
                        hierarchy_path = hierarchy_paths.get((parent_label, leaf_label))
                        if hierarchy_path is None:
                            store.record_result(
                                row_number, "error", filename,
                                error_text="Input parent/leaf pair ({!r}, {!r}) is absent from the class tree".format(parent_label, leaf_label),
                            )
                            counts["error"] += 1
                            continue
                        parent_name, leaf_name = hierarchy_path
                        destination_dir = output_dir / parent_name / leaf_name
                        existing = find_existing_image(destination_dir, image_id)
                        if existing is not None:
                            store.record_result(row_number, "existing", filename, str(existing))
                            counts["existing"] += 1
                            continue
                        destination = destination_dir / (image_id + ".jpg")
                        try:
                            save_image(image_array, destination)
                            store.record_result(row_number, "downloaded", filename, str(destination))
                            counts["downloaded"] += 1
                        except Exception as error:
                            store.record_result(row_number, "error", filename, error_text=str(error))
                            counts["error"] += 1
                progress.update(1)
                progress.set_postfix(found=len(found_ids), **counts)
                if found_ids == requested_set:
                    break
        if args.max_image_ids is None:
            store.mark_not_found(requested_set.difference(found_ids))
        else:
            print("TFDS scan stopped at --max-image-ids; unselected rows remain pending.", flush=True)
        write_status_output(input_path, status_output, store)
        print("Input rows: {}".format(input_count))
        print("TFDS status counts: {}".format(store.summary()))
        print("Status CSV: {}".format(status_output.resolve()))
        print("Checkpoint database: {}".format(checkpoint_db.resolve()))
    finally:
        store.close()
    if args.cleanup_checkpoint:
        remove_checkpoint(checkpoint_db)


if __name__ == "__main__":
    main()
