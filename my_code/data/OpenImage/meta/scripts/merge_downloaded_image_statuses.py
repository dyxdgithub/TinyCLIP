"""Merge successfully obtained images from direct-download and sync CSV files.

One output row represents one relative image path in images_fiftyone_all.
Different class paths for the same ImageID remain separate samples.
"""

import argparse
import csv
import json
import sqlite3
from pathlib import Path

from tqdm import tqdm


META_DIR = Path(__file__).resolve().parent.parent
N2_DIR = META_DIR / "Hierarchy" / "n2"
DEFAULT_DIRECT = N2_DIR / "Image_IDs_n2_all_direct_download_status.csv"
DEFAULT_SYNC = N2_DIR / "sync_images_fiftyone_all_status.csv"
DEFAULT_OUTPUT = N2_DIR / "Image_IDs_n2_all_downloaded_merged.csv"
DIRECT_SUCCESS = {"downloaded_s3", "downloaded_url", "existing"}
FIFTYONE_SUCCESS = {"downloaded", "existing"}
SYNC_SUCCESS = {"copied", "overwritten", "existing"}
FIELDS = (
    "ImageID", "RelativePath", "ParentLabelName", "ParentDisplayName",
    "LabelName", "DisplayName", "FiftyOneDownloadStatus",
    "DirectDownloadStatus", "DirectDownloadedImagePath",
    "FiftyOneDownloadedImagePath", "SyncStatus", "SyncSourcePath",
    "SyncDestinationPath", "SourceTables",
)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--direct-input", type=Path, default=DEFAULT_DIRECT,
                        help="Direct-download status CSV. Default: %(default)s")
    parser.add_argument("--sync-input", type=Path, default=DEFAULT_SYNC,
                        help="Image sync status CSV. Default: %(default)s")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT,
                        help="Merged downloaded-image CSV. Default: %(default)s")
    parser.add_argument("--checkpoint-db", type=Path, default=None,
                        help="SQLite resume checkpoint. Default: hidden database beside --output")
    parser.add_argument("--commit-interval", type=int, default=1000,
                        help="Persist input progress every N rows. Default: %(default)s")
    parser.add_argument("--progress-refresh-seconds", type=float, default=0.1,
                        help="Minimum seconds between progress updates. Default: %(default)s")
    parser.add_argument("--resume", action="store_true",
                        help="Continue an interrupted merge using the checkpoint. Default: disabled")
    parser.add_argument("--overwrite-state", action="store_true",
                        help="Discard this script's previous checkpoint and merged CSV before starting. Input CSVs are preserved. Default: disabled")
    args = parser.parse_args()
    if args.resume and args.overwrite_state:
        parser.error("Use either --resume or --overwrite-state, not both")
    if args.commit_interval <= 0 or args.progress_refresh_seconds <= 0:
        parser.error("--commit-interval and --progress-refresh-seconds must be positive")
    return args


def checkpoint_path(args):
    if args.checkpoint_db is not None:
        return args.checkpoint_db.expanduser().resolve()
    output = args.output.expanduser().resolve()
    return output.with_name("." + output.stem + "_checkpoint.sqlite3")


def remove_checkpoint(path):
    path.unlink(missing_ok=True)
    path.with_name(path.name + "-wal").unlink(missing_ok=True)
    path.with_name(path.name + "-shm").unlink(missing_ok=True)


def relative_from_image_path(image_path):
    normalized = image_path.replace("\\", "/")
    marker = "/images_fiftyone_all/"
    if marker not in normalized:
        raise ValueError("Image path is not within images_fiftyone_all: {!r}".format(image_path))
    relative = normalized.rsplit(marker, 1)[1]
    parts = relative.split("/")
    if not relative or any(part in ("", ".", "..") for part in parts):
        raise ValueError("Unsafe image relative path: {!r}".format(relative))
    return relative


def sync_relative(raw):
    relative = raw.strip().replace("\\", "/")
    if not relative or relative.startswith("/") or ":" in relative.split("/")[0]:
        raise ValueError("Unsafe sync RelativePath: {!r}".format(raw))
    if any(part in ("", ".", "..") for part in relative.split("/")):
        raise ValueError("Unsafe sync RelativePath: {!r}".format(raw))
    return relative


def open_store(path, settings):
    path.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(path)
    connection.execute("PRAGMA journal_mode=WAL")
    connection.execute("CREATE TABLE IF NOT EXISTS state (key TEXT PRIMARY KEY, value TEXT NOT NULL)")
    connection.execute(
        "CREATE TABLE IF NOT EXISTS samples ("
        "relative_path TEXT PRIMARY KEY, image_id TEXT NOT NULL, "
        "parent_label TEXT NOT NULL, parent_name TEXT NOT NULL, "
        "leaf_label TEXT NOT NULL, leaf_name TEXT NOT NULL, "
        "fiftyone_status TEXT NOT NULL, direct_status TEXT NOT NULL, "
        "direct_path TEXT NOT NULL, fiftyone_path TEXT NOT NULL, "
        "sync_status TEXT NOT NULL, sync_source TEXT NOT NULL, "
        "sync_destination TEXT NOT NULL, has_direct INTEGER NOT NULL, "
        "has_sync INTEGER NOT NULL)"
    )
    stored = connection.execute("SELECT value FROM state WHERE key = 'settings'").fetchone()
    encoded = json.dumps(settings, sort_keys=True)
    if stored is not None and stored[0] != encoded:
        connection.close()
        raise ValueError("Input CSVs changed; use --overwrite-state for a fresh merge")
    if stored is None:
        connection.execute("INSERT INTO state VALUES ('settings', ?)", (encoded,))
        connection.execute("INSERT INTO state VALUES ('direct_row', '0')")
        connection.execute("INSERT INTO state VALUES ('sync_row', '0')")
    connection.commit()
    return connection


def last_row(connection, key):
    return int(connection.execute("SELECT value FROM state WHERE key = ?", (key,)).fetchone()[0])


def save_progress(connection, key, number):
    connection.execute("UPDATE state SET value = ? WHERE key = ?", (str(number), key))
    connection.commit()


def require_columns(reader, required, path):
    missing = set(required).difference(reader.fieldnames or [])
    if missing:
        raise ValueError("{} lacks columns: {}".format(path, ", ".join(sorted(missing))))


def merge_direct(connection, path, interval, refresh_seconds):
    required = (
        "ImageID", "ParentLabelName", "ParentDisplayName", "LabelName",
        "DisplayName", "FiftyOneDownloadStatus", "DirectDownloadStatus",
        "DirectDownloadedImagePath", "FiftyOneDownloadedImagePath",
    )
    start = last_row(connection, "direct_row")
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        require_columns(reader, required, path)
        with tqdm(initial=start, desc="Merging direct CSV", unit="row",
                  mininterval=refresh_seconds) as progress:
            for number, row in enumerate(reader, start=1):
                if number <= start:
                    continue
                direct_status = row["DirectDownloadStatus"].strip()
                fiftyone_status = row["FiftyOneDownloadStatus"].strip()
                if direct_status in DIRECT_SUCCESS or fiftyone_status in FIFTYONE_SUCCESS:
                    image_path = row["DirectDownloadedImagePath"] or row["FiftyOneDownloadedImagePath"]
                    relative = relative_from_image_path(image_path)
                    image_id = row["ImageID"].strip()
                    if not image_id or Path(relative).stem != image_id:
                        raise ValueError("ImageID/path mismatch at direct CSV row {}".format(number + 1))
                    connection.execute(
                        "INSERT OR IGNORE INTO samples VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                        (relative, image_id, row["ParentLabelName"], row["ParentDisplayName"],
                         row["LabelName"], row["DisplayName"], fiftyone_status, direct_status,
                         row["DirectDownloadedImagePath"], row["FiftyOneDownloadedImagePath"],
                         "", "", "", 1, 0),
                    )
                progress.update(1)
                if number % interval == 0:
                    save_progress(connection, "direct_row", number)
            save_progress(connection, "direct_row", number if "number" in locals() else start)


def merge_sync(connection, path, interval, refresh_seconds):
    required = ("RelativePath", "SourcePath", "DestinationPath", "SyncStatus")
    start = last_row(connection, "sync_row")
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        require_columns(reader, required, path)
        with tqdm(initial=start, desc="Merging sync CSV", unit="row",
                  mininterval=refresh_seconds) as progress:
            for number, row in enumerate(reader, start=1):
                if number <= start:
                    continue
                status = row["SyncStatus"].strip()
                if status in SYNC_SUCCESS:
                    relative = sync_relative(row["RelativePath"])
                    image_id = Path(relative).stem
                    parent = Path(relative).parts[-3] if len(Path(relative).parts) >= 3 else ""
                    leaf = Path(relative).parts[-2] if len(Path(relative).parts) >= 2 else ""
                    connection.execute(
                        "INSERT INTO samples VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?) "
                        "ON CONFLICT(relative_path) DO UPDATE SET "
                        "sync_status = excluded.sync_status, sync_source = excluded.sync_source, "
                        "sync_destination = excluded.sync_destination, has_sync = 1",
                        (relative, image_id, "", parent, "", leaf, "", "", "", "",
                         status, row["SourcePath"], row["DestinationPath"], 0, 1),
                    )
                progress.update(1)
                if number % interval == 0:
                    save_progress(connection, "sync_row", number)
            save_progress(connection, "sync_row", number if "number" in locals() else start)


def export_csv(connection, output):
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(output.name + ".inprogress")
    try:
        with temporary.open("w", encoding="utf-8-sig", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(FIELDS)
            for row in connection.execute(
                "SELECT image_id, relative_path, parent_label, parent_name, leaf_label, "
                "leaf_name, fiftyone_status, direct_status, direct_path, fiftyone_path, "
                "sync_status, sync_source, sync_destination, "
                "CASE WHEN has_direct = 1 AND has_sync = 1 THEN 'direct_and_sync' "
                "WHEN has_direct = 1 THEN 'direct_only' ELSE 'sync_only' END "
                "FROM samples ORDER BY relative_path"
            ):
                writer.writerow(row)
        temporary.replace(output)
    finally:
        temporary.unlink(missing_ok=True)


def main():
    args = parse_args()
    direct = args.direct_input.expanduser().resolve()
    sync = args.sync_input.expanduser().resolve()
    output = args.output.expanduser().resolve()
    database = checkpoint_path(args)
    for path in (direct, sync):
        if not path.is_file():
            raise FileNotFoundError("Input CSV does not exist: {}".format(path))
    if args.overwrite_state:
        remove_checkpoint(database)
        output.unlink(missing_ok=True)
    elif (database.exists() or output.exists()) and not args.resume:
        raise FileExistsError("Previous merge exists; use --resume or --overwrite-state: {}".format(database))
    settings = {
        str(path): (path.stat().st_size, path.stat().st_mtime_ns)
        for path in (direct, sync)
    }
    connection = open_store(database, settings)
    try:
        merge_direct(connection, direct, args.commit_interval, args.progress_refresh_seconds)
        merge_sync(connection, sync, args.commit_interval, args.progress_refresh_seconds)
        export_csv(connection, output)
        source_counts = dict(connection.execute(
            "SELECT CASE WHEN has_direct = 1 AND has_sync = 1 THEN 'direct_and_sync' "
            "WHEN has_direct = 1 THEN 'direct_only' ELSE 'sync_only' END, COUNT(*) "
            "FROM samples GROUP BY 1"
        ))
        total, unique_ids = connection.execute(
            "SELECT COUNT(*), COUNT(DISTINCT image_id) FROM samples"
        ).fetchone()
        print("Merged image paths: {}".format(total))
        print("Unique ImageIDs: {}".format(unique_ids))
        print("Source counts: {}".format(source_counts))
        print("Output: {}".format(output))
        print("Checkpoint: {}".format(database))
    finally:
        connection.close()


if __name__ == "__main__":
    main()
