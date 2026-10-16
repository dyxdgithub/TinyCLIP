"""Decode every image under an OpenImages directory and report damaged files.

Pillow's full decoding detects truncated image data that Image.verify() alone
can miss. A SQLite progress database makes repeated or interrupted scans resume
without decoding unchanged images again.
"""

import argparse
import csv
import os
import sqlite3
import time
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from pathlib import Path

from PIL import Image, ImageFile
from tqdm import tqdm


META_DIR = Path(__file__).resolve().parent.parent
N2_DIR = META_DIR / "Hierarchy/n2"
DEFAULT_ROOT = N2_DIR / "images_all"
DEFAULT_STATUS = N2_DIR / "images_all_integrity_status.csv"
DEFAULT_TRUNCATED = N2_DIR / "images_all_truncated_images.csv"
IMAGE_SUFFIXES = "jpg,jpeg,png,webp,bmp,gif,tif,tiff"
CSV_FIELDS = ("RelativePath", "Status", "Error", "FileSize", "ModifiedNs")
TRUNCATION_MARKERS = (
    "truncat", "broken data stream", "unexpected end", "end of file",
    "not enough image data", "premature end", "incomplete",
)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--images-dir", type=Path, default=DEFAULT_ROOT,
                        help="Image root to scan recursively; paths in reports are relative to this root.")
    parser.add_argument("--status-csv", type=Path, default=DEFAULT_STATUS,
                        help="Complete CSV with one current status per image.")
    parser.add_argument("--truncated-csv", type=Path, default=DEFAULT_TRUNCATED,
                        help="CSV containing only images Pillow identifies as truncated.")
    parser.add_argument("--state-db", type=Path, default=None,
                        help="SQLite resume database; default: <status-csv>.sqlite3.")
    parser.add_argument("--extensions", default=IMAGE_SUFFIXES,
                        help="Comma-separated image filename extensions, with or without dots.")
    parser.add_argument("--workers", type=int, default=8,
                        help="Concurrent image decoding threads; use 1 for slow storage.")
    parser.add_argument("--force", action="store_true",
                        help="Decode every image again even when size and modification time match the saved result.")
    args = parser.parse_args(argv)
    if args.workers < 1:
        parser.error("--workers must be positive")
    args.suffixes = {"." + part.strip().lower().lstrip(".") for part in args.extensions.split(",") if part.strip()}
    if not args.suffixes:
        parser.error("--extensions must contain at least one extension")
    if args.state_db is None:
        args.state_db = args.status_csv.with_suffix(".sqlite3")
    return args


def discover_images(root, suffixes):
    files = []
    with tqdm(desc="Finding images", unit="image") as progress:
        for directory, _, names in os.walk(root):
            for name in names:
                path = Path(directory) / name
                if path.suffix.lower() not in suffixes:
                    continue
                relative = path.relative_to(root).as_posix()
                try:
                    stat = path.stat()
                    size, modified_ns = stat.st_size, stat.st_mtime_ns
                except OSError:
                    # The decode pass will report a vanished or inaccessible file.
                    size, modified_ns = -1, -1
                files.append((relative, size, modified_ns))
                progress.update()
    files.sort(key=lambda item: item[0])
    return files


def inspect_image(root, relative):
    path = root / relative
    try:
        with Image.open(path) as image:
            image.verify()
        with Image.open(path) as image:
            for frame in range(getattr(image, "n_frames", 1)):
                image.seek(frame)
                image.load()
    except (OSError, EOFError, ValueError, SyntaxError, Image.DecompressionBombError) as error:
        message = str(error)
        status = "truncated" if isinstance(error, EOFError) or any(marker in message.lower() for marker in TRUNCATION_MARKERS) else "unreadable"
        return status, message
    return "ok", ""


def open_state(path):
    path.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(path, isolation_level=None)
    connection.execute("PRAGMA journal_mode=WAL")
    connection.execute("PRAGMA synchronous=NORMAL")
    connection.execute(
        "CREATE TABLE IF NOT EXISTS image_status ("
        "relative_path TEXT PRIMARY KEY, file_size INTEGER NOT NULL, "
        "modified_ns INTEGER NOT NULL, status TEXT NOT NULL, "
        "error TEXT NOT NULL, scan_id INTEGER NOT NULL)"
    )
    return connection


def write_result(connection, entry, status, error, scan_id):
    relative, size, modified_ns = entry
    connection.execute(
        "INSERT INTO image_status VALUES (?, ?, ?, ?, ?, ?) "
        "ON CONFLICT(relative_path) DO UPDATE SET "
        "file_size=excluded.file_size, modified_ns=excluded.modified_ns, "
        "status=excluded.status, error=excluded.error, scan_id=excluded.scan_id",
        (relative, size, modified_ns, status, error, scan_id),
    )


def check_pending(connection, root, pending, workers, scan_id, progress):
    iterator = iter(pending)
    futures = {}
    executor = ThreadPoolExecutor(max_workers=workers)
    try:
        for _ in range(min(len(pending), workers * 4)):
            entry = next(iterator)
            futures[executor.submit(inspect_image, root, entry[0])] = entry
        while futures:
            done, _ = wait(futures, return_when=FIRST_COMPLETED)
            for future in done:
                entry = futures.pop(future)
                status, error = future.result()
                write_result(connection, entry, status, error, scan_id)
                progress.update()
                replacement = next(iterator, None)
                if replacement is not None:
                    futures[executor.submit(inspect_image, root, replacement[0])] = replacement
    finally:
        executor.shutdown(wait=False, cancel_futures=True)


def export_csv(connection, path, status=None):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    query = "SELECT relative_path, status, error, file_size, modified_ns FROM image_status"
    if status is not None:
        query += " WHERE status = ?"
    query += " ORDER BY relative_path"
    parameters = () if status is None else (status,)
    with temporary.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(CSV_FIELDS)
        for row in connection.execute(query, parameters):
            writer.writerow(row)
    temporary.replace(path)


def main(argv=None):
    args = parse_args(argv)
    root = args.images_dir.resolve()
    if not root.is_dir():
        raise NotADirectoryError("Image directory does not exist: {}".format(root))
    if args.status_csv.resolve() == args.truncated_csv.resolve():
        raise ValueError("--status-csv and --truncated-csv must be different files")
    ImageFile.LOAD_TRUNCATED_IMAGES = False
    files = discover_images(root, args.suffixes)
    if not files:
        raise RuntimeError("No images with extensions {} found under {}".format(sorted(args.suffixes), root))

    scan_id = time.time_ns()
    connection = open_state(args.state_db)
    try:
        pending = []
        reused = 0
        with tqdm(total=len(files), desc="Checking images", unit="image") as progress:
            for entry in files:
                relative, size, modified_ns = entry
                saved = connection.execute(
                    "SELECT file_size, modified_ns FROM image_status WHERE relative_path = ?",
                    (relative,),
                ).fetchone()
                if saved == (size, modified_ns) and not args.force:
                    connection.execute(
                        "UPDATE image_status SET scan_id = ? WHERE relative_path = ?",
                        (scan_id, relative),
                    )
                    reused += 1
                    progress.update()
                else:
                    pending.append(entry)
            check_pending(connection, root, pending, args.workers, scan_id, progress)

        connection.execute("DELETE FROM image_status WHERE scan_id != ?", (scan_id,))
        export_csv(connection, args.status_csv)
        export_csv(connection, args.truncated_csv, status="truncated")
        counts = dict(connection.execute("SELECT status, COUNT(*) FROM image_status GROUP BY status"))
        print("Scanned {} images ({} reused, {} decoded): {}".format(
            len(files), reused, len(pending), counts
        ), flush=True)
        print("Status CSV: {}\nTruncated images CSV: {}\nResume DB: {}".format(
            args.status_csv, args.truncated_csv, args.state_db
        ), flush=True)
    finally:
        connection.close()


if __name__ == "__main__":
    main()
