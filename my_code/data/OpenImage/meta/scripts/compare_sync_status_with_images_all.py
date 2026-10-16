"""Compare a sync-status CSV with images_all on the machine holding the images.

The report contains one row per relative image path. It includes paths found
only in images_all, and never changes the image tree.
"""

import argparse
import csv
import os
import sqlite3
from pathlib import Path

from tqdm import tqdm


META_DIR = Path(__file__).resolve().parent.parent
N2_DIR = META_DIR / "Hierarchy" / "n2"
DEFAULT_INPUT = N2_DIR / "sync_images_fiftyone_all_status.csv"
DEFAULT_IMAGES = N2_DIR / "images_all"
DEFAULT_OUTPUT = N2_DIR / "sync_images_fiftyone_all_vs_images_all_status.csv"
DEFAULT_EXTENSIONS = ".jpg,.jpeg,.png,.webp,.bmp"
OUTPUT_FIELDS = (
    "RelativePath", "SyncStatus", "ExpectedSize", "ActualSize",
    "ImageStatus", "Error",
)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT,
                        help="Sync status CSV with RelativePath, SourceSize, and SyncStatus. Default: %(default)s")
    parser.add_argument("--images-dir", type=Path, default=DEFAULT_IMAGES,
                        help="Image tree to inspect recursively. Default: %(default)s")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT,
                        help="Per-path comparison CSV. Default: %(default)s")
    parser.add_argument("--checkpoint-db", type=Path, default=None,
                        help="SQLite checkpoint. Default: hidden database beside --output")
    parser.add_argument("--compare", choices=("size", "existence"), default="size",
                        help="size flags unequal SourceSize; existence only checks that a file exists. Default: %(default)s")
    parser.add_argument("--extensions", default=DEFAULT_EXTENSIONS,
                        help="Comma-separated, case-insensitive image suffixes for finding extra files. Default: %(default)s")
    parser.add_argument("--commit-interval", type=int, default=100,
                        help="Persist results every N inspected images. Default: %(default)s")
    parser.add_argument("--progress-refresh-seconds", type=float, default=0.1,
                        help="Minimum seconds between progress updates. Default: %(default)s")
    parser.add_argument("--resume", action="store_true",
                        help="Continue an interrupted checkpoint without rechecking recorded paths. Use --overwrite-state for a fresh comparison. Default: disabled")
    parser.add_argument("--overwrite-state", action="store_true",
                        help="Discard this script's prior checkpoint and report, then compare again. Images are never deleted. Default: disabled")
    args = parser.parse_args()
    if args.resume and args.overwrite_state:
        parser.error("Use either --resume or --overwrite-state, not both")
    if args.commit_interval <= 0 or args.progress_refresh_seconds <= 0:
        parser.error("--commit-interval and --progress-refresh-seconds must be positive")
    extensions = tuple(dict.fromkeys(part.strip().lower() for part in args.extensions.split(",")))
    if any(not part.startswith(".") or len(part) < 2 for part in extensions):
        parser.error("Every --extensions value must be a suffix beginning with '.'")
    args.extensions = extensions
    return args


def checkpoint_path(args):
    if args.checkpoint_db is not None:
        return args.checkpoint_db.expanduser()
    output = args.output.expanduser()
    return output.with_name("." + output.stem + "_checkpoint.sqlite3")


def remove_checkpoint(path):
    path.unlink(missing_ok=True)
    path.with_name(path.name + "-wal").unlink(missing_ok=True)
    path.with_name(path.name + "-shm").unlink(missing_ok=True)


def normalized_relative_path(raw):
    text = raw.strip().replace("\\", "/")
    path = Path(text)
    if (not text or text.startswith("/") or ":" in text.split("/")[0]
            or any(part in ("", ".", "..") for part in text.split("/"))):
        raise ValueError("Unsafe or empty RelativePath: {!r}".format(raw))
    return path.as_posix()


def open_store(path, settings):
    path.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(path)
    connection.execute("PRAGMA journal_mode=WAL")
    connection.execute("CREATE TABLE IF NOT EXISTS settings (key TEXT PRIMARY KEY, value TEXT NOT NULL)")
    connection.execute(
        "CREATE TABLE IF NOT EXISTS images ("
        "relative_path TEXT PRIMARY KEY, sync_status TEXT NOT NULL, "
        "expected_size INTEGER, actual_size INTEGER, status TEXT NOT NULL, "
        "error_text TEXT NOT NULL)"
    )
    stored = dict(connection.execute("SELECT key, value FROM settings"))
    if stored and stored != settings:
        connection.close()
        raise ValueError("Checkpoint inputs or options changed; use --overwrite-state for a fresh comparison")
    if not stored:
        connection.executemany("INSERT INTO settings VALUES (?, ?)", settings.items())
    connection.commit()
    return connection


def seed_csv(connection, input_path, refresh_seconds):
    with input_path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        required = {"RelativePath", "SourceSize", "SyncStatus"}
        if not required.issubset(reader.fieldnames or []):
            raise ValueError("Input CSV must contain RelativePath, SourceSize, and SyncStatus")
        batch = []
        with tqdm(desc="Reading sync CSV", unit="row", mininterval=refresh_seconds) as progress:
            for row in reader:
                relative = normalized_relative_path(row["RelativePath"])
                try:
                    expected_size = int(row["SourceSize"])
                except (TypeError, ValueError) as error:
                    raise ValueError("Invalid SourceSize for {!r}".format(relative)) from error
                if expected_size < 0:
                    raise ValueError("Negative SourceSize for {!r}".format(relative))
                batch.append((relative, row["SyncStatus"], expected_size, None, "pending", ""))
                if len(batch) >= 1000:
                    insert_rows(connection, batch)
                    connection.commit()
                    batch.clear()
                progress.update(1)
        if batch:
            insert_rows(connection, batch)
    connection.commit()


def insert_rows(connection, rows):
    connection.executemany(
        "INSERT OR IGNORE INTO images "
        "(relative_path, sync_status, expected_size, actual_size, status, error_text) "
        "VALUES (?, ?, ?, ?, ?, ?)", rows,
    )


def scan_images(connection, images_dir, compare, extensions, commit_interval, refresh_seconds):
    def walk_error(error):
        raise error

    checked = 0
    with tqdm(desc="Inspecting images_all", unit="image", mininterval=refresh_seconds) as progress:
        for directory, _, filenames in os.walk(images_dir, onerror=walk_error):
            for filename in filenames:
                path = Path(directory) / filename
                relative = path.relative_to(images_dir).as_posix()
                prior = connection.execute(
                    "SELECT status, expected_size FROM images WHERE relative_path = ?", (relative,)
                ).fetchone()
                if prior is None and path.suffix.lower() not in extensions:
                    continue
                if prior is not None and prior[0] != "pending":
                    progress.update(1)
                    continue
                try:
                    size = path.stat().st_size
                    if prior is None:
                        connection.execute(
                            "INSERT INTO images VALUES (?, ?, ?, ?, ?, ?)",
                            (relative, "", None, size, "extra", ""),
                        )
                    else:
                        status = "size_mismatch" if compare == "size" and size != prior[1] else "present"
                        connection.execute(
                            "UPDATE images SET actual_size = ?, status = ?, error_text = '' "
                            "WHERE relative_path = ?", (size, status, relative),
                        )
                except OSError as error:
                    if prior is None:
                        connection.execute(
                            "INSERT INTO images VALUES (?, ?, ?, ?, ?, ?)",
                            (relative, "", None, None, "error", str(error)),
                        )
                    else:
                        connection.execute(
                            "UPDATE images SET status = 'error', error_text = ? WHERE relative_path = ?",
                            (str(error), relative),
                        )
                checked += 1
                if checked % commit_interval == 0:
                    connection.commit()
                progress.update(1)
    connection.execute("UPDATE images SET status = 'missing' WHERE status = 'pending'")
    connection.commit()


def export_report(connection, output):
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(output.name + ".inprogress")
    try:
        with temporary.open("w", encoding="utf-8-sig", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(OUTPUT_FIELDS)
            for row in connection.execute(
                "SELECT relative_path, sync_status, expected_size, actual_size, status, error_text "
                "FROM images ORDER BY relative_path"
            ):
                writer.writerow(row)
        temporary.replace(output)
    finally:
        temporary.unlink(missing_ok=True)


def main():
    args = parse_args()
    input_path = args.input.expanduser().resolve()
    images_dir = args.images_dir.expanduser().resolve()
    output = args.output.expanduser().resolve()
    database = checkpoint_path(args).resolve()
    if not input_path.is_file():
        raise FileNotFoundError("Sync CSV does not exist: {}".format(input_path))
    if not images_dir.is_dir():
        raise NotADirectoryError("Image directory does not exist: {}".format(images_dir))
    if args.overwrite_state:
        remove_checkpoint(database)
        output.unlink(missing_ok=True)
    elif (database.exists() or output.exists()) and not args.resume:
        raise FileExistsError("Previous state exists; use --resume or --overwrite-state: {}".format(database))
    settings = {
        "input": str(input_path),
        "input_size": str(input_path.stat().st_size),
        "input_mtime_ns": str(input_path.stat().st_mtime_ns),
        "images_dir": str(images_dir),
        "compare": args.compare,
        "extensions": ",".join(args.extensions),
    }
    connection = open_store(database, settings)
    try:
        seed_csv(connection, input_path, args.progress_refresh_seconds)
        scan_images(connection, images_dir, args.compare, args.extensions,
                    args.commit_interval, args.progress_refresh_seconds)
        export_report(connection, output)
        counts = dict(connection.execute("SELECT status, COUNT(*) FROM images GROUP BY status"))
        csv_count = connection.execute("SELECT COUNT(*) FROM images WHERE expected_size IS NOT NULL").fetchone()[0]
        image_count = sum(counts.get(status, 0) for status in ("present", "size_mismatch", "extra", "error"))
        print("Sync CSV paths: {}".format(csv_count))
        print("Image paths inspected: {}".format(image_count))
        print("Status counts: {}".format(counts))
        print("Report: {}".format(output))
        print("Checkpoint: {}".format(database))
    finally:
        connection.close()


if __name__ == "__main__":
    main()
