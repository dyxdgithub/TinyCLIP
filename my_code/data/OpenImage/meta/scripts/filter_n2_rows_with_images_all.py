"""Select n2 metadata rows whose image exists in images_all.

Each row is checked under its parent/leaf class directory by ImageID. The
default CSV contains only nonempty existing images; --include-missing writes
all input rows with a status for diagnostics. Images are never modified.
"""

import argparse
import csv
import sqlite3
from pathlib import Path

from tqdm import tqdm


META_DIR = Path(__file__).resolve().parent.parent
N2_DIR = META_DIR / "Hierarchy" / "n2"
DEFAULT_INPUT = N2_DIR / "Image IDs_with_last_two_layers_multi_members_n2_one_label_per_parent.csv"
DEFAULT_IMAGES_DIR = N2_DIR / "images_all"
DEFAULT_OUTPUT = N2_DIR / "Image_IDs_n2_one_label_per_parent_in_images_all.csv"
DEFAULT_EXTENSIONS = ".jpg,.jpeg,.png,.webp,.bmp"
INVALID_PATH_CHARS = '<>:"/\\|?*'
STATUS_FIELDS = ("ImageStatus", "ImageRelativePath", "ImageCheckError")


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT,
                        help="Metadata CSV containing ImageID, ParentDisplayName, and DisplayName. Default: %(default)s")
    parser.add_argument("--images-dir", type=Path, default=DEFAULT_IMAGES_DIR,
                        help="Parent/leaf image tree to check. Default: %(default)s")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT,
                        help="Filtered status CSV. Default: %(default)s")
    parser.add_argument("--checkpoint-db", type=Path, default=None,
                        help="SQLite resume checkpoint. Default: hidden database beside --output")
    parser.add_argument("--extensions", default=DEFAULT_EXTENSIONS,
                        help="Comma-separated image extensions to try, in priority order. Default: %(default)s")
    parser.add_argument("--include-missing", action="store_true",
                        help="Write present, missing, empty, and error rows instead of present rows only. Default: disabled")
    parser.add_argument("--commit-interval", type=int, default=100,
                        help="Persist results after every N checked rows. Default: %(default)s")
    parser.add_argument("--progress-refresh-seconds", type=float, default=0.1,
                        help="Minimum seconds between progress updates. Default: %(default)s")
    parser.add_argument("--resume", action="store_true",
                        help="Continue a checkpoint and retry error rows. Previously checked rows are reused. Default: disabled")
    parser.add_argument("--overwrite-state", action="store_true",
                        help="Discard this script's prior checkpoint and output, then check all rows again. Images are preserved. Default: disabled")
    args = parser.parse_args()
    if args.resume and args.overwrite_state:
        parser.error("Use either --resume or --overwrite-state, not both")
    if args.commit_interval <= 0 or args.progress_refresh_seconds <= 0:
        parser.error("--commit-interval and --progress-refresh-seconds must be positive")
    extensions = tuple(dict.fromkeys(part.strip().lower() for part in args.extensions.split(",")))
    if any(not part.startswith(".") or len(part) < 2 for part in extensions):
        parser.error("Every --extensions value must be a nonempty suffix beginning with '.'")
    args.extensions = extensions
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


def safe_component(value):
    cleaned = "".join("_" if char in INVALID_PATH_CHARS else char for char in str(value))
    return cleaned.strip().rstrip(".") or "unnamed"


def inspect_image(row, images_dir, extensions):
    image_id = (row["ImageID"] or "").strip()
    if not image_id or image_id in (".", "..") or any(char in image_id for char in "/\\"):
        return "error", "", "Invalid or empty ImageID"
    parent = safe_component(row["ParentDisplayName"])
    leaf = safe_component(row["DisplayName"])
    relative_dir = Path(parent) / leaf
    empty_path = ""
    try:
        for suffix in extensions:
            relative = relative_dir / (image_id + suffix)
            candidate = images_dir / relative
            if candidate.is_file():
                if candidate.stat().st_size > 0:
                    return "present", relative.as_posix(), ""
                empty_path = relative.as_posix()
    except OSError as error:
        return "error", "", str(error)
    if empty_path:
        return "empty", empty_path, "Image file is zero bytes"
    return "missing", "", ""


def open_store(path, settings):
    path.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(path)
    connection.execute("PRAGMA journal_mode=WAL")
    connection.execute("CREATE TABLE IF NOT EXISTS settings (key TEXT PRIMARY KEY, value TEXT NOT NULL)")
    connection.execute(
        "CREATE TABLE IF NOT EXISTS results ("
        "row_number INTEGER PRIMARY KEY, status TEXT NOT NULL, "
        "relative_path TEXT NOT NULL, error_text TEXT NOT NULL)"
    )
    stored = dict(connection.execute("SELECT key, value FROM settings"))
    if stored and stored != settings:
        connection.close()
        raise ValueError("Input or options changed; use --overwrite-state to start a fresh check")
    if not stored:
        connection.executemany("INSERT INTO settings VALUES (?, ?)", settings.items())
    connection.commit()
    return connection


def check_rows(connection, input_path, images_dir, extensions, interval, refresh_seconds):
    completed = connection.execute("SELECT COUNT(*) FROM results WHERE status != 'error'").fetchone()[0]
    updated = 0
    with input_path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        required = {"ImageID", "ParentDisplayName", "DisplayName"}
        missing = required.difference(reader.fieldnames or [])
        if missing:
            raise ValueError("Input CSV lacks columns: {}".format(", ".join(sorted(missing))))
        conflict = set(STATUS_FIELDS).intersection(reader.fieldnames or [])
        if conflict:
            raise ValueError("Input CSV already has output columns: {}".format(", ".join(sorted(conflict))))
        with tqdm(initial=completed, desc="Checking images_all", unit="row",
                  mininterval=refresh_seconds) as progress:
            for number, row in enumerate(reader, start=2):
                prior = connection.execute(
                    "SELECT status FROM results WHERE row_number = ?", (number,)
                ).fetchone()
                if prior is not None and prior[0] != "error":
                    continue
                status, relative, error_text = inspect_image(row, images_dir, extensions)
                connection.execute(
                    "INSERT OR REPLACE INTO results VALUES (?, ?, ?, ?)",
                    (number, status, relative, error_text),
                )
                updated += 1
                if updated % interval == 0:
                    connection.commit()
                progress.update(1)
    connection.commit()


def export_csv(connection, input_path, output, include_missing):
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(output.name + ".inprogress")
    result_rows = iter(connection.execute(
        "SELECT row_number, status, relative_path, error_text FROM results ORDER BY row_number"
    ))
    written = 0
    try:
        with input_path.open("r", encoding="utf-8-sig", newline="") as source, \
                temporary.open("w", encoding="utf-8-sig", newline="") as destination:
            reader = csv.DictReader(source)
            writer = csv.DictWriter(destination, fieldnames=reader.fieldnames + list(STATUS_FIELDS))
            writer.writeheader()
            for number, row in enumerate(reader, start=2):
                stored_number, status, relative, error_text = next(result_rows)
                if stored_number != number:
                    raise RuntimeError("Checkpoint rows do not match the input CSV")
                if status != "present" and not include_missing:
                    continue
                row.update(zip(STATUS_FIELDS, (status, relative, error_text)))
                writer.writerow(row)
                written += 1
        temporary.replace(output)
    finally:
        temporary.unlink(missing_ok=True)
    return written


def main():
    args = parse_args()
    input_path = args.input.expanduser().resolve()
    images_dir = args.images_dir.expanduser().resolve()
    output = args.output.expanduser().resolve()
    database = checkpoint_path(args)
    if not input_path.is_file():
        raise FileNotFoundError("Input CSV does not exist: {}".format(input_path))
    if not images_dir.is_dir():
        raise NotADirectoryError("Image directory does not exist: {}".format(images_dir))
    if args.overwrite_state:
        remove_checkpoint(database)
        output.unlink(missing_ok=True)
    elif (database.exists() or output.exists()) and not args.resume:
        raise FileExistsError("Previous output or checkpoint exists; use --resume or --overwrite-state")
    input_stat = input_path.stat()
    settings = {
        "input": str(input_path),
        "input_size": str(input_stat.st_size),
        "input_mtime_ns": str(input_stat.st_mtime_ns),
        "images_dir": str(images_dir),
        "extensions": ",".join(args.extensions),
    }
    connection = open_store(database, settings)
    try:
        check_rows(connection, input_path, images_dir, args.extensions,
                   args.commit_interval, args.progress_refresh_seconds)
        written = export_csv(connection, input_path, output, args.include_missing)
        counts = dict(connection.execute("SELECT status, COUNT(*) FROM results GROUP BY status"))
        print("Status counts: {}".format(counts))
        print("Output rows: {}".format(written))
        print("Output: {}".format(output))
        print("Checkpoint: {}".format(database))
    finally:
        connection.close()


if __name__ == "__main__":
    main()
