"""Incrementally synchronize one images_fiftyone_all tree into another.

The synchronization is one-way and preserves relative directory structure.
Files that exist only in the destination are never deleted. Completed work is
recorded in SQLite so interrupted runs can continue with ``--resume``.
"""

import argparse
import csv
import hashlib
import os
import shutil
import sqlite3
import time
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

from tqdm import tqdm


DEFAULT_SOURCE = Path("/media/xyycyc/Elements1/dyx/images_fiftyone_all")
DEFAULT_DESTINATION = Path(
    "/media/xyycyc/Elements/dyx/TinyCLIP/my_code/data/OpenImage/meta/Hierarchy/n2/images_fiftyone_all"
)
META_DIR = Path(__file__).resolve().parent.parent
DEFAULT_STATUS_CSV = (
    META_DIR / "Hierarchy" / "n2" / "sync_images_fiftyone_all_status.csv"
)
DEFAULT_EXTENSIONS = ".jpg,.jpeg,.png,.webp,.bmp"


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE, help="Source image-tree root. Default: %(default)s")
    parser.add_argument("--destination", type=Path, default=DEFAULT_DESTINATION, help="Destination image-tree root. Extra destination files are retained. Default: %(default)s")
    parser.add_argument("--status-csv", type=Path, default=DEFAULT_STATUS_CSV, help="Per-file synchronization status CSV. Default: %(default)s")
    parser.add_argument("--checkpoint-db", type=Path, default=None, help="SQLite progress database. Default: hidden database beside --status-csv")
    parser.add_argument("--extensions", default=DEFAULT_EXTENSIONS, help="Comma-separated case-insensitive image suffixes, each beginning with '.'. Default: %(default)s")
    parser.add_argument("--workers", type=int, default=8, help="Maximum concurrent file-copy workers. Lower this when the two disks become saturated. Default: %(default)s")
    parser.add_argument("--existing-policy", choices=("overwrite", "skip", "newer"), default="overwrite", help="Behavior for different same-path files: overwrite always copies the source, skip keeps the destination, and newer copies only when the source mtime is newer. Identical files are always skipped. Default: %(default)s")
    parser.add_argument("--compare", choices=("size-mtime", "checksum"), default="size-mtime", help="How existing files are considered identical. size-mtime is fast; checksum computes SHA-256 when sizes match. Default: %(default)s")
    parser.add_argument("--copy-retries", type=int, default=3, help="Maximum attempts for each failed copy, including the first attempt. Default: %(default)s")
    parser.add_argument("--retry-backoff", type=float, default=1.0, help="Base exponential delay in seconds between copy retries. Default: %(default)s")
    parser.add_argument("--checkpoint-interval", type=int, default=25, help="Commit SQLite progress after this many completed files. Destination files themselves remain resumable even before a commit. Default: %(default)s")
    parser.add_argument("--progress-refresh-seconds", type=float, default=0.1, help="Minimum seconds between real-time tqdm refreshes. Default: %(default)s")
    parser.add_argument("--resume", action="store_true", help="Reuse an existing checkpoint and retry unfinished or failed files. Completed destination files are not copied again.")
    parser.add_argument("--overwrite-state", action="store_true", help="Delete the previous checkpoint and status CSV before synchronizing. Destination images are not deleted.")
    args = parser.parse_args()
    if args.workers <= 0 or args.copy_retries <= 0 or args.checkpoint_interval <= 0:
        parser.error("--workers, --copy-retries, and --checkpoint-interval must be positive")
    if args.retry_backoff < 0.0 or args.progress_refresh_seconds <= 0.0:
        parser.error("--retry-backoff must be nonnegative and --progress-refresh-seconds positive")
    extensions = []
    for value in args.extensions.split(","):
        suffix = value.strip().lower()
        if not suffix or not suffix.startswith("."):
            parser.error("Every --extensions value must begin with '.': {!r}".format(value))
        extensions.append(suffix)
    args.extensions = tuple(dict.fromkeys(extensions))
    return args


def checkpoint_path(args):
    if args.checkpoint_db is not None:
        return args.checkpoint_db.expanduser()
    status_path = args.status_csv.expanduser()
    return status_path.with_name("." + status_path.stem + "_checkpoint.sqlite3")


def remove_checkpoint(path):
    path.unlink(missing_ok=True)
    path.with_name(path.name + "-wal").unlink(missing_ok=True)
    path.with_name(path.name + "-shm").unlink(missing_ok=True)


def validate_roots(source, destination):
    if not source.is_dir():
        raise FileNotFoundError("Source directory does not exist: {}".format(source))
    source = source.resolve()
    destination = destination.resolve()
    if (
        source == destination
        or source in destination.parents
        or destination in source.parents
    ):
        raise ValueError(
            "Source and destination must be separate, non-nested directories: {} / {}"
            .format(source, destination)
        )
    return source, destination


def discover_images(source, extensions, refresh_seconds):
    paths = []
    with tqdm(
        desc="Scanning source images",
        unit="file",
        mininterval=refresh_seconds,
    ) as progress:
        for path in source.rglob("*"):
            if path.is_file() and path.suffix.lower() in extensions:
                paths.append(path)
                progress.update(1)
    paths.sort(key=lambda path: path.relative_to(source).as_posix())
    if not paths:
        raise ValueError(
            "No images with extensions {} were found under {}".format(
                ", ".join(extensions), source
            )
        )
    return paths


def sha256(path, chunk_size=1024 * 1024):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def files_identical(source, destination, compare_mode):
    source_stat = source.stat()
    destination_stat = destination.stat()
    if source_stat.st_size != destination_stat.st_size:
        return False
    if compare_mode == "checksum":
        return sha256(source) == sha256(destination)
    return source_stat.st_mtime_ns == destination_stat.st_mtime_ns


def copied_file_is_valid(source, destination, compare_mode):
    if source.stat().st_size != destination.stat().st_size:
        return False
    if compare_mode == "checksum":
        return sha256(source) == sha256(destination)
    return True


def atomic_copy(source, destination):
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(
        ".{}.inprogress.{}".format(destination.name, uuid.uuid4().hex)
    )
    try:
        shutil.copy2(source, temporary)
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)


def synchronize_file(source, destination, args):
    try:
        if destination.is_file():
            if files_identical(source, destination, args.compare):
                return "existing", ""
            if args.existing_policy == "skip":
                return "skipped_conflict", "Destination differs and policy is skip"
            if args.existing_policy == "newer":
                if source.stat().st_mtime_ns <= destination.stat().st_mtime_ns:
                    return "skipped_conflict", "Destination is not older than source"
            success_status = "overwritten"
        else:
            success_status = "copied"

        errors = []
        for attempt in range(args.copy_retries):
            try:
                atomic_copy(source, destination)
                if not copied_file_is_valid(source, destination, args.compare):
                    raise OSError("Post-copy verification failed")
                return success_status, ""
            except OSError as error:
                errors.append(
                    "attempt {}/{}: {}".format(
                        attempt + 1, args.copy_retries, error
                    )
                )
                if attempt + 1 < args.copy_retries and args.retry_backoff:
                    time.sleep(args.retry_backoff * (2 ** attempt))
        return "error", " | ".join(errors)
    except OSError as error:
        return "error", str(error)


class SyncStore:
    def __init__(self, path, commit_interval):
        path.parent.mkdir(parents=True, exist_ok=True)
        self.connection = sqlite3.connect(path)
        self.connection.execute("PRAGMA journal_mode=WAL")
        self.connection.execute(
            """
            CREATE TABLE IF NOT EXISTS files (
                relative_path TEXT PRIMARY KEY,
                source_path TEXT NOT NULL,
                destination_path TEXT NOT NULL,
                source_size INTEGER NOT NULL,
                source_mtime_ns INTEGER NOT NULL,
                status TEXT NOT NULL,
                error_text TEXT NOT NULL,
                updated_at TEXT NOT NULL
            )
            """
        )
        self.connection.commit()
        self.commit_interval = commit_interval
        self.pending_updates = 0

    def seed(self, source_root, destination_root, paths):
        rows = []
        now = datetime.now(timezone.utc).isoformat()
        for source in paths:
            relative = source.relative_to(source_root)
            stat = source.stat()
            rows.append((
                relative.as_posix(),
                str(source),
                str(destination_root / relative),
                stat.st_size,
                stat.st_mtime_ns,
                "pending",
                "",
                now,
            ))
        self.connection.executemany(
            """
            INSERT INTO files (
                relative_path, source_path, destination_path, source_size,
                source_mtime_ns, status, error_text, updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(relative_path) DO UPDATE SET
                source_path = excluded.source_path,
                destination_path = excluded.destination_path,
                status = CASE
                    WHEN files.source_size != excluded.source_size
                      OR files.source_mtime_ns != excluded.source_mtime_ns
                    THEN 'pending'
                    ELSE files.status
                END,
                error_text = CASE
                    WHEN files.source_size != excluded.source_size
                      OR files.source_mtime_ns != excluded.source_mtime_ns
                    THEN ''
                    ELSE files.error_text
                END,
                source_size = excluded.source_size,
                source_mtime_ns = excluded.source_mtime_ns,
                updated_at = excluded.updated_at
            """,
            rows,
        )
        self.connection.commit()

    def record(self, relative_path, status, error_text):
        self.connection.execute(
            "UPDATE files SET status = ?, error_text = ?, updated_at = ? WHERE relative_path = ?",
            (
                status,
                error_text,
                datetime.now(timezone.utc).isoformat(),
                relative_path,
            ),
        )
        self.pending_updates += 1
        if self.pending_updates >= self.commit_interval:
            self.commit()

    def commit(self):
        self.connection.commit()
        self.pending_updates = 0

    def summary(self):
        return dict(
            self.connection.execute(
                "SELECT status, COUNT(*) FROM files GROUP BY status"
            ).fetchall()
        )

    def export_csv(self, path):
        self.commit()
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_name(path.name + ".inprogress")
        temporary.unlink(missing_ok=True)
        fields = [
            "RelativePath", "SourcePath", "DestinationPath", "SourceSize",
            "SourceMTimeNS", "SyncStatus", "SyncError", "UpdatedAt",
        ]
        with temporary.open("w", encoding="utf-8-sig", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(fields)
            for row in self.connection.execute(
                """
                SELECT relative_path, source_path, destination_path,
                       source_size, source_mtime_ns, status, error_text,
                       updated_at
                FROM files ORDER BY relative_path
                """
            ):
                writer.writerow(row)
        temporary.replace(path)

    def close(self):
        self.commit()
        self.connection.close()


def main():
    args = parse_args()
    source, destination = validate_roots(
        args.source.expanduser(), args.destination.expanduser()
    )
    status_csv = args.status_csv.expanduser()
    database = checkpoint_path(args)
    if args.overwrite_state:
        remove_checkpoint(database)
        status_csv.unlink(missing_ok=True)
    elif (database.exists() or status_csv.exists()) and not args.resume:
        raise FileExistsError(
            "Previous state exists. Use --resume or --overwrite-state: {}"
            .format(database)
        )

    destination.mkdir(parents=True, exist_ok=True)
    paths = discover_images(
        source, args.extensions, args.progress_refresh_seconds
    )
    store = SyncStore(database, args.checkpoint_interval)
    interrupted = False
    try:
        store.seed(source, destination, paths)
        counts = {
            "copied": 0,
            "overwritten": 0,
            "existing": 0,
            "skipped_conflict": 0,
            "error": 0,
        }
        executor = ThreadPoolExecutor(max_workers=args.workers)
        futures = {}
        try:
            for source_path in paths:
                relative = source_path.relative_to(source)
                future = executor.submit(
                    synchronize_file,
                    source_path,
                    destination / relative,
                    args,
                )
                futures[future] = relative.as_posix()
            with tqdm(
                total=len(futures),
                desc="Synchronizing images",
                unit="file",
                mininterval=args.progress_refresh_seconds,
            ) as progress:
                for future in as_completed(futures):
                    relative_path = futures[future]
                    try:
                        status, error_text = future.result()
                    except Exception as error:
                        status, error_text = "error", str(error)
                    store.record(relative_path, status, error_text)
                    counts[status] += 1
                    progress.update(1)
                    progress.set_postfix(**counts)
        except KeyboardInterrupt:
            interrupted = True
            print(
                "Interruption requested; cancelling queued copies and saving progress...",
                flush=True,
            )
            for future in futures:
                future.cancel()
        finally:
            executor.shutdown(wait=True, cancel_futures=True)
        store.export_csv(status_csv)
        print("Source images: {}".format(len(paths)))
        print("Status counts: {}".format(store.summary()))
        print("Status CSV: {}".format(status_csv.resolve()))
        print("Checkpoint database: {}".format(database.resolve()))
        print("Destination: {}".format(destination))
    finally:
        store.close()
    if interrupted:
        raise SystemExit(130)


if __name__ == "__main__":
    main()
