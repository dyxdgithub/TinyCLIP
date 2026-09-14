"""Randomly retain a fixed sample size when an n2 leaf class has enough rows.

Rows are grouped by the single LabelName column. Classes with at least
--samples-per-class rows retain exactly that many rows; smaller classes retain
all rows. Selection uses a deterministic random rank, so the same source and
seed always yield the same sample. The script writes a sampled-row CSV and a
per-class count CSV. A SQLite checkpoint supports --resume after interruption.
"""

import argparse
import csv
import hashlib
import shutil
import sqlite3
from pathlib import Path

from tqdm import tqdm


META_DIR = Path(__file__).resolve().parent.parent
N2_DIR = META_DIR / "Hierarchy" / "n2"
DEFAULT_INPUT = N2_DIR / "Image IDs_with_last_two_layers_multi_members_n2_one_label_per_parent.csv"
DEFAULT_OUTPUT = N2_DIR / "Image IDs_with_last_two_layers_multi_members_n2_one_label_per_parent_sampled_300_per_class.csv"
DEFAULT_COUNTS_OUTPUT = N2_DIR / "Image IDs_with_last_two_layers_multi_members_n2_one_label_per_parent_sampled_300_per_class_counts.csv"
COUNT_FIELDS = [
    "LabelName",
    "DisplayName",
    "ParentLabelName",
    "ParentDisplayName",
    "InputSampleCount",
    "SelectedSampleCount",
    "SamplesPerClass",
    "RandomSeed",
]


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT, help="One-label-per-row input CSV. Default: %(default)s")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT, help="Sampled rows output CSV. Default: %(default)s")
    parser.add_argument("--counts-output", type=Path, default=DEFAULT_COUNTS_OUTPUT, help="Per-class input and selected-count CSV. Default: %(default)s")
    parser.add_argument("--checkpoint-db", type=Path, default=None, help="SQLite checkpoint database. Default: hidden database beside --output.")
    parser.add_argument("--label-column", default="LabelName", help="Single leaf-label ID column used for per-class sampling. Default: %(default)s")
    parser.add_argument("--display-name-column", default="DisplayName", help="Single leaf display-name column used in the count table. Default: %(default)s")
    parser.add_argument("--parent-label-column", default="ParentLabelName", help="Parent class label-ID column used in the count table. Default: %(default)s")
    parser.add_argument("--parent-display-name-column", default="ParentDisplayName", help="Parent class display-name column used in the count table. Default: %(default)s")
    parser.add_argument("--image-id-column", default="ImageID", help="Image ID used in deterministic random ranking. Default: %(default)s")
    parser.add_argument("--samples-per-class", type=int, default=300, help="Exact random sample count for each leaf class with at least this many rows; smaller classes retain all rows. Default: %(default)s")
    parser.add_argument("--random-seed", type=int, default=42, help="Seed used in deterministic random ranking. Default: %(default)s")
    parser.add_argument("--commit-rows", type=int, default=1000, help="Rows processed between durable checkpoint commits. Default: %(default)s")
    parser.add_argument("--progress-refresh-seconds", type=float, default=0.1, help="Minimum seconds between tqdm refreshes. Default: %(default)s")
    parser.add_argument("--resume", action="store_true", help="Continue from the saved checkpoint without recalculating completed candidate rows.")
    parser.add_argument("--overwrite", action="store_true", help="Remove prior outputs and checkpoint state before sampling from the beginning.")
    parser.add_argument("--cleanup-checkpoint", action="store_true", help="Delete the checkpoint database after both outputs are written successfully.")
    args = parser.parse_args()
    if args.samples_per_class <= 0:
        parser.error("--samples-per-class must be positive")
    if args.commit_rows <= 0:
        parser.error("--commit-rows must be positive")
    if args.progress_refresh_seconds <= 0:
        parser.error("--progress-refresh-seconds must be positive")
    return args


def read_header(path):
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return csv.DictReader(handle).fieldnames or []


def count_rows(input_path, refresh_seconds):
    row_count = 0
    with input_path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.reader(handle)
        next(reader, None)
        with tqdm(desc="Counting input rows", unit="row", mininterval=refresh_seconds) as progress:
            for row_count, _ in enumerate(reader, start=1):
                progress.update(1)
    return row_count


def random_rank(label, image_id, input_row_number, seed):
    material = "{}\0{}\0{}\0{}".format(seed, label, image_id, input_row_number)
    return hashlib.sha256(material.encode("utf-8")).hexdigest()


def temporary_path(path):
    return path.with_name(path.name + ".inprogress")


class SamplingCheckpoint:
    def __init__(self, path):
        path.parent.mkdir(parents=True, exist_ok=True)
        self.connection = sqlite3.connect(path)
        self.connection.execute("PRAGMA journal_mode=WAL")
        self.connection.execute(
            "CREATE TABLE IF NOT EXISTS state (key TEXT PRIMARY KEY, value TEXT NOT NULL)"
        )
        self.connection.execute(
            "CREATE TABLE IF NOT EXISTS classes ("
            "label TEXT PRIMARY KEY, display_name TEXT NOT NULL, "
            "parent_label TEXT NOT NULL, parent_display_name TEXT NOT NULL, "
            "input_count INTEGER NOT NULL)"
        )
        self.connection.execute(
            "CREATE TABLE IF NOT EXISTS candidates ("
            "label TEXT NOT NULL, random_rank TEXT NOT NULL, input_row_number INTEGER NOT NULL, "
            "PRIMARY KEY (label, input_row_number))"
        )
        self.connection.execute(
            "CREATE INDEX IF NOT EXISTS candidates_rank_index "
            "ON candidates (label, random_rank, input_row_number)"
        )
        self.connection.commit()

    def get_state(self, key, default=None):
        row = self.connection.execute("SELECT value FROM state WHERE key = ?", (key,)).fetchone()
        return row[0] if row else default

    def set_state(self, key, value):
        self.connection.execute(
            "INSERT INTO state (key, value) VALUES (?, ?) "
            "ON CONFLICT(key) DO UPDATE SET value = excluded.value",
            (key, str(value)),
        )

    def add_row(self, row, input_row_number, args):
        label = (row.get(args.label_column) or "").strip()
        if not label:
            raise ValueError(
                "Missing '{}' at {} row {}".format(args.label_column, args.input, input_row_number)
            )
        display_name = row.get(args.display_name_column, "") or ""
        parent_label = row.get(args.parent_label_column, "") or ""
        parent_display_name = row.get(args.parent_display_name_column, "") or ""
        self.connection.execute(
            "INSERT INTO classes (label, display_name, parent_label, parent_display_name, input_count) "
            "VALUES (?, ?, ?, ?, 1) "
            "ON CONFLICT(label) DO UPDATE SET "
            "display_name = CASE WHEN classes.display_name = '' THEN excluded.display_name ELSE classes.display_name END, "
            "parent_label = CASE WHEN classes.parent_label = '' THEN excluded.parent_label ELSE classes.parent_label END, "
            "parent_display_name = CASE WHEN classes.parent_display_name = '' THEN excluded.parent_display_name ELSE classes.parent_display_name END, "
            "input_count = classes.input_count + 1",
            (label, display_name, parent_label, parent_display_name),
        )
        rank = random_rank(label, row.get(args.image_id_column, "") or "", input_row_number, args.random_seed)
        candidate_count = self.connection.execute(
            "SELECT COUNT(*) FROM candidates WHERE label = ?", (label,)
        ).fetchone()[0]
        if candidate_count < args.samples_per_class:
            self.connection.execute(
                "INSERT INTO candidates (label, random_rank, input_row_number) VALUES (?, ?, ?)",
                (label, rank, input_row_number),
            )
            return
        worst_candidate = self.connection.execute(
            "SELECT random_rank, input_row_number FROM candidates WHERE label = ? "
            "ORDER BY random_rank DESC, input_row_number DESC LIMIT 1",
            (label,),
        ).fetchone()
        if (rank, input_row_number) < (worst_candidate[0], worst_candidate[1]):
            self.connection.execute(
                "DELETE FROM candidates WHERE label = ? AND input_row_number = ?",
                (label, worst_candidate[1]),
            )
            self.connection.execute(
                "INSERT INTO candidates (label, random_rank, input_row_number) VALUES (?, ?, ?)",
                (label, rank, input_row_number),
            )

    def selected_row_numbers(self):
        return {
            row[0]
            for row in self.connection.execute("SELECT input_row_number FROM candidates")
        }

    def class_counts(self, samples_per_class, random_seed):
        query = (
            "SELECT classes.label, classes.display_name, classes.parent_label, "
            "classes.parent_display_name, classes.input_count, COUNT(candidates.input_row_number) "
            "FROM classes LEFT JOIN candidates ON classes.label = candidates.label "
            "GROUP BY classes.label, classes.display_name, classes.parent_label, "
            "classes.parent_display_name, classes.input_count "
            "ORDER BY classes.parent_display_name, classes.display_name, classes.label"
        )
        for row in self.connection.execute(query):
            yield {
                "LabelName": row[0],
                "DisplayName": row[1],
                "ParentLabelName": row[2],
                "ParentDisplayName": row[3],
                "InputSampleCount": row[4],
                "SelectedSampleCount": row[5],
                "SamplesPerClass": samples_per_class,
                "RandomSeed": random_seed,
            }

    def commit(self):
        self.connection.commit()

    def close(self):
        self.connection.close()


def validate_state(output_path, counts_output_path, checkpoint_path, overwrite, resume):
    if overwrite:
        output_path.unlink(missing_ok=True)
        counts_output_path.unlink(missing_ok=True)
        checkpoint_path.unlink(missing_ok=True)
        checkpoint_path.with_name(checkpoint_path.name + "-wal").unlink(missing_ok=True)
        checkpoint_path.with_name(checkpoint_path.name + "-shm").unlink(missing_ok=True)
        return
    existing_outputs = [path for path in (output_path, counts_output_path) if path.exists()]
    if existing_outputs and not resume:
        raise FileExistsError(
            "Output already exists: {}. Use --resume or --overwrite.".format(existing_outputs[0])
        )
    if checkpoint_path.exists() and not resume:
        raise FileExistsError(
            "Checkpoint database exists: {}. Use --resume or --overwrite.".format(checkpoint_path)
        )


def scan_input(input_path, checkpoint, args, total_rows):
    next_input_row = int(checkpoint.get_state("next_input_row", "2"))
    if checkpoint.get_state("scan_complete", "0") == "1":
        return 0
    newly_scanned = 0
    with input_path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        with tqdm(
            total=total_rows,
            initial=max(0, next_input_row - 2),
            desc="Selecting random samples",
            unit="row",
            mininterval=args.progress_refresh_seconds,
        ) as progress:
            for input_row_number, row in enumerate(reader, start=2):
                if input_row_number < next_input_row:
                    continue
                checkpoint.add_row(row, input_row_number, args)
                newly_scanned += 1
                progress.update(1)
                if newly_scanned % args.commit_rows == 0:
                    checkpoint.set_state("next_input_row", input_row_number + 1)
                    checkpoint.commit()
            checkpoint.set_state("next_input_row", total_rows + 2)
            checkpoint.set_state("scan_complete", "1")
            checkpoint.commit()
    return newly_scanned


def write_sampled_rows(input_path, output_path, selected_rows, fields, total_rows, refresh_seconds):
    temporary_output = temporary_path(output_path)
    temporary_output.unlink(missing_ok=True)
    with input_path.open("r", encoding="utf-8-sig", newline="") as input_handle:
        reader = csv.DictReader(input_handle)
        with temporary_output.open("w", encoding="utf-8-sig", newline="") as output_handle:
            writer = csv.DictWriter(output_handle, fieldnames=fields)
            writer.writeheader()
            with tqdm(
                total=total_rows,
                desc="Writing sampled rows",
                unit="row",
                mininterval=refresh_seconds,
            ) as progress:
                for input_row_number, row in enumerate(reader, start=2):
                    if input_row_number in selected_rows:
                        writer.writerow({field: row.get(field, "") for field in fields})
                    progress.update(1)
    temporary_output.replace(output_path)


def write_class_counts(output_path, checkpoint, args):
    temporary_output = temporary_path(output_path)
    temporary_output.unlink(missing_ok=True)
    with temporary_output.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=COUNT_FIELDS)
        writer.writeheader()
        for row in checkpoint.class_counts(args.samples_per_class, args.random_seed):
            writer.writerow(row)
    temporary_output.replace(output_path)


def remove_checkpoint(path):
    path.unlink(missing_ok=True)
    path.with_name(path.name + "-wal").unlink(missing_ok=True)
    path.with_name(path.name + "-shm").unlink(missing_ok=True)


def main():
    args = parse_args()
    input_path = args.input.expanduser()
    output_path = args.output.expanduser()
    counts_output_path = args.counts_output.expanduser()
    checkpoint_path = (
        args.checkpoint_db.expanduser()
        if args.checkpoint_db
        else output_path.parent / ".{}_checkpoint.sqlite3".format(output_path.stem)
    )
    if not input_path.is_file():
        raise FileNotFoundError("Input CSV does not exist: {}".format(input_path))
    fields = read_header(input_path)
    required_columns = [args.label_column, args.image_id_column]
    missing_columns = [column for column in required_columns if column not in fields]
    if missing_columns:
        raise ValueError("Input CSV is missing required columns: {}".format(", ".join(missing_columns)))
    validate_state(output_path, counts_output_path, checkpoint_path, args.overwrite, args.resume)
    total_rows = count_rows(input_path, args.progress_refresh_seconds)
    checkpoint = SamplingCheckpoint(checkpoint_path)
    try:
        newly_scanned = scan_input(input_path, checkpoint, args, total_rows)
        selected_rows = checkpoint.selected_row_numbers()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        counts_output_path.parent.mkdir(parents=True, exist_ok=True)
        write_sampled_rows(
            input_path,
            output_path,
            selected_rows,
            fields,
            total_rows,
            args.progress_refresh_seconds,
        )
        write_class_counts(counts_output_path, checkpoint, args)
        class_count = checkpoint.connection.execute("SELECT COUNT(*) FROM classes").fetchone()[0]
    finally:
        checkpoint.close()
    print("Input rows: {}".format(total_rows))
    print("Newly scanned rows: {}".format(newly_scanned))
    print("Leaf classes: {}".format(class_count))
    print("Exact samples for sufficiently large classes: {}".format(args.samples_per_class))
    print("Sampled rows: {}".format(len(selected_rows)))
    print("Sampled rows CSV: {}".format(output_path.resolve()))
    print("Per-class counts CSV: {}".format(counts_output_path.resolve()))
    if args.cleanup_checkpoint:
        remove_checkpoint(checkpoint_path)
        print("Removed checkpoint database: {}".format(checkpoint_path.resolve()))
    else:
        print("Checkpoint database: {}".format(checkpoint_path.resolve()))


if __name__ == "__main__":
    main()
