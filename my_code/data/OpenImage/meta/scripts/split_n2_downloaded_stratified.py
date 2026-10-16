"""Split downloaded n2 rows into train/validation/test at about 8:1:1.

All rows sharing an ImageID receive the same split. A deterministic greedy
multi-label assignment balances overall row counts and each parent/leaf class.
Small and overlapping classes may prevent exact per-class ratios; the summary
CSV records every target and actual count.
"""

import argparse
import csv
import json
import math
import random
import sqlite3
from collections import Counter, defaultdict
from pathlib import Path

from tqdm import tqdm


META_DIR = Path(__file__).resolve().parent.parent
N2_DIR = META_DIR / "Hierarchy" / "n2"
DEFAULT_INPUT = N2_DIR / "Image_IDs_n2_all_downloaded.csv"
DEFAULT_OUTPUT = N2_DIR / "Image_IDs_n2_all_downloaded_split_8_1_1.csv"
DEFAULT_SUMMARY = N2_DIR / "Image_IDs_n2_all_downloaded_split_8_1_1_summary.csv"
SPLITS = ("train", "validation", "test")
SUMMARY_FIELDS = (
    "ParentLabelName", "ParentDisplayName", "LabelName", "DisplayName",
    "TotalRows", "TargetTrain", "ActualTrain", "TargetValidation",
    "ActualValidation", "TargetTest", "ActualTest", "TrainRatio",
    "ValidationRatio", "TestRatio",
)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT,
                        help="Downloaded sample CSV. Default: %(default)s")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT,
                        help="Input rows plus Split column. Default: %(default)s")
    parser.add_argument("--summary-output", type=Path, default=DEFAULT_SUMMARY,
                        help="Overall and per-class target/actual counts. Default: %(default)s")
    parser.add_argument("--checkpoint-db", type=Path, default=None,
                        help="SQLite resume checkpoint. Default: hidden database beside --output")
    parser.add_argument("--seed", type=int, default=42,
                        help="Deterministic ordering seed; any integer is accepted. Default: %(default)s")
    parser.add_argument("--commit-interval", type=int, default=1000,
                        help="Persist assignments every N unique ImageIDs. Default: %(default)s")
    parser.add_argument("--progress-refresh-seconds", type=float, default=0.1,
                        help="Minimum seconds between progress updates. Default: %(default)s")
    parser.add_argument("--resume", action="store_true",
                        help="Continue an interrupted checkpoint without reassigning completed ImageIDs. Default: disabled")
    parser.add_argument("--overwrite-state", action="store_true",
                        help="Discard this script's checkpoint and outputs, then split again. Input CSV is preserved. Default: disabled")
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


def integer_targets(count):
    """Use largest remainders to turn 80/10/10 into counts summing to count."""
    weights = (8, 1, 1)
    result = [count * weight // 10 for weight in weights]
    remainders = [count * weight % 10 for weight in weights]
    order = sorted(range(3), key=lambda index: (-remainders[index], index))
    for index in order[:count - sum(result)]:
        result[index] += 1
    return result


def load_groups(path, refresh_seconds):
    groups = defaultdict(Counter)
    class_index = {}
    class_names = []
    class_totals = Counter()
    row_total = 0
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        required = {"ImageID", "ParentLabelName", "LabelName",
                    "ParentDisplayName", "DisplayName"}
        missing = required.difference(reader.fieldnames or [])
        if missing:
            raise ValueError("Input CSV lacks columns: {}".format(", ".join(sorted(missing))))
        if "Split" in reader.fieldnames:
            raise ValueError("Input CSV already has a Split column")
        with tqdm(desc="Reading downloaded rows", unit="row",
                  mininterval=refresh_seconds) as progress:
            for row in reader:
                image_id = (row["ImageID"] or "").strip()
                parent = (row["ParentLabelName"] or "").strip()
                leaf = (row["LabelName"] or "").strip()
                if not image_id or not parent or not leaf:
                    raise ValueError("Input row {} lacks ImageID or class labels".format(row_total + 2))
                key = (parent, leaf)
                if key not in class_index:
                    class_index[key] = len(class_names)
                    class_names.append((parent, row["ParentDisplayName"],
                                        leaf, row["DisplayName"]))
                index = class_index[key]
                groups[image_id][index] += 1
                class_totals[index] += 1
                row_total += 1
                progress.update(1)
    if row_total == 0:
        raise ValueError("Input CSV contains no data rows")
    return groups, class_names, class_totals, row_total


def open_store(path, settings):
    path.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(path)
    connection.execute("PRAGMA journal_mode=WAL")
    connection.execute("CREATE TABLE IF NOT EXISTS settings (key TEXT PRIMARY KEY, value TEXT NOT NULL)")
    connection.execute(
        "CREATE TABLE IF NOT EXISTS assignments (image_id TEXT PRIMARY KEY, split TEXT NOT NULL)"
    )
    stored = dict(connection.execute("SELECT key, value FROM settings"))
    if stored and stored != settings:
        connection.close()
        raise ValueError("Input or seed changed; use --overwrite-state for a new split")
    if not stored:
        connection.executemany("INSERT INTO settings VALUES (?, ?)", settings.items())
    connection.commit()
    return connection


def current_counts(assignments, groups, class_count):
    overall = [0, 0, 0]
    by_class = [[0, 0, 0] for _ in range(class_count)]
    ids = [0, 0, 0]
    split_index = {name: index for index, name in enumerate(SPLITS)}
    for image_id, name in assignments.items():
        if image_id not in groups or name not in split_index:
            raise ValueError("Checkpoint assignment is absent from the current input")
        index = split_index[name]
        ids[index] += 1
        for class_id, count in groups[image_id].items():
            by_class[class_id][index] += count
            overall[index] += count
    return overall, by_class, ids


def squared_change(current, target, added):
    before = current - target
    return ((before + added) ** 2 - before ** 2) / max(target, 1)


def choose_split(labels, overall, by_class, overall_target, class_targets, class_totals):
    row_weight = sum(labels.values())
    feasible = [index for index in range(3)
                if overall[index] + row_weight <= overall_target[index]]
    candidates = feasible or list(range(3))
    scores = []
    for index in candidates:
        score = 0.15 * squared_change(overall[index], overall_target[index], row_weight)
        for class_id, count in labels.items():
            score += (
                squared_change(by_class[class_id][index], class_targets[class_id][index], count)
                / math.sqrt(class_totals[class_id])
            )
        overflow = max(0, overall[index] + row_weight - overall_target[index])
        scores.append((overflow, score, -overall_target[index] + overall[index], index))
    return min(scores)[-1]


def assign_groups(connection, groups, class_totals, row_total, seed, interval, refresh_seconds):
    assignments = dict(connection.execute("SELECT image_id, split FROM assignments"))
    overall, by_class, id_counts = current_counts(assignments, groups, len(class_totals))
    overall_target = integer_targets(row_total)
    class_targets = [integer_targets(class_totals[index]) for index in range(len(class_totals))]
    order = list(groups)
    random.Random(seed).shuffle(order)
    order.sort(key=lambda image_id: (
        min(class_totals[index] for index in groups[image_id]),
        -len(groups[image_id]),
    ))
    new_count = 0
    with tqdm(total=len(order), initial=len(assignments), desc="Assigning ImageIDs",
              unit="ID", mininterval=refresh_seconds) as progress:
        for image_id in order:
            if image_id in assignments:
                continue
            labels = groups[image_id]
            index = choose_split(labels, overall, by_class, overall_target,
                                 class_targets, class_totals)
            split = SPLITS[index]
            connection.execute("INSERT INTO assignments VALUES (?, ?)", (image_id, split))
            assignments[image_id] = split
            id_counts[index] += 1
            for class_id, count in labels.items():
                by_class[class_id][index] += count
                overall[index] += count
            new_count += 1
            if new_count % interval == 0:
                connection.commit()
                progress.set_postfix(train=overall[0], val=overall[1], test=overall[2])
            progress.update(1)
    connection.commit()
    if sum(overall) != row_total or len(assignments) != len(groups):
        raise RuntimeError("Not all input rows or ImageIDs were assigned")
    return assignments, overall, by_class, id_counts, overall_target, class_targets


def export_split(input_path, output, assignments):
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(output.name + ".inprogress")
    try:
        with input_path.open("r", encoding="utf-8-sig", newline="") as source, \
                temporary.open("w", encoding="utf-8-sig", newline="") as destination:
            reader = csv.DictReader(source)
            writer = csv.DictWriter(destination, fieldnames=reader.fieldnames + ["Split"])
            writer.writeheader()
            for row in reader:
                row["Split"] = assignments[row["ImageID"].strip()]
                writer.writerow(row)
        temporary.replace(output)
    finally:
        temporary.unlink(missing_ok=True)


def export_summary(output, class_names, class_totals, overall, by_class,
                   overall_target, class_targets):
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(output.name + ".inprogress")
    try:
        with temporary.open("w", encoding="utf-8-sig", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(SUMMARY_FIELDS)

            def write_row(names, total, target, actual):
                writer.writerow([
                    *names, total, target[0], actual[0], target[1], actual[1],
                    target[2], actual[2], *("{:.6f}".format(value / total) for value in actual),
                ])

            write_row(("__ALL__", "All classes", "", ""), sum(overall),
                      overall_target, overall)
            for index in sorted(range(len(class_names)), key=lambda item: class_names[item]):
                write_row(class_names[index], class_totals[index],
                          class_targets[index], by_class[index])
        temporary.replace(output)
    finally:
        temporary.unlink(missing_ok=True)


def main():
    args = parse_args()
    input_path = args.input.expanduser().resolve()
    output = args.output.expanduser().resolve()
    summary = args.summary_output.expanduser().resolve()
    database = checkpoint_path(args)
    if not input_path.is_file():
        raise FileNotFoundError("Input CSV does not exist: {}".format(input_path))
    if len({input_path, output, summary, database}) != 4:
        raise ValueError("Input, output, summary, and checkpoint paths must be distinct")
    if args.overwrite_state:
        remove_checkpoint(database)
        output.unlink(missing_ok=True)
        summary.unlink(missing_ok=True)
    elif (database.exists() or output.exists() or summary.exists()) and not args.resume:
        raise FileExistsError("Previous output or checkpoint exists; use --resume or --overwrite-state")
    if args.resume and not database.exists():
        raise FileNotFoundError("No checkpoint exists to resume: {}".format(database))
    groups, class_names, class_totals, row_total = load_groups(
        input_path, args.progress_refresh_seconds
    )
    input_stat = input_path.stat()
    settings = {
        "input": str(input_path),
        "input_size": str(input_stat.st_size),
        "input_mtime_ns": str(input_stat.st_mtime_ns),
        "seed": str(args.seed),
        "algorithm": "greedy_multilabel_v1",
    }
    connection = open_store(database, settings)
    try:
        results = assign_groups(connection, groups, class_totals, row_total,
                                args.seed, args.commit_interval, args.progress_refresh_seconds)
        assignments, overall, by_class, id_counts, overall_target, class_targets = results
        export_split(input_path, output, assignments)
        export_summary(summary, class_names, class_totals, overall, by_class,
                       overall_target, class_targets)
        print("Rows by split: {}".format(dict(zip(SPLITS, overall))))
        print("Unique ImageIDs by split: {}".format(dict(zip(SPLITS, id_counts))))
        print("Target rows: {}".format(dict(zip(SPLITS, overall_target))))
        print("Split CSV: {}".format(output))
        print("Summary CSV: {}".format(summary))
        print("Checkpoint: {}".format(database))
    finally:
        connection.close()


if __name__ == "__main__":
    main()
