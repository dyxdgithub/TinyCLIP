"""Build separate captioned test/validation CSVs in the AllSet format.

Only positive image-level labels (Confidence=1) are included. Each non-empty
Localized Narratives caption becomes one output row. LabelNames and
DisplayNames are aligned JSON arrays, including [] for unlabeled images.
The SQLite index and completed CSV parts allow an interrupted run to resume.
"""

import argparse
import csv
import json
import shutil
import sqlite3
from pathlib import Path

from tqdm import tqdm


META_DIR = Path(__file__).resolve().parent.parent
DEFAULT_SOURCE_ROOT = Path("E:/数据集/Open Image")
DEFAULT_OUTPUT_DIR = META_DIR / "AllSet"
SPLIT_DIRS = {"test": "test", "validation": "val"}
IMAGE_FIELDS = (
    "ImageID", "Subset", "OriginalURL", "OriginalLandingURL", "License",
    "AuthorProfileURL", "Author", "Title", "OriginalSize", "OriginalMD5",
    "Thumbnail300KURL", "Rotation",
)
OUTPUT_FIELDS = IMAGE_FIELDS + ("caption", "voice_recording", "LabelNames", "DisplayNames")


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, default=DEFAULT_SOURCE_ROOT,
                        help="Directory containing test/, val/, and Hierarchy for 600 boxable classes/. Default: %(default)s")
    parser.add_argument("--class-names", type=Path,
                        help="LabelName/DisplayName CSV. Default: Class Names.csv under --source-root's hierarchy directory.")
    parser.add_argument("--splits", nargs="+", choices=("test", "validation"), default=("test", "validation"),
                        help="Splits to write as separate files. Accepted: test, validation. Default: test validation.")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR,
                        help="Directory for separate test and validation CSVs with the training file's 16 columns. Default: %(default)s")
    parser.add_argument("--checkpoint-dir", type=Path,
                        help="Persistent SQLite index and CSV parts. Default: hidden directory inside --output-dir.")
    parser.add_argument("--rows-per-part", type=int, default=10000,
                        help="Image IDs rows per resumable CSV part. Default: %(default)s")
    parser.add_argument("--checkpoint-interval", type=int, default=5000,
                        help="Source rows between committed index checkpoints. Default: %(default)s")
    parser.add_argument("--progress-refresh-seconds", type=float, default=0.1,
                        help="Minimum seconds between progress updates. Default: %(default)s")
    args = parser.parse_args(argv)
    if args.rows_per_part <= 0 or args.checkpoint_interval <= 0 or args.progress_refresh_seconds <= 0:
        parser.error("--rows-per-part, --checkpoint-interval, and --progress-refresh-seconds must be positive")
    if len(set(args.splits)) != len(args.splits):
        parser.error("--splits must not repeat a split")
    return args


def source_files(source_root, split):
    directory = source_root / SPLIT_DIRS[split]
    return {
        "images": directory / "Image IDs.csv",
        "labels": directory / "Image labels.csv",
        "narratives": directory / "Localized narratives.jsonl",
    }


def require_columns(path, expected):
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        fields = csv.DictReader(handle).fieldnames or []
    missing = set(expected) - set(fields)
    if missing:
        raise ValueError("{} is missing columns: {}".format(path, ", ".join(sorted(missing))))


def input_signature(paths, args, outputs):
    sources = []
    for path in paths:
        if not path.is_file():
            raise FileNotFoundError("Required metadata file does not exist: {}".format(path))
        stat = path.stat()
        sources.append((str(path.resolve()), stat.st_size, stat.st_mtime_ns))
    return json.dumps({
        "sources": sources,
        "splits": list(args.splits),
        "outputs": {split: str(path) for split, path in outputs.items()},
        "rows_per_part": args.rows_per_part,
    }, ensure_ascii=False, sort_keys=True)


def open_checkpoint(path, signature):
    path.parent.mkdir(parents=True, exist_ok=True)
    db = sqlite3.connect(path)
    db.execute("CREATE TABLE IF NOT EXISTS metadata (key TEXT PRIMARY KEY, value TEXT NOT NULL)")
    db.execute("CREATE TABLE IF NOT EXISTS progress (stage TEXT PRIMARY KEY, position INTEGER NOT NULL, count INTEGER NOT NULL, done INTEGER NOT NULL)")
    db.execute("CREATE TABLE IF NOT EXISTS labels (split TEXT NOT NULL, image_id TEXT NOT NULL, label_name TEXT NOT NULL, display_name TEXT NOT NULL, PRIMARY KEY (split, image_id, label_name)) WITHOUT ROWID")
    db.execute("CREATE TABLE IF NOT EXISTS captions (split TEXT NOT NULL, image_id TEXT NOT NULL, ordinal INTEGER NOT NULL, caption TEXT NOT NULL, voice_recording TEXT NOT NULL, PRIMARY KEY (split, image_id, ordinal)) WITHOUT ROWID")
    current = db.execute("SELECT value FROM metadata WHERE key = 'input_signature'").fetchone()
    if current and current[0] != signature:
        db.close()
        raise ValueError("Source files or settings changed; choose a new --checkpoint-dir and --output-dir")
    if current is None:
        db.execute("INSERT INTO metadata VALUES ('input_signature', ?)", (signature,))
    db.commit()
    return db


def stage_state(db, stage):
    return db.execute("SELECT position, count, done FROM progress WHERE stage = ?", (stage,)).fetchone() or (0, 0, 0)


def save_stage(db, stage, position, count, done=False):
    db.execute("INSERT INTO progress VALUES (?, ?, ?, ?) ON CONFLICT(stage) DO UPDATE SET position = excluded.position, count = excluded.count, done = excluded.done",
               (stage, position, count, int(done)))
    db.commit()


def load_class_names(path):
    require_columns(path, ("LabelName", "DisplayName"))
    names = {}
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            label = (row["LabelName"] or "").strip()
            display = row["DisplayName"] or ""
            if label in names and names[label] != display:
                raise ValueError("Conflicting display names for {} in {}".format(label, path))
            if label:
                names[label] = display
    return names


def index_labels(db, path, split, class_names, interval, refresh_seconds):
    stage = split + "_labels"
    completed, _, done = stage_state(db, stage)
    if done:
        print("Reusing {} label index".format(split), flush=True)
        return
    require_columns(path, ("ImageID", "LabelName", "Confidence"))
    batch = []
    count = completed
    with path.open("r", encoding="utf-8-sig", newline="") as handle, \
            tqdm(desc="Indexing {} labels (total unknown)".format(split), unit="row",
                 initial=completed, mininterval=refresh_seconds) as progress:
        for count, row in enumerate(csv.DictReader(handle), start=1):
            if count <= completed:
                continue
            image_id = (row["ImageID"] or "").strip()
            label = (row["LabelName"] or "").strip()
            confidence = (row["Confidence"] or "").strip()
            if not image_id or not label:
                raise ValueError("Empty ImageID or LabelName in {} row {}".format(path, count + 1))
            if confidence in ("1", "1.0"):
                batch.append((split, image_id, label, class_names.get(label, "")))
            if count % interval == 0:
                db.executemany("INSERT OR IGNORE INTO labels VALUES (?, ?, ?, ?)", batch)
                save_stage(db, stage, count, count)
                batch.clear()
            progress.update(1)
        db.executemany("INSERT OR IGNORE INTO labels VALUES (?, ?, ?, ?)", batch)
        save_stage(db, stage, count, count, done=True)


def index_narratives(db, path, split, interval, refresh_seconds):
    stage = split + "_narratives"
    offset, count, done = stage_state(db, stage)
    if done:
        print("Reusing {} caption index".format(split), flush=True)
        return
    batch = []
    total_bytes = path.stat().st_size
    with path.open("rb") as handle, tqdm(total=total_bytes, initial=offset,
            desc="Indexing {} narratives".format(split), unit="B", unit_scale=True,
            mininterval=refresh_seconds) as progress:
        handle.seek(offset)
        while raw_line := handle.readline():
            count += 1
            try:
                annotation = json.loads(raw_line.decode("utf-8-sig" if count == 1 else "utf-8"))
            except (UnicodeDecodeError, json.JSONDecodeError) as error:
                raise ValueError("Invalid JSON in {} line {}: {}".format(path, count, error)) from error
            if not isinstance(annotation, dict):
                raise ValueError("Expected JSON object in {} line {}".format(path, count))
            image_id = str(annotation.get("image_id") or "").strip()
            caption = str(annotation.get("caption") or "").strip()
            if caption:
                if not image_id:
                    raise ValueError("Caption without image_id in {} line {}".format(path, count))
                batch.append((split, image_id, count, caption, str(annotation.get("voice_recording") or "")))
            if count % interval == 0:
                db.executemany("INSERT OR IGNORE INTO captions VALUES (?, ?, ?, ?, ?)", batch)
                save_stage(db, stage, handle.tell(), count)
                batch.clear()
            progress.update(len(raw_line))
        db.executemany("INSERT OR IGNORE INTO captions VALUES (?, ?, ?, ?, ?)", batch)
        save_stage(db, stage, handle.tell(), count, done=True)


def part_path(checkpoint_dir, split, number):
    return checkpoint_dir / "{}_part_{:06d}.csv".format(split, number)


def write_part(path, rows, split, db):
    temporary = path.with_name(path.name + ".inprogress")
    temporary.unlink(missing_ok=True)
    label_cursor = db.cursor()
    caption_cursor = db.cursor()
    try:
        with temporary.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=OUTPUT_FIELDS)
            writer.writeheader()
            for row in rows:
                image_id = (row["ImageID"] or "").strip()
                if not image_id or (row["Subset"] or "").strip().lower() != split:
                    raise ValueError("Image IDs.csv has a missing ImageID or unexpected Subset: {}".format(row))
                labels = label_cursor.execute(
                    "SELECT label_name, display_name FROM labels WHERE split = ? AND image_id = ? ORDER BY label_name, display_name",
                    (split, image_id),
                ).fetchall()
                names = json.dumps([label for label, _ in labels], ensure_ascii=False, separators=(",", ":"))
                displays = json.dumps([display for _, display in labels], ensure_ascii=False, separators=(",", ":"))
                for caption, voice in caption_cursor.execute(
                    "SELECT caption, voice_recording FROM captions WHERE split = ? AND image_id = ? ORDER BY ordinal",
                    (split, image_id),
                ):
                    output = {field: row[field] for field in IMAGE_FIELDS}
                    output.update(caption=caption, voice_recording=voice,
                                  LabelNames=names, DisplayNames=displays)
                    writer.writerow(output)
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def build_parts(db, images_path, split, checkpoint_dir, rows_per_part, refresh_seconds):
    require_columns(images_path, IMAGE_FIELDS)
    parts = []
    with images_path.open("r", encoding="utf-8-sig", newline="") as handle, \
            tqdm(desc="Converting {} images (total unknown)".format(split), unit="image",
                 mininterval=refresh_seconds) as progress:
        reader = csv.DictReader(handle)
        part_number = 0
        while True:
            rows = []
            for _ in range(rows_per_part):
                row = next(reader, None)
                if row is None:
                    break
                rows.append(row)
            if not rows:
                break
            part_number += 1
            path = part_path(checkpoint_dir, split, part_number)
            if not path.is_file():
                write_part(path, rows, split, db)
            parts.append(path)
            progress.update(len(rows))
            progress.set_postfix(parts=part_number)
    return parts


def combine_parts(parts, output, refresh_seconds):
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(output.name + ".inprogress")
    temporary.unlink(missing_ok=True)
    try:
        with temporary.open("w", encoding="utf-8-sig", newline="") as target:
            target.write(",".join(OUTPUT_FIELDS) + "\n")
            with tqdm(total=len(parts), desc="Combining CSV parts", unit="part",
                      mininterval=refresh_seconds) as progress:
                for path in parts:
                    with path.open("r", encoding="utf-8", newline="") as source:
                        next(source)
                        shutil.copyfileobj(source, target)
                    progress.update(1)
        temporary.replace(output)
    finally:
        temporary.unlink(missing_ok=True)


def main(argv=None):
    args = parse_args(argv)
    source_root = args.source_root.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    outputs = {
        split: output_dir / "Image IDs_with_captions_and_labels_captioned_only_{}.csv".format(split)
        for split in args.splits
    }
    checkpoint_dir = (args.checkpoint_dir.expanduser().resolve() if args.checkpoint_dir
                      else output_dir / ("." + "_".join(args.splits) + "_captioned_metadata_parts"))
    class_names_path = (args.class_names.expanduser().resolve() if args.class_names
                        else source_root / "Hierarchy for 600 boxable classes" / "Class Names.csv")
    files = {split: source_files(source_root, split) for split in args.splits}
    inputs = [class_names_path] + [path for split in args.splits for path in files[split].values()]
    if any(output in inputs for output in outputs.values()):
        raise ValueError("Output CSV paths must differ from every input file")
    signature = input_signature(inputs, args, outputs)
    db_path = checkpoint_dir / "metadata_index.sqlite3"
    if not db_path.exists():
        for output in outputs.values():
            if output.exists():
                raise FileExistsError("Output exists without its checkpoint: {}".format(output))
    db = open_checkpoint(db_path, signature)
    try:
        class_names = load_class_names(class_names_path)
        for split in args.splits:
            output = outputs[split]
            stage = "final_done_" + split
            if db.execute("SELECT value FROM metadata WHERE key = ?", (stage,)).fetchone() and output.is_file():
                print("Already complete: {}".format(output), flush=True)
                continue
            print("Processing {} metadata".format(split), flush=True)
            index_labels(db, files[split]["labels"], split, class_names,
                         args.checkpoint_interval, args.progress_refresh_seconds)
            index_narratives(db, files[split]["narratives"], split,
                             args.checkpoint_interval, args.progress_refresh_seconds)
            parts = build_parts(db, files[split]["images"], split, checkpoint_dir,
                                args.rows_per_part, args.progress_refresh_seconds)
            combine_parts(parts, output, args.progress_refresh_seconds)
            db.execute("INSERT OR REPLACE INTO metadata VALUES (?, '1')", (stage,))
            db.commit()
            print("Wrote {} from {} completed parts".format(output, len(parts)), flush=True)
        print("Checkpoint directory: {}".format(checkpoint_dir), flush=True)
        return 0
    finally:
        db.close()


if __name__ == "__main__":
    raise SystemExit(main())
