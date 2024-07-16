"""Join the audio filename table with n2 training annotations by ImageID."""

import argparse
import csv
from collections import defaultdict
from pathlib import Path

from tqdm import tqdm


META_DIR = Path(__file__).resolve().parent
DEFAULT_AUDIO_TABLE = META_DIR / "audio_filename_table.csv"
DEFAULT_ANNOTATIONS = (
    META_DIR / "Hierarchy" / "n2" / "train_annotations_valid_n2.csv"
)
DEFAULT_OUTPUT = META_DIR / "audio_filename_annotations_n2.csv"


def read_annotation_rows(annotation_path):
    rows_by_id = defaultdict(list)
    with Path(annotation_path).open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        fields = reader.fieldnames or []
        if "ImageID" not in fields:
            raise ValueError("Annotation CSV must contain ImageID")
        for row in tqdm(reader, desc="Reading annotations", unit="row"):
            image_id = str(row.get("ImageID", "")).strip()
            if image_id:
                rows_by_id[image_id].append(row)
    return rows_by_id, fields


def join_tables(audio_table, annotations, output_path):
    annotations_by_id, annotation_fields = read_annotation_rows(annotations)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    matched_audio_rows = 0
    matched_ids = set()
    joined_rows = 0
    with Path(audio_table).open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        audio_fields = reader.fieldnames or []
        if "ImageID" not in audio_fields:
            raise ValueError("Audio filename CSV must contain ImageID")
        output_fields = audio_fields + [
            field for field in annotation_fields if field not in audio_fields
        ]
        with output_path.open("w", encoding="utf-8-sig", newline="") as output_handle:
            writer = csv.DictWriter(output_handle, fieldnames=output_fields)
            writer.writeheader()
            for audio_row in tqdm(reader, desc="Joining audio and annotations", unit="row"):
                image_id = str(audio_row.get("ImageID", "")).strip()
                annotation_rows = annotations_by_id.get(image_id, [])
                if not annotation_rows:
                    continue
                matched_audio_rows += 1
                matched_ids.add(image_id)
                for annotation_row in annotation_rows:
                    joined_row = dict(audio_row)
                    for field, value in annotation_row.items():
                        if field not in audio_fields:
                            joined_row[field] = value
                    writer.writerow(joined_row)
                    joined_rows += 1
    print("Audio rows with matching annotations: {}".format(matched_audio_rows))
    print("Common ImageIDs: {}".format(len(matched_ids)))
    print("Joined rows: {}".format(joined_rows))
    print("Output: {}".format(output_path.resolve()))
    return joined_rows


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audio-table", type=Path, default=DEFAULT_AUDIO_TABLE)
    parser.add_argument("--annotations", type=Path, default=DEFAULT_ANNOTATIONS)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def main():
    args = parse_args()
    join_tables(args.audio_table, args.annotations, args.output)


if __name__ == "__main__":
    main()
