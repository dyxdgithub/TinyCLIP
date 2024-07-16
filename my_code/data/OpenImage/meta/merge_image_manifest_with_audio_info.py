"""Merge image download records with audio-image metadata by ImageID."""

import argparse
import csv
from collections import defaultdict
from pathlib import Path

from tqdm import tqdm


META_DIR = Path(__file__).resolve().parent
DEFAULT_SOURCE = META_DIR / "Hierarchy" / "n2" / "audio_image_n2.csv"
DEFAULT_MANIFEST = (
    META_DIR
    / "Hierarchy"
    / "n2"
    / "audio_image"
    / "images"
    / "image_download_manifest.csv"
)
DEFAULT_OUTPUT = (
    META_DIR
    / "Hierarchy"
    / "n2"
    / "audio_image"
    / "images"
    / "image_download_manifest_with_audio_info.csv"
)


def read_source_rows(source_path):
    rows_by_id = defaultdict(list)
    with Path(source_path).open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        fields = reader.fieldnames or []
        if "ImageID" not in fields:
            raise ValueError("Source CSV must contain ImageID")
        for row in tqdm(reader, desc="Reading source metadata", unit="row"):
            image_id = str(row.get("ImageID", "")).strip()
            if image_id:
                rows_by_id[image_id].append(row)
    return rows_by_id, fields


def merge_tables(source_path, manifest_path, output_path):
    source_rows_by_id, source_fields = read_source_rows(source_path)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    matched_manifest_rows = 0
    matched_ids = set()
    joined_rows = 0
    with Path(manifest_path).open(
        "r", encoding="utf-8-sig", newline=""
    ) as manifest_handle:
        reader = csv.DictReader(manifest_handle)
        manifest_fields = reader.fieldnames or []
        if "ImageID" not in manifest_fields:
            raise ValueError("Image manifest CSV must contain ImageID")
        output_fields = manifest_fields + [
            field for field in source_fields if field not in manifest_fields
        ]
        with output_path.open("w", encoding="utf-8-sig", newline="") as output_handle:
            writer = csv.DictWriter(output_handle, fieldnames=output_fields)
            writer.writeheader()
            for manifest_row in tqdm(
                reader, desc="Merging image manifest", unit="row"
            ):
                image_id = str(manifest_row.get("ImageID", "")).strip()
                source_rows = source_rows_by_id.get(image_id, [])
                if not source_rows:
                    continue
                matched_manifest_rows += 1
                matched_ids.add(image_id)
                for source_row in source_rows:
                    joined_row = dict(manifest_row)
                    for field, value in source_row.items():
                        if field not in manifest_fields:
                            joined_row[field] = value
                    writer.writerow(joined_row)
                    joined_rows += 1

    print("Manifest rows with matching metadata: {}".format(matched_manifest_rows))
    print("Common ImageIDs: {}".format(len(matched_ids)))
    print("Joined rows: {}".format(joined_rows))
    print("Output: {}".format(output_path.resolve()))
    return joined_rows


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def main():
    args = parse_args()
    merge_tables(args.source, args.manifest, args.output)


if __name__ == "__main__":
    main()
