import csv
from collections import defaultdict
from pathlib import Path

base = Path(__file__).parent
audio_path = base / "audio_captions_n4.csv"
train_path = base / "train_annotations_valid_n4.csv"
output_path = base / "train_annotations_valid_n4_with_caption.csv"

captions = defaultdict(list)
with audio_path.open("r", encoding="utf-8-sig", newline="") as source:
    for row in csv.DictReader(source):
        if row.get("Caption", "").strip():
            captions[row["ImageID"]].append(row["Caption"])

with train_path.open("r", encoding="utf-8-sig", newline="") as source, output_path.open(
    "w", encoding="utf-8-sig", newline=""
) as target:
    reader = csv.DictReader(source)
    fieldnames = reader.fieldnames + ["Caption"]
    writer = csv.DictWriter(target, fieldnames=fieldnames, lineterminator="\n")
    writer.writeheader()
    for row in reader:
        row["Caption"] = " || ".join(captions.get(row["ImageID"], []))
        writer.writerow(row)

print(f"Wrote {output_path}")
