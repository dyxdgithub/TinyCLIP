"""Select globally unique training pairs from a combined DINOv2 table.

All valid candidate rows are sorted by cosine similarity descending. A row is
selected only when neither ImageID has appeared in a previously selected pair.
The resulting table is self-contained: it includes local image paths and the
longest nonempty caption found for each ImageID.
"""

import argparse
import csv
import math
from dataclasses import dataclass
from pathlib import Path

from tqdm import tqdm


META_DIR = Path(__file__).resolve().parent.parent
SAMPLED_DIR = META_DIR / "Hierarchy" / "n2" / "300"
DEFAULT_SIMILARITY_CSV = SAMPLED_DIR / "dinov2_similarity_fiftyone_sampled_300_per_class" / "top_n_cross_leaf_dinov2_similarity.csv"
DEFAULT_MANIFEST = SAMPLED_DIR / "Image IDs_with_last_two_layers_multi_members_n2_one_label_per_parent_sampled_300_per_class.csv"
DEFAULT_IMAGES_ROOT = SAMPLED_DIR / "images_fiftyone_sampled_300_per_class"
DEFAULT_OUTPUT = SAMPLED_DIR / "selected_training_pairs_from_dinov2_similarity.csv"
PAIR_FIELDS = [
    "PairIndex", "ParentCategory", "AnchorLeafCategory", "AnchorImageID",
    "AnchorImagePath", "AnchorCaption", "HardNegativeLeafCategory",
    "HardNegativeImageID", "HardNegativeImagePath", "HardNegativeCaption",
    "CosineSimilarity",
]
REQUIRED_FIELDS = {
    "ParentCategory", "SourceLeafCategory", "SourceImageID", "SourceImagePath",
    "MatchedLeafCategory", "MatchedImageID", "MatchedImagePath",
    "CosineSimilarity",
}


@dataclass(frozen=True)
class Candidate:
    parent: str
    anchor_leaf: str
    anchor_id: str
    anchor_path: Path
    negative_leaf: str
    negative_id: str
    negative_path: Path
    similarity: float


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--similarity-csv", type=Path, default=DEFAULT_SIMILARITY_CSV, help="Combined DINOv2 similarity CSV. Default: %(default)s")
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST, help="CSV containing ImageID and captions. Default: %(default)s")
    parser.add_argument("--images-root", type=Path, default=DEFAULT_IMAGES_ROOT, help="Local parent/leaf image root used when table paths are unavailable. Default: %(default)s")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT, help="Selected training-pair CSV. Default: %(default)s")
    parser.add_argument("--image-id-column", default="ImageID", help="Manifest ImageID column. Default: %(default)s")
    parser.add_argument("--caption-column", default="caption", help="Manifest caption column. Default: %(default)s")
    parser.add_argument("--caption-conflict", choices=("error", "first", "longest"), default="longest", help="Caption conflict policy. longest selects the longest caption and keeps the first on ties. Default: %(default)s")
    parser.add_argument("--progress-refresh-seconds", type=float, default=0.1, help="tqdm refresh interval. Default: %(default)s")
    args = parser.parse_args()
    if args.progress_refresh_seconds <= 0:
        parser.error("--progress-refresh-seconds must be positive")
    return args


def headers(path):
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return csv.DictReader(handle).fieldnames or []


def count_rows(path, refresh_seconds):
    count = 0
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.reader(handle)
        next(reader, None)
        with tqdm(
            desc="Counting similarity rows",
            unit="row",
            mininterval=refresh_seconds,
        ) as progress:
            for count, _ in enumerate(reader, start=1):
                progress.update(1)
    return count


def load_captions(path, args):
    fields = headers(path)
    for field in (args.image_id_column, args.caption_column):
        if field not in fields:
            raise ValueError("Manifest lacks required column {!r}: {}".format(field, path))
    captions = {}
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        with tqdm(desc="Reading captions", unit="row", mininterval=args.progress_refresh_seconds) as progress:
            for row_number, row in enumerate(reader, start=2):
                image_id = (row.get(args.image_id_column) or "").strip()
                caption = (row.get(args.caption_column) or "").strip()
                if image_id and caption:
                    prior = captions.get(image_id)
                    if prior is None:
                        captions[image_id] = caption
                    elif prior != caption:
                        if args.caption_conflict == "error":
                            raise ValueError("Conflicting captions for ImageID {!r} at row {}".format(image_id, row_number))
                        if args.caption_conflict == "longest" and len(caption) > len(prior):
                            captions[image_id] = caption
                progress.update(1)
    return captions


def resolve_path(raw, parent, leaf, images_root):
    direct = Path((raw or "").strip())
    if direct.is_file():
        return direct.resolve()
    filename = Path(str(direct).replace("\\", "/")).name
    fallback = images_root / parent / leaf / filename
    return fallback.resolve() if filename and fallback.is_file() else direct


def load_candidates(path, args):
    fields = set(headers(path))
    missing = REQUIRED_FIELDS - fields
    if missing:
        raise ValueError("Similarity CSV lacks columns: {}".format(", ".join(sorted(missing))))
    total = count_rows(path, args.progress_refresh_seconds)
    candidates = []
    path_resolution = {"recorded": 0, "rebuilt": 0, "missing": 0}
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        with tqdm(total=max(total, 0), desc="Reading similarity rows", unit="row", mininterval=args.progress_refresh_seconds) as progress:
            for row_number, row in enumerate(reader, start=2):
                try:
                    similarity = float((row.get("CosineSimilarity") or "").strip())
                except ValueError as error:
                    raise ValueError("Invalid CosineSimilarity at {}:{}".format(path, row_number)) from error
                parent = (row.get("ParentCategory") or "").strip()
                anchor_leaf = (row.get("SourceLeafCategory") or "").strip()
                negative_leaf = (row.get("MatchedLeafCategory") or "").strip()
                anchor_id = (row.get("SourceImageID") or "").strip()
                negative_id = (row.get("MatchedImageID") or "").strip()
                recorded_anchor_path = Path((row.get("SourceImagePath") or "").strip())
                recorded_negative_path = Path((row.get("MatchedImagePath") or "").strip())
                anchor_path = resolve_path(
                    row.get("SourceImagePath"), parent, anchor_leaf, args.images_root
                )
                negative_path = resolve_path(
                    row.get("MatchedImagePath"), parent, negative_leaf, args.images_root
                )
                for recorded, resolved in (
                    (recorded_anchor_path, anchor_path),
                    (recorded_negative_path, negative_path),
                ):
                    if recorded.is_file():
                        path_resolution["recorded"] += 1
                    elif resolved.is_file():
                        path_resolution["rebuilt"] += 1
                    else:
                        path_resolution["missing"] += 1
                if math.isfinite(similarity) and similarity < 1.0 and anchor_id and negative_id and anchor_id != negative_id:
                    candidates.append(Candidate(parent, anchor_leaf, anchor_id, anchor_path, negative_leaf, negative_id, negative_path, similarity))
                progress.update(1)
    return candidates, path_resolution


def candidate_key(candidate):
    return (-candidate.similarity, candidate.anchor_id, candidate.negative_id, str(candidate.anchor_path), str(candidate.negative_path))


def select_pairs(candidates, captions):
    selected = []
    used = set()
    for candidate in sorted(candidates, key=candidate_key):
        if candidate.anchor_id not in captions or candidate.negative_id not in captions:
            continue
        if not candidate.anchor_path.is_file() or not candidate.negative_path.is_file():
            continue
        if candidate.anchor_id in used or candidate.negative_id in used:
            continue
        used.update((candidate.anchor_id, candidate.negative_id))
        selected.append(candidate)
    if not selected:
        raise ValueError("No usable pairs remain after caption, path, similarity, and ImageID filtering")
    return selected


def write_output(path, pairs, captions):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".inprogress")
    temporary.unlink(missing_ok=True)
    with temporary.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=PAIR_FIELDS)
        writer.writeheader()
        for index, pair in enumerate(pairs, start=1):
            writer.writerow({
                "PairIndex": index,
                "ParentCategory": pair.parent,
                "AnchorLeafCategory": pair.anchor_leaf,
                "AnchorImageID": pair.anchor_id,
                "AnchorImagePath": str(pair.anchor_path),
                "AnchorCaption": captions[pair.anchor_id],
                "HardNegativeLeafCategory": pair.negative_leaf,
                "HardNegativeImageID": pair.negative_id,
                "HardNegativeImagePath": str(pair.negative_path),
                "HardNegativeCaption": captions[pair.negative_id],
                "CosineSimilarity": "{:.8f}".format(pair.similarity),
            })
    temporary.replace(path)


def main():
    args = parse_args()
    similarity_csv = args.similarity_csv.expanduser()
    manifest = args.manifest.expanduser()
    args.images_root = args.images_root.expanduser()
    output = args.output.expanduser()
    if not similarity_csv.is_file():
        raise FileNotFoundError("Similarity CSV does not exist: {}".format(similarity_csv))
    if not manifest.is_file():
        raise FileNotFoundError("Manifest does not exist: {}".format(manifest))
    captions = load_captions(manifest, args)
    candidates, path_resolution = load_candidates(similarity_csv, args)
    pairs = select_pairs(candidates, captions)
    write_output(output, pairs, captions)
    print("Candidate rows: {}".format(len(candidates)))
    print("Selected unique pairs: {}".format(len(pairs)))
    print("Image path resolution: {}".format(path_resolution))
    print("Output: {}".format(output.resolve()))


if __name__ == "__main__":
    main()
