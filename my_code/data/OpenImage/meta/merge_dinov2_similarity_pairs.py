"""Merge per-small-class DINOv2 similarity tables into disjoint high-score pairs."""

import argparse
import csv
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path


META_DIR = Path(__file__).resolve().parents[1] / "data" / "OpenImage" / "meta"
DEFAULT_INPUT_DIR = (
    META_DIR / "Hierarchy" / "n2" / "audio_image" / "dinov2_similarity" / "tables"
)
DEFAULT_OUTPUT_DIR = (
    META_DIR / "Hierarchy" / "n2" / "audio_image" / "dinov2_similarity" / "merged_pairs"
)
REQUIRED_FIELDS = {"ParentClass", "QueryImagePath", "MatchedImagePath", "Similarity"}
OUTPUT_FIELDS = [
    "ParentClass",
    "QuerySmallClass",
    "MatchedSmallClass",
    "QueryImageID",
    "MatchedImageID",
    "QueryImagePath",
    "MatchedImagePath",
    "Similarity",
    "SourceFile",
]
SUMMARY_FIELDS = [
    "ParentClass",
    "InputRows",
    "SelfPairsSkipped",
    "DuplicatePairsRemoved",
    "UniqueCandidatePairs",
    "PairsSkippedForUsedImage",
    "SelectedPairs",
    "OutputFile",
]


@dataclass(frozen=True)
class PairCandidate:
    parent_class: str
    query_small_class: str
    matched_small_class: str
    query_image_id: str
    matched_image_id: str
    query_path: str
    matched_path: str
    similarity: float
    source_file: str

    @property
    def image_id_pair(self):
        return tuple(sorted((self.query_image_id, self.matched_image_id)))

    @property
    def tie_break_key(self):
        return (
            self.image_id_pair,
            self.query_path,
            self.matched_path,
            self.source_file,
        )


def image_id_from_path(path):
    image_id = Path(path).stem.strip()
    if not image_id:
        raise ValueError("Image path has no filename: {}".format(path))
    return image_id


def small_class_from_path(path):
    return Path(path).parent.name


def safe_file_name(value):
    return "".join(
        character if character.isalnum() or character in " ._-" else "_"
        for character in value
    )


def read_parent_candidates(parent_dir):
    """Read all leaf-class CSV rows for one parent directory."""
    candidates = []
    input_rows = 0
    self_pairs = 0
    csv_paths = sorted(parent_dir.glob("*.csv"))
    for csv_path in csv_paths:
        with csv_path.open("r", encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle)
            fields = set(reader.fieldnames or [])
            missing = REQUIRED_FIELDS.difference(fields)
            if missing:
                raise ValueError(
                    "{} is missing required columns: {}".format(
                        csv_path, ", ".join(sorted(missing))
                    )
                )
            for row_number, row in enumerate(reader, start=2):
                input_rows += 1
                query_path = str(row.get("QueryImagePath", "")).strip()
                matched_path = str(row.get("MatchedImagePath", "")).strip()
                parent_class = str(row.get("ParentClass", "")).strip()
                if not query_path or not matched_path or not parent_class:
                    raise ValueError("{}:{} has empty required fields".format(csv_path, row_number))
                try:
                    similarity = float(row["Similarity"])
                except (TypeError, ValueError) as error:
                    raise ValueError(
                        "{}:{} has invalid Similarity: {}".format(
                            csv_path, row_number, row.get("Similarity", "")
                        )
                    ) from error
                query_image_id = image_id_from_path(query_path)
                matched_image_id = image_id_from_path(matched_path)
                if query_image_id == matched_image_id:
                    self_pairs += 1
                    continue
                candidates.append(
                    PairCandidate(
                        parent_class=parent_class,
                        query_small_class=str(row.get("SmallClass", "")).strip()
                        or small_class_from_path(query_path),
                        matched_small_class=small_class_from_path(matched_path),
                        query_image_id=query_image_id,
                        matched_image_id=matched_image_id,
                        query_path=query_path,
                        matched_path=matched_path,
                        similarity=similarity,
                        source_file=str(csv_path.resolve()),
                    )
                )
    return candidates, input_rows, self_pairs, len(csv_paths)


def deduplicate_pairs(candidates):
    """Keep the highest-scoring row for every undirected ImageID pair."""
    best_by_pair = {}
    for candidate in candidates:
        current = best_by_pair.get(candidate.image_id_pair)
        if current is None:
            best_by_pair[candidate.image_id_pair] = candidate
            continue
        if candidate.similarity > current.similarity or (
            candidate.similarity == current.similarity
            and candidate.tie_break_key < current.tie_break_key
        ):
            best_by_pair[candidate.image_id_pair] = candidate
    return list(best_by_pair.values()), len(candidates) - len(best_by_pair)


def select_disjoint_pairs(candidates):
    """Greedily retain descending-score pairs so each ImageID appears at most once."""
    selected = []
    used_image_ids = set()
    skipped_for_used_image = 0
    for candidate in sorted(
        candidates, key=lambda item: (-item.similarity, item.tie_break_key)
    ):
        if (
            candidate.query_image_id in used_image_ids
            or candidate.matched_image_id in used_image_ids
        ):
            skipped_for_used_image += 1
            continue
        selected.append(candidate)
        used_image_ids.add(candidate.query_image_id)
        used_image_ids.add(candidate.matched_image_id)
    return selected, skipped_for_used_image


def write_pairs(output_path, pairs):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=OUTPUT_FIELDS)
        writer.writeheader()
        for pair in pairs:
            writer.writerow(
                {
                    "ParentClass": pair.parent_class,
                    "QuerySmallClass": pair.query_small_class,
                    "MatchedSmallClass": pair.matched_small_class,
                    "QueryImageID": pair.query_image_id,
                    "MatchedImageID": pair.matched_image_id,
                    "QueryImagePath": pair.query_path,
                    "MatchedImagePath": pair.matched_path,
                    "Similarity": "{:.8f}".format(pair.similarity),
                    "SourceFile": pair.source_file,
                }
            )


def merge_parent_directory(parent_dir, output_dir, write_output):
    candidates, input_rows, self_pairs, source_count = read_parent_candidates(parent_dir)
    unique_pairs, duplicates_removed = deduplicate_pairs(candidates)
    selected_pairs, skipped_for_used_image = select_disjoint_pairs(unique_pairs)
    parent_class = selected_pairs[0].parent_class if selected_pairs else parent_dir.name
    output_path = Path(output_dir) / (safe_file_name(parent_class) + ".csv")
    if write_output:
        write_pairs(output_path, selected_pairs)
    return {
        "ParentClass": parent_class,
        "InputRows": input_rows,
        "SelfPairsSkipped": self_pairs,
        "DuplicatePairsRemoved": duplicates_removed,
        "UniqueCandidatePairs": len(unique_pairs),
        "PairsSkippedForUsedImage": skipped_for_used_image,
        "SelectedPairs": len(selected_pairs),
        "OutputFile": str(output_path.resolve()),
        "SourceTables": source_count,
    }


def write_summary(output_dir, summaries):
    summary_path = Path(output_dir) / "merge_summary.csv"
    with summary_path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=SUMMARY_FIELDS)
        writer.writeheader()
        for summary in summaries:
            writer.writerow({field: summary[field] for field in SUMMARY_FIELDS})
    return summary_path


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=DEFAULT_INPUT_DIR,
        help="Directory containing one subdirectory of small-class CSV tables per parent class. Default: %(default)s",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Output directory for one disjoint-pair CSV per parent class and merge_summary.csv. Default: %(default)s",
    )
    parser.add_argument(
        "--parent",
        default=None,
        help="Optional exact parent-directory name to merge. Default: merge every parent directory.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate and report merge statistics without writing output CSV files.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    input_dir = Path(args.input_dir)
    if not input_dir.is_dir():
        raise FileNotFoundError("Input directory does not exist: {}".format(input_dir))
    parent_dirs = sorted(path for path in input_dir.iterdir() if path.is_dir())
    if args.parent is not None:
        parent_dirs = [path for path in parent_dirs if path.name == args.parent]
        if not parent_dirs:
            raise ValueError("No parent directory named '{}' under {}".format(args.parent, input_dir))
    if not parent_dirs:
        raise ValueError("No parent directories found under {}".format(input_dir))

    summaries = []
    for parent_dir in parent_dirs:
        summary = merge_parent_directory(parent_dir, args.output_dir, not args.dry_run)
        summaries.append(summary)
        print(
            "{}: {} input rows -> {} selected pairs ({} duplicate pairs removed)".format(
                summary["ParentClass"],
                summary["InputRows"],
                summary["SelectedPairs"],
                summary["DuplicatePairsRemoved"],
            )
        )
    if args.dry_run:
        return
    summary_path = write_summary(args.output_dir, summaries)
    print("Summary: {}".format(summary_path.resolve()))


if __name__ == "__main__":
    main()
