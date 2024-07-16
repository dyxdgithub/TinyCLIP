"""Download images into the leaf-class structure defined by an n2 hierarchy."""

import argparse
import asyncio
import csv
import json
import random
import re
import sqlite3
from pathlib import Path
from urllib.parse import urlparse

from tqdm import tqdm


META_DIR = Path(__file__).resolve().parent
DEFAULT_CSV = META_DIR / "Hierarchy" / "n2" / "audio_image_n2.csv"
DEFAULT_JSON = META_DIR / "Hierarchy" / "n2" / "Last_Two_Layers_Multi_Members_n2.json"
DEFAULT_OUTPUT = META_DIR / "Hierarchy" / "n2" / "audio_image" / "images"
IMAGE_ID_PATTERN = re.compile(r"^[A-Za-z0-9._-]+$")
INVALID_PATH_CHARS = re.compile(r'[<>:"/\\|?*]')


def safe_component(value):
    value = INVALID_PATH_CHARS.sub("_", str(value)).strip().rstrip(".")
    return value or "unnamed"


def load_leaf_paths(json_path):
    with Path(json_path).open("r", encoding="utf-8") as handle:
        root = json.load(handle)
    mapping = {}

    def visit(node, parents):
        if not isinstance(node, dict):
            raise ValueError("Every hierarchy node must be a JSON object")
        children = node.get("Subcategory", [])
        if children and not isinstance(children, list):
            raise ValueError("Subcategory must be a list")
        label = node.get("LabelName")
        if not label:
            raise ValueError("Every hierarchy node must contain LabelName")
        text = node.get("TextName", label)
        if not children:
            if parents:
                mapping.setdefault(label, set()).add(
                    tuple(safe_component(item) for item in parents + [text])
                )
            return
        for child in children:
            visit(child, parents + [text])

    for child in root.get("Subcategory", []):
        visit(child, [])
    return mapping


def collect_candidates(csv_path, leaf_paths, seed):
    candidates = {}
    seen = {}
    required = {"ImageID", "OriginalURL", "Thumbnail300KURL", "Human_LabelName"}
    with Path(csv_path).open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames or not required.issubset(reader.fieldnames):
            raise ValueError(
                "CSV must contain ImageID, OriginalURL, Thumbnail300KURL, and Human_LabelName"
            )
        for row in tqdm(reader, desc="Reading image candidates", unit="row"):
            image_id = str(row.get("ImageID", "")).strip()
            label = str(row.get("Human_LabelName", "")).strip()
            if not image_id or not IMAGE_ID_PATTERN.match(image_id) or label not in leaf_paths:
                continue
            label_seen = seen.setdefault(label, set())
            if image_id in label_seen:
                continue
            label_seen.add(image_id)
            candidates.setdefault(label, []).append(row)
    generator = random.Random(seed)
    for rows in candidates.values():
        generator.shuffle(rows)
    return candidates


def load_previous_errors(manifest_path):
    manifest = Path(manifest_path)
    if not manifest.exists():
        return set()
    errors = set()
    with manifest.open("r", encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            if str(row.get("Status", "")).strip().lower() == "error":
                errors.add((row.get("ClassName", ""), row.get("ImageID", "")))
    return errors


def prioritize_errors(candidates, previous_errors):
    count = 0
    for label, rows in candidates.items():
        failed = [row for row in rows if (label, row["ImageID"]) in previous_errors]
        rest = [row for row in rows if (label, row["ImageID"]) not in previous_errors]
        candidates[label] = failed + rest
        count += len(failed)
    return count


def extension_from_url(url):
    suffix = Path(urlparse(url).path).suffix.lower()
    return suffix if suffix in {".jpg", ".jpeg", ".png", ".webp", ".bmp"} else ".jpg"


async def download_one(session, semaphore, label, row, output_dir, overwrite):
    image_id = row["ImageID"]
    destinations = [Path(output_dir).joinpath(*path, image_id + ".jpg") for path in row["Paths"]]
    if not overwrite and all(path.exists() for path in destinations):
        return label, image_id, "skipped", "", len(destinations)
    urls = [
        url for url in (row.get("Thumbnail300KURL", ""), row.get("OriginalURL", ""))
        if url and url.startswith("http")
    ]
    async with semaphore:
        last_error = "no valid URL"
        for url in urls:
            try:
                async with session.get(url) as response:
                    if response.status != 200:
                        last_error = "HTTP {}".format(response.status)
                        continue
                    content = await response.read()
                    suffix = extension_from_url(url)
                    for destination in destinations:
                        destination.parent.mkdir(parents=True, exist_ok=True)
                        destination.with_suffix(suffix).write_bytes(content)
                    return label, image_id, "downloaded", "", len(destinations)
            except Exception as error:
                last_error = str(error)
        return label, image_id, "error", last_error, len(destinations)


async def download_class(
    session, semaphore, label, rows, quota, output_dir, concurrency, overwrite, progress, counts
):
    target = min(quota, len(rows))
    successful = 0
    position = 0
    results = []
    while position < len(rows) and successful < target:
        batch = rows[position:position + min(concurrency, target - successful)]
        position += len(batch)
        batch_results = await asyncio.gather(
            *(download_one(session, semaphore, label, row, output_dir, overwrite) for row in batch)
        )
        for result in batch_results:
            results.append(result)
            _, _, status, _, _ = result
            counts[status] += 1
            successful += status in {"downloaded", "skipped"}
            progress.update(1)
            progress.set_postfix(**counts)
    return label, results, successful, target


async def run_download(candidates, output_dir, manifest_path, quota, concurrency, overwrite):
    import aiohttp

    previous_errors = load_previous_errors(manifest_path)
    retry_count = prioritize_errors(candidates, previous_errors)
    total = sum(min(quota, len(rows)) for rows in candidates.values())
    semaphore = asyncio.Semaphore(concurrency)
    counts = {"downloaded": 0, "skipped": 0, "error": 0}
    shortages = []
    timeout = aiohttp.ClientTimeout(total=60)
    headers = {"User-Agent": "Mozilla/5.0 TinyCLIP image downloader"}
    manifest_path = Path(manifest_path)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with manifest_path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["ClassName", "ImageID", "Status", "Error", "DestinationCount"])
        async with aiohttp.ClientSession(headers=headers, timeout=timeout) as session:
            with tqdm(total=total, desc="Downloading images", unit="image") as progress:
                results = await asyncio.gather(
                    *(
                        download_class(
                            session,
                            semaphore,
                            label,
                            rows,
                            quota,
                            output_dir,
                            concurrency,
                            overwrite,
                            progress,
                            counts,
                        )
                        for label, rows in sorted(candidates.items())
                    )
                )
        for label, class_results, successful, target in results:
            for _, image_id, status, error, destination_count in class_results:
                writer.writerow([label, image_id, status, error, destination_count])
            if successful < target:
                shortages.append((label, successful, target))
    return counts, shortages, retry_count


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", type=Path, default=DEFAULT_CSV)
    parser.add_argument("--json", type=Path, default=DEFAULT_JSON)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--manifest", type=Path, default=None)
    parser.add_argument("--samples-per-class", type=int, default=300)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--concurrency", type=int, default=20)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    if args.samples_per_class < 1:
        raise ValueError("--samples-per-class must be positive")
    if args.concurrency < 1:
        raise ValueError("--concurrency must be positive")
    output_dir = args.output_dir
    manifest = args.manifest or output_dir / "image_download_manifest.csv"
    leaf_paths = load_leaf_paths(args.json)
    candidates = collect_candidates(args.csv, leaf_paths, args.seed)
    for label, rows in candidates.items():
        for row in rows:
            row["Paths"] = leaf_paths[label]
    counts, shortages, retry_count = asyncio.run(
        run_download(
            candidates,
            output_dir,
            manifest,
            args.samples_per_class,
            args.concurrency,
            args.overwrite,
        )
    )
    print("Leaf classes: {}".format(len(candidates)))
    print("Previous errors queued for retry: {}".format(retry_count))
    print("Download results: {}".format(counts))
    if shortages:
        print("Classes below requested count: {}".format(len(shortages)))
        for label, successful, target in shortages:
            print("  {}: {}/{}".format(label, successful, target))
    print("Manifest: {}".format(Path(manifest).resolve()))


if __name__ == "__main__":
    main()
