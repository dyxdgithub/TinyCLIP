"""Download manifest images once and place them in all matching JSON hierarchy folders."""

import argparse
import asyncio
import csv
import json
import re
import shutil
from collections import OrderedDict
from pathlib import Path
from urllib.parse import urlparse

from tqdm import tqdm


META_DIR = Path(__file__).resolve().parent
DEFAULT_MANIFEST = (
    META_DIR
    / "Hierarchy"
    / "n2"
    / "audio_image"
    / "images"
    / "image_download_manifest_with_audio_info.csv"
)
DEFAULT_JSON = META_DIR / "Hierarchy" / "n2" / "Last_Two_Layers_Multi_Members_n2.json"
DEFAULT_OUTPUT_DIR = META_DIR / "Hierarchy" / "n2" / "audio_image" / "images"
IMAGE_ID_PATTERN = re.compile(r"^[A-Za-z0-9._-]+$")
INVALID_PATH_CHARS = re.compile(r'[<>:"/\\|?*]')
IMAGE_SUFFIXES = (".jpg", ".jpeg", ".png", ".webp", ".bmp")
PROGRESS_FIELDS = ["ImageID", "Status", "Error", "DestinationCount", "ImagePath"]


def safe_component(value):
    value = INVALID_PATH_CHARS.sub("_", str(value)).strip().rstrip(".")
    return value or "unnamed"


def load_leaf_paths(json_path):
    """Return every hierarchy destination path keyed by its leaf LabelName."""
    with Path(json_path).open("r", encoding="utf-8") as handle:
        root = json.load(handle)
    paths_by_label = {}

    def visit(node, parents):
        if not isinstance(node, dict):
            raise ValueError("Every hierarchy node must be a JSON object")
        label = node.get("LabelName")
        if not label:
            raise ValueError("Every hierarchy node must contain LabelName")
        name = safe_component(node.get("TextName", label))
        children = node.get("Subcategory", [])
        if children and not isinstance(children, list):
            raise ValueError("Subcategory must be a list")
        if not children:
            paths_by_label.setdefault(label, set()).add(tuple(parents + [name]))
            return
        for child in children:
            visit(child, parents + [name])

    for child in root.get("Subcategory", []):
        visit(child, [])
    return paths_by_label


def preferred_urls(row):
    """Prefer the smaller image URL and retain the original URL as a fallback."""
    return tuple(
        url
        for url in (row.get("Thumbnail300KURL", ""), row.get("OriginalURL", ""))
        if url and url.startswith(("http://", "https://"))
    )


def collect_images(manifest_path, paths_by_label):
    """Combine repeated manifest rows into one download record per image ID."""
    required = {"ImageID", "Human_LabelName", "OriginalURL", "Thumbnail300KURL"}
    images = OrderedDict()
    with Path(manifest_path).open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        fields = set(reader.fieldnames or [])
        missing = required.difference(fields)
        if missing:
            raise ValueError("Manifest must contain {}".format(", ".join(sorted(missing))))
        for row in tqdm(reader, desc="Reading image manifest", unit="row"):
            image_id = str(row.get("ImageID", "")).strip()
            label = str(row.get("Human_LabelName", "")).strip()
            urls = preferred_urls(row)
            if not IMAGE_ID_PATTERN.match(image_id) or label not in paths_by_label or not urls:
                continue
            image = images.setdefault(
                image_id,
                {"urls": urls, "paths": set()},
            )
            image["paths"].update(paths_by_label[label])
    return images


def extension_from_url(url):
    suffix = Path(urlparse(url).path).suffix.lower()
    return suffix if suffix in IMAGE_SUFFIXES else ".jpg"


def find_existing_image(image_id, destination_dirs):
    for destination_dir in destination_dirs:
        for suffix in IMAGE_SUFFIXES:
            candidate = destination_dir / (image_id + suffix)
            if candidate.is_file() and candidate.stat().st_size > 0:
                return candidate
    return None


def materialize_destinations(source_path, destination_dirs):
    """Copy one local image into every missing class directory."""
    destination_paths = []
    for destination_dir in destination_dirs:
        destination_dir.mkdir(parents=True, exist_ok=True)
        destination = destination_dir / source_path.name
        if destination.resolve() != source_path.resolve() and not destination.is_file():
            shutil.copy2(source_path, destination)
        destination_paths.append(destination)
    return destination_paths


def append_progress(progress_path, image_id, status, error, destination_count, image_path):
    progress_path = Path(progress_path)
    progress_path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not progress_path.is_file() or progress_path.stat().st_size == 0
    with progress_path.open("a", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=PROGRESS_FIELDS)
        if write_header:
            writer.writeheader()
        writer.writerow(
            {
                "ImageID": image_id,
                "Status": status,
                "Error": error,
                "DestinationCount": destination_count,
                "ImagePath": image_path,
            }
        )
        handle.flush()


async def download_one(session, semaphore, image_id, image, output_dir, timeout_seconds):
    destination_dirs = [Path(output_dir).joinpath(*path) for path in sorted(image["paths"])]
    existing = find_existing_image(image_id, destination_dirs)
    if existing:
        destination_paths = materialize_destinations(existing, destination_dirs)
        return image_id, "skipped", "", len(destination_paths), str(existing.resolve())

    last_error = "no valid URL"
    async with semaphore:
        for url in image["urls"]:
            try:
                async with session.get(url) as response:
                    if response.status != 200:
                        last_error = "HTTP {} from {}".format(response.status, url)
                        continue
                    content = await response.read()
                if not content:
                    last_error = "empty response from {}".format(url)
                    continue
                suffix = extension_from_url(url)
                first_directory = destination_dirs[0]
                first_directory.mkdir(parents=True, exist_ok=True)
                first_path = first_directory / (image_id + suffix)
                first_path.write_bytes(content)
                destination_paths = materialize_destinations(first_path, destination_dirs)
                return image_id, "downloaded", "", len(destination_paths), str(first_path.resolve())
            except TimeoutError:
                last_error = "timeout after {} seconds from {}".format(timeout_seconds, url)
            except Exception as error:
                last_error = "{} from {}".format(error, url)
    return image_id, "error", last_error, len(destination_dirs), ""


async def run_download(images, output_dir, progress_path, concurrency, timeout_seconds, limit):
    try:
        import aiohttp
    except ImportError as error:
        raise RuntimeError(
            "The aiohttp package is required. Install it with "
            "`python -m pip install aiohttp`."
        ) from error

    selected = list(images.items())[:limit] if limit is not None else list(images.items())
    semaphore = asyncio.Semaphore(concurrency)
    timeout = aiohttp.ClientTimeout(total=timeout_seconds)
    headers = {"User-Agent": "Mozilla/5.0 TinyCLIP OpenImage downloader"}
    counts = {"downloaded": 0, "skipped": 0, "error": 0}
    connector = aiohttp.TCPConnector(limit=concurrency)
    async with aiohttp.ClientSession(
        headers=headers, timeout=timeout, connector=connector
    ) as session:
        tasks = [
            download_one(
                session,
                semaphore,
                image_id,
                image,
                output_dir,
                timeout_seconds,
            )
            for image_id, image in selected
        ]
        with tqdm(total=len(tasks), desc="Downloading images", unit="image") as progress:
            for task in asyncio.as_completed(tasks):
                image_id, status, error, destination_count, image_path = await task
                counts[status] += 1
                append_progress(
                    progress_path,
                    image_id,
                    status,
                    error,
                    destination_count,
                    image_path,
                )
                progress.update(1)
                progress.set_postfix(**counts)
    return counts, len(selected)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest",
        type=Path,
        default=DEFAULT_MANIFEST,
        help="Source CSV with OpenImage URLs and Human_LabelName. Default: %(default)s",
    )
    parser.add_argument(
        "--json",
        type=Path,
        default=DEFAULT_JSON,
        help="Hierarchy JSON used to build parent/leaf destination folders. Default: %(default)s",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Root directory for hierarchy-organized images. Default: %(default)s",
    )
    parser.add_argument(
        "--progress",
        type=Path,
        default=None,
        help="Append-only per-image checkpoint CSV. Default: image_download_by_hierarchy.progress.csv under --output-dir.",
    )
    parser.add_argument(
        "--concurrency",
        type=int,
        default=20,
        help="Maximum concurrent HTTP requests; must be positive. Default: %(default)s",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=60,
        help="Per-request total timeout in seconds; must be positive. Default: %(default)s",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Process only the first N unique images for a smoke test. Default: all images.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Report the hierarchy mapping and download scope without requesting or writing images.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    if args.concurrency < 1:
        raise ValueError("--concurrency must be positive")
    if args.timeout < 1:
        raise ValueError("--timeout must be positive")
    if args.limit is not None and args.limit < 1:
        raise ValueError("--limit must be positive")

    paths_by_label = load_leaf_paths(args.json)
    images = collect_images(args.manifest, paths_by_label)
    destination_count = sum(len(image["paths"]) for image in images.values())
    print("Hierarchy leaf labels: {}".format(len(paths_by_label)))
    print("Unique images to process: {}".format(len(images)))
    print("Image-to-folder destinations: {}".format(destination_count))
    if args.dry_run:
        return

    progress_path = args.progress or Path(args.output_dir) / "image_download_by_hierarchy.progress.csv"
    counts, selected_count = asyncio.run(
        run_download(
            images,
            args.output_dir,
            progress_path,
            args.concurrency,
            args.timeout,
            args.limit,
        )
    )
    print("Processed unique images: {}".format(selected_count))
    print("Download results: {}".format(counts))
    print("Progress checkpoint: {}".format(progress_path.resolve()))


if __name__ == "__main__":
    main()
