"""List audio filenames and extract the image ID from each filename."""

import argparse
import csv
import zipfile
from pathlib import Path

from tqdm import tqdm


META_DIR = Path(__file__).resolve().parent
DEFAULT_ZIP = Path(r"E:\数据集\Open Image\train\open_images_train_audio.zip")
DEFAULT_OUTPUT = META_DIR / "audio_filename_table.csv"


def extract_image_id(filename):
    stem = Path(filename).stem
    parts = stem.split("_")
    return parts[-2] if len(parts) >= 2 else ""


def build_audio_filename_table(zip_path, output_path):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    audio_count = 0
    with zipfile.ZipFile(zip_path) as archive, output_path.open(
        "w", encoding="utf-8-sig", newline=""
    ) as output_handle:
        writer = csv.DictWriter(
            output_handle,
            fieldnames=["FileName", "ArchivePath", "ImageID", "Suffix", "FileSize"],
        )
        writer.writeheader()
        members = [
            info
            for info in archive.infolist()
            if not info.is_dir() and Path(info.filename).suffix.lower() in {".ogg", ".wav", ".mp3", ".flac"}
        ]
        for info in tqdm(members, desc="Listing audio files", unit="file"):
            path = Path(info.filename)
            writer.writerow(
                {
                    "FileName": path.name,
                    "ArchivePath": info.filename,
                    "ImageID": extract_image_id(path.name),
                    "Suffix": path.suffix.lower(),
                    "FileSize": info.file_size,
                }
            )
            audio_count += 1
    print("Audio files: {}".format(audio_count))
    print("Output: {}".format(output_path.resolve()))
    return audio_count


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--zip", type=Path, default=DEFAULT_ZIP)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def main():
    args = parse_args()
    build_audio_filename_table(args.zip, args.output)


if __name__ == "__main__":
    main()
