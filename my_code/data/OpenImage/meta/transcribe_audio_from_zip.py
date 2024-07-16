"""Extract CSV-referenced audio from a ZIP and transcribe it with Whisper."""

import argparse
import csv
import os
import shutil
import sys
import zipfile
from collections import OrderedDict
from pathlib import Path

from tqdm import tqdm


META_DIR = Path(__file__).resolve().parent
DEFAULT_CSV = META_DIR / "Hierarchy" / "n2" / "audio_image" / "images" / "image_download_manifest_with_audio_info.csv"
DEFAULT_ZIP = Path(r"E:\数据集\Open Image\train\open_images_train_audio.zip")
DEFAULT_AUDIO_DIR = META_DIR / "Hierarchy" / "n2" / "audio_image" / "audio"
DEFAULT_OUTPUT = META_DIR / "Hierarchy" / "n2" / "audio_image" / "audio_transcriptions.csv"
OUTPUT_FIELDS = ["ClassName", "Human_LabelName", "ImageID", "Caption"]
PROGRESS_FIELDS = ["FileName", "ImageID", "Status", "Caption", "Error"]


def configure_ffmpeg(ffmpeg_path=None):
    candidates = []
    if ffmpeg_path:
        requested = Path(ffmpeg_path)
        candidates.append(requested / "ffmpeg.exe" if requested.is_dir() else requested)
    executable_dir = Path(sys.executable).resolve().parent
    candidates.extend(
        [
            executable_dir / "ffmpeg.exe",
            executable_dir / "Library" / "bin" / "ffmpeg.exe",
            executable_dir.parent / "Library" / "bin" / "ffmpeg.exe",
            executable_dir.parent / "bin" / "ffmpeg.exe",
        ]
    )
    for conda_root in (executable_dir, executable_dir.parent):
        candidates.extend(conda_root.glob("envs/*/Library/bin/ffmpeg.exe"))
    conda_prefix = os.environ.get("CONDA_PREFIX")
    if conda_prefix:
        candidates.append(Path(conda_prefix) / "Library" / "bin" / "ffmpeg.exe")
    path_entries = os.environ.get("PATH", "").split(os.pathsep)
    for entry in path_entries:
        candidates.append(Path(entry) / "ffmpeg.exe")
    for candidate in candidates:
        if candidate.is_file():
            bin_dir = str(candidate.parent)
            if bin_dir not in path_entries:
                os.environ["PATH"] = bin_dir + os.pathsep + os.environ.get("PATH", "")
            return str(candidate.resolve())
    raise RuntimeError(
        "ffmpeg.exe was not found. Install FFmpeg, add its bin folder to PATH, "
        "or pass --ffmpeg-path C:\\path\\to\\ffmpeg\\bin"
    )


def read_audio_rows(csv_path):
    rows = []
    seen = set()
    with Path(csv_path).open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        fields = set(reader.fieldnames or [])
        filename_field = "FileName" if "FileName" in fields else "AudioFileName"
        if "ImageID" not in fields or filename_field not in fields:
            raise ValueError("CSV must contain FileName or AudioFileName, and ImageID")
        for row in reader:
            filename = str(row.get(filename_field, "")).strip()
            image_id = str(row.get("ImageID", "")).strip()
            if not filename or not image_id:
                continue
            row = dict(row)
            row["FileName"] = filename
            key = (filename, image_id, row.get("Human_LabelName", ""), row.get("Human_DisplayName", ""))
            if key not in seen:
                rows.append(row)
                seen.add(key)
    return rows


def index_zip_audio(zip_path):
    members = OrderedDict()
    with zipfile.ZipFile(zip_path) as archive:
        for info in archive.infolist():
            if not info.is_dir():
                members.setdefault(Path(info.filename).name, info.filename)
    return members


def extract_audio_files(rows, zip_path, audio_dir):
    audio_dir = Path(audio_dir)
    audio_dir.mkdir(parents=True, exist_ok=True)
    members = index_zip_audio(zip_path)
    extracted = {}
    missing = set()
    with zipfile.ZipFile(zip_path) as archive:
        for row in tqdm(rows, desc="Extracting audio", unit="row"):
            filename = row["FileName"]
            if filename in extracted:
                continue
            archive_name = members.get(filename)
            if not archive_name:
                missing.add(filename)
                continue
            destination = audio_dir / filename
            if not destination.exists():
                with archive.open(archive_name) as source, destination.open("wb") as target:
                    shutil.copyfileobj(source, target)
            extracted[filename] = str(destination.resolve())
    return extracted, missing


def group_rows_by_filename(rows):
    rows_by_filename = OrderedDict()
    for row in rows:
        rows_by_filename.setdefault(row["FileName"], []).append(row)
    return rows_by_filename


def load_completed_progress(progress_path):
    completed = {}
    progress_path = Path(progress_path)
    if not progress_path.is_file():
        return completed
    with progress_path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            filename = str(row.get("FileName", "")).strip()
            if filename and row.get("Status") == "ok":
                completed[filename] = row.get("Caption", "")
    return completed


def load_existing_captions(output_path):
    captions_by_image_id = {}
    output_path = Path(output_path)
    if not output_path.is_file():
        return captions_by_image_id
    with output_path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            image_id = str(row.get("ImageID", "")).strip()
            caption = str(row.get("Caption", "")).strip()
            if image_id and caption:
                captions_by_image_id.setdefault(image_id, caption)
    return captions_by_image_id


def append_progress(progress_path, filename, image_id, status, caption="", error=""):
    progress_path = Path(progress_path)
    progress_path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not progress_path.exists() or progress_path.stat().st_size == 0
    with progress_path.open("a", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=PROGRESS_FIELDS)
        if write_header:
            writer.writeheader()
        writer.writerow(
            {
                "FileName": filename,
                "ImageID": image_id,
                "Status": status,
                "Caption": caption,
                "Error": error,
            }
        )
        handle.flush()


def write_completed_output(rows, completed, output_path):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = output_path.with_suffix(output_path.suffix + ".tmp")
    written = set()
    with temporary_path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=OUTPUT_FIELDS)
        writer.writeheader()
        for row in rows:
            filename = row["FileName"]
            if filename not in completed:
                continue
            output_row = {
                "ClassName": row.get("ClassName", ""),
                "Human_LabelName": row.get("Human_LabelName", ""),
                "ImageID": row.get("ImageID", ""),
                "Caption": completed[filename],
            }
            key = tuple(output_row[field] for field in OUTPUT_FIELDS[:-1])
            if key not in written:
                writer.writerow(output_row)
                written.add(key)
    temporary_path.replace(output_path)
    return len(written)


def transcribe(
    rows,
    zip_path,
    audio_dir,
    output_path,
    progress_path,
    model_size,
    device,
    language,
    beam_size,
    resume,
    completed,
):
    import torch
    import whisper

    if device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable. Use --device cpu or fix the GPU environment.")
    rows_by_filename = group_rows_by_filename(rows)
    if resume:
        existing_captions = load_existing_captions(output_path)
        for filename, filename_rows in rows_by_filename.items():
            if filename in completed:
                continue
            image_id = filename_rows[0].get("ImageID", "")
            if image_id in existing_captions:
                completed[filename] = existing_captions[image_id]
                append_progress(
                    progress_path,
                    filename,
                    image_id,
                    "ok",
                    completed[filename],
                )

    pending_rows = [
        filename_rows[0]
        for filename, filename_rows in rows_by_filename.items()
        if filename not in completed
    ]
    if not pending_rows:
        return completed, 0, 0

    audio_paths, missing = extract_audio_files(pending_rows, zip_path, audio_dir)
    ready_rows = [row for row in pending_rows if row["FileName"] in audio_paths]
    for row in pending_rows:
        if row["FileName"] not in audio_paths:
            append_progress(
                progress_path,
                row["FileName"],
                row.get("ImageID", ""),
                "error",
                error="audio not found in ZIP",
            )
    if not ready_rows:
        return completed, 0, len(missing)

    model = whisper.load_model(model_size, device=device)
    use_fp16 = device.startswith("cuda")
    print("Using GPU: {}".format(use_fp16))
    transcribed = 0
    for row in tqdm(ready_rows, desc="Transcribing audio", unit="file"):
        filename = row["FileName"]
        try:
            caption = whisper_result(
                model,
                audio_paths[filename],
                language=language,
                beam_size=beam_size,
                fp16=use_fp16,
            )["Text"]
        except Exception as error:
            append_progress(
                progress_path,
                filename,
                row.get("ImageID", ""),
                "error",
                error=str(error),
            )
            continue
        completed[filename] = caption
        append_progress(
            progress_path,
            filename,
            row.get("ImageID", ""),
            "ok",
            caption,
        )
        transcribed += 1
    return completed, transcribed, len(missing)


def whisper_result(model, audio_path, language, beam_size, fp16):
    options = {"fp16": fp16, "beam_size": beam_size}
    if language:
        options["language"] = language
    result = model.transcribe(audio_path, **options)
    return {"Text": result.get("text", "").strip(), "Status": "ok", "Error": ""}


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", type=Path, default=DEFAULT_CSV)
    parser.add_argument("--zip", type=Path, default=DEFAULT_ZIP)
    parser.add_argument("--audio-dir", type=Path, default=DEFAULT_AUDIO_DIR)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--progress",
        type=Path,
        default=None,
        help="Audio-level checkpoint CSV. Defaults beside --output.",
    )
    parser.add_argument(
        "--no-resume",
        action="store_false",
        dest="resume",
        help="Ignore the checkpoint and re-transcribe every audio file.",
    )
    parser.set_defaults(resume=True)
    parser.add_argument("--model-size", default="small")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--language", default=None)
    parser.add_argument("--beam-size", type=int, default=5)
    parser.add_argument(
        "--ffmpeg-path",
        default=None,
        help="Path to ffmpeg.exe or the directory containing it.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    # print("Using ffmpeg: {}".format(configure_ffmpeg(args.ffmpeg_path)))
    rows = read_audio_rows(args.csv)
    progress_path = args.progress or args.output.with_suffix(".progress.csv")
    completed = load_completed_progress(progress_path) if args.resume else {}
    try:
        completed, transcribed, missing_count = transcribe(
            rows,
            args.zip,
            args.audio_dir,
            args.output,
            progress_path,
            args.model_size,
            args.device,
            args.language,
            args.beam_size,
            args.resume,
            completed,
        )
    finally:
        output_rows = write_completed_output(rows, completed, args.output)
    print("Input rows: {}".format(len(rows)))
    print("Newly transcribed audio files: {}".format(transcribed))
    print("Completed audio files: {}".format(len(completed)))
    print("Missing audio filenames: {}".format(missing_count))
    print("Output rows: {}".format(output_rows))
    print("Progress checkpoint: {}".format(progress_path.resolve()))
    print("Transcription output: {}".format(Path(args.output).resolve()))


if __name__ == "__main__":
    main()
