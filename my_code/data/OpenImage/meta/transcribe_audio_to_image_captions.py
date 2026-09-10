"""Transcribe manifest-referenced English audio and append captions to its image rows."""

import argparse
import csv
import os
import sys
from collections import OrderedDict
from pathlib import Path

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
DEFAULT_AUDIO_DIR = META_DIR / "Hierarchy" / "n2" / "audio_image" / "audio"
DEFAULT_OUTPUT = DEFAULT_MANIFEST.with_name(
    "image_download_manifest_with_audio_captions.csv"
)
PROGRESS_FIELDS = ["FileName", "Status", "Caption", "Error"]


def configure_ffmpeg(ffmpeg_path=None):
    """Find ffmpeg.exe and prepend its directory to PATH for Whisper."""
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
    conda_prefix = os.environ.get("CONDA_PREFIX")
    if conda_prefix:
        candidates.append(Path(conda_prefix) / "Library" / "bin" / "ffmpeg.exe")

    path_entries = os.environ.get("PATH", "").split(os.pathsep)
    candidates.extend(Path(entry) / "ffmpeg.exe" for entry in path_entries if entry)
    for candidate in candidates:
        if candidate.is_file():
            ffmpeg_dir = str(candidate.parent.resolve())
            if ffmpeg_dir not in path_entries:
                os.environ["PATH"] = ffmpeg_dir + os.pathsep + os.environ.get("PATH", "")
            return str(candidate.resolve())
    raise RuntimeError(
        "ffmpeg.exe was not found. Install FFmpeg, add its bin directory to PATH, "
        "or pass --ffmpeg-path C:\\path\\to\\ffmpeg\\bin"
    )


def read_manifest(manifest_path):
    """Read all rows and retain one ordered entry for each referenced audio file."""
    with Path(manifest_path).open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        fields = reader.fieldnames or []
        required = {"ImageID", "FileName"}
        missing = required.difference(fields)
        if missing:
            raise ValueError(
                "Manifest must contain {}".format(
                    ", ".join(sorted(missing))
                )
            )
        rows = []
        filenames = OrderedDict()
        for row in reader:
            filename = str(row.get("FileName", "")).strip()
            image_id = str(row.get("ImageID", "")).strip()
            if not filename or not image_id:
                continue
            row = dict(row)
            row["FileName"] = filename
            rows.append(row)
            filenames.setdefault(filename, None)
    return fields, rows, list(filenames)


def load_completed_progress(progress_path):
    completed = {}
    progress_path = Path(progress_path)
    if not progress_path.is_file():
        return completed
    with progress_path.open("r", encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            filename = str(row.get("FileName", "")).strip()
            if filename and row.get("Status") == "ok":
                completed[filename] = str(row.get("Caption", "")).strip()
    return completed


def append_progress(progress_path, filename, status, caption="", error=""):
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
                "Status": status,
                "Caption": caption,
                "Error": error,
            }
        )
        handle.flush()


def transcribe_pending(
    filenames,
    audio_dir,
    completed,
    progress_path,
    model_size,
    device,
    beam_size,
):
    """Transcribe each pending file once, even when it appears on multiple image rows."""
    import torch
    try:
        import whisper
    except ImportError as error:
        raise RuntimeError(
            "The Whisper package is required. Install it with "
            "`python -m pip install -U openai-whisper`."
        ) from error

    if device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable. Use --device cpu or fix the GPU environment.")

    pending = [filename for filename in filenames if filename not in completed]
    missing = []
    ready = []
    for filename in pending:
        audio_path = Path(audio_dir) / filename
        if audio_path.is_file():
            ready.append((filename, audio_path))
        else:
            missing.append(filename)
            append_progress(progress_path, filename, "error", error="audio file not found")

    if not ready:
        return 0, len(missing)

    model = whisper.load_model(model_size, device=device)
    fp16 = device.startswith("cuda")
    transcribed = 0
    for filename, audio_path in tqdm(ready, desc="Transcribing English audio", unit="file"):
        try:
            result = model.transcribe(
                str(audio_path),
                task="transcribe",
                language="en",
                beam_size=beam_size,
                fp16=fp16,
                temperature=0,
            )
            caption = str(result.get("text", "")).strip()
        except Exception as error:
            append_progress(progress_path, filename, "error", error=str(error))
            continue
        completed[filename] = caption
        append_progress(progress_path, filename, "ok", caption=caption)
        transcribed += 1
    return transcribed, len(missing)


def write_captioned_manifest(fields, rows, captions, output_path):
    """Atomically create a complete manifest with Caption appended as its final column."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_fields = [field for field in fields if field != "Caption"] + ["Caption"]
    temporary_path = output_path.with_suffix(output_path.suffix + ".tmp")
    written = 0
    with temporary_path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=output_fields, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            output_row = dict(row)
            output_row["Caption"] = captions.get(row["FileName"], "")
            writer.writerow(output_row)
            written += 1
    temporary_path.replace(output_path)
    return written


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest",
        type=Path,
        default=DEFAULT_MANIFEST,
        help="Input CSV with ImageID and FileName columns. Default: %(default)s",
    )
    parser.add_argument(
        "--audio-dir",
        type=Path,
        default=DEFAULT_AUDIO_DIR,
        help="Directory containing manifest-referenced audio files. Default: %(default)s",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="Caption-augmented CSV. Caption is appended as the final column. Default: %(default)s",
    )
    parser.add_argument(
        "--progress",
        type=Path,
        default=None,
        help="Per-audio checkpoint CSV. Default: --output with a .progress.csv suffix.",
    )
    parser.add_argument(
        "--model-size",
        default="small.en",
        help="Installed Whisper model name. Default: %(default)s",
    )
    parser.add_argument(
        "--device",
        default="cuda",
        help="Torch device, normally cuda or cpu. Default: %(default)s",
    )
    parser.add_argument(
        "--beam-size",
        type=int,
        default=5,
        help="Whisper beam-search width; must be positive. Default: %(default)s",
    )
    parser.add_argument(
        "--ffmpeg-path",
        default=None,
        help="Path to ffmpeg.exe or the directory containing it. Default: search PATH and common conda locations.",
    )
    parser.add_argument(
        "--no-resume",
        action="store_false",
        dest="resume",
        help="Ignore successful entries in --progress and transcribe all available audio again.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate the manifest and report audio coverage without loading Whisper or writing files.",
    )
    parser.set_defaults(resume=True)
    return parser.parse_args()


def main():
    args = parse_args()
    if args.beam_size < 1:
        raise ValueError("--beam-size must be positive")

    fields, rows, filenames = read_manifest(args.manifest)
    audio_dir = Path(args.audio_dir)
    available = sum((audio_dir / filename).is_file() for filename in filenames)
    if args.dry_run:
        print("Manifest rows: {}".format(len(rows)))
        print("Unique audio files: {}".format(len(filenames)))
        print("Audio files present: {}".format(available))
        print("Audio files missing: {}".format(len(filenames) - available))
        return

    print("Using ffmpeg: {}".format(configure_ffmpeg(args.ffmpeg_path)))
    progress_path = args.progress or args.output.with_suffix(".progress.csv")
    completed = load_completed_progress(progress_path) if args.resume else {}
    transcribed, missing = transcribe_pending(
        filenames,
        audio_dir,
        completed,
        progress_path,
        args.model_size,
        args.device,
        args.beam_size,
    )
    output_rows = write_captioned_manifest(fields, rows, completed, args.output)
    print("Manifest rows: {}".format(len(rows)))
    print("Unique audio files: {}".format(len(filenames)))
    print("Newly transcribed audio files: {}".format(transcribed))
    print("Completed audio files: {}".format(len(completed)))
    print("Audio files missing: {}".format(missing))
    print("Output rows: {}".format(output_rows))
    print("Progress checkpoint: {}".format(progress_path.resolve()))
    print("Captioned manifest: {}".format(Path(args.output).resolve()))


if __name__ == "__main__":
    main()
