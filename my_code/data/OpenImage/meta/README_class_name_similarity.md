# Class-name similarity

Run from the repository root after installing `sentence-transformers`:

```powershell
python my_code/data/OpenImage/meta/class_name_similarity.py `
  --input "my_code/data/OpenImage/meta/Class Names Map.csv" `
  --output-dir outputs/class_name_similarity `
  --model BAAI/bge-m3 --batch-size 32 --device auto --metric cosine `
  --top-k 3 --threshold 0.7 `
  --cache outputs/class_name_similarity/bge-m3_embeddings.npy
```

The command creates a nearest-neighbor CSV, a complete CSV matrix, and a NumPy
matrix. Matrix rows and columns use the same stable key: zero-padded source row
number followed by the display name. The diagonal is excluded from ranking.
Use `--top-k N` to control how many similar classes are exported; the default
is 3. The output contains `Top1Name`/`Top1Cosine` through the selected `TopN`
columns.
Use `--metric cosine`, `--metric euclidean`, or `--metric manhattan` to select
the distance measure. Cosine ranks larger values first; Euclidean and Manhattan
rank smaller values first. With `--threshold T`, cosine keeps values strictly
greater than `T`, while distance metrics keep values strictly less than `T`.
The complete selected-metric matrix is still exported without filtering.
For 20,931 classes, the float32 matrix is about 1.63 GiB before CSV overhead.
The model is downloaded by `sentence-transformers` on first use.

The loader requests `safetensors` weights explicitly. The environment should
use PyTorch 2.6 or newer because current Transformers blocks unsafe legacy
`torch.load` checkpoints on older versions. Upgrade with the command matching
your CUDA installation, for example:

```powershell
python -m pip install --upgrade "torch>=2.6.0" "torchvision" "torchaudio"
```

For CUDA-specific wheels, use the corresponding index URL from the official
PyTorch installation selector.

The `--model` value may also be a local directory containing a downloaded
`BAAI/bge-m3` model. For unstable Hugging Face connections, retry with the
standard transfer backend disabled:

```powershell
$env:HF_HUB_DISABLE_XET = "1"
python my_code/data/OpenImage/meta/class_name_similarity.py --model BAAI/bge-m3
```

If `huggingface.co` is blocked on the current network, configure a reachable
Hugging Face endpoint before running, or download the model on another machine
and use its local directory via `--model C:\models\bge-m3`.

If importing the model fails with `RuntimeError: operator torchvision::nms does
not exist`, the installed PyTorch and torchvision wheels are incompatible.
Install a matching pair from the official PyTorch selector, then verify:

```powershell
python -c "import torch, torchvision; print(torch.__version__, torchvision.__version__)"
```

This is an environment issue, not a missing `sentence-transformers` package.

If the error is `ssl.SSLError: [ASN1: NOT_ENOUGH_DATA]`, repair the conda
environment and optionally point Python at the certifi bundle:

```powershell
conda activate clip
conda install -n clip openssl ca-certificates certifi --update-deps
$env:SSL_CERT_FILE = (python -c "import certifi; print(certifi.where())").Trim()
python my_code/data/OpenImage/meta/class_name_similarity.py --help
```

If it still fails, update the Windows root certificates and inspect recently
installed antivirus or proxy certificates. Do not disable TLS verification.

## Export leaf classes

To export every deepest-level classification from the hierarchy JSON:

```powershell
python my_code/data/OpenImage/meta/extract_leaf_classes.py `
  --input "my_code/data/OpenImage/meta/Hierarchy/temp/Hierarchy_with_TextName.json" `
  --output outputs/leaf_classes.csv
```

The output includes each leaf `LabelName` and `TextName`, its immediate parent,
depth, and the complete label and display-name paths.

## Similarity inside each category

Use the following command to compare child class names only within their own
top-level category:

```powershell
python my_code/data/OpenImage/meta/category_name_similarity.py `
  --input "my_code/data/OpenImage/meta/Hierarchy/n2/Last_Two_Layers_Multi_Members_n2.json" `
  --output-dir "Hierarchy/category_name_similarity" `
  --top-k 5 --metric cosine --threshold 0.7
```

It creates `category_nearest_neighbors.csv`,
`category_pairwise_similarity.csv`, and one square matrix CSV per category in
`category_matrices`. Similarity is never calculated between different
top-level categories. The pairwise output uses the generic `MetricValue` field,
and neighbor outputs use `TopNValue` fields for the selected metric.

## Audio transcription

The filtered `.ogg` files can be transcribed with Whisper:

```powershell
python my_code/data/OpenImage/meta/audio_to_text.py `
  --audio-dir "my_code/data/OpenImage/meta/Hierarchy/n2/filtered_audio/audio" `
  --output-csv "my_code/data/OpenImage/meta/Hierarchy/n2/filtered_audio/audio_captions_n2.csv" `
  --model base --device auto
```

The output CSV contains `ImageID`, `AudioFilename`, `Caption`, and `Status`.
Existing successful or failed rows are reused on later runs; pass
`--overwrite` to transcribe every file again. Install the runtime with
`python -m pip install -U openai-whisper` before running.

Whisper also requires `ffmpeg.exe` to decode `.ogg` files. Install the pip
package below; the script automatically detects its bundled executable and
creates a temporary `ffmpeg.exe` entry for the current process:

```powershell
python -m pip install -U imageio-ffmpeg
```

Alternatively, add FFmpeg's `bin` directory to `PATH`, or pass its executable
or directory explicitly:

```powershell
python my_code/data/OpenImage/meta/audio_to_text.py `
  --ffmpeg-path "C:\ffmpeg\bin"
```

The script also handles an incomplete or older output CSV: missing audio rows
are added as `pending` records before transcription continues.

The default device is `auto`: it uses GPU when CUDA is available and falls back
to CPU when the installed PyTorch is CPU-only. Explicit `--device cuda` requires
`torch.cuda.is_available()` to be true.

## Filter audio by hierarchy

The following command uses the leaf `LabelName` values in the n2 hierarchy,
selects matching rows from `train_annotations_valid_n2.csv`, and copies every
matching `.ogg` file from the training ZIP by `ImageID`:

```powershell
python my_code/data/OpenImage/meta/Hierarchy/n2/filter_audio_by_hierarchy.py `
  --csv "my_code/data/OpenImage/meta/Hierarchy/n2/train_annotations_valid_n2.csv" `
  --json "my_code/data/OpenImage/meta/Hierarchy/n2/Last_Two_Layers_Multi_Members_n2.json" `
  --zip "E:\数据集\Open Image\train\open_images_train_audio.zip" `
  --output-dir "my_code/data/OpenImage/meta/filtered_audio"
```

The default output is rooted at `meta/Hierarchy/n2/filtered_audio`: the filtered
`matched_annotations.csv`, copied `audio` folder, manifest, and missing-ID list
are all at the same output-folder level.

## Download images by hierarchy

Use the matched annotation CSV and n2 hierarchy JSON to download images into
the same two-level category structure:

```powershell
python my_code/data/OpenImage/meta/download_images_by_hierarchy.py `
  --csv "my_code/data/OpenImage/meta/Hierarchy/n2/filtered_audio/matched_annotations.csv" `
  --json "my_code/data/OpenImage/meta/Hierarchy/n2/Last_Two_Layers_Multi_Members_n2.json" `
  --output-dir "my_code/data/OpenImage/meta/Hierarchy/n2/filtered_audio/images" `
  --concurrency 32
```

Images are stored as `images/<Category>/<Class>/<ImageID>.jpg`. The script
prefers `Thumbnail300KURL` and falls back to `OriginalURL`, deduplicates by
`ImageID`, and creates `download_index.sqlite3` plus
`image_download_manifest.csv` in the output directory. Existing files are
skipped unless `--overwrite` is supplied. Progress bars are shown while reading
the annotation CSV and while downloading images; the download bar reports
downloaded, skipped, and failed counts.
