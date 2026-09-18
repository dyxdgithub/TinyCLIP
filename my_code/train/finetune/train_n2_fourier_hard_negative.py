"""Fine-tune TinyCLIP with Fourier-augmented cross-leaf hard negatives.

The script reads all top-N rows from one combined DINOv2 similarity CSV,
keeps a globally unique greedy maximum-similarity matching of ImageIDs, and trains
TinyCLIP on two image-text examples per pair:

* the anchor image with its own caption; and
* the Fourier-augmented hard-negative image with its own caption.

The standard TinyCLIP image-text ``ClipLoss`` is combined with a weighted
margin loss that pushes each anchor away from its paired augmented negative.
Checkpoints include optimizer, scheduler, AMP scaler, and within-epoch state
so ``--resume`` continues from the next training batch.
"""

import argparse
import csv
import functools
import hashlib
import importlib.util
import json
import math
import random
import sys
from contextlib import nullcontext
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as functional
from PIL import Image
from torch import nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

try:
    from torch.utils.tensorboard import SummaryWriter
except ImportError:
    SummaryWriter = None


REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
SRC_DIR = REPOSITORY_ROOT / "src"
FOURIER_AUGMENTOR_PATH = REPOSITORY_ROOT / "my_code" / "module" / "fourier_augmentor.py"
for import_path in (SRC_DIR,):
    if str(import_path) not in sys.path:
        sys.path.insert(0, str(import_path))

from open_clip import ClipLoss, create_model_and_transforms, get_tokenizer


def load_fourier_augmentor():
    """Load the local preprocessing component from its explicit file path."""
    specification = importlib.util.spec_from_file_location(
        "tinyclip_fourier_augmentor", FOURIER_AUGMENTOR_PATH
    )
    if specification is None or specification.loader is None:
        raise ImportError(
            "Unable to load FourierAugmentor from {}".format(
                FOURIER_AUGMENTOR_PATH
            )
        )
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    augmentor = getattr(module, "FourierAugmentor", None)
    if augmentor is None:
        raise ImportError(
            "FourierAugmentor is missing from {}".format(FOURIER_AUGMENTOR_PATH)
        )
    return augmentor


FourierAugmentor = load_fourier_augmentor()


META_DIR = REPOSITORY_ROOT / "my_code" / "data" / "OpenImage" / "meta"
SAMPLED_DIR = META_DIR / "Hierarchy" / "n2" / "300"
DEFAULT_SIMILARITY_CSV = (
    SAMPLED_DIR / "dinov2_similarity_fiftyone_sampled_300_per_class"
    / "top_n_cross_leaf_dinov2_similarity.csv"
)
DEFAULT_MANIFEST = (
    SAMPLED_DIR
    / "Image IDs_with_last_two_layers_multi_members_n2_one_label_per_parent_sampled_300_per_class.csv"
)
DEFAULT_IMAGES_ROOT = SAMPLED_DIR / "images_fiftyone_sampled_300_per_class"
DEFAULT_OUTPUT_DIR = SAMPLED_DIR / "tinyclip_fourier_hard_negative_finetune"
PAIR_FIELDS = [
    "PairIndex",
    "ParentCategory",
    "AnchorLeafCategory",
    "AnchorImageID",
    "AnchorImagePath",
    "AnchorCaption",
    "HardNegativeLeafCategory",
    "HardNegativeImageID",
    "HardNegativeImagePath",
    "HardNegativeCaption",
    "CosineSimilarity",
]
REQUIRED_TABLE_FIELDS = {
    "ParentCategory",
    "SourceLeafCategory",
    "SourceImageID",
    "SourceImagePath",
    "Rank",
    "MatchedLeafCategory",
    "MatchedImageID",
    "MatchedImagePath",
    "CosineSimilarity",
}
CHECKPOINT_VERSION = 1


@dataclass(frozen=True)
class PairCandidate:
    parent: str
    anchor_leaf: str
    anchor_id: str
    anchor_path: Path
    negative_leaf: str
    negative_id: str
    negative_path: Path
    similarity: float
    source_table: Path


@dataclass(frozen=True)
class TrainingPair:
    parent: str
    anchor_leaf: str
    anchor_id: str
    anchor_path: Path
    anchor_caption: str
    negative_leaf: str
    negative_id: str
    negative_path: Path
    negative_caption: str
    similarity: float


class LoRALinear(nn.Module):
    """Frozen linear layer with a trainable low-rank residual."""

    def __init__(self, base: nn.Linear, rank: int, alpha: float, dropout: float):
        super().__init__()
        if rank <= 0:
            raise ValueError("LoRA rank must be positive")
        self.base = base
        for parameter in self.base.parameters():
            parameter.requires_grad = False
        self.rank = rank
        self.scaling = float(alpha) / rank
        self.dropout = nn.Dropout(dropout) if dropout > 0.0 else nn.Identity()
        parameter_options = {
            "device": base.weight.device,
            "dtype": base.weight.dtype,
        }
        self.lora_a = nn.Parameter(
            torch.empty(rank, base.in_features, **parameter_options)
        )
        self.lora_b = nn.Parameter(
            torch.zeros(base.out_features, rank, **parameter_options)
        )
        nn.init.kaiming_uniform_(self.lora_a, a=math.sqrt(5))

    def forward(self, value):
        base_output = self.base(value)
        lora_output = functional.linear(
            functional.linear(self.dropout(value), self.lora_a), self.lora_b
        )
        return base_output + lora_output * self.scaling


class LoRAMultiheadAttention(nn.Module):
    """Low-rank adapters for query/key/value and output attention projections."""

    def __init__(self, base: nn.MultiheadAttention, rank: int, alpha: float, dropout: float):
        super().__init__()
        if base._qkv_same_embed_dim is False:
            raise ValueError("LoRA only supports MultiheadAttention with shared QKV dimensions")
        self.base = base
        for parameter in self.base.parameters():
            parameter.requires_grad = False
        width = base.embed_dim
        self.scaling = float(alpha) / rank
        self.dropout = nn.Dropout(dropout) if dropout > 0.0 else nn.Identity()
        parameter_options = {
            "device": base.in_proj_weight.device,
            "dtype": base.in_proj_weight.dtype,
        }
        self.qkv_a = nn.Parameter(torch.empty(rank, width, **parameter_options))
        self.qkv_b = nn.Parameter(
            torch.zeros(3 * width, rank, **parameter_options)
        )
        self.out_a = nn.Parameter(torch.empty(rank, width, **parameter_options))
        self.out_b = nn.Parameter(torch.zeros(width, rank, **parameter_options))
        nn.init.kaiming_uniform_(self.qkv_a, a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.out_a, a=math.sqrt(5))

    def _qkv_weight(self):
        return self.base.in_proj_weight + (self.qkv_b @ self.qkv_a) * self.scaling

    def _out_weight(self):
        return self.base.out_proj.weight + (self.out_b @ self.out_a) * self.scaling

    def forward(
        self,
        query,
        key,
        value,
        key_padding_mask=None,
        need_weights=True,
        attn_mask=None,
        average_attn_weights=True,
        is_causal=False,
    ):
        if self.base.batch_first:
            raise ValueError("LoRA MultiheadAttention requires batch_first=False")
        # The TinyCLIP attention blocks use this standard PyTorch path.
        return functional.multi_head_attention_forward(
            query,
            key,
            value,
            self.base.embed_dim,
            self.base.num_heads,
            self._qkv_weight(),
            self.base.in_proj_bias,
            self.base.bias_k,
            self.base.bias_v,
            self.base.add_zero_attn,
            self.base.dropout,
            self._out_weight(),
            self.base.out_proj.bias,
            training=self.training,
            key_padding_mask=key_padding_mask,
            need_weights=need_weights,
            attn_mask=attn_mask,
            average_attn_weights=average_attn_weights,
            is_causal=is_causal,
        )


class FourierPairDataset(Dataset):
    def __init__(self, pairs, transform, phi, size_policy):
        self.pairs = pairs
        self.transform = transform
        self.phi = phi
        self.size_policy = size_policy
        # Created lazily in each DataLoader worker. This keeps the dataset
        # pickle-safe under Windows' spawn multiprocessing method.
        self._augmentor = None

    def __len__(self):
        return len(self.pairs)

    def __getstate__(self):
        state = self.__dict__.copy()
        state["_augmentor"] = None
        return state

    def _get_augmentor(self):
        if self._augmentor is None:
            self._augmentor = FourierAugmentor(
                phi=self.phi, size_policy=self.size_policy
            )
        return self._augmentor

    def __getitem__(self, index):
        pair = self.pairs[index]
        with Image.open(pair.anchor_path) as anchor_file:
            anchor = anchor_file.convert("RGB")
        with Image.open(pair.negative_path) as negative_file:
            negative = negative_file.convert("RGB")
        augmented_negative = self._get_augmentor()(anchor, negative)
        return (
            self.transform(anchor),
            self.transform(augmented_negative),
            pair.anchor_caption,
            pair.negative_caption,
        )


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--similarity-csv", type=Path, default=DEFAULT_SIMILARITY_CSV, help="Combined top-N cross-leaf DINOv2 similarity CSV. All valid rows are ranked by cosine similarity before global ImageID-unique pair selection. Default: %(default)s")
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST, help="CSV mapping ImageID values to captions. Default: %(default)s")
    parser.add_argument("--images-root", type=Path, default=DEFAULT_IMAGES_ROOT, help="Local parent/leaf image root. When a CSV image path does not exist on this machine, the script rebuilds it as <images-root>/<parent>/<leaf>/<filename>. Default: %(default)s")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR, help="Directory for prepared pairs, logs, and checkpoints. Default: %(default)s")
    parser.add_argument("--prepared-pairs", type=Path, default=None, help="Auditable selected-pair CSV. Default: <output-dir>/prepared_pairs.csv")
    parser.add_argument("--model", default="TinyCLIP-ViT-40M-32-Text-19M", help="TinyCLIP model configuration name. Default: %(default)s")
    parser.add_argument("--pretrained", default="LAION400M", help="Registered pretrained tag or local TinyCLIP checkpoint path. The user-run script downloads a registered tag only when it is absent from the model cache. Default: %(default)s")
    parser.add_argument("--cache-dir", type=Path, default=None, help="Optional local cache for a registered --pretrained checkpoint. Default: the open_clip cache location")
    parser.add_argument("--fine-tune-mode", choices=("full", "lora"), default="lora", help="full updates both encoder towers and logit scale; lora freezes base weights and trains adapters in both encoder towers plus logit scale. Default: %(default)s")
    parser.add_argument("--lora-rank", type=int, default=16, help="Adapter rank in lora mode. Ignored in full mode. Default: %(default)s")
    parser.add_argument("--lora-alpha", type=float, default=32.0, help="Adapter scaling numerator in lora mode. Ignored in full mode. Default: %(default)s")
    parser.add_argument("--lora-dropout", type=float, default=0.05, help="Dropout applied inside LoRA adapters in lora mode. Ignored in full mode. Default: %(default)s")
    parser.add_argument("--epochs", type=int, default=10, help="Total epochs, including epochs already completed by --resume. Default: %(default)s")
    parser.add_argument("--batch-size", type=int, default=32, help="Number of image pairs per GPU batch; the ClipLoss sees twice this many image-text items. Default: %(default)s")
    parser.add_argument("--learning-rate", type=float, default=None, help="AdamW learning rate. Default: 1e-5 in full mode and 1e-4 in lora mode.")
    parser.add_argument("--weight-decay", type=float, default=0.2, help="AdamW decay for matrix parameters. Biases, normalization parameters, and logit scale use zero decay. Default: %(default)s")
    parser.add_argument("--beta1", type=float, default=0.9, help="AdamW beta1. Default: %(default)s")
    parser.add_argument("--beta2", type=float, default=0.98, help="AdamW beta2. Default: %(default)s")
    parser.add_argument("--eps", type=float, default=1e-6, help="AdamW epsilon. Default: %(default)s")
    parser.add_argument("--warmup-steps", type=int, default=200, help="Linear warmup steps before cosine decay. Default: %(default)s")
    parser.add_argument("--negative-loss-weight", type=float, default=1.0, help="Multiplier lambda for the specified-negative margin contrastive loss. Default: %(default)s")
    parser.add_argument("--negative-margin", type=float, default=0.2, help="Maximum allowed cosine similarity for an anchor and its augmented hard negative. The extra loss is mean(relu(similarity - margin)). Default: %(default)s")
    parser.add_argument("--fourier-phi", type=float, default=0.1, help="Centered low-frequency side-length proportion passed to FourierAugmentor. Accepted range: (0, 1]. Default: %(default)s")
    parser.add_argument("--fourier-size-policy", choices=("anchor-to-negative", "error"), default="anchor-to-negative", help="FourierAugmentor behavior when original image sizes differ. Default: %(default)s")
    parser.add_argument("--validation-fraction", type=float, default=0.1, help="Fraction of globally unique pairs assigned to validation. Set 0 to disable validation and best.pt selection. Default: %(default)s")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for pair split, data order, and PyTorch. Default: %(default)s")
    parser.add_argument("--num-workers", type=int, default=4, help="DataLoader worker processes. Set 0 for in-process loading. Default: %(default)s")
    parser.add_argument("--device", default="cuda", help="Training device. Default: %(default)s. Use cpu only when GPU execution is intentionally unavailable.")
    parser.add_argument("--gpu", type=int, choices=(0, 1, 2, 3), default=0, help="CUDA logical GPU index used when --device is CUDA. Available choices: 0, 1, 2, 3. Default: %(default)s")
    parser.add_argument("--precision", choices=("amp", "amp_bfloat16", "fp32"), default="amp", help="CUDA precision mode. amp uses fp16 autocast and GradScaler; amp_bfloat16 uses bf16 autocast; fp32 disables autocast. Default: %(default)s")
    parser.add_argument("--grad-clip-norm", type=float, default=1.0, help="Maximum global gradient norm. Set 0 to disable clipping. Default: %(default)s")
    parser.add_argument("--checkpoint-steps", type=int, default=500, help="Write last.pt every N optimizer steps to resume from the next batch after interruption. Default: %(default)s")
    parser.add_argument("--save-every", type=int, default=1, help="Write epoch_NNN.pt every N completed epochs. Default: %(default)s")
    parser.add_argument("--validate-every", type=int, default=1, help="Run validation every N completed epochs. Set 0 to disable validation. Default: %(default)s")
    parser.add_argument("--log-every", type=int, default=20, help="Refresh loss postfix and append JSONL metrics every N optimizer steps. Default: %(default)s")
    parser.add_argument("--tensorboard-dir", type=Path, default=None, help="TensorBoard event-log directory. Default: <output-dir>/tensorboard")
    parser.add_argument("--progress-refresh-seconds", type=float, default=0.1, help="Minimum seconds between real-time tqdm progress refreshes. Default: %(default)s")
    parser.add_argument("--caption-column", default="caption", help="Caption column in --manifest. Default: %(default)s")
    parser.add_argument("--image-id-column", default="ImageID", help="ImageID column in --manifest. Default: %(default)s")
    parser.add_argument("--caption-conflict", choices=("error", "first", "longest"), default="longest", help="Behavior when a manifest ImageID has nonidentical nonempty captions. longest selects the longest caption and keeps the first CSV occurrence on equal lengths. Default: %(default)s")
    parser.add_argument("--resume", type=Path, default=None, help="Checkpoint produced by this script. It restores model, optimizer, scheduler, AMP scaler, RNG, epoch, and next batch. Default: disabled")
    args = parser.parse_args()
    for name in ("lora_rank", "epochs", "batch_size", "num_workers", "warmup_steps", "checkpoint_steps", "save_every", "log_every"):
        if getattr(args, name) <= 0:
            parser.error("--{} must be positive".format(name.replace("_", "-")))
    if args.validate_every < 0:
        parser.error("--validate-every must be zero or positive")
    if args.learning_rate is None:
        args.learning_rate = 1e-5 if args.fine_tune_mode == "full" else 1e-4
    if args.learning_rate <= 0.0 or args.weight_decay < 0.0:
        parser.error("--learning-rate must be positive and --weight-decay must be nonnegative")
    if not 0.0 <= args.lora_dropout < 1.0:
        parser.error("--lora-dropout must be in [0, 1)")
    if args.lora_alpha <= 0.0 or args.negative_loss_weight < 0.0:
        parser.error("--lora-alpha must be positive and --negative-loss-weight must be nonnegative")
    if not -1.0 <= args.negative_margin <= 1.0:
        parser.error("--negative-margin must be in [-1, 1]")
    if not 0.0 < args.fourier_phi <= 1.0:
        parser.error("--fourier-phi must be in (0, 1]")
    if not 0.0 <= args.validation_fraction < 1.0:
        parser.error("--validation-fraction must be in [0, 1)")
    if args.grad_clip_norm < 0.0 or args.progress_refresh_seconds <= 0.0:
        parser.error("--grad-clip-norm must be nonnegative and --progress-refresh-seconds must be positive")
    return args


def read_csv_header(path):
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return csv.DictReader(handle).fieldnames or []


def load_captions(path, args):
    headers = read_csv_header(path)
    for field in (args.image_id_column, args.caption_column):
        if field not in headers:
            raise ValueError("Manifest lacks required column {!r}: {}".format(field, path))
    captions = {}
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        with tqdm(desc="Reading captions", unit="row", mininterval=args.progress_refresh_seconds) as progress:
            for row_number, row in enumerate(reader, start=2):
                image_id = (row.get(args.image_id_column) or "").strip()
                caption = (row.get(args.caption_column) or "").strip()
                if not image_id or not caption:
                    progress.update(1)
                    continue
                prior = captions.get(image_id)
                if prior is None:
                    captions[image_id] = caption
                elif prior != caption:
                    if args.caption_conflict == "error":
                        raise ValueError("Conflicting nonempty captions for ImageID {!r} at manifest row {}".format(image_id, row_number))
                    if args.caption_conflict == "longest" and len(caption) > len(prior):
                        captions[image_id] = caption
                progress.update(1)
    if not captions:
        raise ValueError("No nonempty ImageID/caption mappings found in {}".format(path))
    return captions


def candidate_sort_key(candidate):
    return (
        -candidate.similarity,
        candidate.anchor_id,
        candidate.negative_id,
        str(candidate.anchor_path),
        str(candidate.negative_path),
        str(candidate.source_table),
    )


def count_csv_rows(path, refresh_seconds):
    count = 0
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.reader(handle)
        next(reader, None)
        with tqdm(desc="Counting similarity rows", unit="row", mininterval=refresh_seconds) as progress:
            for count, _ in enumerate(reader, start=1):
                progress.update(1)
    return count


def resolve_image_path(recorded_path, parent, leaf, images_root):
    """Use the CSV path when valid; otherwise rebuild it under local images root."""
    recorded_path = recorded_path.strip()
    direct_path = Path(recorded_path)
    if direct_path.is_file():
        return direct_path.resolve(), "recorded"
    filename = Path(recorded_path.replace("\\", "/")).name
    if filename:
        rebuilt_path = images_root / parent / leaf / filename
        if rebuilt_path.is_file():
            return rebuilt_path.resolve(), "rebuilt"
    return direct_path, "missing"


def load_candidates(similarity_csv, args):
    headers = read_csv_header(similarity_csv)
    missing = REQUIRED_TABLE_FIELDS.difference(headers)
    if missing:
        raise ValueError("Similarity CSV {} lacks columns: {}".format(similarity_csv, ", ".join(sorted(missing))))
    total_rows = count_csv_rows(similarity_csv, args.progress_refresh_seconds)
    candidates = []
    path_resolution_counts = {"recorded": 0, "rebuilt": 0, "missing": 0}
    images_root = args.images_root.expanduser()
    with similarity_csv.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        with tqdm(total=total_rows, desc="Reading similarity rows", unit="row", mininterval=args.progress_refresh_seconds) as progress:
            for row_number, row in enumerate(reader, start=2):
                try:
                    similarity = float(row.get("CosineSimilarity") or "")
                except ValueError as error:
                    raise ValueError("Invalid CosineSimilarity at {}:{}".format(similarity_csv, row_number)) from error
                if not math.isfinite(similarity) or similarity >= 1.0:
                    progress.update(1)
                    continue
                anchor_id = (row.get("SourceImageID") or "").strip()
                negative_id = (row.get("MatchedImageID") or "").strip()
                parent = (row.get("ParentCategory") or "").strip()
                anchor_leaf = (row.get("SourceLeafCategory") or "").strip()
                negative_leaf = (row.get("MatchedLeafCategory") or "").strip()
                anchor_path, anchor_resolution = resolve_image_path(
                    row.get("SourceImagePath") or "", parent, anchor_leaf, images_root
                )
                negative_path, negative_resolution = resolve_image_path(
                    row.get("MatchedImagePath") or "", parent, negative_leaf, images_root
                )
                path_resolution_counts[anchor_resolution] += 1
                path_resolution_counts[negative_resolution] += 1
                if anchor_id and negative_id and anchor_id != negative_id:
                    candidates.append(PairCandidate(
                        parent=parent,
                        anchor_leaf=anchor_leaf,
                        anchor_id=anchor_id,
                        anchor_path=anchor_path,
                        negative_leaf=negative_leaf,
                        negative_id=negative_id,
                        negative_path=negative_path,
                        similarity=similarity,
                        source_table=similarity_csv,
                    ))
                progress.update(1)
    return candidates, path_resolution_counts


def build_unique_pairs(candidates, captions):
    """Greedily select descending-similarity pairs with globally unique ImageIDs."""
    pairs = []
    used_image_ids = set()
    for candidate in sorted(candidates, key=candidate_sort_key):
        if candidate.anchor_id not in captions or candidate.negative_id not in captions:
            continue
        if not candidate.anchor_path.is_file() or not candidate.negative_path.is_file():
            continue
        if candidate.anchor_id in used_image_ids or candidate.negative_id in used_image_ids:
            continue
        used_image_ids.add(candidate.anchor_id)
        used_image_ids.add(candidate.negative_id)
        pairs.append(TrainingPair(
            parent=candidate.parent,
            anchor_leaf=candidate.anchor_leaf,
            anchor_id=candidate.anchor_id,
            anchor_path=candidate.anchor_path.resolve(),
            anchor_caption=captions[candidate.anchor_id],
            negative_leaf=candidate.negative_leaf,
            negative_id=candidate.negative_id,
            negative_path=candidate.negative_path.resolve(),
            negative_caption=captions[candidate.negative_id],
            similarity=candidate.similarity,
        ))
    if not pairs:
        raise ValueError("No usable pairs remain after similarity, caption, path, and ImageID-uniqueness filtering")
    return pairs


def write_prepared_pairs(path, pairs):
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
                "AnchorCaption": pair.anchor_caption,
                "HardNegativeLeafCategory": pair.negative_leaf,
                "HardNegativeImageID": pair.negative_id,
                "HardNegativeImagePath": str(pair.negative_path),
                "HardNegativeCaption": pair.negative_caption,
                "CosineSimilarity": "{:.8f}".format(pair.similarity),
            })
    temporary.replace(path)


def pair_signature(pairs):
    encoded = json.dumps([asdict(pair) | {"anchor_path": str(pair.anchor_path), "negative_path": str(pair.negative_path)} for pair in pairs], ensure_ascii=True, sort_keys=True).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def split_pairs(pairs, validation_fraction, seed):
    indices = list(range(len(pairs)))
    random.Random(seed).shuffle(indices)
    if validation_fraction == 0.0 or len(pairs) < 2:
        return [pairs[index] for index in indices], []
    validation_count = max(1, int(round(len(pairs) * validation_fraction)))
    validation_count = min(validation_count, len(pairs) - 1)
    validation_indices = set(indices[:validation_count])
    train_pairs = [pair for index, pair in enumerate(pairs) if index not in validation_indices]
    validation_pairs = [pair for index, pair in enumerate(pairs) if index in validation_indices]
    return train_pairs, validation_pairs


def replace_submodule(root, name, module):
    parent_name, _, child_name = name.rpartition(".")
    parent = root.get_submodule(parent_name) if parent_name else root
    setattr(parent, child_name, module)


def add_lora_adapters(model, args):
    for parameter in model.parameters():
        parameter.requires_grad = False

    replaced = {"image_attention": 0, "text_attention": 0, "image_linear": 0, "text_linear": 0}
    towers = (("image", model.image_encoder_without_ddp), ("text", model.text_encoder_without_ddp))
    for tower_name, tower in towers:
        mha_names = [name for name, module in tower.named_modules() if isinstance(module, nn.MultiheadAttention)]
        for name in mha_names:
            replace_submodule(tower, name, LoRAMultiheadAttention(tower.get_submodule(name), args.lora_rank, args.lora_alpha, args.lora_dropout))
            replaced[tower_name + "_attention"] += 1
        for name, module in list(tower.named_modules()):
            if not isinstance(module, nn.Linear):
                continue
            if any(name.startswith(mha_name + ".") for mha_name in mha_names):
                continue
            replace_submodule(tower, name, LoRALinear(module, args.lora_rank, args.lora_alpha, args.lora_dropout))
            replaced[tower_name + "_linear"] += 1

    model.logit_scale.requires_grad = True
    if not any(parameter.requires_grad for parameter in model.parameters()):
        raise RuntimeError("LoRA insertion produced no trainable parameters")
    return replaced


def configure_trainable_parameters(model, args):
    if args.fine_tune_mode == "full":
        for parameter in model.parameters():
            parameter.requires_grad = True
        return {"mode": "full"}
    return add_lora_adapters(model, args)


def make_optimizer(model, args):
    decay_parameters = []
    no_decay_parameters = []
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        if parameter.ndim < 2 or name.endswith("bias") or "norm" in name.lower() or "logit_scale" in name:
            no_decay_parameters.append(parameter)
        else:
            decay_parameters.append(parameter)
    if not decay_parameters and not no_decay_parameters:
        raise RuntimeError("No trainable parameters were configured")
    return AdamW(
        [
            {"params": decay_parameters, "weight_decay": args.weight_decay},
            {"params": no_decay_parameters, "weight_decay": 0.0},
        ],
        lr=args.learning_rate,
        betas=(args.beta1, args.beta2),
        eps=args.eps,
    )


def make_scheduler(optimizer, total_steps, warmup_steps):
    warmup_steps = min(warmup_steps, total_steps)

    def multiplier(step):
        if step < warmup_steps:
            return float(step + 1) / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1.0 + math.cos(math.pi * min(progress, 1.0)))

    return LambdaLR(optimizer, multiplier)


def ensure_device(device_name, gpu_index):
    device = torch.device(device_name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("--device {} was requested, but CUDA is unavailable".format(device_name))
    if device.type == "cuda":
        if device.index is not None and device.index != gpu_index:
            raise ValueError(
                "--device {} conflicts with --gpu {}; use --device cuda or matching values"
                .format(device_name, gpu_index)
            )
        visible_count = torch.cuda.device_count()
        if gpu_index >= visible_count:
            raise ValueError(
                "--gpu {} is unavailable; {} CUDA device(s) are visible to this process"
                .format(gpu_index, visible_count)
            )
        device = torch.device("cuda", gpu_index)
        torch.cuda.set_device(device)
    return device


def autocast_context(precision, device):
    if device.type != "cuda" or precision == "fp32":
        return nullcontext()
    if precision == "amp_bfloat16":
        return torch.amp.autocast("cuda", dtype=torch.bfloat16)
    return torch.amp.autocast("cuda")


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def initialize_data_worker(worker_id, base_seed, epoch):
    worker_seed = base_seed + epoch * 100003 + worker_id
    random.seed(worker_seed)
    np.random.seed(worker_seed)
    torch.manual_seed(worker_seed)


def make_loader(pairs, transform, args, epoch, shuffle):
    dataset = FourierPairDataset(pairs, transform, args.fourier_phi, args.fourier_size_policy)
    generator = torch.Generator()
    generator.manual_seed(args.seed + epoch * 100003 + (0 if shuffle else 1))

    return DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=shuffle,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False,
        persistent_workers=False,
        worker_init_fn=(
            functools.partial(
                initialize_data_worker, base_seed=args.seed, epoch=epoch
            )
            if args.num_workers
            else None
        ),
        generator=generator,
    )


def batch_losses(model, clip_loss_fn, tokenizer, batch, device, autocast, negative_margin):
    anchors, augmented_negatives, anchor_captions, negative_captions = batch
    images = torch.cat((anchors, augmented_negatives), dim=0).to(device, non_blocking=True)
    tokens = tokenizer(list(anchor_captions) + list(negative_captions)).to(device, non_blocking=True)
    with autocast:
        image_features, text_features, logit_scale = model(images, tokens, normalized=True)
        clip_loss = clip_loss_fn(image_features, text_features, logit_scale)
        anchor_features, negative_features = image_features.chunk(2, dim=0)
        pair_similarity = (anchor_features * negative_features).sum(dim=-1)
        negative_loss = functional.relu(pair_similarity - negative_margin).mean()
    return clip_loss, negative_loss, pair_similarity.detach()


def checkpoint_payload(model, optimizer, scheduler, scaler, epoch, next_batch, global_step, pairs_hash, args):
    return {
        "checkpoint_version": CHECKPOINT_VERSION,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "scaler": scaler.state_dict(),
        "epoch": epoch,
        "next_batch": next_batch,
        "global_step": global_step,
        "pair_signature": pairs_hash,
        "fine_tune_mode": args.fine_tune_mode,
        "model_name": args.model,
        "rng": {
            "python": random.getstate(),
            "numpy": np.random.get_state(),
            "torch": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
        },
        "args": vars(args),
    }


def save_checkpoint(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".inprogress")
    temporary.unlink(missing_ok=True)
    torch.save(payload, temporary)
    temporary.replace(path)


def restore_checkpoint(path, model, optimizer, scheduler, scaler, pairs_hash, args, device):
    checkpoint = torch.load(path, map_location=device)
    if checkpoint.get("checkpoint_version") != CHECKPOINT_VERSION:
        raise ValueError("Unsupported checkpoint version in {}".format(path))
    for key, expected in (("pair_signature", pairs_hash), ("fine_tune_mode", args.fine_tune_mode), ("model_name", args.model)):
        if checkpoint.get(key) != expected:
            raise ValueError("Checkpoint {} does not match current {}".format(path, key))
    model.load_state_dict(checkpoint["model"], strict=True)
    optimizer.load_state_dict(checkpoint["optimizer"])
    scheduler.load_state_dict(checkpoint["scheduler"])
    scaler.load_state_dict(checkpoint["scaler"])
    rng = checkpoint.get("rng", {})
    if rng.get("python") is not None:
        random.setstate(rng["python"])
    if rng.get("numpy") is not None:
        np.random.set_state(rng["numpy"])
    if rng.get("torch") is not None:
        torch.set_rng_state(rng["torch"])
    if torch.cuda.is_available() and rng.get("cuda") is not None:
        torch.cuda.set_rng_state_all(rng["cuda"])
    return checkpoint["epoch"], checkpoint["next_batch"], checkpoint["global_step"]


def append_metric(path, payload):
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=True) + "\n")


def create_tensorboard_writer(path, args, run_metadata):
    if SummaryWriter is None:
        raise RuntimeError(
            "TensorBoard logging requires the 'tensorboard' package. "
            "Install requirements-training.txt in the active environment."
        )
    writer = SummaryWriter(log_dir=str(path), flush_secs=1)
    writer.add_text("run/config", "<pre>{}</pre>".format(
        json.dumps(run_metadata, ensure_ascii=True, indent=2, default=str)
    ))
    writer.flush()
    return writer


def write_tensorboard_scalars(writer, prefix, metrics, step):
    for name, value in metrics.items():
        writer.add_scalar("{}/{}".format(prefix, name), float(value), step)
    writer.flush()


def run_validation(model, pairs, transform, tokenizer, clip_loss_fn, args, device, autocast):
    if not pairs:
        return None
    loader = make_loader(pairs, transform, args, epoch=0, shuffle=False)
    total_loss = total_clip_loss = total_negative_loss = total_similarity = 0.0
    total_samples = 0
    model.eval()
    with torch.no_grad():
        with tqdm(total=len(loader), desc="Validation", unit="batch", mininterval=args.progress_refresh_seconds) as progress:
            for batch in loader:
                clip_loss, negative_loss, similarities = batch_losses(model, clip_loss_fn, tokenizer, batch, device, autocast, args.negative_margin)
                count = similarities.numel()
                total_clip_loss += clip_loss.item() * count
                total_negative_loss += negative_loss.item() * count
                total_loss += (clip_loss + args.negative_loss_weight * negative_loss).item() * count
                total_similarity += similarities.float().sum().item()
                total_samples += count
                progress.update(1)
    return {
        "loss": total_loss / total_samples,
        "clip_loss": total_clip_loss / total_samples,
        "negative_loss": total_negative_loss / total_samples,
        "anchor_negative_similarity": total_similarity / total_samples,
    }


def main():
    args = parse_args()
    similarity_csv = args.similarity_csv.expanduser()
    manifest = args.manifest.expanduser()
    output_dir = args.output_dir.expanduser()
    prepared_pairs_path = args.prepared_pairs.expanduser() if args.prepared_pairs else output_dir / "prepared_pairs.csv"
    tensorboard_dir = args.tensorboard_dir.expanduser() if args.tensorboard_dir else output_dir / "tensorboard"
    if not similarity_csv.is_file():
        raise FileNotFoundError("Combined similarity CSV does not exist: {}".format(similarity_csv))
    if not manifest.is_file():
        raise FileNotFoundError("Caption manifest does not exist: {}".format(manifest))
    if args.resume is not None and not args.resume.expanduser().is_file():
        raise FileNotFoundError("Resume checkpoint does not exist: {}".format(args.resume))

    device = ensure_device(args.device, args.gpu)
    print("Training device: {}".format(device), flush=True)
    seed_everything(args.seed)
    captions = load_captions(manifest, args)
    candidates, path_resolution_counts = load_candidates(similarity_csv, args)
    pairs = build_unique_pairs(candidates, captions)
    pairs_hash = pair_signature(pairs)
    output_dir.mkdir(parents=True, exist_ok=True)
    write_prepared_pairs(prepared_pairs_path, pairs)
    train_pairs, validation_pairs = split_pairs(pairs, args.validation_fraction, args.seed)
    if not train_pairs:
        raise ValueError("No pairs were assigned to training")
    run_metadata = {"arguments": vars(args), "pair_signature": pairs_hash, "candidate_rows": len(candidates), "selected_pairs": len(pairs), "train_pairs": len(train_pairs), "validation_pairs": len(validation_pairs), "image_path_resolution": path_resolution_counts}
    with (output_dir / "run_config.json").open("w", encoding="utf-8") as handle:
        json.dump(run_metadata, handle, ensure_ascii=True, indent=2, default=str)
    tensorboard_writer = create_tensorboard_writer(tensorboard_dir, args, run_metadata)

    cache_dir = str(args.cache_dir.expanduser()) if args.cache_dir else None
    model, train_transform, validation_transform = create_model_and_transforms(
        args.model,
        pretrained=args.pretrained,
        precision="fp32",
        device=device,
        cache_dir=cache_dir,
    )
    lora_summary = configure_trainable_parameters(model, args)
    optimizer = make_optimizer(model, args)
    initial_loader = make_loader(train_pairs, train_transform, args, epoch=0, shuffle=True)
    total_steps = max(1, len(initial_loader) * args.epochs)
    scheduler = make_scheduler(optimizer, total_steps, args.warmup_steps)
    scaler = torch.amp.GradScaler(
        "cuda", enabled=device.type == "cuda" and args.precision == "amp"
    )
    tokenizer = get_tokenizer(args.model)
    clip_loss_fn = ClipLoss(cache_labels=True)
    autocast = autocast_context(args.precision, device)
    metrics_path = output_dir / "metrics.jsonl"
    start_epoch = start_batch = global_step = 0
    if args.resume is not None:
        start_epoch, start_batch, global_step = restore_checkpoint(args.resume.expanduser(), model, optimizer, scheduler, scaler, pairs_hash, args, device)
        print("Resumed from {} at epoch {}, next batch {}, global step {}".format(args.resume.expanduser(), start_epoch + 1, start_batch + 1, global_step), flush=True)

    trainable_parameters = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    all_parameters = sum(parameter.numel() for parameter in model.parameters())
    print("Prepared pairs: {} from {} top-N candidate rows".format(len(pairs), len(candidates)), flush=True)
    print("Image path resolution: {}".format(path_resolution_counts), flush=True)
    print("Train pairs: {}; validation pairs: {}".format(len(train_pairs), len(validation_pairs)), flush=True)
    print("Fine-tune mode: {}; adapter summary: {}".format(args.fine_tune_mode, lora_summary), flush=True)
    print("Trainable parameters: {} / {}".format(trainable_parameters, all_parameters), flush=True)
    best_validation_loss = float("inf")
    last_checkpoint = output_dir / "last.pt"

    for epoch in range(start_epoch, args.epochs):
        loader = make_loader(train_pairs, train_transform, args, epoch=epoch, shuffle=True)
        model.train()
        epoch_clip_loss = epoch_negative_loss = epoch_total_loss = epoch_similarity = 0.0
        epoch_samples = 0
        skipped_batches = start_batch if epoch == start_epoch else 0
        with tqdm(total=len(loader), desc="Epoch {}/{}".format(epoch + 1, args.epochs), unit="batch", mininterval=args.progress_refresh_seconds) as progress:
            if skipped_batches:
                progress.update(skipped_batches)
            for batch_index, batch in enumerate(loader):
                if batch_index < skipped_batches:
                    continue
                optimizer.zero_grad(set_to_none=True)
                clip_loss, negative_loss, similarities = batch_losses(model, clip_loss_fn, tokenizer, batch, device, autocast, args.negative_margin)
                total_loss = clip_loss + args.negative_loss_weight * negative_loss
                scaler.scale(total_loss).backward()
                if args.grad_clip_norm > 0.0:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip_norm)
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
                global_step += 1

                count = similarities.numel()
                epoch_clip_loss += clip_loss.item() * count
                epoch_negative_loss += negative_loss.item() * count
                epoch_total_loss += total_loss.item() * count
                epoch_similarity += similarities.float().sum().item()
                epoch_samples += count
                progress.update(1)
                if global_step % args.log_every == 0:
                    postfix = {
                        "loss": "{:.4f}".format(total_loss.item()),
                        "clip": "{:.4f}".format(clip_loss.item()),
                        "neg": "{:.4f}".format(negative_loss.item()),
                        "pair_sim": "{:.4f}".format(similarities.float().mean().item()),
                        "lr": "{:.2e}".format(optimizer.param_groups[0]["lr"]),
                    }
                    progress.set_postfix(postfix)
                    step_metrics = {
                        "loss": total_loss.item(),
                        "clip_loss": clip_loss.item(),
                        "negative_loss": negative_loss.item(),
                        "anchor_negative_similarity": similarities.float().mean().item(),
                        "learning_rate": optimizer.param_groups[0]["lr"],
                    }
                    append_metric(metrics_path, {"split": "train_step", "epoch": epoch + 1, "global_step": global_step, **step_metrics})
                    write_tensorboard_scalars(
                        tensorboard_writer, "train_step", step_metrics, global_step
                    )
                if global_step % args.checkpoint_steps == 0:
                    save_checkpoint(last_checkpoint, checkpoint_payload(model, optimizer, scheduler, scaler, epoch, batch_index + 1, global_step, pairs_hash, args))

        start_batch = 0
        train_metrics = {
            "loss": epoch_total_loss / epoch_samples,
            "clip_loss": epoch_clip_loss / epoch_samples,
            "negative_loss": epoch_negative_loss / epoch_samples,
            "anchor_negative_similarity": epoch_similarity / epoch_samples,
        }
        append_metric(metrics_path, {"split": "train_epoch", "epoch": epoch + 1, "global_step": global_step, **train_metrics})
        write_tensorboard_scalars(
            tensorboard_writer, "train_epoch", train_metrics, epoch + 1
        )
        validation_metrics = None
        if validation_pairs and args.validate_every and (epoch + 1) % args.validate_every == 0:
            validation_metrics = run_validation(model, validation_pairs, validation_transform, tokenizer, clip_loss_fn, args, device, autocast)
            append_metric(metrics_path, {"split": "validation", "epoch": epoch + 1, "global_step": global_step, **validation_metrics})
            write_tensorboard_scalars(
                tensorboard_writer, "validation", validation_metrics, epoch + 1
            )
            print("Epoch {} validation: {}".format(epoch + 1, validation_metrics), flush=True)
            if validation_metrics["loss"] < best_validation_loss:
                best_validation_loss = validation_metrics["loss"]
                save_checkpoint(output_dir / "best.pt", checkpoint_payload(model, optimizer, scheduler, scaler, epoch + 1, 0, global_step, pairs_hash, args))
        if (epoch + 1) % args.save_every == 0:
            save_checkpoint(output_dir / "checkpoints" / "epoch_{:03d}.pt".format(epoch + 1), checkpoint_payload(model, optimizer, scheduler, scaler, epoch + 1, 0, global_step, pairs_hash, args))
        save_checkpoint(last_checkpoint, checkpoint_payload(model, optimizer, scheduler, scaler, epoch + 1, 0, global_step, pairs_hash, args))
        print("Epoch {} train: {}".format(epoch + 1, train_metrics), flush=True)

    tensorboard_writer.close()
    print("Training complete. Last checkpoint: {}".format(last_checkpoint.resolve()))
    print("TensorBoard logs: {}".format(tensorboard_dir.resolve()))
    if validation_pairs:
        print("Best validation checkpoint: {}".format((output_dir / "best.pt").resolve()))
    print("Prepared pairs: {}".format(prepared_pairs_path.resolve()))


if __name__ == "__main__":
    main()
