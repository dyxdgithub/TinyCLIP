"""Fine-tune TinyCLIP with LoRA and the symmetric CLIP image-text loss.

The split table supplies image paths and split names. Captions are joined from
the n2 metadata table by (ImageID, ParentLabelName, LabelName). The first
nonempty caption for a repeated key is used deterministically. Validation and
test Top1Acc classify each unique image against all n2 DisplayName prompts.
"""

import argparse
import csv
import math
import random
from contextlib import nullcontext
from pathlib import Path

import torch
from PIL import Image
from torch import nn
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from open_clip import ClipLoss, create_model_and_transforms, get_tokenizer


SCRIPT_DIR = Path(__file__).resolve().parent
N2_DIR = SCRIPT_DIR.parents[1] / "data/OpenImage/meta/Hierarchy/n2"
MODEL_NAME = "TinyCLIP-auto-ViT-45M-32-Text-18M"
DEFAULT_OUTPUT = SCRIPT_DIR / "output/clip_lora_TinyCLIP-auto-ViT-45M-32-Text-18M-LAION400M"
JOIN_FIELDS = ("ImageID", "ParentLabelName", "LabelName")


class LoRAWeight(nn.Module):
    """Add a trainable low-rank update to a frozen linear weight matrix."""

    def __init__(self, weight, rank, alpha):
        super().__init__()
        self.scaling = alpha / rank
        self.lora_a = nn.Parameter(torch.empty(rank, weight.shape[1], device=weight.device, dtype=weight.dtype))
        self.lora_b = nn.Parameter(torch.zeros(weight.shape[0], rank, device=weight.device, dtype=weight.dtype))
        nn.init.kaiming_uniform_(self.lora_a, a=math.sqrt(5))

    def forward(self, weight):
        return weight + (self.lora_b @ self.lora_a) * self.scaling


def add_lora_adapters(model, rank, alpha):
    for parameter in model.parameters():
        parameter.requires_grad = False
    counts = {"image_attention": 0, "text_attention": 0, "image_linear": 0, "text_linear": 0}
    towers = (("image", model.image_encoder_without_ddp), ("text", model.text_encoder_without_ddp))
    for tower_name, tower in towers:
        modules = list(tower.named_modules())
        for _, module in modules:
            if isinstance(module, nn.MultiheadAttention):
                if not module._qkv_same_embed_dim:
                    raise ValueError("LoRA requires shared QKV dimensions in MultiheadAttention")
                nn.utils.parametrize.register_parametrization(module, "in_proj_weight", LoRAWeight(module.in_proj_weight, rank, alpha))
                counts[tower_name + "_attention"] += 1
            elif isinstance(module, nn.Linear):
                nn.utils.parametrize.register_parametrization(module, "weight", LoRAWeight(module.weight, rank, alpha))
                counts[tower_name + "_linear"] += 1
    if counts["image_attention"] == 0 or counts["text_attention"] == 0:
        raise RuntimeError("LoRA did not find attention blocks in both TinyCLIP towers")
    model.logit_scale.requires_grad = True
    return counts


def arguments(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--split-csv", type=Path, default=N2_DIR / "Image_IDs_n2_all_downloaded_split_8_1_1.csv", help="CSV with ImageID, RelativePath, parent/child labels and Split.")
    parser.add_argument("--caption-csv", type=Path, default=N2_DIR / "Image IDs_with_last_two_layers_multi_members_n2_one_label_per_parent.csv", help="Metadata CSV with caption and the three join columns.")
    parser.add_argument("--images-dir", type=Path, default=N2_DIR / "images_all", help="Root containing RelativePath entries; set this to the server's image directory.")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT, help="Checkpoint and TensorBoard output directory.")
    parser.add_argument("--pretrained", default="LAION400M", help="Registered TinyCLIP pretrained tag or local checkpoint; a registered tag may download when user runs this script.")
    parser.add_argument("--cache-dir", type=Path, default=None, help="Optional pretrained weight cache; default is open_clip's cache.")
    parser.add_argument("--lora-rank", type=int, default=16, help="Rank of each LoRA update in both encoders; must be positive.")
    parser.add_argument("--lora-alpha", type=float, default=32.0, help="LoRA scale numerator, divided by rank; must be positive.")
    parser.add_argument("--class-prompt-template", default="a photo of {label}", help="Text prompt for each DisplayName; must contain {label}.")
    parser.add_argument("--class-text-batch-size", type=int, default=256, help="Number of class prompts encoded together during validation/test.")
    parser.add_argument("--gpu", type=int, default=0, help="Nonnegative logical CUDA GPU index; default 0. Use --device cpu only if CUDA is unavailable.")
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda", help="Training device.")
    parser.add_argument("--precision", choices=("amp", "amp_bfloat16", "fp32"), default="amp", help="CUDA autocast mode; CPU uses fp32.")
    parser.add_argument("--epochs", type=int, default=6, help="Total epochs, including completed epochs when resuming.")
    parser.add_argument("--batch-size", type=int, default=64, help="Image-text pairs per GPU batch; lower this if GPU memory is insufficient.")
    parser.add_argument("--workers", type=int, default=8, help="DataLoader worker processes; 0 loads in the main process.")
    parser.add_argument("--lr", type=float, default=1e-4, help="Peak AdamW learning rate.")
    parser.add_argument("--weight-decay", type=float, default=0.2, help="AdamW decay for matrix weights; biases, norms and logit scale have zero decay.")
    parser.add_argument("--beta1", type=float, default=0.9, help="AdamW beta1.")
    parser.add_argument("--beta2", type=float, default=0.98, help="AdamW beta2.")
    parser.add_argument("--eps", type=float, default=1e-6, help="AdamW epsilon.")
    parser.add_argument("--warmup-steps", type=int, default=8000, help="Optimizer steps of linear warmup, capped at total steps; followed by cosine decay.")
    parser.add_argument("--seed", type=int, default=0, help="Random seed for batch order and model operations.")
    parser.add_argument("--checkpoint-steps", type=int, default=500, help="Save resumable last.pt after this many optimizer steps.")
    parser.add_argument("--bad-image-policy", choices=("skip", "error"), default="skip", help="Skip unreadable images and record them in <output-dir>/skipped_images.csv, or stop at the first error.")
    parser.add_argument("--resume", type=Path, default=None, help="Resume from this script's last.pt, including the next unprocessed batch; default is a new run.")
    args = parser.parse_args(argv)
    for name in ("epochs", "batch_size", "checkpoint_steps", "lora_rank", "class_text_batch_size"):
        if getattr(args, name) < 1:
            parser.error("--{} must be positive".format(name.replace("_", "-")))
    if args.workers < 0 or args.gpu < 0 or args.warmup_steps < 0 or args.weight_decay < 0 or args.lr <= 0:
        parser.error("workers, gpu, warmup-steps and weight-decay must be nonnegative; lr must be positive")
    if not (0 <= args.beta1 < 1 and 0 <= args.beta2 < 1 and args.eps > 0):
        parser.error("beta1/beta2 must be in [0, 1), and eps must be positive")
    if args.lora_alpha <= 0 or "{label}" not in args.class_prompt_template:
        parser.error("--lora-alpha must be positive and --class-prompt-template must contain {label}")
    try:
        args.class_prompt_template.format(label="example")
    except (KeyError, ValueError) as error:
        parser.error("invalid --class-prompt-template: {}".format(error))
    return args


def read_rows(split_csv, caption_csv):
    captions = {}
    with caption_csv.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        required = set(JOIN_FIELDS) | {"caption"}
        if not required.issubset(reader.fieldnames or []):
            raise ValueError("Caption CSV lacks: " + ", ".join(sorted(required - set(reader.fieldnames or []))))
        for row in tqdm(reader, desc="Reading captions", unit="row"):
            key = tuple((row[field] or "").strip() for field in JOIN_FIELDS)
            caption = (row["caption"] or "").strip()
            if caption and key not in captions:
                captions[key] = caption

    rows = {"train": [], "validation": [], "test": []}
    class_names = {}
    with split_csv.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        required = set(JOIN_FIELDS) | {"RelativePath", "Split", "DisplayName"}
        if not required.issubset(reader.fieldnames or []):
            raise ValueError("Split CSV lacks: " + ", ".join(sorted(required - set(reader.fieldnames or []))))
        for number, row in enumerate(tqdm(reader, desc="Joining samples", unit="row"), 2):
            key = tuple((row[field] or "").strip() for field in JOIN_FIELDS)
            if key not in captions:
                raise ValueError("No nonempty caption for row {}: {}".format(number, key))
            split = (row["Split"] or "").strip().lower()
            if split == "val":
                split = "validation"
            if split not in rows:
                raise ValueError("Unknown Split at row {}: {}".format(number, split))
            image_id, _, label_id = key
            display_name = (row["DisplayName"] or "").strip()
            if not image_id or not label_id or not display_name:
                raise ValueError("Missing ImageID, LabelName or DisplayName at row {}".format(number))
            if label_id in class_names and class_names[label_id] != display_name:
                raise ValueError("Conflicting DisplayName for LabelName {}".format(label_id))
            class_names[label_id] = display_name
            relative = Path((row["RelativePath"] or "").strip().replace("\\", "/"))
            if relative.is_absolute() or not relative.parts or ".." in relative.parts:
                raise ValueError("Unsafe RelativePath at row {}: {}".format(number, relative))
            rows[split].append((relative, captions[key], image_id, (label_id,)))
    for split, entries in rows.items():
        if not entries:
            raise ValueError("Split {} has no samples".format(split))
    return rows, class_names


def unique_images(entries):
    """Count each validation/test ImageID once and accept any of its labels."""
    positions = {}
    grouped = []
    for relative, caption, image_id, labels in entries:
        if image_id not in positions:
            positions[image_id] = len(grouped)
            grouped.append((relative, caption, image_id, set(labels)))
        else:
            grouped[positions[image_id]][3].update(labels)
    return [(relative, caption, image_id, tuple(sorted(labels))) for relative, caption, image_id, labels in grouped]


class ImageCaptionDataset(Dataset):
    def __init__(self, entries, images_dir, transform, split, bad_image_policy):
        self.entries = entries
        self.images_dir = images_dir
        self.transform = transform
        self.split = split
        self.bad_image_policy = bad_image_policy

    def __len__(self):
        return len(self.entries)

    def __getitem__(self, index):
        relative, caption, _, labels = self.entries[index]
        try:
            with Image.open(self.images_dir / relative) as image:
                return self.transform(image.convert("RGB")), caption, labels, None
        except OSError as error:
            if self.bad_image_policy == "error":
                raise OSError("Cannot load image {}: {}".format(self.images_dir / relative, error)) from error
            return None, None, None, (self.split, relative.as_posix(), str(error))


def collate_image_captions(samples):
    valid = [(image, caption, labels) for image, caption, labels, error in samples if error is None]
    bad = [error for _, _, _, error in samples if error is not None]
    if not valid:
        return None, [], [], bad
    images, captions, labels = zip(*valid)
    return torch.stack(images), list(captions), list(labels), bad


def record_bad_images(bad, report_path, seen):
    new = []
    pending = set()
    for item in bad:
        key = (item[0], item[1])
        if key not in seen and key not in pending:
            new.append(item)
            pending.add(key)
    if not new:
        return
    report_path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not report_path.exists() or report_path.stat().st_size == 0
    with report_path.open("a", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        if write_header:
            writer.writerow(("Split", "RelativePath", "Error"))
        for split, relative, error in new:
            writer.writerow((split, relative, error))
            seen.add((split, relative))
            tqdm.write("Skipping unreadable image [{}] {}: {}".format(split, relative, error))


def known_bad_images(report_path):
    if not report_path.exists():
        return set()
    with report_path.open(encoding="utf-8", newline="") as handle:
        return {(row["Split"], row["RelativePath"]) for row in csv.DictReader(handle)}


def optimizer_for(model, args):
    decay, no_decay = [], []
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        target = no_decay if parameter.ndim < 2 or name.endswith("bias") or "norm" in name.lower() or "logit_scale" in name else decay
        target.append(parameter)
    return AdamW([{"params": decay, "weight_decay": args.weight_decay}, {"params": no_decay, "weight_decay": 0.0}], lr=args.lr, betas=(args.beta1, args.beta2), eps=args.eps)


def learning_rate(step, total, warmup):
    warmup = min(warmup, total)
    if step < warmup:
        return (step + 1) / max(1, warmup)
    return 0.5 * (1 + math.cos(math.pi * min(1, (step - warmup) / max(1, total - warmup))))


def batches_for_epoch(size, batch_size, seed, epoch):
    generator = torch.Generator().manual_seed(seed + epoch)
    order = torch.randperm(size, generator=generator).tolist()
    return [order[start:start + batch_size] for start in range(0, size, batch_size)]


def loader(dataset, batch_indices, args):
    # A batch sampler avoids decoding batches already completed before --resume.
    return DataLoader(dataset, batch_sampler=batch_indices, num_workers=args.workers, pin_memory=args.device == "cuda", collate_fn=collate_image_captions)


def context(args):
    if args.device == "cpu" or args.precision == "fp32":
        return nullcontext()
    dtype = torch.bfloat16 if args.precision == "amp_bfloat16" else torch.float16
    return torch.autocast("cuda", dtype=dtype)


def forward(model, tokenizer, loss_fn, images, captions, device, args):
    images = images.to(device, non_blocking=True)
    tokens = tokenizer(list(captions)).to(device, non_blocking=True)
    with context(args):
        image_features, text_features, scale = model(images, tokens, normalized=True)
        loss = loss_fn(image_features, text_features, scale)
    return loss, image_features


def encode_class_text(model, tokenizer, class_ids, class_names, device, args):
    prompts = [args.class_prompt_template.format(label=class_names[class_id]) for class_id in class_ids]
    features = []
    with torch.no_grad():
        for start in range(0, len(prompts), args.class_text_batch_size):
            tokens = tokenizer(prompts[start:start + args.class_text_batch_size]).to(device)
            with context(args):
                features.append(model.encode_text(tokens, normalized=True).float())
    return torch.cat(features)


def evaluate(model, dataset, tokenizer, loss_fn, class_ids, class_names, device, args, label, writer, step, report_path, seen):
    model.eval()
    class_features = encode_class_text(model, tokenizer, class_ids, class_names, device, args)
    total_loss = loss_count = correct = count = 0
    skipped = 0
    indices = [list(range(start, min(start + args.batch_size, len(dataset)))) for start in range(0, len(dataset), args.batch_size)]
    with torch.no_grad():
        for images, captions, targets, bad in tqdm(loader(dataset, indices, args), total=len(indices), desc=label, unit="batch"):
            record_bad_images(bad, report_path, seen)
            skipped += len(bad)
            if not captions:
                continue
            loss, image_features = forward(model, tokenizer, loss_fn, images, captions, device, args)
            n = len(captions)
            if n > 1:
                total_loss += loss.item() * n
                loss_count += n
            predictions = (image_features.float() @ class_features.T).argmax(dim=1).tolist()
            correct += sum(class_ids[index] in labels for index, labels in zip(predictions, targets))
            count += n
    if not count:
        raise RuntimeError("No usable {} image-text batches; see {}".format(label, report_path))
    metrics = {"clip_loss": total_loss / loss_count if loss_count else float("nan"), "Top1Acc": correct / count, "skipped_images": skipped}
    for key, value in metrics.items():
        writer.add_scalar(label + "/" + key, value, step)
    print("{}: {}".format(label, metrics), flush=True)
    return metrics


def save_checkpoint(path, state):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    torch.save(state, temporary)
    temporary.replace(path)


def main(argv=None):
    args = arguments(argv)
    from torch.utils.tensorboard import SummaryWriter
    if args.device == "cuda":
        if not torch.cuda.is_available() or args.gpu >= torch.cuda.device_count():
            raise RuntimeError("Requested GPU {} is unavailable; use --device cpu only if necessary".format(args.gpu))
        device = torch.device("cuda", args.gpu)
        torch.cuda.set_device(device)
    else:
        device = torch.device("cpu")
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(args.seed)

    rows, class_names = read_rows(args.split_csv, args.caption_csv)
    class_ids = tuple(sorted(class_names))
    rows["validation"] = unique_images(rows["validation"])
    rows["test"] = unique_images(rows["test"])
    data_signature = tuple((str(path.resolve()), path.stat().st_size, path.stat().st_mtime_ns) for path in (args.split_csv, args.caption_csv))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    print("Samples: " + ", ".join("{}={}".format(key, len(value)) for key, value in rows.items()) + "; classes={}".format(len(class_ids)), flush=True)
    # This auto model is pruned while loading LAION weights. Resume must build
    # the same pruned structure before restoring its LoRA state dictionary.
    model, train_transform, val_transform = create_model_and_transforms(MODEL_NAME, pretrained=args.pretrained, precision="fp32", device=device, cache_dir=str(args.cache_dir) if args.cache_dir else None)
    adapter_counts = add_lora_adapters(model, args.lora_rank, args.lora_alpha)
    trainable = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    total = sum(parameter.numel() for parameter in model.parameters())
    print("LoRA adapters: {}; trainable parameters: {:,}/{:,}".format(adapter_counts, trainable, total), flush=True)
    tokenizer = get_tokenizer(MODEL_NAME)
    datasets = {key: ImageCaptionDataset(value, args.images_dir, train_transform if key == "train" else val_transform, key, args.bad_image_policy) for key, value in rows.items()}
    bad_image_report = args.output_dir / "skipped_images.csv"
    seen_bad_images = known_bad_images(bad_image_report)
    loss_fn = ClipLoss(cache_labels=True)
    optimizer = optimizer_for(model, args)
    batches_per_epoch = math.ceil(len(datasets["train"]) / args.batch_size)
    total_steps = batches_per_epoch * args.epochs
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda" and args.precision == "amp")
    start_epoch = start_batch = step = 0
    resumed_loss_sum = resumed_count = 0
    best_top1 = -1.0
    if args.resume:
        checkpoint = torch.load(args.resume, map_location="cpu", weights_only=False)
        if checkpoint.get("fine_tune_mode") != "lora":
            raise ValueError("--resume requires a LoRA checkpoint from this script; full fine-tuning checkpoints cannot be resumed")
        if checkpoint.get("model_name") != MODEL_NAME or checkpoint.get("pretrained") != args.pretrained or checkpoint.get("batch_size") != args.batch_size or checkpoint.get("seed") != args.seed or checkpoint.get("train_size") != len(datasets["train"]) or checkpoint.get("data_signature") != data_signature or checkpoint.get("lora_rank") != args.lora_rank or checkpoint.get("lora_alpha") != args.lora_alpha or checkpoint.get("class_ids") != class_ids or checkpoint.get("class_prompt_template") != args.class_prompt_template:
            raise ValueError("Resume checkpoint model, LoRA settings, class prompts, batch size, seed or input CSV files differ")
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        scaler.load_state_dict(checkpoint["scaler"])
        start_epoch, start_batch, step = checkpoint["epoch"], checkpoint["next_batch"], checkpoint["step"]
        resumed_loss_sum, resumed_count = checkpoint.get("epoch_loss_sum", 0), checkpoint.get("epoch_count", 0)
        best_top1 = checkpoint["best_top1"]
        random.setstate(checkpoint["python_rng"])
        torch.set_rng_state(checkpoint["torch_rng"])
        if device.type == "cuda" and checkpoint.get("cuda_rng") is not None:
            torch.cuda.set_rng_state(checkpoint["cuda_rng"], device)
    writer = SummaryWriter(args.output_dir / "tensorboard", purge_step=step if args.resume else None)

    def state(epoch, next_batch, epoch_loss_sum=0, epoch_count=0):
        return {"fine_tune_mode": "lora", "lora_rank": args.lora_rank, "lora_alpha": args.lora_alpha, "class_ids": class_ids, "class_prompt_template": args.class_prompt_template, "model_name": MODEL_NAME, "pretrained": args.pretrained, "batch_size": args.batch_size, "seed": args.seed, "train_size": len(datasets["train"]), "data_signature": data_signature, "model": model.state_dict(), "optimizer": optimizer.state_dict(), "scaler": scaler.state_dict(), "epoch": epoch, "next_batch": next_batch, "epoch_loss_sum": epoch_loss_sum, "epoch_count": epoch_count, "step": step, "best_top1": best_top1, "python_rng": random.getstate(), "torch_rng": torch.get_rng_state(), "cuda_rng": torch.cuda.get_rng_state(device) if device.type == "cuda" else None}

    for epoch in range(start_epoch, args.epochs):
        model.train()
        batches = batches_for_epoch(len(datasets["train"]), args.batch_size, args.seed, epoch)
        first = start_batch if epoch == start_epoch else 0
        loss_sum = resumed_loss_sum if epoch == start_epoch else 0
        count = resumed_count if epoch == start_epoch else 0
        progress = tqdm(enumerate(loader(datasets["train"], batches[first:], args), start=first), total=len(batches), initial=first, desc="train epoch {}/{}".format(epoch + 1, args.epochs), unit="batch")
        skipped = 0
        for batch_index, (images, captions, _, bad) in progress:
            record_bad_images(bad, bad_image_report, seen_bad_images)
            skipped += len(bad)
            if len(captions) < 2:
                continue
            factor = learning_rate(step, total_steps, args.warmup_steps)
            for group in optimizer.param_groups:
                group["lr"] = args.lr * factor
            optimizer.zero_grad(set_to_none=True)
            loss, _ = forward(model, tokenizer, loss_fn, images, captions, device, args)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            step += 1
            n = len(captions)
            loss_sum += loss.item() * n
            count += n
            writer.add_scalar("train_step/clip_loss", loss.item(), step)
            writer.add_scalar("train_step/total_loss", loss.item(), step)
            writer.add_scalar("train_step/learning_rate", args.lr * factor, step)
            progress.set_postfix(loss="{:.4f}".format(loss.item()))
            if step % args.checkpoint_steps == 0:
                save_checkpoint(args.output_dir / "last.pt", state(epoch, batch_index + 1, loss_sum, count))
        if count:
            writer.add_scalar("train_epoch/clip_loss", loss_sum / count, epoch + 1)
            writer.add_scalar("train_epoch/total_loss", loss_sum / count, epoch + 1)
        writer.add_scalar("train_epoch/skipped_images", skipped, epoch + 1)
        if not count:
            raise RuntimeError("No usable training batches; see {}".format(bad_image_report))
        metrics = evaluate(model, datasets["validation"], tokenizer, loss_fn, class_ids, class_names, device, args, "validation", writer, epoch + 1, bad_image_report, seen_bad_images)
        if metrics["Top1Acc"] > best_top1:
            best_top1 = metrics["Top1Acc"]
            save_checkpoint(args.output_dir / "best.pt", state(epoch + 1, 0))
        save_checkpoint(args.output_dir / "last.pt", state(epoch + 1, 0))
        writer.flush()

    best = torch.load(args.output_dir / "best.pt", map_location="cpu", weights_only=False)
    model.load_state_dict(best["model"])
    evaluate(model, datasets["test"], tokenizer, loss_fn, class_ids, class_names, device, args, "test", writer, step, bad_image_report, seen_bad_images)
    writer.close()
    print("Saved checkpoints and TensorBoard logs to {}".format(args.output_dir), flush=True)


if __name__ == "__main__":
    main()
