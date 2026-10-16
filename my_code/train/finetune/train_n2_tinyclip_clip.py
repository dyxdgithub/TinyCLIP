"""Fine-tune TinyCLIP with the original symmetric CLIP image-text loss.

The split table supplies image paths and split names. Captions are joined from
the n2 metadata table by (ImageID, ParentLabelName, LabelName). The first
nonempty caption for a repeated key is used deterministically.
"""

import argparse
import csv
import math
import random
from contextlib import nullcontext
from pathlib import Path

import torch
from PIL import Image
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from open_clip import ClipLoss, create_model_and_transforms, get_tokenizer


SCRIPT_DIR = Path(__file__).resolve().parent
N2_DIR = SCRIPT_DIR.parents[1] / "data/OpenImage/meta/Hierarchy/n2"
MODEL_NAME = "TinyCLIP-auto-ViT-45M-32-Text-18M"
DEFAULT_OUTPUT = SCRIPT_DIR / "output/clip_full_TinyCLIP-auto-ViT-45M-32-Text-18M-LAION400M"
JOIN_FIELDS = ("ImageID", "ParentLabelName", "LabelName")


def arguments(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--split-csv", type=Path, default=N2_DIR / "Image_IDs_n2_all_downloaded_split_8_1_1.csv", help="CSV with ImageID, RelativePath, parent/child labels and Split.")
    parser.add_argument("--caption-csv", type=Path, default=N2_DIR / "Image IDs_with_last_two_layers_multi_members_n2_one_label_per_parent.csv", help="Metadata CSV with caption and the three join columns.")
    parser.add_argument("--images-dir", type=Path, default=N2_DIR / "images_all", help="Root containing RelativePath entries; set this to the server's image directory.")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT, help="Checkpoint and TensorBoard output directory.")
    parser.add_argument("--pretrained", default="LAION400M", help="Registered TinyCLIP pretrained tag or local checkpoint; a registered tag may download when user runs this script.")
    parser.add_argument("--cache-dir", type=Path, default=None, help="Optional pretrained weight cache; default is open_clip's cache.")
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
    for name in ("epochs", "batch_size", "checkpoint_steps"):
        if getattr(args, name) < 1:
            parser.error("--{} must be positive".format(name.replace("_", "-")))
    if args.workers < 0 or args.gpu < 0 or args.warmup_steps < 0 or args.weight_decay < 0 or args.lr <= 0:
        parser.error("workers, gpu, warmup-steps and weight-decay must be nonnegative; lr must be positive")
    if not (0 <= args.beta1 < 1 and 0 <= args.beta2 < 1 and args.eps > 0):
        parser.error("beta1/beta2 must be in [0, 1), and eps must be positive")
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
    with split_csv.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        required = set(JOIN_FIELDS) | {"RelativePath", "Split"}
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
            relative = Path((row["RelativePath"] or "").strip().replace("\\", "/"))
            if relative.is_absolute() or not relative.parts or ".." in relative.parts:
                raise ValueError("Unsafe RelativePath at row {}: {}".format(number, relative))
            rows[split].append((relative, captions[key]))
    for split, entries in rows.items():
        if not entries:
            raise ValueError("Split {} has no samples".format(split))
    return rows


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
        relative, caption = self.entries[index]
        try:
            with Image.open(self.images_dir / relative) as image:
                return self.transform(image.convert("RGB")), caption, None
        except OSError as error:
            if self.bad_image_policy == "error":
                raise OSError("Cannot load image {}: {}".format(self.images_dir / relative, error)) from error
            return None, None, (self.split, relative.as_posix(), str(error))


def collate_image_captions(samples):
    valid = [(image, caption) for image, caption, error in samples if error is None]
    bad = [error for _, _, error in samples if error is not None]
    if not valid:
        return None, [], bad
    images, captions = zip(*valid)
    return torch.stack(images), list(captions), bad


def record_bad_images(bad, report_path, seen):
    new = []
    pending = set()
    for item in bad:
        if item[:2] not in seen and item[:2] not in pending:
            new.append(item)
            pending.add(item[:2])
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
    logits = scale.detach().float() * image_features.detach().float() @ text_features.detach().float().T
    labels = torch.arange(logits.size(0), device=device)
    correct_i2t = (logits.argmax(dim=1) == labels).sum().item()
    correct_t2i = (logits.argmax(dim=0) == labels).sum().item()
    return loss, correct_i2t, correct_t2i


def evaluate(model, dataset, tokenizer, loss_fn, device, args, label, writer, step, report_path, seen):
    model.eval()
    total_loss = total_i2t = total_t2i = count = 0
    skipped = 0
    indices = [list(range(start, min(start + args.batch_size, len(dataset)))) for start in range(0, len(dataset), args.batch_size)]
    with torch.no_grad():
        for images, captions, bad in tqdm(loader(dataset, indices, args), total=len(indices), desc=label, unit="batch"):
            record_bad_images(bad, report_path, seen)
            skipped += len(bad)
            if len(captions) < 2:
                continue
            loss, i2t, t2i = forward(model, tokenizer, loss_fn, images, captions, device, args)
            n = len(captions)
            total_loss += loss.item() * n
            total_i2t += i2t
            total_t2i += t2i
            count += n
    if not count:
        raise RuntimeError("No usable {} image-text batches; see {}".format(label, report_path))
    metrics = {"clip_loss": total_loss / count, "image_to_text_recall_at_1": total_i2t / count, "text_to_image_recall_at_1": total_t2i / count, "skipped_images": skipped}
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

    rows = read_rows(args.split_csv, args.caption_csv)
    data_signature = tuple((str(path.resolve()), path.stat().st_size, path.stat().st_mtime_ns) for path in (args.split_csv, args.caption_csv))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    print("Samples: " + ", ".join("{}={}".format(key, len(value)) for key, value in rows.items()), flush=True)
    model, train_transform, val_transform = create_model_and_transforms(MODEL_NAME, pretrained="" if args.resume else args.pretrained, precision="fp32", device=device, cache_dir=str(args.cache_dir) if args.cache_dir else None)
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
    best_loss = float("inf")
    if args.resume:
        checkpoint = torch.load(args.resume, map_location="cpu", weights_only=False)
        if checkpoint.get("model_name") != MODEL_NAME or checkpoint.get("batch_size") != args.batch_size or checkpoint.get("seed") != args.seed or checkpoint.get("train_size") != len(datasets["train"]) or checkpoint.get("data_signature") != data_signature:
            raise ValueError("Resume checkpoint model, batch size, seed or input CSV files differ")
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        scaler.load_state_dict(checkpoint["scaler"])
        start_epoch, start_batch, step = checkpoint["epoch"], checkpoint["next_batch"], checkpoint["step"]
        resumed_loss_sum, resumed_count = checkpoint.get("epoch_loss_sum", 0), checkpoint.get("epoch_count", 0)
        best_loss = checkpoint["best_loss"]
        random.setstate(checkpoint["python_rng"])
        torch.set_rng_state(checkpoint["torch_rng"])
        if device.type == "cuda" and checkpoint.get("cuda_rng") is not None:
            torch.cuda.set_rng_state(checkpoint["cuda_rng"], device)
    writer = SummaryWriter(args.output_dir / "tensorboard", purge_step=step if args.resume else None)

    def state(epoch, next_batch, epoch_loss_sum=0, epoch_count=0):
        return {"model_name": MODEL_NAME, "batch_size": args.batch_size, "seed": args.seed, "train_size": len(datasets["train"]), "data_signature": data_signature, "model": model.state_dict(), "optimizer": optimizer.state_dict(), "scaler": scaler.state_dict(), "epoch": epoch, "next_batch": next_batch, "epoch_loss_sum": epoch_loss_sum, "epoch_count": epoch_count, "step": step, "best_loss": best_loss, "python_rng": random.getstate(), "torch_rng": torch.get_rng_state(), "cuda_rng": torch.cuda.get_rng_state(device) if device.type == "cuda" else None}

    for epoch in range(start_epoch, args.epochs):
        model.train()
        batches = batches_for_epoch(len(datasets["train"]), args.batch_size, args.seed, epoch)
        first = start_batch if epoch == start_epoch else 0
        loss_sum = resumed_loss_sum if epoch == start_epoch else 0
        count = resumed_count if epoch == start_epoch else 0
        progress = tqdm(enumerate(loader(datasets["train"], batches[first:], args), start=first), total=len(batches), initial=first, desc="train epoch {}/{}".format(epoch + 1, args.epochs), unit="batch")
        skipped = 0
        for batch_index, (images, captions, bad) in progress:
            record_bad_images(bad, bad_image_report, seen_bad_images)
            skipped += len(bad)
            if len(captions) < 2:
                continue
            factor = learning_rate(step, total_steps, args.warmup_steps)
            for group in optimizer.param_groups:
                group["lr"] = args.lr * factor
            optimizer.zero_grad(set_to_none=True)
            loss, _, _ = forward(model, tokenizer, loss_fn, images, captions, device, args)
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
        metrics = evaluate(model, datasets["validation"], tokenizer, loss_fn, device, args, "validation", writer, epoch + 1, bad_image_report, seen_bad_images)
        if metrics["clip_loss"] < best_loss:
            best_loss = metrics["clip_loss"]
            save_checkpoint(args.output_dir / "best.pt", state(epoch + 1, 0))
        save_checkpoint(args.output_dir / "last.pt", state(epoch + 1, 0))
        writer.flush()

    best = torch.load(args.output_dir / "best.pt", map_location="cpu", weights_only=False)
    model.load_state_dict(best["model"])
    evaluate(model, datasets["test"], tokenizer, loss_fn, device, args, "test", writer, step, bad_image_report, seen_bad_images)
    writer.close()
    print("Saved checkpoints and TensorBoard logs to {}".format(args.output_dir), flush=True)


if __name__ == "__main__":
    main()
