#!/usr/bin/env python3
"""Train and evaluate one deep learning baseline on burn2scar.

Normally started by run_baselines.py. To run one model by hand:

  python tools/train_baseline.py --spec unet_r34 --seed 0 --index data/v2/index.csv \
      --images-root data/v1 --masks-root data/v2 --out results

FLOGA's deep learning models are all change detection models that need a pre-fire
image. burn2scar has one post-fire image per sample, so each is matched here by its
closest single-image family. As in FLOGA, the best epoch is chosen by burnt-class F1
on the validation set, and burnt-class metrics treat fresh and old burn as one class.

Pixels labelled nodata or older_burn are ignored in training and evaluation. Pixels
labelled cloud or shadow are also left out of the burnt-class metrics, because the
ground under them can't be seen.
"""
import argparse
import json
import math
import os
import random
import time
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio
import segmentation_models_pytorch as smp
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

IGNORE = 255
TASKS = {
    "multiclass": {"classes": ["clear", "fresh_burn", "old_burn", "cloud", "shadow", "water"],
                   "values": {0: 0, 2: 1, 3: 2, 4: 3, 5: 4, 6: 5}, "burnt": [1, 2], "unburnt": [0, 5]},
    "burn": {"classes": ["unburnt", "burnt"], "values": {0: 0, 6: 0, 2: 1, 3: 1}, "burnt": [1], "unburnt": [0]},
}
SPECS = {
    # Closest FLOGA family is in the comment.
    "unet_r18_scratch": dict(arch="Unet", encoder="resnet18", weights=None, lr=3e-4, batch=32),  # FC-EF
    "unet_r34": dict(arch="Unet", encoder="resnet34", weights="imagenet", lr=3e-4, batch=32),  # FC-EF, pretrained
    "unetpp_r34": dict(arch="UnetPlusPlus", encoder="resnet34", weights="imagenet", lr=3e-4, batch=24),  # SNUNet-CD
    "unetpp_r101": dict(arch="UnetPlusPlus", encoder="resnet101", weights="imagenet", lr=1e-4, batch=12),  # BAM-CD
    "hrnet_w18": dict(arch="Unet", encoder="tu-hrnet_w18", weights="imagenet", lr=3e-4, batch=16),  # ADHR-CDNet
    "segformer_b2": dict(arch="Segformer", encoder="mit_b2", weights="imagenet", lr=6e-5, batch=24),  # ChangeFormer
    "deeplabv3p_r50": dict(arch="DeepLabV3Plus", encoder="resnet50", weights="imagenet", lr=3e-4, batch=24),
    "unet_r50_ssl4eo": dict(arch="Unet", encoder="resnet50", weights="ssl4eo", lr=1e-4, batch=24,
                            norm="reflectance"),  # Pretrained on Sentinel-2 L1C, all 13 bands
    # Small model for smoke tests.
    "tiny_test": dict(arch="Unet", encoder="resnet18", weights=None, lr=1e-3, batch=4),
}


def label_table(task):
    table = np.full(256, IGNORE, np.uint8)
    for value, index in TASKS[task]["values"].items():
        table[value] = index
    return table


def read_image(path):
    with rasterio.open(path) as src:
        return np.clip(src.read().astype(np.float32) / 10000.0, 0, 1.5)


class Tiles(Dataset):
    def __init__(self, frame, images_root, masks_root, task, mean, std, crop=0, cache_root=""):
        if cache_root:  # Plain NumPy files from build_cache.py, far quicker to read.
            self.images = [Path(cache_root) / f"{p}.npy" for p in frame["image_path"]]
            self.masks = [Path(cache_root) / f"{p}.npy" for p in frame["mask_path"]]
        else:
            self.images = [Path(images_root) / p for p in frame["image_path"]]
            self.masks = [Path(masks_root) / p for p in frame["mask_path"]]
        self.cached = bool(cache_root)
        self.table, self.crop = label_table(task), crop
        self.mean, self.std = mean[:, None, None], std[:, None, None]

    def __len__(self):
        return len(self.images)

    def __getitem__(self, i):
        if self.cached:
            image, mask = np.load(self.images[i], mmap_mode="r"), np.load(self.masks[i], mmap_mode="r")
        else:
            with rasterio.open(self.images[i]) as src:
                image = src.read()
            with rasterio.open(self.masks[i]) as src:
                mask = src.read(1)
        if self.crop:
            size = image.shape[-1]
            top, left = random.randint(0, size - self.crop), random.randint(0, size - self.crop)
            image = image[:, top:top + self.crop, left:left + self.crop]
            mask = mask[top:top + self.crop, left:left + self.crop]
        image = np.clip(np.asarray(image, dtype=np.float32) / 10000.0, 0, 1.5)
        label = self.table[np.asarray(mask)]
        if self.crop:
            if random.random() < .5:
                image, label = image[:, :, ::-1], label[:, ::-1]
            turns = random.randint(0, 3)
            if turns:
                image, label = np.rot90(image, turns, (1, 2)), np.rot90(label, turns)
        image = (image - self.mean) / self.std
        return torch.from_numpy(image.copy()), torch.from_numpy(label.copy()).long(), i


def seed_worker(worker_id):
    seed = torch.initial_seed() % 2**32
    random.seed(seed)
    np.random.seed(seed)


def band_stats(frame, images_root, cache, key):
    """Band means and standard deviations from a fixed sample of training tiles. Zero means nodata."""
    cache = Path(cache)
    if cache.exists():
        stored = json.loads(cache.read_text())
        if key in stored:
            return np.array(stored[key]["mean"], np.float32), np.array(stored[key]["std"], np.float32)
    sample = frame.sample(min(1000, len(frame)), random_state=0)
    total, squares, count = np.zeros(13), np.zeros(13), np.zeros(13)
    for path in sample["image_path"]:
        image = read_image(Path(images_root) / path)
        valid = image != 0
        total += (image * valid).sum((1, 2))
        squares += ((image * valid) ** 2).sum((1, 2))
        count += valid.sum((1, 2))
    mean = total / np.maximum(count, 1)
    std = np.sqrt(np.maximum(squares / np.maximum(count, 1) - mean ** 2, 1e-8))
    stored = json.loads(cache.read_text()) if cache.exists() else {}
    stored[key] = {"mean": mean.tolist(), "std": std.tolist()}
    cache.parent.mkdir(parents=True, exist_ok=True)
    temp = cache.with_suffix(f".{os.getpid()}.tmp")
    temp.write_text(json.dumps(stored, indent=1))
    temp.replace(cache)
    return mean.astype(np.float32), std.astype(np.float32)


def build_model(spec, classes):
    architecture = getattr(smp, spec["arch"])
    weights = spec["weights"] if spec["weights"] in ("imagenet",) else None
    model = architecture(encoder_name=spec["encoder"], encoder_weights=weights, in_channels=13, classes=classes)
    if spec["weights"] == "ssl4eo":
        # SSL4EO-S12 MoCo weights, the same file torchgeo uses for SENTINEL2_ALL_MOCO. Licence: CC BY 4.0.
        from huggingface_hub import hf_hub_download
        path = hf_hub_download("torchgeo/resnet50_sentinel2_all_moco", "resnet50_sentinel2_all_moco-df8b932e.pth")
        load_encoder_weights(model, torch.load(path, map_location="cpu", weights_only=True))
    return model


def load_encoder_weights(model, state):
    """Strict load, so weights that don't fit raise an error instead of being silently skipped."""
    model.encoder.load_state_dict({k: v for k, v in state.items() if not k.startswith("fc.")}, strict=True)


def confusion_per_tile(model, loader, classes, device, amp):
    """One classes-by-classes confusion matrix per tile, with rows as labels and columns as predictions."""
    model.eval()
    matrices = np.zeros((len(loader.dataset), classes, classes), np.int64)
    with torch.no_grad():
        for images, labels, indices in loader:
            images, labels = images.to(device, non_blocking=True), labels.to(device, non_blocking=True)
            with torch.amp.autocast(device.type, enabled=amp):
                predictions = model(images).argmax(1)
            for row, truth, predicted in zip(indices.tolist(), labels, predictions):
                valid = truth != IGNORE
                codes = classes * truth[valid] + predicted[valid]
                matrices[row] = torch.bincount(codes, minlength=classes ** 2).reshape(classes, classes).cpu().numpy()
    return matrices


def two_class(tp, fp, fn, tn):
    def scores(tp, fp, fn):
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * tp / (2 * tp + fp + fn) if tp else 0.0
        iou = tp / (tp + fp + fn) if tp else 0.0
        return precision, recall, f1, iou
    burnt = scores(tp, fp, fn)
    unburnt = scores(tn, fn, fp)
    return {
        "burnt_precision": burnt[0], "burnt_recall": burnt[1], "burnt_f1": burnt[2], "burnt_iou": burnt[3],
        "unburnt_precision": unburnt[0], "unburnt_recall": unburnt[1], "unburnt_f1": unburnt[2], "unburnt_iou": unburnt[3],
        "mean_f1": (burnt[2] + unburnt[2]) / 2, "mean_iou": (burnt[3] + unburnt[3]) / 2,
    }


def summarise_matrix(matrix, task):
    """Metrics from one confusion matrix. Burnt metrics use only burnt and unburnt label rows."""
    info = TASKS[task]
    burnt, unburnt = info["burnt"], info["unburnt"]
    tp = matrix[np.ix_(burnt, burnt)].sum()
    fn = matrix[burnt].sum() - tp
    fp = matrix[np.ix_(unburnt, burnt)].sum()
    tn = matrix[unburnt].sum() - fp
    result = two_class(int(tp), int(fp), int(fn), int(tn))
    per_class = {}
    for i, name in enumerate(info["classes"]):
        hit = matrix[i, i]
        wrong_label, wrong_prediction = matrix[i].sum() - hit, matrix[:, i].sum() - hit
        per_class[name] = {"iou": float(hit / max(hit + wrong_label + wrong_prediction, 1)),
                           "f1": float(2 * hit / max(2 * hit + wrong_label + wrong_prediction, 1)),
                           "pixels": int(matrix[i].sum())}
    result["per_class"] = per_class
    result["class_miou"] = float(np.mean([c["iou"] for c in per_class.values()]))
    if task == "multiclass":
        pairs = matrix[np.ix_([1, 2], [1, 2])]
        result["fresh_old_matrix"] = pairs.tolist()
        result["fresh_old_accuracy"] = float(np.trace(pairs) / max(pairs.sum(), 1))
    return result


def learning_rate(step, warmup, total):
    if step < warmup:
        return (step + 1) / warmup
    return 0.5 * (1 + math.cos(math.pi * (step - warmup) / max(total - warmup, 1)))


def train(args):
    spec = SPECS[args.spec]
    run_name = f"{args.spec}__{args.split_col}__{args.task}__seed{args.seed}"
    run_dir = Path(args.out) / run_name
    if (run_dir / "metrics.json").exists():
        print(f"{run_name}: already finished")
        return
    run_dir.mkdir(parents=True, exist_ok=True)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.benchmark = True
    device = torch.device(args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu"))
    amp = device.type == "cuda"

    index = pd.read_csv(args.index, low_memory=False)
    frames = {name: index[index[args.split_col] == name] for name in ("train", "val", "test")}
    if args.limit:
        frames = {k: v.head(args.limit) for k, v in frames.items()}
    if any(frame.empty for frame in frames.values()):
        raise SystemExit(f"{args.split_col}: a split is empty: { {k: len(v) for k, v in frames.items()} }")
    info = TASKS[args.task]
    classes = len(info["classes"])

    if spec.get("norm") == "reflectance":
        mean, std = np.zeros(13, np.float32), np.ones(13, np.float32)
    else:
        mean, std = band_stats(frames["train"], args.images_root, Path(args.out) / "band_stats.json", args.split_col)

    batch = args.batch_size or max(int(spec["batch"] * args.batch_scale), 2)
    def make_loader(frame, training):
        data = Tiles(frame, args.images_root, args.masks_root, args.task, mean, std, args.crop if training else 0, args.cache_root)
        return DataLoader(data, batch_size=batch if training else batch * 2, shuffle=training, drop_last=training and len(frame) > batch,
                          num_workers=args.workers, pin_memory=amp, persistent_workers=args.workers > 0,
                          worker_init_fn=seed_worker, generator=torch.Generator().manual_seed(args.seed))
    train_loader, val_loader, test_loader = make_loader(frames["train"], True), make_loader(frames["val"], False), make_loader(frames["test"], False)

    model = build_model(spec, classes).to(device)
    parameters = sum(p.numel() for p in model.parameters())
    frequency = np.zeros(classes)
    fractions = {"clear": "frac_clear", "fresh_burn": "frac_fresh", "old_burn": "frac_old", "cloud": "frac_cloud",
                 "shadow": "frac_shadow", "water": "frac_water"}
    if args.task == "multiclass":
        frequency = np.array([frames["train"][fractions[name]].mean() for name in info["classes"]])
    else:
        frame = frames["train"]
        frequency = np.array([(frame["frac_clear"] + frame["frac_water"]).mean(), (frame["frac_fresh"] + frame["frac_old"]).mean()])
    weights = 1 / np.sqrt(np.maximum(frequency, 1e-6))
    weights = torch.tensor(weights / weights.mean(), dtype=torch.float32, device=device)
    dice = smp.losses.DiceLoss("multiclass", ignore_index=IGNORE, from_logits=True)

    optimiser = torch.optim.AdamW(model.parameters(), lr=spec["lr"], weight_decay=1e-4)
    steps = args.epochs * len(train_loader)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimiser, lambda s: learning_rate(s, len(train_loader), steps))
    scaler = torch.amp.GradScaler(device.type, enabled=amp)

    best, stale, history, started = -1.0, 0, [], time.time()
    best_path = run_dir / "best.pt"
    for epoch in range(1, args.epochs + 1):
        model.train()
        total, seen = 0.0, 0
        for images, labels, _ in train_loader:
            if (labels == IGNORE).all():  # Nothing to learn from, and the loss would be undefined.
                scheduler.step()
                continue
            images, labels = images.to(device, non_blocking=True), labels.to(device, non_blocking=True)
            optimiser.zero_grad(set_to_none=True)
            with torch.amp.autocast(device.type, enabled=amp):
                logits = model(images)
                loss = F.cross_entropy(logits, labels, weight=weights, ignore_index=IGNORE) + dice(logits, labels)
            scaler.scale(loss).backward()
            scaler.step(optimiser)
            scaler.update()
            scheduler.step()
            total += loss.item() * len(images)
            seen += len(images)
        val = summarise_matrix(confusion_per_tile(model, val_loader, classes, device, amp).sum(0), args.task)
        history.append({"epoch": epoch, "loss": total / max(seen, 1), "val_burnt_f1": val["burnt_f1"],
                        "val_class_miou": val["class_miou"], "minutes": (time.time() - started) / 60})
        print(json.dumps(history[-1]), flush=True)
        if val["burnt_f1"] > best:
            best, stale = val["burnt_f1"], 0
            torch.save(model.state_dict(), best_path)
        else:
            stale += 1
            if stale >= args.patience:
                break

    model.load_state_dict(torch.load(best_path, map_location=device, weights_only=True))
    val_tiles = confusion_per_tile(model, val_loader, classes, device, amp)
    test_tiles = confusion_per_tile(model, test_loader, classes, device, amp)
    np.savez_compressed(run_dir / "test_tile_confusion.npz", tile_id=frames["test"]["tile_id"].to_numpy(dtype=str),
                        matrices=test_tiles, classes=np.array(info["classes"]))
    result = {
        "name": args.spec, "kind": "deep", "split": args.split_col, "task": args.task, "seed": args.seed, "spec": spec,
        "parameters": parameters, "epochs_run": len(history), "best_epoch": int(np.argmax([h["val_burnt_f1"] for h in history]) + 1),
        "minutes": (time.time() - started) / 60, "tiles": {k: len(v) for k, v in frames.items()}, "crop": args.crop,
        "val": summarise_matrix(val_tiles.sum(0), args.task), "test": summarise_matrix(test_tiles.sum(0), args.task),
        "history": history,
    }
    temp = run_dir / "metrics.tmp"
    temp.write_text(json.dumps(result, indent=1))
    temp.replace(run_dir / "metrics.json")  # Written last: its presence marks the run as finished.
    print(f"{run_name}: test burnt F1 {result['test']['burnt_f1']:.4f}, IoU {result['test']['burnt_iou']:.4f}")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--spec", required=True, choices=sorted(SPECS))
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--index", default="data/v2/index.csv")
    parser.add_argument("--images-root", default="data/v1")
    parser.add_argument("--masks-root", default="data/v2")
    parser.add_argument("--cache-root", default="", help="NumPy files from build_cache.py, which are much quicker to read")
    parser.add_argument("--split-col", default="split")
    parser.add_argument("--task", default="multiclass", choices=sorted(TASKS))
    parser.add_argument("--out", default="results")
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument("--crop", type=int, default=192, help="random training crop; must be divisible by 32")
    parser.add_argument("--batch-size", type=int, default=0, help="0 uses the model's default")
    parser.add_argument("--batch-scale", type=float, default=1.0, help="multiply the model's default, e.g. 0.5 for a small GPU")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--device", default="")
    parser.add_argument("--limit", type=int, default=0, help="use only this many tiles per split, for a quick trial")
    train(parser.parse_args())


if __name__ == "__main__":
    main()
