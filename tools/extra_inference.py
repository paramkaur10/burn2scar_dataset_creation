#!/usr/bin/env python3
"""Two tests that need the trained models run again on the test tiles (minutes on one GPU, no training).

  python tools/extra_inference.py --results results_v4 --index data/v4/index.csv --images-root data/v1 \
      --masks-root data/v4 --out results_v4/extra_inference.md

1. older_burn: what the models predict on pixels whose scar is more than 365 days old, a class left out of training
   and of the scores. Called fresh_burn, it is a false alarm of exactly the kind the age classes are meant to stop.
2. Off-centre crops: every test tile is centred on its fire. Each tile is also cut into four 192 x 192 windows offset
   by 64 pixels, so the fire sits off-centre, and predicted again. If the fresh_burn F1 on the windows matches the
   full tiles' F1 on the same pixels, the models do not lean on the fire being in the middle.
Runs on seed 0 of each model by default (--seeds for more).
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio
import torch
from torch.utils.data import DataLoader, Dataset

sys.path.insert(0, str(Path(__file__).resolve().parent))
from summarize_results import fresh_from_matrix  # noqa: E402
from train_baseline import ALL_BANDS, IGNORE, TASKS, build_model, label_table  # noqa: E402

CLASSES = TASKS["multiclass"]["classes"]
OFFSETS = [(0, 0), (0, 64), (64, 0), (64, 64)]
CROP = 192


class TestTiles(Dataset):
    def __init__(self, frame, images_root, masks_root, mean, std, channels):
        self.images = [Path(images_root) / p for p in frame["image_path"]]
        self.masks = [Path(masks_root) / p for p in frame["mask_path"]]
        self.channels, self.mean, self.std = channels, mean[channels][:, None, None], std[channels][:, None, None]

    def __len__(self):
        return len(self.images)

    def __getitem__(self, i):
        with rasterio.open(self.images[i]) as src:
            image = src.read()
        with rasterio.open(self.masks[i]) as src:
            mask = src.read(1)
        image = (np.clip(image.astype(np.float32)[self.channels] / 10000.0, 0, 1.5) - self.mean) / self.std
        return torch.from_numpy(image), torch.from_numpy(mask.astype(np.int64)), i


def confusion(truth, predicted, classes=6):
    valid = truth != IGNORE
    codes = classes * truth[valid] + predicted[valid]
    return torch.bincount(codes, minlength=classes * classes).reshape(classes, classes).cpu().numpy()


def run_model(folder, index, args, device):
    metrics = json.loads((folder / "metrics.json").read_text())
    channels = [ALL_BANDS.index(b) for b in metrics.get("bands", ALL_BANDS)]
    if metrics["spec"].get("norm") == "reflectance":
        mean, std = np.zeros(13, np.float32), np.ones(13, np.float32)
    else:
        stats = json.loads((Path(args.results) / "band_stats.json").read_text())[metrics["split"]]
        mean, std = np.array(stats["mean"], np.float32), np.array(stats["std"], np.float32)
    model = build_model({**metrics["spec"], "weights": None}, len(CLASSES), len(channels)).to(device)
    model.load_state_dict(torch.load(folder / "best.pt", map_location=device, weights_only=True))
    model.eval()
    frame = index[index[metrics["split"]] == "test"]
    if args.limit:
        frame = frame.head(args.limit)
    loader = DataLoader(TestTiles(frame, args.images_root, args.masks_root, mean, std, channels),
                        batch_size=args.batch, num_workers=args.workers)
    table = torch.from_numpy(label_table("multiclass").astype(np.int64)).to(device)
    windows = frame["window"].to_numpy()
    older = {w: np.zeros(len(CLASSES), np.int64) for w in ("fresh", "old", "all")}
    old_burn = np.zeros(len(CLASSES), np.int64)
    full_on_crops, crops = np.zeros((6, 6), np.int64), np.zeros((6, 6), np.int64)
    agree, compared = 0, 0
    amp = device.type == "cuda"
    with torch.no_grad():
        for images, masks, indices in loader:
            images, masks = images.to(device), masks.to(device)
            with torch.autocast(device.type, enabled=amp):
                full = model(images).argmax(1)
            labels = table[masks]
            for k, i in enumerate(indices.tolist()):
                on_older = masks[k] == 7
                if on_older.any():
                    counts = torch.bincount(full[k][on_older], minlength=len(CLASSES)).cpu().numpy()
                    older["all"] += counts
                    if windows[i] in older:
                        older[windows[i]] += counts
                on_old = labels[k] == 2
                old_burn += torch.bincount(full[k][on_old], minlength=len(CLASSES)).cpu().numpy()
            for top, left in OFFSETS:
                window = images[:, :, top:top + CROP, left:left + CROP]
                with torch.autocast(device.type, enabled=amp):
                    cut = model(window).argmax(1)
                truth = labels[:, top:top + CROP, left:left + CROP]
                same = full[:, top:top + CROP, left:left + CROP]
                crops += confusion(truth, cut)
                full_on_crops += confusion(truth, same)
                agree += int((cut == same).sum())
                compared += cut.numel()
    return {"older": older, "old_burn": old_burn, "crops": crops, "full_on_crops": full_on_crops,
            "agreement": agree / max(compared, 1), "tiles": len(frame)}


def shares(counts):
    total = counts.sum()
    return {name: f"{100 * counts[i] / total:.1f}" if total else "" for i, name in enumerate(CLASSES)} | {"Pixels": f"{int(total):,}"}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--results", default="results_v4")
    parser.add_argument("--index", default="data/v4/index.csv")
    parser.add_argument("--images-root", default="data/v1")
    parser.add_argument("--masks-root", default="data/v4")
    parser.add_argument("--models", nargs="+", default=["unet_r34", "segformer_b2"])
    parser.add_argument("--seeds", nargs="+", type=int, default=[0])
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--limit", type=int, default=0, help="test tiles, for a quick trial")
    parser.add_argument("--out", default="results_v4/extra_inference.md")
    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    index = pd.read_csv(args.index, low_memory=False)
    older_rows, crop_rows = [], []
    for model_name in args.models:
        for seed in args.seeds:
            folder = Path(args.results) / f"{model_name}__split__multiclass__seed{seed}"
            if not (folder / "best.pt").exists():
                print(f"skip {folder}: no best.pt")
                continue
            print(f"{folder.name} ...", flush=True)
            r = run_model(folder, index, args, device)
            for window in ("all", "fresh", "old"):
                older_rows.append({"Model": model_name, "Seed": seed, "Pixels labelled": f"older_burn, {window} tiles",
                                   **shares(r["older"][window])})
            older_rows.append({"Model": model_name, "Seed": seed, "Pixels labelled": "old_burn, all tiles", **shares(r["old_burn"])})
            full = fresh_from_matrix(r["full_on_crops"], CLASSES)
            cut = fresh_from_matrix(r["crops"], CLASSES)
            crop_rows.append({"Model": model_name, "Seed": seed, "Test tiles": r["tiles"],
                              "Fresh F1, full tile on the windows' pixels": f"{100 * full['burnt_f1']:.2f}",
                              "Fresh F1, off-centre windows": f"{100 * cut['burnt_f1']:.2f}",
                              "Same class, window vs full %": f"{100 * r['agreement']:.2f}"})
    if not older_rows:
        raise SystemExit("No trained models found.")
    md = lambda f: "\n".join(["| " + " | ".join(map(str, f.columns)) + " |", "|" + "---|" * len(f.columns)] +
                             ["| " + " | ".join(str(v) for v in row) + " |" for row in f.itertuples(index=False)])
    lines = ["# Extra tests with the trained models", "",
             "## What the models predict on older_burn (scars more than 365 days old)", "",
             "Percent of the pixels predicted as each class. old_burn pixels are shown for comparison.", "",
             md(pd.DataFrame(older_rows)), "",
             "## Off-centre windows", "", f"Four {CROP} x {CROP} windows per test tile, offsets {OFFSETS}.", "",
             md(pd.DataFrame(crop_rows)), ""]
    Path(args.out).write_text("\n".join(lines))
    print("\n".join(lines))


if __name__ == "__main__":
    main()
