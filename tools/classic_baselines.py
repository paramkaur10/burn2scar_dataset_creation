#!/usr/bin/env python3
"""Spectral index and pixel classifier baselines on burn2scar, following FLOGA.

  python tools/classic_baselines.py --index data/v2/index.csv --images-root data/v1 \
      --masks-root data/v2 --out results --workers 16

FLOGA differences a pre-fire and a post-fire index. burn2scar has only the post-fire
image, so the same indices are used on one image:

  Indices      NBR, NBR2, NBR+, MIRBI, BAIS2, NDVI, NDWI, EVI, CSI, SAVI. One global
               threshold per index, found by grid search on the training set.
  Classifiers  Random Forest, XGBoost, SVM, MLP, on single pixels with all bands and
               indices. Training samples: N/2 burnt, N/2 unburnt, with 10% of the
               unburnt pixels taken from water, as in FLOGA.

All of these answer burnt or unburnt. With --target burn (the default), burnt is fresh and
old burn together. With --target fresh, burnt is fresh burn only and old burn counts as
unburnt, as an alert for new fires needs. Cloud, shadow, older_burn and nodata pixels are
left out. Classifier evaluation uses a regular 1-in-16 pixel grid to keep prediction
time reasonable; index evaluation uses every pixel.
"""
import argparse
import json
import os
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio

BANDS = ["B01", "B02", "B03", "B04", "B05", "B06", "B07", "B08", "B8A", "B09", "B10", "B11", "B12"]
B = {name: i for i, name in enumerate(BANDS)}
IGNORE = 255
LABELS = np.full(256, IGNORE, np.uint8)
TASK = {"name": "burn"}
WATER_VALUE = 6


def set_target(target):
    """burn: fresh and old burn are burnt. fresh: only fresh burn is, and old burn joins clear and water."""
    LABELS[:] = IGNORE
    LABELS[[0, 6]] = 0
    LABELS[2] = 1
    LABELS[3] = 1 if target == "burn" else 0
    TASK["name"] = target


set_target("burn")


def indices(image):
    """Single-date indices from reflectance with bands first."""
    b = {name: image[i] for name, i in B.items()}
    eps = 1e-6
    return {
        "NBR": (b["B8A"] - b["B12"]) / (b["B8A"] + b["B12"] + eps),
        "NBR2": (b["B11"] - b["B12"]) / (b["B11"] + b["B12"] + eps),
        "NBR+": (b["B12"] - b["B8A"] - b["B03"] - b["B02"]) / (b["B12"] + b["B8A"] + b["B03"] + b["B02"] + eps),
        "MIRBI": 10 * b["B12"] - 9.8 * b["B11"] + 2,
        "BAIS2": (1 - np.sqrt(b["B06"] * b["B07"] * b["B8A"] / (b["B04"] + eps)))
                 * ((b["B12"] - b["B8A"]) / np.sqrt(b["B12"] + b["B8A"] + eps) + 1),
        "NDVI": (b["B08"] - b["B04"]) / (b["B08"] + b["B04"] + eps),
        "NDWI": (b["B03"] - b["B08"]) / (b["B03"] + b["B08"] + eps),
        "EVI": 2.5 * (b["B08"] - b["B04"]) / (b["B08"] + 6 * b["B04"] - 7.5 * b["B02"] + 1 + eps),
        "CSI": b["B08"] / (b["B11"] + eps),
        "SAVI": 1.5 * (b["B08"] - b["B04"]) / (b["B08"] + b["B04"] + 0.5),
    }


INDEX_NAMES = ["NBR", "NBR2", "NBR+", "MIRBI", "BAIS2", "NDVI", "NDWI", "EVI", "CSI", "SAVI"]


def load_tile(args):
    image_path, mask_path = args
    if image_path.endswith(".npy"):  # From build_cache.py.
        return np.clip(np.load(image_path).astype(np.float32) / 10000.0, 0, 1.5), np.load(mask_path)
    with rasterio.open(image_path) as src:
        image = np.clip(src.read().astype(np.float32) / 10000.0, 0, 1.5)
    with rasterio.open(mask_path) as src:
        mask = src.read(1)
    return image, mask


def tile_index_values(paths):
    image, mask = load_tile(paths)
    return indices(image), LABELS[mask]


def _sample_values(paths):
    values, labels = tile_index_values(paths)
    valid = labels != IGNORE
    return {name: values[name][valid][::37] for name in INDEX_NAMES}


def _histogram_tile(job):
    paths, ranges, bins = job
    values, labels = tile_index_values(paths)
    out = {}
    for name in INDEX_NAMES:
        # Clipping keeps values outside the search range in the first and last bins.
        low, high = ranges[name]
        clipped = np.clip(np.nan_to_num(values[name], nan=low, posinf=high, neginf=low), low, high)
        burnt, _ = np.histogram(clipped[labels == 1], bins=bins, range=(low, high))
        unburnt, _ = np.histogram(clipped[labels == 0], bins=bins, range=(low, high))
        out[name] = (burnt, unburnt)
    return out


def _threshold_tile(job):
    paths, rules = job
    values, labels = tile_index_values(paths)
    out = {}
    for name, (threshold, lower_is_burnt) in rules.items():
        predicted = values[name] <= threshold if lower_is_burnt else values[name] > threshold
        burnt, unburnt = labels == 1, labels == 0
        out[name] = [[int((unburnt & ~predicted).sum()), int((unburnt & predicted).sum())],
                     [int((burnt & ~predicted).sum()), int((burnt & predicted).sum())]]
    return out


def two_class_metrics(matrix):
    (tn, fp), (fn, tp) = np.asarray(matrix, dtype=np.int64)

    def scores(hit, false_positive, false_negative):
        precision = hit / (hit + false_positive) if hit + false_positive else 0.0
        recall = hit / (hit + false_negative) if hit + false_negative else 0.0
        f1 = 2 * hit / (2 * hit + false_positive + false_negative) if hit else 0.0
        iou = hit / (hit + false_positive + false_negative) if hit else 0.0
        return precision, recall, f1, iou
    burnt, unburnt = scores(tp, fp, fn), scores(tn, fn, fp)
    return {"burnt_precision": burnt[0], "burnt_recall": burnt[1], "burnt_f1": burnt[2], "burnt_iou": burnt[3],
            "unburnt_precision": unburnt[0], "unburnt_recall": unburnt[1], "unburnt_f1": unburnt[2],
            "unburnt_iou": unburnt[3], "mean_f1": (burnt[2] + unburnt[2]) / 2, "mean_iou": (burnt[3] + unburnt[3]) / 2}


def save_run(out, name, kind, split, seed, extra, val_matrix, test_tiles, tile_ids, started):
    run_dir = Path(out) / f"{name}__{split}__{TASK['name']}__seed{seed}"
    run_dir.mkdir(parents=True, exist_ok=True)
    test_tiles = np.asarray(test_tiles, dtype=np.int64)
    np.savez_compressed(run_dir / "test_tile_confusion.npz", tile_id=np.asarray(tile_ids, dtype=str), matrices=test_tiles,
                        classes=np.array(["unburnt", "burnt"]))
    result = {"name": name, "kind": kind, "split": split, "task": TASK["name"], "seed": seed, **extra,
              "minutes": (time.time() - started) / 60, "val": two_class_metrics(val_matrix),
              "test": two_class_metrics(test_tiles.sum(0))}
    (run_dir / "metrics.json").write_text(json.dumps(result, indent=1))
    print(f"{name} seed {seed}: test burnt F1 {result['test']['burnt_f1']:.4f}, IoU {result['test']['burnt_iou']:.4f}", flush=True)


def run_spectral(args, paths, ids, pool):
    started = time.time()
    if all((Path(args.out) / f"{n}__{args.split_col}__{TASK['name']}__seed0" / "metrics.json").exists() for n in INDEX_NAMES):
        print("Spectral indices already finished")
        return
    rng = np.random.default_rng(0)
    sample_ids = rng.choice(len(paths["train"]), min(500, len(paths["train"])), replace=False)
    samples = list(pool.map(_sample_values, [paths["train"][i] for i in sample_ids], chunksize=8))
    ranges = {}
    for name in INDEX_NAMES:
        values = np.concatenate([s[name] for s in samples])
        low, high = np.percentile(values[np.isfinite(values)], [0.1, 99.9])
        ranges[name] = (float(low), float(high) if high > low else float(low) + 1e-3)
    bins = args.threshold_bins
    burnt = {n: np.zeros(bins, np.int64) for n in INDEX_NAMES}
    unburnt = {n: np.zeros(bins, np.int64) for n in INDEX_NAMES}
    for hist in pool.map(_histogram_tile, [(p, ranges, bins) for p in paths["train"]], chunksize=8):
        for name in INDEX_NAMES:
            burnt[name] += hist[name][0]
            unburnt[name] += hist[name][1]

    rules = {}
    for name in INDEX_NAMES:
        # F1 for every threshold, in both directions, from cumulative histograms.
        cum_b, cum_u = np.cumsum(burnt[name])[:-1], np.cumsum(unburnt[name])[:-1]
        total_b, total_u = burnt[name].sum(), unburnt[name].sum()
        f1_lower = 2 * cum_b / np.maximum(cum_b + cum_u + total_b, 1)
        tp_upper, fp_upper = total_b - cum_b, total_u - cum_u
        f1_upper = 2 * tp_upper / np.maximum(tp_upper + fp_upper + total_b, 1)
        thresholds = np.linspace(*ranges[name], bins + 1)[1:-1]
        if f1_lower.max() >= f1_upper.max():
            rules[name] = (float(thresholds[int(f1_lower.argmax())]), True)
        else:
            rules[name] = (float(thresholds[int(f1_upper.argmax())]), False)

    val_total = {n: np.zeros((2, 2), np.int64) for n in INDEX_NAMES}
    for result in pool.map(_threshold_tile, [(p, rules) for p in paths["val"]], chunksize=8):
        for name in INDEX_NAMES:
            val_total[name] += np.asarray(result[name])
    test_tiles = {n: [] for n in INDEX_NAMES}
    for result in pool.map(_threshold_tile, [(p, rules) for p in paths["test"]], chunksize=8):
        for name in INDEX_NAMES:
            test_tiles[name].append(result[name])
    for name in INDEX_NAMES:
        threshold, lower = rules[name]
        save_run(args.out, name, "index", args.split_col, 0, {"threshold": threshold, "lower_is_burnt": lower},
                 val_total[name], test_tiles[name], ids["test"], started)


def features(image):
    values = indices(image)
    stacked = np.concatenate([image, np.stack([values[n] for n in INDEX_NAMES])], axis=0)
    return np.nan_to_num(stacked, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)


def _sample_pixels(job):
    paths, seed, per_tile = job
    image, mask = load_tile(paths)
    rng = np.random.default_rng(seed)
    feats = features(image)
    labels = LABELS[mask]
    picked = []
    for wanted, limit in ((labels == 1, per_tile), ((labels == 0) & (mask != WATER_VALUE), per_tile),
                          (mask == WATER_VALUE, max(per_tile // 9, 1))):
        rows, cols = np.nonzero(wanted)
        if len(rows):
            chosen = rng.choice(len(rows), min(limit, len(rows)), replace=False)
            picked.append((rows[chosen], cols[chosen]))
    if not picked:
        return np.empty((0, feats.shape[0]), np.float32), np.empty(0, np.uint8), np.empty(0, bool)
    rows, cols = np.concatenate([p[0] for p in picked]), np.concatenate([p[1] for p in picked])
    return feats[:, rows, cols].T, labels[rows, cols], mask[rows, cols] == WATER_VALUE


def _grid_pixels(job):
    paths, stride = job
    image, mask = load_tile(paths)
    feats = features(image)[:, ::stride, ::stride]
    labels = LABELS[mask][::stride, ::stride]
    return feats.reshape(feats.shape[0], -1).T, labels.ravel()


def training_pixels(paths, seed, n_pixels, pool):
    per_tile = max(int(n_pixels / max(len(paths), 1)), 20)
    jobs = [(p, seed + i, per_tile) for i, p in enumerate(paths)]
    parts = list(pool.map(_sample_pixels, jobs, chunksize=8))
    x = np.concatenate([p[0] for p in parts])
    y = np.concatenate([p[1] for p in parts])
    water = np.concatenate([p[2] for p in parts])
    rng = np.random.default_rng(seed)
    half = n_pixels // 2
    burnt = np.flatnonzero(y == 1)
    water_ids = np.flatnonzero(water)
    land = np.flatnonzero((y == 0) & ~water)
    chosen = np.concatenate([
        rng.choice(burnt, min(half, len(burnt)), replace=False),
        rng.choice(land, min(half - half // 10, len(land)), replace=False),
        rng.choice(water_ids, min(half // 10, len(water_ids)), replace=False) if len(water_ids) else water_ids,
    ])
    rng.shuffle(chosen)
    return x[chosen], y[chosen]


def make_classifier(method, seed, args):
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.kernel_approximation import Nystroem
    from sklearn.neural_network import MLPClassifier
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.svm import LinearSVC
    if method == "RF":
        return RandomForestClassifier(n_estimators=200, max_depth=24, min_samples_leaf=5, n_jobs=args.workers, random_state=seed)
    if method == "XGBoost":
        from xgboost import XGBClassifier
        return XGBClassifier(n_estimators=400, max_depth=8, learning_rate=.1, subsample=.8, colsample_bytree=.8,
                             tree_method="hist", device="cuda" if args.xgb_gpu else "cpu", n_jobs=args.workers, random_state=seed)
    if method == "SVM":
        # An RBF kernel, approximated: an exact one can't fit this many pixels.
        return make_pipeline(StandardScaler(), Nystroem(n_components=500, random_state=seed),
                             LinearSVC(C=1.0, dual=False, random_state=seed, max_iter=5000))
    if method == "MLP":
        return make_pipeline(StandardScaler(), MLPClassifier(hidden_layer_sizes=(256, 256), batch_size=4096, max_iter=60,
                                                             early_stopping=True, random_state=seed))
    raise ValueError(method)


def evaluate_classifier(model, paths, stride, pool, tiles_per_batch=256):
    """One confusion matrix per tile. Predicts in batches of tiles, which is far faster than tile by tile."""
    matrices = []
    features_batch, labels_batch = [], []

    def flush():
        predictions = model.predict(np.concatenate(features_batch)) if sum(map(len, features_batch)) else np.empty(0)
        start = 0
        for truth in labels_batch:
            predicted = predictions[start:start + len(truth)].astype(np.int64)
            start += len(truth)
            matrices.append(np.bincount(2 * truth.astype(np.int64) + predicted, minlength=4).reshape(2, 2))
        features_batch.clear()
        labels_batch.clear()

    for x, y in pool.map(_grid_pixels, [(p, stride) for p in paths], chunksize=4):
        valid = y != IGNORE
        features_batch.append(x[valid])
        labels_batch.append(y[valid])
        if len(labels_batch) >= tiles_per_batch:
            flush()
    flush()
    return np.stack(matrices)


def run_classifier(method, seed, args, paths, ids, pool):
    run_dir = Path(args.out) / f"{method}__{args.split_col}__{TASK['name']}__seed{seed}"
    if (run_dir / "metrics.json").exists():
        print(f"{method} seed {seed}: already finished")
        return
    started = time.time()
    n_pixels = args.svm_pixels if method == "SVM" else args.n_pixels
    x, y = training_pixels(paths["train"], seed, n_pixels, pool)
    print(f"{method} seed {seed}: fitting on {len(y):,} pixels", flush=True)
    model = make_classifier(method, seed, args)
    model.fit(x, y.astype(np.int64))
    val = evaluate_classifier(model, paths["val"], args.eval_stride, pool).sum(0)
    test = evaluate_classifier(model, paths["test"], args.eval_stride, pool)
    save_run(args.out, method, "pixel", args.split_col, seed, {"training_pixels": int(len(y)), "eval_stride": args.eval_stride},
             val, test, ids["test"], started)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--index", default="data/v2/index.csv")
    parser.add_argument("--images-root", default="data/v1")
    parser.add_argument("--masks-root", default="data/v2")
    parser.add_argument("--split-col", default="split")
    parser.add_argument("--out", default="results")
    parser.add_argument("--workers", type=int, default=os.cpu_count())
    parser.add_argument("--cache-root", default="", help="NumPy files from build_cache.py, which are much quicker to read")
    parser.add_argument("--methods", nargs="+", default=["spectral", "RF", "XGBoost", "SVM", "MLP"])
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    parser.add_argument("--n-pixels", type=int, default=2_000_000)
    parser.add_argument("--svm-pixels", type=int, default=300_000)
    parser.add_argument("--eval-stride", type=int, default=4)
    parser.add_argument("--threshold-bins", type=int, default=2000)
    parser.add_argument("--xgb-gpu", action="store_true")
    parser.add_argument("--limit", type=int, default=0, help="use only this many tiles per split, for a quick trial")
    parser.add_argument("--target", choices=["burn", "fresh"], default="burn",
                        help="burn: fresh and old burn are burnt. fresh: only fresh burn is")
    args = parser.parse_args()
    set_target(args.target)

    index = pd.read_csv(args.index, low_memory=False)
    paths, ids = {}, {}
    for split in ("train", "val", "test"):
        frame = index[index[args.split_col] == split]
        if args.limit:
            frame = frame.head(args.limit)
        ids[split] = frame["tile_id"].tolist()
        if args.cache_root:
            paths[split] = [(str(Path(args.cache_root) / f"{i}.npy"), str(Path(args.cache_root) / f"{m}.npy"))
                            for i, m in zip(frame["image_path"], frame["mask_path"])]
        else:
            paths[split] = [(str(Path(args.images_root) / i), str(Path(args.masks_root) / m))
                            for i, m in zip(frame["image_path"], frame["mask_path"])]
    print({k: len(v) for k, v in paths.items()}, flush=True)
    with ProcessPoolExecutor(args.workers, initializer=set_target, initargs=(args.target,)) as pool:
        if "spectral" in args.methods:
            run_spectral(args, paths, ids, pool)
        for method in [m for m in args.methods if m != "spectral"]:
            for seed in args.seeds:
                run_classifier(method, seed, args, paths, ids, pool)


if __name__ == "__main__":
    main()
