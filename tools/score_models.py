#!/usr/bin/env python3
"""The trained baselines scored the ways the referee asks, on the CPU, with no training.

  python tools/score_models.py matrices                      # seconds: from the saved test matrices
  python tools/score_models.py additions --seeds 0 1 2 3 4   # the new tiles of acquire_additions.py
  python tools/score_models.py cems                          # against the CEMS maps, beside the labels
  python tools/score_models.py points                        # against the reviewed points, beside the labels

matrices   Fresh-burn F1 with cloud and cloud shadow pixels counted as negatives (the paper's scores leave them out),
           and the share of cloud and shadow pixels predicted fresh_burn, from each run's test_tile_confusion.npz.
additions  negative: fresh_burn predicted on burn-free tiles (share of pixels; tiles with at least 0.1 ha and 1 ha).
           boundary: on the burn pixels of scenes 61 to 120 days after the fire, the share predicted fresh and old by
           days from the fire, and the age accuracy either side of 90 days. large: fresh F1 on the edges of fires above
           2000 ha. firstlook: fresh F1 and recall 0 to 5 days after the fire, cloud and shadow counted as negatives.
cems       The models' fresh_burn against the CEMS burned area, on the pixels where the labels were compared
           (make_label_figures.py, part cems), pooled with intervals over activations; all pairs and test-split pairs.
points     The models' class at each reviewed point against the final labels and each interpreter, weighted by the
           strata as for the labels (Stehman 2014: a sample stratified by one map can score another), on all points and on
           the test split.

Models run on the CPU (CUDA_VISIBLE_DEVICES is emptied), so booked GPUs are untouched. Writes results_v4/model_scores_<command>.md.
"""
import os
import sys

os.environ["CUDA_VISIBLE_DEVICES"] = ""   # before torch loads

import argparse  # noqa: E402
import json  # noqa: E402
from argparse import Namespace  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))


def two_class(matrix, positive, negative):
    """One class against others, rows labels and columns predictions; label rows in neither list are left out."""
    tp = matrix[np.ix_(positive, positive)].sum()
    fn = matrix[positive].sum() - tp
    fp = matrix[np.ix_(negative, positive)].sum()
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    return {"burnt_precision": precision, "burnt_recall": recall, "burnt_f1": 2 * tp / max(2 * tp + fp + fn, 1),
            "positive_px": int(matrix[positive].sum())}


CLASSES = ["clear", "fresh_burn", "old_burn", "cloud", "shadow", "water"]
FRESH, NEGATIVE, WITH_CLOUD = [1], [0, 2, 5], [0, 2, 3, 4, 5]


def md(frame):
    return "\n".join(["| " + " | ".join(map(str, frame.columns)) + " |", "|" + "---|" * len(frame.columns)]
                     + ["| " + " | ".join(str(v) for v in row) + " |" for row in frame.itertuples(index=False)])


def mean_sd(values):
    values = [v for v in values if v == v]
    if not values:
        return ""
    return f"{100 * np.mean(values):.2f} ± {100 * np.std(values, ddof=1):.2f}" if len(values) > 1 else f"{100 * values[0]:.2f}"


def runs(results, models, seeds=None):
    for model in models:
        for folder in sorted(Path(results).glob(f"{model}__split__multiclass__seed*")):
            seed = int(folder.name.rsplit("seed", 1)[1])
            if (seeds is None or seed in seeds) and (folder / "best.pt").exists():
                yield model, seed, folder


# ---------------------------------------------------------------- matrices

def run_matrices(args):
    rows = []
    for model in args.models:
        found = sorted(Path(args.results).glob(f"{model}__split__multiclass__seed*/test_tile_confusion.npz"))
        paper, cloud, on_cloud, on_shadow = [], [], [], []
        for path in found:
            matrix = np.load(path, allow_pickle=False)["matrices"].sum(0)
            paper.append(two_class(matrix, FRESH, NEGATIVE)["burnt_f1"])
            cloud.append(two_class(matrix, FRESH, WITH_CLOUD)["burnt_f1"])
            on_cloud.append(matrix[3, 1] / max(matrix[3].sum(), 1))
            on_shadow.append(matrix[4, 1] / max(matrix[4].sum(), 1))
        if found:
            rows.append({"Model": model, "Seeds": len(found), "Fresh F1, as in the paper": mean_sd(paper),
                         "Fresh F1, cloud and shadow as negatives": mean_sd(cloud),
                         "Cloud pixels predicted fresh %": mean_sd(on_cloud), "Shadow pixels predicted fresh %": mean_sd(on_shadow)})
    write(args, "matrices", ["# Fresh-burn scores with cloud and shadow counted", "",
                             "Test set of the spatial split; mean ± standard deviation over seeds.", "", md(pd.DataFrame(rows))])


# ---------------------------------------------------------------- inference

def load_model(folder, results):
    import torch

    from train_baseline import ALL_BANDS, build_model
    metrics = json.loads((folder / "metrics.json").read_text())
    channels = [ALL_BANDS.index(b) for b in metrics.get("bands", ALL_BANDS)]
    if metrics["spec"].get("norm") == "reflectance":
        mean, std = np.zeros(13, np.float32), np.ones(13, np.float32)
    else:
        stats = json.loads((Path(results) / "band_stats.json").read_text())[metrics["split"]]
        mean, std = np.array(stats["mean"], np.float32), np.array(stats["std"], np.float32)
    model = build_model({**metrics["spec"], "weights": None}, len(CLASSES), len(channels))
    model.load_state_dict(torch.load(folder / "best.pt", map_location="cpu", weights_only=True))
    model.eval()
    return model, channels, mean, std


def predict(model, channels, mean, std, frame, images_root, masks_root, args):
    """Predicted classes and mask values for every row of frame, in order, as two (n, 256, 256) arrays."""
    import torch
    from torch.utils.data import DataLoader

    from extra_inference import TestTiles
    loader = DataLoader(TestTiles(frame, images_root, masks_root, mean, std, channels), batch_size=args.batch,
                        num_workers=args.workers)
    predicted, masks = [], []
    with torch.no_grad():
        for images, mask, _ in loader:
            predicted.append(model(images).argmax(1).numpy().astype(np.uint8))
            masks.append(mask.numpy().astype(np.uint8))
    return np.concatenate(predicted), np.concatenate(masks)


def label_classes(mask):
    """Mask values to the models' classes (clear 0, fresh 1, old 2, cloud 3, shadow 4, water 5); 255 elsewhere."""
    table = np.full(256, 255, np.uint8)
    for value, index in {0: 0, 2: 1, 3: 2, 4: 3, 5: 4, 6: 5}.items():
        table[value] = index
    return table[mask]


def confusion(labels, predicted):
    valid = labels != 255
    return np.bincount(6 * labels[valid].astype(np.int64) + predicted[valid], minlength=36).reshape(6, 6)


# ---------------------------------------------------------------- additions

def run_additions(args):
    import torch
    torch.set_num_threads(args.threads)
    out = Path(args.additions)
    index = pd.read_csv(out / "v4" / "additions_index.csv", low_memory=False)
    rows = {part: [] for part in ("negative", "boundary", "large", "firstlook")}
    for model_name, seed, folder in runs(args.results, args.models, set(args.seeds)):
        print(f"{folder.name} on {len(index):,} new tiles ...", flush=True)
        model, channels, mean, std = load_model(folder, args.results)
        predicted, masks = predict(model, channels, mean, std, index, out / "raw", out / "v4", args)
        labels = label_classes(masks)
        valid = masks != 255
        for part in rows:
            mine = (index["subset"] == part).to_numpy()
            if not mine.any():
                continue
            frame, pred, lab, ok = index[mine], predicted[mine], labels[mine], valid[mine]
            result = {"Model": model_name, "Seed": seed, "Tiles": int(mine.sum())}
            if part == "negative":
                fresh = ((pred == 1) & ok).sum(axis=(1, 2))
                result.update({"Pixels predicted fresh %": 100 * fresh.sum() / max(ok.sum(), 1),
                               "Tiles with 0.1 ha or more %": 100 * (fresh >= 10).mean(),
                               "Tiles with 1 ha or more %": 100 * (fresh >= 100).mean(),
                               "Pixels predicted fresh or old %": 100 * (((pred == 1) | (pred == 2)) & ok).sum() / max(ok.sum(), 1)})
            elif part == "boundary":
                burn = (lab == 1) | (lab == 2)
                days = frame["age_days"].to_numpy()
                for low, high in ((61, 75), (76, 90), (91, 105), (106, 120)):
                    pick = (days >= low) & (days < high + 1)
                    sel = burn[pick]
                    result[f"{low}-{high} d, predicted fresh %"] = 100 * (pred[pick][sel] == 1).sum() / max(sel.sum(), 1)
                predicted_burn = burn & ((pred == 1) | (pred == 2))
                right = predicted_burn & (pred == lab)
                for name, side in (("90 days or less", days <= 90), ("over 90 days", days > 90)):
                    result[f"Age accuracy, {name} %"] = 100 * right[side].sum() / max(predicted_burn[side].sum(), 1)
            else:
                matrix = sum(confusion(lab[i], pred[i]) for i in range(len(frame)))
                scores = two_class(matrix, FRESH, WITH_CLOUD)
                result.update({"Fresh F1 %": 100 * scores["burnt_f1"], "Precision %": 100 * scores["burnt_precision"],
                               "Recall %": 100 * scores["burnt_recall"]})
                if part == "large":
                    test = (frame["split"] == "test").to_numpy()
                    if test.any():
                        tm = sum(confusion(lab[i], pred[i]) for i in np.flatnonzero(test))
                        result["Fresh F1, test blocks %"] = 100 * two_class(tm, FRESH, WITH_CLOUD)["burnt_f1"]
                else:
                    days = frame["age_days"].round().to_numpy()
                    for name, pick in (("day 0", days <= 0), ("days 1-2", (days >= 1) & (days <= 2)), ("days 3-5", days >= 3)):
                        if pick.any():
                            dm = sum(confusion(lab[i], pred[i]) for i in np.flatnonzero(pick))
                            result[f"Fresh recall, {name} %"] = 100 * two_class(dm, FRESH, WITH_CLOUD)["burnt_recall"]
            rows[part].append(result)
    lines = ["# The baselines on the new tiles", "",
             "Trained on the released tiles only; nothing was retrained. Cloud and shadow pixels count as negatives.", ""]
    titles = {"negative": "Burn-free tiles: false alarms", "boundary": "Scenes 61-120 days after the fire: the 90-day line",
              "large": "Edges of fires above 2000 ha", "firstlook": "First look, 0-5 days after the fire"}
    for part, found in rows.items():
        if found:
            frame = pd.DataFrame(found).round(2)
            lines += [f"## {titles[part]}", "", md(frame), "", "Mean over seeds: " + ", ".join(
                f"{c} {frame[c].mean():.2f}" for c in frame.columns if c not in ("Model", "Seed", "Tiles")), ""]
    write(args, "additions", lines)


# ---------------------------------------------------------------- cems

def run_cems(args):
    import torch
    torch.set_num_threads(args.threads)
    import make_label_figures as mlf
    index = pd.read_csv(args.index, low_memory=False)
    options = Namespace(cems_tiles=args.cems_tiles, cems_dnbr=args.cems_dnbr, cems_root=args.cems_root, index=args.index,
                        masks_root=args.masks_root, images_root=args.images_root)
    jobs, skipped, total = mlf.cems_jobs(index[index["split"].notna()], options)
    print(f"{len(jobs)} of {total} pairs compared; left out: {dict(skipped)}", flush=True)
    arrays = []
    for job in jobs:
        mask, target, dnbr = mlf.cems_arrays(job)
        keep = (target > 0) & np.isin(mask, mlf.KEEP_CLASSES) & np.isfinite(dnbr)
        arrays.append((keep, target == 2, mask == 2, job["activation"]))
    frame = index.drop_duplicates("tile_id").set_index("tile_id").loc[[job["tile_id"] for job in jobs]].reset_index()
    test = frame["split"].eq("test").to_numpy()

    def pooled(burned_maps, name):
        counts = pd.DataFrame([{"activation": a, **{f"m_{k}": v for k, v in mlf.confusion(b, r, keep).items()}}
                               for b, (keep, r, _, a) in zip(burned_maps, arrays)])
        rows = []
        for label, pick in (("all pairs", np.ones(len(frame), bool)), ("test-split pairs", test)):
            if pick.any():
                scores = mlf.by_activation(counts[pick], "m", args.reps, args.seed)
                rows.append({"Map": name, "Pairs": label, "Pairs n": int(pick.sum()),
                             "Activations": counts[pick]["activation"].nunique(), **{k.capitalize(): v for k, v in scores.items()}})
        return rows

    table = pooled([labels for _, _, labels, _ in arrays], "burn2scar fresh_burn labels")
    for model_name, seed, folder in runs(args.results, args.models, set(args.seeds)):
        print(f"{folder.name} on {len(frame)} CEMS tiles ...", flush=True)
        model, channels, mean, std = load_model(folder, args.results)
        predicted, _ = predict(model, channels, mean, std, frame, args.images_root, args.masks_root, args)
        table += pooled([p == 1 for p in predicted], f"{model_name}, seed {seed}")
    write(args, "cems", ["# Against the CEMS maps: labels and models on the same pixels", "",
                         "Pixels as in make_label_figures.py (part cems): inside the CEMS area, label classes clear, burn, "
                         "water or older_burn, and a dNBR value. Intervals resample activations. The models were trained on "
                         "the EFFIS labels of the training tiles, so the test-split pairs are the fair comparison.", "",
                         md(pd.DataFrame(table))])


# ---------------------------------------------------------------- points

def run_points(args):
    import torch
    torch.set_num_threads(args.threads)
    from review_points import MAP_CLASS, final_labels, latest_answers, reference_class
    review = Path(args.review)
    points = pd.read_csv(review / "points.csv", low_memory=False)
    first = points[points["repeat_of"] < 0].copy()
    area = pd.read_csv(review / "strata_area.csv").set_index("stratum")["pixels"].astype(float)
    index = pd.read_csv(args.index, low_memory=False).drop_duplicates("tile_id").set_index("tile_id")
    tiles = index.loc[first["tile_id"].unique()].reset_index()
    first["split"] = first["tile_id"].map(index["split"])
    references = {f"Interpreter {i}": latest_answers(review / f"answers_{n}.csv") for i, n in enumerate(args.reviewers, 1)}
    references["Final"] = final_labels(points, review, args.reviewers, args.judge)
    to_reference = {0: "not_burned", 1: "burned", 2: "burned", 3: "cloud", 4: "shadow", 5: "water"}

    def accuracy(classes, reference, pick):
        """Mean agreement within each stratum, weighted by the strata's areas, as review_points.py scores the labels."""
        rows = first[pick].assign(cls=classes[pick])
        rows["ref"] = rows["point_id"].map(lambda p: reference_class(reference[p]) if reference.get(p) not in (None, "cant_tell") else None)
        rows = rows[rows["ref"].notna()]
        if rows.empty:
            return np.nan
        hit = (rows["cls"] == rows["ref"]).groupby(rows["stratum"]).mean()
        w = area.reindex(hit.index).fillna(0)
        return 100 * float((hit * w).sum() / w.sum())

    table = []
    label_classes_ = first["stratum"].map(lambda s: "burned" if MAP_CLASS[s] in ("burned", "older_burn") else MAP_CLASS[s]).to_numpy()
    for name, pick in (("all points", np.ones(len(first), bool)), ("test split", first["split"].eq("test").to_numpy())):
        table.append({"Map": "burn2scar labels (older_burn counted as burned)", "Points": name, "n": int(pick.sum()),
                      **{f"Overall accuracy vs {r} %": round(accuracy(label_classes_, ref, pick), 1) for r, ref in references.items()}})
    for model_name, seed, folder in runs(args.results, args.models, set(args.seeds)):
        print(f"{folder.name} on {len(tiles)} reviewed tiles ...", flush=True)
        model, channels, mean, std = load_model(folder, args.results)
        predicted, _ = predict(model, channels, mean, std, tiles, args.images_root, args.masks_root, args)
        position = {t: i for i, t in enumerate(tiles["tile_id"])}
        classes = np.array([to_reference[int(predicted[position[r.tile_id], int(r.row), int(r.col)])] for r in first.itertuples()])
        for name, pick in (("all points", np.ones(len(first), bool)), ("test split", first["split"].eq("test").to_numpy())):
            table.append({"Map": f"{model_name}, seed {seed}", "Points": name, "n": int(pick.sum()),
                          **{f"Overall accuracy vs {r} %": round(accuracy(classes, ref, pick), 1) for r, ref in references.items()}})
    write(args, "points", ["# Against the reviewed points: labels and models", "",
                           "Area-weighted overall accuracy: the agreement within each stratum, weighted by the stratum's area. For the "
                           "models this is the estimator for a map other than the one the sample was stratified by (Stehman, 2014). "
                           "Classes: burned, not burned, water, cloud, shadow; can't tell left out. The models were trained on "
                           "the labels of the training tiles, so the test split is the fair comparison.", "", md(pd.DataFrame(table))])


# ---------------------------------------------------------------- burns a regional agency mapped and EFFIS did not

def run_agency(args):
    """On the released tiles of one region: what the models call the burn of agency perimeters that the labels call
    clear, split into the outline differences of fires EFFIS also mapped and the fires EFFIS did not map, against the
    burn both mapped and against clear land with no agency perimeter at all; for all tiles and for the test split."""
    import geopandas as gpd
    import torch
    import yaml
    from rasterio.features import rasterize
    from scipy.ndimage import distance_transform_edt
    from rasterio.transform import from_bounds
    from shapely.geometry import box

    from burn_labels import load_perimeters
    from compare_agency import match, naive
    torch.set_num_threads(args.threads)
    agency = gpd.read_file(args.agency_file)
    agency["date"] = naive(pd.to_datetime(agency["cfva_date"] if "cfva_date" in agency else agency[args.agency_date_col]))
    agency = agency[agency["date"].notna() & ~agency.geometry.is_empty].reset_index(drop=True)
    agency["geometry"] = agency.geometry.make_valid()
    agency["agency_id"] = np.arange(len(agency))
    first, last = agency["date"].min(), agency["date"].max()
    # Which agency fires EFFIS mapped: the same rule as compare_agency.py.
    equal = agency.to_crs(3035)
    equal["agency_date"] = equal["date"]
    equal["agency_ha"] = equal.geometry.area / 1e4
    cfg = yaml.safe_load(Path(args.config).read_text())
    effis = load_perimeters(cfg, args.effis_extra)
    effis = effis[effis.intersects(box(*agency.to_crs(4326).total_bounds).buffer(.05))].drop_duplicates("fire_id")
    effis = effis.to_crs(3035).reset_index(drop=True)
    effis["geometry"] = effis.geometry.make_valid()
    effis["fire_date"] = naive(effis["fire_date"])
    pairs = match(equal, effis, args.days_before, args.days_after)
    agency["matched"] = agency["agency_id"].isin(pairs["agency_id"])
    agency["large"] = equal["agency_ha"].to_numpy() >= args.min_ha
    print(f"Agency fires: {len(agency):,}; matched by EFFIS: {int(agency['matched'].sum()):,}", flush=True)

    index = pd.read_csv(args.index, low_memory=False)
    tiles = index[index["split"].notna() & (index["region"] == args.agency_region)].copy()
    tiles["scene"] = pd.to_datetime(tiles["scene_time"], format="mixed", utc=True).dt.tz_convert(None)
    # Only scenes whose 90 days before are all inside the agency's record.
    tiles = tiles[(tiles["scene"] >= first + pd.Timedelta(days=90)) & (tiles["scene"] <= last)].reset_index(drop=True)
    if tiles.empty:
        raise SystemExit(f"No released tiles in region {args.agency_region} inside the agency's dates.")
    layers = {key: [] for key in ("any_recent", "matched", "missed_small", "missed_large", "any_date")}
    by_crs = {}
    for r in tiles.itertuples():
        crs = f"EPSG:{int(r.epsg)}"
        if crs not in by_crs:
            by_crs[crs] = agency.to_crs(crs)
        part = by_crs[crs].cx[r.minx:r.maxx, r.miny:r.maxy]
        transform = from_bounds(r.minx, r.miny, r.maxx, r.maxy, 256, 256)
        burn = lambda rows: (rasterize(((g, 1) for g in rows.geometry), out_shape=(256, 256), transform=transform,
                                       fill=0, dtype="uint8").astype(bool) if len(rows) else np.zeros((256, 256), bool))
        recent = part[(part["date"] <= r.scene) & (part["date"] >= r.scene - pd.Timedelta(days=90))]
        layers["any_recent"].append(burn(recent))
        layers["matched"].append(burn(recent[recent["matched"]]))
        layers["missed_small"].append(burn(recent[~recent["matched"] & ~recent["large"]]))
        layers["missed_large"].append(burn(recent[~recent["matched"] & recent["large"]]))
        layers["any_date"].append(burn(part[part["date"] <= r.scene]))
    layers = {key: np.stack(value) for key, value in layers.items()}
    missed = (layers["missed_small"] | layers["missed_large"]) & ~layers["matched"]
    trained = (tiles["split"] == "train").to_numpy()[:, None, None]
    splits = [("all", np.ones_like(trained)), ("train: learned as clear", trained), ("val or test: not trained on", ~trained)]
    results = {}
    for model_name, seed, folder in runs(args.results, args.models, args.seeds):
        print(f"{folder.name} on {len(tiles)} tiles ...", flush=True)
        model, channels, mean, std = load_model(folder, args.results)
        pred, masks = predict(model, channels, mean, std, tiles, args.images_root, args.masks_root, args)
        clear = masks == 0
        near = np.zeros(masks.shape, bool)       # within 20 m (2 pixels) of a fresh_burn label
        for k in range(len(masks)):
            if (masks[k] == 2).any():
                near[k] = distance_transform_edt(masks[k] != 2) <= 2
        edge = clear & layers["matched"]
        groups = {"Both mapped it (fresh_burn, agency 0-90 days)": (masks == 2) & layers["any_recent"],
                  "Only the agency: edge of a fire EFFIS also mapped (clear)": edge,
                  "... of it, within 20 m of the EFFIS burn": edge & near,
                  "... of it, farther than 20 m": edge & ~near,
                  "Only the agency: a fire EFFIS did not map (clear)": clear & missed,
                  f"... of it, agency fires under {args.min_ha:g} ha": clear & missed & layers["missed_small"] & ~layers["missed_large"],
                  f"... of it, agency fires of {args.min_ha:g} ha or more": clear & missed & layers["missed_large"],
                  "Control: clear within 20 m of the EFFIS burn, no agency perimeter of any date": clear & near & ~layers["any_date"],
                  "Clear, no agency perimeter of any date": clear & ~layers["any_date"]}
        for split_name, keep in splits:
            for name, where in groups.items():
                where = where & keep
                key = (split_name, name)
                results.setdefault(key, {"pixels": int(where.sum()), "tiles": int(where.any(axis=(1, 2)).sum())})
                results[key].setdefault((model_name, "fresh"), []).append(float((pred[where] == 1).mean()) if where.any() else np.nan)
                results[key].setdefault((model_name, "burn"), []).append(float(np.isin(pred[where], [1, 2]).mean()) if where.any() else np.nan)
    rows = []
    for (split_name, name), r in results.items():
        if not r["pixels"]:
            continue
        row = {"Tiles of": split_name, "Pixels": name, "Tiles": r["tiles"], "Area, ha": f"{r['pixels'] / 100:,.1f}"}
        for model_name in args.models:
            if (model_name, "fresh") in r:
                row[f"{model_name}: called fresh_burn %"] = mean_sd(r[(model_name, "fresh")])
                row[f"{model_name}: called fresh or old burn %"] = mean_sd(r[(model_name, "burn")])
        rows.append(row)
    write(args, "agency", [f"# The models on burns that {args.agency_region}'s agency mapped and EFFIS did not", "",
                           f"{len(tiles)} released tiles in {args.agency_region} ({int(trained.sum())} in the training split), "
                           f"scenes {tiles['scene'].min().date()} to {tiles['scene'].max().date()}. Agency perimeters dated "
                           "0-90 days before the scene, rasterised on the tile grid. An agency fire counts as mapped by EFFIS "
                           f"when an EFFIS perimeter dated {args.days_before} days before to {args.days_after} days after it "
                           "overlaps it, as in compare_agency.py. The models were trained on the EFFIS labels, in which the "
                           "agency-only burn is clear: on training tiles they learned these very pixels as clear. The control "
                           "is clear land next to the EFFIS burn where the agency mapped nothing, to tell real burn beyond "
                           "the EFFIS outline from a model's habit of widening the burn it finds. Mean ± standard deviation "
                           "over seeds.", "", md(pd.DataFrame(rows))])


def write(args, name, lines):
    path = Path(args.results) / f"model_scores_{name}.md"
    path.write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    print(f"Wrote {path}")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("command", choices=["matrices", "additions", "cems", "points", "agency"])
    parser.add_argument("--results", default="results_v4")
    parser.add_argument("--models", nargs="+", default=["unet_r34", "segformer_b2"])
    parser.add_argument("--seeds", nargs="+", type=int, default=[0])
    parser.add_argument("--index", default="data/v4/index.csv")
    parser.add_argument("--images-root", default="data/v1")
    parser.add_argument("--masks-root", default="data/v4")
    parser.add_argument("--additions", default="data/additions")
    parser.add_argument("--cems-tiles", default="validation/cems_tiles.csv")
    parser.add_argument("--cems-dnbr", default="validation/dnbr_cems")
    parser.add_argument("--cems-root", default="data/cems")
    parser.add_argument("--review", default="review2")
    parser.add_argument("--reviewers", nargs=2, default=["param", "silvia"])
    parser.add_argument("--judge", default="judge")
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--threads", type=int, default=16, help="CPU threads for the models")
    parser.add_argument("--reps", type=int, default=2000)
    parser.add_argument("--agency-file", default="data/agency/sardinia/cfva_perimeters.gpkg")
    parser.add_argument("--agency-region", default="Sardegna")
    parser.add_argument("--agency-date-col", default="cfva_date")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--effis-extra", nargs="*", default=[])
    parser.add_argument("--days-before", type=int, default=15)
    parser.add_argument("--days-after", type=int, default=30)
    parser.add_argument("--min-ha", type=float, default=5)
    parser.add_argument("--seed", type=int, default=20261120)
    args = parser.parse_args()
    {"matrices": run_matrices, "additions": run_additions, "cems": run_cems, "points": run_points,
     "agency": run_agency}[args.command](args)


if __name__ == "__main__":
    main()
