#!/usr/bin/env python3
"""Turn every finished baseline run into the tables for the paper.

  python tools/summarize_results.py --results results --index data/v2/index.csv

Writes results/summary.md and CSV files with the same numbers. Values are the mean and
standard deviation across seeds. FLOGA's published numbers are listed for reference
only: FLOGA uses pre-fire and post-fire L2A images of Greece at 20 m, so the two sets
of results measure different tasks and should not be ranked against each other.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

METRICS = ["burnt_precision", "burnt_recall", "burnt_f1", "burnt_iou", "unburnt_f1", "unburnt_iou", "mean_f1", "mean_iou"]
HEADERS = {"burnt_precision": "Burnt P", "burnt_recall": "Burnt R", "burnt_f1": "Burnt F1", "burnt_iou": "Burnt IoU",
           "unburnt_f1": "Unburnt F1", "unburnt_iou": "Unburnt IoU", "mean_f1": "Mean F1", "mean_iou": "Mean IoU"}
FLOGA_FAMILY = {
    "unet_r18_scratch": "FC-EF-Conc", "unet_r34": "FC-EF-Conc", "unetpp_r34": "SNUNet-CD", "unetpp_r101": "BAM-CD",
    "hrnet_w18": "ADHR-CDNet", "segformer_b2": "ChangeFormer",
    "NBR": "dNBR", "NBR2": "dNBR2", "NBR+": "dNBR+", "MIRBI": "dMIRBI", "BAIS2": "dBAIS2", "NDVI": "dNDVI",
    "NDWI": "dNDWI", "EVI": "dEVI", "CSI": "dCSI", "SAVI": "dSAVI", "RF": "RF", "XGBoost": "XGBoost", "SVM": "SVM", "MLP": "MLP",
}
FLOGA_F1 = {"dNBR": 45.14, "dNBR2": 52.78, "dNBR+": 46.57, "dMIRBI": 64.06, "dBAIS2": 31.04, "dNDVI": 45.93,
            "dNDWI": 32.22, "dEVI": 30.89, "dCSI": 40.31, "dSAVI": 45.96, "RF": 67.38, "XGBoost": 61.49, "SVM": 68.93,
            "MLP": 56.52, "FC-EF-Conc": 72.18, "SNUNet-CD": 86.53, "ADHR-CDNet": 83.58, "ChangeFormer": 85.65, "BAM-CD": 88.48}
SIZE_BINS = [0, 10, 30, 100, 500, np.inf]
SIZE_LABELS = ["5-10 ha", "10-30 ha", "30-100 ha", "100-500 ha", "over 500 ha"]
KIND_ORDER = {"index": 0, "pixel": 1, "deep": 2}


def md_table(frame):
    head = "| " + " | ".join(map(str, frame.columns)) + " |"
    rule = "|" + "---|" * len(frame.columns)
    body = ["| " + " | ".join("" if pd.isna(v) else str(v) for v in row) + " |" for row in frame.itertuples(index=False)]
    return "\n".join([head, rule, *body])


def pm(values, scale=100):
    values = np.asarray(values, float) * scale
    return f"{values.mean():.2f}" if len(values) == 1 else f"{values.mean():.2f} ± {values.std(ddof=1):.2f}"


def load_runs(results):
    runs = []
    for path in sorted(Path(results).glob("*/metrics.json")):
        run = json.loads(path.read_text())
        run["dir"] = path.parent
        runs.append(run)
    return runs


def comparison_table(runs, split):
    rows = []
    groups = {}
    for run in runs:
        if run["split"] == split:
            groups.setdefault((run["kind"], run["name"]), []).append(run)
    for (kind, name), members in sorted(groups.items(), key=lambda kv: (KIND_ORDER.get(kv[0][0], 9), -np.mean([m["test"]["burnt_f1"] for m in kv[1]]))):
        family = FLOGA_FAMILY.get(name, "")
        row = {"Method": name, "Type": kind, "Seeds": len(members)}
        row.update({HEADERS[m]: pm([r["test"][m] for r in members]) for m in METRICS})
        row["FLOGA family"] = family
        row["FLOGA burnt F1 (reference)"] = FLOGA_F1.get(family, "")
        rows.append(row)
    return pd.DataFrame(rows)


def per_class_table(runs, split):
    rows = []
    groups = {}
    for run in runs:
        if run["split"] == split and run["task"] == "multiclass":
            groups.setdefault(run["name"], []).append(run)
    for name, members in sorted(groups.items(), key=lambda kv: -np.mean([m["test"]["class_miou"] for m in kv[1]])):
        row = {"Method": name, "Seeds": len(members), "mIoU": pm([r["test"]["class_miou"] for r in members])}
        for class_name in members[0]["test"]["per_class"]:
            row[f"{class_name} IoU"] = pm([r["test"]["per_class"][class_name]["iou"] for r in members])
        row["Fresh vs old accuracy"] = pm([r["test"]["fresh_old_accuracy"] for r in members])
        rows.append(row)
    return pd.DataFrame(rows)


def burnt_from_matrix(matrix, classes):
    names = list(classes)
    if "fresh_burn" in names:
        burnt, unburnt = [names.index("fresh_burn"), names.index("old_burn")], [names.index("clear"), names.index("water")]
    else:
        burnt, unburnt = [names.index("burnt")], [names.index("unburnt")]
    tp = matrix[np.ix_(burnt, burnt)].sum()
    fn = matrix[burnt].sum() - tp
    fp = matrix[np.ix_(unburnt, burnt)].sum()
    return {"f1": 2 * tp / max(2 * tp + fp + fn, 1), "iou": tp / max(tp + fp + fn, 1), "burnt_px": int(matrix[burnt].sum())}


def stratified(runs, index):
    """Burnt F1 and IoU by country, year, fire size, window and other strata, per method and seed."""
    index = index.copy()
    index["size_bin"] = pd.cut(index["area_ha"], SIZE_BINS, labels=SIZE_LABELS, right=False).astype(str)
    index["days_bin"] = pd.cut(index["age_days"], [0, 30, 60, 150, 225, 400], right=False).astype(str)
    index = index.set_index("tile_id")
    rows = []
    for run in runs:
        data = np.load(run["dir"] / "test_tile_confusion.npz", allow_pickle=False)
        ids, matrices, classes = data["tile_id"], data["matrices"], data["classes"]
        meta = index.reindex(ids)
        for column in ["country", "fire_year", "size_bin", "window", "days_bin", "tile_filled", "land_cover"]:
            if column not in meta:
                continue
            for value in pd.unique(meta[column]):
                chosen = (meta[column] == value).to_numpy()
                if chosen.any():
                    scores = burnt_from_matrix(matrices[chosen].sum(0), classes)
                    rows.append({"method": run["name"], "kind": run["kind"], "split": run["split"], "seed": run["seed"],
                                 "stratum": column, "value": str(value), "tiles": int(chosen.sum()), **scores})
    return pd.DataFrame(rows)


def shift_table(runs, index, spec):
    """How much does a model lose on a year or country it never saw? Both models are scored on the same tiles."""
    matrices = {}
    for run in runs:
        if run["name"] == spec:
            data = np.load(run["dir"] / "test_tile_confusion.npz", allow_pickle=False)
            matrices.setdefault(run["split"], []).append((list(data["tile_id"]), data["matrices"], data["classes"]))
    if "split" not in matrices:
        return pd.DataFrame()
    rows = []
    main_tiles = set(matrices["split"][0][0])

    def score(entries, tiles):
        values = []
        for ids, stack, classes in entries:
            chosen = np.isin(np.asarray(ids), list(tiles))
            values.append(burnt_from_matrix(stack[chosen].sum(0), classes)["f1"])
        return values

    for column, entries in sorted(matrices.items()):
        if column == "split":
            continue
        shared = main_tiles & set(entries[0][0])
        if not shared:
            continue
        with_data, without = score(matrices["split"], shared), score(entries, shared)
        rows.append({"Unseen in training": column.replace("split_loco_", "country ").replace("split_time", "fires from 2025 on"),
                     "Test tiles": len(shared), "Burnt F1, trained on all": pm(with_data), "Burnt F1, trained without": pm(without),
                     "Change": round(100 * (np.mean(without) - np.mean(with_data)), 2)})
    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--results", default="results")
    parser.add_argument("--index", default="data/v2/index.csv")
    parser.add_argument("--shift-spec", default="unet_r34", help="the model used for the time and country tests")
    args = parser.parse_args()

    runs = load_runs(args.results)
    if not runs:
        raise SystemExit(f"No finished runs under {args.results}")
    out = Path(args.results)
    lines = ["# burn2scar baseline results", "",
             f"{len(runs)} finished runs. Values are test-set percentages, as mean ± standard deviation across seeds.",
             "Burnt metrics merge fresh and old burn and leave out cloud, shadow, older_burn and nodata labels.", "",
             "The FLOGA column is for reference only. FLOGA uses pre-fire and post-fire L2A images of Greece at 20 m,",
             "so the two sets of results measure different tasks.", ""]
    main_table = comparison_table(runs, "split")
    if not main_table.empty:
        main_table.to_csv(out / "comparison.csv", index=False)
        lines += ["## Burnt or unburnt, spatial split", "", md_table(main_table), ""]
    classes = per_class_table(runs, "split")
    if not classes.empty:
        classes.to_csv(out / "per_class.csv", index=False)
        lines += ["## All classes, spatial split", "", md_table(classes), ""]
    others = sorted({r["split"] for r in runs if r["split"] != "split"})
    for split in others:
        table = comparison_table(runs, split)
        lines += [f"## Generalisation: {split}", "", md_table(table), ""]

    index = pd.read_csv(args.index, low_memory=False)
    shift = shift_table(runs, index, args.shift_spec)
    if not shift.empty:
        shift.to_csv(out / "shift.csv", index=False)
        lines += [f"## New years and countries, on the same test tiles: {args.shift_spec}", "", md_table(shift), ""]
    strata = stratified(runs, index)
    strata.to_csv(out / "stratified.csv", index=False)
    deep = strata[(strata["split"] == "split") & (strata["kind"] == "deep")]
    if not deep.empty:
        best_name = main_table[main_table["Type"] == "deep"].iloc[0]["Method"] if (main_table["Type"] == "deep").any() else None
        table = deep[deep["method"] == best_name].groupby(["stratum", "value"], sort=True).agg(
            tiles=("tiles", "first"), f1=("f1", "mean"), iou=("iou", "mean"), burnt_px=("burnt_px", "first")).reset_index()
        table["f1"] = (100 * table["f1"]).round(2)
        table["iou"] = (100 * table["iou"]).round(2)
        lines += [f"## Best deep model by subgroup: {best_name}", "", md_table(table), ""]
    text = "\n".join(lines)
    (out / "summary.md").write_text(text)
    print(text)
    print(f"Wrote {out}/summary.md, comparison.csv, per_class.csv, shift.csv and stratified.csv")


if __name__ == "__main__":
    main()
