#!/usr/bin/env python3
"""Turn every finished baseline run into the tables for the paper.

  python tools/summarize_results.py --results results --index data/v3/index.csv

The main task is fresh burn against everything else: fresh burn is the positive class, and
clear, old burn and water are negative. That is what an alert for new fires needs, and it is
how FLOGA counts burnt (the event's own burn, with older scars as unburnt). The deep models
predict fresh and old burn as separate classes, so their fresh scores come from the same runs.
The classic baselines are run once for each target (classic_baselines.py --target fresh).
Burnt metrics with fresh and old burn together are kept for reference. Cloud, shadow,
older_burn and nodata labels are left out of both.

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
FRESH_HEADERS = {"burnt_precision": "Fresh P", "burnt_recall": "Fresh R", "burnt_f1": "Fresh F1", "burnt_iou": "Fresh IoU",
                 "unburnt_f1": "Not fresh F1", "unburnt_iou": "Not fresh IoU", "mean_f1": "Mean F1", "mean_iou": "Mean IoU"}
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
SEASONS = {12: "Dec-Feb", 1: "Dec-Feb", 2: "Dec-Feb", 3: "Mar-May", 4: "Mar-May", 5: "Mar-May",
           6: "Jun-Aug", 7: "Jun-Aug", 8: "Jun-Aug", 9: "Sep-Nov", 10: "Sep-Nov", 11: "Sep-Nov"}
KIND_ORDER = {"index": 0, "pixel": 1, "deep": 2}
MERGED_TASKS = {"multiclass", "burn"}


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


def two_class(matrix, positive, negative):
    """Metrics for one class against others. Rows are labels, columns predictions; other label rows are left out."""
    tp = matrix[np.ix_(positive, positive)].sum()
    fn = matrix[positive].sum() - tp
    fp = matrix[np.ix_(negative, positive)].sum()
    tn = matrix[negative].sum() - fp

    def scores(hit, false_positive, false_negative):
        precision = hit / (hit + false_positive) if hit + false_positive else 0.0
        recall = hit / (hit + false_negative) if hit + false_negative else 0.0
        return precision, recall, 2 * hit / max(2 * hit + false_positive + false_negative, 1), hit / max(hit + false_positive + false_negative, 1)
    yes, no = scores(tp, fp, fn), scores(tn, fn, fp)
    return {"burnt_precision": yes[0], "burnt_recall": yes[1], "burnt_f1": yes[2], "burnt_iou": yes[3],
            "unburnt_f1": no[2], "unburnt_iou": no[3], "mean_f1": (yes[2] + no[2]) / 2, "mean_iou": (yes[3] + no[3]) / 2,
            "positive_px": int(matrix[positive].sum())}


def fresh_from_matrix(matrix, classes):
    names = list(classes)
    return two_class(matrix, [names.index("fresh_burn")], [names.index(n) for n in ("clear", "old_burn", "water")])


def burnt_from_matrix(matrix, classes):
    names = list(classes)
    if "fresh_burn" in names:
        return two_class(matrix, [names.index("fresh_burn"), names.index("old_burn")], [names.index("clear"), names.index("water")])
    return two_class(matrix, [names.index("burnt")], [names.index("unburnt")])


def fresh_runs(runs):
    """Every run scored on fresh burn against the rest: deep runs from their saved test matrices,
    classic runs only where fresh burn was their target."""
    out = []
    for run in runs:
        if run["task"] == "multiclass":
            data = np.load(run["dir"] / "test_tile_confusion.npz", allow_pickle=False)
            out.append({**run, "test": fresh_from_matrix(data["matrices"].sum(0), data["classes"])})
        elif run["task"] == "fresh":
            out.append(run)
    return out


def comparison_table(runs, split, headers=HEADERS):
    rows = []
    groups = {}
    for run in runs:
        if run["split"] == split:
            groups.setdefault((run["kind"], run["name"]), []).append(run)
    for (kind, name), members in sorted(groups.items(), key=lambda kv: (KIND_ORDER.get(kv[0][0], 9), -np.mean([m["test"]["burnt_f1"] for m in kv[1]]))):
        family = FLOGA_FAMILY.get(name, "")
        row = {"Method": name, "Type": kind, "Seeds": len(members)}
        row.update({headers[m]: pm([r["test"][m] for r in members]) for m in METRICS})
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


def generalisation_table(merged, fresh, splits):
    """One row per design and method: fresh burn against the rest, and fresh and old burn together."""
    rows = []
    for split in splits:
        for name in sorted({r["name"] for r in fresh if r["split"] == split}):
            f = [r["test"] for r in fresh if r["split"] == split and r["name"] == name]
            m = [r["test"] for r in merged if r["split"] == split and r["name"] == name]
            rows.append({"Design": split, "Method": name, "Seeds": len(f),
                         "Fresh P": pm([x["burnt_precision"] for x in f]), "Fresh R": pm([x["burnt_recall"] for x in f]),
                         "Fresh F1": pm([x["burnt_f1"] for x in f]), "Fresh IoU": pm([x["burnt_iou"] for x in f]),
                         "Burnt F1 (fresh + old)": pm([x["burnt_f1"] for x in m]) if m else "",
                         "Burnt IoU (fresh + old)": pm([x["burnt_iou"] for x in m]) if m else ""})
    return pd.DataFrame(rows)


def stratified(runs, index):
    """Scores by country, year, fire size, window, season and other strata, per method and seed."""
    index = index.copy()
    index["size_bin"] = pd.cut(index["area_ha"], SIZE_BINS, labels=SIZE_LABELS, right=False).astype(str)
    index["days_bin"] = pd.cut(index["age_days"], [0, 30, 60, 150, 225, 400], right=False).astype(str)
    for column, source in (("fire_season", "fire_date"), ("scene_season", "scene_time")):
        if source in index:
            index[column] = pd.to_datetime(index[source], format="mixed").dt.month.map(SEASONS)
    index = index.set_index("tile_id")
    rows = []
    for run in runs:
        data = np.load(run["dir"] / "test_tile_confusion.npz", allow_pickle=False)
        ids, matrices, classes = data["tile_id"], data["matrices"], data["classes"]
        names = list(classes)
        meta = index.reindex(ids)
        for column in ["country", "fire_year", "size_bin", "window", "days_bin", "fire_season", "scene_season",
                       "tile_filled", "land_cover"]:
            if column not in meta:
                continue
            for value in pd.unique(meta[column]):
                chosen = (meta[column] == value).to_numpy()
                if not chosen.any():
                    continue
                matrix = matrices[chosen].sum(0)
                scores = {"f1": np.nan, "iou": np.nan, "burnt_px": np.nan, "fresh_f1": np.nan, "fresh_iou": np.nan,
                          "fresh_px": np.nan, "old_iou": np.nan}
                if run["task"] in MERGED_TASKS:
                    merged = burnt_from_matrix(matrix, classes)
                    scores.update(f1=merged["burnt_f1"], iou=merged["burnt_iou"], burnt_px=merged["positive_px"])
                if run["task"] == "multiclass" or run["task"] == "fresh":
                    fresh = fresh_from_matrix(matrix, classes) if run["task"] == "multiclass" else two_class(
                        matrix, [names.index("burnt")], [names.index("unburnt")])
                    scores.update(fresh_f1=fresh["burnt_f1"], fresh_iou=fresh["burnt_iou"], fresh_px=fresh["positive_px"])
                if "old_burn" in names:
                    c = names.index("old_burn")
                    union = matrix[c].sum() + matrix[:, c].sum() - matrix[c, c]
                    scores["old_iou"] = matrix[c, c] / union if union else np.nan
                rows.append({"method": run["name"], "kind": run["kind"], "task": run["task"], "split": run["split"],
                             "seed": run["seed"], "stratum": column, "value": str(value), "tiles": int(chosen.sum()), **scores})
    return pd.DataFrame(rows)


def shift_table(runs, index, spec):
    """How much does a model lose on a year or country it never saw? Both models are scored on the same tiles."""
    matrices = {}
    for run in runs:
        if run["name"] == spec and run["task"] == "multiclass":
            data = np.load(run["dir"] / "test_tile_confusion.npz", allow_pickle=False)
            matrices.setdefault(run["split"], []).append((list(data["tile_id"]), data["matrices"], data["classes"]))
    if "split" not in matrices:
        return pd.DataFrame()
    rows = []
    main_tiles = set(matrices["split"][0][0])

    def score(entries, tiles, scorer):
        values = []
        for ids, stack, classes in entries:
            chosen = np.isin(np.asarray(ids), list(tiles))
            values.append(scorer(stack[chosen].sum(0), classes)["burnt_f1"])
        return values

    for column, entries in sorted(matrices.items()):
        if column == "split":
            continue
        shared = main_tiles & set(entries[0][0])
        if not shared:
            continue
        row = {"Unseen in training": column.replace("split_loco_", "country ").replace("split_time", "fires from 2025 on"),
               "Test tiles": len(shared)}
        for label, scorer in (("Fresh F1", fresh_from_matrix), ("Burnt F1 (fresh + old)", burnt_from_matrix)):
            with_data, without = score(matrices["split"], shared, scorer), score(entries, shared, scorer)
            row[f"{label}, trained on all"] = pm(with_data)
            row[f"{label}, trained without"] = pm(without)
            row[f"{label}, change"] = round(100 * (np.mean(without) - np.mean(with_data)), 2)
        rows.append(row)
    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--results", default="results")
    parser.add_argument("--index", default="data/v3/index.csv")
    parser.add_argument("--shift-spec", default="unet_r34", help="the model used for the time and country tests")
    args = parser.parse_args()

    runs = load_runs(args.results)
    if not runs:
        raise SystemExit(f"No finished runs under {args.results}")
    out = Path(args.results)
    merged = [r for r in runs if r["task"] in MERGED_TASKS]
    fresh = fresh_runs(runs)
    lines = ["# burn2scar baseline results", "",
             f"{len(runs)} finished runs. Values are test-set percentages, as mean ± standard deviation across seeds.", "",
             "Main task: fresh burn against everything else. Fresh burn is positive; clear, old burn and water are negative.",
             "Deep models predict fresh and old burn separately, so their fresh scores come from the same runs. The classic",
             "baselines are run once with fresh burn as their only target. Burnt metrics with fresh and old burn together",
             "follow for reference. Cloud, shadow, older_burn and nodata labels are left out of both.", "",
             "The FLOGA column is for reference only. FLOGA uses pre-fire and post-fire L2A images of Greece at 20 m,",
             "so the two sets of results measure different tasks.", ""]
    fresh_table = comparison_table(fresh, "split", FRESH_HEADERS)
    if not fresh_table.empty:
        fresh_table.to_csv(out / "comparison_fresh.csv", index=False)
        lines += ["## Fresh burn against everything else, spatial split", "", md_table(fresh_table), ""]
        missing = sorted({r["name"] for r in merged if r["split"] == "split" and r["task"] == "burn"}
                         - {r["name"] for r in fresh if r["split"] == "split"})
        if missing:
            lines += [f"Not yet run with fresh burn as the target: {', '.join(missing)}.", ""]
    main_table = comparison_table(merged, "split")
    if not main_table.empty:
        main_table.to_csv(out / "comparison.csv", index=False)
        lines += ["## Fresh and old burn together against unburnt, spatial split", "", md_table(main_table), ""]
    classes = per_class_table(runs, "split")
    if not classes.empty:
        classes.to_csv(out / "per_class.csv", index=False)
        lines += ["## All classes, spatial split", "",
                  "Each class against every other class, so these IoUs differ slightly from the two-class tables above.", "",
                  md_table(classes), ""]
    others = sorted({r["split"] for r in runs if r["split"] != "split"})
    if others:
        table = generalisation_table(merged, fresh, others)
        table.to_csv(out / "generalisation.csv", index=False)
        lines += ["## Generalisation designs", "", md_table(table), ""]

    index = pd.read_csv(args.index, low_memory=False)
    shift = shift_table(runs, index, args.shift_spec)
    if not shift.empty:
        shift.to_csv(out / "shift.csv", index=False)
        lines += [f"## New years and countries, on the same test tiles: {args.shift_spec}", "", md_table(shift), ""]
    strata = stratified(runs, index)
    strata.to_csv(out / "stratified.csv", index=False)
    deep = strata[(strata["split"] == "split") & (strata["kind"] == "deep")]
    ranked = fresh_table if not fresh_table.empty else main_table  # Rows come sorted best first within each type.
    deep_rows = ranked[ranked["Type"] == "deep"] if not ranked.empty else ranked
    if not deep.empty and len(deep_rows):
        best_name = deep_rows.iloc[0]["Method"]
        table = deep[deep["method"] == best_name].groupby(["stratum", "value"], sort=True).agg(
            tiles=("tiles", "first"), fresh_f1=("fresh_f1", "mean"), fresh_iou=("fresh_iou", "mean"),
            old_burn_iou=("old_iou", "mean"), burnt_f1=("f1", "mean"), burnt_iou=("iou", "mean"),
            fresh_px=("fresh_px", "first"), burnt_px=("burnt_px", "first")).reset_index()
        for column in ("fresh_f1", "fresh_iou", "old_burn_iou", "burnt_f1", "burnt_iou"):
            table[column] = (100 * table[column]).round(2)
        for column in ("fresh_px", "burnt_px"):
            table[column] = table[column].round().astype("Int64")
        lines += [f"## Best deep model by subgroup: {best_name}", "",
                  "fresh_f1 and fresh_iou: fresh burn against the rest. burnt_f1 and burnt_iou: fresh and old burn together.", "",
                  md_table(table), ""]
    text = "\n".join(lines)
    (out / "summary.md").write_text(text)
    print(text)
    print(f"Wrote {out}/summary.md and the CSV files next to it")


if __name__ == "__main__":
    main()
