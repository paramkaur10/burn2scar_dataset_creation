#!/usr/bin/env python3
"""Scores the paper needs that come from the saved per-tile confusion matrices, so nothing is retrained.

  python tools/extra_scores.py --results results_v4 --index data/v4/index.csv --out results_v4/extra_scores.md

1. Check: burnt F1 recomputed against metrics.json.
2. Confirmed fires: the test scores on fires whose dNBR flag is burn (needs fire_dnbr_flag in the index).
3. False alarms: fresh_burn predicted where the label is clear, old_burn or water, on old-window and fresh-window tiles.
4. Hold-outs on fires above 27 ha, against all fires.
5. Age: fresh and old recall, balanced accuracy, plain accuracy, the all-fresh baseline and the fresh:old pixel ratio,
   on all test tiles and on mixed tiles (at least --min-share of both ages).
6. Fresh F1 by fire size.
Every number is the mean and standard deviation over seeds.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from summarize_results import burnt_from_matrix, fresh_from_matrix  # noqa: E402

CLEAR, FRESH, OLD, WATER = 0, 1, 2, 5
SIZE_BINS, SIZE_LABELS = [0, 10, 30, 100, 500, np.inf], ["5-10 ha", "10-30 ha", "30-100 ha", "100-500 ha", "500-2000 ha"]


def md(frame):
    head = "| " + " | ".join(map(str, frame.columns)) + " |"
    rule = "|" + "---|" * len(frame.columns)
    body = ["| " + " | ".join("" if pd.isna(v) else str(v) for v in row) + " |" for row in frame.itertuples(index=False)]
    return "\n".join([head, rule, *body])


def runs(results):
    for path in sorted(Path(results).glob("*__*__multiclass__seed*/test_tile_confusion.npz")):
        model, split, _, seed = path.parent.name.split("__")
        data = np.load(path, allow_pickle=True)
        yield model, split, int(seed.replace("seed", "")), data["tile_id"].astype(str), data["matrices"], list(data["classes"]), path.parent


def mean_sd(values):
    values = [v for v in values if not pd.isna(v)]
    if not values:
        return ""
    return f"{100 * np.mean(values):.2f} ± {100 * np.std(values):.2f}" if len(values) > 1 else f"{100 * values[0]:.2f}"


def age_scores(matrices):
    """Among pixels labelled and predicted as burn (fresh or old)."""
    pairs = matrices[:, [FRESH, OLD]][:, :, [FRESH, OLD]].sum(0)
    fresh_n, old_n = pairs[0].sum(), pairs[1].sum()
    fresh_recall = pairs[0, 0] / fresh_n if fresh_n else np.nan
    old_recall = pairs[1, 1] / old_n if old_n else np.nan
    total = fresh_n + old_n
    return {"fresh_recall": fresh_recall, "old_recall": old_recall, "balanced": np.nanmean([fresh_recall, old_recall]),
            "accuracy": np.trace(pairs) / total if total else np.nan,
            "all_fresh": fresh_n / total if total else np.nan,
            "ratio": fresh_n / old_n if old_n else np.nan}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--results", default="results_v4")
    parser.add_argument("--index", default="data/v4/index.csv")
    parser.add_argument("--min-share", type=float, default=0.01)
    parser.add_argument("--min-ha", type=float, default=27)
    parser.add_argument("--out", default="results_v4/extra_scores.md")
    args = parser.parse_args()
    index = pd.read_csv(args.index, low_memory=False).set_index("tile_id")
    collected = {k: [] for k in ("check", "confirmed", "alarms", "holdout", "age", "size")}

    for model, split, seed, tiles, matrices, classes, folder in runs(args.results):
        meta = index.loc[tiles]
        key = {"Model": model, "Split": split, "Seed": seed}
        stored = json.loads((folder / "metrics.json").read_text())["test"]["burnt_f1"] if (folder / "metrics.json").exists() else np.nan
        collected["check"].append({**key, "recomputed": burnt_from_matrix(matrices.sum(0), classes)["burnt_f1"], "metrics.json": stored})

        if split == "split":
            if "fire_dnbr_flag" in meta:
                flag = meta["fire_dnbr_flag"].fillna("none").to_numpy()
                for name, chosen in (("all test fires", np.ones(len(tiles), bool)), ("dNBR burn", flag == "burn"),
                                     ("dNBR burn or weak", np.isin(flag, ["burn", "weak"])), ("dNBR weak or none", np.isin(flag, ["weak", "none"]))):
                    if chosen.any():
                        part = matrices[chosen].sum(0)
                        collected["confirmed"].append({**key, "Subset": name, "Tiles": int(chosen.sum()),
                                                       "fresh": fresh_from_matrix(part, classes)["burnt_f1"],
                                                       "burnt": burnt_from_matrix(part, classes)["burnt_f1"]})
            for window in ("old", "fresh"):
                chosen = (meta["window"] == window).to_numpy()
                if not chosen.any():
                    continue
                m = matrices[chosen]
                negatives = m[:, [CLEAR, OLD, WATER], :].sum((1, 2))
                false_fresh = m[:, [CLEAR, OLD, WATER], FRESH].sum(1)
                no_fresh_label = m[:, FRESH, :].sum(1) == 0
                collected["alarms"].append({**key, "Window": window, "Tiles": int(chosen.sum()),
                                            "rate": false_fresh.sum() / max(negatives.sum(), 1),
                                            "tiles_1pc": float(((false_fresh / np.maximum(negatives, 1)) >= 0.01)[no_fresh_label].mean())
                                            if no_fresh_label.any() else np.nan,
                                            "Tiles without fresh label": int(no_fresh_label.sum())})
            mixed = ((meta["frac_fresh"] >= args.min_share) & (meta["frac_old"] >= args.min_share)).to_numpy()
            for name, chosen in (("all test tiles", np.ones(len(tiles), bool)), ("mixed tiles", mixed)):
                if chosen.any():
                    collected["age"].append({**key, "Tiles": name, "n": int(chosen.sum()), **age_scores(matrices[chosen])})
            size = pd.cut(meta["area_ha"], SIZE_BINS, right=False, labels=SIZE_LABELS).astype(str).to_numpy()
            for label in SIZE_LABELS:
                chosen = size == label
                if chosen.any():
                    collected["size"].append({**key, "Size": label, "Tiles": int(chosen.sum()),
                                              "fresh": fresh_from_matrix(matrices[chosen].sum(0), classes)["burnt_f1"]})
        big = (meta["area_ha"] > args.min_ha).to_numpy()
        collected["holdout"].append({**key, "all": fresh_from_matrix(matrices.sum(0), classes)["burnt_f1"],
                                     "big": fresh_from_matrix(matrices[big].sum(0), classes)["burnt_f1"] if big.any() else np.nan,
                                     "Tiles": len(tiles), "Big tiles": int(big.sum())})

    lines = ["# Extra scores from the saved confusion matrices", "", "Mean ± standard deviation over seeds, in percent.", ""]
    check = pd.DataFrame(collected["check"])
    if check.empty:
        raise SystemExit(f"No multiclass runs with test_tile_confusion.npz under {args.results}")
    worst = (check["recomputed"] - check["metrics.json"]).abs().max()
    lines += [f"Check: burnt F1 recomputed from the matrices differs from metrics.json by at most {100 * worst:.4f} points "
              f"over {len(check)} runs.", ""]

    def summary(name, groups, values, rename):
        frame = pd.DataFrame(collected[name])
        if frame.empty:
            return []
        rows = []
        for group, part in frame.groupby(groups, sort=False):
            row = dict(zip(groups, group if isinstance(group, tuple) else (group,)))
            row["Seeds"] = part["Seed"].nunique()
            for value in values:
                if value in ("Tiles", "n", "Tiles without fresh label", "Big tiles"):
                    row[rename.get(value, value)] = int(part[value].iloc[0])
                elif value == "ratio":
                    row[rename[value]] = f"{part[value].mean():.2f}"
                else:
                    row[rename.get(value, value)] = mean_sd(part[value].tolist())
            rows.append(row)
        return [md(pd.DataFrame(rows)), ""]

    lines += ["## Test scores on fires the burn evidence confirms", "",
              "Spatial split. fresh: fresh_burn against clear, old_burn and water. burnt: fresh and old burn against clear and water.", ""]
    lines += summary("confirmed", ["Model", "Subset"], ["Tiles", "fresh", "burnt"], {"fresh": "Fresh F1", "burnt": "Burnt F1"})
    lines += ["## False alarms: fresh_burn predicted on clear, old_burn or water", "",
              "rate: share of those pixels predicted fresh_burn. tiles_1pc: share of tiles with no fresh_burn label where at least "
              "1 % of them are predicted fresh_burn.", ""]
    lines += summary("alarms", ["Model", "Window"], ["Tiles", "Tiles without fresh label", "rate", "tiles_1pc"],
                     {"rate": "Pixel rate %", "tiles_1pc": "Tiles with a false alarm %"})
    lines += [f"## Fresh F1 on all fires and on fires above {args.min_ha:g} ha", ""]
    lines += summary("holdout", ["Model", "Split"], ["Tiles", "Big tiles", "all", "big"],
                     {"all": "All fires", "big": f"Above {args.min_ha:g} ha"})
    lines += ["## Age: fresh against old among pixels labelled and predicted as burn", "",
              "balanced: mean of fresh and old recall. all_fresh: the accuracy of calling every burn pixel fresh. "
              "ratio: fresh to old label pixels.", ""]
    lines += summary("age", ["Model", "Tiles"], ["n", "fresh_recall", "old_recall", "balanced", "accuracy", "all_fresh", "ratio"],
                     {"n": "Tiles n", "fresh_recall": "Fresh recall", "old_recall": "Old recall", "balanced": "Balanced accuracy",
                      "accuracy": "Accuracy", "all_fresh": "All-fresh baseline", "ratio": "Fresh:old pixels"})
    lines += ["## Fresh F1 by fire size", ""]
    lines += summary("size", ["Model", "Size"], ["Tiles", "fresh"], {"fresh": "Fresh F1"})
    text = "\n".join(lines)
    Path(args.out).write_text(text)
    print(text)


if __name__ == "__main__":
    main()
