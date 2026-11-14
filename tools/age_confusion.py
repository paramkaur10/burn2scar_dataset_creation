#!/usr/bin/env python3
"""How much of the fresh-against-old accuracy is the image, and how much is the calendar.

Reads the per-tile confusion matrices of finished multiclass runs and the release index, and reports:
  1. the age accuracy on all test tiles (pixels labelled and predicted as burn that get the right age class);
  2. the same on mixed tiles only, those holding at least --min-share of both fresh_burn and old_burn, where the
     scene date cannot settle the age;
  3. the recall of each age class among burn pixels;
  4. a calendar control: the rule "fresh if the scene month says so", with the month's fresh share learned on
     the training tiles, scored on the same test pixels. A model that only beats this rule a little is reading
     the season, not the scar;
  5. how many test tiles hold one age class only.

  python tools/age_confusion.py --results results --index data/v3/index.csv --runs segformer_b2 unet_r34 --out results/age_confusion.md
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

FRESH, OLD = 1, 2   # class order of the multiclass task: clear, fresh_burn, old_burn, cloud, shadow, water


def md(frame):
    head = "| " + " | ".join(map(str, frame.columns)) + " |"
    rule = "|" + "---|" * len(frame.columns)
    body = ["| " + " | ".join("" if pd.isna(v) else str(v) for v in row) + " |" for row in frame.itertuples(index=False)]
    return "\n".join([head, rule, *body])


def age_accuracy(matrices):
    pairs = matrices[:, [FRESH, OLD]][:, :, [FRESH, OLD]].sum(0)
    total = pairs.sum()
    return (np.trace(pairs) / total if total else np.nan,
            pairs[0, 0] / pairs[0].sum() if pairs[0].sum() else np.nan,
            pairs[1, 1] / pairs[1].sum() if pairs[1].sum() else np.nan, int(total))


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--results", default="results")
    parser.add_argument("--index", default="data/v3/index.csv")
    parser.add_argument("--runs", nargs="+", default=["segformer_b2", "unet_r34"])
    parser.add_argument("--split-col", default="split")
    parser.add_argument("--min-share", type=float, default=0.01, help="a mixed tile holds at least this share of each age class")
    parser.add_argument("--out", default="results/age_confusion.md")
    args = parser.parse_args()

    index = pd.read_csv(args.index, low_memory=False).set_index("tile_id")
    index["month"] = pd.to_datetime(index["scene_time"]).dt.month
    train = index[index[args.split_col] == "train"]
    pixels = 256 * 256
    by_month = train.groupby("month").apply(lambda g: (g["frac_fresh"] * pixels).sum() / max(((g["frac_fresh"] + g["frac_old"]) * pixels).sum(), 1),
                                            include_groups=False)
    month_rule = (by_month > 0.5).to_dict()   # month -> predict fresh

    lines = ["# Fresh against old: image or calendar?", "",
             f"Test tiles of the `{args.split_col}` split. Age accuracy is the share of pixels labelled and predicted as burn that get the right age class.",
             f"Mixed tiles hold at least {100 * args.min_share:.0f} % of both fresh_burn and old_burn.", ""]
    lines += ["## Calendar rule learned on the training tiles", "", "| Scene month | Fresh share of burn pixels | Rule says |", "|---|---|---|"]
    lines += [f"| {m} | {100 * by_month[m]:.1f} % | {'fresh' if month_rule[m] else 'old'} |" for m in sorted(month_rule)]
    lines.append("")

    rows = []
    for run_name in args.runs:
        for path in sorted(Path(args.results).glob(f"{run_name}__{args.split_col}__multiclass__seed*/test_tile_confusion.npz")):
            data = np.load(path, allow_pickle=True)
            tiles, matrices = data["tile_id"].astype(str), data["matrices"]
            meta = index.loc[tiles]
            mixed = ((meta["frac_fresh"] >= args.min_share) & (meta["frac_old"] >= args.min_share)).to_numpy()
            one_age = ((meta["frac_fresh"] < 1e-4) | (meta["frac_old"] < 1e-4)).to_numpy()
            acc_all, fresh_recall, old_recall, n_all = age_accuracy(matrices)
            acc_mixed, fresh_mixed, old_mixed, n_mixed = age_accuracy(matrices[mixed]) if mixed.any() else (np.nan,) * 4
            # Calendar rule on the same label pixels: fresh if the month says fresh.
            fresh_px = matrices[:, FRESH, [FRESH, OLD]].sum(1)
            old_px = matrices[:, OLD, [FRESH, OLD]].sum(1)
            says_fresh = meta["month"].map(month_rule).fillna(False).to_numpy()
            correct = np.where(says_fresh, fresh_px, old_px)
            rule_all = correct.sum() / max((fresh_px + old_px).sum(), 1)
            rule_mixed = correct[mixed].sum() / max((fresh_px + old_px)[mixed].sum(), 1) if mixed.any() else np.nan
            # Window oracle: fresh on fresh-window tiles. Not available at inference; shows what the scene date would give.
            is_fresh_window = (meta["window"] == "fresh").to_numpy()
            oracle = np.where(is_fresh_window, fresh_px, old_px).sum() / max((fresh_px + old_px).sum(), 1)
            seed = path.parent.name.split("seed")[-1]
            rows.append({"Run": run_name, "Seed": seed, "Test tiles": len(tiles), "One-age tiles %": f"{100 * one_age.mean():.1f}",
                         "Mixed tiles": int(mixed.sum()),
                         "Age accuracy, all": f"{100 * acc_all:.2f}", "Age accuracy, mixed": f"{100 * acc_mixed:.2f}",
                         "Fresh recall, all / mixed": f"{100 * fresh_recall:.1f} / {100 * fresh_mixed:.1f}",
                         "Old recall, all / mixed": f"{100 * old_recall:.1f} / {100 * old_mixed:.1f}",
                         "Calendar rule, all": f"{100 * rule_all:.2f}", "Calendar rule, mixed": f"{100 * rule_mixed:.2f}",
                         "Window oracle, all": f"{100 * oracle:.2f}", "Burn pixels, all / mixed": f"{n_all:,} / {n_mixed:,}"})
    if not rows:
        raise SystemExit("No finished multiclass runs found for these names.")
    table = pd.DataFrame(rows)
    lines += ["## Models against the calendar", "", md(table), "",
              "Reading: if a model's age accuracy on mixed tiles is far above the calendar rule's, it reads the scar's age from the image. "
              "If the two are close, the scene date is doing the work. The window oracle is the ceiling the scene date alone could reach "
              "if the window were known, which it is not at inference.", ""]
    text = "\n".join(lines)
    Path(args.out).write_text(text)
    print(text)
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
