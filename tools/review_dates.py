#!/usr/bin/env python3
"""The point review, scored on dates as well as on burned or not, from the answers already given. CPU only.

  python tools/review_dates.py --out review2 --reviewers param silvia --judge judge

The interpreters answered burned (the scar appears after the earliest image shown) or burned_before (the scar is
already on the earliest image). With the date of the EFFIS perimeter that labels the point, read from the age raster
(scene date minus the pixel's age), that answer is a date check:

  earliest image before the perimeter date   burned is consistent with the date; burned_before means the scar is older
                                             than the mapped fire (date_early)
  earliest image on or after it              burned_before is consistent; burned means a scar newer than the mapped
                                             fire (date_late)
  no perimeter (clear points)                burned is a recent burn EFFIS did not map (unmapped_recent); burned_before
                                             an older one (unmapped_older)

Writes <out>/review_dates.md:
  1. Date-aware answers per stratum, for each interpreter and the final labels.
  2. The area-weighted error matrix with the date-aware reference: map class in rows, reference in columns.
  3. User's accuracy of fresh_burn and old_burn counting only date-consistent burn as correct, and producer's accuracy
     of the burn classes for burn that appears after the earliest image, with bootstrap intervals within strata.
  4. A check that each point's map class is the class of the released (v4) mask at its pixel.
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio

sys.path.insert(0, str(Path(__file__).resolve().parent))
from review_points import MAP_CLASS, final_labels, latest_answers  # noqa: E402

DATE_CLASSES = ["date_ok", "date_early", "date_late", "unmapped_recent", "unmapped_older", "not_burned", "water", "cloud",
                "shadow"]
MAP_ROWS = {"fresh_early": "fresh_burn", "fresh_mid": "fresh_burn", "fresh_late": "fresh_burn", "old": "old_burn",
            "older": "older_burn", "clear_near": "clear", "clear_far": "clear", "water": "water", "cloud": "cloud",
            "shadow": "cloud shadow"}
MASK_VALUE = {"fresh_burn": 2, "old_burn": 3, "older_burn": 7, "clear": 0, "water": 6, "cloud": 4, "cloud shadow": 5}


def point_ages(points, index, masks_root):
    """Age raster value (days since the most recent perimeter, 0 = none) and v4 mask value at every point."""
    rows = index.drop_duplicates("tile_id").set_index("tile_id")
    ages, values = {}, {}
    for tile_id, group in points.groupby("tile_id"):
        record = rows.loc[tile_id]
        with rasterio.open(Path(masks_root) / record["age_path"]) as src:
            age = src.read(1)
        with rasterio.open(Path(masks_root) / record["mask_path"]) as src:
            mask = src.read(1)
        for _, point in group.iterrows():
            r, c = int(point["row"]), int(point["col"])
            ages[point["point_id"]], values[point["point_id"]] = int(age[r, c]), int(mask[r, c])
    return ages, values


def date_class(answer, earliest, perimeter_date):
    if answer not in ("burned", "burned_before"):
        return answer
    if perimeter_date is None:
        return "unmapped_recent" if answer == "burned" else "unmapped_older"
    if earliest < perimeter_date:
        return "date_ok" if answer == "burned" else "date_early"
    return "date_ok" if answer == "burned_before" else "date_late"


def classify(first, answers, ages):
    out = {}
    for _, point in first.iterrows():
        answer = answers.get(point["point_id"])
        if answer is None or answer == "cant_tell":
            continue
        age = ages.get(point["point_id"], 0)
        scene = pd.Timestamp(point["scene_time"]).normalize()
        perimeter = scene - pd.Timedelta(days=age) if age > 0 else None
        out[point["point_id"]] = (date_class(answer, pd.Timestamp(point["earliest_date"]), perimeter), answer == "burned")
    return out


def vectors(first, classes):
    """Per stratum, the date classes of its answered points."""
    return {s: [classes[p] for p in g["point_id"] if p in classes] for s, g in first.groupby("stratum")}


def shares(vec):
    """Per stratum: answered points, the share of each date class, and the share of burn appearing after the earliest
    image (the answer burned)."""
    rows = {}
    for s, v in vec.items():
        classes = [c for c, _ in v]
        rows[s] = {"n": len(v), **{c: (classes.count(c) / len(v) if v else np.nan) for c in DATE_CLASSES},
                   "new_burn": (sum(new for _, new in v) / len(v) if v else np.nan)}
    return pd.DataFrame(rows).T


def estimates(vec, weights):
    """Area-weighted matrix, date-aware user's accuracies, and the producer's accuracy of the burn classes for burn that
    appears after the earliest image (date-consistent burn on them, and recent or late burn anywhere)."""
    table = shares({s: v for s, v in vec.items() if v})
    per = table[DATE_CLASSES].astype(float)
    w = pd.Series({s: weights.get(s, 0.0) for s in per.index})
    w = w / w.sum()
    matrix = per.mul(w, axis=0).groupby(lambda s: MAP_ROWS[s]).sum()
    totals = matrix.sum(axis=1)
    user = {name: matrix.loc[name, "date_ok"] / totals[name] for name in ("fresh_burn", "old_burn") if name in matrix.index}
    new = table["new_burn"].astype(float).mul(w).groupby(lambda s: MAP_ROWS[s]).sum()
    mapped = new[[m for m in ("fresh_burn", "old_burn") if m in new.index]].sum()
    return matrix, user, (mapped / new.sum() if new.sum() else np.nan)


def bootstrap(vec, weights, reps, seed):
    rng = np.random.default_rng(seed)
    draws = []
    for _ in range(reps):
        sample = {s: [v[i] for i in rng.integers(0, len(v), len(v))] for s, v in vec.items() if v}
        _, user, producer = estimates(sample, weights)
        draws.append({**{f"user_{k}": x for k, x in user.items()}, "producer": producer})
    frame = pd.DataFrame(draws)
    return frame.quantile(.025), frame.quantile(.975)


def pct(value):
    return "" if pd.isna(value) else f"{100 * value:.1f}"


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", default="review2")
    parser.add_argument("--reviewers", nargs=2, default=["param", "silvia"])
    parser.add_argument("--judge", default="judge")
    parser.add_argument("--index", default="data/v4/index.csv")
    parser.add_argument("--masks-root", default="data/v4", help="released masks and age rasters")
    parser.add_argument("--reps", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=20261120)
    args = parser.parse_args()
    out = Path(args.out)
    points = pd.read_csv(out / "points.csv", low_memory=False)
    shown = pd.read_csv(out / "chips_shown.csv")
    points = points.merge(shown[["point_id", "earliest_date", "dates_shown", "prefire_shown"]], on="point_id", how="left")
    first = points[points["repeat_of"] < 0].copy()
    weights = pd.read_csv(out / "strata_area.csv").set_index("stratum")["pixels"].astype(float).to_dict()
    index = pd.read_csv(args.index, low_memory=False)
    ages, values = point_ages(first, index, args.masks_root)

    sets = {f"Interpreter {i}": latest_answers(out / f"answers_{name}.csv") for i, name in enumerate(args.reviewers, 1)}
    sets["Final"] = final_labels(points, out, args.reviewers, args.judge)
    lines = ["# The point review, scored on dates", "",
             "Reference classes: date_ok, a burn whose timing agrees with the EFFIS perimeter at the point; date_early, a "
             "scar already on an image taken before the mapped fire; date_late, a scar that appears after an image taken "
             "after the mapped fire; unmapped_recent and unmapped_older, burn where no perimeter lies, appearing after the "
             "earliest image or already on it. Percent of the answered points; can't tell left out.", ""]
    summary = []
    for name, answers in sets.items():
        vec = vectors(first, classify(first, answers, ages))
        table = shares(vec)
        lines += [f"## {name}: answers per stratum", "",
                  "| Stratum | n | " + " | ".join(DATE_CLASSES) + " |", "|---|---:|" + "---:|" * len(DATE_CLASSES)]
        for stratum in [s for s in MAP_ROWS if s in table.index]:
            row = table.loc[stratum]
            lines.append(f"| {stratum} | {int(row['n'])} | " + " | ".join(pct(row[c]) for c in DATE_CLASSES) + " |")
        matrix, user, producer = estimates(vec, weights)
        low, high = bootstrap(vec, weights, args.reps, args.seed)
        lines += ["", f"Area-weighted matrix, percent of the data set's area ({name}):", "",
                  "| Map class | " + " | ".join(DATE_CLASSES) + " |", "|---|" + "---:|" * len(DATE_CLASSES)]
        for map_class in [m for m in MASK_VALUE if m in matrix.index]:
            lines.append(f"| {map_class} | " + " | ".join(f"{100 * matrix.loc[map_class, c]:.2f}" for c in DATE_CLASSES) + " |")
        lines.append("")
        for key, value in user.items():
            lines.append(f"- User's accuracy of {key}, date-consistent burn only: {pct(value)} % "
                         f"({pct(low.get(f'user_{key}'))}-{pct(high.get(f'user_{key}'))})")
        lines += [f"- Producer's accuracy of fresh_burn and old_burn together, for burn that appears after the earliest "
                  f"image (the answer burned, wherever it was given): {pct(producer)} % "
                  f"({pct(low['producer'])}-{pct(high['producer'])})", ""]
        summary.append({"Answers": name, **{f"User's, {k}": pct(v) for k, v in user.items()}, "Producer's, new burn": pct(producer)})

    mismatch = first.assign(v4=first["point_id"].map(values))
    mismatch["expected"] = mismatch["stratum"].map(lambda s: MASK_VALUE[MAP_ROWS[s]])
    differ = mismatch[mismatch["v4"] != mismatch["expected"]]
    lines += ["## Map class at the point against the released mask", "",
              f"{len(differ)} of {len(first)} points have a v4 mask value other than their stratum's class"
              + (": " + ", ".join(f"{s} -> {v} ({n})" for (s, v), n in differ.groupby(["stratum", "v4"]).size().items())
                 if len(differ) else "."), ""]
    if "batch" in first:
        lines.append("By batch: " + ", ".join(f"batch {b}: {int((differ['batch'] == b).sum())} of {int((first['batch'] == b).sum())}"
                                              for b in sorted(first["batch"].unique())))
    (out / "review_dates.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines[-6:]))
    print(pd.DataFrame(summary).to_string(index=False))
    print(f"Wrote {out / 'review_dates.md'}")


if __name__ == "__main__":
    main()
