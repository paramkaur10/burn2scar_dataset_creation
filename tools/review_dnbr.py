#!/usr/bin/env python3
"""dNBR under every reviewed point: an objective second opinion for the review and its adjudication.

  python tools/burn_evidence.py --index data/v4/index.csv --masks-root data/v4 --out validation/dnbr_review \
      --tiles-from review2/points.csv --save-dnbr --workers 24
  python tools/review_dnbr.py --out review2 --dnbr validation/dnbr_review/dnbr --reviewers param silvia --index data/v4/index.csv

With --index it also reads, at the point's location, the dNBR of the fire's fresh-window scene (dnbr_fresh). For points
on old-window tiles that is the evidence that matters: the scar may have faded by THIS DATE, but did the fire burn there?

Reads the dNBR (median of the 3 x 3 pixels around the point) and writes review2/review_dnbr.md: for each stratum and
answer, how many points and their median dNBR and share at or above 0.10; then review2/review_dnbr_points.csv with
every point, its answers and its dNBR, flagging the answers the imagery contradicts (not burned with dNBR >= 0.27,
burned with dNBR < 0.05), for the adjudication.
"""
import argparse
import csv
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio
from rasterio.warp import transform

BURNED = {"burned", "burned_before"}


def latest(path):
    with open(path, newline="") as handle:
        return {row["point_id"]: row["answer"] for row in csv.DictReader(handle)}


def dnbr_at(folder, tile_id, row, col):
    fire, ts = tile_id.split("/")
    path = folder / f"{fire}_{ts}_dnbr.tif"
    if not path.exists():
        return np.nan
    with rasterio.open(path) as src:
        window = src.read(1)[max(row - 1, 0):row + 2, max(col - 1, 0):col + 2].astype(float)
    window[window == -32768] = np.nan
    return float(np.nanmedian(window)) / 1000 if np.isfinite(window).any() else np.nan


def dnbr_at_lonlat(folder, tile_id, lon, lat):
    fire, ts = tile_id.split("/")
    path = folder / f"{fire}_{ts}_dnbr.tif"
    if not path.exists():
        return np.nan
    with rasterio.open(path) as src:
        xs, ys = transform("EPSG:4326", src.crs, [lon], [lat])
        row, col = src.index(xs[0], ys[0])
        if not (0 <= row < src.height and 0 <= col < src.width):
            return np.nan
    return dnbr_at(folder, tile_id, row, col)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", default="review2")
    parser.add_argument("--dnbr", default="validation/dnbr_review/dnbr")
    parser.add_argument("--reviewers", nargs="+", required=True)
    parser.add_argument("--index", default="", help="release index, to find each fire's fresh-window scene")
    args = parser.parse_args()
    out, folder = Path(args.out), Path(args.dnbr)
    points = pd.read_csv(out / "points.csv", low_memory=False)
    points = points[points["repeat_of"] < 0].copy()
    points["dnbr"] = [dnbr_at(folder, t, int(r), int(c)) for t, r, c in zip(points["tile_id"], points["row"], points["col"])]
    points["dnbr_fresh"] = points["dnbr"].where(points["window"] == "fresh")
    if args.index:
        index = pd.read_csv(args.index, usecols=["tile_id", "fire_id", "window"], low_memory=False)
        fresh = index[index["window"] == "fresh"].groupby("fire_id")["tile_id"].first()
        old = points["window"] != "fresh"
        points.loc[old, "dnbr_fresh"] = [dnbr_at_lonlat(folder, fresh[f], lon, lat) if f in fresh.index else np.nan
                                         for f, lon, lat in zip(points.loc[old, "fire_id"], points.loc[old, "lon"], points.loc[old, "lat"])]
    # The evidence for "did it burn here": the fire's fresh scene where there is one, else this scene.
    points["evidence"] = points["dnbr_fresh"].fillna(points["dnbr"])
    lines = ["# dNBR under the reviewed points", "",
             f"{points['dnbr'].notna().sum():,} of {len(points):,} points have a dNBR value (no value: no image, or cloud on a date).", ""]
    flags = []
    for name in args.reviewers:
        path = out / f"answers_{name}.csv"
        if not path.exists():
            continue
        points[name] = points["point_id"].map(latest(path))
        answer = points[name].map(lambda a: "burned" if a in BURNED else a)
        grouped = points.assign(answer=answer).dropna(subset=["answer"]).groupby(["stratum", "answer"])
        table = pd.DataFrame({"points": grouped.size(),
                              "this": grouped["dnbr"].median().round(3),
                              "this_010": grouped["dnbr"].apply(lambda s: round(100 * (s.dropna() >= 0.10).mean(), 1) if s.notna().any() else np.nan),
                              "fresh": grouped["dnbr_fresh"].median().round(3),
                              "fresh_010": grouped["dnbr_fresh"].apply(lambda s: round(100 * (s.dropna() >= 0.10).mean(), 1) if s.notna().any() else np.nan)}).reset_index()
        lines += [f"## {name}", "", "dNBR on THIS DATE's scene, and on the fire's fresh scene at the same place.", "",
                  "| Stratum | Answer | Points | Median dNBR, this scene | >= 0.10 % | Median dNBR, fire's fresh scene | >= 0.10 % |",
                  "|---|---|---|---|---|---|---|"]
        lines += ["| " + " | ".join("" if pd.isna(v) else str(v) for v in r) + " |" for r in table.itertuples(index=False)]
        contradicted = ((answer == "not_burned") & (points["evidence"] >= 0.27)) | ((answer == "burned") & (points["evidence"] < 0.05))
        points[f"{name}_check"] = np.where(contradicted, "look again", "")
        flags.append(int(contradicted.sum()))
        lines += ["", f"Answers the dNBR contradicts (not burned at >= 0.27, burned below 0.05): {int(contradicted.sum())}", ""]
    keep = [c for c in ["point_id", "tile_id", "stratum", "window", "age_days", "dnbr", "dnbr_fresh", *args.reviewers,
                        *[f"{n}_check" for n in args.reviewers]] if c in points]
    points[keep].to_csv(out / "review_dnbr_points.csv", index=False)
    (out / "review_dnbr.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    print(f"Wrote {out}/review_dnbr.md and {out}/review_dnbr_points.csv (it holds the strata: open it only after both reviews).")


if __name__ == "__main__":
    main()
