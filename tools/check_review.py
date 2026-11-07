#!/usr/bin/env python3
"""Explain a finished review: where the reviewer confirmed burn, and whether a burn signal is there.

It reads every point shown for the first time, not only the disagreements, and never changes an
answer, so the review's own numbers stay as they are. Run it once, after the score step.

  python tools/check_review.py --out review_final

Two parts:
  1. Confirmed burned, for fresh_burn and old_burn points, by the size of the tile's fire, days from
     the fire to the scene, season of the fire, and distance from the point to the edge of the burn.
  2. A spectral check. dNBR is the drop in the normalised burn ratio, NBR = (B8A - B12) / (B8A + B12),
     from the pre-fire image to a later scene, at the point itself. The threshold is the 95th
     percentile of dNBR at clear_far points in tiles of the same window, so 5 % of clear pixels pass it.
     Images are top-of-atmosphere, so the usual surface-reflectance severity classes don't apply.

Writes <out>/review_check.md, and two sheets of 12 random burn points answered not burned.
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio
from PIL import Image
from rasterio.windows import Window

BAND = {name: i for i, name in enumerate(["B01", "B02", "B03", "B04", "B05", "B06", "B07", "B08", "B8A", "B09", "B10", "B11", "B12"])}
BURN_VALUES = (2, 3)
SEASONS = {12: "Dec-Feb", 1: "Dec-Feb", 2: "Dec-Feb", 3: "Mar-May", 4: "Mar-May", 5: "Mar-May",
           6: "Jun-Aug", 7: "Jun-Aug", 8: "Jun-Aug", 9: "Sep-Nov", 10: "Sep-Nov", 11: "Sep-Nov"}


def wilson(k, n, z=1.96):
    if n == 0:
        return np.nan, np.nan
    p, d = k / n, 1 + z * z / n
    centre = (p + z * z / (2 * n)) / d
    half = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return 100 * (centre - half), 100 * (centre + half)


def md_table(frame):
    columns = list(frame.columns)
    lines = ["| " + " | ".join(map(str, columns)) + " |", "|" + "---|" * len(columns)]
    lines += ["| " + " | ".join(str(value) for value in row) + " |" for row in frame.itertuples(index=False)]
    return "\n".join(lines)


def confirmed_by(frame, column, label):
    rows = []
    for value, group in frame.groupby(column, observed=True, sort=True):
        judged = group[group["answer"] != "cant_tell"]
        k, n = int((judged["answer"] == "burned").sum()), len(judged)
        low, high = wilson(k, n)
        rows.append({label: value, "Points": n, "Confirmed burned %": f"{100 * k / n:.1f}" if n else "",
                     "95% interval": f"{low:.0f}-{high:.0f}" if n else ""})
    return pd.DataFrame(rows)


def edge_distance(mask, row, col, reach=30):
    """Metres from the point to the nearest pixel that isn't fresh or old burn, looking up to `reach` pixels away."""
    r0, c0 = max(row - reach, 0), max(col - reach, 0)
    outside = np.argwhere(~np.isin(mask[r0:row + reach + 1, c0:col + reach + 1], BURN_VALUES))
    if not len(outside):
        return np.inf
    return 10 * float(np.sqrt(((outside - [row - r0, col - c0]) ** 2).sum(axis=1)).min())


def read_pixel(path, row, col):
    """Blue, B8A and B12 reflectance at one pixel."""
    with rasterio.open(path) as src:
        values = src.read([BAND["B02"] + 1, BAND["B8A"] + 1, BAND["B12"] + 1], window=Window(col, row, 1, 1))
    return values.reshape(-1).astype(float) / 10000


def nbr(values, check_bright=False):
    blue, nir, swir = values
    if nir <= 0 or swir <= 0 or (check_bright and blue > .2):  # 0 is no data. A bright pre-fire pixel is likely cloud or snow.
        return np.nan
    return (nir - swir) / (nir + swir)


def signal_table(frame, column, threshold, label):
    rows = []
    for answer in ("burned", "not_burned", "all"):
        group = frame if answer == "all" else frame[frame["answer"] == answer]
        values = group[column].dropna()
        if not len(values):
            continue
        rows.append({"Answer": answer, "Points": len(values), "Median dNBR": f"{values.median():.3f}",
                     label: f"{100 * (values > threshold).mean():.1f}", "Above 0.1, %": f"{100 * (values > .1).mean():.1f}"})
    return pd.DataFrame(rows)


def sheet(frame, chips, target, seed):
    picks = frame.sample(n=min(12, len(frame)), random_state=seed)
    if not len(picks):
        return 0
    w, h = 540, 380
    canvas = Image.new("RGB", (3 * w, 4 * h))
    for i, point in enumerate(picks["point_id"]):
        canvas.paste(Image.open(chips / f"{point}.jpg").resize((w, h)), ((i % 3) * w, (i // 3) * h))
    canvas.save(target, quality=85)
    return len(picks)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", default="review_final")
    parser.add_argument("--reviewer", default="param")
    parser.add_argument("--index", default="data/v3/index.csv")
    parser.add_argument("--masks-root", default="data/v3")
    parser.add_argument("--images-root", default="data/v1")
    parser.add_argument("--prefire-root", default="data/prefire")
    parser.add_argument("--seed", type=int, default=1)
    args = parser.parse_args()
    out = Path(args.out)

    answers = pd.read_csv(out / f"answers_{args.reviewer}.csv").drop_duplicates("point_id", keep="last")[["point_id", "answer"]]
    points = pd.read_csv(out / "points.csv", low_memory=False)
    index = pd.read_csv(args.index, low_memory=False).set_index("tile_id")
    frame = points[points["repeat_of"] < 0].merge(answers, on="point_id")
    frame = frame.join(index[["area_ha", "age_days", "fire_date", "mask_path"]], on="tile_id")

    # A point's burn should come from the tile's own fire for size, dates and the pre-fire image to apply.
    # fresh_burn in an old-window tile, or old_burn in a fresh-window tile, comes from another fire, so it's left out.
    matched = ((frame["stratum"] == "fresh") & (frame["window"] == "fresh")) | ((frame["stratum"] == "old") & (frame["window"] == "old"))
    burn = frame[matched].copy()
    left_out = int(frame["stratum"].isin(["fresh", "old"]).sum() - len(burn))
    burn["size"] = pd.cut(burn["area_ha"], [0, 30, 100, 500, np.inf], labels=["5-30 ha", "30-100 ha", "100-500 ha", "500-2000 ha"])
    burn["season"] = pd.Categorical(pd.to_datetime(burn["fire_date"], format="mixed").dt.month.map(SEASONS),
                                    categories=["Dec-Feb", "Mar-May", "Jun-Aug", "Sep-Nov"], ordered=True)
    burn["all"] = "All"
    masks = {}
    distances = []
    for row in burn.itertuples():
        if row.mask_path not in masks:
            with rasterio.open(Path(args.masks_root) / row.mask_path) as src:
                masks[row.mask_path] = src.read(1)
        distances.append(edge_distance(masks[row.mask_path], int(row.row), int(row.col)))
    burn["edge"] = pd.cut(pd.Series(distances, index=burn.index), [0, 15, 30, 100, np.inf],
                          labels=["touching the edge", "20-30 m", "30-100 m", "over 100 m"])

    lines = ["# What the review confirmed, and why", "",
             f"Reviewer {args.reviewer}. Points shown for the first time only. Can't tell is left out of the percentages.",
             f"fresh_burn and old_burn points whose burn comes from another fire than the tile's own: {left_out}, left out.", ""]
    for stratum, days_bins, days_labels in (("fresh", [0, 15, 30, 45, np.inf], ["5-15", "15-30", "30-45", "45-60"]),
                                            ("old", [0, 180, 240, np.inf], ["120-180", "180-240", "240-300"])):
        part = burn[burn["stratum"] == stratum].copy()
        part["days"] = pd.cut(part["age_days"], days_bins, labels=days_labels)
        lines += [f"## {stratum}_burn points: confirmed burned", ""]
        for column, label in (("all", "Group"), ("size", "Size of the tile's fire"), ("days", "Days from fire to scene"),
                              ("season", "Season of the fire"), ("edge", "Distance to the burn edge")):
            lines += [md_table(confirmed_by(part, column, label)), ""]

    # Spectral check.
    prefire = {}
    for fire_id in frame["fire_id"].unique():
        found = sorted((Path(args.prefire_root) / "acquisitions" / fire_id).glob("*_prefire_s2.tif"))
        if found:
            prefire[fire_id] = found[0]
    fresh_image = index[index["window"] == "fresh"].reset_index().drop_duplicates("fire_id").set_index("fire_id")["image_path"]
    checked = pd.concat([burn, frame[frame["stratum"].isin(["clear_far", "clear_near"])]])
    checked = checked[checked["fire_id"].isin(prefire)].copy()
    for name in ("dnbr_scene", "dnbr_fresh"):
        checked[name] = np.nan
    for key, row in checked.iterrows():
        r, c = int(row["row"]), int(row["col"])
        before = nbr(read_pixel(prefire[row["fire_id"]], r, c), check_bright=True)
        checked.at[key, "dnbr_scene"] = before - nbr(read_pixel(Path(args.images_root) / row["image_path"], r, c))
        if row["stratum"] == "old" and row["fire_id"] in fresh_image.index:
            checked.at[key, "dnbr_fresh"] = before - nbr(read_pixel(Path(args.images_root) / fresh_image[row["fire_id"]], r, c))
    control = checked[checked["stratum"] == "clear_far"]
    threshold = {window: control.loc[control["window"] == window, "dnbr_scene"].quantile(.95) for window in ("fresh", "old")}
    lines += ["## Spectral check", "",
              f"Points with a pre-fire image: {len(checked):,}. Pre-fire images: {len(prefire):,} fires.",
              f"Threshold, the 95th percentile of dNBR at clear_far points: fresh-window tiles {threshold['fresh']:.3f}, "
              f"old-window tiles {threshold['old']:.3f}.", ""]
    parts = [("fresh_burn points, pre-fire to THIS DATE", "fresh", "dnbr_scene", threshold["fresh"]),
             ("old_burn points, pre-fire to the EARLIER fresh scene", "old", "dnbr_fresh", threshold["fresh"]),
             ("old_burn points, pre-fire to THIS DATE", "old", "dnbr_scene", threshold["old"]),
             ("clear_near points, pre-fire to THIS DATE, fresh-window tiles", "clear_near", "dnbr_scene", threshold["fresh"]),
             ("clear_far points, pre-fire to THIS DATE, fresh-window tiles", "clear_far", "dnbr_scene", threshold["fresh"])]
    for title, stratum, column, cut in parts:
        group = checked[checked["stratum"] == stratum]
        if stratum.startswith("clear"):
            group = group[group["window"] == "fresh"]
        lines += [f"### {title}", "", md_table(signal_table(group, column, cut, "Above threshold, %")), ""]

    chips = out / "chips"
    for stratum in ("fresh", "old"):
        target = out / f"check_{stratum}_not_burned.jpg"
        count = sheet(burn[(burn["stratum"] == stratum) & (burn["answer"] == "not_burned")], chips, target, args.seed)
        lines.append(f"Wrote {target}, {count} random {stratum}_burn points answered not burned.")
    text = "\n".join(lines) + "\n"
    (out / "review_check.md").write_text(text)
    print(text)
    print(f"Wrote {out / 'review_check.md'}")


if __name__ == "__main__":
    main()
