#!/usr/bin/env python3
"""Compare burn2scar labels with an independent source: Copernicus EMS analyst maps.

The reference is wildfires-cems (Arnaudo et al. 2023, CC BY 4.0), which holds rasterised
delineations from Copernicus EMS rapid mapping, 2017 to early 2023. They are made separately from
EFFIS, by other analysts and often from other imagery, which is what makes them a useful check.

  python tools/compare_cems.py --config config.yaml --cems-root data/cems \\
      --index data/v2/index.csv --masks-root data/v2 --out validation

Each area's date and country come from the table kept by the authors of the CEMS dataset
(satelliteData.csv), fetched once from a fixed commit: about 190 KB, no imagery. The date is the
activation date of the area. The time stamp in each sample's JSON is when the file was downloaded,
so it is never used.

Two comparisons, both limited to the countries and dates of burn2scar:

  perimeters  EFFIS perimeters against the CEMS map over each CEMS area. This tests the source
              of the labels, and doesn't depend on whether a burn2scar tile happens to be there.
              It is reported twice. CEMS maps only the fire it was activated for, inside its area
              of interest, so a fire elsewhere in the box isn't necessarily a mistake: one table
              keeps the EFFIS fires that touch the CEMS burn, the other keeps them all.
  tiles       The fresh_burn class of burn2scar v2 masks against the CEMS map, where a tile
              overlaps a CEMS area and was acquired up to 60 days after the CEMS date.

Intervals resample whole CEMS activations, since pixels of one fire aren't independent.
"""
import argparse
import json
import sys
import urllib.request
from collections import Counter
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
import yaml
from rasterio.warp import Resampling, reproject, transform_bounds
from shapely.geometry import box

sys.path.insert(0, str(Path(__file__).resolve().parent))
from burn_labels import load_perimeters, md_table, query_many, to_raster  # noqa: E402

AREA_TABLE_URL = ("https://raw.githubusercontent.com/MatteoM95/CEMS-Wildfire-Dataset/"
                  "d635dea054b2756b9645659e60d1c9b09a611367/csv_files/satelliteData.csv")
COUNTRY_CODES = {"Italy": "IT", "France": "FR", "Spain": "ES", "Greece": "EL"}
# Days before and after the CEMS date that an EFFIS fire can be dated. "main" can be changed with options.
WINDOWS = {"narrow": (15, 7), "main": (60, 15), "wide": (90, 30), "year": (365, 15)}
COUNTS = ["tp", "fp", "fn", "tn"]


def load_area_table(path):
    """Date and country of each CEMS sample, from the table the dataset's authors keep. Fetched once if missing."""
    path = Path(path)
    if not path.exists():
        print(f"Fetching the CEMS area table (about 190 KB, no imagery):\n  {AREA_TABLE_URL}", flush=True)
        path.parent.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(AREA_TABLE_URL, path)
    table = pd.read_csv(path)
    table["sample"] = table["folderPath"].str.rsplit("/", n=1).str[-1]
    table["area"] = table["EMSR"] + "_" + table["AOI"]
    table["date"] = pd.to_datetime(table["activationDate"], format="ISO8601", errors="coerce").dt.normalize()
    table["request_start"] = table["interval_startDate"].astype(str).str[:10]
    table["country"] = table["country"].map(lambda name: COUNTRY_CODES.get(name, name))
    return table.dropna(subset=["date"]).drop_duplicates("sample")


def request_start(meta_path):
    """Start of the Sentinel-2 search in a sample's JSON, used only to check the table against the sample."""
    try:
        meta = json.loads(Path(meta_path).read_text())
        return str(meta["payload"]["input"]["data"][0]["dataFilter"]["timeRange"]["from"])[:10]
    except (OSError, ValueError, KeyError, IndexError, TypeError):
        return None


def load_samples(root, table):
    by_sample = table.set_index("sample")
    by_area = table.drop_duplicates("area").set_index("area")
    samples = []
    for del_path in sorted(Path(root).rglob("*_DEL.tif")):
        name = del_path.name.replace("_DEL.tif", "")
        activation, aoi = name.split("_")[:2]
        row, found = None, "none"
        if name in by_sample.index:
            row, found = by_sample.loc[name], "sample"
        elif f"{activation}_{aoi}" in by_area.index:
            row, found = by_area.loc[f"{activation}_{aoi}"], "area"
        with rasterio.open(del_path) as src:
            if src.crs is None:
                continue
            footprint = box(*transform_bounds(src.crs, "EPSG:4326", *src.bounds, densify_pts=21))
        from_json = request_start(del_path.with_name(f"{name}_S2L2A.json"))
        samples.append({"sample": name, "activation": activation, "del_path": del_path, "found": found,
                        "date": row["date"] if row is not None else pd.NaT,
                        "country": row["country"] if row is not None else None,
                        "window_agrees": (from_json == row["request_start"]) if row is not None and from_json else None,
                        "geometry": footprint})
    return gpd.GeoDataFrame(samples, geometry="geometry", crs=4326) if samples else None


def read_reference(sample):
    """Returns burned, valid, transform, crs. In these rasters 1 is burned, 0 is not, 255 is outside the image."""
    with rasterio.open(sample["del_path"]) as src:
        data = src.read(1)
        transform, crs = src.transform, src.crs
    return data == 1, data != 255, transform, crs


def confusion(burned_map, burned_reference, valid):
    return {"tp": int((burned_map & burned_reference & valid).sum()), "fp": int((burned_map & ~burned_reference & valid).sum()),
            "fn": int((~burned_map & burned_reference & valid).sum()), "tn": int((~burned_map & ~burned_reference & valid).sum())}


def scores(tp, fp, fn, tn):
    return {"precision": tp / (tp + fp) if tp + fp else np.nan, "recall": tp / (tp + fn) if tp + fn else np.nan,
            "f1": 2 * tp / (2 * tp + fp + fn) if tp + fp + fn else np.nan, "iou": tp / (tp + fp + fn) if tp + fp + fn else np.nan,
            "accuracy": (tp + tn) / max(tp + fp + fn + tn, 1)}


def bootstrap_by_activation(frame, repeats, seed):
    """95% intervals, resampling whole activations."""
    rng = np.random.default_rng(seed)
    totals = frame.groupby("activation")[["tp", "fp", "fn", "tn"]].sum().to_numpy()
    if len(totals) < 2:
        return {}
    draws = [scores(*totals[rng.integers(len(totals), size=len(totals))].sum(0)) for _ in range(repeats)]
    table = pd.DataFrame(draws)
    return {name: (table[name].quantile(.025), table[name].quantile(.975)) for name in table}


def report(frame, title, repeats):
    rows = []
    groups = [("All", frame)]
    for column in ("country", "year"):
        if column in frame:
            groups += [(str(value), part) for value, part in frame.groupby(column)]
    for label, part in groups:
        total = part[["tp", "fp", "fn", "tn"]].sum()
        result = scores(*total)
        interval = bootstrap_by_activation(part, repeats, 0)
        row = {"Group": label, "Activations": part["activation"].nunique(), "Areas": len(part),
               "Burned ha, reference": round(int(total["tp"] + total["fn"]) * .01, 1)}
        for name in ("precision", "recall", "f1", "iou"):
            text = f"{100 * result[name]:.1f}"
            if name in interval:
                text += f" ({100 * interval[name][0]:.1f}-{100 * interval[name][1]:.1f})"
            row[name.capitalize() if name != "iou" else "IoU"] = text
        rows.append(row)
    return [f"## {title}", "", md_table(pd.DataFrame(rows)), ""]


def touching(geoms, burned, transform):
    """The perimeters that overlap the CEMS burn."""
    return [geom for geom in geoms if to_raster([geom], burned.shape, transform)[burned].any()]


def compare_perimeters(samples, perimeters, cfg, windows):
    start, end = pd.Timestamp(cfg["start_date"]), pd.Timestamp(cfg["end_date"])
    rows, skipped = [], Counter()
    sindex = perimeters.sindex
    for sample in samples.to_dict("records"):
        if sample["found"] == "none":
            skipped["not in the CEMS area table"] += 1
            continue
        if sample["country"] not in cfg["countries"]:
            skipped["outside the dataset's countries"] += 1
            continue
        date = sample["date"]
        if not start <= date <= end:
            skipped["outside the dates of the EFFIS export"] += 1
            continue
        burned, valid, transform, crs = read_reference(sample)
        burned_valid = burned & valid
        if not burned_valid.any():
            skipped["no burned pixels to compare"] += 1
            continue
        nearby = perimeters.iloc[sindex.query(sample["geometry"], predicate="intersects")]
        projected = nearby.geometry.to_crs(crs).to_numpy()
        days = ((nearby["fire_date"].dt.normalize() - date).dt.days).to_numpy()
        row = {"sample": sample["sample"], "activation": sample["activation"], "country": sample["country"],
               "year": date.year, "date": date.date().isoformat()}
        for name, (before, after) in windows.items():
            within = projected[(days >= -before) & (days <= after)]
            row.update({f"{name}_{k}": v for k, v in confusion(to_raster(within, burned.shape, transform), burned, valid).items()})
            if name == "main":
                touch = touching(within, burned_valid, transform)
                row.update(effis_fires=len(within), touching_fires=len(touch),
                           **confusion(to_raster(touch, burned.shape, transform), burned, valid))
        rows.append(row)
    return pd.DataFrame(rows), skipped


def window_table(frame, windows):
    rows = []
    for name, (before, after) in windows.items():
        total = frame[[f"{name}_{k}" for k in COUNTS]].sum().to_numpy()
        result = scores(*total)
        rows.append({"EFFIS fires dated": f"{before} days before to {after} after", "Precision": f"{100 * result['precision']:.1f}",
                     "Recall": f"{100 * result['recall']:.1f}", "F1": f"{100 * result['f1']:.1f}", "IoU": f"{100 * result['iou']:.1f}"})
    return pd.DataFrame(rows)


def compare_tiles(samples, index, masks_root, days_before, days_after):
    rows = []
    if index.empty:
        return pd.DataFrame()
    usable = samples[samples["date"].notna()]
    if usable.empty:
        return pd.DataFrame()
    sindex = usable.sindex
    footprints = gpd.GeoSeries([box(r.lon_min, r.lat_min, r.lon_max, r.lat_max) for r in index.itertuples()], crs=4326)
    left, right = query_many(sindex, footprints.values)
    for tile_pos, sample_pos in zip(left, right):
        tile, sample = index.iloc[tile_pos], usable.iloc[sample_pos]
        gap = (pd.Timestamp(tile["scene_time"]) - sample["date"]).total_seconds() / 86400
        if not -days_before <= gap <= days_after:
            continue
        with rasterio.open(Path(masks_root) / tile["mask_path"]) as src:
            mask, transform, crs = src.read(1), src.transform, src.crs
        burned, valid, ref_transform, ref_crs = read_reference(sample)
        # 2 is burned, 1 is not burned, 0 is outside the CEMS area or not valid there.
        source = np.where(valid, burned.astype(np.uint8) + 1, 0).astype(np.uint8)
        target = np.zeros(mask.shape, np.uint8)
        reproject(source, target, src_transform=ref_transform, src_crs=ref_crs, dst_transform=transform, dst_crs=crs,
                  src_nodata=0, dst_nodata=0, resampling=Resampling.nearest)
        # Cloud, shadow and nodata can't be judged. Fresh burn is the class this fire must appear as,
        # and leaving out older scars keeps fires CEMS never mapped from counting against the labels.
        keep = (target > 0) & np.isin(mask, [0, 2, 3, 6, 7])
        if not keep.any() or not ((target == 2) & keep).any():
            continue
        rows.append({"tile_id": tile["tile_id"], "sample": sample["sample"], "activation": sample["activation"],
                     "country": tile["country"], "year": sample["date"].year, "window": tile["window"], "days_after_cems_date": round(gap, 1),
                     **confusion(mask == 2, target == 2, keep)})
    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--cems-root", default="data/cems")
    parser.add_argument("--index", default="data/v2/index.csv")
    parser.add_argument("--masks-root", default="data/v2")
    parser.add_argument("--out", default="validation")
    parser.add_argument("--area-table", default="", help="the authors' satelliteData.csv. Default: inside --cems-root, fetched if missing")
    parser.add_argument("--days-before", type=int, default=WINDOWS["main"][0], help="EFFIS fires dated up to this long before the CEMS date count as the same event")
    parser.add_argument("--days-after", type=int, default=WINDOWS["main"][1], help="and up to this long after it")
    parser.add_argument("--tile-days-before", type=int, default=0, help="above 0 risks tiles from before the fire")
    parser.add_argument("--tile-days-after", type=int, default=60)
    parser.add_argument("--bootstrap", type=int, default=1000)
    args = parser.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text())
    table = load_area_table(args.area_table or Path(args.cems_root) / "satelliteData.csv")
    samples = load_samples(args.cems_root, table)
    if samples is None:
        raise SystemExit(f"No *_DEL.tif files with a CRS under {args.cems_root}")
    found = Counter(samples["found"])
    checked = samples["window_agrees"].dropna()
    agree = int((checked == True).sum())  # noqa: E712
    print(f"{len(samples):,} CEMS areas. In the table by sample: {found['sample']:,}, by area: {found['area']:,}, not found: {found['none']:,}. "
          f"Search window agrees with the sample's JSON: {agree:,} of {len(checked):,}", flush=True)

    perimeters = load_perimeters(cfg)
    windows = {**WINDOWS, "main": (args.days_before, args.days_after)}
    frame, skipped = compare_perimeters(samples, perimeters, cfg, windows)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    lines = ["# burn2scar labels against Copernicus EMS maps", "",
             f"- CEMS areas found: {len(samples):,}",
             f"- In the authors' table by sample {found['sample']:,}, by area {found['area']:,}, not found {found['none']:,}",
             f"- Search window in the sample's JSON agrees with the table: {agree:,} of {len(checked):,}",
             f"- Compared: {len(frame):,}"]
    lines += [f"- Left out, {reason}: {count:,}" for reason, count in skipped.items()]
    before, after = windows["main"]
    if not frame.empty:
        lost = frame[frame["touching_fires"] == 0]
        burned_ha = (frame["tp"] + frame["fn"]).sum() * .01
        lines += [f"- Areas where no EFFIS fire touches the CEMS burn: {len(lost):,} of {len(frame):,}, "
                  f"holding {(lost['tp'] + lost['fn']).sum() * .01:,.0f} of {burned_ha:,.0f} burned ha"]
    lines += ["", f"An EFFIS fire counts as the same event if it is dated {before} days before to {after} days after the CEMS date.",
              "Precision is the share of EFFIS or burn2scar burned pixels that CEMS also maps as burned. Recall is the",
              "share of CEMS burned pixels that EFFIS or burn2scar also has. Brackets show a 95% interval.", ""]
    if not frame.empty:
        frame.to_csv(out / "cems_perimeters.csv", index=False)
        lines += report(frame, "EFFIS fires that touch the CEMS burn", args.bootstrap)
        every = frame.drop(columns=COUNTS).rename(columns=lambda c: c.replace("main_", ""))
        lines += report(every, "Every EFFIS fire in the box and date window", args.bootstrap)
        lines += ["CEMS maps only the fire it was activated for, inside its area of interest, so precision here is a lower bound.", "",
                  "## Does the date window matter? Every EFFIS fire in the box", "", md_table(window_table(frame, windows)), ""]

    index = pd.read_csv(args.index, low_memory=False)
    index = index[index["country"].notna()].reset_index(drop=True)
    tiles = compare_tiles(samples, index, args.masks_root, args.tile_days_before, args.tile_days_after)
    if not tiles.empty:
        tiles.to_csv(out / "cems_tiles.csv", index=False)
        lines += report(tiles, f"burn2scar v2 fresh_burn where a tile overlaps a CEMS area ({len(tiles):,} tile and area pairs)", args.bootstrap)
    else:
        lines += ["No burn2scar tile overlaps a CEMS area with burned pixels within the date limits.", ""]
    text = "\n".join(lines)
    (out / "cems_summary.md").write_text(text)
    print(text)
    print(f"Wrote {out}/cems_summary.md")


if __name__ == "__main__":
    main()
