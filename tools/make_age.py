#!/usr/bin/env python3
"""Days since the fire, per pixel: one uint16 raster beside every mask, so users can cut burn age where they like.

For each pixel, the number of days from the most recent EFFIS perimeter that covers it to the scene date,
counting every perimeter in the export (and in effis_extra) dated on or before the scene. 0 means no perimeter.
Days are rounded up, with a fire earlier on the scene's day counting as 1, so the raster agrees with the mask
classes: 1-90 fresh_burn, 91-365 old_burn, over 365 older_burn.

  python tools/make_age.py --config config.yaml --masks-dir data/v4 --out-dir data/v4 --workers 16 \
      --index data/v4/index.csv --carry-from data/v3/index.csv

Writes <fire>_<ts>_age.tif next to each mask (uint16, nodata 0, deflate, same grid and tags), and
<out-dir>/age_summary.md with the distribution of ages over burn pixels. Add the file to the release index with
--index data/v4/index.csv, which writes age_path and the per-tile median age of burn pixels.

Also adds three flags to the index, from columns it already has: winter_fire (fire dated November to February),
agricultural (EFFIS land cover is agriculture) and modis_mapped (perimeter from MODIS, not Sentinel-2).
--carry-from copies, by tile_id, the columns an older index has and this one lacks (build_index.py does not
rebuild product_id, processing_baseline or water_checked).
"""
import argparse
import math
import os
import sys
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio
import yaml
from rasterio.features import rasterize
from rasterio.warp import transform_bounds
from shapely.geometry import box

sys.path.insert(0, str(Path(__file__).resolve().parent))
from burn_labels import DAY, TILE_RE, TS_FORMAT, load_perimeters  # noqa: E402

NODATA = 0
AGE_BINS = [1, 31, 61, 91, 121, 181, 241, 301, 366, 731, 65536]   # whole days; each bin includes its left edge
AGE_LABELS = ["1-30", "31-60", "61-90", "91-120", "121-180", "181-240", "241-300", "301-365", "366-730", "over 730"]
_AGE = {}


def _init(raw, masks_dir, out_dir):
    _AGE.update(raw=raw, sindex=raw.sindex, masks_dir=masks_dir, out_dir=out_dir)


def age_one(mask_path):
    raw = _AGE["raw"]
    rel = mask_path.relative_to(_AGE["masks_dir"])
    match = TILE_RE.search(rel.as_posix())
    scene_time = np.datetime64(datetime.strptime(match["ts"], TS_FORMAT))
    with rasterio.open(mask_path) as src:
        mask = src.read(1)
        profile, tags = src.profile, src.tags()
        shape, transform, crs, bounds = mask.shape, src.transform, src.crs, src.bounds
    footprint = box(*transform_bounds(crs, "EPSG:4326", *bounds, densify_pts=21))
    local = raw.iloc[_AGE["sindex"].query(footprint, predicate="intersects")].to_crs(crs)
    age_days = ((scene_time - local["fire_date"].values) / DAY).astype(float)
    local = local[age_days >= 0].assign(age=age_days[age_days >= 0])
    # Oldest first, so a more recent perimeter overwrites an older one where they overlap.
    local = local.sort_values("age", ascending=False)
    age = np.zeros(shape, np.uint16)
    if len(local):
        shapes = ((geom, int(min(max(math.ceil(a), 1), 65535))) for geom, a in zip(local.geometry, local["age"])
                  if geom is not None and not geom.is_empty)
        age = rasterize(shapes, out_shape=shape, transform=transform, fill=0, dtype="uint16")
    age[mask == 255] = NODATA
    target = _AGE["out_dir"] / rel.parent / rel.name.replace("_mask.tif", "_age.tif")
    target.parent.mkdir(parents=True, exist_ok=True)
    profile.update(dtype="uint16", nodata=NODATA, count=1, compress="deflate")
    with rasterio.open(target, "w", **profile) as dst:
        dst.write(age, 1)
        dst.update_tags(**tags, description="days from the most recent EFFIS perimeter covering the pixel to the scene; 0 = none")
    burn = np.isin(mask, [2, 3])
    ages = age[burn & (age > 0)]
    return {"tile": rel.as_posix(), "age_path": target.relative_to(_AGE["out_dir"]).as_posix(),
            "burn_px": int(burn.sum()), "labelled_px": int((age > 0).sum()),
            "burn_age_median": float(np.median(ages)) if len(ages) else np.nan,
            "burn_age_min": float(ages.min()) if len(ages) else np.nan, "burn_age_max": float(ages.max()) if len(ages) else np.nan,
            "hist": np.histogram(ages, bins=AGE_BINS)[0].tolist()}


def add_flags(index):
    month = pd.to_datetime(index["fire_date"], errors="coerce").dt.month
    index["winter_fire"] = month.isin([11, 12, 1, 2])
    if "land_cover" in index:
        index["agricultural"] = index["land_cover"].astype(str).str.lower().str.startswith("agri")
    if "map_source" in index:
        index["modis_mapped"] = index["map_source"].astype(str).str.upper().str.contains("MODIS")
    return index


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--masks-dir", default="data/v4")
    parser.add_argument("--out-dir", default="data/v4", help="may equal --masks-dir: the age file sits beside the mask")
    parser.add_argument("--index", default="", help="release index to extend with age_path and the flags")
    parser.add_argument("--carry-from", default="", help="older index whose extra columns to copy over, e.g. data/v3/index.csv")
    parser.add_argument("--effis-extra", nargs="*", default=[], help="further exports, e.g. the 2018-2019 perimeters")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text())
    masks_dir, out_dir = Path(args.masks_dir).resolve(), Path(args.out_dir).resolve()
    paths = sorted(masks_dir.glob("acquisitions/*/*/*_mask.tif"))
    if args.limit:
        paths = paths[:args.limit]
    if not paths:
        sys.exit(f"No masks found under {masks_dir}/acquisitions")
    raw = load_perimeters(cfg, args.effis_extra)
    print(f"{len(raw):,} perimeters, {len(paths):,} masks", flush=True)
    results = []
    with ProcessPoolExecutor(args.workers, initializer=_init, initargs=(raw, masks_dir, out_dir)) as pool:
        for i, result in enumerate(pool.map(age_one, paths, chunksize=16), 1):
            results.append(result)
            if i % 1000 == 0:
                print(f"{i:,}/{len(paths):,}", flush=True)
    frame = pd.DataFrame(results)
    hist = np.sum(np.stack(frame["hist"].tolist()), axis=0)
    edges = AGE_LABELS
    total = hist.sum() or 1
    lines = ["# Burn age per pixel", "", f"- Masks: {len(frame):,}", f"- Burn pixels (fresh_burn and old_burn) with an age: {int(total):,}",
             f"- Burn pixels with no perimeter in the export: {int(frame['burn_px'].sum() - total):,}", "",
             "| Days from the fire | Burn pixels | % |", "|---|---|---|"]
    lines += [f"| {e} | {int(h):,} | {100 * h / total:.2f} |" for e, h in zip(edges, hist)]
    (out_dir / "age_summary.md").write_text("\n".join(lines) + "\n")
    frame.drop(columns=["hist"]).to_csv(out_dir / "age_report.csv", index=False)
    print("\n".join(lines))
    if args.index:
        index = pd.read_csv(args.index, low_memory=False)
        ages = frame.set_index("tile")
        index["age_path"] = index["mask_path"].map(ages["age_path"])
        index["burn_age_median"] = index["mask_path"].map(ages["burn_age_median"])
        index = add_flags(index)
        carried = []
        if args.carry_from:
            old = pd.read_csv(args.carry_from, low_memory=False).drop_duplicates("tile_id").set_index("tile_id")
            carried = [c for c in old.columns if c not in index.columns]
            for column in carried:
                index[column] = index["tile_id"].map(old[column])
        tmp = f"{args.index}.tmp"   # written whole, then swapped in, so a run reading the index never sees half a file
        index.to_csv(tmp, index=False)
        os.replace(tmp, args.index)
        missing = int(index["age_path"].isna().sum())
        print(f"Wrote age_path, burn_age_median, winter_fire, agricultural and modis_mapped to {args.index}"
              f" ({missing:,} rows without an age file)")
        if carried:
            print(f"Carried from {args.carry_from}: {', '.join(carried)}")


if __name__ == "__main__":
    main()
