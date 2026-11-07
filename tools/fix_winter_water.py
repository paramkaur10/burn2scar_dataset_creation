#!/usr/bin/env python3
"""Remove false water from November to February masks, using the same place on another date.

In low winter sun, NDWI marks mountain shadow and snow as water. Real water is still there on a
March to October date, so in a winter scene a pixel stays water only if the same fire's March to
October scene has water there, or can't see the ground (cloud, shadow or no data). Other masks are
copied unchanged. Burn, cloud and shadow are never touched. Originals are never changed.

  python tools/fix_winter_water.py fix --index data/v2/index.csv --masks-dir data/v2 --out-dir data/v3_water
  python tools/burn_labels.py patch --config config.yaml --masks-dir data/v3_water --out-dir data/v3
  python tools/build_index.py --config config.yaml --masks-dir data/v3 --out data/v3/index.csv \
      --patch-report data/v2/patch_report.csv
  python tools/fix_winter_water.py finish --old-index data/v2/index.csv --report data/v3_water/water_report.csv \
      --index data/v3/index.csv

The patch step gives pixels that turned clear their older_burn class where a perimeter says so.
A winter scene with no March to October partner keeps its water, and index.csv marks it
water_checked = False. finish adds that column, checks the splits are unchanged, and prints the numbers.
"""
import argparse
import shutil
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio

MASK_CLEAR, MASK_CLOUD, MASK_SHADOW, MASK_WATER, MASK_NODATA = 0, 4, 5, 6, 255
WINTER_MONTHS = [11, 12, 1, 2]
SPLITS = ["split", "split_time", "split_loco_IT", "split_loco_ES", "split_loco_FR", "split_loco_EL"]
_ROOTS = {}


def _init(masks_dir, out_dir):
    _ROOTS.update(masks=Path(masks_dir), out=Path(out_dir))


def fix_tile(job):
    relative, partner = job
    source = _ROOTS["masks"] / relative
    target = _ROOTS["out"] / relative
    target.parent.mkdir(parents=True, exist_ok=True)
    if partner is None:
        shutil.copy2(source, target)
        return {"mask_path": relative, "water_before": -1, "water_after": -1}
    with rasterio.open(source) as src:
        mask, profile, colormap, tags = src.read(1), src.profile, src.colormap(1), src.tags()
    water_before = int((mask == MASK_WATER).sum())
    with rasterio.open(_ROOTS["masks"] / partner) as src:
        other = src.read(1)
    if other.shape != mask.shape:
        raise ValueError(f"{relative} and {partner} aren't on the same grid")
    keep = np.isin(other, [MASK_WATER, MASK_CLOUD, MASK_SHADOW, MASK_NODATA])
    fixed = mask.copy()
    fixed[(mask == MASK_WATER) & ~keep] = MASK_CLEAR
    if (fixed == mask).all():
        shutil.copy2(source, target)
    else:
        with rasterio.open(target, "w", **profile) as dst:
            dst.write_colormap(1, colormap)
            dst.write(fixed, 1)
            dst.update_tags(**tags)
    return {"mask_path": relative, "water_before": water_before, "water_after": int((fixed == MASK_WATER).sum())}


def run_fix(args):
    out = Path(args.out_dir)
    if out.resolve() == Path(args.masks_dir).resolve():
        raise SystemExit("--out-dir must be different from --masks-dir, so the originals stay untouched.")
    index = pd.read_csv(args.index, low_memory=False)
    months = pd.to_datetime(index["scene_time"], format="mixed").dt.month
    index["winter"] = months.isin(WINTER_MONTHS)
    summer = index[~index["winter"]].drop_duplicates("fire_id").set_index("fire_id")["mask_path"]
    jobs, status = [], {}
    for row in index.itertuples():
        partner = summer.get(row.fire_id) if row.winter else None
        jobs.append((row.mask_path, partner))
        status[row.mask_path] = "not_winter" if not row.winter else ("checked" if partner else "no_partner")
    with ProcessPoolExecutor(args.workers, initializer=_init, initargs=(args.masks_dir, args.out_dir)) as pool:
        results = list(pool.map(fix_tile, jobs, chunksize=32))
    report = pd.DataFrame(results)
    report["status"] = report["mask_path"].map(status)
    report.to_csv(out / "water_report.csv", index=False)
    checked = report[report["status"] == "checked"]
    print(f"Masks: {len(report):,}. Winter with a March to October partner: {len(checked):,}. "
          f"Winter with no partner, water kept: {(report['status'] == 'no_partner').sum():,}.")
    print(f"Water pixels in checked tiles: {checked['water_before'].sum():,} before, {checked['water_after'].sum():,} after.")


def run_finish(args):
    old = pd.read_csv(args.old_index, low_memory=False).set_index("tile_id")
    new = pd.read_csv(args.index, low_memory=False)
    report = pd.read_csv(args.report)
    new["water_checked"] = ~new["mask_path"].map(report.set_index("mask_path")["status"]).eq("no_partner")
    new.to_csv(args.index, index=False)
    aligned = old.loc[new["tile_id"]]
    for column in [c for c in SPLITS if c in new.columns]:
        same = aligned[column].fillna("").to_numpy() == new[column].fillna("").to_numpy()
        if not same.all():
            raise SystemExit(f"{column} changed for {(~same).sum()} tiles. Stop, and send this output.")
    print("Splits: unchanged.")
    for column in ("frac_fresh", "frac_old"):
        gap = np.abs(aligned[column].to_numpy() - new[column].to_numpy()).max()
        print(f"Largest change in {column}: {gap:.6f}. It should be 0.")
    kept = new["split"].notna()
    valid = 1 - new.loc[kept, "frac_nodata"]
    shares = {}
    for name, frame in (("before", aligned[kept.to_numpy()]), ("after", new[kept])):
        months = pd.to_datetime(frame["scene_time"], format="mixed").dt.month
        shares[name] = frame["frac_water"].groupby(months).mean().mul(100).round(2)
        print(f"Water, % of valid pixels, {name}: {100 * frame['frac_water'].sum() / valid.sum():.2f}. "
              f"Older burn: {100 * frame['frac_older'].sum() / valid.sum():.2f}. Clear: {100 * frame['frac_clear'].sum() / valid.sum():.2f}")
    print("Mean water share by scene month, %")
    print(pd.DataFrame(shares).to_string())
    print(f"Tiles with water_checked False: {(~new.loc[kept, 'water_checked']).sum():,}")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    commands = parser.add_subparsers(dest="command", required=True)
    fix = commands.add_parser("fix")
    fix.add_argument("--index", default="data/v2/index.csv")
    fix.add_argument("--masks-dir", default="data/v2")
    fix.add_argument("--out-dir", default="data/v3_water")
    fix.add_argument("--workers", type=int, default=4)
    finish = commands.add_parser("finish")
    finish.add_argument("--old-index", default="data/v2/index.csv")
    finish.add_argument("--report", default="data/v3_water/water_report.csv")
    finish.add_argument("--index", default="data/v3/index.csv")
    args = parser.parse_args()
    {"fix": run_fix, "finish": run_finish}[args.command](args)


if __name__ == "__main__":
    main()
