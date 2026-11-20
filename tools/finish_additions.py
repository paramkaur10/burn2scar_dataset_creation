#!/usr/bin/env python3
"""After acquire_additions.py: the release's label rules, the age rasters and the index for the new tiles. CPU only.

  python tools/finish_additions.py --workers 16          # water check, burn classes, ages, index
  python tools/finish_additions.py --workers 16 --zip    # and additions.zip for the release

1. Winter water, by the rule of fix_winter_water.py: in a November to February scene, water stays water only where the
   same place on a March to October scene of the same fire is water or can't be seen. Boundary and first-look tiles
   share their fire's grid, so the fire's released March to October tile is the partner. Burn-free and large-fire tiles
   have none; they keep their water and get water_checked False, like the release's winter tiles without a partner.
2. Burn classes from every EFFIS perimeter since 2018, older_burn included: burn_labels.py patch, as for the release.
3. Age rasters: make_age.py, as for the release.
4. additions_index.csv: one row per tile, with the release index's columns where they apply, plus subset (the part),
   source_tile_id, offset_km, edge_burn_frac, history_days (days of label history before the scene, from 1 January
   2018) and the split. Boundary and first-look tiles take every split of their fire's released tile, as they share its
   ground; burn-free and large-fire tiles take the split of their 0.5 degree block (acquire_additions.py).

Reads data/additions/targets.csv and acquired.csv; writes data/additions/water/, data/additions/v4/ (masks and ages) and
data/additions/v4/additions_index.csv. Images stay in data/additions/raw/.
"""
import argparse
import shutil
import subprocess
import sys
import warnings
import zipfile
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio
import yaml

TOOLS = Path(__file__).resolve().parent
sys.path.insert(0, str(TOOLS))
sys.path.insert(0, str(TOOLS.parent))
warnings.filterwarnings("ignore", message=".*has no geotransform.*")
warnings.filterwarnings("ignore", message=".*geographic CRS.*")

from acquire_additions import footprint  # noqa: E402

CLEAR, FRESH, OLD, CLOUD, SHADOW, WATER, OLDER, NODATA = 0, 2, 3, 4, 5, 6, 7, 255
CLASS_COLUMNS = {CLEAR: "frac_clear", FRESH: "frac_fresh", OLD: "frac_old", CLOUD: "frac_cloud", SHADOW: "frac_shadow",
                 WATER: "frac_water", OLDER: "frac_older", NODATA: "frac_nodata"}
TS_FORMAT = "%Y-%m-%dT%H-%M-%S"
HISTORY_START = pd.Timestamp("2018-01-01")
FIRE_COLUMNS = ["region", "land_cover", "forest_pct", "shrubland_pct", "agriculture_pct", "map_source", "fire_duration_days",
                "winter_fire", "agricultural", "modis_mapped", "fire_dnbr_flag", "group", "split_time", "split_loco_IT",
                "split_loco_ES", "split_loco_FR", "split_loco_EL"]
ZIP_DATE = (2026, 1, 1, 0, 0, 0)


def finished_tiles(out):
    targets = pd.read_csv(out / "targets.csv", low_memory=False)
    log = pd.read_csv(out / "acquired.csv").drop_duplicates("target_id", keep="last")
    done = log[log["status"].isin(["ok", "kept_best_available"])]
    tiles = done[["target_id", "status", "scene_time", "valid_frac", "obscured_frac", "rule_pass", "image_path", "mask_path"]]
    return tiles.merge(targets, on="target_id", how="left")


def is_winter(time):
    return pd.Timestamp(time).month in (11, 12, 1, 2)


def partner_mask(row, release, release_root):
    """The fire's released tile on the same grid with a March to October scene, nearest in time, or None."""
    if row["part"] not in ("boundary", "firstlook") or release is None:
        return None
    same = release[(release["fire_id"] == row["fire_id"]) & (release["epsg"] == row["epsg"])
                   & ((release["minx"] - row["minx"]).abs() < 1) & ((release["miny"] - row["miny"]).abs() < 1)]
    same = same[~pd.to_datetime(same["scene_time"]).dt.month.isin([11, 12, 1, 2])]
    if same.empty:
        return None
    gap = (pd.to_datetime(same["scene_time"]) - pd.Timestamp(row["scene_time"])).abs()
    return release_root / same.loc[gap.idxmin(), "mask_path"]


def water_step(tiles, out, release, release_root):
    """Copies every raw mask to water/, with the winter rule applied where a partner exists. Returns water_checked."""
    checked = {}
    for _, row in tiles.iterrows():
        source = out / "raw" / row["mask_path"]
        target = out / "water" / row["mask_path"]
        target.parent.mkdir(parents=True, exist_ok=True)
        if not is_winter(row["scene_time"]):
            shutil.copy2(source, target)
            checked[row["target_id"]] = True
            continue
        partner = partner_mask(row, release, release_root)
        if partner is None or not partner.exists():
            shutil.copy2(source, target)
            checked[row["target_id"]] = False
            continue
        with rasterio.open(source) as src:
            mask, profile, colormap = src.read(1), src.profile, src.colormap(1)
        with rasterio.open(partner) as src:
            other = src.read(1)
        keep = np.isin(other, [WATER, CLOUD, SHADOW, NODATA])
        fixed = mask.copy()
        fixed[(mask == WATER) & ~keep] = CLEAR
        with rasterio.open(target, "w", **profile) as dst:
            dst.write_colormap(1, colormap)
            dst.write(fixed, 1)
        checked[row["target_id"]] = True
    return checked


def own_fire_shares(row, mask, perimeters):
    """Share of the tile inside the tile's own fire, and share of that part hidden by cloud or shadow."""
    from rasterio.features import rasterize
    from rasterio.transform import from_bounds
    if not row["fire_id"] or pd.isna(row["fire_id"]) or row["fire_id"] not in perimeters.index:
        return np.nan, np.nan
    geometry = perimeters.loc[[row["fire_id"]]].to_crs(int(row["epsg"])).geometry
    transform = from_bounds(row["minx"], row["miny"], row["maxx"], row["maxy"], mask.shape[1], mask.shape[0])
    own = rasterize(((g, 1) for g in geometry if g is not None and not g.is_empty), out_shape=mask.shape,
                    transform=transform, fill=0, dtype=np.uint8).astype(bool)
    hidden = np.isin(mask, [CLOUD, SHADOW])
    return own.sum() / mask.size, (own & hidden).sum() / max(int(own.sum()), 1)


def index_step(tiles, out, checked, cfg, release):
    from build_index import fire_attributes
    from src.effis_loader import load_all_perimeters
    perimeters = load_all_perimeters(cfg["effis_path"], cfg.get("effis_extra")).drop_duplicates("fire_id").set_index("fire_id")
    attributes = fire_attributes(cfg["effis_path"]).set_index("fire_id")
    source_rows = release.set_index("tile_id") if release is not None else pd.DataFrame()
    rows = []
    for _, row in tiles.iterrows():
        mask_rel = row["mask_path"]
        age_rel = mask_rel.replace("_mask.tif", "_age.tif")
        with rasterio.open(out / "v4" / mask_rel) as src:
            mask = src.read(1)
        age_path = out / "v4" / age_rel
        burn_age = np.nan
        if age_path.exists():
            with rasterio.open(age_path) as src:
                age = src.read(1)
            burn = np.isin(mask, [FRESH, OLD]) & (age > 0)
            burn_age = float(np.median(age[burn])) if burn.any() else np.nan
        values, counts = np.unique(mask, return_counts=True)
        found = dict(zip(values.tolist(), counts.tolist()))
        stamp = mask_rel.split("/")[2]
        scene_time = datetime.strptime(stamp, TS_FORMAT)
        lon_min, lat_min, lon_max, lat_max = footprint(int(row["epsg"]), (row["minx"], row["miny"], row["maxx"], row["maxy"])).bounds
        fire_date = pd.Timestamp(row["fire_date"]) if isinstance(row["fire_date"], str) and row["fire_date"] else None
        own_frac, own_hidden = own_fire_shares(row, mask, perimeters)
        record = {
            "tile_id": f"{row['tile_name']}/{stamp}", "subset": row["part"], "fire_id": row["fire_id"] if fire_date else "",
            "source_tile_id": row.get("source_tile_id", ""), "ts": stamp, "scene_time": scene_time.isoformat(),
            "scene_year": scene_time.year, "epsg": int(row["epsg"]), "minx": row["minx"], "miny": row["miny"],
            "maxx": row["maxx"], "maxy": row["maxy"], "lat": (lat_min + lat_max) / 2, "lon": (lon_min + lon_max) / 2,
            "lon_min": lon_min, "lat_min": lat_min, "lon_max": lon_max, "lat_max": lat_max,
            "image_path": mask_rel.replace("_mask.tif", "_s2.tif"), "mask_path": mask_rel,
            "age_path": age_rel if age_path.exists() else "",
            **{column: found.get(value, 0) / mask.size for value, column in CLASS_COLUMNS.items()},
            "other_values": ";".join(str(v) for v in found if v not in CLASS_COLUMNS),
            "burn_age_median": burn_age, "country": row["country"],
            "fire_date": fire_date.date().isoformat() if fire_date else "", "fire_year": fire_date.year if fire_date else -1,
            "area_ha": row["area_ha"],
            "age_days": round((pd.Timestamp(scene_time) - fire_date) / pd.Timedelta(days=1), 2) if fire_date else np.nan,
            "window": row["part"], "own_frac": own_frac, "own_obscured_frac": own_hidden,
            "scene_rule_pass": bool(row["rule_pass"]) if row["part"] in ("boundary", "large") else True,
            "valid_frac": row["valid_frac"], "obscured_frac": row["obscured_frac"],
            "water_checked": checked.get(row["target_id"], True), "offset_km": row["offset_km"],
            "edge_burn_frac": row["edge_burn_frac"], "history_days": (pd.Timestamp(scene_time.date()) - HISTORY_START).days,
            "block": row["block"], "split": row["split"],
        }
        source = row.get("source_tile_id")
        if row["part"] in ("boundary", "firstlook") and isinstance(source, str) and source in source_rows.index:
            for column in FIRE_COLUMNS:
                if column in source_rows.columns:
                    record[column] = source_rows.at[source, column]
        elif row["part"] == "large" and row["fire_id"] in attributes.index:
            for column in attributes.columns:
                record[column] = attributes.at[row["fire_id"], column]
            month = fire_date.month if fire_date else 0
            record["winter_fire"] = month in (11, 12, 1, 2)
            record["agricultural"] = str(record.get("land_cover", "")).lower().startswith("agri")
            record["modis_mapped"] = "MODIS" in str(record.get("map_source", "")).upper()
        rows.append(record)
    index = pd.DataFrame(rows).sort_values(["subset", "tile_id"])
    # A burn-free tile must have no EFFIS perimeter of any date since 2018: the plan may have used fewer exports.
    burned = (index["subset"] == "negative") & (index[["frac_fresh", "frac_old", "frac_older"]].sum(axis=1) > 0)
    index[burned].to_csv(out / "v4" / "negatives_dropped.csv", index=False)
    index = index[~burned]
    index.to_csv(out / "v4" / "additions_index.csv", index=False)
    return index, int(burned.sum())


def summary(index, out, dropped=0):
    lines = ["# New tiles", "", f"{len(index):,} tiles. Burn-free tiles left out because a perimeter since 2018 touches them: "
             f"{dropped:,} (negatives_dropped.csv).", "",
             "| Subset | Tiles | train | val | test | Fresh burn % | Old burn % | Older burn % | Cloud % | Water checked |",
             "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for subset, group in index.groupby("subset"):
        splits = group["split"].value_counts()
        valid = 1 - group["frac_nodata"]
        share = lambda c: 100 * (group[c] / valid.where(valid > 0)).mean()
        lines.append(f"| {subset} | {len(group):,} | {splits.get('train', 0):,} | {splits.get('val', 0):,} | {splits.get('test', 0):,} | "
                     f"{share('frac_fresh'):.2f} | {share('frac_old'):.2f} | {share('frac_older'):.2f} | {share('frac_cloud'):.2f} | "
                     f"{100 * group['water_checked'].mean():.1f} % |")
    boundary = index[index["subset"] == "boundary"]
    if len(boundary):
        lines += ["", f"Boundary scenes: {int((boundary['age_days'] <= 90).sum()):,} at 90 days or less, "
                      f"{int((boundary['age_days'] > 90).sum()):,} above; ages {boundary['age_days'].min():.0f} to {boundary['age_days'].max():.0f} days."]
    first = index[index["subset"] == "firstlook"]
    if len(first):
        lines += [f"First looks: {first['age_days'].round().value_counts().sort_index().to_dict()} (days after the EFFIS date: tiles)."]
    negative = index[index["subset"] == "negative"]
    if len(negative):
        lines += [f"Burn-free tiles kept: {len(negative):,}, in {negative['block'].nunique():,} blocks; median distance to the "
                  f"released tile they were cut beside {negative['offset_km'].median():.1f} km."]
    (out / "v4" / "additions_summary.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


def make_zip(index, out):
    target = out / "additions.zip"
    temporary = target.with_name(target.name + ".part")
    with zipfile.ZipFile(temporary, "w", zipfile.ZIP_DEFLATED, allowZip64=True) as archive:
        for _, row in index.iterrows():
            for root, column in ((out / "raw", "image_path"), (out / "v4", "mask_path"), (out / "v4", "age_path")):
                if not row[column]:
                    continue
                info = zipfile.ZipInfo(row[column], date_time=ZIP_DATE)
                info.external_attr = 0o644 << 16
                info.compress_type = zipfile.ZIP_STORED
                archive.writestr(info, (root / row[column]).read_bytes())
        info = zipfile.ZipInfo("additions_index.csv", date_time=ZIP_DATE)
        info.external_attr = 0o644 << 16
        archive.writestr(info, (out / "v4" / "additions_index.csv").read_bytes(), compress_type=zipfile.ZIP_DEFLATED)
    temporary.replace(target)
    print(f"Wrote {target}, {target.stat().st_size / 1e6:,.0f} MB")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--out", default="data/additions")
    parser.add_argument("--release-index", default="data/v4/index.csv")
    parser.add_argument("--release-masks", default="data/v4", help="the released masks, for the winter water partners")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--zip", action="store_true", help="also write data/additions/additions.zip")
    args = parser.parse_args()
    cfg = yaml.safe_load(Path(args.config).read_text())
    if not cfg.get("effis_extra"):
        raise SystemExit("config.yaml has no effis_extra. The labels need the EFFIS export with the perimeters from 2018, "
                         "as in the release: add  effis_extra: [effis/<file>.zip]  to config.yaml.")
    out = Path(args.out)
    tiles = finished_tiles(out)
    if tiles.empty:
        raise SystemExit("No finished tiles in acquired.csv yet.")
    release = pd.read_csv(args.release_index, low_memory=False) if Path(args.release_index).exists() else None
    print(f"{len(tiles):,} tiles: {tiles['part'].value_counts().to_dict()}", flush=True)

    for folder in ("water", "v4"):
        if (out / folder).exists():
            shutil.rmtree(out / folder)
    checked = water_step(tiles, out, release, Path(args.release_masks))
    print(f"Winter water: {sum(not v for v in checked.values()):,} winter tiles without a partner keep their water", flush=True)
    run = lambda *command: subprocess.run([sys.executable, *map(str, command)], check=True)
    run(TOOLS / "burn_labels.py", "patch", "--config", args.config, "--masks-dir", out / "water", "--out-dir", out / "v4",
        "--workers", args.workers)
    run(TOOLS / "make_age.py", "--config", args.config, "--masks-dir", out / "v4", "--out-dir", out / "v4", "--workers", args.workers)
    index, dropped = index_step(tiles, out, checked, cfg, release)
    summary(index, out, dropped)
    if args.zip:
        make_zip(index, out)


if __name__ == "__main__":
    main()
