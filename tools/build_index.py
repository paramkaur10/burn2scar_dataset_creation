#!/usr/bin/env python3
"""Build the per-tile index and the fixed splits for burn2scar.

Run from the burn2scar repo root, after `burn_labels.py patch`:

  python tools/build_index.py --config config.yaml --masks-dir data/v2 --out data/v2/index.csv

One row per tile: fire, country, dates, location, land cover, class fractions and flags.

Splits. No two tiles that share pixels are ever in different splits.
  split              Spatial 60/20/20 split, the same ratio as FLOGA. Whole blocks
                     (0.5 degrees by default) go to one split, balanced by country,
                     acquisition window and year. Chains of overlapping tiles stay together.
  split_time         The same blocks as split, kept to fires up to 2023 for train, 2024 for
                     validation, and 2025 onwards for test. Its test tiles are also in the test
                     part of split, so a model that saw all years can be scored on the same tiles.
  split_loco_<CC>    One country held out as test, the other three train/validation. A tile
                     from another country that overlaps a test tile is marked unused.
"""
import argparse
import sys
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
import yaml
from rasterio.warp import transform_bounds
from shapely.geometry import box
from shapely.ops import unary_union

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.effis_loader import _read_vector  # noqa: E402
from burn_labels import (  # noqa: E402
    DAY, MASK_CLEAR, MASK_CLOUD, MASK_FRESH, MASK_NODATA, MASK_OLD, MASK_OLDER, MASK_SHADOW, MASK_WATER,
    TILE_RE, TS_FORMAT, load_perimeters, md_table, query_many, to_raster,
)

CLASS_COLUMNS = {
    MASK_CLEAR: "frac_clear", MASK_FRESH: "frac_fresh", MASK_OLD: "frac_old", MASK_CLOUD: "frac_cloud",
    MASK_SHADOW: "frac_shadow", MASK_WATER: "frac_water", MASK_OLDER: "frac_older", MASK_NODATA: "frac_nodata",
}
_STATE = {}


def _init(own, cfg, masks_dir):
    _STATE.update(own=own, cfg=cfg, masks_dir=masks_dir)


def describe_tile(mask_path):
    own, cfg = _STATE["own"], _STATE["cfg"]
    rel = mask_path.relative_to(_STATE["masks_dir"]).as_posix()
    match = TILE_RE.search(rel)
    fire_id, ts = match["fire"], match["ts"]
    scene_time = datetime.strptime(ts, TS_FORMAT)
    with rasterio.open(mask_path) as src:
        mask = src.read(1)
        crs, bounds, transform = src.crs, src.bounds, src.transform
    lon0, lat0, lon1, lat1 = transform_bounds(crs, "EPSG:4326", *bounds, densify_pts=21)
    row = {
        "tile_id": f"{fire_id}/{ts}", "fire_id": fire_id, "ts": ts, "scene_time": scene_time.isoformat(),
        "scene_year": scene_time.year, "epsg": crs.to_epsg(),
        "minx": bounds.left, "miny": bounds.bottom, "maxx": bounds.right, "maxy": bounds.top,
        "lat": (lat0 + lat1) / 2, "lon": (lon0 + lon1) / 2,
        "lon_min": lon0, "lat_min": lat0, "lon_max": lon1, "lat_max": lat1,
        "mask_path": rel, "image_path": rel.replace("_mask.tif", "_s2.tif"),
    }
    values, counts = np.unique(mask, return_counts=True)
    found = dict(zip(values.tolist(), counts.tolist()))
    row.update({column: found.get(value, 0) / mask.size for value, column in CLASS_COLUMNS.items()})
    row["other_values"] = ";".join(str(v) for v in found if v not in CLASS_COLUMNS)

    info = own.get(fire_id)
    if info is None:
        row.update(country="", fire_date="", fire_year=-1, area_ha=np.nan, age_days=np.nan, window="unknown",
                   own_frac=np.nan, own_obscured_frac=np.nan)
        return row
    age = (np.datetime64(scene_time) - np.datetime64(info["fire_date"])) / DAY
    # The pipeline searches whole calendar days, so windows compare dates, not hours.
    days = (np.datetime64(scene_time, "D") - np.datetime64(info["fire_date"], "D")) / DAY
    if cfg["post_fire_min_days"] <= days <= cfg["post_fire_max_days"]:
        window = "fresh"
    elif cfg.get("old_scene_min_days", np.inf) <= days <= cfg.get("old_scene_max_days", -np.inf):
        window = "old"
    else:
        window = "other"
    geometry = gpd.GeoSeries([info["geometry"]], crs=4326).to_crs(crs)
    own_px = to_raster(geometry, mask.shape, transform)
    obscured = np.isin(mask, [MASK_CLOUD, MASK_SHADOW])
    row.update(
        country=info["country"], fire_date=pd.Timestamp(info["fire_date"]).date().isoformat(),
        fire_year=pd.Timestamp(info["fire_date"]).year, area_ha=info["area_ha"], age_days=round(float(age), 2),
        window=window, own_frac=own_px.sum() / mask.size,
        own_obscured_frac=(own_px & obscured).sum() / max(int(own_px.sum()), 1),
    )
    return row


LAND_COVER = {
    "forest": ["broadleaved_forest_percent", "coniferous_forest_percent", "mixed_forest_percent"],
    "shrubland": ["sclerophillous_vegetation_percent", "transitional_vegetation_percent"],
    "other_natural": ["other_natural_percent"],
    "agriculture": ["agriculture_percent"],
    "artificial": ["artificial_percent"],
    "other": ["other_percent"],
}


def fire_attributes(path):
    """Extra EFFIS fields per fire: land cover, mapping source, duration and region. Empty if the export lacks them."""
    gdf = _read_vector(path)
    wanted = [c for columns in LAND_COVER.values() for c in columns]
    if not set(wanted) <= set(gdf.columns) or "id" not in gdf.columns:
        return pd.DataFrame(columns=["fire_id"])
    out = pd.DataFrame({"fire_id": "effis_" + gdf["id"].astype(str).str.replace(r"[^\w\-]", "_", regex=True)})
    shares = pd.DataFrame({name: gdf[columns].apply(pd.to_numeric, errors="coerce").sum(axis=1) for name, columns in LAND_COVER.items()})
    out["land_cover"] = shares.idxmax(axis=1).where(shares.sum(axis=1) > 0, "unknown")
    out["forest_pct"] = shares["forest"].round(1)
    out["shrubland_pct"] = shares["shrubland"].round(1)
    out["agriculture_pct"] = shares["agriculture"].round(1)
    if "map_source" in gdf:
        out["map_source"] = gdf["map_source"]
    if {"initialdate", "finaldate"} <= set(gdf.columns):
        start = pd.to_datetime(gdf["initialdate"], errors="coerce", format="mixed", utc=True)
        end = pd.to_datetime(gdf["finaldate"], errors="coerce", format="mixed", utc=True)
        out["fire_duration_days"] = ((end - start).dt.total_seconds() / 86400).round(2)
    if "admlvl2" in gdf:
        out["region"] = gdf["admlvl2"]
    return out.drop_duplicates("fire_id")


def overlap_pairs(fires):
    """fires: one row per fire with a WGS84 box. Returns the positions of fires whose tiles overlap."""
    boxes = gpd.GeoSeries([box(r.lon_min, r.lat_min, r.lon_max, r.lat_max) for r in fires.itertuples()], crs=4326)
    left, right = query_many(boxes.sindex, boxes.values)
    keep = left < right
    return left[keep], right[keep]


def overlap_groups(count, left, right):
    """Connected groups of overlapping fires. Returns a group number per fire."""
    parent = list(range(count))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for a, b in zip(left, right):
        parent[find(int(a))] = find(int(b))
    roots = {}
    return [roots.setdefault(find(i), len(roots)) for i in range(count)]


def covered_area(fires):
    """Area under at least one tile, and the sum of all tile areas, in square kilometres."""
    parts = []
    for epsg, group in fires.groupby("epsg"):
        squares = [box(r.minx, r.miny, r.maxx, r.maxy) for r in group.itertuples()]
        parts.extend(gpd.GeoSeries(squares, crs=int(epsg)).to_crs(3035).values)
    return unary_union(parts).area / 1e6, sum(part.area for part in parts) / 1e6


def balance_blocks(blocks, targets, seed):
    """Greedy assignment of blocks to splits, largest first.

    blocks: {block: Counter of tiles per balance key}. targets: {split: share}.
    Each key is pushed towards the target share in every split.
    """
    rng = np.random.default_rng(seed)
    names = list(blocks)
    rng.shuffle(names)
    names.sort(key=lambda b: -blocks[b]["tiles"])
    totals = Counter()
    for block in names:
        totals.update(blocks[block])
    assigned, current = {}, {s: Counter() for s in targets}
    for block in names:
        best, best_score = None, np.inf
        for split, share in targets.items():
            score = 0.0
            for key, total in totals.items():
                goal = max(share * total, 1.0)
                after = current[split][key] + blocks[block].get(key, 0)
                # Only overshooting a target costs anything, so splits fill up evenly.
                score += (max(after - goal, 0) / goal) ** 2 * (4.0 if key == "tiles" else 1.0)
            score += current[split]["tiles"] / max(share * totals["tiles"], 1.0) * 1e-3
            if score < best_score:
                best, best_score = split, score
        assigned[block] = best
        current[best].update(blocks[block])
    return assigned


def block_counters(frame):
    counters = defaultdict(Counter)
    for row in frame.itertuples():
        counter = counters[row.block]
        counter["tiles"] += 1
        counter[f"country:{row.country}"] += 1
        counter[f"window:{row.window}"] += 1
        counter[f"year:{row.fire_year}"] += 1
    return counters


def split_table(frame, column):
    rows = []
    for split, group in frame.groupby(column):
        counts = group["country"].value_counts(normalize=True)
        windows = group["window"].value_counts()
        rows.append({
            "Split": split, "Tiles": f"{len(group):,}", "Fires": f"{group['fire_id'].nunique():,}",
            "Blocks": f"{group['block'].nunique():,}",
            "Countries": ", ".join(f"{k} {100 * v:.0f}%" for k, v in counts.items()),
            "Fresh / old": f"{windows.get('fresh', 0):,} / {windows.get('old', 0):,}",
            "Mean fresh %": round(100 * group["frac_fresh"].mean(), 2),
            "Mean old %": round(100 * group["frac_old"].mean(), 2),
        })
    return md_table(pd.DataFrame(rows))


def make_splits(frame, block_deg, seed, pairs):
    """pairs: columns a and b, the fire IDs of every two fires whose tiles overlap."""
    frame = frame.copy()
    neighbours = defaultdict(set)
    for a, b in zip(pairs["a"], pairs["b"]):
        neighbours[a].add(b)
        neighbours[b].add(a)
    fires = frame.drop_duplicates("fire_id")
    # Each overlap group goes to the block holding most of its tiles, so an overlap can't cross a split.
    block_of_tile = (np.floor(frame["lat"] / block_deg).astype(int).astype(str) + "_"
                     + np.floor(frame["lon"] / block_deg).astype(int).astype(str))
    frame["block"] = block_of_tile.groupby(frame["group"]).transform(lambda s: s.mode().iloc[0])

    assigned = balance_blocks(block_counters(frame), {"train": .6, "val": .2, "test": .2}, seed)
    frame["split"] = frame["block"].map(assigned)

    # Time split, on the blocks of the main split, so no pixels are shared. Dropping every tile that overlaps
    # a later one would leave only about 5,000 training tiles, since most fires overlap another fire.
    years = frame["fire_year"]
    frame["split_time"] = "unused"
    frame.loc[(frame["split"] == "train") & (years <= 2023), "split_time"] = "train"
    frame.loc[(frame["split"] == "val") & (years == 2024), "split_time"] = "val"
    frame.loc[(frame["split"] == "test") & (years >= 2025), "split_time"] = "test"

    # Leave one country out. Test holds only that country. Tiles of other countries that overlap
    # its tiles (fires near a border) are dropped, so they can't leak into training.
    for country in sorted(c for c in frame["country"].unique() if c):
        held_out = set(fires.loc[fires["country"] == country, "fire_id"])
        blocked = {n for f in held_out for n in neighbours[f]} - held_out
        rest = frame[~frame["fire_id"].isin(held_out | blocked)]
        chosen = balance_blocks(block_counters(rest), {"train": .85, "val": .15}, seed)
        column = f"split_loco_{country}"
        frame[column] = np.where(frame["fire_id"].isin(held_out), "test", "unused")
        frame.loc[rest.index, column] = rest["block"].map(chosen)
    return frame


def check_no_leak(frame, columns, pairs):
    """Stops the run if two tiles that share pixels sit in two different splits."""
    for column in columns:
        by_fire = frame.drop_duplicates("fire_id").set_index("fire_id")[column]
        assert frame.groupby("fire_id")[column].nunique().max() == 1, f"{column}: one fire in two splits"
        a, b = by_fire.reindex(pairs["a"]).to_numpy(), by_fire.reindex(pairs["b"]).to_numpy()
        crossing = (a != b) & (a != "unused") & (b != "unused")
        assert not crossing.any(), f"{column}: {int(crossing.sum())} overlapping fire pairs fall in different splits"


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--masks-dir", required=True, help="folder that contains acquisitions/")
    parser.add_argument("--out", required=True, help="path of the index CSV")
    parser.add_argument("--effis-extra", nargs="*", default=[], help="same wider exports used for the patch")
    parser.add_argument("--patch-report", help="patch_report.csv, to record how much each mask changed")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--block-deg", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text())
    masks_dir = Path(args.masks_dir).resolve()
    paths = sorted(masks_dir.glob("acquisitions/*/*/*_mask.tif"))
    if not paths:
        sys.exit(f"No masks found under {masks_dir}/acquisitions")
    raw = load_perimeters(cfg, args.effis_extra)
    selected = raw[raw["reason"] == ""].drop_duplicates("fire_id")
    own = {r.fire_id: {"fire_date": r.fire_date, "geometry": r.geometry, "country": r.country, "area_ha": r.area_ha}
           for r in selected.itertuples()}
    print(f"{len(paths):,} masks, {len(own):,} selected fires", flush=True)
    with ProcessPoolExecutor(args.workers, initializer=_init, initargs=(own, cfg, masks_dir)) as pool:
        rows = list(pool.map(describe_tile, paths, chunksize=32))
    frame = pd.DataFrame(rows)

    orphan = frame["country"] == ""
    if orphan.any():
        print(f"Warning: {int(orphan.sum()):,} tiles have no matching fire in the export; they get no split.")
    fires = frame.drop_duplicates("fire_id").reset_index(drop=True)
    left, right = overlap_pairs(fires)
    fires["group"] = overlap_groups(len(fires), left, right)
    pairs = pd.DataFrame({"a": fires["fire_id"].to_numpy()[left], "b": fires["fire_id"].to_numpy()[right]})
    frame = frame.merge(fires[["fire_id", "group"]], on="fire_id")
    frame = frame.merge(fire_attributes(cfg["effis_path"]), on="fire_id", how="left")
    frame["scene_rule_pass"] = frame["own_obscured_frac"] <= cfg.get("burn_cloud_max_fraction", 0.5)
    frame["tile_filled"] = frame["own_frac"] >= .95
    if args.patch_report:
        report = pd.read_csv(args.patch_report)
        report["tile_id"] = report["tile"].map(lambda p: "/".join(p.split("/")[1:3]))
        frame = frame.merge(report[["tile_id", "changed_px"]], on="tile_id", how="left")

    usable = make_splits(frame[frame["country"] != ""], args.block_deg, args.seed, pairs)
    known = set(usable["fire_id"])
    pairs = pairs[pairs["a"].isin(known) & pairs["b"].isin(known)]
    split_columns = ["split", "split_time"] + [c for c in usable.columns if c.startswith("split_loco_")]
    check_no_leak(usable, split_columns, pairs)
    unique_km2, total_km2 = covered_area(fires)
    frame = frame.merge(usable[["tile_id", "block", *split_columns]], on="tile_id", how="left")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(out, index=False)

    lines = ["# burn2scar index and splits", "",
             f"- Tiles: {len(frame):,}", f"- Fires: {frame['fire_id'].nunique():,}",
             f"- Overlap groups: {frame['group'].nunique():,}",
             f"- Largest overlap group: {int(frame.drop_duplicates('fire_id')['group'].value_counts().max()):,} fires",
             f"- Fires whose tile overlaps another fire's tile: {len(set(pairs['a']) | set(pairs['b'])):,}",
             f"- Land covered by at least one tile: {unique_km2:,.0f} km2 (tiles add up to {total_km2:,.0f} km2)",
             f"- Blocks: {usable['block'].nunique():,} at {args.block_deg} degrees",
             f"- Tiles where the fire's own scar is over "
             f"{100 * cfg.get('burn_cloud_max_fraction', .5):.0f}% obscured: {int((~frame['scene_rule_pass']).sum()):,}",
             f"- Tiles at least 95% covered by their own fire: {int(frame['tile_filled'].sum()):,}",
             f"- Tiles from a scene outside both windows: {int((frame['window'] == 'other').sum()):,}",
             f"- Tiles with unexpected mask values: {int((frame['other_values'] != '').sum()):,}", ""]
    for column in split_columns:
        lines += [f"## {column}", "", split_table(usable, column), ""]
    text = "\n".join(lines)
    summary = out.with_name("index_summary.md")
    summary.write_text(text)
    print(text)
    print(f"Wrote {out} and {summary}")


if __name__ == "__main__":
    main()
