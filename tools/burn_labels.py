#!/usr/bin/env python3
"""Measure and fix missing burn labels in burn2scar masks.

The extraction pipeline rasterises only the fires that passed its selection
filters, so burned areas from every other EFFIS perimeter stay `clear`.

  audit  Measures this from the EFFIS export and the Hugging Face file list. It
         also reports old scars, filled tiles, overlapping tiles and why fires
         are missing. It needs no imagery and changes nothing.

  patch  Writes corrected copies of existing masks to a new folder. It never
         changes the originals and never downloads imagery. The result equals
         regenerating the masks from all perimeters, with cloud, shadow, water
         and nodata kept as they are. It also marks EFFIS scars older than
         burn_max_age_days as a new class, older_burn (7), so they stop counting
         as clear. Turn that off with --no-older-burn.

Run from the burn2scar repo root:

  python tools/burn_labels.py audit --config config.yaml --out audit

  python tools/burn_labels.py patch --config config.yaml \
      --masks-dir data/v1 --out-dir data/v2 --limit 200

If your export covers only the selected countries or dates, add a wider one to
measure the rest (the public layer is linked from the EFFIS data page). List it
under effis_extra in config.yaml, so these tools and the pipeline all use it.
"""
import argparse
import json
import re
import shutil
import sys
import warnings
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
import yaml
from pyproj import Transformer
from rasterio.features import rasterize
from rasterio.transform import from_bounds
from rasterio.warp import transform_bounds
from shapely.geometry import box

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
from src.effis_loader import load_effis  # noqa: E402

warnings.filterwarnings("ignore", message=".*Geometry is in a geographic CRS.*")
warnings.filterwarnings("ignore", message=".*wgs84_to_utm.*")

# Same values as src/s2_pipeline.py.
MASK_CLEAR, MASK_FRESH, MASK_OLD = 0, 2, 3
MASK_CLOUD, MASK_SHADOW, MASK_WATER, MASK_NODATA = 4, 5, 6, 255
MASK_OLDER = 7  # New in v2: burned more than burn_max_age_days before the scene.
OLDER_COLOUR = (120, 95, 70, 255)

TILE_RE = re.compile(r"acquisitions/(?P<fire>[^/]+)/(?P<ts>[^/]+)/(?P=fire)_(?P=ts)_mask\.tif$")
TS_FORMAT = "%Y-%m-%dT%H-%M-%S"
DAY = np.timedelta64(1, "D")


# ----------------------------------------------------------------- perimeters

def exclusion_reasons(gdf, cfg):
    """Why each perimeter fails the filters in run_extraction.py ('' if it passes)."""
    reason = pd.Series("", index=gdf.index, dtype=object)
    lat = gdf.geometry.centroid.y
    checks = [
        ("outside_countries", ~gdf["country"].isin({c.upper() for c in cfg["countries"]})),
        ("before_start_date", gdf["fire_date"] < pd.Timestamp(cfg["start_date"])),
        ("after_end_date", gdf["fire_date"] > pd.Timestamp(cfg["end_date"])),
        ("below_min_area", gdf["area_ha"].fillna(0) < cfg["min_area_ha"]),
        ("above_max_area", gdf["area_ha"] > cfg.get("max_area_ha", np.inf)),
        ("france_north_of_limit", (gdf["country"] == "FR") & (lat > cfg["southern_france_max_lat"])),
    ]
    for name, failed in checks:
        reason[(reason == "") & failed] = name
    return reason


def load_perimeters(cfg, extra_paths=()):
    """All EFFIS perimeters, with `reason` empty for the ones the pipeline rasterised."""
    primary = load_effis(cfg["effis_path"])
    primary["reason"] = exclusion_reasons(primary, cfg)
    parts = [primary]
    known = set(primary["fire_id"])
    for path in [*(cfg.get("effis_extra") or []), *extra_paths]:
        extra = load_effis(path)
        extra = extra[~extra["fire_id"].isin(known)].copy()
        known |= set(extra["fire_id"])
        reason = exclusion_reasons(extra, cfg)
        reason[reason == ""] = "not_in_build_export"
        extra["reason"] = reason
        parts.append(extra)
    gdf = pd.concat(parts, ignore_index=True)
    return gpd.GeoDataFrame(gdf, geometry="geometry", crs=primary.crs)


def query_many(sindex, geoms):
    """Pairs of (input index, tree index) that intersect, on old and new geopandas."""
    values = np.asarray(geoms)
    try:
        return sindex.query(values, predicate="intersects")
    except (TypeError, ValueError, AttributeError):
        return sindex.query_bulk(values, predicate="intersects")


def to_raster(geoms, shape, transform):
    geoms = [g for g in geoms if g is not None and not g.is_empty]
    if not geoms:
        return np.zeros(shape, bool)
    return rasterize(((g, 1) for g in geoms), out_shape=shape, transform=transform,
                     fill=0, dtype=np.uint8).astype(bool)


def tile_bounds(lat, lon, size_m):
    """Same grid as get_utm_bbox in src/s2_pipeline.py."""
    from sentinelhub.geo_utils import get_utm_crs, wgs84_to_utm
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        east, north = wgs84_to_utm(lon, lat)
    east, north = 10 * int(east / 10), 10 * int(north / 10)
    half = size_m // 2
    return int(get_utm_crs(lon, lat).epsg), (east - half, north - half, east + half, north + half)


_TRANSFORMERS = {}


def footprint_wgs84(epsg, bounds):
    if epsg not in _TRANSFORMERS:
        _TRANSFORMERS[epsg] = Transformer.from_crs(epsg, 4326, always_xy=True)
    x0, y0, x1, y1 = bounds
    lon, lat = _TRANSFORMERS[epsg].transform([x0, x1, x1, x0], [y0, y0, y1, y1])
    return box(min(lon), min(lat), max(lon), max(lat))


def parse_tiles(paths):
    rows = []
    for path in paths:
        match = TILE_RE.search(path)
        if match:
            rows.append((match["fire"], match["ts"], datetime.strptime(match["ts"], TS_FORMAT)))
    return pd.DataFrame(rows, columns=["fire_id", "ts", "scene_time"])


# ---------------------------------------------------------------------- audit

def list_dataset_files(args, cfg):
    """(path, size) pairs from a local list or from Hugging Face."""
    if args.file_list:
        pairs = []
        for line in Path(args.file_list).read_text().splitlines():
            if line.strip():
                bits = line.split()
                pairs.append((bits[0], int(bits[1]) if len(bits) > 1 else 0))
        return pairs
    from huggingface_hub import HfApi
    api = HfApi()
    if not hasattr(api, "list_repo_tree"):
        return [(path, 0) for path in api.list_repo_files(cfg["hf_repo_id"], repo_type="dataset")]
    return [(item.path, getattr(item, "size", 0) or 0)
            for item in api.list_repo_tree(cfg["hf_repo_id"], repo_type="dataset", recursive=True)
            if hasattr(item, "size")]


def audit_tiles(tiles, raw, cfg):
    size_m, res_m = cfg["bbox_size_m"], cfg["s2_resolution_m"]
    n = size_m // res_m
    total_px = n * n
    max_age = cfg["burn_max_age_days"]
    selected = raw[raw["reason"] == ""].drop_duplicates("fire_id").set_index("fire_id")
    sindex = raw.sindex
    rows, footprints, orphans = [], {}, []

    for fire_id, group in tiles.groupby("fire_id", sort=False):
        if fire_id not in selected.index:
            orphans.append(fire_id)
            continue
        own = selected.loc[fire_id]
        centre = own.geometry.centroid
        epsg, bounds = tile_bounds(centre.y, centre.x, size_m)
        footprint = footprint_wgs84(epsg, bounds)
        footprints[fire_id] = footprint
        transform = from_bounds(*bounds, n, n)
        local = raw.iloc[sindex.query(footprint, predicate="intersects")].to_crs(epsg)
        dates = local["fire_date"].values
        was_rasterised = (local["reason"] == "").values
        own_px = to_raster(local.loc[local["fire_id"] == fire_id, "geometry"], (n, n), transform)

        for row in group.itertuples():
            scene_time = np.datetime64(row.scene_time)
            age = (scene_time - dates) / DAY
            in_window = (age >= 0) & (age <= max_age)
            included = to_raster(local.geometry[was_rasterised & in_window], (n, n), transform)
            reasons, missing = {}, np.zeros((n, n), bool)
            for reason in sorted(set(local["reason"].values[~was_rasterised & in_window])):
                layer = to_raster(local.geometry[(local["reason"].values == reason) & in_window],
                                  (n, n), transform) & ~included
                if layer.any():
                    reasons[reason] = int(layer.sum())
                    missing |= layer
            older = to_raster(local.geometry[age > max_age], (n, n), transform) & ~included & ~missing
            own_age = (scene_time - np.datetime64(own["fire_date"])) / DAY
            # The pipeline searches whole calendar days, so windows compare dates, not hours.
            days = (np.datetime64(scene_time, "D") - np.datetime64(own["fire_date"], "D")) / DAY
            if cfg["post_fire_min_days"] <= days <= cfg["post_fire_max_days"]:
                window = "fresh"
            elif cfg.get("old_scene_min_days", np.inf) <= days <= cfg.get("old_scene_max_days", -np.inf):
                window = "old"
            else:
                window = "other"
            rows.append({
                "fire_id": fire_id, "ts": row.ts, "country": own["country"], "area_ha": own["area_ha"],
                "fire_date": own["fire_date"], "age_days": round(float(own_age), 2), "window": window,
                "epsg": epsg, "minx": bounds[0], "miny": bounds[1], "maxx": bounds[2], "maxy": bounds[3],
                "own_frac": own_px.sum() / total_px,
                "burn_rasterised_frac": included.sum() / total_px,
                "burn_missing_frac": missing.sum() / total_px,
                "older_scar_frac": older.sum() / total_px,
                "missing_reasons": ";".join(f"{k}={v}" for k, v in reasons.items()),
            })
    return pd.DataFrame(rows), footprints, orphans


def overlap_groups(footprints):
    """Connected groups of fires whose tiles overlap, for leakage-safe splits."""
    ids = list(footprints)
    if not ids:
        return pd.DataFrame(columns=["fire_id", "group", "overlapping_fires"])
    series = gpd.GeoSeries([footprints[i] for i in ids], crs=4326)
    left, right = query_many(series.sindex, series.values)
    parent = list(range(len(ids)))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    neighbours = Counter()
    for a, b in zip(left, right):
        if a != b:
            neighbours[int(a)] += 1
            parent[find(int(a))] = find(int(b))
    roots = {}
    return pd.DataFrame({
        "fire_id": ids,
        "group": [roots.setdefault(find(i), len(roots)) for i in range(len(ids))],
        "overlapping_fires": [neighbours[i] for i in range(len(ids))],
    })


def status_group(status):
    status = str(status)
    if "quota" in status or "budget" in status:
        return "skipped: quota or budget"
    if status.startswith(("already done", "ok")):
        return "done"
    if status == "skipped":
        return "skipped: no usable scene"
    if "low disk" in status:
        return "skipped: low disk"
    if status.startswith("error"):
        return "error"
    if "upload failed" in status:
        return "upload failed"
    return "other"


def md_table(frame):
    frame = frame.copy().astype(object).where(frame.notna(), "")
    head = "| " + " | ".join(map(str, frame.columns)) + " |"
    rule = "|" + "---|" * len(frame.columns)
    body = ["| " + " | ".join(map(str, row)) + " |" for row in frame.itertuples(index=False)]
    return "\n".join([head, rule, *body])


def share(part, whole):
    return f"{part:,} ({100 * part / whole:.1f}%)" if whole else f"{part:,}"


def summarise(raw, cfg, tile_df, groups, orphans, files, out_dir):
    selected = raw[raw["reason"] == ""].drop_duplicates("fire_id")
    n_tiles = len(tile_df)
    lines = ["# burn2scar audit", ""]

    lines += ["## EFFIS export", "",
              f"- Build export: `{cfg['effis_path']}`",
              f"- Perimeters loaded: {len(raw):,}",
              f"- Dates: {raw['fire_date'].min().date()} to {raw['fire_date'].max().date()}",
              f"- Countries: {', '.join(f'{k} {v:,}' for k, v in raw['country'].value_counts().items())}",
              f"- Passing the build filters: {len(selected):,}",
              f"- Duplicate fire IDs in the export: {int(raw['fire_id'].duplicated().sum()):,}", ""]
    if set(raw["country"]) <= {c.upper() for c in cfg["countries"]} or \
            raw["fire_date"].min() >= pd.Timestamp(cfg["start_date"]):
        lines += ["**Warning:** this export looks limited to the selected countries or dates, so burned",
                  "areas from other countries or earlier years can't be measured. Rerun with",
                  "`--effis-extra` and a wider EFFIS export.", ""]

    lines += ["## Missing burn labels", "",
              f"- Tiles checked: {n_tiles:,}",
              f"- Tile IDs not found among selected fires: {len(orphans):,}"]
    if orphans:
        why = raw.drop_duplicates("fire_id").set_index("fire_id")["reason"]
        lines += ["  - " + ", ".join(f"{fire} ({why.get(fire, 'not in the export')})" for fire in orphans[:20])]
    if n_tiles:
        missing = tile_df["burn_missing_frac"]
        burned, absent = tile_df["burn_rasterised_frac"].sum(), missing.sum()
        lines += [f"- Tiles with any missing burn: {share(int((missing > 0).sum()), n_tiles)}",
                  f"- Tiles with at least 1% of the tile missing: {share(int((missing >= .01).sum()), n_tiles)}",
                  f"- Tiles with at least 5% missing: {share(int((missing >= .05).sum()), n_tiles)}",
                  f"- Tiles with at least 25% missing: {share(int((missing >= .25).sum()), n_tiles)}",
                  f"- Missing burn as a share of all tile pixels: {100 * missing.mean():.3f}%",
                  f"- Missing burn compared with rasterised burn: {100 * absent / burned if burned else 0:.1f}%", ""]
        per_reason = defaultdict(lambda: [0, 0])
        for text in tile_df["missing_reasons"]:
            for item in filter(None, str(text).split(";")):
                reason, px = item.split("=")
                per_reason[reason][0] += 1
                per_reason[reason][1] += int(px)
        if per_reason:
            lines += [md_table(pd.DataFrame(
                [(k, f"{v[0]:,}", f"{v[1]:,}") for k, v in sorted(per_reason.items(), key=lambda kv: -kv[1][1])],
                columns=["Reason perimeter was left out", "Tiles", "Pixels"])), ""]
        by_country = tile_df.groupby("country").agg(
            tiles=("fire_id", "size"),
            affected=("burn_missing_frac", lambda s: int((s > 0).sum())),
            mean_missing_pct=("burn_missing_frac", lambda s: round(100 * s.mean(), 3)),
        ).reset_index()
        lines += [md_table(by_country), ""]

        older = tile_df["older_scar_frac"]
        own = tile_df["own_frac"]
        lines += ["## Other label checks", "",
                  f"- Tiles containing scars older than {cfg['burn_max_age_days']} days, labelled `clear`: "
                  f"{share(int((older > 0).sum()), n_tiles)}",
                  f"- Those covering at least 5% of the tile: {share(int((older >= .05).sum()), n_tiles)}",
                  f"- Tiles at least 95% covered by their own fire: {share(int((own >= .95).sum()), n_tiles)}",
                  f"- Tiles from a scene outside both search windows: "
                  f"{share(int((tile_df['window'] == 'other').sum()), n_tiles)}",
                  "- Window counts: " + ", ".join(f"{k} {int(v):,}" for k, v in tile_df["window"].value_counts().items()),
                  ""]

    if len(groups):
        counts = groups["group"].value_counts()
        lines += ["## Overlapping tiles", "",
                  f"- Fires with a tile: {len(groups):,}",
                  f"- Fires whose tile overlaps another fire's tile: "
                  f"{share(int((groups['overlapping_fires'] > 0).sum()), len(groups))}",
                  f"- Separate overlap groups: {len(counts):,}",
                  f"- Largest group: {int(counts.max()):,} fires", ""]

    lines += ["## Missing fires", ""]
    present = set(tile_df["fire_id"]) if n_tiles else set()
    scenes = tile_df["fire_id"].value_counts() if n_tiles else pd.Series(dtype=int)
    state_path = Path(cfg["out_dir"]) / "quota_state.json"
    used = {}
    if state_path.exists():
        used = json.loads(state_path.read_text()).get("running_by_country", {})
    size_by_fire = Counter()
    for path, size in files:
        match = re.match(r"acquisitions/([^/]+)/", path)
        if match:
            size_by_fire[match.group(1)] += size
    rows = []
    for country, group in selected.groupby("country"):
        has = group["fire_id"].isin(present)
        smallest = group.loc[has, "area_ha"].min() if has.any() else np.nan
        absent = group[~has]
        budget = cfg.get("country_quota", {}).get(country, 0) * cfg.get("total_budget_gb", 0)
        rows.append({
            "Country": country,
            "Queued": f"{len(group):,}",
            "Two scenes": f"{int(group['fire_id'].map(scenes).eq(2).sum()):,}",
            "One scene": f"{int(group['fire_id'].map(scenes).eq(1).sum()):,}",
            "None": f"{len(absent):,}",
            "Missing, smaller than smallest kept": f"{int((absent['area_ha'] < smallest).sum()):,}",
            "Smallest kept ha": round(float(smallest), 1) if has.any() else "",
            "Median queued ha": round(float(group["area_ha"].median()), 1),
            "Median kept ha": round(float(group.loc[has, "area_ha"].median()), 1) if has.any() else "",
            "Budget GB": round(budget, 2),
            "Used GB (state file)": round(used[country] / 1e9, 2) if country in used else "",
            "Size on Hub GB": round(sum(size_by_fire[f] for f in group["fire_id"]) / 1e9, 2),
        })
    lines += [md_table(pd.DataFrame(rows)), ""]
    lines += ["Fires are processed largest first, so a country that reached its budget loses its",
              "smallest fires. Missing fires smaller than the smallest kept fire suggest that cut.", ""]

    index_path = Path(cfg["out_dir"]) / "processing_index.csv"
    if index_path.exists():
        index = pd.read_csv(index_path)
        index["group"] = index["status"].map(status_group)
        table = index.pivot_table(index="group", columns="country", values="fire_id",
                                  aggfunc="count", fill_value=0).reset_index()
        lines += [f"`{index_path}` (this reflects the most recent run only):", "", md_table(table), ""]
    else:
        lines += [f"`{index_path}` was not found.", ""]
    lines += [f"Quota state file found: {state_path.exists()}"]
    mask_bytes = sum(size for path, size in files if path.endswith("_mask.tif"))
    image_bytes = sum(size for path, size in files if path.endswith("_s2.tif"))
    lines += [f"Download size on the Hub: masks {mask_bytes / 1e9:.2f} GB, images {image_bytes / 1e9:.2f} GB", ""]

    text = "\n".join(lines)
    (out_dir / "summary.md").write_text(text)
    return text


def run_audit(args):
    cfg = yaml.safe_load(Path(args.config).read_text())
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    raw = load_perimeters(cfg, args.effis_extra)
    files = list_dataset_files(args, cfg)
    tiles = parse_tiles(path for path, _ in files)
    print(f"{len(raw):,} perimeters, {len(tiles):,} tiles", flush=True)
    tile_df, footprints, orphans = audit_tiles(tiles, raw, cfg)
    groups = overlap_groups(footprints)
    tile_df.to_csv(out_dir / "tile_audit.csv", index=False)
    groups.to_csv(out_dir / "fire_overlap_groups.csv", index=False)
    print(summarise(raw, cfg, tile_df, groups, orphans, files, out_dir))
    print(f"Wrote {out_dir}/summary.md, tile_audit.csv and fire_overlap_groups.csv")


# ---------------------------------------------------------------------- patch

_PATCH = {}


def _init_patch(raw, cfg, masks_dir, out_dir, write, mark_older):
    _PATCH.update(raw=raw, sindex=raw.sindex, cfg=cfg, masks_dir=masks_dir, out_dir=out_dir,
                  write=write, mark_older=mark_older)


def patch_one(mask_path):
    raw, cfg = _PATCH["raw"], _PATCH["cfg"]
    rel = mask_path.relative_to(_PATCH["masks_dir"])
    match = TILE_RE.search(rel.as_posix())
    scene_time = np.datetime64(datetime.strptime(match["ts"], TS_FORMAT))

    with rasterio.open(mask_path) as src:
        mask = src.read(1)
        profile = src.profile
        colormap = src.colormap(1)
        shape, transform = mask.shape, src.transform
        crs, bounds = src.crs, src.bounds
        tags = src.tags()
    footprint = box(*transform_bounds(crs, "EPSG:4326", *bounds, densify_pts=21))
    local = raw.iloc[_PATCH["sindex"].query(footprint, predicate="intersects")].to_crs(crs)
    age = (scene_time - local["fire_date"].values) / DAY
    fresh_age = (age >= 0) & (age <= cfg["fresh_burn_max_days"])
    old_age = (age > cfg["fresh_burn_max_days"]) & (age <= cfg["burn_max_age_days"])
    was_rasterised = (local["reason"] == "").values

    expect_fresh = to_raster(local.geometry[was_rasterised & fresh_age], shape, transform)
    expect_old = to_raster(local.geometry[was_rasterised & old_age], shape, transform)
    add_fresh = to_raster(local.geometry[~was_rasterised & fresh_age], shape, transform)
    add_old = to_raster(local.geometry[~was_rasterised & old_age], shape, transform)

    # Do the existing burn classes match the export? They should if it's the one used to build them.
    mismatch = int(((mask == MASK_FRESH) & ~expect_fresh).sum()
                   + ((mask == MASK_OLD) & (~expect_old | expect_fresh)).sum()
                   + (np.isin(mask, [MASK_CLEAR, MASK_WATER]) & (expect_fresh | expect_old)).sum())

    # What kind of disagreement is it? Burn from a perimeter outside the build filters is expected for
    # the few tiles of fires now above the size limit, and isn't counted here.
    all_fresh, all_old = expect_fresh | add_fresh, expect_old | add_old
    burn = np.isin(mask, [MASK_FRESH, MASK_OLD])
    no_perimeter = int((burn & ~(all_fresh | all_old)).sum())
    wrong_class = int((((mask == MASK_FRESH) & ~all_fresh & all_old) | ((mask == MASK_OLD) & expect_fresh)).sum())
    not_labelled = int((np.isin(mask, [MASK_CLEAR, MASK_WATER]) & (expect_fresh | expect_old)).sum())

    # Burn classes come from the perimeters alone, so the result doesn't depend on how a mask was first built.
    # Same priority as CombineMaskTask: clear < water < old < fresh < shadow < cloud < nodata.
    # older_burn only ever replaces clear, so it never hides water or a recent burn.
    can_change = ~np.isin(mask, [MASK_CLOUD, MASK_SHADOW, MASK_NODATA])
    patched = mask.copy()
    patched[can_change & burn] = MASK_CLEAR
    patched[can_change & all_old] = MASK_OLD
    patched[can_change & all_fresh] = MASK_FRESH
    if _PATCH["mark_older"]:
        older = to_raster(local.geometry[age > cfg["burn_max_age_days"]], shape, transform)
        patched[older & (patched == MASK_CLEAR)] = MASK_OLDER
    changed = int((patched != mask).sum())

    if _PATCH["write"]:
        target = _PATCH["out_dir"] / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        if changed:
            colormap[MASK_OLDER] = OLDER_COLOUR
            with rasterio.open(target, "w", **profile) as dst:
                dst.write_colormap(1, colormap)
                dst.write(patched, 1)
                dst.update_tags(**tags)
        else:
            shutil.copy2(mask_path, target)

    values, counts = np.unique(patched, return_counts=True)
    before_values, before_counts = np.unique(mask, return_counts=True)
    return {"tile": rel.as_posix(), "changed_px": changed, "verify_mismatch_px": mismatch,
            "no_perimeter_px": no_perimeter, "wrong_class_px": wrong_class, "not_labelled_px": not_labelled,
            "after": dict(zip(values.tolist(), counts.tolist())),
            "before": dict(zip(before_values.tolist(), before_counts.tolist()))}


def run_patch(args):
    cfg = yaml.safe_load(Path(args.config).read_text())
    masks_dir, out_dir = Path(args.masks_dir).resolve(), Path(args.out_dir).resolve()
    if out_dir == masks_dir:
        sys.exit("--out-dir must be different from --masks-dir, so the originals stay untouched.")
    paths = sorted(masks_dir.glob("acquisitions/*/*/*_mask.tif"))
    if args.limit:
        rng = np.random.default_rng(0)
        paths = [paths[i] for i in sorted(rng.choice(len(paths), min(args.limit, len(paths)), replace=False))]
    if not paths:
        sys.exit(f"No masks found under {masks_dir}/acquisitions")
    raw = load_perimeters(cfg, args.effis_extra)
    write = not args.verify_only
    print(f"{len(raw):,} perimeters, {len(paths):,} masks, writing: {write}", flush=True)
    results = []
    with ProcessPoolExecutor(args.workers, initializer=_init_patch,
                             initargs=(raw, cfg, masks_dir, out_dir, write, not args.no_older_burn)) as pool:
        for i, result in enumerate(pool.map(patch_one, paths, chunksize=16), 1):
            results.append(result)
            if i % 1000 == 0:
                print(f"{i:,}/{len(paths):,}", flush=True)

    report_dir = out_dir if write else Path(args.report_dir)
    report_dir.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame(results)
    columns = ["changed_px", "verify_mismatch_px", "no_perimeter_px", "wrong_class_px", "not_labelled_px"]
    frame[["tile", *columns]].to_csv(report_dir / "patch_report.csv", index=False)
    before, after = Counter(), Counter()
    for result in results:
        before.update(result["before"])
        after.update(result["after"])
    names = {0: "clear", 2: "fresh_burn", 3: "old_burn", 4: "cloud", 5: "shadow", 6: "water",
             7: "older_burn", 255: "nodata"}
    valid_before = sum(v for k, v in before.items() if k != MASK_NODATA) or 1
    valid_after = sum(v for k, v in after.items() if k != MASK_NODATA) or 1
    table = pd.DataFrame([{
        "Class": names.get(k, k),
        "Pixels before": f"{before[k]:,}",
        "Pixels after": f"{after[k]:,}",
        "% of valid before": round(100 * before[k] / valid_before, 2) if k != MASK_NODATA else "",
        "% of valid after": round(100 * after[k] / valid_after, 2) if k != MASK_NODATA else "",
    } for k in sorted(set(before) | set(after))])
    text = "\n".join([
        "# burn2scar mask patch", "",
        f"- Masks processed: {len(frame):,}",
        f"- Masks changed: {int((frame['changed_px'] > 0).sum()):,}",
        f"- Pixels changed: {int(frame['changed_px'].sum()):,}",
        f"- Masks that differ from the selected fires: {int((frame['verify_mismatch_px'] > 0).sum()):,}",
        f"- Pixels that differ from the selected fires: {int(frame['verify_mismatch_px'].sum()):,}",
        f"  - Burn with no matching perimeter in the export, now clear: {int(frame['no_perimeter_px'].sum()):,} "
        f"in {int((frame['no_perimeter_px'] > 0).sum()):,} masks",
        f"  - Burn in the wrong class for its age, now corrected: {int(frame['wrong_class_px'].sum()):,} "
        f"in {int((frame['wrong_class_px'] > 0).sum()):,} masks",
        f"  - Selected fires missing from the mask, now added: {int(frame['not_labelled_px'].sum()):,} "
        f"in {int((frame['not_labelled_px'] > 0).sum()):,} masks", "",
        "Burn classes now come from the export alone. The three corrected rows should be small. Anything else in the",
        "difference is burn from perimeters outside the build filters, such as tiles of fires above the size limit.", "",
        md_table(table), ""])
    (report_dir / "patch_summary.md").write_text(text)
    print(text)
    print(f"Wrote {report_dir}/patch_summary.md and patch_report.csv")


# ----------------------------------------------------------------------- main

def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    commands = parser.add_subparsers(dest="command", required=True)

    audit = commands.add_parser("audit", help="measure label problems; changes nothing")
    audit.add_argument("--config", default="config.yaml")
    audit.add_argument("--out", default="audit")
    audit.add_argument("--effis-extra", nargs="*", default=[], help="wider EFFIS exports")
    audit.add_argument("--file-list", help="local list of 'path size' lines instead of asking Hugging Face")
    audit.set_defaults(func=run_audit)

    patch = commands.add_parser("patch", help="write corrected copies of the masks")
    patch.add_argument("--config", default="config.yaml")
    patch.add_argument("--masks-dir", required=True, help="folder that contains acquisitions/")
    patch.add_argument("--out-dir", required=True, help="new folder for the patched masks")
    patch.add_argument("--effis-extra", nargs="*", default=[], help="wider EFFIS exports")
    patch.add_argument("--workers", type=int, default=8)
    patch.add_argument("--limit", type=int, default=0, help="random sample of masks, for a trial run")
    patch.add_argument("--no-older-burn", action="store_true", help="leave scars over burn_max_age_days as clear")
    patch.add_argument("--verify-only", action="store_true", help="check masks against the export; write no masks")
    patch.add_argument("--report-dir", default="audit", help="where --verify-only writes its report")
    patch.set_defaults(func=run_patch)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
