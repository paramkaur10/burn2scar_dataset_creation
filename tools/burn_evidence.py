#!/usr/bin/env python3
"""dNBR burn evidence: an independent check that each labelled fire left a burn signal in the imagery.

For every released fresh-window tile, the differenced Normalized Burn Ratio (Key and Benson, 2006), with
NBR = (B8A - B12) / (B8A + B12) from Sentinel-2 L2A surface reflectance, pre-fire minus post-fire:
  pre-fire   the most recent L2A in the --pre-days before the EFFIS fire date whose scene classification is valid
             over at least --min-valid of the pixels tested, else the best of the --max-dates most recent dates;
  post-fire  the L2A of the tile's own acquisition (same datatake as the released L1C); cloud, shadow, water and
             nodata are taken from the release mask, because Sen2Cor often classes fresh burns as dark or shadow.
Compared inside the fire's own perimeter (pixels labelled fresh_burn) and in a ring 100-300 m outside it (pixels
labelled clear). The L2A comes from the public Earth Search catalogue on AWS: no account, no Copernicus quota.

  python tools/burn_evidence.py --index data/v4/index.csv --masks-root data/v4 --out validation/dnbr --limit 20
  python tools/burn_evidence.py --index data/v4/index.csv --masks-root data/v4 --out validation/dnbr --workers 24
  python tools/burn_evidence.py --index data/v4/index.csv --masks-root data/v4 --out validation/dnbr --summarize --write-index

Flag per tile (Key and Benson put unburned at -0.1 to 0.1):
  burn     median dNBR inside >= 0.10, and >= 0.10 above the ring's median (or no usable ring)
  weak     not burn, but the inside median >= 0.10 or the contrast with the ring >= 0.05
  none     neither
  no_data  fewer than --min-pixels pixels valid on both dates inside the perimeter, or no image
Resumable: rows are appended to <out>/burn_evidence.csv as tiles finish; a rerun skips them and retries errors.
"""
import argparse
import json
import os
import sys
import time
import urllib.error
import urllib.request
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

for _key, _value in {"GDAL_DISABLE_READDIR_ON_OPEN": "EMPTY_DIR", "CPL_VSIL_CURL_ALLOWED_EXTENSIONS": ".tif",
                     "GDAL_HTTP_MAX_RETRY": "5", "GDAL_HTTP_RETRY_DELAY": "3", "GDAL_HTTP_TIMEOUT": "60",
                     "GDAL_HTTP_MULTIRANGE": "YES", "GDAL_HTTP_MERGE_CONSECUTIVE_RANGES": "YES",
                     "VSI_CACHE": "TRUE", "AWS_NO_SIGN_REQUEST": "YES"}.items():
    os.environ.setdefault(_key, _value)

import geopandas as gpd  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import rasterio  # noqa: E402
import yaml  # noqa: E402
from rasterio.enums import Resampling  # noqa: E402
from rasterio.features import rasterize  # noqa: E402
from rasterio.vrt import WarpedVRT  # noqa: E402
from rasterio.warp import transform_bounds  # noqa: E402
from shapely.geometry import box, mapping  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from burn_labels import (MASK_CLEAR, MASK_CLOUD, MASK_FRESH, MASK_NODATA, MASK_OLD, MASK_SHADOW,  # noqa: E402
                         MASK_WATER, load_perimeters)

STAC = "https://earth-search.aws.element84.com/v1/search"
ASSETS = {"nir": ("nir08", "B8A"), "swir": ("swir22", "B12"), "scl": ("scl", "SCL")}
SCL_INVALID = [0, 1, 2, 3, 8, 9, 10, 11]   # no data, defective, dark area, cloud shadow, cloud, cirrus, snow
BURN, CONTRAST, WEAK_CONTRAST = 0.10, 0.10, 0.05
RING_M = (100, 300)
COLUMNS = ["tile_id", "fire_id", "flag", "note", "pre_date", "pre_valid", "pre_pb", "post_pb", "n_in", "n_ring",
           "n_clear", "dnbr_in_median", "dnbr_in_p25", "dnbr_ring_median", "contrast", "in_010", "in_027",
           "ring_010", "clear_010", "nbr_pre_in", "nbr_post_in"]
FLAGS = ["burn", "weak", "none", "no_data", "error"]   # best first: a fire's flag is the best of its tiles
_W = {}


def _init(perimeters, args):
    _W.update(perimeters=perimeters, args=args, masks_root=Path(args.masks_root))


def iso(stamp):
    return stamp.strftime("%Y-%m-%dT%H:%M:%SZ")


def naive(stamp):
    stamp = pd.Timestamp(stamp)
    return stamp.tz_convert(None) if stamp.tzinfo else stamp


def stac_search(geometry, start, end):
    """Items over the tile between two times, newest first."""
    args = _W["args"]
    body = {"collections": [args.collection], "intersects": geometry, "datetime": f"{start}/{end}", "limit": 100,
            "sortby": [{"field": "properties.datetime", "direction": "desc"}]}
    for attempt in range(6):
        request = urllib.request.Request(args.stac, data=json.dumps(body).encode(),
                                         headers={"Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(request, timeout=60) as response:
                items = json.load(response).get("features", [])
            return sorted(items, key=lambda item: item["properties"]["datetime"], reverse=True)
        except urllib.error.HTTPError as error:
            if error.code == 400 and "sortby" in body:   # a catalogue without sorting
                body.pop("sortby")
                continue
            if error.code not in (429, 500, 502, 503, 504) or attempt == 5:
                raise
        except (urllib.error.URLError, TimeoutError):
            if attempt == 5:
                raise
        time.sleep(2 ** attempt)
    return []


def asset(item, kind):
    for key in ASSETS[kind]:
        if key in item["assets"]:
            href = item["assets"][key]["href"]
            return ("/vsis3/" + href[5:] if href.startswith("s3://") else href), item["assets"][key]
    raise KeyError(f"{item['id']}: no {kind} asset")


def read_chip(item, kind, grid, resampling):
    """One band of one item, warped onto the tile's 10 m grid."""
    crs, transform, (height, width) = grid
    with rasterio.open(asset(item, kind)[0]) as src, WarpedVRT(src, crs=crs, transform=transform, width=width,
                                                                height=height, resampling=resampling,
                                                                src_nodata=0, nodata=0) as vrt:
        return vrt.read(1)


def reflectance(dn, item, kind):
    """L2A digital numbers to reflectance. From processing baseline 04.00 (25 January 2022) DN carry a +1000 offset."""
    baseline = item["properties"].get("s2:processing_baseline")
    if item["properties"].get("earthsearch:boa_offset_applied") is True:
        offset = 0.0   # Earth Search already removed it from these files (e.g. the reprocessed 2020-2021 archive)
    elif baseline is not None:
        offset = -1000.0 if float(baseline) >= 4.0 else 0.0
    else:
        band = (asset(item, kind)[1].get("raster:bands") or [{}])[0]
        offset = band.get("offset", 0.0) / band.get("scale", 1e-4)
    out = (dn.astype(np.float32) + offset) / 10000.0
    out[dn == 0] = np.nan
    return out


def nbr(items, grid, source=None):
    """NBR mosaic: each pixel from the item `source` names, or else from the first item with data there."""
    out = np.full(grid[2], np.nan, np.float32)
    for i, item in enumerate(items):
        need = (source == i) if source is not None else np.isnan(out)
        if not need.any():
            continue
        nir = reflectance(read_chip(item, "nir", grid, Resampling.bilinear), item, "nir")
        swir = reflectance(read_chip(item, "swir", grid, Resampling.bilinear), item, "swir")
        with np.errstate(invalid="ignore", divide="ignore"):
            value = (nir - swir) / (nir + swir)
            take = need & np.isfinite(value) & (nir + swir > 0.02)
        out[take] = value[take]
    return out


def prefire(items, grid, area, args):
    """(valid share, date, items, source) for the most recent date valid over --min-valid of `area`, else the best."""
    dates = {}
    for item in items:
        dates.setdefault(item["properties"]["datetime"][:10], []).append(item)
    best, tried = None, 0
    for date in sorted(dates, reverse=True):
        group = sorted(dates[date], key=lambda item: item["properties"].get("eo:cloud_cover", 0))
        if group[0]["properties"].get("eo:cloud_cover", 0) > args.max_scene_cloud:
            continue
        scl, source = np.zeros(grid[2], np.uint8), np.full(grid[2], -1, np.int16)
        for i, item in enumerate(group):
            empty = scl == 0
            if not empty[area].any():
                break
            part = read_chip(item, "scl", grid, Resampling.nearest)
            take = empty & (part > 0)
            scl[take], source[take] = part[take], i
        valid = ~np.isin(scl, SCL_INVALID)
        share = float(valid[area].mean())
        if best is None or share > best[0]:
            best = (share, date, group, np.where(valid, source, -1))
        tried += 1
        if share >= args.min_valid or tried >= args.max_dates:
            break
    return best


def cover(geometry, grid):
    if geometry is None or geometry.is_empty:
        return np.zeros(grid[2], bool)
    return rasterize([(geometry, 1)], out_shape=grid[2], transform=grid[1], fill=0, dtype="uint8").astype(bool)


def median(values):
    return round(float(np.median(values)), 4) if values.size else np.nan


def flag_of(n_in, in_median, contrast, min_pixels):
    if n_in < min_pixels:
        return "no_data"
    no_ring = np.isnan(contrast)
    if in_median >= BURN and (no_ring or contrast >= CONTRAST):
        return "burn"
    if in_median >= BURN or (not no_ring and contrast >= WEAK_CONTRAST):
        return "weak"
    return "none"


def evidence(row):
    args = _W["args"]
    out = {"tile_id": row["tile_id"], "fire_id": row["fire_id"]}
    try:
        with rasterio.open(_W["masks_root"] / row["mask_path"]) as src:
            mask, bounds = src.read(1), src.bounds
            grid = (src.crs, src.transform, (src.height, src.width))
        geometry = _W["perimeters"].get(row["fire_id"])
        if geometry is None:
            return {**out, "flag": "no_data", "note": "no perimeter"}
        perimeter = gpd.GeoSeries([geometry], crs=4326).to_crs(grid[0]).buffer(0).iloc[0]
        inside = cover(perimeter, grid) & np.isin(mask, [MASK_FRESH, MASK_OLD])   # old_burn on old-window tiles
        ring = (cover(perimeter.buffer(RING_M[1]), grid) & ~cover(perimeter.buffer(RING_M[0]), grid)
                & (mask == MASK_CLEAR))
        if inside.sum() < args.min_pixels:
            return {**out, "flag": "no_data", "note": "too few fresh_burn pixels in the perimeter", "n_in": int(inside.sum())}

        footprint = mapping(box(*transform_bounds(grid[0], "EPSG:4326", *bounds, densify_pts=21)))
        scene = naive(row["scene_time"])
        post_items = stac_search(footprint, iso(scene - pd.Timedelta(hours=2)), iso(scene + pd.Timedelta(hours=2)))
        if not post_items:
            return {**out, "flag": "no_data", "note": "no L2A of the acquisition"}
        post_items.sort(key=lambda item: abs(naive(item["properties"]["datetime"]) - scene))
        post = nbr(post_items, grid)
        post[np.isin(mask, [MASK_CLOUD, MASK_SHADOW, MASK_WATER, MASK_NODATA])] = np.nan
        area = (inside | ring) & np.isfinite(post)
        if area.sum() < args.min_pixels:
            return {**out, "flag": "no_data", "note": "post-fire L2A empty over the fire"}

        fire = pd.Timestamp(row["fire_date"])
        pre_items = stac_search(footprint, iso(fire - pd.Timedelta(days=args.pre_days)), iso(fire - pd.Timedelta(seconds=1)))
        best = prefire(pre_items, grid, area, args) if pre_items else None
        if best is None:
            return {**out, "flag": "no_data", "note": "no pre-fire L2A"}
        share, date, group, source = best
        pre = nbr(group, grid, source)

        dnbr = pre - post
        ok = np.isfinite(dnbr)
        d_in, d_ring, d_clear = dnbr[inside & ok], dnbr[ring & ok], dnbr[(mask == MASK_CLEAR) & ok]
        in_median = median(d_in)
        ring_median = median(d_ring) if d_ring.size >= args.min_pixels else np.nan
        contrast = round(in_median - ring_median, 4) if d_in.size and not np.isnan(ring_median) else np.nan
        if args.save_dnbr:
            target = Path(args.out) / "dnbr" / Path(row["mask_path"]).name.replace("_mask.tif", "_dnbr.tif")
            target.parent.mkdir(parents=True, exist_ok=True)
            scaled = np.where(ok, np.clip(np.round(dnbr * 1000), -32767, 32767), -32768).astype(np.int16)
            with rasterio.open(target, "w", driver="GTiff", width=grid[2][1], height=grid[2][0], count=1,
                               dtype="int16", crs=grid[0], transform=grid[1], nodata=-32768, compress="deflate") as dst:
                dst.write(scaled, 1)
                dst.update_tags(scale="0.001", pre_date=date, description="dNBR x 1000, L2A B8A and B12")
        return {**out, "flag": flag_of(d_in.size, in_median, contrast, args.min_pixels), "note": "",
                "pre_date": date, "pre_valid": round(share, 3),
                "pre_pb": group[0]["properties"].get("s2:processing_baseline"),
                "post_pb": post_items[0]["properties"].get("s2:processing_baseline"),
                "n_in": int(d_in.size), "n_ring": int(d_ring.size), "n_clear": int(d_clear.size),
                "dnbr_in_median": in_median,
                "dnbr_in_p25": round(float(np.percentile(d_in, 25)), 4) if d_in.size else np.nan,
                "dnbr_ring_median": ring_median, "contrast": contrast,
                "in_010": int((d_in >= 0.10).sum()), "in_027": int((d_in >= 0.27).sum()),
                "ring_010": int((d_ring >= 0.10).sum()), "clear_010": int((d_clear >= 0.10).sum()),
                "nbr_pre_in": median(pre[inside & ok]), "nbr_post_in": median(post[inside & ok])}
    except Exception as error:   # one bad tile must not stop the run; it is retried next time
        return {**out, "flag": "error", "note": f"{type(error).__name__}: {error}"[:300]}


def run(todo, perimeters, args, rows_path):
    jobs = todo[["tile_id", "fire_id", "mask_path", "scene_time", "fire_date"]].to_dict("records")
    start, counts = time.time(), {}
    with ProcessPoolExecutor(args.workers, initializer=_init, initargs=(perimeters, args)) as pool:
        futures = [pool.submit(evidence, job) for job in jobs]
        for i, future in enumerate(as_completed(futures), 1):
            row = future.result()
            pd.DataFrame([row]).reindex(columns=COLUMNS).to_csv(rows_path, mode="a", header=not rows_path.exists(),
                                                                 index=False)
            counts[row["flag"]] = counts.get(row["flag"], 0) + 1
            if i % 100 == 0 or i == len(jobs):
                print(f"{i:,}/{len(jobs):,}  {counts}  {3600 * i / (time.time() - start):,.0f} tiles/h", flush=True)


def md(frame):
    head = "| " + " | ".join(map(str, frame.columns)) + " |"
    rule = "|" + "---|" * len(frame.columns)
    body = ["| " + " | ".join("" if pd.isna(v) else str(v) for v in row) + " |" for row in frame.itertuples(index=False)]
    return "\n".join([head, rule, *body])


def group_table(rows, by):
    rows = rows[rows[by].notna()]
    assessed = rows[rows["flag"].isin(["burn", "weak", "none"])]
    every, done = rows.groupby(by, observed=True), assessed.groupby(by, observed=True)
    table = pd.DataFrame({"Tiles": every.size(), "Assessed": done.size()}).fillna(0).astype(int)
    for flag in ["burn", "weak", "none"]:
        table[f"{flag} %"] = (100 * done["flag"].apply(lambda s, f=flag: (s == f).mean())).round(1)
    table["Median dNBR inside"] = done["dnbr_in_median"].median().round(3)
    table["Median dNBR ring"] = done["dnbr_ring_median"].median().round(3)
    table["Inside pixels >= 0.10 %"] = (100 * done["in_010"].sum() / done["n_in"].sum()).round(1)
    return table.reset_index().rename(columns={by: by.replace("_", " ").capitalize()})


def summarize(rows_path, index, out, write_index, index_path):
    if not rows_path.exists():
        raise SystemExit(f"No rows yet in {rows_path}")
    rows = pd.read_csv(rows_path).drop_duplicates("tile_id", keep="last")
    extra = [c for c in ["country", "area_ha", "fire_date", "age_days", "land_cover", "map_source"] if c in index]
    rows = rows.merge(index[["tile_id", *extra]], on="tile_id", how="left")
    fire_date = pd.to_datetime(rows["fire_date"], errors="coerce")
    rows["size"] = pd.cut(rows["area_ha"], [0, 50, 100, 500, 1000, np.inf], right=False,
                          labels=["< 50 ha", "50-100 ha", "100-500 ha", "500-1000 ha", ">= 1000 ha"])
    rows["season"] = fire_date.dt.month.map({12: "1 Dec-Feb", 1: "1 Dec-Feb", 2: "1 Dec-Feb", 3: "2 Mar-May",
                                             4: "2 Mar-May", 5: "2 Mar-May", 6: "3 Jun-Aug", 7: "3 Jun-Aug",
                                             8: "3 Jun-Aug", 9: "4 Sep-Nov", 10: "4 Sep-Nov", 11: "4 Sep-Nov"})
    rows["fire_year"] = fire_date.dt.year
    rows["days_after_fire"] = pd.cut(rows["age_days"], [0, 10, 20, 40, 61, np.inf], right=False,
                                     labels=["0-9", "10-19", "20-39", "40-60", "over 60"])
    assessed = rows[rows["flag"].isin(["burn", "weak", "none"])]
    rank = {flag: i for i, flag in enumerate(FLAGS)}
    fire_flag = rows.assign(rank=rows["flag"].map(rank)).sort_values("rank").drop_duplicates("fire_id").set_index("fire_id")["flag"]

    def share(column, total):
        return f"{100 * assessed[column].sum() / max(assessed[total].sum(), 1):.1f} %"

    lines = ["# Burn evidence from dNBR (Sentinel-2 L2A, B8A and B12)", "",
             f"- Tiles: {len(rows):,}; fires: {rows['fire_id'].nunique():,}",
             "- Tiles by flag: " + ", ".join(f"{f} {int((rows['flag'] == f).sum()):,}" for f in FLAGS),
             "- Fires by their best tile: " + ", ".join(f"{f} {int((fire_flag == f).sum()):,}" for f in FLAGS),
             f"- Assessed tiles: burn {100 * (assessed['flag'] == 'burn').mean():.1f} %, "
             f"weak {100 * (assessed['flag'] == 'weak').mean():.1f} %, none {100 * (assessed['flag'] == 'none').mean():.1f} %",
             f"- Pixels inside the perimeters (labelled fresh_burn) with dNBR >= 0.10: {share('in_010', 'n_in')}; "
             f">= 0.27: {share('in_027', 'n_in')}",
             f"- Pixels in the 100-300 m rings (labelled clear) with dNBR >= 0.10: {share('ring_010', 'n_ring')}",
             f"- All clear pixels of the tiles with dNBR >= 0.10 (unmapped burns, but also harvest and other change): "
             f"{share('clear_010', 'n_clear')}",
             f"- Median dNBR inside / ring / contrast: {assessed['dnbr_in_median'].median():.3f} / "
             f"{assessed['dnbr_ring_median'].median():.3f} / {assessed['contrast'].median():.3f}",
             f"- Check on the reflectance offset: median pre-fire NBR inside {assessed['nbr_pre_in'].median():.3f}, "
             f"post-fire {assessed['nbr_post_in'].median():.3f} (vegetation sits near 0.2-0.6 before a fire)",
             f"- Median valid share of the pre-fire image: {assessed['pre_valid'].median():.2f}", ""]
    notes = rows.loc[rows["flag"].isin(["no_data", "error"]), "note"].fillna("").astype(str).str.split(":").str[0].value_counts()
    if len(notes):
        lines += ["## Why tiles were not assessed", "", md(notes.rename_axis("Reason").reset_index(name="Tiles")), ""]
    for by in ["country", "size", "season", "fire_year", "days_after_fire", "land_cover", "map_source"]:
        if by in rows and rows[by].notna().any():
            lines += [f"## By {by.replace('_', ' ')}", "", md(group_table(rows, by)), ""]
    text = "\n".join(lines)
    (out / "burn_evidence.md").write_text(text)
    print(text)
    if write_index:
        tiles = rows.set_index("tile_id")
        index["dnbr_flag"] = index["tile_id"].map(tiles["flag"])
        index["dnbr_in_median"] = index["tile_id"].map(tiles["dnbr_in_median"])
        index["dnbr_contrast"] = index["tile_id"].map(tiles["contrast"])
        index["fire_dnbr_flag"] = index["fire_id"].map(fire_flag)
        tmp = Path(f"{index_path}.tmp")
        index.to_csv(tmp, index=False)
        os.replace(tmp, index_path)
        print(f"Wrote dnbr_flag, dnbr_in_median, dnbr_contrast and fire_dnbr_flag to {index_path}")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--index", default="data/v4/index.csv")
    parser.add_argument("--masks-root", default="data/v4")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--effis-extra", nargs="*", default=[], help="further exports; the own perimeters are in the build export")
    parser.add_argument("--out", default="validation/dnbr")
    parser.add_argument("--window", default="fresh", choices=["fresh", "old", "all"])
    parser.add_argument("--workers", type=int, default=16, help="parallel tiles; each reads a few small windows over HTTPS")
    parser.add_argument("--limit", type=int, default=0, help="a random sample of this many tiles, for a trial")
    parser.add_argument("--tiles-from", default="", help="any CSV with a tile_id column, e.g. review2/points.csv; overrides --window")
    parser.add_argument("--pre-days", type=int, default=90)
    parser.add_argument("--min-valid", type=float, default=0.8)
    parser.add_argument("--max-dates", type=int, default=6)
    parser.add_argument("--max-scene-cloud", type=float, default=80, help="skip pre-fire dates whose granules are all cloudier")
    parser.add_argument("--min-pixels", type=int, default=25, help="0.25 ha at 10 m")
    parser.add_argument("--collection", default="sentinel-2-l2a", help="or sentinel-2-c1-l2a")
    parser.add_argument("--stac", default=STAC)
    parser.add_argument("--save-dnbr", action="store_true", help="also write each tile's dNBR (int16, x1000) for figures")
    parser.add_argument("--summarize", action="store_true", help="only summarize the rows already written")
    parser.add_argument("--write-index", action="store_true", help="add the flags to --index (all tiles of a fire get fire_dnbr_flag)")
    args = parser.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    rows_path = out / "burn_evidence.csv"
    index = pd.read_csv(args.index, low_memory=False)
    if not args.summarize:
        released = index[index["split"].notna()]
        if args.tiles_from:
            todo = released[released["tile_id"].isin(set(pd.read_csv(args.tiles_from, usecols=["tile_id"])["tile_id"]))]
        else:
            todo = released if args.window == "all" else released[released["window"] == args.window]
        done = set()
        if rows_path.exists():
            previous = pd.read_csv(rows_path).drop_duplicates("tile_id", keep="last")
            done = set(previous.loc[previous["flag"] != "error", "tile_id"])
        todo = todo[~todo["tile_id"].isin(done)]
        if args.limit:
            todo = todo.sample(min(args.limit, len(todo)), random_state=0)
        print(f"{len(todo):,} tiles to do, {len(done):,} done before", flush=True)
        if len(todo):
            raw = load_perimeters(yaml.safe_load(Path(args.config).read_text()), args.effis_extra)
            raw = raw[raw["fire_id"].isin(set(todo["fire_id"]))].drop_duplicates("fire_id")
            run(todo, dict(zip(raw["fire_id"], raw.geometry)), args, rows_path)
    summarize(rows_path, index, out, args.write_index, args.index)


if __name__ == "__main__":
    main()
