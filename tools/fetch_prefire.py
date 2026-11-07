#!/usr/bin/env python3
"""Download one pre-fire Sentinel-2 L1C image per fire, on the same grid as its tile.

Uses the same collection, upsampling and request code as the extraction pipeline, except
that it asks for 16-bit values instead of floats, which costs half the processing units:
13 bands at 256 x 256 cost about 1.08 PU per image. Finished fires are skipped on restart.

  Check it first. This fetches 3 images you already have and compares them with your files:
  python tools/fetch_prefire.py --index data/v2/index.csv --images-root data/v1 --verify 3

  Then one of these:
  python tools/fetch_prefire.py --index data/v2/index.csv --points review/points.csv --out data/prefire
  python tools/fetch_prefire.py --index data/v2/index.csv --all --out data/prefire --max-pu 5000

Needs SH_CLIENT_ID and SH_CLIENT_SECRET in .env, as the pipeline does.
"""
import argparse
import csv
import datetime as dt
import os
import sys
import threading
import warnings
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio
from rasterio.transform import from_bounds

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

warnings.filterwarnings("ignore", message=".*timestamp.*")
PU_PER_IMAGE = (256 * 256) / (512 * 512) * 13 / 3  # Area factor times band factor, for 16-bit output.
FIELDS = ["fire_id", "status", "scene_date", "cloud_cover", "valid_frac", "bright_frac", "candidates_fetched", "path"]


def fetch_scene(row, date, sh, cache_folder):
    """Bands as uint16 (reflectance x 10000) plus dataMask for one date, or None if there's no scene."""
    from eolearn.core import FeatureType
    from eolearn.io import SentinelHubInputTask
    from sentinelhub import BBox, CRS

    from src.s2_pipeline import EXPORT_BANDS, S2L1C_COLLECTION

    bbox = BBox((row["minx"], row["miny"], row["maxx"], row["maxy"]), crs=CRS(int(row["epsg"])))
    eop = SentinelHubInputTask(
        data_collection=S2L1C_COLLECTION, resolution=10,
        bands_feature=(FeatureType.DATA, "BANDS"), additional_data=[(FeatureType.MASK, "dataMask")],
        bands=EXPORT_BANDS, bands_dtype=np.uint16, aux_request_args={"processing": {"upsampling": "BICUBIC"}},
        config=sh, cache_folder=cache_folder, time_difference=dt.timedelta(minutes=180),
    )(bbox=bbox, time_interval=(date, date))
    if len(eop.timestamp) == 0:
        return None
    return eop.data["BANDS"][0], eop.mask["dataMask"][0, ..., 0].astype(bool)


def search_candidates(row, days_min, days_max, max_cloud, limit, sh):
    from sentinelhub import BBox, CRS, SentinelHubCatalog

    from src.s2_pipeline import S2L1C_COLLECTION

    fire_date = pd.Timestamp(row["fire_date"])
    bbox = BBox((row["minx"], row["miny"], row["maxx"], row["maxy"]), crs=CRS(int(row["epsg"])))
    results = list(SentinelHubCatalog(sh).search(
        collection=S2L1C_COLLECTION, bbox=bbox,
        time=((fire_date - pd.Timedelta(days=days_max)).date().isoformat(),
              (fire_date - pd.Timedelta(days=days_min)).date().isoformat()),
        filter=f"eo:cloud_cover < {max_cloud:.0f}",
        fields={"include": ["properties.datetime", "properties.eo:cloud_cover"], "exclude": []}))
    results.sort(key=lambda r: r["properties"]["eo:cloud_cover"])
    seen, picked = set(), []
    for result in results:
        date = result["properties"]["datetime"].split("T")[0]
        if date not in seen:
            seen.add(date)
            picked.append((date, result["properties"]["eo:cloud_cover"]))
    return picked[:limit]


def write_image(path, bands_hwc, valid, row):
    from src.s2_pipeline import EXPORT_BANDS
    height, width, count = bands_hwc.shape
    data = bands_hwc.astype(np.uint16).copy()
    data[~valid] = 0
    profile = dict(driver="GTiff", height=height, width=width, count=count, dtype="uint16", crs=f"EPSG:{int(row['epsg'])}",
                   compress="deflate", predictor=2, tiled=True, blockxsize=512, blockysize=512, nodata=0,
                   transform=from_bounds(row["minx"], row["miny"], row["maxx"], row["maxy"], width, height))
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(".tmp.tif")
    with rasterio.open(temp, "w", **profile) as dst:
        for band in range(count):
            dst.write(data[..., band], band + 1)
            dst.set_band_description(band + 1, EXPORT_BANDS[band])
    temp.replace(path)


def judge(bands_hwc, valid):
    """Share of valid pixels, and of very bright ones, which are mostly cloud."""
    from src.s2_pipeline import EXPORT_BANDS
    blue = bands_hwc[..., EXPORT_BANDS.index("B02")]
    return float(valid.mean()), float(((blue > 2500) & valid).sum() / max(int(valid.sum()), 1))


def process_fire(row, args, sh):
    candidates = search_candidates(row, args.days_min, args.days_max, args.max_cloud, args.max_candidates, sh)
    best, fetched = None, 0
    for date, cloud in candidates:
        scene = fetch_scene(row, date, sh, args.cache_folder)
        fetched += 1
        if scene is None:
            continue
        bands, valid = scene
        valid_frac, bright = judge(bands, valid)
        score = bright + (1 - valid_frac)
        if best is None or score < best["score"]:
            best = dict(score=score, bands=bands, valid=valid, date=date, cloud=cloud, valid_frac=valid_frac, bright=bright)
        if valid_frac >= args.min_valid and bright <= args.max_bright:
            break
    record = dict(fire_id=row["fire_id"], status="no_scene", scene_date="", cloud_cover="", valid_frac="", bright_frac="",
                  candidates_fetched=fetched, path="")
    if best is not None:
        path = Path(args.out) / "acquisitions" / row["fire_id"] / f"{row['fire_id']}_{best['date']}_prefire_s2.tif"
        write_image(path, best["bands"], best["valid"], row)
        passed = best["valid_frac"] >= args.min_valid and best["bright"] <= args.max_bright
        record.update(status="ok" if passed else "kept_best_available", scene_date=best["date"], cloud_cover=best["cloud"],
                      valid_frac=round(best["valid_frac"], 4), bright_frac=round(best["bright"], 4),
                      path=path.relative_to(args.out).as_posix())
    return record


def make_config():
    from dotenv import load_dotenv

    from src.s2_pipeline import make_sh_config
    load_dotenv()
    return make_sh_config(os.environ["SH_CLIENT_ID"], os.environ["SH_CLIENT_SECRET"])


def run_verify(index, args, sh):
    """Fetch tiles you already have. If the two agree, this script reads Sentinel Hub the way the pipeline did."""
    # Mostly recent scenes: products from 2022 onwards are where a scaling mistake would show.
    recent = index[index["scene_year"] >= 2023]
    rows = pd.concat([recent.sample(min(max(args.verify - 1, 1), len(recent)), random_state=0),
                      index[index["scene_year"] < 2022].sample(min(1, int((index["scene_year"] < 2022).sum())), random_state=0)])
    all_close = True
    for _, row in rows.iterrows():
        date = str(row["scene_time"])[:10]
        scene = fetch_scene(row, date, sh, args.cache_folder)
        if scene is None:
            print(f"{row['tile_id']}: no scene came back for {date}")
            all_close = False
            continue
        bands, valid = scene
        with rasterio.open(Path(args.images_root) / row["image_path"]) as src:
            stored = np.moveaxis(src.read(), 0, -1)
        difference = np.abs(bands.astype(np.int32) - stored.astype(np.int32))[valid & (stored != 0).all(-1)]
        close = float((difference <= 2).mean())
        print(f"{row['tile_id']}: {100 * close:.2f}% of values within 2 of your file, largest gap {int(difference.max())}")
        all_close &= close > .99
    print("Verified: same values as the published images." if all_close else "These don't match. Don't fetch more; send me this output.")
    return all_close


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--index", default="data/v2/index.csv")
    parser.add_argument("--points", help="points.csv from review_points.py: fetch only those fires")
    parser.add_argument("--all", action="store_true", help="fetch every fire")
    parser.add_argument("--verify", type=int, default=0, help="fetch this many existing tiles and compare, then stop")
    parser.add_argument("--images-root", default="data/v1")
    parser.add_argument("--out", default="data/prefire")
    parser.add_argument("--days-min", type=int, default=2, help="closest allowed days before the fire")
    parser.add_argument("--days-max", type=int, default=60, help="furthest allowed days before the fire")
    parser.add_argument("--max-cloud", type=float, default=30.0, help="largest scene cloud cover to consider, percent")
    parser.add_argument("--max-candidates", type=int, default=3, help="most images to try per fire")
    parser.add_argument("--min-valid", type=float, default=.99, help="keep looking if less of the tile has data")
    parser.add_argument("--max-bright", type=float, default=.05, help="keep looking if more of the tile looks like cloud")
    parser.add_argument("--max-pu", type=float, default=0, help="stop after spending about this many PU; 0 means no limit")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--limit", type=int, default=0, help="only this many fires, for a quick trial")
    parser.add_argument("--cache-folder", default=None)
    args = parser.parse_args()

    index = pd.read_csv(args.index, low_memory=False)
    index = index[index["country"].notna()]
    sh = make_config()
    if args.verify:
        sys.exit(0 if run_verify(index, args, sh) else 1)
    if bool(args.points) == bool(args.all):
        raise SystemExit("Choose one: --points review/points.csv, --all, or --verify N")

    fires = index.drop_duplicates("fire_id")
    if args.points:
        wanted = set(pd.read_csv(args.points, low_memory=False)["fire_id"])
        fires = fires[fires["fire_id"].isin(wanted)]
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    log_path = out / "prefire_index.csv"
    finished = set(pd.read_csv(log_path)["fire_id"]) if log_path.exists() else set()
    pending = fires[~fires["fire_id"].isin(finished)]
    if args.limit:
        pending = pending.head(args.limit)
    if not log_path.exists():
        with open(log_path, "w", newline="") as handle:
            csv.DictWriter(handle, FIELDS).writeheader()
    print(f"{len(pending):,} fires to fetch, {len(finished):,} already finished. At most about "
          f"{len(pending) * args.max_candidates * PU_PER_IMAGE:,.0f} PU, usually close to {len(pending) * PU_PER_IMAGE:,.0f}.", flush=True)

    lock = threading.Lock()
    spent = [0.0]
    stop = threading.Event()
    counts = {}

    def work(row):
        if stop.is_set():
            return None
        return process_fire(row, args, sh)

    with ThreadPoolExecutor(args.workers) as pool:
        futures = {pool.submit(work, row): row["fire_id"] for _, row in pending.iterrows()}
        for done, future in enumerate(as_completed(futures), 1):
            try:
                record = future.result()
            except Exception as exc:  # A failed fire stays unfinished and is tried again on restart.
                print(f"{futures[future]}: failed: {str(exc)[:200]}", flush=True)
                counts["failed"] = counts.get("failed", 0) + 1
                continue
            if record is None:
                continue
            with lock, open(log_path, "a", newline="") as handle:
                csv.DictWriter(handle, FIELDS).writerow(record)
            counts[record["status"]] = counts.get(record["status"], 0) + 1
            spent[0] += record["candidates_fetched"] * PU_PER_IMAGE
            if args.max_pu and spent[0] >= args.max_pu and not stop.is_set():
                print(f"Reached about {spent[0]:,.0f} PU. Stopping; run again to continue.", flush=True)
                stop.set()
            if done % 100 == 0:
                print(f"{done:,}/{len(pending):,} {counts}, about {spent[0]:,.0f} PU", flush=True)
    print(f"Done: {counts}, about {spent[0]:,.0f} PU spent. Log: {log_path}")


if __name__ == "__main__":
    main()
