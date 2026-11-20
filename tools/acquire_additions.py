#!/usr/bin/env python3
"""New tiles for burn2scar 5.0, from the same Sentinel-2 source and with the same request code as the release.

  negative   burn-free tiles from the same acquisitions as released tiles, 5 to 15 km away, where no EFFIS perimeter
             since 2018 touches the tile, so false alarms over unburned land can be measured
  boundary   a third scene of released fires, 61 to 90 days after the fire for half of them and 91 to 120 for the
             other half, so the 90-day line between fresh_burn and old_burn is sampled on both sides
  large      tiles on the edge of fires above 2000 ha, which no released tile is centred on, 20 to 80 % inside the
             perimeter
  firstlook  the first acquisition 0 to 5 days after the fire, with cloud and smoke allowed

  python tools/acquire_additions.py plan                       # picks the targets and prices them; sends no requests
  python tools/acquire_additions.py run --max-requests 9000    # downloads; stops at the budget; safe to restart
  python tools/acquire_additions.py status

Budget. Copernicus General users get 10,000 Sentinel Hub requests and 10,000 PU a month, reset on the 1st. Every
catalogue search, scene-classification (L2A) request and L1C request counts as one request here, and the budget is
counted over the current calendar month, from the log, so restarts never overspend. A tile costs about 1.17 PU: 1.08
for the 13 L1C bands as 16-bit values (as fetch_prefire.py asks, which gives the release's values at half the PU of the
pipeline's floats) and 0.08 for the scene classification. Requests are the limit: about 2 for a negative tile, 3 to 7
for a tile whose scene has to be searched.

Everything runs on the CPU. CUDA_VISIBLE_DEVICES is emptied before OmniCloudMask loads, so booked GPUs are untouched.

Tiles go to data/additions/raw/acquisitions/<tile>/<timestamp>/ as <tile>_<timestamp>_s2.tif and _mask.tif, with the
masks built as the pipeline builds them (src/s2_pipeline.py). finish_additions.py then applies the release's label rules
(the winter water check, burn_labels.py patch, make_age.py) and writes the index. Boundary and first-look tiles keep the
fire's own tile grid and folder name, so they sit beside the fire's released tiles; negative tiles are named neg_NNNN,
and large-fire tiles <fire_id>-eN.

Needs SH_CLIENT_ID and SH_CLIENT_SECRET in .env, as the pipeline does.
"""
import argparse
import csv
import math
import os
import sys
import threading
import warnings
import zlib
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta, timezone
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))
warnings.filterwarnings("ignore", message=".*timestamp.*")
warnings.filterwarnings("ignore", message=".*shapely.ops.transform.*")
warnings.filterwarnings("ignore", message=".*geographic CRS.*")
warnings.filterwarnings("ignore", message=".*has no geotransform.*")

TILE_M = 2560
PIXELS = 256
PU_L1C = (PIXELS * PIXELS) / (512 * 512) * 13 / 3
PU_SCL = (PIXELS * PIXELS) / (512 * 512) * 1 / 3
TS_FORMAT = "%Y-%m-%dT%H-%M-%S"
BANDS = ["B01", "B02", "B03", "B04", "B05", "B06", "B07", "B08", "B8A", "B09", "B10", "B11", "B12"]
B03, B04, B08 = BANDS.index("B03"), BANDS.index("B04"), BANDS.index("B08")
CLEAR, FRESH, OLD, CLOUD, SHADOW, WATER, OLDER, NODATA = 0, 2, 3, 4, 5, 6, 7, 255
COLOURS = {0: (200, 200, 200, 255), 2: (220, 60, 20, 255), 3: (150, 40, 20, 255), 4: (255, 255, 255, 255),
           5: (60, 60, 90, 255), 6: (30, 90, 200, 255), 7: (120, 95, 70, 255), 255: (0, 0, 0, 0)}

# Scene search per part: window in days after the fire, scene cloud filter, candidates tried, and how one is chosen.
PARTS = {
    "negative": dict(window=None, maxcc=None, candidates=1, rule="same_scene", expected=1.0),
    "boundary": dict(window=(61, 120), maxcc=.70, candidates=3, rule="least_obscured", expected=1.6),
    "large": dict(window=(5, 60), maxcc=.70, candidates=3, rule="least_obscured", expected=1.6),
    "firstlook": dict(window=(0, 5), maxcc=None, candidates=2, rule="earliest", expected=1.15),
}
TARGET_FIELDS = ["target_id", "part", "tile_name", "source_tile_id", "fire_id", "country", "fire_date", "area_ha",
                 "scene_date", "day_min", "day_max", "epsg", "minx", "miny", "maxx", "maxy", "lon", "lat", "block", "split",
                 "offset_km", "edge_burn_frac"]
LOG_FIELDS = ["target_id", "part", "tile_name", "status", "scene_time", "candidates_fetched", "requests", "pu",
              "valid_frac", "obscured_frac", "rule_pass", "image_path", "mask_path", "utc_time", "error"]
FINISHED = {"ok", "kept_best_available", "no_scene", "no_data"}
REFUSED = ("invalid_client", "invalid_grant", "Unauthorized", "unauthorized_client")   # the login, not the data


# ---------------------------------------------------------------- geometry

@lru_cache(maxsize=None)
def transformer(source, target):
    from pyproj import Transformer
    return Transformer.from_crs(source, target, always_xy=True)


def utm_tile(lon, lat, size=TILE_M):
    """The pipeline's tile grid (get_utm_bbox in src/s2_pipeline.py): the centre's UTM zone, snapped to 10 m."""
    zone = int((lon + 180) / 6) % 60 + 1
    epsg = (32600 if lat >= 0 else 32700) + zone
    east, north = transformer(4326, epsg).transform(lon, lat)
    east, north = 10 * int(east / 10), 10 * int(north / 10)
    return epsg, (east - size // 2, north - size // 2, east + size // 2, north + size // 2)


def footprint(epsg, bounds):
    """The tile as a lon/lat polygon, densified so its edges follow the projection."""
    from shapely.geometry import Polygon
    minx, miny, maxx, maxy = bounds
    steps = np.linspace(0, 1, 11)
    xs = np.concatenate([minx + (maxx - minx) * steps, np.full(11, maxx), maxx - (maxx - minx) * steps, np.full(11, minx)])
    ys = np.concatenate([np.full(11, miny), miny + (maxy - miny) * steps, np.full(11, maxy), maxy - (maxy - miny) * steps])
    lon, lat = transformer(int(epsg), 4326).transform(xs, ys)
    return Polygon(zip(lon, lat))


def block_of(lat, lon, block_deg=.5):
    """The spatial block of build_index.py."""
    return f"{int(np.floor(lat / block_deg))}_{int(np.floor(lon / block_deg))}"


def split_for_block(block, block_split):
    """The block's split in the release; a block with no released tile is assigned 60/20/20 by a fixed hash of its name,
    which can't leak, since no released tile shares its ground."""
    if block in block_split:
        return block_split[block]
    draw = zlib.crc32(block.encode()) % 10
    return "train" if draw < 6 else "val" if draw < 8 else "test"


# ---------------------------------------------------------------- targets

class Footprints:
    """Released and chosen tiles, so a new tile never shares ground with another."""

    def __init__(self, polygons):
        import geopandas as gpd
        self.released = gpd.GeoSeries(list(polygons), crs=4326)
        self.chosen, self._series, self._size = [], None, -1

    def clear_of(self, polygon):
        import geopandas as gpd
        if len(self.released.sindex.query(polygon, predicate="intersects")):
            return False
        if self.chosen and self._size != len(self.chosen):
            self._series, self._size = gpd.GeoSeries(self.chosen, crs=4326), len(self.chosen)
        return not self.chosen or len(self._series.sindex.query(polygon, predicate="intersects")) == 0

    def add(self, polygon):
        self.chosen.append(polygon)


def released_footprints(index):
    from shapely.geometry import box
    return Footprints(box(r.lon_min, r.lat_min, r.lon_max, r.lat_max) for r in index.itertuples())


def pick_negatives(index, perimeters, land, taken, n, rng, block_split):
    """Up to n burn-free tiles, the same number for each country and window, each 5-15 km from a released tile and on
    the date of that tile's scene: no EFFIS perimeter of any date touches them, they are at least 90 % land, and they
    share no ground with any released or chosen tile."""
    from shapely.prepared import prep
    land_prepared = prep(land)
    strata = sorted(index.groupby(["country", "window"]).groups.items())
    per_stratum = math.ceil(n / max(len(strata), 1))
    rows = []
    for (country, window), labels in strata:
        order = rng.permutation(np.asarray(labels))
        got = 0
        for label in order[: 6 * per_stratum]:
            if got >= per_stratum or len(rows) >= n:
                break
            source = index.loc[label]
            cx, cy = (source["minx"] + source["maxx"]) / 2, (source["miny"] + source["maxy"]) / 2
            for _ in range(25):
                distance, bearing = rng.uniform(5000, 15000), rng.uniform(0, 2 * math.pi)
                lon, lat = transformer(int(source["epsg"]), 4326).transform(cx + distance * math.cos(bearing),
                                                                             cy + distance * math.sin(bearing))
                epsg, bounds = utm_tile(lon, lat)
                shape = footprint(epsg, bounds)
                if len(perimeters.sindex.query(shape, predicate="intersects")):
                    continue
                if not land_prepared.contains(shape) and shape.intersection(land).area / shape.area < .9:
                    continue
                if not taken.clear_of(shape):
                    continue
                taken.add(shape)
                block = block_of(lat, lon)
                rows.append(dict(part="negative", tile_name=f"neg_{len(rows) + 1:04d}", source_tile_id=source["tile_id"],
                                 fire_id="", country=country, fire_date="", area_ha=np.nan,
                                 scene_date=str(source["scene_time"])[:10], day_min=np.nan, day_max=np.nan, epsg=epsg,
                                 minx=bounds[0], miny=bounds[1], maxx=bounds[2], maxy=bounds[3], lon=lon, lat=lat,
                                 block=block, split=split_for_block(block, block_split),
                                 offset_km=round(distance / 1000, 2), edge_burn_frac=np.nan))
                got += 1
                break
    return rows


def pick_fire_scenes(index, part, n, rng, latest_fire_date):
    """n released fires, the same number per country, on the grid of their fresh-window tile. Boundary targets alternate
    between the two halves of the part's window, 61-90 and 91-120 days, so both sides of the 90-day line are sampled."""
    fresh = index[(index["window"] == "fresh") & (pd.to_datetime(index["fire_date"]) <= latest_fire_date)]
    fresh = fresh.drop_duplicates("fire_id")
    countries = sorted(fresh["country"].unique())
    per_country = math.ceil(n / max(len(countries), 1))
    windows = [(61, 90), (91, 120)] if part == "boundary" else [PARTS[part]["window"]]
    rows = []
    for country in countries:
        group = fresh[fresh["country"] == country]
        chosen = group.loc[rng.permutation(group.index.to_numpy())[:per_country]]
        for _, tile in chosen.iterrows():
            if len(rows) >= n:
                break
            day_min, day_max = windows[len(rows) % len(windows)]
            rows.append(dict(part=part, tile_name=tile["fire_id"], source_tile_id=tile["tile_id"], fire_id=tile["fire_id"],
                             country=country, fire_date=str(tile["fire_date"])[:10], area_ha=tile.get("area_ha", np.nan),
                             scene_date="", day_min=day_min, day_max=day_max, epsg=int(tile["epsg"]), minx=tile["minx"],
                             miny=tile["miny"], maxx=tile["maxx"], maxy=tile["maxy"], lon=tile["lon"], lat=tile["lat"],
                             block=tile.get("block", ""), split=tile["split"], offset_km=np.nan, edge_burn_frac=np.nan))
    return rows


def pick_large(fires, taken, per_fire, max_tiles, rng, block_split):
    """Tiles centred on the perimeter of each fire above the size limit, 20-80 % inside it, not sharing ground with any
    released or chosen tile. Fires are taken in random order, so a cap keeps every country in proportion."""
    from shapely.geometry import box
    rows = []
    for _, fire in fires.loc[rng.permutation(fires.index.to_numpy())].iterrows():
        if len(rows) >= max_tiles:
            break
        geom = make_valid(fire.geometry)
        if geom is None or geom.is_empty:
            continue
        centre = geom.centroid
        epsg0, _ = utm_tile(centre.x, centre.y)
        local = make_valid(shapely_transform(transformer(4326, epsg0).transform, geom))
        parts = [g for g in getattr(local, "geoms", [local]) if g.geom_type == "Polygon"]
        if not parts:
            continue
        ring = max(parts, key=lambda p: p.area).exterior
        got = 0
        for _ in range(40):
            if got >= per_fire or len(rows) >= max_tiles:
                break
            point = ring.interpolate(rng.uniform(0, ring.length))
            lon, lat = transformer(epsg0, 4326).transform(point.x, point.y)
            epsg, bounds = utm_tile(lon, lat)
            tile = box(*bounds)
            inside = local if epsg == epsg0 else make_valid(shapely_transform(transformer(4326, epsg).transform, geom))
            share = inside.intersection(tile).area / tile.area
            if not .2 <= share <= .8:
                continue
            shape = footprint(epsg, bounds)
            if not taken.clear_of(shape):
                continue
            taken.add(shape)
            got += 1
            block = block_of(lat, lon)
            rows.append(dict(part="large", tile_name=f"{fire['fire_id']}-e{got}", source_tile_id="", fire_id=fire["fire_id"],
                             country=fire["country"], fire_date=str(fire["fire_date"])[:10], area_ha=fire["area_ha"],
                             scene_date="", day_min=PARTS["large"]["window"][0], day_max=PARTS["large"]["window"][1],
                             epsg=epsg, minx=bounds[0], miny=bounds[1], maxx=bounds[2], maxy=bounds[3], lon=lon, lat=lat,
                             block=block, split=split_for_block(block, block_split), offset_km=np.nan,
                             edge_burn_frac=round(share, 3)))
    return rows


def make_valid(geom):
    from shapely.validation import make_valid as fix
    return geom if geom is None or geom.is_valid else fix(geom)


def shapely_transform(func, geom):
    from shapely.ops import transform
    return transform(func, geom)


def large_fires(cfg):
    """The fires the build left out for size: above max_area_ha, in its countries, dates and French latitude limit."""
    from src.effis_loader import load_effis
    fires = load_effis(cfg["effis_path"], countries=cfg["countries"], start_date=cfg["start_date"],
                       end_date=cfg["end_date"], min_area_ha=cfg["max_area_ha"])
    fires = fires[fires["area_ha"] > cfg["max_area_ha"]]
    southern = (fires["country"] != "FR") | (fires.geometry.centroid.y <= cfg["southern_france_max_lat"])
    return fires[southern].reset_index(drop=True)


def cost(part, count):
    """Expected and largest requests and PU for count tiles of a part."""
    spec = PARTS[part]
    search = 0 if spec["window"] is None else 1
    per_candidate_pu = PU_L1C + PU_SCL
    return dict(expected_requests=count * (search + 2 * spec["expected"]), worst_requests=count * (search + 2 * spec["candidates"]),
                expected_pu=count * per_candidate_pu * spec["expected"], worst_pu=count * per_candidate_pu * spec["candidates"])


def budget_table(counts):
    lines = ["| Part | Tiles | Requests, expected | Requests, at most | PU, expected | PU, at most |", "|---|---:|---:|---:|---:|---:|"]
    total = dict(expected_requests=0, worst_requests=0, expected_pu=0, worst_pu=0)
    for part, count in counts.items():
        c = cost(part, count)
        total = {k: total[k] + c[k] for k in total}
        lines.append(f"| {part} | {count:,} | {c['expected_requests']:,.0f} | {c['worst_requests']:,.0f} | "
                     f"{c['expected_pu']:,.0f} | {c['worst_pu']:,.0f} |")
    lines.append(f"| all | {sum(counts.values()):,} | {total['expected_requests']:,.0f} | {total['worst_requests']:,.0f} | "
                 f"{total['expected_pu']:,.0f} | {total['worst_pu']:,.0f} |")
    return lines, total


def markdown(frame):
    try:
        return frame.to_markdown()
    except ImportError:
        return frame.to_string()


def run_plan(args, cfg):
    import geopandas as gpd
    from shapely.ops import unary_union

    from src.effis_loader import load_all_perimeters
    out = Path(args.out)
    targets_path, log_path = out / "targets.csv", out / "acquired.csv"
    if log_path.exists() and len(pd.read_csv(log_path)) and not args.replan:
        raise SystemExit(f"{log_path} already has downloads: the targets are fixed. Use --replan to start again from scratch.")
    rng = np.random.default_rng(args.seed)
    index = pd.read_csv(args.index, low_memory=False)
    index = index[index["split"].notna()].copy()
    block_split = index.groupby("block")["split"].agg(lambda s: s.mode().iloc[0]).to_dict() if "block" in index else {}
    perimeters = load_all_perimeters(cfg["effis_path"], cfg.get("effis_extra"))
    land = unary_union(gpd.read_file(ROOT / "tools" / "data" / "land_med_ne50m.geojson").geometry)
    taken = released_footprints(index)
    today = pd.Timestamp(datetime.now(timezone.utc).date())
    print(f"{len(index):,} released tiles, {len(perimeters):,} perimeters", flush=True)

    rows = []
    if args.negatives:
        rows += pick_negatives(index, perimeters, land, taken, args.negatives, rng, block_split)
        print(f"negative: {sum(r['part'] == 'negative' for r in rows):,} tiles", flush=True)
    if args.boundary:
        rows += pick_fire_scenes(index, "boundary", args.boundary, rng, today - pd.Timedelta(days=PARTS["boundary"]["window"][1] + 2))
    if args.firstlook:
        rows += pick_fire_scenes(index, "firstlook", args.firstlook, rng, today - pd.Timedelta(days=PARTS["firstlook"]["window"][1] + 2))
    if args.large:
        fires = large_fires(cfg)
        print(f"{len(fires):,} fires above {cfg['max_area_ha']:,.0f} ha", flush=True)
        rows += pick_large(fires, taken, args.per_fire, args.large, rng, block_split)
    if not rows:
        raise SystemExit("No targets: every part is set to 0.")
    targets = pd.DataFrame(rows)
    targets.insert(0, "target_id", [f"{part}-{i:05d}" for i, part in enumerate(targets["part"])])
    out.mkdir(parents=True, exist_ok=True)
    targets[TARGET_FIELDS].to_csv(targets_path, index=False)

    counts = targets["part"].value_counts().reindex(list(PARTS)).dropna().astype(int).to_dict()
    lines, total = budget_table(counts)
    by_split = targets.groupby(["part", "split"]).size().unstack(fill_value=0)
    text = ["# New tiles: plan", "", f"Targets: {targets_path}", "", *lines, "",
            "Requests count every catalogue search, scene-classification request and L1C request. Expected costs assume "
            "1.6 candidate scenes for a searched window, as in the release build.", "", "Tiles by split:", "",
            markdown(by_split), ""]
    if "large" in counts:
        large = targets[targets["part"] == "large"]
        text += [f"Large fires: {large['fire_id'].nunique():,} fires, median area {large['area_ha'].median():,.0f} ha; "
                 f"tiles {large['edge_burn_frac'].min():.0%} to {large['edge_burn_frac'].max():.0%} inside the perimeter.", ""]
    if total["expected_requests"] > args.budget:
        text += [f"Expected requests ({total['expected_requests']:,.0f}) exceed --budget {args.budget:,}: lower a part's count."]
    (out / "plan.md").write_text("\n".join(text))
    print("\n".join(text))


# ---------------------------------------------------------------- downloads

class Spend:
    """Requests and PU spent by one target, counted before each call so failures count too."""

    def __init__(self):
        self.requests, self.pu, self.candidates = 0, 0.0, 0


def sh_bbox(target):
    from sentinelhub import BBox, CRS
    return BBox((target["minx"], target["miny"], target["maxx"], target["maxy"]), crs=CRS(int(target["epsg"])))


def search_dates(target, sh, spend):
    """Candidate dates in the part's window, one per day: least cloudy first, or earliest first for the first look."""
    from sentinelhub import SentinelHubCatalog

    from src.s2_pipeline import S2L2A_COLLECTION
    spec = PARTS[target["part"]]
    fire_date = pd.Timestamp(target["fire_date"])
    start = (fire_date + pd.Timedelta(days=int(target["day_min"]))).date().isoformat()
    end = (fire_date + pd.Timedelta(days=int(target["day_max"]))).date().isoformat()
    spend.requests += 1
    query = dict(collection=S2L2A_COLLECTION, bbox=sh_bbox(target), time=(start, end),
                 fields={"include": ["properties.datetime", "properties.eo:cloud_cover"], "exclude": []})
    if spec["maxcc"] is not None:
        query["filter"] = f"eo:cloud_cover < {spec['maxcc'] * 100:.0f}"
    found = {}
    for item in SentinelHubCatalog(sh).search(**query):
        date = item["properties"]["datetime"].split("T")[0]
        cover = float(item["properties"].get("eo:cloud_cover", 100))
        found[date] = min(cover, found.get(date, 101))
    order = sorted(found.items(), key=(lambda kv: kv[0]) if spec["rule"] == "earliest" else (lambda kv: (kv[1], kv[0])))
    return [date for date, _ in order[: spec["candidates"]]]


def stamps(eop):
    """Acquisition times of an EOPatch, under either name eo-learn has used."""
    found = getattr(eop, "timestamps", None)
    return found if found is not None else eop.timestamp


def fetch(target, date, sh, spend, cache_folder):
    """Scene classification from L2A, then the 13 L1C bands as uint16, for one date: as the pipeline asks for them,
    within 180 minutes of the date's acquisitions, bicubic upsampling. None when there is no scene that day."""
    import datetime as dt

    from eolearn.core import FeatureType
    from eolearn.io import SentinelHubInputTask

    from src.s2_pipeline import S2L1C_COLLECTION, S2L2A_COLLECTION
    aux = {"processing": {"upsampling": "BICUBIC"}}
    spend.requests += 1
    spend.pu += PU_SCL
    eop = SentinelHubInputTask(data_collection=S2L2A_COLLECTION, resolution=10, additional_data=[(FeatureType.MASK, "SCL")],
                               aux_request_args=aux, config=sh, cache_folder=cache_folder,
                               time_difference=dt.timedelta(minutes=180))(bbox=sh_bbox(target), time_interval=(date, date))
    if len(stamps(eop)) == 0:
        return None
    spend.requests += 1
    spend.pu += PU_L1C
    spend.candidates += 1
    eop = SentinelHubInputTask(data_collection=S2L1C_COLLECTION, resolution=10, bands_feature=(FeatureType.DATA, "BANDS"),
                               additional_data=[(FeatureType.MASK, "dataMask")], bands=BANDS, bands_dtype=np.uint16,
                               aux_request_args=aux, config=sh, cache_folder=cache_folder,
                               time_difference=dt.timedelta(minutes=180))(eopatch=eop)
    return dict(time=stamps(eop)[0], bands=np.asarray(eop.data["BANDS"][0], np.uint16),
                valid=eop.mask["dataMask"][0, ..., 0].astype(bool), scl=eop.mask["SCL"][0, ..., 0])


OCM_LOCK = threading.Lock()


def cloud_and_shadow(bands):
    """OmniCloudMask on B04, B03 and B08 in the release's units (reflectance x 10,000), as OmniCloudMaskTask does, on the CPU."""
    from omnicloudmask import predict_from_array
    stack = np.stack([bands[..., B04], bands[..., B03], bands[..., B08]], axis=0).astype(np.float32)
    with OCM_LOCK:
        prediction = np.asarray(predict_from_array(stack, batch_size=1, apply_no_data_mask=True, no_data_value=0,
                                                   inference_device="cpu"))[0]
    return (prediction == 1) | (prediction == 2), prediction == 3


def burn_layers(perimeters, target, scene_time, cfg):
    """fresh, old and older burn on the tile from every perimeter, as BurnScarRasterTask and burn_labels.py rasterise them."""
    from rasterio.features import rasterize
    from rasterio.transform import from_bounds
    bounds = (target["minx"], target["miny"], target["maxx"], target["maxy"])
    transform = from_bounds(*bounds, PIXELS, PIXELS)
    nearby = perimeters.iloc[perimeters.sindex.query(footprint(int(target["epsg"]), bounds), predicate="intersects")]
    nearby = nearby.to_crs(int(target["epsg"]))
    age = (np.datetime64(scene_time) - nearby["fire_date"].values) / np.timedelta64(1, "D")

    def draw(selection):
        geoms = [g for g in nearby.geometry[selection] if g is not None and not g.is_empty]
        if not geoms:
            return np.zeros((PIXELS, PIXELS), bool)
        return rasterize(((g, 1) for g in geoms), out_shape=(PIXELS, PIXELS), transform=transform, fill=0,
                         dtype=np.uint8).astype(bool)

    fresh = draw((age >= 0) & (age <= cfg["fresh_burn_max_days"]))
    old = draw((age > cfg["fresh_burn_max_days"]) & (age <= cfg["burn_max_age_days"]))
    older = draw(age > cfg["burn_max_age_days"])
    return fresh, old, older


def make_mask(scene, cloud, shadow, fresh, old, older, cfg):
    """CombineMaskTask's order: clear < water < old < fresh < shadow < cloud < no data; older_burn only replaces clear."""
    bands = scene["bands"].astype(np.float32)
    green, nir = bands[..., B03], bands[..., B08]
    water = (green - nir) / np.maximum(green + nir, 1e-6) > cfg["ndwi_threshold"]
    if cfg.get("use_scl_water", True):
        water |= scene["scl"] == 6
    mask = np.full((PIXELS, PIXELS), CLEAR, np.uint8)
    mask[water] = WATER
    mask[old] = OLD
    mask[fresh] = FRESH
    mask[shadow] = SHADOW
    mask[cloud] = CLOUD
    mask[~scene["valid"]] = NODATA
    mask[older & (mask == CLEAR)] = OLDER
    return mask


def write_tile(out, target, scene, mask):
    import rasterio
    from rasterio.transform import from_bounds
    stamp = pd.Timestamp(scene["time"]).strftime(TS_FORMAT)
    folder = out / "raw" / "acquisitions" / target["tile_name"] / stamp
    folder.mkdir(parents=True, exist_ok=True)
    stem = f"{target['tile_name']}_{stamp}"
    transform = from_bounds(target["minx"], target["miny"], target["maxx"], target["maxy"], PIXELS, PIXELS)
    common = dict(driver="GTiff", height=PIXELS, width=PIXELS, crs=f"EPSG:{int(target['epsg'])}", transform=transform,
                  compress="deflate", tiled=True, blockxsize=512, blockysize=512)
    bands = scene["bands"].copy()
    bands[~scene["valid"]] = 0
    image_path, mask_path = folder / f"{stem}_s2.tif", folder / f"{stem}_mask.tif"
    for path, write in ((image_path, "image"), (mask_path, "mask")):
        temp = path.with_suffix(".tmp.tif")
        if write == "image":
            with rasterio.open(temp, "w", count=13, dtype="uint16", nodata=0, predictor=2, **common) as dst:
                for b in range(13):
                    dst.write(bands[..., b], b + 1)
                    dst.set_band_description(b + 1, BANDS[b])
        else:
            with rasterio.open(temp, "w", count=1, dtype="uint8", nodata=NODATA, **common) as dst:
                dst.write_colormap(1, COLOURS)
                dst.write(mask, 1)
        temp.replace(path)
    return image_path.relative_to(out / "raw").as_posix(), mask_path.relative_to(out / "raw").as_posix()


def process_target(target, perimeters, cfg, sh, out, cache_folder, spend):
    part = target["part"]
    rule = PARTS[part]["rule"]
    record = dict(status="no_scene", scene_time="", valid_frac="", obscured_frac="", rule_pass="", image_path="", mask_path="")
    dates = [target["scene_date"]] if rule == "same_scene" else search_dates(target, sh, spend)
    best = None
    for date in dates:
        scene = fetch(target, date, sh, spend, cache_folder)
        if scene is None:
            continue
        valid = float(scene["valid"].mean())
        if valid < .5:
            record.update(status="no_data", valid_frac=round(valid, 4))
            continue
        cloud, shadow = cloud_and_shadow(scene["bands"])
        fresh, old, older = burn_layers(perimeters, target, scene["time"], cfg)
        burn = fresh | old
        obscured = float((burn & (cloud | shadow)).sum() / max(int(burn.sum()), 1))
        candidate = dict(scene=scene, cloud=cloud, shadow=shadow, fresh=fresh, old=old, older=older, valid=valid,
                         obscured=obscured)
        passes = rule != "least_obscured" or obscured <= cfg["burn_cloud_max_fraction"]
        if best is None or (rule == "least_obscured" and obscured < best["obscured"]):
            best = candidate
        if passes:
            best = candidate
            break
    if best is None:
        return record
    passed = rule != "least_obscured" or best["obscured"] <= cfg["burn_cloud_max_fraction"]
    mask = make_mask(best["scene"], best["cloud"], best["shadow"], best["fresh"], best["old"], best["older"], cfg)
    image_path, mask_path = write_tile(out, target, best["scene"], mask)
    record.update(status="ok" if passed else "kept_best_available", scene_time=pd.Timestamp(best["scene"]["time"]).isoformat(),
                  valid_frac=round(best["valid"], 4), obscured_frac=round(best["obscured"], 4), rule_pass=passed,
                  image_path=image_path, mask_path=mask_path)
    return record


def check_login(sh):
    """Ask Copernicus for a token before any request: a refused login stops here, costing nothing."""
    try:
        from sentinelhub import SentinelHubSession
        SentinelHubSession(config=sh)
    except (ImportError, TypeError, AttributeError):
        return   # an older sentinelhub-py: the first request will tell
    except Exception as exc:
        raise SystemExit(f"Copernicus refused the login in .env: {str(exc)[:300]}\n"
                         "Make a new OAuth client in the Sentinel Hub dashboard of the Copernicus Data Space Ecosystem "
                         "(User settings, OAuth clients) and put its ID and secret in .env as SH_CLIENT_ID and "
                         "SH_CLIENT_SECRET. Nothing was spent.")


def month_spend(log):
    """Requests and PU logged in the current calendar month (UTC), when the Copernicus quota last reset."""
    if log.empty:
        return 0, 0.0
    stamps = pd.to_datetime(log["utc_time"], errors="coerce", utc=True)
    now = pd.Timestamp.now(tz="UTC")
    refused = log["error"].fillna("").astype(str).str.contains("|".join(REFUSED))   # never reached Sentinel Hub
    this_month = (stamps.dt.year == now.year) & (stamps.dt.month == now.month) & ~refused
    return int(log.loc[this_month, "requests"].sum()), float(log.loc[this_month, "pu"].sum())


def run_download(args, cfg):
    os.environ["CUDA_VISIBLE_DEVICES"] = ""   # before torch loads: OmniCloudMask stays on the CPU
    import torch
    torch.set_num_threads(args.threads)
    from dotenv import load_dotenv

    from src.effis_loader import load_all_perimeters
    from src.s2_pipeline import make_sh_config
    out = Path(args.out)
    targets = pd.read_csv(out / "targets.csv", low_memory=False)
    if args.parts:
        targets = targets[targets["part"].isin(args.parts)]
    log_path = out / "acquired.csv"
    log = pd.read_csv(log_path) if log_path.exists() else pd.DataFrame(columns=LOG_FIELDS)
    finished = set(log.loc[log["status"].isin(FINISHED), "target_id"])
    pending = targets[~targets["target_id"].isin(finished)]
    # Round robin over the parts, so a budget that runs out leaves every part with tiles.
    pending = pending.assign(_rank=pending.groupby("part").cumcount()).sort_values(["_rank", "part"]).drop(columns="_rank")
    if args.limit:
        pending = pending.head(args.limit)
    spent_requests, spent_pu = month_spend(log)
    print(f"{len(pending):,} targets to fetch, {len(finished):,} finished. This month so far: {spent_requests:,} requests, "
          f"{spent_pu:,.0f} PU; budget {args.max_requests:,} requests, {args.max_pu:,.0f} PU.", flush=True)
    if not log_path.exists():
        with open(log_path, "w", newline="") as handle:
            csv.DictWriter(handle, LOG_FIELDS).writeheader()

    load_dotenv(ROOT / ".env", override=True)   # the file wins over anything left exported in the shell
    sh = make_sh_config(os.environ["SH_CLIENT_ID"], os.environ["SH_CLIENT_SECRET"])
    check_login(sh)
    perimeters = load_all_perimeters(cfg["effis_path"], cfg.get("effis_extra"))
    lock = threading.Lock()
    state = dict(requests=spent_requests, pu=spent_pu, reserved_requests=0, reserved_pu=0.0, stopped=False)
    counts = {}

    def worst(part):
        spec = PARTS[part]
        return (0 if spec["window"] is None else 1) + 2 * spec["candidates"], (PU_L1C + PU_SCL) * spec["candidates"]

    def work(target):
        need_requests, need_pu = worst(target["part"])
        with lock:
            if state["stopped"] or (state["requests"] + state["reserved_requests"] + need_requests > args.max_requests
                                    or state["pu"] + state["reserved_pu"] + need_pu > args.max_pu):
                state["stopped"] = True
                return None
            state["reserved_requests"] += need_requests
            state["reserved_pu"] += need_pu
        spend = Spend()
        cache = os.path.join(args.cache_folder, target["target_id"]) if args.cache_folder else None
        try:
            record = process_target(target, perimeters, cfg, sh, out, cache, spend)
            record["error"] = ""
        except Exception as exc:   # left unfinished, so a restart tries it again
            record = dict(status="failed", error=str(exc)[:300])
            if any(word in str(exc) for word in REFUSED):   # a refused login costs nothing, and every target would fail
                spend.requests, spend.pu = 0, 0.0
                with lock:
                    state["stopped"], state["refused"] = True, str(exc)[:200]
        finally:
            with lock:
                state["reserved_requests"] -= need_requests
                state["reserved_pu"] -= need_pu
                state["requests"] += spend.requests
                state["pu"] += spend.pu
        record.update(target_id=target["target_id"], part=target["part"], tile_name=target["tile_name"],
                      candidates_fetched=spend.candidates, requests=spend.requests, pu=round(spend.pu, 3),
                      utc_time=datetime.now(timezone.utc).isoformat(timespec="seconds"))
        with lock, open(log_path, "a", newline="") as handle:
            csv.DictWriter(handle, LOG_FIELDS, extrasaction="ignore").writerow({k: record.get(k, "") for k in LOG_FIELDS})
        return record

    with ThreadPoolExecutor(args.workers) as pool:
        futures = [pool.submit(work, target) for _, target in pending.iterrows()]
        for done, future in enumerate(as_completed(futures), 1):
            record = future.result()
            if record is None:
                continue
            key = f"{record['part']} {record['status']}"
            counts[key] = counts.get(key, 0) + 1
            if record["status"] == "failed":
                print(f"{record['target_id']}: failed: {record['error']}", flush=True)
            if done % 50 == 0:
                print(f"{done:,}/{len(pending):,} | {state['requests']:,} requests, {state['pu']:,.0f} PU this month | "
                      f"{dict(sorted(counts.items()))}", flush=True)
    if state.get("refused"):
        raise SystemExit(f"Stopped: Copernicus refused the login ({state['refused']}). Check SH_CLIENT_ID and "
                         "SH_CLIENT_SECRET in .env; the targets are kept, and a rerun carries on.")
    if state["stopped"]:
        print(f"Stopped at the budget: {state['requests']:,} requests and {state['pu']:,.0f} PU this month. "
              "Run again next month, or with a higher --max-requests if the dashboard shows room.", flush=True)
    print(f"Done: {dict(sorted(counts.items()))}. This month: {state['requests']:,} requests, {state['pu']:,.0f} PU. Log: {log_path}")


def run_status(args):
    out = Path(args.out)
    targets = pd.read_csv(out / "targets.csv", low_memory=False)
    log = pd.read_csv(out / "acquired.csv") if (out / "acquired.csv").exists() else pd.DataFrame(columns=LOG_FIELDS)
    last = log.drop_duplicates("target_id", keep="last")
    table = targets[["target_id", "part"]].merge(last[["target_id", "status"]], on="target_id", how="left")
    table["status"] = table["status"].fillna("pending")
    print(table.groupby(["part", "status"]).size().unstack(fill_value=0).to_string())
    requests, pu = month_spend(log)
    print(f"\nThis month: {requests:,} requests, {pu:,.0f} PU. All runs: {int(log['requests'].sum()):,} requests, "
          f"{float(log['pu'].sum()):,.0f} PU.")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("command", choices=["plan", "run", "status"])
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--index", default="data/v4/index.csv")
    parser.add_argument("--out", default="data/additions")
    parser.add_argument("--negatives", type=int, default=1200, help="burn-free tiles")
    parser.add_argument("--boundary", type=int, default=600, help="fires to get a scene 61-120 days after")
    parser.add_argument("--large", type=int, default=600, help="tiles on the edge of fires above the size limit")
    parser.add_argument("--per-fire", type=int, default=2, help="edge tiles per large fire")
    parser.add_argument("--firstlook", type=int, default=400, help="fires to get a scene 0-5 days after")
    parser.add_argument("--seed", type=int, default=5)
    parser.add_argument("--budget", type=int, default=9000, help="plan: warn above this many expected requests")
    parser.add_argument("--replan", action="store_true", help="plan again although downloads have started")
    parser.add_argument("--max-requests", type=int, default=9000, help="run: stop at this many requests this month")
    parser.add_argument("--max-pu", type=float, default=9000, help="run: stop at this many PU this month")
    parser.add_argument("--parts", nargs="*", choices=list(PARTS), help="run: only these parts")
    parser.add_argument("--limit", type=int, default=0, help="run: only this many targets, for a trial")
    parser.add_argument("--workers", type=int, default=3)
    parser.add_argument("--threads", type=int, default=8, help="CPU threads for OmniCloudMask")
    parser.add_argument("--cache-folder", default="", help="Sentinel Hub cache; empty keeps nothing")
    args = parser.parse_args()
    cfg = yaml.safe_load(Path(args.config).read_text())
    if args.command == "plan":
        run_plan(args, cfg)
    elif args.command == "run":
        run_download(args, cfg)
    else:
        run_status(args)


if __name__ == "__main__":
    main()
