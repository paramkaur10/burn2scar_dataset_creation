#!/usr/bin/env python3
"""Numbers the referee asks for, from the released files and the existing outputs. CPU only; no Copernicus quota.

  python tools/referee_checks.py cloud --workers 16     # cloud and shadow inside the perimeters they hide
  python tools/referee_checks.py history --workers 16   # label history length, and older_burn with a fixed 2-year look-back
  python tools/referee_checks.py burns                  # fires that may be prescribed or pastoral burns
  python tools/referee_checks.py water --workers 16     # labels against JRC Global Surface Water (downloads its tiles)
  python tools/referee_checks.py none                   # dNBR "none" fires: scar already on the pre-fire image, or no change
  python tools/referee_checks.py old                    # old-window scars against a same-season (anniversary) reference
  python tools/referee_checks.py misc --workers 16      # two-pass images, zeros in valid pixels, train-test distances,
                                                        # overlap groups, the time split's test set, strata coverage,
                                                        # review counts, CEMS per pair, snow counts, tiles never built
  python tools/referee_checks.py columns                # new index columns for 5.0 from the parts above

Each part writes validation/referee/<part>.md and prints it. none needs the fresh-window fires flagged none run again
(burn_evidence.py now records the ring's NBR before and after):
  python tools/referee_checks.py none --make-list
  python tools/burn_evidence.py --tiles-from validation/referee/none_tiles.csv --out validation/dnbr_none --workers 16
old needs the anniversary run:
  python tools/burn_evidence.py --window old --reference anniversary --out validation/dnbr_old_anniversary --workers 16
"""
import argparse
import json
import math
import sys
import urllib.request
import warnings
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
warnings.filterwarnings("ignore", message=".*geographic CRS.*")
warnings.filterwarnings("ignore", message=".*has no geotransform.*")

CLEAR, FRESH, OLD, CLOUD, SHADOW, WATER, OLDER, NODATA = 0, 2, 3, 4, 5, 6, 7, 255
OUT = Path("validation/referee")
GSW = "https://storage.googleapis.com/global-surface-water/downloads2021/occurrence/occurrence_{x}_{y}v1_4_2021.tif"


def md(frame):
    cells = lambda v: "" if (isinstance(v, float) and math.isnan(v)) else str(v)
    return "\n".join(["| " + " | ".join(map(str, frame.columns)) + " |", "|" + "---|" * len(frame.columns)]
                     + ["| " + " | ".join(cells(v) for v in row) + " |" for row in frame.itertuples(index=False)])


def pct(part, whole, digits=2):
    return f"{100 * part / whole:.{digits}f}" if whole else ""


def write(name, lines):
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / f"{name}.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    print(f"Wrote {OUT / f'{name}.md'}")


def released(args):
    index = pd.read_csv(args.index, low_memory=False)
    return index[index["split"].notna()].copy()


def pooled(function, jobs, workers):
    with ProcessPoolExecutor(workers) as pool:
        return list(pool.map(function, jobs, chunksize=32))


# ---------------------------------------------------------------- cloud and shadow inside perimeters

def cloud_job(job):
    mask_path, age_path = job
    with rasterio.open(mask_path) as src:
        mask = src.read(1)
    with rasterio.open(age_path) as src:
        age = src.read(1).astype(np.int32)
    fresh, old = (age >= 1) & (age <= 90), (age >= 91) & (age <= 365)
    out = {}
    for name, value in (("cloud", CLOUD), ("shadow", SHADOW)):
        here = mask == value
        out[f"{name}_px"], out[f"{name}_in_fresh"], out[f"{name}_in_old"] = int(here.sum()), int((here & fresh).sum()), int((here & old).sum())
    out["fresh_perimeter_px"], out["fresh_perimeter_hidden"] = int(fresh.sum()), int((fresh & np.isin(mask, [CLOUD, SHADOW])).sum())
    return out


def run_cloud(args):
    tiles = released(args)
    jobs = [(str(Path(args.masks_root) / r.mask_path), str(Path(args.masks_root) / r.age_path)) for r in tiles.itertuples()]
    table = pd.DataFrame(pooled(cloud_job, jobs, args.workers)).assign(window=tiles["window"].to_numpy())
    rows = []
    for window, part in [("all", table), *table.groupby("window")]:
        s = part.sum(numeric_only=True)
        rows.append({"Tiles": window, "Cloud px": f"{int(s['cloud_px']):,}", "Cloud inside fresh perimeters %": pct(s["cloud_in_fresh"], s["cloud_px"], 1),
                     "Cloud inside old perimeters %": pct(s["cloud_in_old"], s["cloud_px"], 1),
                     "Shadow px": f"{int(s['shadow_px']):,}", "Shadow inside fresh perimeters %": pct(s["shadow_in_fresh"], s["shadow_px"], 1),
                     "Shadow inside old perimeters %": pct(s["shadow_in_old"], s["shadow_px"], 1),
                     "Fresh perimeters hidden by cloud or shadow %": pct(s["fresh_perimeter_hidden"], s["fresh_perimeter_px"], 2)})
    write("cloud", ["# Cloud and cloud shadow inside burn perimeters", "",
                    "Perimeters from the age raster: 1-90 days fresh, 91-365 old. Cloud and shadow outrank burn in the mask, so these "
                    "pixels are burn the mask cannot show; some may be smoke or dark burn taken for cloud or shadow.", "",
                    md(pd.DataFrame(rows))])


# ---------------------------------------------------------------- label history

def history_job(job):
    mask_path, age_path = job
    with rasterio.open(mask_path) as src:
        mask = src.read(1)
    with rasterio.open(age_path) as src:
        age = src.read(1).astype(np.int32)
    older = mask == OLDER
    return {"valid_px": int((mask != NODATA).sum()), "older_px": int(older.sum()), "older_2y_px": int((older & (age <= 730)).sum())}


def run_history(args):
    tiles = released(args)
    jobs = [(str(Path(args.masks_root) / r.mask_path), str(Path(args.masks_root) / r.age_path)) for r in tiles.itertuples()]
    table = pd.DataFrame(pooled(history_job, jobs, args.workers))
    table["tile_id"] = tiles["tile_id"].to_numpy()
    scene = pd.to_datetime(tiles["scene_time"], format="mixed", utc=True).dt.tz_convert(None)
    table["scene_year"] = scene.dt.year.to_numpy()
    table["history_days"] = (scene.dt.normalize() - pd.Timestamp("2018-01-01")).dt.days.to_numpy()
    OUT.mkdir(parents=True, exist_ok=True)
    table[["tile_id", "history_days", "older_px", "older_2y_px", "valid_px"]].to_csv(OUT / "history_tiles.csv", index=False)
    rows = []
    for year, part in table.groupby("scene_year"):
        s = part.sum(numeric_only=True)
        rows.append({"Scene year": year, "Tiles": len(part), "History, days (median)": int(part["history_days"].median()),
                     "older_burn % of valid": pct(s["older_px"], s["valid_px"]),
                     "older_burn within 2 years % of valid": pct(s["older_2y_px"], s["valid_px"])})
    s = table.sum(numeric_only=True)
    write("history", ["# Label history and older_burn", "",
                      "The label history starts on 1 January 2018 for every scene, so a 2026 scene can show scars up to 8 years old and "
                      "a 2020 scene up to 2 years. Cut at 730 days, which every scene since 2020 can see, older_burn has the same "
                      "meaning in every year; the age raster gives any cut.", "", md(pd.DataFrame(rows)), "",
                      f"All scenes: older_burn {pct(s['older_px'], s['valid_px'])} % of valid pixels; within 2 years "
                      f"{pct(s['older_2y_px'], s['valid_px'])} %; scars over 730 days old {pct(s['older_px'] - s['older_2y_px'], s['valid_px'])} %."])


# ---------------------------------------------------------------- prescribed or pastoral burns

def reburn_shares(fires, perimeters):
    """Share of each fire's area on an earlier EFFIS perimeter (any earlier date in the label history)."""
    fires3035, perims3035 = fires.to_crs(3035), perimeters.to_crs(3035)
    sindex = perims3035.sindex
    out = {}
    for fire in fires3035.itertuples():
        geom = fire.geometry.buffer(0)
        hits = perims3035.iloc[sindex.query(geom, predicate="intersects")]
        earlier = hits[(hits["fire_date"] < fire.fire_date) & (hits["fire_id"] != fire.fire_id)]
        if earlier.empty or geom.area == 0:
            out[fire.fire_id] = 0.0
            continue
        from shapely.ops import unary_union
        out[fire.fire_id] = geom.intersection(unary_union(list(earlier.geometry.buffer(0)))).area / geom.area
    return out


def run_burns(args):
    from src.effis_loader import load_all_perimeters
    cfg = yaml.safe_load(Path(args.config).read_text())
    tiles = released(args)
    fires = tiles.drop_duplicates("fire_id").set_index("fire_id")
    perimeters = load_all_perimeters(cfg["effis_path"], cfg.get("effis_extra"))
    own = perimeters[perimeters["fire_id"].isin(fires.index)].drop_duplicates("fire_id")
    print(f"Re-burn shares for {len(own):,} fires ...", flush=True)
    reburn = reburn_shares(own, perimeters)
    frame = pd.DataFrame({"fire_id": fires.index, "country": fires["country"].to_numpy(),
                          "month": pd.to_datetime(fires["fire_date"]).dt.month.to_numpy(),
                          "land_cover": fires.get("land_cover", pd.Series("unknown", index=fires.index)).astype(str).to_numpy(),
                          "area_ha": fires["area_ha"].to_numpy()})
    frame["reburn"] = frame["fire_id"].map(reburn).fillna(0)
    frame["flag"] = fires["fire_dnbr_flag"].to_numpy() if "fire_dnbr_flag" in fires else ""
    assessed = frame["flag"].isin(["burn", "weak", "none"])
    frame["cold_season"] = frame["month"].isin([11, 12, 1, 2, 3])
    frame["non_forest"] = frame["land_cover"].isin(["shrubland", "other_natural", "agriculture"])
    frame["recurrent"] = frame["reburn"] >= .1
    frame["likely_managed"] = frame["cold_season"] & frame["non_forest"]
    frame["likely_managed_recurrent"] = frame["likely_managed"] & frame["recurrent"]
    OUT.mkdir(parents=True, exist_ok=True)
    frame.to_csv(OUT / "burns_fires.csv", index=False)
    rows = []
    for country, part in [("all", frame), *frame.groupby("country")]:
        rows.append({"Country": country, "Fires": len(part), "November-March %": pct(part["cold_season"].sum(), len(part), 1),
                     "Shrub, other natural or farm land %": pct(part["non_forest"].sum(), len(part), 1),
                     "On an earlier perimeter (>= 10 %) %": pct(part["recurrent"].sum(), len(part), 1),
                     "Cold season and non-forest %": pct(part["likely_managed"].sum(), len(part), 1),
                     "... and recurrent %": pct(part["likely_managed_recurrent"].sum(), len(part), 1),
                     "Burned area of cold-season non-forest fires %": pct(part.loc[part["likely_managed"], "area_ha"].sum(), part["area_ha"].sum(), 1),
                     "Flagged burn, all fires %": pct((part["flag"] == "burn").sum(), assessed[part.index].sum(), 1),
                     "Flagged burn, cold-season non-forest %": pct(((part["flag"] == "burn") & part["likely_managed"]).sum(),
                                                                   (assessed[part.index] & part["likely_managed"]).sum(), 1)})
    months = frame.groupby("month").agg(fires=("fire_id", "size"), non_forest=("non_forest", "mean"), recurrent=("recurrent", "mean"))
    months = months.assign(non_forest=(100 * months["non_forest"]).round(1), recurrent=(100 * months["recurrent"]).round(1)).reset_index()
    write("burns", ["# Fires that may be prescribed, agricultural or pastoral burns", "",
                    "EFFIS does not separate them from wildfires. An indicator, not a classification: a fire dated November to March "
                    "on land EFFIS classes as mostly shrubland, other natural vegetation or agriculture; recurrent where at least a "
                    "tenth of its area lies on an earlier EFFIS perimeter.", "", md(pd.DataFrame(rows)), "", "By month:", "",
                    md(months.rename(columns={"month": "Month", "fires": "Fires", "non_forest": "Non-forest %", "recurrent": "Recurrent %"}))])


# ---------------------------------------------------------------- permanent water

def gsw_name(lon, lat):
    x = int(math.floor(lon / 10) * 10)
    y = int(math.floor(lat / 10) * 10) + 10
    return f"{abs(x)}{'E' if x >= 0 else 'W'}", f"{abs(y)}{'N' if y >= 0 else 'S'}"


def water_job(job):
    from rasterio.vrt import WarpedVRT
    from rasterio.enums import Resampling
    mask_path, sources = job
    with rasterio.open(mask_path) as src:
        mask, crs, transform, shape = src.read(1), src.crs, src.transform, src.shape
    occurrence = np.zeros(shape, np.uint8)
    seen = np.zeros(shape, bool)
    for source in sources:
        with rasterio.open(source) as src, WarpedVRT(src, crs=crs, transform=transform, width=shape[1], height=shape[0],
                                                      resampling=Resampling.nearest) as vrt:
            part = vrt.read(1)
        take = ~seen & (part != 255)
        occurrence[take], seen[take] = part[take], True
    permanent = seen & (occurrence >= 90)
    never = seen & (occurrence == 0)
    burn = np.isin(mask, [FRESH, OLD])
    return {"permanent_px": int(permanent.sum()), "permanent_as_water": int((permanent & (mask == WATER)).sum()),
            "permanent_as_burn": int((permanent & burn).sum()), "permanent_as_clear": int((permanent & (mask == CLEAR)).sum()),
            "permanent_as_older": int((permanent & (mask == OLDER)).sum()), "water_px": int((mask == WATER).sum()),
            "water_never": int(((mask == WATER) & never).sum()), "burn_px": int(burn.sum())}


def run_water(args):
    from shapely.geometry import box
    tiles = released(args)
    folder = Path(args.gsw)
    folder.mkdir(parents=True, exist_ok=True)
    needed, per_tile = set(), []
    for r in tiles.itertuples():
        names = {gsw_name(lon, lat) for lon in (r.lon_min, r.lon_max) for lat in (r.lat_min, r.lat_max)}
        needed |= names
        per_tile.append(names)
    for x, y in sorted(needed):
        path = folder / f"occurrence_{x}_{y}.tif"
        if not path.exists():
            print(f"Downloading GSW occurrence {x} {y} ...", flush=True)
            urllib.request.urlretrieve(GSW.format(x=x, y=y), path.with_suffix(".part"))
            path.with_suffix(".part").replace(path)
    jobs = [(str(Path(args.masks_root) / r.mask_path), [str(folder / f"occurrence_{x}_{y}.tif") for x, y in names])
            for r, names in zip(tiles.itertuples(), per_tile)]
    table = pd.DataFrame(pooled(water_job, jobs, args.workers))
    table["tile_id"] = tiles["tile_id"].to_numpy()
    OUT.mkdir(parents=True, exist_ok=True)
    table.to_csv(OUT / "water_tiles.csv", index=False)
    s = table.sum(numeric_only=True)
    write("water", ["# Labels against JRC Global Surface Water (Pekel et al., 2016; occurrence 1984-2021)", "",
                    "Permanent water: occurrence of 90 % or more, resampled to the tile grid by nearest neighbour.", "",
                    f"- Permanent-water pixels: {int(s['permanent_px']):,}; labelled water {pct(s['permanent_as_water'], s['permanent_px'], 1)} %, "
                    f"fresh or old burn {pct(s['permanent_as_burn'], s['permanent_px'], 1)} %, clear {pct(s['permanent_as_clear'], s['permanent_px'], 1)} %, "
                    f"older_burn {pct(s['permanent_as_older'], s['permanent_px'], 1)} % (the rest is cloud, shadow or no data).",
                    f"- Burn pixels on permanent water: {int(s['permanent_as_burn']):,}, {pct(s['permanent_as_burn'], s['burn_px'], 3)} % of all burn "
                    f"pixels, in {int((table['permanent_as_burn'] > 0).sum()):,} tiles. Overriding burn with permanent water would move these pixels.",
                    f"- Water pixels where GSW never saw water: {int(s['water_never']):,}, {pct(s['water_never'], s['water_px'], 1)} % of the water class "
                    "(new water, NDWI on wet or dark ground, or the 30 m GSW grid missing narrow water)."])


# ---------------------------------------------------------------- dNBR "none"

def evidence_table(path):
    """A burn_evidence.py output folder, or a burn evidence CSV such as data/release/burn_evidence_old_window.csv."""
    path = Path(path)
    return pd.read_csv(path if path.is_file() else path / "burn_evidence.csv", low_memory=False)


def run_none(args):
    if args.make_list:
        tiles = released(args)
        none = tiles[(tiles["window"] == "fresh") & (tiles["dnbr_flag"] == "none")][["tile_id"]]
        OUT.mkdir(parents=True, exist_ok=True)
        none.to_csv(OUT / "none_tiles.csv", index=False)
        print(f"{len(none):,} fresh-window tiles flagged none: {OUT / 'none_tiles.csv'}. Now:\n  python tools/burn_evidence.py "
              f"--tiles-from {OUT / 'none_tiles.csv'} --out validation/dnbr_none --workers 16")
        return
    rows = evidence_table(args.none_dnbr).drop_duplicates("tile_id", keep="last")
    rows = rows[rows["flag"].isin(["burn", "weak", "none"]) & rows["nbr_pre_ring"].notna()]
    rows["pre_contrast"] = rows["nbr_pre_ring"] - rows["nbr_pre_in"]
    rows["post_contrast"] = rows["nbr_post_ring"] - rows["nbr_post_in"]
    rows["kind"] = np.select([rows["pre_contrast"] >= .1, rows["post_contrast"] < .1],
                             ["scar already on the pre-fire image (date suspect)", "no change in either image (label suspect)"],
                             "darker after the fire but under the dNBR threshold")
    tiles = released(args).set_index("tile_id")
    rows["size"] = pd.cut(rows["tile_id"].map(tiles["area_ha"]), [5, 10, 30, 100, 500, 2001], right=False,
                          labels=["5-10 ha", "10-30 ha", "30-100 ha", "100-500 ha", "500-2000 ha"])
    table = rows.groupby(["size", "kind"], observed=True).size().unstack(fill_value=0)
    table.insert(0, "Tiles", table.sum(axis=1))
    counts = rows["kind"].value_counts()
    write("none", ["# Fires the dNBR flags none, split", "",
                   "Contrast: the 100-300 m ring's median NBR minus the perimeter's, before and after the fire; 0.10 or more means "
                   "the inside is darker, as a scar is.", "",
                   *[f"- {kind}: {n:,} of {len(rows):,} ({pct(n, len(rows), 1)} %)" for kind, n in counts.items()], "",
                   md(table.reset_index())])


# ---------------------------------------------------------------- old window, same season

def standing(rows):
    rows = rows[rows["flag"].isin(["burn", "weak", "none"])]
    return {"Tiles": len(rows), "Contrast >= 0.05 %": pct((rows["contrast"] >= .05).sum(), len(rows), 1),
            "Contrast >= 0.10 %": pct((rows["contrast"] >= .10).sum(), len(rows), 1),
            "Inside median >= 0.10 %": pct((rows["dnbr_in_median"] >= .10).sum(), len(rows), 1),
            "Median inside": f"{rows['dnbr_in_median'].median():.3f}", "Median ring": f"{rows['dnbr_ring_median'].median():.3f}"}


def run_old(args):
    prefire = evidence_table(args.old_dnbr).drop_duplicates("tile_id", keep="last")
    anniversary = evidence_table(args.anniversary_dnbr).drop_duplicates("tile_id", keep="last")
    both = set(prefire.loc[prefire["flag"].isin(["burn", "weak", "none"]), "tile_id"]) & set(
        anniversary.loc[anniversary["flag"].isin(["burn", "weak", "none"]), "tile_id"])
    tiles = released(args).set_index("tile_id")
    lines = ["# Old-window scars: pre-fire against same-season reference", "",
             "Pre-fire: the most recent valid image before the fire, usually summer. Same season: the image nearest to a year "
             "before the old scene, within 30 days and before the fire, so phenology cancels. Contrast is the dNBR inside the "
             "perimeter minus the ring's.", ""]
    rows = []
    for name, frame in (("pre-fire reference", prefire), ("same-season reference", anniversary)):
        rows.append({"Reference": name, "Tiles in both": "", **standing(frame[frame["tile_id"].isin(both)])})
    lines += [md(pd.DataFrame(rows)), "", "By scene month:", ""]
    monthly = []
    for name, frame in (("pre-fire", prefire), ("same season", anniversary)):
        f = frame[frame["tile_id"].isin(both)].assign(month=lambda d: pd.to_datetime(d["tile_id"].map(tiles["scene_time"]),
                                                                                      format="mixed", utc=True).dt.month)
        for month, part in f.groupby("month"):
            monthly.append({"Reference": name, "Month": month, **standing(part)})
    lines += [md(pd.DataFrame(monthly))]
    write("old", lines)


# ---------------------------------------------------------------- misc

def zero_job(job):
    image_path, mask_path = job
    with rasterio.open(image_path) as src:
        image = src.read()
    with rasterio.open(mask_path) as src:
        valid = src.read(1) != NODATA
    return ([int(((image[b] == 0) & valid).sum()) for b in range(image.shape[0])]
            + [int(((image == 0).any(axis=0) & valid).sum()), int(valid.sum())])


def run_misc(args):
    tiles = released(args)
    lines = ["# Other numbers the referee asks for", ""]

    # Two passes in one image
    if "product_id" in tiles:
        def passes(value):
            names = [n for n in str(value).split(";") if n and n != "nan"]
            return len({(n.split("_")[0], n.split("_")[2][:15]) for n in names if len(n.split("_")) > 2})
        count = tiles["product_id"].map(passes)
        lines += [f"## Two passes in one image", "", f"{int((count > 1).sum()):,} of {len(tiles):,} tiles "
                  f"({pct((count > 1).sum(), len(tiles), 1)} %) combine products from more than one pass (satellite and sensing "
                  "time); they get the column two_pass.", ""]

    # Zeros inside valid pixels, on a sample of tiles
    sample = tiles.sample(min(args.zero_sample, len(tiles)), random_state=0)
    sample = sample[[(Path(args.images_root) / p).exists() for p in sample["image_path"]]]   # fetch_images.py --for zeros
    jobs = [(str(Path(args.images_root) / r.image_path), str(Path(args.masks_root) / r.mask_path)) for r in sample.itertuples()]
    zeros = np.array(pooled(zero_job, jobs, args.workers))
    bands = ["B01", "B02", "B03", "B04", "B05", "B06", "B07", "B08", "B8A", "B09", "B10", "B11", "B12"]
    lines += ["## Zero values inside valid pixels", "",
              f"{len(sample):,} random tiles. 0 is no data in the images; Sentinel Hub sets negative reflectance to 0, so a valid "
              "pixel can read 0 in a band.", "",
              md(pd.DataFrame([{"Band": b, "Valid pixels at 0 %": pct(zeros[:, i].sum(), zeros[:, -1].sum(), 4),
                                "Tiles with any": int((zeros[:, i] > 0).sum())} for i, b in enumerate(bands)])), "",
              f"Valid pixels at 0 in at least one band: {pct(zeros[:, -2].sum(), zeros[:, -1].sum(), 3)} %, in "
              f"{int((zeros[:, -2] > 0).sum()):,} of {len(sample):,} tiles.", ""]

    # Nearest train tile to each test tile
    import geopandas as gpd
    from shapely.geometry import box
    shapes = gpd.GeoSeries([box(r.lon_min, r.lat_min, r.lon_max, r.lat_max) for r in tiles.itertuples()], crs=4326).to_crs(3035)
    train = shapes[(tiles["split"] == "train").to_numpy()]
    test = shapes[(tiles["split"] == "test").to_numpy()]
    nearest = train.sindex.nearest(test.values, return_distance=True, return_all=False)[1]
    lines += ["## Distance from each test tile to the nearest training tile", "",
              f"{len(test):,} test tiles: median {np.median(nearest) / 1000:.1f} km, 5th percentile {np.percentile(nearest, 5) / 1000:.2f} km; "
              f"{pct((nearest == 0).sum(), len(test), 1)} % touch a training tile's edge (the splits keep overlapping tiles together, so "
              f"none share pixels); {pct((nearest < 1000).sum(), len(test), 1)} % lie within 1 km.", ""]

    # Overlap groups
    if "group" in tiles:
        groups = tiles.groupby("group").agg(fires=("fire_id", "nunique"), tiles=("tile_id", "size"),
                                            split=("split", lambda s: s.mode().iloc[0]), countries=("country", lambda s: ",".join(sorted(set(s)))))
        top = groups.sort_values("fires", ascending=False).head(3).reset_index()
        lines += ["## Largest overlap groups", "", md(top), "",
                  f"The largest holds {pct(top.loc[0, 'tiles'], (tiles['split'] == top.loc[0, 'split']).sum(), 1)} % of the "
                  f"{top.loc[0, 'split']} tiles.", ""]

    # The time split's test set against the spatial split's
    if "split_time" in tiles:
        rows = []
        for name, part in (("split test", tiles[tiles["split"] == "test"]), ("split_time test", tiles[tiles["split_time"] == "test"])):
            rows.append({"Test set": name, "Tiles": len(part), **{f"{c} %": pct((part["country"] == c).sum(), len(part), 1) for c in ("IT", "ES", "FR", "EL")},
                         "Median fire ha": f"{part['area_ha'].median():.0f}", "fresh_burn % of pixels": f"{100 * part['frac_fresh'].mean():.2f}",
                         "Fire years": f"{int(part['fire_year'].min())}-{int(part['fire_year'].max())}"})
        lines += ["## The time split's test set", "", md(pd.DataFrame(rows)), ""]

    # Strata coverage of the review
    fresh_px = tiles["frac_fresh"] * 65536
    outside = ~tiles["age_days"].between(5, 61, inclusive="left")
    valid = (1 - tiles["frac_nodata"]) * 65536
    lines += ["## Review strata against the class shares", "",
              f"fresh_burn on tiles whose scene is outside the 5-60-day bins of the fresh strata (fresh burn of another fire on an "
              f"old-window tile, mostly): {pct(fresh_px[outside].sum(), valid.sum(), 3)} % of valid pixels. These pixels are in no "
              "stratum, which is why the strata's map totals differ slightly from the class shares.", ""]

    # Review counts the text quotes as shares
    review = Path(args.review)
    if (review / "points.csv").exists():
        from review_points import latest_answers
        points = pd.read_csv(review / "points.csv", low_memory=False)
        first = points[points["repeat_of"] < 0]
        rows = []
        for i, name in enumerate(args.reviewers, 1):
            answers = latest_answers(review / f"answers_{name}.csv")
            for stratum in ("clear_near", "clear_far"):
                ids = first.loc[first["stratum"] == stratum, "point_id"]
                given = [answers.get(p) for p in ids if answers.get(p) not in (None, "cant_tell")]
                rows.append({"Interpreter": i, "Stratum": stratum, "Answered": len(given), "burned": given.count("burned"),
                             "burned_before": given.count("burned_before")})
        lines += ["## Clear points answered burned", "", md(pd.DataFrame(rows)), ""]

    # CEMS per pair
    pairs_path = Path(args.labels_work) / "cems_dnbr_pairs.csv"
    if pairs_path.exists():
        pairs = pd.read_csv(pairs_path)
        pairs = pairs[pairs.get("error", pd.Series(np.nan, index=pairs.index)).isna()] if "error" in pairs else pairs
        f1 = 2 * pairs["label_tp"] / (2 * pairs["label_tp"] + pairs["label_fp"] + pairs["label_fn"]).replace(0, np.nan)
        area = pairs["tile_id"].map(tiles.set_index("tile_id")["area_ha"])
        sizes = pd.cut(area, [5, 30, 100, 500, 2001], right=False, labels=["5-30 ha", "30-100 ha", "100-500 ha", "500-2000 ha"])
        by_size = pd.DataFrame({"f1": f1, "fire size": sizes}).groupby("fire size", observed=True)["f1"].agg(["size", "median"])
        lines += ["## CEMS, pair by pair", "",
                  f"{len(pairs)} pairs: per-pair F1 of the fresh_burn labels median {100 * f1.median():.1f} % (quartiles "
                  f"{100 * f1.quantile(.25):.1f}-{100 * f1.quantile(.75):.1f}).", "",
                  md(by_size.reset_index().rename(columns={"size": "Pairs", "median": "Median F1"}).assign(**{"Median F1": lambda d: (100 * d["Median F1"]).round(1)})), ""]

    # Snow counts
    snow_path = Path(args.labels_work) / "snow_tiles.csv"
    if snow_path.exists():
        snow = pd.read_csv(snow_path)
        lines += ["## Snow: tiles counted", ""]
        for window, part in snow.groupby("window"):
            errors = int(part["error"].notna().sum()) if "error" in part else 0
            empty = int((part["own_px"].fillna(0) == 0).sum()) - errors
            lines.append(f"- {window}: {len(part):,} tiles read, {errors} unreadable, {empty} with no valid pixel of their own "
                         f"perimeter; {len(part) - errors - empty:,} counted.")
        lines.append("")

    # Eligible fires that never got a tile
    log_path = Path(args.processing_log)
    if log_path.exists():
        log = pd.read_csv(log_path)
        missing = log[~log["fire_id"].isin(set(tiles["fire_id"]))]
        lines += ["## Fires processed without a tile", "", md(missing.groupby(["country", "status"]).size().rename("Fires").reset_index()), ""]
    write("misc", lines)


def run_columns(args):
    """Columns for the 5.0 index: history_days, older_2y_frac, two_pass, cold_season_fire, likely_managed_burn."""
    tiles = released(args)
    out = tiles[["tile_id"]].copy()
    history = OUT / "history_tiles.csv"
    if history.exists():
        h = pd.read_csv(history).set_index("tile_id")
        out["history_days"] = out["tile_id"].map(h["history_days"])
        out["older_2y_frac"] = out["tile_id"].map(h["older_2y_px"] / 65536).round(5)
    if "product_id" in tiles:
        out["two_pass"] = tiles["product_id"].map(lambda v: len({(n.split("_")[0], n.split("_")[2][:15]) for n in str(v).split(";")
                                                                  if len(n.split("_")) > 2}) > 1).to_numpy()
    month = pd.to_datetime(tiles["fire_date"]).dt.month
    out["cold_season_fire"] = month.isin([11, 12, 1, 2, 3]).to_numpy()
    burns = OUT / "burns_fires.csv"
    if burns.exists():
        b = pd.read_csv(burns).set_index("fire_id")
        out["likely_managed_burn"] = tiles["fire_id"].map(b["likely_managed"]).to_numpy()
        out["reburn_frac"] = tiles["fire_id"].map(b["reburn"]).round(3).to_numpy()
    out.to_csv(OUT / "index_columns.csv", index=False)
    print(f"Wrote {OUT / 'index_columns.csv'}: {', '.join(out.columns[1:])}")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("part", choices=["cloud", "history", "burns", "water", "none", "old", "misc", "columns"])
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--index", default="data/v4/index.csv")
    parser.add_argument("--images-root", default="data/v1")
    parser.add_argument("--masks-root", default="data/v4")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--gsw", default="data/gsw", help="where the JRC occurrence tiles are kept")
    parser.add_argument("--make-list", action="store_true", help="none: write the list of tiles to run burn_evidence.py on")
    parser.add_argument("--none-dnbr", default="validation/dnbr_none")
    parser.add_argument("--old-dnbr", default="validation/dnbr_old")
    parser.add_argument("--anniversary-dnbr", default="validation/dnbr_old_anniversary")
    parser.add_argument("--zero-sample", type=int, default=2000)
    parser.add_argument("--review", default="review2")
    parser.add_argument("--reviewers", nargs=2, default=["param", "silvia"])
    parser.add_argument("--labels-work", default="validation/labels", help="make_label_figures.py --work")
    parser.add_argument("--processing-log", default="output/processing_index.csv")
    args = parser.parse_args()
    {"cloud": run_cloud, "history": run_history, "burns": run_burns, "water": run_water, "none": run_none, "old": run_old,
     "misc": run_misc, "columns": run_columns}[args.part](args)


if __name__ == "__main__":
    main()
