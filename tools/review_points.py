#!/usr/bin/env python3
"""Label review, second edition: two people judge random pixels from the imagery alone, blind to the mask.

Design, after Olofsson et al. (2014) and Stehman and Foody (2019):
  - Pixels are drawn at random within strata, with every pixel of a stratum equally likely, so the
    sample has known inclusion probabilities and the scores are estimates for the whole data set.
  - Strata are the mask classes, with fresh_burn split by days from the fire to the scene (5-20, 20-40,
    40-60) and clear split by distance to the nearest burn (within 300 m, farther), so the groups the
    paper reports are designed in, not read off afterwards.
  - The reviewer sees a window centred on the point, never the whole tile, so the tile centre cannot
    give the fire away. Up to three dates are shown in time order with their dates only: no caption says
    which date is the fire, and the mask and the EFFIS perimeter are never shown.
  - Two reviewers answer every point independently, each in their own random order. Their agreement is
    reported. A third person adjudicates the points where they disagree and the answers the dNBR contradicts
    (review_adjudicate.py); her answer is the final label wherever she gave one, and the agreed answer elsewhere.
  - The score step gives the full error matrix with area-weighted overall, user's and producer's accuracy,
    with bootstrap intervals that respect the strata, besides the per-class agreement and the groups.

Steps:

  1. Draw the sample. Reads masks only. 500 points by default; --counts changes any stratum.
     python tools/review_points.py sample --index data/v3/index.csv --masks-root data/v3 --out review2

  2. Fetch a pre-fire image for every sampled fire (Sentinel Hub, about 1 request per fire).
     python tools/fetch_prefire.py --index data/v3/index.csv --points review2/points.csv --out data/prefire

  3. One picture per point.
     python tools/review_points.py chips --out review2 --index data/v3/index.csv --images-root data/v1 --prefire-root data/prefire

  4. Each reviewer, on their own port, answers every point. Answers save on every key press.
     python tools/review_points.py serve --out review2 --reviewer param --port 8765
     python tools/review_points.py serve --out review2 --reviewer silvia --port 8766
     On the laptop: ssh -L 8765:localhost:8765 you@server, then open http://localhost:8765

  5. Adjudicate the disagreements and the answers the dNBR contradicts (a third person), on a page served here
     or, after review_dnbr.py, as a folder that opens in any browser:
     python tools/review_points.py serve --out review2 --reviewer judge --disagreements param silvia --port 8767
     python tools/review_adjudicate.py --out review2 --reviewers param silvia --judge judge

  6. Score, at any time; the final label needs both reviewers and the judge.
     python tools/review_points.py score --out review2 --reviewers param silvia --judge judge

  7. Files for the release, with the answer columns named by role (interpreter1, interpreter2, adjudicator).
     python tools/review_points.py export --out review2 --reviewers param silvia --judge judge

  8. The overall accuracy under the two scoring rules that bound it.
     python tools/review_bounds.py --out review2 --reviewers param silvia --judge judge
"""
import argparse
import csv
import json
import threading
import time
import zlib
from concurrent.futures import ProcessPoolExecutor
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio
from rasterio.warp import transform as warp_transform
from scipy.ndimage import distance_transform_edt

MASK_CLEAR, MASK_FRESH, MASK_OLD, MASK_CLOUD, MASK_SHADOW, MASK_WATER, MASK_OLDER = 0, 2, 3, 4, 5, 6, 7
NEAR_PX = 30                      # 300 m: clear pixels this close to a burn are where edge errors live.
FRESH_DAY_BINS = ((5, 20), (20, 40), (40, 61))   # days from the fire to the scene; the last bin takes 60.5
STRATA = ["fresh_early", "fresh_mid", "fresh_late", "old", "clear_near", "clear_far", "water", "cloud", "shadow", "older"]
DEFAULT_COUNTS = {"fresh_early": 50, "fresh_mid": 50, "fresh_late": 50, "old": 100, "clear_near": 75, "clear_far": 75,
                  "water": 35, "cloud": 35, "shadow": 30, "older": 0}
MAP_CLASS = {"fresh_early": "burned", "fresh_mid": "burned", "fresh_late": "burned", "old": "burned",
             "clear_near": "not_burned", "clear_far": "not_burned", "water": "water", "cloud": "cloud", "shadow": "shadow",
             "older": "older_burn"}   # older_burn is right when the place had burned (any burned answer)


def expected(map_class):
    """The reference answer that agrees with a map class."""
    return "burned" if map_class == "older_burn" else map_class
REF_CLASSES = ["burned", "not_burned", "water", "cloud", "shadow"]
ANSWERS = {"1": "burned", "2": "burned_before", "3": "not_burned", "4": "water", "5": "cloud", "6": "shadow", "7": "cant_tell"}
BURNED_ANSWERS = ("burned", "burned_before")
BAND = {name: i for i, name in enumerate(
    ["B01", "B02", "B03", "B04", "B05", "B06", "B07", "B08", "B8A", "B09", "B10", "B11", "B12"])}
FALSE_COLOUR = (("B12", .40), ("B8A", .50), ("B04", .28))
TRUE_COLOUR = (("B04", .28), ("B03", .28), ("B02", .32))
PANEL, ZOOM_HALF, CONTEXT_HALF = 320, 32, 96    # 640 m and 1.92 km windows, both centred on the point
FILL = (40, 40, 44)


# ---------------------------------------------------------------------- sample

def strata_of(mask, age_days):
    """Boolean mask per stratum for one tile. The fresh bins depend on the tile's days from the fire."""
    burn = np.isin(mask, [MASK_FRESH, MASK_OLD])
    clear = mask == MASK_CLEAR
    near = distance_transform_edt(~burn) <= NEAR_PX if burn.any() else np.zeros(mask.shape, bool)
    fresh = mask == MASK_FRESH
    out = {name: np.zeros(mask.shape, bool) for name in ("fresh_early", "fresh_mid", "fresh_late")}
    for name, (low, high) in zip(("fresh_early", "fresh_mid", "fresh_late"), FRESH_DAY_BINS):
        if low <= age_days < high:
            out[name] = fresh
    out.update({"old": mask == MASK_OLD, "clear_near": clear & near, "clear_far": clear & ~near,
                "water": mask == MASK_WATER, "cloud": mask == MASK_CLOUD, "shadow": mask == MASK_SHADOW,
                "older": mask == MASK_OLDER})
    return out


def _count_tile(job):
    path, age_days = job
    with rasterio.open(path) as src:
        mask = src.read(1)
    return {name: int(chosen.sum()) for name, chosen in strata_of(mask, age_days).items()}


def _pick_point(job):
    path, age_days, stratum, seed = job
    with rasterio.open(path) as src:
        mask, transform, crs = src.read(1), src.transform, src.crs
    rows, cols = np.nonzero(strata_of(mask, age_days)[stratum])
    rng = np.random.default_rng(seed)
    pick = int(rng.integers(len(rows)))
    row, col = int(rows[pick]), int(cols[pick])
    x, y = transform * (col + .5, row + .5)
    lon, lat = warp_transform(crs, "EPSG:4326", [x], [y])
    burn = np.isin(mask, [MASK_FRESH, MASK_OLD])
    # Distance to the edge of the burn, in metres: inside the burn for burn points, to the nearest burn otherwise.
    if burn[row, col]:
        edge = distance_transform_edt(burn)[row, col]
    elif burn.any():
        edge = distance_transform_edt(~burn)[row, col]
    else:
        edge = np.nan
    return {"row": row, "col": col, "lon": round(lon[0], 6), "lat": round(lat[0], 6),
            "mask_value": int(mask[row, col]), "edge_m": None if np.isnan(edge) else round(10 * float(edge), 1)}


def run_sample(args):
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    previous = None
    if (out / "points.csv").exists() and args.supplement:
        previous = pd.read_csv(out / "points.csv", low_memory=False)
        if "batch" not in previous:
            previous["batch"] = 1
    elif (out / "points.csv").exists() and not args.overwrite:
        raise SystemExit(f"{out}/points.csv already exists. A new sample would throw away the old one; "
                         "pass --overwrite if you really want that, or --supplement to add a batch.")
    index = pd.read_csv(args.index, low_memory=False)
    index = index[index["split"].notna()].reset_index(drop=True)   # released tiles only
    paths = [str(Path(args.masks_root) / p) for p in index["mask_path"]]
    ages = index["age_days"].astype(float).tolist()
    with ProcessPoolExecutor(args.workers) as pool:
        counts = pd.DataFrame(list(pool.map(_count_tile, zip(paths, ages), chunksize=32)))
    areas = counts.sum()
    areas.rename("pixels").rename_axis("stratum").reset_index().to_csv(out / "strata_area.csv", index=False)

    wanted = dict(DEFAULT_COUNTS) if previous is None else {name: 0 for name in STRATA}
    for item in args.counts or []:
        name, number = item.split("=")
        if name not in wanted:
            raise SystemExit(f"Unknown stratum {name}. Strata: {', '.join(STRATA)}")
        wanted[name] = int(number)
    rng = np.random.default_rng(args.seed)
    jobs, rows = [], []
    for stratum, number in wanted.items():
        weights = counts[stratum].to_numpy(float)
        available = int((weights > 0).sum())
        number = min(number, available)
        if number == 0:
            continue
        # A tile's chance follows its pixel count, then a uniform pixel, so every pixel of the stratum is equally likely.
        chosen = rng.choice(len(index), number, replace=False, p=weights / weights.sum())
        for tile in chosen:
            record = index.iloc[tile]
            jobs.append((paths[tile], float(record["age_days"]), stratum, int(rng.integers(2 ** 32))))
            rows.append({"tile_id": record["tile_id"], "fire_id": record["fire_id"], "scene_time": record["scene_time"],
                         "image_path": record["image_path"], "country": record["country"], "window": record["window"],
                         "fire_year": record["fire_year"], "fire_date": record.get("fire_date", ""),
                         "age_days": record["age_days"], "area_ha": record.get("area_ha", np.nan),
                         "land_cover": record.get("land_cover", ""), "stratum": stratum})
    with ProcessPoolExecutor(args.workers) as pool:
        picks = list(pool.map(_pick_point, jobs, chunksize=8))
    points = pd.DataFrame([{**row, **pick} for row, pick in zip(rows, picks)])

    # A few points shown twice to each reviewer, to measure one person's consistency.
    repeats = points.sample(frac=args.repeat, random_state=args.seed).copy()
    repeats["repeat_of"] = repeats.index
    points["repeat_of"] = -1
    points = pd.concat([points, repeats], ignore_index=False).reset_index(names="source_row")
    taken = set() if previous is None else set(previous["point_id"])
    ids = [f"p{i:06d}" for i in rng.permutation(10 ** 6) if f"p{i:06d}" not in taken][:len(points)]
    points.insert(0, "point_id", ids)
    points = points.sample(frac=1, random_state=args.seed).reset_index(drop=True)
    if previous is not None:
        # A second random draw within the same strata: within each stratum the points stay an equal-probability sample.
        same = points.merge(previous[["tile_id", "row", "col"]], on=["tile_id", "row", "col"], how="left", indicator=True)["_merge"] == "both"
        dropped = set(points.loc[same.to_numpy() & (points["repeat_of"] < 0).to_numpy(), "source_row"])
        points = points[~points["source_row"].isin(dropped) & ~points["repeat_of"].isin(dropped)].copy()
        base = int(previous["source_row"].max()) + 1
        points["source_row"] += base
        points.loc[points["repeat_of"] >= 0, "repeat_of"] += base
        points["batch"] = int(previous["batch"].max()) + 1
        points = pd.concat([previous, points], ignore_index=True)
        print(f"Added batch {points['batch'].max()}: {int((points['batch'] == points['batch'].max()).sum()):,} showings"
              f"{f', {len(dropped)} duplicates of earlier points dropped' if dropped else ''}.")
    else:
        points["batch"] = 1
    points.to_csv(out / "points.csv", index=False)

    table = pd.DataFrame({"pixels": areas, "share_%": (100 * areas / areas.sum()).round(3),
                          "points": points[points["repeat_of"] < 0]["stratum"].value_counts()})
    print(table.fillna(0).to_string())
    print(f"\n{len(points):,} showings of {points['source_row'].nunique():,} points in {points['tile_id'].nunique():,} tiles; "
          f"about {len(points) * 30 / 3600:.1f} hours per reviewer at 30 seconds a point.")
    print(f"Don't open {out}/points.csv before the review: it holds the mask classes.")


# ----------------------------------------------------------------------- chips

def read_bands(path):
    with rasterio.open(path) as src:
        return src.read().astype(np.float32) / 10000.0


def compose(image, recipe):
    channels = [np.clip(image[BAND[name]] / limit, 0, 1) ** .85 for name, limit in recipe]
    return np.round(np.stack(channels, -1) * 255).astype(np.uint8)


def crop_centred(rgb, row, col, half):
    """A (2 half) square with the point at its centre; outside the tile is filled with grey, so the point never moves."""
    size = rgb.shape[0]
    out = np.empty((2 * half, 2 * half, 3), np.uint8)
    out[:] = FILL
    r0, c0 = row - half, col - half
    sr0, sc0 = max(r0, 0), max(c0, 0)
    sr1, sc1 = min(r0 + 2 * half, size), min(c0 + 2 * half, size)
    out[sr0 - r0:sr1 - r0, sc0 - c0:sc1 - c0] = rgb[sr0:sr1, sc0:sc1]
    return out


def panel(image, recipe, row, col, half, font, caption):
    from PIL import Image, ImageDraw
    crop = crop_centred(compose(image, recipe), row, col, half)
    scale = PANEL / (2 * half)
    picture = Image.fromarray(crop).resize((PANEL, PANEL), Image.NEAREST if half <= ZOOM_HALF else Image.BILINEAR)
    draw = ImageDraw.Draw(picture)
    # A hollow square around the pixel, so the pixel itself stays visible.
    reach = 1 if half <= ZOOM_HALF else 3
    x0, y0 = (half - reach) * scale, (half - reach) * scale
    x1, y1 = (half + 1 + reach) * scale, (half + 1 + reach) * scale
    draw.rectangle([x0 - 1, y0 - 1, x1 + 1, y1 + 1], outline=(0, 0, 0))
    draw.rectangle([x0, y0, x1, y1], outline=(255, 235, 0))
    return with_caption(picture, caption, font)


def panel_blank(font, caption):
    from PIL import Image
    return with_caption(Image.new("RGB", (PANEL, PANEL), FILL), caption, font)


def with_caption(picture, caption, font):
    from PIL import Image, ImageDraw
    canvas = Image.new("RGB", (PANEL, PANEL + 24), (18, 18, 20))
    canvas.paste(picture, (0, 24))
    ImageDraw.Draw(canvas).text((6, 5), caption, fill=(235, 235, 235), font=font)
    return canvas


def find_font():
    from PIL import ImageFont
    for name in ("DejaVuSans.ttf", "LiberationSans-Regular.ttf", "Arial.ttf"):
        try:
            return ImageFont.truetype(name, 14)
        except OSError:
            continue
    return ImageFont.load_default()


_CHIP = {}


def _init_chips(images_root, prefire_root, out, other_scene):
    _CHIP.update(images_root=Path(images_root), prefire_root=Path(prefire_root) if prefire_root else None,
                 out=Path(out), other_scene=other_scene, font=find_font())


def prefire_for(fire_id):
    root = _CHIP["prefire_root"]
    if root is None:
        return None, ""
    found = sorted((root / "acquisitions" / fire_id).glob("*_prefire_s2.tif"))
    if not found:
        return None, ""
    return found[0], found[0].name.split("_prefire")[0].split("_")[-1][:10]


def make_chip(point):
    """Three dates in time order, captioned by date only, then true colour and two context windows."""
    from PIL import Image
    font = _CHIP["font"]
    row, col = int(point["row"]), int(point["col"])
    this_date = str(point["scene_time"])[:10]
    dates = [(this_date, read_bands(_CHIP["images_root"] / point["image_path"]), True)]
    pre_path, pre_date = prefire_for(point["fire_id"])
    if pre_path:
        dates.append((pre_date, read_bands(pre_path), False))
    other = _CHIP["other_scene"].get(point["tile_id"])
    if other:
        dates.append((other[1], read_bands(_CHIP["images_root"] / other[0]), False))
    dates.sort(key=lambda d: d[0])
    this = next(d for d in dates if d[2])

    top = [panel(img, FALSE_COLOUR, row, col, ZOOM_HALF, font, f"{'THIS DATE' if is_this else 'Date'}  {date}")
           for date, img, is_this in dates]
    while len(top) < 3:
        top.append(panel_blank(font, "No other image"))
    earliest = dates[0]
    bottom = [panel(this[1], TRUE_COLOUR, row, col, ZOOM_HALF, font, f"THIS DATE, true colour  {this[0]}"),
              panel(this[1], FALSE_COLOUR, row, col, CONTEXT_HALF, font, f"THIS DATE, 1.9 km  {this[0]}"),
              panel(earliest[1], FALSE_COLOUR, row, col, CONTEXT_HALF, font,
                    f"{'THIS DATE' if earliest[2] else 'Date'}, 1.9 km  {earliest[0]}")]
    panels = top + bottom
    width, height = panels[0].size
    sheet = Image.new("RGB", (3 * width + 8, 2 * height + 4), (18, 18, 20))
    for i, picture in enumerate(panels):
        sheet.paste(picture, ((i % 3) * (width + 4), (i // 3) * (height + 4)))
    sheet.save(_CHIP["out"] / "chips" / f"{point['point_id']}.jpg", quality=88)
    return {"point_id": point["point_id"], "dates_shown": len(dates), "earliest_date": dates[0][0],
            "prefire_shown": bool(pre_path)}


def run_chips(args):
    out = Path(args.out)
    points = pd.read_csv(out / "points.csv", low_memory=False)
    (out / "chips").mkdir(exist_ok=True)
    index = pd.read_csv(args.index, low_memory=False)
    other_scene = {}
    # The same place on the fire's other date, nearest in time.
    for _, group in index[index["fire_id"].isin(points["fire_id"])].groupby("fire_id"):
        if len(group) < 2:
            continue
        for rec in group.itertuples():
            others = group[group["tile_id"] != rec.tile_id]
            gap = (pd.to_datetime(others["scene_time"]) - pd.Timestamp(rec.scene_time)).abs()
            best = others.loc[gap.idxmin()]
            other_scene[rec.tile_id] = (best["image_path"], str(best["scene_time"])[:10])
    records = points.to_dict("records")
    with ProcessPoolExecutor(args.workers, initializer=_init_chips,
                             initargs=(args.images_root, args.prefire_root, out, other_scene)) as pool:
        shown = pd.DataFrame(list(pool.map(make_chip, records, chunksize=8)))
    shown.to_csv(out / "chips_shown.csv", index=False)
    size = sum(p.stat().st_size for p in (out / "chips").glob("*.jpg")) / 1e6
    print(f"{len(records):,} pictures in {out}/chips, {size:,.0f} MB. {int(shown['prefire_shown'].sum()):,} show a pre-fire image; "
          f"dates shown: {shown['dates_shown'].value_counts().sort_index().to_dict()}.")


# ----------------------------------------------------------------------- serve

PAGE = """<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>burn2scar review</title>
<style>
:root { color-scheme: dark; }
* { box-sizing: border-box; }
body { margin: 0; background: #121214; color: #ececec; font: 15px/1.4 system-ui, sans-serif; display: flex; flex-direction: column; height: 100vh; }
header { display: flex; gap: 16px; align-items: baseline; padding: 8px 14px; flex-wrap: wrap; }
header b { font-size: 16px; }
#question { flex: 1; min-width: 240px; }
#progress { color: #a8a8b0; font-variant-numeric: tabular-nums; }
main { flex: 1; min-height: 0; display: flex; justify-content: center; align-items: center; padding: 0 8px; }
img { max-width: 100%; max-height: 100%; object-fit: contain; display: block; }
footer { display: flex; gap: 8px; padding: 10px 14px; justify-content: center; flex-wrap: wrap; align-items: center; }
button { background: #26262b; color: inherit; border: 1px solid #3c3c44; border-radius: 6px; padding: 9px 14px; font: inherit; cursor: pointer; min-height: 44px; }
button:hover { background: #33333a; }
button.primary { border-color: #e0a030; }
button kbd { color: #e0a030; font: inherit; font-weight: 700; margin-right: 6px; }
details { padding: 0 14px 8px; color: #c8c8d0; max-width: 1040px; margin: 0 auto; width: 100%; }
#done { display: none; text-align: center; padding: 40px 16px; font-size: 18px; }
</style></head><body>
<header><b>burn2scar review</b>
<span id="question">Had the spot in the yellow square burned at any time up to THIS DATE? Ignore pictures after THIS DATE.</span>
<span id="progress"></span></header>
<main><img id="chip" alt="Satellite views of the point under review"><div id="done">All points answered. You can close this tab.</div></main>
<footer id="keys"></footer>
<details><summary>How to answer</summary>
<p>The top row shows the same spot on up to three dates, oldest on the left; the dates are the only captions. Judge only the pixel in the middle of the yellow square. In false colour, healthy vegetation is green and fresh burn is a dark red or brown patch; pale pink, lilac or tan is usually soil, dry grass or fields, above all if it looks the same on every date or follows field edges.</p>
<p><b>Burned:</b> a scar shows at this spot on THIS DATE or on an earlier date, even if the spot looks green again by THIS DATE.
<b>Burned, already on the earliest date:</b> the same, and the scar is already there on the leftmost image.
<b>Not burned:</b> no sign of fire by THIS DATE; a scar that appears only on a later date counts as not burned.
<b>Water, Cloud, Shadow:</b> the pixel is water, or cloud or any shadow, from a cloud or a mountain, covers it on THIS DATE. Real water stays dark on the other dates too.
<b>Can't tell:</b> the images aren't enough, or snow hides the ground. Use it freely: a guess is worse than no answer.</p>
<p>Backspace goes back one point. Your answers save on every key press.</p>
</details>
<script>
const CHOICES = [["1","burned","Burned"],["2","burned_before","Burned, already on the earliest date"],["3","not_burned","Not burned"],["4","water","Water"],["5","cloud","Cloud"],["6","shadow","Shadow"],["7","cant_tell","Can't tell"]];
let ids = [], answers = {}, at = 0, shownAt = 0, times = [];
const chip = document.getElementById("chip"), progress = document.getElementById("progress"), done = document.getElementById("done");
const keys = document.getElementById("keys");
for (const [key, value, label] of CHOICES) {
  const button = document.createElement("button");
  button.className = "primary";
  button.innerHTML = `<kbd>${key}</kbd>${label}`;
  button.onclick = () => answer(value);
  keys.append(button);
}
const back = document.createElement("button");
back.innerHTML = "<kbd>&#9003;</kbd>Back";
back.onclick = goBack;
keys.append(back);
function firstOpen() { const i = ids.findIndex(id => !(id in answers)); return i < 0 ? ids.length : i; }
function label() {
  const count = Object.keys(answers).length;
  let text = `${count.toLocaleString()} of ${ids.length.toLocaleString()} answered`;
  if (times.length >= 5) {
    const sorted = [...times].sort((a, b) => a - b);
    const typical = sorted[Math.floor(sorted.length / 2)];
    const minutes = Math.round((ids.length - count) * typical / 60000);
    text += `, about ${minutes >= 60 ? (minutes / 60).toFixed(1) + " hours" : minutes + " min"} left`;
  }
  progress.textContent = text;
}
function preload() { for (let i = at + 1; i <= at + 4 && i < ids.length; i++) new Image().src = `/chips/${ids[i]}.jpg`; }
function show() {
  label();
  if (at >= ids.length) { chip.style.display = "none"; done.style.display = "block"; return; }
  chip.style.display = "block"; done.style.display = "none";
  chip.src = `/chips/${ids[at]}.jpg`;
  shownAt = performance.now();
  preload();
}
async function answer(value) {
  if (at >= ids.length) return;
  const id = ids[at], ms = Math.round(performance.now() - shownAt);
  answers[id] = value;
  if (ms < 120000) times.push(ms);
  at += 1;
  while (at < ids.length && ids[at] in answers) at += 1;
  show();
  try {
    const response = await fetch("/answer", {method: "POST", headers: {"Content-Type": "application/json"}, body: JSON.stringify({id, answer: value, ms})});
    if (!response.ok) throw new Error(response.status);
  } catch (error) {
    progress.textContent = "Could not save the last answer. Check the server, then reload this page.";
  }
}
function goBack() { if (at > 0) { at -= 1; show(); } }
document.addEventListener("keydown", event => {
  if (event.metaKey || event.ctrlKey || event.altKey) return;
  const choice = CHOICES.find(c => c[0] === event.key);
  if (choice) { event.preventDefault(); answer(choice[1]); }
  else if (event.key === "Backspace" || event.key === "ArrowLeft") { event.preventDefault(); goBack(); }
});
chip.onerror = () => { progress.textContent = "This picture did not load. Reload the page."; };
fetch("/state").then(r => r.json()).then(state => { ids = state.ids; answers = state.answers; at = firstOpen(); show(); });
</script></body></html>"""


def latest_answers(path):
    answers = {}
    if path.exists():
        with open(path, newline="") as handle:
            for row in csv.DictReader(handle):
                answers[row["point_id"]] = row["answer"]
    return answers


def reviewer_order(points, reviewer):
    """Every reviewer gets their own random order, so two people never move through the sample in step."""
    rng = np.random.default_rng(zlib.crc32(reviewer.encode()))
    ids = points["point_id"].tolist()
    return [ids[i] for i in rng.permutation(len(ids))]


def disagreements(out, points, names):
    """First showings where the named reviewers gave different answers, or one of them has not answered."""
    tables = [latest_answers(out / f"answers_{name}.csv") for name in names]
    first = points[points["repeat_of"] < 0]["point_id"]
    return [pid for pid in first if len({t.get(pid) for t in tables}) > 1 and all(pid in t for t in tables)]


def run_serve(args):
    out = Path(args.out)
    points = pd.read_csv(out / "points.csv", low_memory=False)
    if args.disagreements:
        ids = disagreements(out, points, args.disagreements)
        print(f"{len(ids):,} points where {' and '.join(args.disagreements)} disagree.")
    else:
        ids = reviewer_order(points, args.reviewer)
    known = set(ids)
    answers_path = out / f"answers_{args.reviewer}.csv"
    if not answers_path.exists():
        answers_path.write_text("point_id,answer,ms,saved_at\n")
    lock = threading.Lock()
    allowed = set(ANSWERS.values())
    chips = (out / "chips").resolve()

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *_):
            pass

        def send_body(self, body, content_type, status=200, cache=False):
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "max-age=86400" if cache else "no-store")
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            path = self.path.split("?")[0]
            if path == "/":
                self.send_body(PAGE.encode(), "text/html; charset=utf-8")
            elif path == "/state":
                with lock:
                    answered = {k: v for k, v in latest_answers(answers_path).items() if k in known}
                self.send_body(json.dumps({"ids": ids, "answers": answered}).encode(), "application/json")
            elif path.startswith("/chips/") and Path(path).stem in known and path.endswith(".jpg"):
                file = chips / f"{Path(path).stem}.jpg"
                if file.exists():
                    self.send_body(file.read_bytes(), "image/jpeg", cache=True)
                else:
                    self.send_body(b"missing", "text/plain", 404)
            else:
                self.send_body(b"not found", "text/plain", 404)

        def do_POST(self):
            if self.path != "/answer":
                self.send_body(b"not found", "text/plain", 404)
                return
            data = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))) or b"{}")
            if data.get("id") not in known or data.get("answer") not in allowed:
                self.send_body(b"bad answer", "text/plain", 400)
                return
            with lock, open(answers_path, "a", newline="") as handle:
                csv.writer(handle).writerow([data["id"], data["answer"], int(data.get("ms", 0)), int(time.time())])
            self.send_body(b"{}", "application/json")

    server = ThreadingHTTPServer((args.host, args.port), Handler)
    print(f"{len(ids):,} points for {args.reviewer}. Answers go to {answers_path}.")
    print(f"From your laptop: ssh -L {args.port}:localhost:{args.port} you@your-server")
    print(f"Then open http://localhost:{args.port}. Press Ctrl+C here when you finish.", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass



# ----------------------------------------------------------------------- score

def wilson(hits, total, z=1.96):
    if total == 0:
        return np.nan, np.nan
    p = hits / total
    centre = (p + z * z / (2 * total)) / (1 + z * z / total)
    half = z * np.sqrt(p * (1 - p) / total + z * z / (4 * total * total)) / (1 + z * z / total)
    return max(centre - half, 0.0), min(centre + half, 1.0)


def interval(hits, total):
    low, high = wilson(hits, total)
    return f"{100 * hits / total:.1f} ({100 * low:.1f}-{100 * high:.1f})" if total else ""


def md(frame):
    head = "| " + " | ".join(map(str, frame.columns)) + " |"
    rule = "|" + "---|" * len(frame.columns)
    body = ["| " + " | ".join("" if pd.isna(v) else str(v) for v in row) + " |" for row in frame.itertuples(index=False)]
    return "\n".join([head, rule, *body])


def reference_class(answer):
    return "burned" if answer in BURNED_ANSWERS else answer


def error_matrix(counts, weights):
    """Area-weighted error matrix from per-stratum answer counts (Olofsson et al. 2014, eq. 1 generalised to strata).

    counts: {stratum: {reference class: n}}, cant_tell already left out. weights: pixel share per stratum.
    Returns the matrix of area proportions, map classes as rows, reference classes as columns, and the accuracies.
    """
    strata = [s for s in counts if s in MAP_CLASS and sum(counts[s].values()) > 0]
    total_weight = sum(weights[s] for s in strata)
    classes = [c for c in REF_CLASSES if any(MAP_CLASS[s] == c for s in strata)]
    classes += ["older_burn"] if any(MAP_CLASS[s] == "older_burn" for s in strata) else []
    matrix = pd.DataFrame(0.0, index=classes, columns=REF_CLASSES)
    for s in strata:
        n = sum(counts[s].values())
        for ref, k in counts[s].items():
            matrix.at[MAP_CLASS[s], ref] += weights[s] / total_weight * k / n
    overall = float(sum(matrix.at[c, expected(c)] for c in classes))
    user = {c: matrix.at[c, expected(c)] / matrix.loc[c].sum() if matrix.loc[c].sum() else np.nan for c in classes}
    producer = {c: matrix.at[c, c] / matrix[c].sum() if c in matrix.columns and matrix[c].sum() else np.nan for c in classes}
    area = {c: matrix[c].sum() for c in REF_CLASSES}
    return matrix, {"overall": overall, "user": user, "producer": producer, "area": area}


def burn_summary(matrix):
    """Burned against not burned, from the matrix: precision, recall, F1, IoU. Water, cloud and shadow rows are not burned.
    The older_burn row is left out, as in the model evaluation: these scores cover the area outside older_burn."""
    matrix = matrix.drop(index="older_burn", errors="ignore")
    both = matrix.at["burned", "burned"]
    map_burned = matrix.loc["burned"].sum()
    ref_burned = matrix["burned"].sum()
    precision = both / map_burned if map_burned else np.nan
    recall = both / ref_burned if ref_burned else np.nan
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else np.nan
    iou = both / (map_burned + ref_burned - both) if map_burned + ref_burned - both else np.nan
    return {"precision": precision, "recall": recall, "f1": f1, "iou": iou, "map_share": map_burned, "ref_share": ref_burned}


def bootstrap_matrix(counts, weights, repeats, seed):
    """Resample answers within each stratum (multinomial), which keeps the design, and collect every statistic."""
    rng = np.random.default_rng(seed)
    rows = []
    for _ in range(repeats):
        drawn = {}
        for s, table in counts.items():
            n = sum(table.values())
            if n == 0:
                continue
            refs = list(table)
            sample = rng.multinomial(n, [table[r] / n for r in refs])
            drawn[s] = dict(zip(refs, sample.tolist()))
        matrix, acc = error_matrix(drawn, weights)
        row = {"overall": acc["overall"]}
        row.update({f"user_{c}": v for c, v in acc["user"].items()})
        row.update({f"producer_{c}": v for c, v in acc["producer"].items()})
        row.update({f"area_{c}": v for c, v in acc["area"].items()})
        if "burned" in matrix.index:
            row.update({f"burn_{k}": v for k, v in burn_summary(matrix).items()})
        rows.append(row)
    table = pd.DataFrame(rows)
    return table.quantile(.025), table.quantile(.975)


def pct(value, low=None, high=None):
    if pd.isna(value):
        return ""
    return f"{100 * value:.1f}" if low is None else f"{100 * value:.1f} ({100 * low:.1f}-{100 * high:.1f})"


def kappa(a, b):
    """Cohen's kappa for two sequences of labels."""
    labels = sorted(set(a) | set(b))
    table = pd.crosstab(pd.Categorical(a, labels), pd.Categorical(b, labels)).reindex(index=labels, columns=labels, fill_value=0)
    n = table.to_numpy().sum()
    observed = np.trace(table.to_numpy()) / n
    expected = float((table.sum(axis=1) * table.sum(axis=0)).sum()) / (n * n)
    return (observed - expected) / (1 - expected) if expected < 1 else np.nan


def final_labels(points, out, reviewers, judge):
    """One answer per first showing: the judge's where she answered (the disagreements, and the answers the dNBR
    contradicts, from review_adjudicate.py), else the agreed answer, else none."""
    tables = {name: latest_answers(out / f"answers_{name}.csv") for name in reviewers}
    judged = latest_answers(out / f"answers_{judge}.csv") if judge else {}
    final = {}
    for pid in points[points["repeat_of"] < 0]["point_id"]:
        given = [tables[name].get(pid) for name in reviewers]
        if any(g is None for g in given):
            continue
        if pid in judged:
            final[pid] = judged[pid]
        elif len(set(given)) == 1:
            final[pid] = given[0]
    return final


def score_one(points, weights, answers, title, bootstrap_n):
    """Markdown and tables for one set of answers (one reviewer, or the final labels)."""
    lines = [f"## {title}", ""]
    first = points[points["repeat_of"] < 0].copy()
    first["answer"] = first["point_id"].map(answers)
    first = first[first["answer"].notna()]
    if first.empty:
        return lines + ["No answers yet.", ""], None
    first["ref"] = first["answer"].map(reference_class)

    # Per stratum: what was answered, and agreement with the map class, with Wilson intervals.
    rows = []
    for stratum in STRATA:
        part = first[first["stratum"] == stratum]
        if part.empty:
            continue
        shares = part["answer"].value_counts(normalize=True)
        row = {"Stratum": stratum, "Map class": MAP_CLASS.get(stratum, "older_burn"), "Points": len(part),
               "Share of data set %": round(100 * weights[stratum], 2)}
        row.update({name: f"{100 * shares.get(name, 0):.1f}" for name in ANSWERS.values()})
        seen = part[part["answer"] != "cant_tell"]
        if stratum in MAP_CLASS:
            hits = int((seen["ref"] == expected(MAP_CLASS[stratum])).sum())
            row["Agreement %"] = interval(hits, len(seen))
        else:
            row["Agreement %"] = f"{int((seen['ref'] == 'burned').sum())} of {len(seen)} judged burned"
        rows.append(row)
    lines += ["Percent of answers per stratum. Agreement is the share of answers, can't tell left out, that match the map class, "
              "with a 95 % Wilson interval. 'burned_before' means burned, with the scar already on the earliest date shown.", "",
              md(pd.DataFrame(rows)), ""]

    # The error matrix, area-weighted, with bootstrap intervals.
    usable = first[(first["answer"] != "cant_tell") & first["stratum"].isin(MAP_CLASS)]
    counts = {s: usable[usable["stratum"] == s]["ref"].value_counts().to_dict() for s in MAP_CLASS}
    counts = {s: {r: int(k) for r, k in t.items()} for s, t in counts.items() if sum(t.values())}
    matrix, acc = error_matrix(counts, weights)
    low, high = bootstrap_matrix(counts, weights, bootstrap_n, 0)
    shown = matrix.copy() * 100
    shown["Map total"] = shown.sum(axis=1)
    shown.loc["Reference total"] = shown.sum(axis=0)
    shown = shown.round(2).rename_axis("Map class \\ reference").reset_index()
    lines += ["### Error matrix, percent of the data set's area", "",
              f"From {len(usable):,} points with a usable answer ({int((first['answer'] == 'cant_tell').sum())} can't tell left out). "
              "Rows are the map classes, columns what the reviewer saw. Each cell is the area share estimated from the stratified sample, "
              "so the cells that agree (the diagonal, and the burned cell of the older_burn row) add up to the overall accuracy.", "", md(shown), ""]
    rows = [{"Measure": "Overall accuracy", "Estimate % (95 % interval)": pct(acc["overall"], low["overall"], high["overall"])}]
    for c in matrix.index:
        rows.append({"Measure": f"User's accuracy, {c} (precision)", "Estimate % (95 % interval)": pct(acc["user"][c], low[f"user_{c}"], high[f"user_{c}"])})
        if c in REF_CLASSES:
            rows.append({"Measure": f"Producer's accuracy, {c} (recall)", "Estimate % (95 % interval)": pct(acc["producer"][c], low[f"producer_{c}"], high[f"producer_{c}"])})
    for c in REF_CLASSES:
        rows.append({"Measure": f"Area of {c}, reviewer's estimate", "Estimate % (95 % interval)": pct(acc["area"][c], low[f"area_{c}"], high[f"area_{c}"])})
    summary = burn_summary(matrix) if "burned" in matrix.index else {}
    for key, label in (("precision", "Burned precision"), ("recall", "Burned recall"), ("f1", "Burned F1"), ("iou", "Burned IoU"),
                       ("map_share", "Burned share, map"), ("ref_share", "Burned share, reviewer")):
        if key in summary:
            rows.append({"Measure": label, "Estimate % (95 % interval)": pct(summary[key], low[f"burn_{key}"], high[f"burn_{key}"])})
    lines += ["### Accuracies from the matrix", "", "Bootstrap intervals from resampling within each stratum.", "", md(pd.DataFrame(rows)), ""]

    # Groups, for the burn strata and the clear strata, with intervals.
    burn = first[first["stratum"].isin([s for s in MAP_CLASS if MAP_CLASS[s] == "burned"]) & (first["answer"] != "cant_tell")].copy()
    burn["size"] = pd.cut(burn["area_ha"].astype(float), [0, 10, 30, 100, 500, np.inf],
                          labels=["5-10 ha", "10-30 ha", "30-100 ha", "100-500 ha", "500-2000 ha"])
    burn["days"] = pd.cut(burn["age_days"].astype(float), [0, 20, 40, 61, 150, 225, 301],
                          labels=["5-20 d", "20-40 d", "40-60 d", "120-150 d", "150-225 d", "225-300 d"])
    burn["edge"] = pd.cut(burn["edge_m"].astype(float), [0, 15, 30, 100, np.inf],
                          labels=["within 15 m", "15-30 m", "30-100 m", "over 100 m"])
    month = pd.to_datetime(burn["fire_date"], errors="coerce").dt.month
    burn["season"] = month.map({12: "Dec-Feb", 1: "Dec-Feb", 2: "Dec-Feb", 3: "Mar-May", 4: "Mar-May", 5: "Mar-May",
                                6: "Jun-Aug", 7: "Jun-Aug", 8: "Jun-Aug", 9: "Sep-Nov", 10: "Sep-Nov", 11: "Sep-Nov"})
    rows = []
    for column, label in (("window", "Window"), ("country", "Country"), ("size", "Fire size"), ("days", "Days from the fire"),
                          ("edge", "Distance inside the burn edge"), ("season", "Season of the fire"), ("fire_year", "Fire year"),
                          ("land_cover", "Land cover")):
        if column not in burn:
            continue
        for value, part in burn.groupby(column, observed=True):
            hits = int((part["ref"] == "burned").sum())
            rows.append({"Group": label, "Value": value, "Points": len(part), "Confirmed burned %": interval(hits, len(part))})
    lines += ["### Burn labels confirmed, by group", "", "fresh and old strata together. Wilson 95 % intervals; the sizes say how much each group can carry.", "",
              md(pd.DataFrame(rows)), ""]
    clear = first[first["stratum"].isin(["clear_near", "clear_far"]) & (first["answer"] != "cant_tell")]
    rows = []
    for stratum, part in clear.groupby("stratum"):
        hits = int((part["ref"] == "burned").sum())
        rows.append({"Stratum": stratum, "Points": len(part), "Judged burned %": interval(hits, len(part))})
    lines += ["### Burn seen where the map says clear", "", md(pd.DataFrame(rows)), ""]

    # The date check: among burned answers, how often the scar was already on the earliest date shown.
    rows = []
    for stratum, part in first[first["ref"] == "burned"].groupby("stratum"):
        hits = int((part["answer"] == "burned_before").sum())
        rows.append({"Stratum": stratum, "Burned answers": len(part), "Scar already on the earliest date %": interval(hits, len(part))})
    lines += ["### Scar already present on the earliest date shown", "",
              "For burn strata this flags a scar older than the pre-fire image, so a fire date later than the scar, or an older burn at the same spot.", "",
              md(pd.DataFrame(rows)), ""]
    return lines, {"matrix": matrix, "accuracies": acc, "low": low, "high": high}


def run_score(args):
    out = Path(args.out)
    points = pd.read_csv(out / "points.csv", low_memory=False)
    area = pd.read_csv(out / "strata_area.csv").set_index("stratum")["pixels"]
    weights = (area / area.sum()).to_dict()
    lines = ["# burn2scar label review, second edition", ""]
    hours = {}
    per_reviewer = {}
    for name in args.reviewers:
        path = out / f"answers_{name}.csv"
        if not path.exists():
            lines.append(f"- {name}: no answers yet")
            continue
        data = pd.read_csv(path)
        hours[name] = data["ms"].clip(upper=120000).sum() / 3.6e6
        per_reviewer[name] = latest_answers(path)
        shown_once = points[points["repeat_of"] < 0]["point_id"]
        lines.append(f"- {name}: {sum(pid in per_reviewer[name] for pid in shown_once):,} of {len(shown_once):,} points, {hours[name]:.1f} hours")
    lines.append("")

    # Agreement between reviewers, on first showings answered by all.
    if len(per_reviewer) >= 2:
        names = list(per_reviewer)
        first = points[points["repeat_of"] < 0]
        table = pd.DataFrame({name: first["point_id"].map(per_reviewer[name]).to_numpy() for name in names}, index=first["point_id"].to_numpy()).dropna()
        if len(table):
            full = table.nunique(axis=1).eq(1)
            coarse = table.apply(lambda col: col.map(reference_class)).nunique(axis=1).eq(1)
            lines += ["## Agreement between reviewers", "",
                      f"- Same answer on {int(full.sum()):,} of {len(table):,} points: {interval(int(full.sum()), len(table))} %",
                      f"- Same class (burned, not burned, water, cloud, shadow, can't tell): {interval(int(coarse.sum()), len(table))} %",
                      f"- Cohen's kappa on the class: {kappa(table[names[0]].map(reference_class), table[names[1]].map(reference_class)):.3f}" if len(names) == 2 else "", ""]
            by_stratum = []
            strata = first.set_index("point_id").loc[table.index, "stratum"]
            for stratum, idx in strata.groupby(strata).groups.items():
                part = coarse.loc[idx]
                by_stratum.append({"Stratum": stratum, "Points": len(part), "Same class %": interval(int(part.sum()), len(part))})
            lines += [md(pd.DataFrame(by_stratum)), ""]
            if args.judge:
                judged = latest_answers(out / f"answers_{args.judge}.csv")
                open_ones = [pid for pid in table.index[~full] if pid not in judged]
                lines += [f"- Disagreements: {int((~full).sum()):,}; adjudicated: {int((~full).sum()) - len(open_ones):,}; still open: {len(open_ones):,}", ""]

    # One person's consistency, on the repeats.
    for name, answers in per_reviewer.items():
        repeats = points[points["repeat_of"] >= 0]
        pairs = []
        for rec in repeats.itertuples():
            source = points[(points["source_row"] == rec.repeat_of) & (points["repeat_of"] < 0)]
            if len(source) and rec.point_id in answers and source.iloc[0]["point_id"] in answers:
                pairs.append(answers[rec.point_id] == answers[source.iloc[0]["point_id"]])
        if pairs:
            lines += [f"- {name}, same point shown twice: {interval(sum(pairs), len(pairs))} % the same answer, {len(pairs)} points", ""]

    results = {}
    final = final_labels(points, out, list(per_reviewer), args.judge) if len(per_reviewer) >= 2 else {}
    if final:
        text, results["final"] = score_one(points, weights, final, f"Final labels ({len(final):,} points: agreed, or adjudicated by {args.judge or 'nobody yet'})", args.bootstrap)
        lines += text
    for name, answers in per_reviewer.items():
        text, results[name] = score_one(points, weights, answers, f"Reviewer {name}", args.bootstrap)
        lines += text
    text = "\n".join(lines)
    (out / "review_summary.md").write_text(text)
    for name, result in results.items():
        if result:
            (result["matrix"] * 100).round(3).to_csv(out / f"error_matrix_{name}.csv")
    print(text)
    print(f"Wrote {out}/review_summary.md and the error matrices as CSV.")


def run_export(args):
    """reviewed_points.csv and review_strata.csv in <out>/release: every point, each reviewer's answer, the final label.

    The answer columns are named by role, not by reviewer: answer_interpreter1, answer_interpreter2 (in the order of
    --reviewers), answer_adjudicator (--judge) and answer_final."""
    out = Path(args.out)
    points = pd.read_csv(out / "points.csv", low_memory=False)
    shown = pd.read_csv(out / "chips_shown.csv") if (out / "chips_shown.csv").exists() else None
    table = points.copy()
    roles = {name: f"interpreter{number}" for number, name in enumerate(args.reviewers, 1)}
    if args.judge:
        roles[args.judge] = "adjudicator"
    for name, role in roles.items():
        table[f"answer_{role}"] = table["point_id"].map(latest_answers(out / f"answers_{name}.csv"))
    final = final_labels(points, out, args.reviewers, args.judge)
    table["answer_final"] = table["point_id"].map(final)
    if shown is not None:
        table = table.merge(shown, on="point_id", how="left")
    dnbr_path = out / "review_dnbr_points.csv"
    if dnbr_path.exists():   # the dNBR at each point, on its scene and on the fire's first scene (review_dnbr.py)
        dnbr = pd.read_csv(dnbr_path, usecols=lambda c: c in {"point_id", "dnbr", "dnbr_fresh"}).drop_duplicates("point_id")
        table = table.merge(dnbr.rename(columns={"dnbr_fresh": "dnbr_first_scene"}), on="point_id", how="left")
    if args.judge:
        table["adjudicated"] = table["answer_adjudicator"].notna()
    first_showing = table[table["repeat_of"] < 0].set_index("source_row")["point_id"]
    table["repeat_of"] = [first_showing.get(source, "") if source >= 0 else "" for source in table["repeat_of"]]
    columns = ["point_id", "repeat_of", "tile_id", "fire_id", "scene_time", "country", "window", "fire_year", "fire_date", "age_days",
               "area_ha", "land_cover", "stratum", "mask_value", "row", "col", "lon", "lat", "edge_m", "dates_shown", "earliest_date",
               "prefire_shown", "dnbr", "dnbr_first_scene", "adjudicated"] + [c for c in table.columns if c.startswith("answer_")]
    table = table[[c for c in columns if c in table.columns]]
    target = out / "release"
    target.mkdir(exist_ok=True)
    table.to_csv(target / "reviewed_points.csv", index=False)
    area = pd.read_csv(out / "strata_area.csv")
    area["share_percent"] = (100 * area["pixels"] / area["pixels"].sum()).round(4)
    area["map_class"] = area["stratum"].map(MAP_CLASS).fillna("older_burn")
    area.to_csv(target / "review_strata.csv", index=False)
    once = table[table["repeat_of"] == ""]
    print(once["stratum"].value_counts().rename("points").to_string())
    print(f"{len(once):,} points, {len(table) - len(once):,} shown a second time, {int(table['answer_final'].notna().sum()):,} with a final label.")
    print(f"Wrote {target / 'reviewed_points.csv'} and {target / 'review_strata.csv'}")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    commands = parser.add_subparsers(dest="command", required=True)
    sample = commands.add_parser("sample")
    sample.add_argument("--index", default="data/v3/index.csv")
    sample.add_argument("--masks-root", default="data/v3")
    sample.add_argument("--out", default="review2")
    sample.add_argument("--counts", nargs="*", help="change a stratum's size, e.g. old=120 older=20")
    sample.add_argument("--repeat", type=float, default=.05, help="share of points shown twice to each reviewer")
    sample.add_argument("--seed", type=int, default=2026)
    sample.add_argument("--workers", type=int, default=8)
    sample.add_argument("--overwrite", action="store_true")
    sample.add_argument("--supplement", action="store_true", help="add a batch to the existing sample; --counts gives the extra points per stratum")
    chips = commands.add_parser("chips")
    chips.add_argument("--out", default="review2")
    chips.add_argument("--index", default="data/v3/index.csv", help="finds the other date of each fire")
    chips.add_argument("--images-root", default="data/v1")
    chips.add_argument("--prefire-root", default="")
    chips.add_argument("--workers", type=int, default=8)
    serve = commands.add_parser("serve")
    serve.add_argument("--out", default="review2")
    serve.add_argument("--reviewer", required=True)
    serve.add_argument("--disagreements", nargs=2, metavar="REVIEWER", help="serve only the points where these two disagree")
    serve.add_argument("--host", default="127.0.0.1")
    serve.add_argument("--port", type=int, default=8765)
    score = commands.add_parser("score")
    score.add_argument("--out", default="review2")
    score.add_argument("--reviewers", nargs="+", required=True)
    score.add_argument("--judge", default="")
    score.add_argument("--bootstrap", type=int, default=2000)
    export = commands.add_parser("export")
    export.add_argument("--out", default="review2")
    export.add_argument("--reviewers", nargs="+", required=True)
    export.add_argument("--judge", default="")
    args = parser.parse_args()
    {"sample": run_sample, "chips": run_chips, "serve": run_serve, "score": run_score, "export": run_export}[args.command](args)


if __name__ == "__main__":
    main()
