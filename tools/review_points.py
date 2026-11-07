#!/usr/bin/env python3
"""Independent check of burn2scar labels: a person judges random pixels from the imagery alone.

The design follows Olofsson et al. (2014). Pixels are drawn at random from each map class,
and the reviewer never sees the mask or EFFIS. Results are weighted by each class's share of
the dataset, so they estimate accuracy over the whole dataset, not just over the sample.

  1. Draw the sample. Reads masks only.
     python tools/review_points.py sample --index data/v2/index.csv --masks-root data/v2 --out review

  2. Optional, recommended: fetch pre-fire images for the sampled tiles.
     python tools/fetch_prefire.py --index data/v2/index.csv --points review/points.csv --out data/prefire

  3. Make one picture per point.
     python tools/review_points.py chips --out review --images-root data/v1 --prefire-root data/prefire

  4. Review. Your answers are saved on every key press, so you can stop and come back at any time.
     python tools/review_points.py serve --out review --reviewer param
     Then on your laptop:  ssh -L 8765:localhost:8765 you@your-server
     and open http://localhost:8765

  5. Score. Works at any point: the order is random, so stopping early only widens the intervals.
     python tools/review_points.py score --out review

  6. Files to publish with the data set: every point with its answer, and the size of each stratum.
     python tools/review_points.py export --out review

To share the work between two people, each runs serve with --part 1/2 or --part 2/2. The first
--shared points (100 by default) go to both people, which measures how much two reviewers agree.
"""
import argparse
import csv
import json
import threading
import time
from concurrent.futures import ProcessPoolExecutor
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio
from rasterio.warp import transform as warp_transform
from scipy.ndimage import distance_transform_edt

MASK_CLEAR, MASK_FRESH, MASK_OLD, MASK_CLOUD, MASK_SHADOW, MASK_WATER, MASK_OLDER = 0, 2, 3, 4, 5, 6, 7
NEAR_PX = 30  # 300 m. Errors on the edges of scars are the ones that matter most.
DEFAULT_COUNTS = {"fresh": 300, "old": 300, "clear_near": 300, "clear_far": 300,
                  "water": 100, "cloud": 100, "shadow": 100, "older": 50}
ANSWERS = {"1": "burned", "2": "not_burned", "3": "water", "4": "cloud", "5": "shadow", "6": "cant_tell"}
BAND = {name: i for i, name in enumerate(
    ["B01", "B02", "B03", "B04", "B05", "B06", "B07", "B08", "B8A", "B09", "B10", "B11", "B12"])}
FALSE_COLOUR = (("B12", .40), ("B8A", .50), ("B04", .28))
TRUE_COLOUR = (("B04", .28), ("B03", .28), ("B02", .32))
PANEL, ZOOM_HALF = 320, 32


# ---------------------------------------------------------------------- sample

def strata_of(mask):
    """Boolean mask per stratum. Near and far clear split on distance to any burn in the tile."""
    burn = np.isin(mask, [MASK_FRESH, MASK_OLD])
    clear = mask == MASK_CLEAR
    if burn.any():
        near = distance_transform_edt(~burn) <= NEAR_PX
    else:
        near = np.zeros(mask.shape, bool)
    return {"fresh": mask == MASK_FRESH, "old": mask == MASK_OLD, "clear_near": clear & near,
            "clear_far": clear & ~near, "water": mask == MASK_WATER, "cloud": mask == MASK_CLOUD,
            "shadow": mask == MASK_SHADOW, "older": mask == MASK_OLDER}


def _count_tile(path):
    with rasterio.open(path) as src:
        mask = src.read(1)
    return {name: int(chosen.sum()) for name, chosen in strata_of(mask).items()}


def _pick_point(job):
    path, stratum, seed = job
    with rasterio.open(path) as src:
        mask, transform, crs = src.read(1), src.transform, src.crs
    rows, cols = np.nonzero(strata_of(mask)[stratum])
    rng = np.random.default_rng(seed)
    pick = int(rng.integers(len(rows)))
    row, col = int(rows[pick]), int(cols[pick])
    x, y = transform * (col + .5, row + .5)
    lon, lat = warp_transform(crs, "EPSG:4326", [x], [y])
    return {"row": row, "col": col, "lon": round(lon[0], 6), "lat": round(lat[0], 6), "mask_value": int(mask[row, col])}


def run_sample(args):
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    if (out / "points.csv").exists() and not args.overwrite:
        raise SystemExit(f"{out}/points.csv already exists. A new sample would throw away the old one; "
                         "pass --overwrite if you really want that.")
    index = pd.read_csv(args.index, low_memory=False)
    index = index[index["split"].notna()].reset_index(drop=True)  # Tiles with no split aren't in the release.
    paths = [str(Path(args.masks_root) / p) for p in index["mask_path"]]
    with ProcessPoolExecutor(args.workers) as pool:
        counts = pd.DataFrame(list(pool.map(_count_tile, paths, chunksize=32)))
    areas = counts.sum()
    (out / "strata_area.csv").write_text(areas.rename("pixels").rename_axis("stratum").reset_index().to_csv(index=False))

    wanted = dict(DEFAULT_COUNTS)
    for item in args.counts or []:
        name, number = item.split("=")
        wanted[name] = int(number)
    rng = np.random.default_rng(args.seed)
    jobs, rows = [], []
    for stratum, number in wanted.items():
        weights = counts[stratum].to_numpy(float)
        available = int((weights > 0).sum())
        number = min(number, available)
        if number == 0:
            print(f"Nothing to sample in {stratum}")
            continue
        # A tile's chance follows its pixel count, then a uniform pixel, so every pixel is equally likely.
        chosen = rng.choice(len(index), number, replace=False, p=weights / weights.sum())
        for tile in chosen:
            jobs.append((paths[tile], stratum, int(rng.integers(2 ** 32))))
            rows.append({"tile_id": index.at[tile, "tile_id"], "fire_id": index.at[tile, "fire_id"],
                         "scene_time": index.at[tile, "scene_time"], "image_path": index.at[tile, "image_path"],
                         "country": index.at[tile, "country"], "window": index.at[tile, "window"],
                         "fire_year": index.at[tile, "fire_year"], "stratum": stratum,
                         "land_cover": index.at[tile, "land_cover"] if "land_cover" in index else ""})
    with ProcessPoolExecutor(args.workers) as pool:
        picks = list(pool.map(_pick_point, jobs, chunksize=8))
    points = pd.DataFrame([{**row, **pick} for row, pick in zip(rows, picks)])

    # Repeats check whether one person answers the same way twice.
    repeats = points.sample(frac=args.repeat, random_state=args.seed).copy()
    repeats["repeat_of"] = repeats.index
    points["repeat_of"] = -1
    points = pd.concat([points, repeats], ignore_index=False).reset_index(names="source_row")
    order = rng.permutation(len(points))
    points = points.iloc[order].reset_index(drop=True)
    ids = rng.choice(10 ** 6, len(points), replace=False)
    points.insert(0, "point_id", [f"p{i:06d}" for i in ids])
    points["position"] = np.arange(len(points))
    points.to_csv(out / "points.csv", index=False)

    table = pd.DataFrame({"pixels": areas, "share_%": (100 * areas / areas.sum()).round(3),
                          "points": points[points["repeat_of"] < 0]["stratum"].value_counts()})
    print(table.fillna(0).to_string())
    minutes = len(points) * 5 / 60
    print(f"\n{len(points):,} points, {points['tile_id'].nunique():,} tiles, about {minutes / 60:.1f} hours at 5 seconds each.")
    print(f"Don't open {out}/points.csv before the review: it holds the mask classes.")


# ----------------------------------------------------------------------- chips

def read_bands(path):
    with rasterio.open(path) as src:
        return src.read().astype(np.float32) / 10000.0


def compose(image, recipe):
    channels = [np.clip(image[BAND[name]] / limit, 0, 1) ** .85 for name, limit in recipe]
    return np.round(np.stack(channels, -1) * 255).astype(np.uint8)


def window_around(row, col, size):
    half = ZOOM_HALF
    top = int(np.clip(row - half, 0, size - 2 * half))
    left = int(np.clip(col - half, 0, size - 2 * half))
    return top, left


def panel_zoom(image, recipe, row, col, font, caption):
    from PIL import Image, ImageDraw
    rgb = compose(image, recipe)
    top, left = window_around(row, col, rgb.shape[0])
    crop = rgb[top:top + 2 * ZOOM_HALF, left:left + 2 * ZOOM_HALF]
    scale = PANEL / (2 * ZOOM_HALF)
    picture = Image.fromarray(crop).resize((PANEL, PANEL), Image.NEAREST)
    draw = ImageDraw.Draw(picture)
    # A hollow square three pixels wide, so the pixel itself stays visible.
    x0, y0 = (col - left - 1) * scale, (row - top - 1) * scale
    x1, y1 = (col - left + 2) * scale, (row - top + 2) * scale
    draw.rectangle([x0 - 1, y0 - 1, x1 + 1, y1 + 1], outline=(0, 0, 0))
    draw.rectangle([x0, y0, x1, y1], outline=(255, 235, 0))
    return with_caption(picture, caption, font)


def panel_whole(image, recipe, row, col, font, caption):
    from PIL import Image, ImageDraw
    rgb = compose(image, recipe)
    scale = PANEL / rgb.shape[0]
    picture = Image.fromarray(rgb).resize((PANEL, PANEL), Image.BILINEAR)
    draw = ImageDraw.Draw(picture)
    x, y = (col + .5) * scale, (row + .5) * scale
    draw.rectangle([x - 11, y - 11, x + 11, y + 11], outline=(0, 0, 0))
    draw.rectangle([x - 10, y - 10, x + 10, y + 10], outline=(255, 235, 0))
    return with_caption(picture, caption, font)


def panel_blank(font, caption):
    from PIL import Image
    return with_caption(Image.new("RGB", (PANEL, PANEL), (40, 40, 44)), caption, font)


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
    from PIL import Image
    font = _CHIP["font"]
    row, col = int(point["row"]), int(point["col"])
    image = read_bands(_CHIP["images_root"] / point["image_path"])
    date = str(point["scene_time"])[:10]

    pre_path, pre_date = prefire_for(point["fire_id"])
    pre = read_bands(pre_path) if pre_path else None
    other_path = _CHIP["other_scene"].get(point["tile_id"])
    other, other_date = None, ""
    if other_path:
        other = read_bands(_CHIP["images_root"] / other_path[0])
        other_date = other_path[1]

    other_word = ("EARLIER date" if other_date < date else "LATER date") if other_date else ""
    panels = [
        panel_zoom(pre, FALSE_COLOUR, row, col, font, f"Before the fire  {pre_date}") if pre is not None else panel_blank(font, "Before the fire: no image"),
        panel_zoom(image, FALSE_COLOUR, row, col, font, f"THIS DATE  {date}"),
        panel_zoom(other, FALSE_COLOUR, row, col, font, f"{other_word}  {other_date}") if other is not None else panel_blank(font, "Other date: no image"),
        panel_zoom(image, TRUE_COLOUR, row, col, font, f"THIS DATE, true colour  {date}"),
        panel_whole(image, FALSE_COLOUR, row, col, font, f"THIS DATE, whole tile  {date}"),
        panel_whole(pre, FALSE_COLOUR, row, col, font, f"Before the fire, whole tile  {pre_date}") if pre is not None
        else (panel_whole(other, FALSE_COLOUR, row, col, font, f"{other_word}, whole tile  {other_date}") if other is not None
              else panel_blank(font, "")),
    ]
    width, height = panels[0].size
    sheet = Image.new("RGB", (3 * width + 8, 2 * height + 4), (18, 18, 20))
    for i, panel in enumerate(panels):
        sheet.paste(panel, ((i % 3) * (width + 4), (i // 3) * (height + 4)))
    sheet.save(_CHIP["out"] / "chips" / f"{point['point_id']}.jpg", quality=88)
    return pre is not None


def run_chips(args):
    out = Path(args.out)
    points = pd.read_csv(out / "points.csv", low_memory=False)
    (out / "chips").mkdir(exist_ok=True)
    index = pd.read_csv(args.index, low_memory=False)
    other_scene = {}
    # The same place on another date, nearest in time, helps most with faded scars.
    for _, group in index[index["fire_id"].isin(points["fire_id"])].groupby("fire_id"):
        if len(group) < 2:
            continue
        for row in group.itertuples():
            others = group[group["tile_id"] != row.tile_id]
            gap = (pd.to_datetime(others["scene_time"]) - pd.Timestamp(row.scene_time)).abs()
            best = others.loc[gap.idxmin()]
            other_scene[row.tile_id] = (best["image_path"], str(best["scene_time"])[:10])
    records = points.to_dict("records")
    with ProcessPoolExecutor(args.workers, initializer=_init_chips,
                             initargs=(args.images_root, args.prefire_root, out, other_scene)) as pool:
        with_prefire = list(pool.map(make_chip, records, chunksize=8))
    size = sum(p.stat().st_size for p in (out / "chips").glob("*.jpg")) / 1e6
    print(f"{len(records):,} pictures in {out}/chips, {size:,.0f} MB. {sum(with_prefire):,} show a pre-fire image.")


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
details { padding: 0 14px 8px; color: #c8c8d0; max-width: 980px; margin: 0 auto; width: 100%; }
#done { display: none; text-align: center; padding: 40px 16px; font-size: 18px; }
</style></head><body>
<header><b>burn2scar review</b>
<span id="question">Was the pixel in the yellow square burned on or before THIS DATE?</span>
<span id="progress"></span></header>
<main><img id="chip" alt="Satellite views of the point under review"><div id="done">All points answered. You can close this tab and run the score step.</div></main>
<footer id="keys"></footer>
<details><summary>How to answer</summary>
<p><b>Burned:</b> a scar shows at this spot on THIS DATE, or on an earlier date, even if the spot looks green again by THIS DATE. In the false-colour views, healthy vegetation is green and fresh burn is a dark red or brown patch. Pale pink, lilac or tan is usually soil, dry grass or fields, above all if it was the same before the fire or follows field edges.
<b>Not burned:</b> no sign of fire by THIS DATE. A scar that appears only on a later date counts as not burned.
<b>Water, Cloud, Shadow:</b> the pixel is water, or cloud or any shadow, from a cloud or a mountain, covers it on THIS DATE. Real water stays dark on the other date too.
<b>Can't tell:</b> the images aren't enough, or snow hides the ground. Use it freely: a guess is worse than no answer.</p>
<p>Judge only the pixel in the middle of the square. Backspace goes back one point. Your answers save on every key press.</p>
</details>
<script>
const CHOICES = [["1","burned","Burned"],["2","not_burned","Not burned"],["3","water","Water"],["4","cloud","Cloud"],["5","shadow","Shadow"],["6","cant_tell","Can't tell"]];
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


def select_ids(points, part, shared):
    ids = points.sort_values("position")["point_id"].tolist()
    if not part:
        return ids
    number, total = (int(x) for x in part.split("/"))
    head, rest = ids[:shared], ids[shared:]
    return head + rest[number - 1::total]


def run_serve(args):
    out = Path(args.out)
    points = pd.read_csv(out / "points.csv", low_memory=False)
    ids = select_ids(points, args.part, args.shared)
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


# ----------------------------------------------------------------- second pass

def run_second_pass(args):
    """The points already answered, in a new order, in their own folder. The first pass isn't touched."""
    source, out = Path(args.source), Path(args.out)
    if (out / "points.csv").exists():
        raise SystemExit(f"{out}/points.csv already exists.")
    points = pd.read_csv(source / "points.csv", low_memory=False)
    answered = latest_answers(source / f"answers_{args.reviewer}.csv")
    keep = points[points["point_id"].isin(answered)].copy()
    keep["position"] = np.random.default_rng(args.seed).permutation(len(keep))
    out.mkdir(parents=True, exist_ok=True)
    keep.sort_values("position").to_csv(out / "points.csv", index=False)
    (out / "strata_area.csv").write_text((source / "strata_area.csv").read_text())
    print(f"{len(keep):,} points in {out}/points.csv, in a new order. Don't open it before the review: it holds the mask classes.")


# ----------------------------------------------------------------------- score

def wilson(hits, total):
    if total == 0:
        return np.nan, np.nan
    z = 1.96
    p = hits / total
    centre = (p + z * z / (2 * total)) / (1 + z * z / total)
    half = z * np.sqrt(p * (1 - p) / total + z * z / (4 * total * total)) / (1 + z * z / total)
    return centre - half, centre + half


MAP_BURNED, MAP_OTHER = ("fresh", "old"), ("clear_near", "clear_far", "water")


def burn_estimates(burned_rate, weights):
    """Area-weighted accuracy for burned against not burned, from a stratified sample (Olofsson et al. 2014).

    burned_rate gives, per stratum, the share of its reviewed points that the reviewer called burned.
    Fresh and old count as burned on the map. Clear and water count as not burned.
    """
    map_burned = sum(weights[s] for s in MAP_BURNED)
    map_other = sum(weights[s] for s in MAP_OTHER)
    both = sum(weights[s] * burned_rate[s] for s in MAP_BURNED)    # On the map and really burned.
    missed = sum(weights[s] * burned_rate[s] for s in MAP_OTHER)   # Not on the map, but really burned.
    total = map_burned + map_other
    user = both / map_burned if map_burned else np.nan
    producer = both / (both + missed) if both + missed else np.nan
    return {
        "precision": user, "recall": producer,
        "f1": 2 * user * producer / (user + producer) if user + producer else np.nan,
        "iou": both / (map_burned + missed) if map_burned + missed else np.nan,
        "overall": (both + (map_other - missed)) / total,
        "burned_share_map": map_burned / total,
        "burned_share_estimated": (both + missed) / total,
    }


def bootstrap(hits, totals, weights, repeats, seed):
    """Resampling within each stratum, which keeps the sample design."""
    rng = np.random.default_rng(seed)
    drawn = {s: rng.binomial(totals[s], hits[s] / totals[s], repeats) / totals[s] for s in totals}
    table = pd.DataFrame([burn_estimates({s: drawn[s][i] for s in totals}, weights) for i in range(repeats)])
    return table.quantile(.025), table.quantile(.975)


def md(frame):
    head = "| " + " | ".join(map(str, frame.columns)) + " |"
    rule = "|" + "---|" * len(frame.columns)
    body = ["| " + " | ".join("" if pd.isna(v) else str(v) for v in row) + " |" for row in frame.itertuples(index=False)]
    return "\n".join([head, rule, *body])


def run_score(args):
    out = Path(args.out)
    points = pd.read_csv(out / "points.csv", low_memory=False)
    area = pd.read_csv(out / "strata_area.csv").set_index("stratum")["pixels"]
    weights = (area / area.sum()).to_dict()
    files = sorted(out.glob("answers_*.csv"))
    if not files:
        raise SystemExit("No answers yet")
    tables = []
    for file in files:
        data = pd.read_csv(file)
        total_ms = data["ms"].clip(upper=120000).sum()
        data = data.drop_duplicates("point_id", keep="last")
        data["reviewer"] = file.stem.replace("answers_", "")
        data.attrs["hours"] = total_ms / 3.6e6
        tables.append(data)
    answers = pd.concat(tables, ignore_index=True)
    merged = answers.merge(points, on="point_id", how="left")
    unknown = merged["stratum"].isna().sum()
    if unknown:
        raise SystemExit(f"{unknown} answers don't match {out}/points.csv")

    # One answer per point for the estimates: the first reviewer's, leaving out repeats.
    first = merged[merged["repeat_of"] < 0].sort_values("reviewer").drop_duplicates("point_id")
    expected = {"fresh": "burned", "old": "burned", "clear_near": "not_burned", "clear_far": "not_burned",
                "water": "water", "cloud": "cloud", "shadow": "shadow"}
    lines = ["# burn2scar label review", ""]
    lines += [f"- Reviewers: {', '.join(sorted(answers['reviewer'].unique()))}",
              f"- Points answered: {len(answers):,} of {len(points):,}",
              f"- Time spent: {sum(t.attrs['hours'] for t in tables):.1f} hours", ""]

    rows = []
    for stratum in DEFAULT_COUNTS:
        part = first[first["stratum"] == stratum]
        if part.empty:
            continue
        shares = part["answer"].value_counts(normalize=True)
        row = {"Mask class": stratum, "Points": len(part), "Share of dataset %": round(100 * weights[stratum], 2)}
        row.update({name: f"{100 * shares.get(name, 0):.1f}" for name in ANSWERS.values()})
        goal = expected.get(stratum)
        if goal:
            seen = part[part["answer"] != "cant_tell"]
            hits = int((seen["answer"] == goal).sum())
            low, high = wilson(hits, len(seen))
            row["Agreement %"] = f"{100 * hits / max(len(seen), 1):.1f} ({100 * low:.1f}-{100 * high:.1f})"
        rows.append(row)
    lines += ["## What the reviewer saw, by mask class",
              "", "Percent of answers. Agreement leaves out `cant_tell`, with a 95% interval.", "", md(pd.DataFrame(rows)), ""]

    # Burned against not burned, area-weighted. Points where the ground can't be seen can't be judged.
    usable = first[first["stratum"].isin(["fresh", "old", "clear_near", "clear_far", "water"])
                   & first["answer"].isin(["burned", "not_burned", "water"])].copy()
    if usable["stratum"].nunique() == 5:
        totals = usable["stratum"].value_counts().to_dict()
        hits = usable[usable["answer"] == "burned"]["stratum"].value_counts().to_dict()
        hits = {s: hits.get(s, 0) for s in totals}
        estimate = burn_estimates({s: hits[s] / totals[s] for s in totals}, weights)
        low, high = bootstrap(hits, totals, weights, args.bootstrap, 0)
        names = {"precision": "Precision (1 - commission error)", "recall": "Recall (1 - omission error)", "f1": "Burnt F1",
                 "iou": "Burnt IoU", "overall": "Overall accuracy", "burned_share_map": "Burned share, from the mask",
                 "burned_share_estimated": "Burned share, estimated from the review"}
        table = pd.DataFrame([{"Measure": label, "Estimate %": f"{100 * estimate[key]:.1f}",
                               "95% interval": f"{100 * low[key]:.1f}-{100 * high[key]:.1f}"} for key, label in names.items()])
        lines += ["## Burned against not burned, weighted by area", "",
                  f"From {len(usable):,} points. Masks count as burned for fresh and old, as not burned for clear and water.", "",
                  md(table), ""]
    else:
        lines += ["Not enough points in every class yet for the area-weighted estimate.", ""]

    # Breakdown for the burn classes.
    rows = []
    for column in ("country", "window", "fire_year", "land_cover"):
        for value, part in first[first["stratum"].isin(["fresh", "old"]) & (first["answer"] != "cant_tell")].groupby(column):
            hits = int((part["answer"] == "burned").sum())
            low, high = wilson(hits, len(part))
            rows.append({"Group": column, "Value": value, "Points": len(part),
                         "Confirmed burned %": f"{100 * hits / len(part):.1f} ({100 * low:.1f}-{100 * high:.1f})"})
    if rows:
        lines += ["## Burn labels confirmed, by group", "", md(pd.DataFrame(rows)), ""]

    # Consistency of one person, on points shown twice.
    repeats = merged[merged["repeat_of"] >= 0]
    pairs = []
    for row in repeats.itertuples():
        source = merged[(merged["source_row"] == row.repeat_of) & (merged["repeat_of"] < 0) & (merged["reviewer"] == row.reviewer)]
        if len(source):
            pairs.append(source.iloc[0]["answer"] == row.answer)
    if pairs:
        low, high = wilson(sum(pairs), len(pairs))
        lines += [f"Same person, same point shown twice: {100 * np.mean(pairs):.1f}% the same answer "
                  f"({100 * low:.1f}-{100 * high:.1f}), {len(pairs)} points.", ""]
    # Agreement between two people.
    if answers["reviewer"].nunique() > 1:
        table = answers.pivot_table(index="point_id", columns="reviewer", values="answer", aggfunc="first").dropna()
        if len(table):
            same = table.nunique(axis=1).eq(1)
            low, high = wilson(int(same.sum()), len(same))
            lines += [f"Two reviewers, same point: {100 * same.mean():.1f}% the same answer ({100 * low:.1f}-{100 * high:.1f}), {len(same)} points.", ""]
    text = "\n".join(lines)
    (out / "review_summary.md").write_text(text)
    print(text)
    print(f"Wrote {out}/review_summary.md")


def run_export(args):
    """reviewed_points.csv and review_strata.csv, in <out>/release. Only what a reader needs to redo the scores."""
    out = Path(args.out)
    points = pd.read_csv(out / "points.csv", low_memory=False)
    answers = pd.read_csv(out / f"answers_{args.reviewer}.csv", dtype=str).drop_duplicates("point_id", keep="last")
    table = points.merge(answers[["point_id", "answer"]], on="point_id", how="left")
    if table["answer"].isna().any():
        print(f"{int(table['answer'].isna().sum()):,} points have no answer; their answer column is empty.")
    first_showing = table[table["repeat_of"] < 0].set_index("source_row")["point_id"]
    table["repeat_of"] = [first_showing.get(source, "") if source >= 0 else "" for source in table["repeat_of"]]
    table["order"] = table["position"] + 1
    columns = ["point_id", "order", "repeat_of", "tile_id", "fire_id", "scene_time", "country", "window", "fire_year",
               "land_cover", "stratum", "mask_value", "row", "col", "lon", "lat", "answer"]
    table = table.sort_values("order")[[c for c in columns if c in table.columns]]
    target = out / "release"
    target.mkdir(exist_ok=True)
    table.to_csv(target / "reviewed_points.csv", index=False)
    area = pd.read_csv(out / "strata_area.csv")
    area["share_percent"] = (100 * area["pixels"] / area["pixels"].sum()).round(4)
    area.to_csv(target / "review_strata.csv", index=False)
    shown_once = table[table["repeat_of"] == ""]
    print(shown_once["stratum"].value_counts().rename("points").to_string())
    print(f"{len(shown_once):,} points, {len(table) - len(shown_once):,} of them shown a second time.")
    print(f"Wrote {target / 'reviewed_points.csv'} and {target / 'review_strata.csv'}")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    commands = parser.add_subparsers(dest="command", required=True)
    sample = commands.add_parser("sample")
    sample.add_argument("--index", default="data/v2/index.csv")
    sample.add_argument("--masks-root", default="data/v2")
    sample.add_argument("--out", default="review")
    sample.add_argument("--counts", nargs="*", help="change a sample size, e.g. fresh=500 clear_far=200")
    sample.add_argument("--repeat", type=float, default=.05, help="share of points shown twice")
    sample.add_argument("--seed", type=int, default=20261104)
    sample.add_argument("--workers", type=int, default=8)
    sample.add_argument("--overwrite", action="store_true")
    chips = commands.add_parser("chips")
    chips.add_argument("--out", default="review")
    chips.add_argument("--index", default="data/v2/index.csv", help="finds the other date of each fire")
    chips.add_argument("--images-root", default="data/v1")
    chips.add_argument("--prefire-root", default="")
    chips.add_argument("--workers", type=int, default=8)
    serve = commands.add_parser("serve")
    serve.add_argument("--out", default="review")
    serve.add_argument("--reviewer", required=True)
    serve.add_argument("--part", default="", help="e.g. 1/2 and 2/2 to share the work between two people")
    serve.add_argument("--shared", type=int, default=100, help="points that both people review when using --part")
    serve.add_argument("--host", default="127.0.0.1")
    serve.add_argument("--port", type=int, default=8765)
    second = commands.add_parser("second-pass")
    second.add_argument("--source", default="review")
    second.add_argument("--out", default="review2")
    second.add_argument("--reviewer", default="param")
    second.add_argument("--seed", type=int, default=20261107)
    score = commands.add_parser("score")
    score.add_argument("--out", default="review")
    score.add_argument("--bootstrap", type=int, default=2000)
    export = commands.add_parser("export")
    export.add_argument("--out", default="review")
    export.add_argument("--reviewer", default="param")
    args = parser.parse_args()
    {"sample": run_sample, "chips": run_chips, "second-pass": run_second_pass, "serve": run_serve, "score": run_score,
     "export": run_export}[args.command](args)


if __name__ == "__main__":
    main()
