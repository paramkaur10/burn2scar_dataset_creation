#!/usr/bin/env python3
"""An independent, date-blind review of the same points, for an interpreter who did not build the data set.

  python tools/review_blind.py chips --out review2 --index data/v4/index.csv --images-root data/v1 --prefire-root data/prefire
  python tools/review_blind.py page --out review2 --reviewer anna
  python tools/review_blind.py score --out review2 --reviewer anna --index data/v4/index.csv

What changes from the first review:
  - No picture is marked as the scene under review. Every point shows its dates in time order (the pre-fire image, the
    fire's fresh-window scene and its old-window scene, as many as exist), labelled Date 1, 2, 3 with their dates, so
    the position of the reviewed scene says nothing about its window.
  - The interpreter answers every date: burned (a scar is visible), not burned, water, cloud, cloud shadow, or can't
    tell. So she says when the scar first appears, and the dates are checked as well as the classes.
Each sheet holds the dates side by side in three rows: false colour 640 m, true colour 640 m, false colour 1.9 km.

score reads her answer on the reviewed scene's date for the classes (area-weighted, as review_points.py scores) and her
answers on all dates for the date check: a point labelled fresh_burn, old_burn or older_burn is date-consistent when
every date before the perimeter date is answered not burned and the first date after it burned. Can't-tell, cloud and
shadow answers make a date uncheckable rather than wrong. Writes <out>/review_blind_<reviewer>.md.
"""
import argparse
import json
import shutil
import sys
import zipfile
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import review_points as rp  # noqa: E402

ANSWERS = {"1": "burned", "2": "not_burned", "3": "water", "4": "cloud", "5": "shadow", "6": "cant_tell"}


def blind_chip(point):
    from PIL import Image
    chip = rp._CHIP
    font = chip["font"]
    row, col = int(point["row"]), int(point["col"])
    dates = [(str(point["scene_time"])[:10], rp.read_bands(chip["images_root"] / point["image_path"]), True)]
    pre_path, pre_date = rp.prefire_for(point["fire_id"])
    if pre_path:
        dates.append((pre_date, rp.read_bands(pre_path), False))
    other = chip["other_scene"].get(point["tile_id"])
    if other:
        dates.append((other[1], rp.read_bands(chip["images_root"] / other[0]), False))
    dates.sort(key=lambda d: d[0])
    panels = []
    for recipe, half, label in ((rp.FALSE_COLOUR, rp.ZOOM_HALF, "640 m"), (rp.TRUE_COLOUR, rp.ZOOM_HALF, "true colour"),
                                (rp.FALSE_COLOUR, rp.CONTEXT_HALF, "1.9 km")):
        line = [rp.panel(image, recipe, row, col, half, font, f"Date {k}  {date}  {label}") for k, (date, image, _) in enumerate(dates, 1)]
        while len(line) < 3:
            line.append(rp.panel_blank(font, "No other image"))
        panels += line
    width, height = panels[0].size
    sheet = Image.new("RGB", (3 * width + 8, 3 * height + 8), (18, 18, 20))
    for i, picture in enumerate(panels):
        sheet.paste(picture, ((i % 3) * (width + 4), (i // 3) * (height + 4)))
    sheet.save(chip["out"] / "chips_blind" / f"{point['point_id']}.jpg", quality=88)
    return {"point_id": point["point_id"], "dates": ";".join(d[0] for d in dates),
            "scene_position": next(k for k, d in enumerate(dates, 1) if d[2])}


def run_chips(args):
    out = Path(args.out)
    points = pd.read_csv(out / "points.csv", low_memory=False)
    points = points[points["repeat_of"] < 0]
    if args.sample:
        points = points.groupby("stratum", group_keys=False).apply(
            lambda g: g.sample(max(1, round(len(g) * args.sample / len(points))), random_state=args.seed))
    (out / "chips_blind").mkdir(exist_ok=True)
    index = pd.read_csv(args.index, low_memory=False)
    other_scene = {}
    for _, group in index[index["fire_id"].isin(points["fire_id"])].groupby("fire_id"):
        for rec in group.itertuples():
            others = group[group["tile_id"] != rec.tile_id]
            if len(others):
                gap = (pd.to_datetime(others["scene_time"]) - pd.Timestamp(rec.scene_time)).abs()
                best = others.loc[gap.idxmin()]
                other_scene[rec.tile_id] = (best["image_path"], str(best["scene_time"])[:10])
    with ProcessPoolExecutor(args.workers, initializer=rp._init_chips,
                             initargs=(args.images_root, args.prefire_root, out, other_scene)) as pool:
        shown = pd.DataFrame(list(pool.map(blind_chip, points.to_dict("records"), chunksize=8)))
    shown.to_csv(out / "chips_blind_shown.csv", index=False)
    print(f"{len(shown):,} date-blind sheets in {out}/chips_blind; dates per point: "
          f"{shown['dates'].str.count(';').add(1).value_counts().sort_index().to_dict()}. "
          f"chips_blind_shown.csv says which date is the reviewed scene: keep it from the interpreter.")


PAGE = """<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>burn2scar review</title>
<style>
:root { color-scheme: dark; }
* { box-sizing: border-box; }
body { margin: 0; background: #121214; color: #ececec; font: 15px/1.4 system-ui, sans-serif; display: flex; flex-direction: column; height: 100vh; }
header { padding: 8px 14px; display: flex; gap: 14px; flex-wrap: wrap; align-items: baseline; }
#dates { display: flex; gap: 10px; flex-wrap: wrap; }
.date { padding: 2px 8px; border: 1px solid #3a3a40; border-radius: 4px; color: #a8a8b0; }
.date.now { border-color: #f2c94c; color: #f2c94c; }
.date.done { color: #ececec; }
#progress { color: #a8a8b0; font-variant-numeric: tabular-nums; margin-left: auto; }
main { flex: 1; min-height: 0; display: flex; justify-content: center; align-items: center; padding: 0 8px; }
img { max-width: 100%; max-height: 100%; object-fit: contain; display: block; }
footer { display: flex; gap: 8px; padding: 10px 14px; justify-content: center; flex-wrap: wrap; align-items: center; }
button { background: #26262b; color: #ececec; border: 1px solid #3a3a40; border-radius: 4px; padding: 6px 10px; font: inherit; cursor: pointer; }
button:hover { border-color: #f2c94c; }
</style></head><body>
<header><b>For each date, what is the spot in the yellow square?</b><div id="dates"></div><span id="progress"></span></header>
<main><img id="sheet" alt="The point on each date"></main>
<footer id="keys">
<button data-k="1">1 burned (a scar)</button><button data-k="2">2 not burned</button><button data-k="3">3 water</button>
<button data-k="4">4 cloud</button><button data-k="5">5 cloud shadow</button><button data-k="6">6 can't tell</button>
<button data-k="Backspace">Back</button>
</footer>
<script>
const POINTS = __POINTS__, KEY = "burn2scar_blind_" + __REVIEWER__, FILE = __FILE__;
const NAMES = {"1": "burned", "2": "not_burned", "3": "water", "4": "cloud", "5": "shadow", "6": "cant_tell"};
let log = [];
try { log = JSON.parse(localStorage.getItem(KEY) || "[]"); } catch (error) { log = []; }
const answers = {};
for (const [id, k, value] of log) { (answers[id] = answers[id] || {})[k] = value; }
let at = 0, step = 0, started = Date.now();
const sheet = document.getElementById("sheet"), dates = document.getElementById("dates"), progress = document.getElementById("progress");
function store() {
  try { localStorage.setItem(KEY, JSON.stringify(log)); }
  catch (error) { progress.textContent = "This browser is not keeping answers: press Save answers often."; }
}
function complete(i) { const [id, ds] = POINTS[i]; return answers[id] && ds.every((_, k) => answers[id][k + 1]); }
function firstOpen() { for (let i = 0; i < POINTS.length; i++) if (!complete(i)) return i; return POINTS.length; }
function show() {
  if (at >= POINTS.length) { sheet.removeAttribute("src"); dates.innerHTML = ""; progress.textContent = "All points answered. Press Save answers (CSV) and send the file to Param."; return; }
  const [id, ds] = POINTS[at];
  sheet.src = `chips_blind/${id}.jpg`;
  dates.innerHTML = ds.map((d, k) => `<span class="date ${k === step ? "now" : (answers[id] && answers[id][k + 1] ? "done" : "")}">Date ${k + 1} ${d}: ${(answers[id] && answers[id][k + 1]) || "?"}</span>`).join("");
  progress.textContent = `${firstOpen()} of ${POINTS.length} points done`;
  started = Date.now();
}
function press(key) {
  if (at >= POINTS.length && key !== "Backspace") return;
  if (key === "Backspace") {
    if (step > 0) step -= 1; else if (at > 0) { at -= 1; step = POINTS[at][1].length - 1; }
    show(); return;
  }
  if (!NAMES[key]) return;
  const [id, ds] = POINTS[at];
  (answers[id] = answers[id] || {})[step + 1] = NAMES[key];
  log.push([id, step + 1, NAMES[key], ds[step], Date.now() - started, Math.floor(Date.now() / 1000)]);
  store();
  step += 1;
  if (step >= ds.length) { at += 1; step = 0; }
  show();
}
document.addEventListener("keydown", e => { if (NAMES[e.key] || e.key === "Backspace") { e.preventDefault(); press(e.key); } });
document.querySelectorAll("#keys button").forEach(b => b.onclick = () => press(b.dataset.k));
const save = document.createElement("button");
save.textContent = "Save answers (CSV)";
save.onclick = () => {
  const rows = ["point_id,date_index,answer,date,ms,saved_at", ...log.map(r => r.join(","))];
  const link = document.createElement("a");
  link.href = URL.createObjectURL(new Blob([rows.join("\\n") + "\\n"], {type: "text/csv"}));
  link.download = FILE; link.click();
};
document.getElementById("keys").append(save);
at = firstOpen(); show();
</script></body></html>
"""


def run_page(args):
    out = Path(args.out)
    shown = pd.read_csv(out / "chips_blind_shown.csv")
    points = pd.read_csv(out / "points.csv", low_memory=False)
    order = [p for p in rp.reviewer_order(points, args.reviewer) if p in set(shown["point_id"])]
    dates = shown.set_index("point_id")["dates"].str.split(";")
    folder = out / f"offline_blind_{args.reviewer}"
    if folder.exists():
        shutil.rmtree(folder)
    (folder / "chips_blind").mkdir(parents=True)
    for point_id in order:
        shutil.copy2(out / "chips_blind" / f"{point_id}.jpg", folder / "chips_blind" / f"{point_id}.jpg")
    page = (PAGE.replace("__POINTS__", json.dumps([[p, dates[p]] for p in order]))
            .replace("__REVIEWER__", json.dumps(args.reviewer)).replace("__FILE__", json.dumps(f"answers_blind_{args.reviewer}.csv")))
    (folder / "index.html").write_text(page, encoding="utf-8")
    target = out / f"offline_blind_{args.reviewer}.zip"
    with zipfile.ZipFile(target, "w", zipfile.ZIP_STORED) as archive:
        for path in sorted(folder.rglob("*")):
            archive.write(path, Path(f"burn2scar_review_{args.reviewer}") / path.relative_to(folder))
    print(f"{len(order):,} points, {sum(len(dates[p]) for p in order):,} answers for {args.reviewer}: {target} "
          f"({target.stat().st_size / 1e6:.0f} MB). She unzips it and double-clicks index.html; the file she sends back, "
          f"answers_blind_{args.reviewer}.csv, goes into {out}/ as it is.")


CLASS_STRATA = {"fresh_burn": ["fresh_early", "fresh_mid", "fresh_late"], "old_burn": ["old"], "older_burn": ["older"],
                "clear": ["clear_near", "clear_far"], "water": ["water"], "cloud": ["cloud"], "cloud shadow": ["shadow"]}


def weighted(vectors, weights, strata, reps, rng):
    """Area-weighted share of hits over the given strata, with a bootstrap interval that resamples within each."""
    use = [s for s in strata if s in vectors and len(vectors[s])]
    if not use:
        return np.nan, np.nan, np.nan, 0
    w = np.array([weights[s] for s in use])
    w = w / w.sum()
    estimate = float(w @ np.array([vectors[s].mean() for s in use]))
    boot = np.array([[rng.choice(vectors[s], len(vectors[s])).mean() for s in use] for _ in range(reps)]) @ w
    low, high = np.percentile(boot, [2.5, 97.5])
    return estimate, low, high, int(sum(len(vectors[s]) for s in use))


def run_score(args):
    import review_bounds as rb
    out = Path(args.out)
    points = pd.read_csv(out / "points.csv", low_memory=False)
    first = points[points["repeat_of"] < 0].set_index("point_id")
    shown = pd.read_csv(out / "chips_blind_shown.csv").set_index("point_id")
    log = pd.read_csv(out / f"answers_blind_{args.reviewer}.csv")
    last = log.drop_duplicates(["point_id", "date_index"], keep="last")
    latest = last.set_index(["point_id", "date_index"])["answer"]
    weights = pd.read_csv(out / "strata_area.csv").set_index("stratum")["pixels"].astype(float)
    weights = weights / weights.sum()
    index = pd.read_csv(args.index, low_memory=False)
    import review_dates
    ages, _ = review_dates.point_ages(first.reset_index().loc[lambda d: d["point_id"].isin(shown.index)], index, args.masks_root)

    rows, scene_answers = [], {}
    for point_id, record in shown.iterrows():
        dates = record["dates"].split(";")
        given = [latest.get((point_id, k)) for k in range(1, len(dates) + 1)]
        if any(g is None for g in given):
            continue
        stratum = first.at[point_id, "stratum"]
        position = int(record["scene_position"]) - 1
        at_scene = given[position]
        # In the vocabulary of the first review: burned_before when the scar is already on an earlier, first date.
        scene_answers[point_id] = ("burned_before" if at_scene == "burned" and position > 0 and given[0] == "burned"
                                   else at_scene)
        scene = pd.Timestamp(first.at[point_id, "scene_time"]).normalize()
        age = ages.get(point_id, 0)
        consistent = np.nan
        if age > 0:
            perimeter = scene - pd.Timedelta(days=age)
            before = [g for d, g in zip(dates, given) if pd.Timestamp(d) < perimeter]
            after = [g for d, g in zip(dates, given) if pd.Timestamp(d) >= perimeter]
            clear_before = all(g in ("not_burned", "water") for g in before)
            seen_after = bool(after) and after[0] == "burned"
            if all(g not in ("cloud", "shadow", "cant_tell") for g in before + after[:1]):
                consistent = float(clear_before and seen_after)
        rows.append({"point_id": point_id, "stratum": stratum, "at_scene": at_scene, "date_consistent": consistent})
    frame = pd.DataFrame(rows)
    rng = np.random.default_rng(20261120)
    # ms is the time from showing a date to answering it, so a point's time is the sum over its answers, corrections included.
    seconds = log.groupby("point_id")["ms"].sum() / 1000
    lines = [f"# Independent date-blind review: {args.reviewer}", "",
             f"{len(frame):,} points answered on every date ({len(last):,} answers, {len(log) - len(last)} corrected); "
             f"median time per point {seconds.median():.1f} s (per date {log['ms'].median() / 1000:.1f} s), "
             f"{seconds.sum() / 3600:.1f} h of answering in all.", ""]

    # Classes, area-weighted over their strata, as in the paper's table of the reviews.
    vectors = {s: v for s, v in rb.hits(points[points["point_id"].isin(scene_answers)], scene_answers, "strict").items() if len(v)}
    table = []
    for name, strata in CLASS_STRATA.items():
        estimate, low, high, n = weighted(vectors, weights, strata, 2000, rng)
        part = frame[frame["stratum"].isin(strata)]
        dated = part["date_consistent"].dropna()
        d_est = np.nan
        if len(dated):
            d_vec = {s: part.loc[part["stratum"] == s, "date_consistent"].dropna().to_numpy() for s in strata}
            d_est, d_low, d_high, _ = weighted({s: v for s, v in d_vec.items() if len(v)}, weights, strata, 2000, rng)
        table.append({"Class": name, "Points": n, "Confirmed %": rp.pct(estimate, low, high),
                      "Points with a date check": len(dated),
                      "Date-consistent %": rp.pct(d_est, d_low, d_high) if len(dated) else ""})
    strict = rb.bounded(points[points["point_id"].isin(scene_answers)], weights, scene_answers, "strict", 2000, 0)
    earliest = rb.bounded(points[points["point_id"].isin(scene_answers)], weights, scene_answers, "earliest", 2000, 0)
    lines += ["## Classes", "", "Area-weighted over the strata of each class, can't tell left out; bootstrap intervals resample "
              "within strata. Date-consistent: every date before the perimeter date answered not burned and the first date "
              "after it burned, among the points whose dates could be checked.", "", rp.md(pd.DataFrame(table)), "",
              f"- Overall accuracy: {strict} %",
              f"- Overall accuracy, scars already on the first date at clear points not counted as errors: {earliest} %", ""]

    # Per stratum, with Wilson intervals.
    per = []
    for stratum, part in frame.groupby("stratum"):
        seen = part[part["at_scene"] != "cant_tell"]
        hits = int(sum(rp.reference_class(a) == rp.expected(rp.MAP_CLASS[stratum]) for a in seen["at_scene"]))
        dated = part["date_consistent"].dropna()
        per.append({"Stratum": stratum, "Points": len(seen), "Map class confirmed %": rp.interval(hits, len(seen)),
                    "Can't tell": int((part["at_scene"] == "cant_tell").sum()),
                    "Points with a date check": len(dated),
                    "Date-consistent %": rp.interval(int(dated.sum()), len(dated)) if len(dated) else ""})
    lines += ["## Strata", "", rp.md(pd.DataFrame(per)), ""]

    # Against the authors' final labels at the same points.
    final = rp.final_labels(points, out, args.reviewers, args.judge)
    both = [(rp.reference_class(scene_answers[p]), rp.reference_class(final[p])) for p in scene_answers
            if p in final and scene_answers[p] != "cant_tell" and final[p] != "cant_tell"]
    if both:
        a, b = zip(*both)
        same = float(np.mean([x == y for x, y in both]))
        lines += ["## Against the authors' final labels", "",
                  f"{len(both)} points answered by both: same class at {100 * same:.1f} %, Cohen's kappa {rp.kappa(a, b):.2f}.", ""]

    try:
        detail, _ = rp.score_one(points[points["point_id"].isin(scene_answers)], weights, scene_answers,
                                 f"Full scoring of {args.reviewer}, in the format of the first review", 2000)
        lines += detail
    except Exception as error:   # the tables above do not depend on it
        lines += [f"Full scoring skipped: {type(error).__name__}: {error}", ""]
    (out / f"review_blind_{args.reviewer}.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines[:40]))


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("command", choices=["chips", "page", "score"])
    parser.add_argument("--out", default="review2")
    parser.add_argument("--reviewer", default="")
    parser.add_argument("--index", default="data/v4/index.csv")
    parser.add_argument("--images-root", default="data/v1")
    parser.add_argument("--prefire-root", default="data/prefire")
    parser.add_argument("--masks-root", default="data/v4")
    parser.add_argument("--sample", type=int, default=0, help="chips: only this many points, in proportion to the strata")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--reviewers", nargs=2, default=["param", "silvia"], help="score: the first review, for comparison")
    parser.add_argument("--judge", default="judge")
    args = parser.parse_args()
    if args.command in ("page", "score") and not args.reviewer:
        parser.error("--reviewer is needed")
    {"chips": run_chips, "page": run_page, "score": run_score}[args.command](args)


if __name__ == "__main__":
    main()
