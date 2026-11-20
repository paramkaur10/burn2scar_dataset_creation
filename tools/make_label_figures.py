#!/usr/bin/env python3
"""Figures and numbers for the label-quality section of the burn2scar paper, in the style of make_paper_figures.py.

  python tools/make_label_figures.py --out paper/figures --workers 16

fig_review.pdf      The point review: (a) how often each interpreter, and the final label once adjudicated, confirms
                    the map class, by stratum; (b) how often the two interpreters agree; (c) the dNBR under the
                    reviewed burn points by answer, on the scene itself and on the fire's first scene.
fig_sources.pdf     The same points called burned by each source: the map, the dNBR and the interpreters, by stratum.
fig_review_examples.pdf  Reviewed points as the interpreters saw them, on up to three dates, with their answers.
fig_oldspectra.pdf  Reflectance of fresh burn, of old burn that still stands out on the second scene, of old burn that
                    has regrown, and of the clear land around it; and the NBR of each fire's burn.
fig_oldburn.pdf     Fires seen on both scenes: scars that still stand out, scars that have regrown, and snow.
fig_reburn.pdf      Fires that burned where an earlier EFFIS fire had burned.
fig_cems_dnbr.pdf   burn2scar labels, CEMS maps and the dNBR on the same tiles.
label_numbers.md    The numbers the text quotes, part by part.

One part alone with --only review, snow, spectra, oldburn, reburn or cems. review reads review2/ and the few tiles of
its examples, and takes seconds; snow, spectra and oldburn read image tiles (minutes, with --workers); reburn reads the
EFFIS perimeters. Tables the parts share go to --work. oldburn uses the snow table for its snow example, so run snow
before it. cems needs the dNBR of the tiles that overlap a CEMS area, made once with (about 100 tiles, read from the
public Earth Search archive, no Copernicus quota):
  python tools/burn_evidence.py --index data/v4/index.csv --masks-root data/v4 --out validation/dnbr_cems \\
      --tiles-from validation/cems_tiles.csv --save-dnbr --workers 16
"""
import argparse
import sys
import textwrap
import warnings
from collections import Counter
from multiprocessing import Pool
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import rasterio  # noqa: E402
import yaml  # noqa: E402
from matplotlib import patheffects  # noqa: E402
from matplotlib.cm import ScalarMappable  # noqa: E402
from matplotlib.colors import ListedColormap, TwoSlopeNorm  # noqa: E402
from matplotlib.gridspec import GridSpec  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Patch, Rectangle  # noqa: E402
from rasterio.warp import Resampling, reproject  # noqa: E402
from shapely.ops import unary_union  # noqa: E402

TOOLS = Path(__file__).resolve().parent
sys.path[:0] = [str(TOOLS), str(TOOLS.parent)]
from make_paper_figures import (AGREE, AQUA, BLUE, CLASS_COLOUR, CLASS_LABEL, CLASSES, CM, INK, INK2, MASK_VALUE,  # noqa: E402
                                ORANGE, OUTSIDE, WIDTH, draw_outline, frame_panel, grid, mask_colours,
                                own_perimeters, read_rgb, save, set_style)
from make_extra_figures import add_scale, clean, snow_free, window_of  # noqa: E402
from review_points import (BAND, FALSE_COLOUR, MAP_CLASS, ZOOM_HALF, compose, crop_centred, expected,  # noqa: E402
                           latest_answers, reference_class, wilson)

PARTS = ["review", "snow", "spectra", "oldburn", "reburn", "cems"]
ORDER = ["fresh_early", "fresh_mid", "fresh_late", "old", "older", "clear_near", "clear_far", "water", "cloud", "shadow"]
STRATUM_LABEL = {"fresh_early": "fresh_burn, days 5–20", "fresh_mid": "fresh_burn, days 20–40",
                 "fresh_late": "fresh_burn, days 40–60", "old": "old_burn", "older": "older_burn",
                 "clear_near": "clear, within 300 m of burn", "clear_far": "clear, farther", "water": "water",
                 "cloud": "cloud", "shadow": "cloud shadow"}
GROUPS = {"fresh_burn": ["fresh_early", "fresh_mid", "fresh_late"], "old_burn": ["old"], "fresh and old burn":
          ["fresh_early", "fresh_mid", "fresh_late", "old"], "older_burn": ["older"], "clear": ["clear_near", "clear_far"],
          "water": ["water"], "cloud": ["cloud"], "shadow": ["shadow"]}
SERIES_COLOUR = {"Interpreter 1": "#a8a69d", "Interpreter 2": BLUE, "Final": ORANGE}
THIS_SCENE, FIRST_SCENE = AQUA, CLASS_COLOUR["fresh"]
DNBR_THIS, DNBR_FIRST = "dNBR ≥ 0.10, this scene", "dNBR ≥ 0.10, the fire's first scene"
SOURCE_COLOUR = {DNBR_THIS: THIS_SCENE, DNBR_FIRST: FIRST_SCENE, **SERIES_COLOUR}
BURN_STRATA, CLEAR_STRATA = ["fresh_early", "fresh_mid", "fresh_late", "old", "older"], ["clear_near", "clear_far"]
SAID = {"burned": "burned", "burned_before": "burned, already on the earliest image", "not_burned": "not burned",
        "water": "water", "cloud": "cloud", "shadow": "shadow", "cant_tell": "can't tell"}
CEMS_LINE, KEEP_CLASSES = "#3fd0ea", [0, 2, 3, 6, 7]   # the classes compare_cems.py compares: not cloud, shadow, no data
CEMS_MAPS = {"label": "burn2scar fresh_burn", "dnbr10": "dNBR ≥ 0.10", "dnbr27": "dNBR ≥ 0.27",
             "label_dnbr10": "fresh_burn where dNBR ≥ 0.10"}
COUNTS = ["tp", "fp", "fn", "tn"]
SPECTRA = {"Fresh burn": (CLASS_COLOUR["fresh"], "-"), "Old burn, stands out": (CLASS_COLOUR["old"], (0, (4, 1.5))),
           "Old burn, regrown": ("#1baf7a", (0, (1, 1.2))), "Clear land around old burn": ("#8a8984", (0, (4, 1.2, 1, 1.2)))}
SNOW_NDSI, SNOW_GREEN = .4, .2
SEASON = {12: "Dec-Feb", 1: "Dec-Feb", 2: "Dec-Feb", 3: "Mar-May", 4: "Mar-May", 5: "Mar-May", 6: "Jun-Aug", 7: "Jun-Aug",
          8: "Jun-Aug", 9: "Sep-Nov", 10: "Sep-Nov", 11: "Sep-Nov"}


# ------------------------------------------------------------------ small helpers

def md_table(frame):
    head = "| " + " | ".join(map(str, frame.columns)) + " |"
    body = ["| " + " | ".join("" if pd.isna(v) else str(v) for v in row) + " |" for row in frame.itertuples(index=False)]
    return "\n".join([head, "|" + "---|" * len(frame.columns), *body])


def ci(estimate, low, high):
    return "" if pd.isna(estimate) else f"{100 * estimate:.1f} ({100 * low:.1f}-{100 * high:.1f})"


def share(hits, total):
    if not total:
        return ""
    low, high = wilson(hits, total)
    return f"{100 * hits / total:.1f} ({100 * low:.1f}-{100 * high:.1f})"


def halo(ax):
    """A thin dark edge on the white text and lines of an image panel, so they read on pale ground too."""
    for text in ax.texts:
        text.set_path_effects([patheffects.withStroke(linewidth=1.5, foreground=(0, 0, 0, .55))])
    for line in ax.lines:
        line.set_path_effects([patheffects.Stroke(linewidth=line.get_linewidth() + .9, foreground=(0, 0, 0, .45)),
                               patheffects.Normal()])


def write_numbers(out, part, lines):
    (Path(out) / f"numbers_{part}.md").write_text("\n".join(lines) + "\n")
    parts = [Path(out) / f"numbers_{name}.md" for name in PARTS]
    text = "# Numbers for the label-quality section\n\n" + "\n".join(p.read_text() for p in parts if p.exists())
    (Path(out) / "label_numbers.md").write_text(text)


def read_evidence(path):
    if not Path(path).exists():
        return None
    wanted = {"tile_id", "flag", "contrast", "dnbr_in_median", "dnbr_ring_median"}
    return pd.read_csv(path, usecols=lambda c: c in wanted).drop_duplicates("tile_id", keep="last")


def legend_classes(fig, y):
    handles = [Patch(facecolor=CLASS_COLOUR[n], edgecolor="#8a8984", linewidth=.3, label=CLASS_LABEL[n]) for n in CLASSES]
    fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(.5, y), ncol=len(handles), columnspacing=.7,
               handlelength=1, handletextpad=.3, fontsize=5.8)


def old_window_table(frame, args):
    """Old-window tiles with their own burn evidence and the flag of the same fire's first (fresh-window) tile."""
    evidence = read_evidence(args.old_evidence)
    if evidence is None:
        raise SystemExit(f"No {args.old_evidence}: the old-window burn evidence is needed.")
    old = frame[frame["window"] == "old"].drop(columns=["flag", "contrast"], errors="ignore").merge(evidence, on="tile_id", how="left")
    fresh = frame[frame["window"] == "fresh"].drop_duplicates("fire_id").set_index("fire_id")
    old["fresh_flag"] = old["fire_id"].map(fresh["dnbr_flag"])
    old["fresh_tile"] = old["fire_id"].map(fresh["tile_id"])
    return old.reset_index(drop=True)


# ------------------------------------------------------------------ the point review

def review_data(args):
    out = Path(args.review)
    points = pd.read_csv(out / "points.csv", low_memory=False)
    names = [n for n in args.reviewers if (out / f"answers_{n}.csv").exists()]
    if len(names) < 2:
        raise SystemExit(f"Need answers_{args.reviewers[0]}.csv and answers_{args.reviewers[1]}.csv in {out}.")
    answers = {n: latest_answers(out / f"answers_{n}.csv") for n in names}
    first = points[points["repeat_of"] < 0].copy()
    series = {f"Interpreter {i}": answers[n] for i, n in enumerate(names, 1)}
    judge_path = out / f"answers_{args.judge}.csv"
    judged = latest_answers(judge_path) if args.judge and judge_path.exists() else {}
    if judged:
        final = {}
        for pid in first["point_id"]:
            given = [answers[n].get(pid) for n in names]
            if None in given:
                continue
            if pid in judged:
                final[pid] = judged[pid]
            elif len(set(given)) == 1:
                final[pid] = given[0]
        series["Final"] = final
    area = pd.read_csv(out / "strata_area.csv").set_index("stratum")["pixels"].astype(float)
    return out, points, first, names, answers, series, (area / area.sum()).to_dict(), judged


def hits_of(first, table, stratum):
    want = expected(MAP_CLASS[stratum])
    ids = first.loc[first["stratum"] == stratum, "point_id"]
    return np.array([reference_class(table[p]) == want for p in ids if p in table and table[p] != "cant_tell"], float)


def weighted_accuracy(first, table, strata, weights, reps, seed):
    """Area-weighted share of answers that confirm the map class over the given strata, with a bootstrap 95 %
    interval that resamples points within each stratum (as review_points.py does for its matrices)."""
    per = {s: hits_of(first, table, s) for s in strata}
    per = {s: h for s, h in per.items() if len(h)}
    if not per:
        return np.nan, np.nan, np.nan
    w = np.array([weights.get(s, 0.0) for s in per])
    w = w / w.sum()
    estimate = float(w @ np.array([h.mean() for h in per.values()]))
    rng = np.random.default_rng(seed)
    boot = np.array([[rng.choice(h, len(h)).mean() for h in per.values()] for _ in range(reps)]) @ w
    low, high = np.percentile(boot, [2.5, 97.5])
    return estimate, low, high


def kappa(a, b):
    a, b = np.asarray(a), np.asarray(b)
    observed = (a == b).mean()
    chance = sum((a == c).mean() * (b == c).mean() for c in set(a) | set(b))
    return (observed - chance) / (1 - chance) if chance < 1 else np.nan


def repeat_agreement(points, table):
    """Same answer on a point and its second showing, matched as review_points.py matches them."""
    originals = points[points["repeat_of"] < 0].set_index("source_row")["point_id"]
    pairs = []
    for row in points[points["repeat_of"] >= 0].itertuples():
        source = originals.get(row.repeat_of)
        if source is not None and source in table and row.point_id in table:
            pairs.append(table[source] == table[row.point_id])
    return sum(pairs), len(pairs)


def dnbr_cases(first, answers, names, dnbr):
    """dNBR this scene and on the fire's first scene, for fresh and old burn points, by answer (both interpreters)."""
    cases = []
    for group, strata in (("fresh_burn", GROUPS["fresh_burn"]), ("old_burn", ["old"])):
        for answer in ("burned", "not_burned"):
            this, first_scene = [], []
            for pid in first.loc[first["stratum"].isin(strata), "point_id"]:
                if pid not in dnbr.index:
                    continue
                for name in names:
                    given = answers[name].get(pid)
                    if given is not None and reference_class(given) == answer:
                        this.append(dnbr.at[pid, "dnbr"])
                        first_scene.append(dnbr.at[pid, "dnbr_fresh"])
            cases.append((group, answer, np.array(this, float), np.array(first_scene, float)))
    return cases


def burned_counts(first, series, dnbr, stratum, level=.10):
    """(points called burned, points with a value) in one stratum, by each source: the dNBR at the point on this
    scene and, for old_burn, on the fire's first scene; each interpreter; the final label. Can't tell is left out."""
    ids = list(first.loc[first["stratum"] == stratum, "point_id"])
    counts = {}
    if dnbr is not None:
        values = dnbr.reindex(ids)
        this = values["dnbr"].dropna()
        counts[DNBR_THIS] = (int((this >= level).sum()), len(this))
        if stratum == "old":
            first_scene = values["dnbr_fresh"].dropna()
            counts[DNBR_FIRST] = (int((first_scene >= level).sum()), len(first_scene))
    for key, table in series.items():
        given = [table[p] for p in ids if p in table and table[p] != "cant_tell"]
        counts[key] = (sum(reference_class(g) == "burned" for g in given), len(given))
    return counts


def figure_sources(first, series, dnbr, args):
    """The reviewed points called burned by each source, stratum by stratum, against what the map says."""
    strata = [s for s in BURN_STRATA + CLEAR_STRATA if (first["stratum"] == s).any()]
    counts = {s: burned_counts(first, series, dnbr, s) for s in strata}
    keys = [k for k in SOURCE_COLOUR if any(counts[s].get(k, (0, 0))[1] for s in strata)]
    fig, ax = plt.subplots(figsize=(WIDTH, 10.4 * CM))
    fig.subplots_adjust(left=.25, right=.86, top=.86, bottom=.1)
    y = np.arange(len(strata))
    height = .84 / len(keys)
    bars = {key: ([], [], [], []) for key in keys}
    for s, centre in zip(strata, y):
        present = [key for key in keys if counts[s].get(key, (0, 0))[1]]   # no gap where a source has no value
        for i, key in enumerate(present):
            hits, total = counts[s][key]
            low, high = wilson(hits, total)
            for store, value in zip(bars[key], (centre + height * (i - (len(present) - 1) / 2), 100 * hits / total,
                                                100 * low, 100 * high)):
                store.append(value)
    for key, (where, values, lows, highs) in bars.items():
        ax.barh(where, values, height=height * .88, color=SOURCE_COLOUR[key], label=key, zorder=3)
        ax.hlines(where, lows, highs, color=INK, linewidth=.45, zorder=4)
    ax.set_yticks(y, [STRATUM_LABEL[s] for s in strata])
    ax.invert_yaxis()
    ax.set_xlim(0, 100)
    ax.set_xlabel("Points called burned (%)")
    burn = [i for i, s in enumerate(strata) if s in BURN_STRATA]
    clear = [i for i, s in enumerate(strata) if s in CLEAR_STRATA]
    if burn and clear:
        ax.axhline(max(burn) + .5, color=INK2, linewidth=.5)
    for rows, text_ in ((burn, "The map:\nburned"), (clear, "The map:\nnot burned")):
        if rows:
            ax.text(102, np.mean(rows), text_, ha="left", va="center", fontsize=6.3, color=INK2, linespacing=1.15,
                    clip_on=False)
    ax.legend(loc="lower left", bbox_to_anchor=(-.02, 1.03), ncol=3, borderaxespad=0, handlelength=1, handletextpad=.4,
              columnspacing=1, fontsize=6)
    grid(ax, "x")
    save(fig, args.out, "fig_sources")

    rows = []
    for s in strata:
        row = {"Stratum": STRATUM_LABEL[s], "Map": "burned" if s in BURN_STRATA else "not burned"}
        strict = burned_counts(first, {}, dnbr, s, level=.27)
        for key in keys:
            hits, total = counts[s].get(key, (0, 0))
            row[key] = f"{share(hits, total)}, n {total}" if total else ""
            if key in (DNBR_THIS, DNBR_FIRST):
                hits, total = strict.get(key, (0, 0))
                row[key.replace("0.10", "0.27")] = f"{share(hits, total)}, n {total}" if total else ""
        rows.append(row)
    return ["### Points called burned by each source, by stratum", "",
            "Percent with Wilson 95 % intervals; n is the points with a dNBR value, or with an answer other than can't tell.",
            "", md_table(pd.DataFrame(rows)), ""]


def signed(value):
    text = f"{value:+.2f}"
    return "0.00" if text in ("+0.00", "-0.00") else text.replace("-", "\u2212")


def example_dates(point, index, args):
    """The dates the interpreters saw for a point, in time order: (role, date, image path)."""
    dates = [("Reviewed", str(point["scene_time"])[:10], Path(args.images_root) / point["image_path"])]
    if args.prefire_root:
        found = sorted((Path(args.prefire_root) / "acquisitions" / point["fire_id"]).glob("*_prefire_s2.tif"))
        if found:
            dates.append(("Pre-fire", found[0].name.split("_prefire")[0].split("_")[-1][:10], found[0]))
    if index is not None:
        others = index[(index["fire_id"] == point["fire_id"]) & (index["tile_id"] != point["tile_id"])]
        if len(others):
            scene = pd.to_datetime(point["scene_time"], format="mixed", utc=True)
            other = others.loc[(pd.to_datetime(others["scene_time"], format="mixed", utc=True) - scene).abs().idxmin()]
            date = str(other["scene_time"])[:10]
            dates.append(("First scene" if date < dates[0][1] else "Later scene", date, Path(args.images_root) / other["image_path"]))
    return sorted(dates, key=lambda d: d[1])


def pick_examples(first, answers, names, dnbr, shown, seed):
    """(label, point) for points that show what the review found: a fresh burn both saw; an old burn still visible; an
    old burn that has regrown although the fire's first scene shows it; and a point the two answered apart."""
    a, b = answers[names[0]], answers[names[1]]
    table = first.set_index("point_id", drop=False)
    table = table[table.index.isin(dnbr.index) & table.index.isin(list(a)) & table.index.isin(list(b))]
    d = dnbr.reindex(table.index)
    one = np.array([reference_class(a[p]) for p in table.index])
    two = np.array([reference_class(b[p]) for p in table.index])
    this, first_scene = d["dnbr"].to_numpy(float), d["dnbr_fresh"].to_numpy(float)
    stratum = table["stratum"].to_numpy()
    fresh, old = np.isin(stratum, GROUPS["fresh_burn"]), stratum == "old"
    with np.errstate(invalid="ignore"):
        rules = [("a fresh burn", fresh & (one == "burned") & (two == "burned") & (this >= .27)),
                 ("an old burn still visible", old & (one == "burned") & (two == "burned") & (this >= .10)),
                 ("an old burn that has regrown", old & (one == "not_burned") & (two == "not_burned") & (this < .05)
                  & (first_scene >= .27)),
                 ("two answers apart", np.isin(stratum, ["older", "old", "clear_near"])
                  & (((one == "burned") & (two == "not_burned")) | ((one == "not_burned") & (two == "burned"))))]
    full = table.index.isin(list(shown)) if shown else np.ones(len(table), bool)
    rng = np.random.default_rng(seed)
    picks = []
    for label, rule in rules:
        candidates = np.flatnonzero(rule & full) if (rule & full).any() else np.flatnonzero(rule)
        if len(candidates):
            picks.append((label, table.iloc[int(rng.choice(candidates))]))
    return picks


def example_title(letter, point, answers, names, dnbr, judged):
    pid = point["point_id"]
    given = [answers[n].get(pid) for n in names]
    short = {**SAID, "burned_before": "burned, scar already on the pre-fire image"}
    said = (f"both interpreters: {short.get(given[0], given[0])}" if given[0] == given[1]
            else "interpreters: " + " | ".join(short.get(g, g) for g in given))
    if pid in judged:
        said += f"; adjudicated: {short.get(judged[pid], judged[pid])}"
    place = STRATUM_LABEL[point["stratum"]]
    if point["stratum"] in GROUPS["fresh and old burn"] and pd.notna(point.get("age_days", np.nan)):
        place = f"{place.split(',')[0]}, day {point['age_days']:.0f} after the fire"
    first_line = f"({letter}) {place}. dNBR {signed(dnbr.at[pid, 'dnbr'])}"
    if point.get("window", "fresh") != "fresh" and pd.notna(dnbr.at[pid, "dnbr_fresh"]):
        first_line += f" on this scene, {signed(dnbr.at[pid, 'dnbr_fresh'])} on the first scene"
    # The answers on their own line, wrapped so that a long pair of answers is never cut at the figure's edge.
    return first_line + ".\n" + textwrap.fill(said[0].upper() + said[1:] + ".", width=96)


def figure_examples(out, first, answers, names, dnbr, judged, args):
    """Reviewed points as the interpreters saw them: the 640 m window on each date, in time order, and the 1.9 km
    window of the reviewed scene, drawn from the image tiles, with the answers and the dNBR above."""
    shown = set()
    if (out / "chips_shown.csv").exists():
        chips = pd.read_csv(out / "chips_shown.csv")
        shown = set(chips.loc[chips["dates_shown"] >= 3, "point_id"])
    picks = pick_examples(first, answers, names, dnbr, shown, args.pick_seed)
    if not picks:
        print("No points for the review examples.", flush=True)
        return []
    index = None
    if Path(args.index).exists():
        index = pd.read_csv(args.index, usecols=["tile_id", "fire_id", "scene_time", "image_path"], low_memory=False)
    n = len(picks)
    panel = WIDTH / CM * .98 / (4 + 3 * .04)          # cm, four square panels across
    lines = max(example_title(letter, point, answers, names, dnbr, judged).count("\n") + 1
                for letter, (_, point) in zip("abcdefgh", picks))
    gap = head = .45 + .27 * lines
    foot = .12
    height = head + n * panel + (n - 1) * gap + foot
    fig = plt.figure(figsize=(WIDTH, height * CM))
    layout = GridSpec(n, 4, figure=fig, left=.01, right=.99, top=1 - head / height, bottom=foot / height, wspace=.04,
                      hspace=gap / panel)
    wide = 3 * ZOOM_HALF
    notes = []
    for k, (letter, (label, point)) in enumerate(zip("abcdefgh", picks)):
        dates = example_dates(point, index, args)
        row, col = int(point["row"]), int(point["col"])
        axes = [fig.add_subplot(layout[k, j]) for j in range(4)]
        for j, ax in enumerate(axes[:3]):
            frame_panel(ax)
            ax.set_xlim(-.5, 2 * ZOOM_HALF - .5)
            ax.set_ylim(2 * ZOOM_HALF - .5, -.5)
            if j >= len(dates):
                ax.set_facecolor("#28282c")
                ax.text(.5, .5, "no other image", transform=ax.transAxes, ha="center", va="center", fontsize=5.5,
                        color="#c8c8c8")
                continue
            role, date, path = dates[j]
            with rasterio.open(path) as src:
                image = src.read().astype(np.float32) / 10000
            ax.imshow(crop_centred(compose(image, FALSE_COLOUR), row, col, ZOOM_HALF))
            for edge, width in ((INK, 1.5), ("#ffeb00", .6)):
                ax.add_patch(Rectangle((ZOOM_HALF - 1.5, ZOOM_HALF - 1.5), 3, 3, fill=False, edgecolor=edge, linewidth=width))
            ax.set_title(f"{role}, {date}", fontsize=5.6, pad=1.5, fontweight="bold" if role == "Reviewed" else "normal")
            if j == 0:
                add_scale(ax, 2 * ZOOM_HALF)
        with rasterio.open(Path(args.images_root) / point["image_path"]) as src:
            image = src.read().astype(np.float32) / 10000
        axes[3].imshow(crop_centred(compose(image, FALSE_COLOUR), row, col, wide))
        axes[3].add_patch(Rectangle((wide - ZOOM_HALF - .5, wide - ZOOM_HALF - .5), 2 * ZOOM_HALF, 2 * ZOOM_HALF,
                                    fill=False, edgecolor="#ffeb00", linewidth=.6))
        axes[3].set_xlim(-.5, 2 * wide - .5)
        axes[3].set_ylim(2 * wide - .5, -.5)
        frame_panel(axes[3])
        axes[3].set_title("Reviewed, 1.9 km around", fontsize=5.6, pad=1.5)
        add_scale(axes[3], 2 * wide)
        box = axes[0].get_position()
        fig.text(box.x0, box.y1 + .31 / height, example_title(letter, point, answers, names, dnbr, judged), fontsize=6.2,
                 ha="left", va="bottom", linespacing=1.25)
        for ax in axes:
            halo(ax)
        notes.append(f"- fig_review_examples ({letter}) {label}: point {point['point_id']}, tile {point['tile_id']}")
    save(fig, args.out, "fig_review_examples")
    return ["### Review examples", "", *notes, ""]


def run_review(args):
    out, points, first, names, answers, series, weights, judged = review_data(args)
    strata = [s for s in ORDER if (first["stratum"] == s).any()]
    a, b = answers[names[0]], answers[names[1]]
    both = [p for p in first["point_id"] if p in a and p in b]
    k = kappa([reference_class(a[p]) for p in both], [reference_class(b[p]) for p in both])
    same_class = sum(reference_class(a[p]) == reference_class(b[p]) for p in both)
    dnbr_path = out / "review_dnbr_points.csv"
    dnbr = pd.read_csv(dnbr_path).drop_duplicates("point_id").set_index("point_id") if dnbr_path.exists() else None

    fig = plt.figure(figsize=(WIDTH, 13.4 * CM))
    layout = GridSpec(2, 2, figure=fig, height_ratios=[1.5, 1], width_ratios=[1.6, 1], hspace=.42, wspace=.06,
                      left=.25, right=.985, top=.93, bottom=.12)
    ax_a, ax_c = fig.add_subplot(layout[0, 0]), fig.add_subplot(layout[1, :])
    ax_b = fig.add_subplot(layout[0, 1], sharey=ax_a)
    y = np.arange(len(strata))
    keys = list(series)
    height = .82 / len(keys)
    for i, key in enumerate(keys):
        values, lows, highs = [], [], []
        for s in strata:
            h = hits_of(first, series[key], s)
            low, high = wilson(h.sum(), len(h)) if len(h) else (np.nan, np.nan)
            values.append(100 * h.mean() if len(h) else np.nan)
            lows.append(100 * low)
            highs.append(100 * high)
        position = y - .41 + height * (i + .5)
        ax_a.barh(position, values, height=height * .9, color=SERIES_COLOUR[key], label=key, zorder=3)
        ax_a.hlines(position, lows, highs, color=INK, linewidth=.45, zorder=4)
    ax_a.set_yticks(y, [STRATUM_LABEL[s] for s in strata])
    ax_a.invert_yaxis()
    ax_a.set_xlim(0, 100)
    ax_a.set_xlabel("Answers confirming the map class (%)")
    ax_a.set_title("(a) Agreement with the map", loc="left")
    ax_a.legend(loc="lower left", bbox_to_anchor=(-.02, 1.07), ncol=len(keys), borderaxespad=0, handlelength=1,
                handletextpad=.4, columnspacing=1, fontsize=6)
    grid(ax_a, "x")
    values, lows, highs = [], [], []
    for s in strata:
        pairs = [(a[p], b[p]) for p in first.loc[first["stratum"] == s, "point_id"] if p in a and p in b]
        same = sum(reference_class(x) == reference_class(z) for x, z in pairs)
        low, high = wilson(same, len(pairs))
        values.append(100 * same / len(pairs) if pairs else np.nan)
        lows.append(100 * low)
        highs.append(100 * high)
    ax_b.barh(y, values, height=.55, color=INK2, zorder=3)
    ax_b.hlines(y, lows, highs, color=INK, linewidth=.45, zorder=4)
    ax_b.set_xlim(0, 100)
    ax_b.set_xticks([25, 50, 75, 100])
    ax_b.set_xlabel("Same class from both (%)")
    ax_b.set_title(f"(b) Interpreters agree, κ = {k:.2f}", loc="left")
    ax_b.tick_params(axis="y", labelleft=False)
    grid(ax_b, "x")

    lines = []
    if dnbr is not None:
        cases = dnbr_cases(first, answers, names, dnbr)
        ticks, labels = [], []
        for x, (group, answer, this, first_scene) in enumerate(cases):
            counts = [int(np.sum(~np.isnan(this)))] + ([int(np.sum(~np.isnan(first_scene)))] if group != "fresh_burn" else [])
            sets = [(this, THIS_SCENE, x)] if group == "fresh_burn" else [(this, THIS_SCENE, x - .17), (first_scene, FIRST_SCENE, x + .17)]
            for data, colour, position in sets:
                data = data[~np.isnan(data)]
                if len(data) < 3:
                    continue
                box = ax_c.boxplot([data], positions=[position], widths=.28, showfliers=False, patch_artist=True,
                                   medianprops={"color": "white", "linewidth": .9}, whiskerprops={"linewidth": .5},
                                   capprops={"linewidth": .5}, boxprops={"linewidth": .4})
                box["boxes"][0].set_facecolor(colour)
            ticks.append(x)
            labels.append(f"{group}\nanswered {answer.replace('_', ' ')}\nn = {' | '.join(map(str, counts))}")
            lines.append(f"| {group} | {answer} | {np.sum(~np.isnan(this))} | {np.nanmedian(this):.3f} | "
                         f"{100 * np.mean(this[~np.isnan(this)] >= .1):.1f} | {np.nanmedian(first_scene):.3f} | "
                         f"{100 * np.mean(first_scene[~np.isnan(first_scene)] >= .1):.1f} |" if np.sum(~np.isnan(this)) else "")
        for level in (.10, .27):
            ax_c.axhline(level, color=INK2, linewidth=.5, linestyle=(0, (2, 2)), zorder=1)
            ax_c.text(-.46, level + .015, f"{level:.2f}", va="bottom", ha="left", fontsize=5, color=INK2)
        ax_c.set_xticks(ticks, labels)
        ax_c.set_xlim(-.5, len(cases) - .5)
        ax_c.set_ylim(-.6, 1.0)
        ax_c.set_ylabel("dNBR at the point")
        ax_c.set_title("(c) dNBR at the reviewed burn points", loc="left")
        ax_c.legend(handles=[Patch(facecolor=THIS_SCENE, label="this scene"),
                             Patch(facecolor=FIRST_SCENE, label="the fire's first scene")],
                    loc="lower right", bbox_to_anchor=(1, 1.0), ncol=2, fontsize=6, handlelength=1, frameon=False,
                    borderaxespad=.2)
        grid(ax_c)
    else:
        ax_c.set_axis_off()
        print(f"No {dnbr_path}: run tools/review_dnbr.py for panel (c).", flush=True)
    save(fig, args.out, "fig_review")

    rows = []
    for s in strata:
        row = {"Stratum": s, "Points": int((first["stratum"] == s).sum()), "Share of area %": f"{100 * weights.get(s, 0):.2f}"}
        for key in keys:
            h = hits_of(first, series[key], s)
            row[key] = share(int(h.sum()), len(h))
        pairs = [(a[p], b[p]) for p in first.loc[first["stratum"] == s, "point_id"] if p in a and p in b]
        row["Interpreters agree"] = share(sum(reference_class(x) == reference_class(z) for x, z in pairs), len(pairs))
        rows.append(row)
    classes = []
    for group, members in [("overall", strata)] + list(GROUPS.items()):
        members = [m for m in members if m in strata]
        if members:
            classes.append({"Map class": group, **{key: ci(*weighted_accuracy(first, series[key], members, weights, args.reps,
                                                                              args.seed)) for key in keys}})
    text = ["## Point review", "",
            f"Points: {len(first)} first showings. Same class from both interpreters: {share(same_class, len(both))} %, "
            f"Cohen's kappa {k:.3f}.", ""]
    for i, name in enumerate(names, 1):
        same, total = repeat_agreement(points, answers[name])
        cant = sum(1 for p in first["point_id"] if answers[name].get(p) == "cant_tell")
        text.append(f"- Interpreter {i} ({name}): same answer on a point shown twice {share(same, total)} %, {total} points; "
                    f"can't tell {cant}.")
    if judged:
        final = series["Final"]
        changed = sum(1 for p in judged if p in a and p in b and a[p] == b[p] and final.get(p) != a[p])
        text.append(f"- Adjudicated: {sum(1 for p in judged if p in set(first['point_id']))} points; the adjudicator changed "
                    f"{changed} answers the two interpreters shared. Final labels: {len(final)} points.")
    text += ["", "### Agreement with the map class, by stratum (Wilson 95 % intervals)", "", md_table(pd.DataFrame(rows)), "",
             "### User's accuracy by map class: area-weighted over its strata, bootstrap 95 % intervals", "",
             md_table(pd.DataFrame(classes)), ""]
    if lines:
        text += ["### dNBR under fresh and old burn points, answers of both interpreters pooled", "",
                 "| Map class | Answer | Answers with a dNBR | Median, this scene | >= 0.10 % | Median, fire's first scene | >= 0.10 % |",
                 "|---|---|---|---|---|---|---|", *[line for line in lines if line], ""]
    text += figure_sources(first, series, dnbr, args)
    if dnbr is not None:
        text += figure_examples(out, first, answers, names, dnbr, judged, args)
    write_numbers(args.out, "review", text)


# ------------------------------------------------------------------ snow on the scenes

def snow_job(job):
    tile_id, window, image_path, mask_path, age_path, age_days = job
    try:
        with rasterio.open(image_path) as src:
            green = src.read(BAND["B03"] + 1).astype(np.float32) / 10000
            swir = src.read(BAND["B11"] + 1).astype(np.float32) / 10000
        with rasterio.open(mask_path) as src:
            mask = src.read(1)
        own = None
        if isinstance(age_path, str) and Path(age_path).exists():
            with rasterio.open(age_path) as src:
                age = src.read(1).astype(np.int64)
            own = (age > 0) & (np.abs(age - int(round(age_days))) <= 1)
        if own is None or own.sum() < 25:   # no age raster: the own fire is the burn class of its window
            own = mask == MASK_VALUE["old" if window == "old" else "fresh"]
        valid = (green > 0) & (mask != MASK_VALUE["nodata"])
        ndsi = (green - swir) / np.maximum(green + swir, 1e-6)
        snow = valid & (ndsi > SNOW_NDSI) & (green > SNOW_GREEN)
        inside = own & valid
        result = {"tile_id": tile_id, "window": window, "own_px": int(inside.sum()), "snow_own_px": int((snow & inside).sum()),
                  "snow_tile_frac": float(snow.sum() / max(int(valid.sum()), 1))}
        for name, value in MASK_VALUE.items():
            if name != "nodata":
                result[f"snow_as_{name}"] = int((snow & inside & (mask == value)).sum())
        return result
    except Exception as error:  # a broken tile must not stop the others
        return {"tile_id": tile_id, "window": window, "error": f"{type(error).__name__}: {error}"[:200]}


def run_snow(frame, args):
    jobs = [(r.tile_id, r.window, str(Path(args.images_root) / r.image_path), str(Path(args.masks_root) / r.mask_path),
             str(Path(args.masks_root) / r.age_path) if isinstance(getattr(r, "age_path", None), str) else None, r.age_days)
            for r in frame.itertuples()]
    print(f"Snow: reading {len(jobs):,} tiles with {args.workers} workers", flush=True)
    with Pool(args.workers) as pool:
        table = pd.DataFrame(pool.imap_unordered(snow_job, jobs, chunksize=16))
    work = Path(args.work)
    work.mkdir(parents=True, exist_ok=True)
    table.to_csv(work / "snow_tiles.csv", index=False)
    if "error" in table:
        print(f"  {table['error'].notna().sum()} tiles could not be read; see {work / 'snow_tiles.csv'}", flush=True)
        table = table[table["error"].isna()]
    t = table.merge(frame[["tile_id", "country", "scene_time"]], on="tile_id")
    t["own_snow"] = t["snow_own_px"] / t["own_px"].where(t["own_px"] > 0)
    t["month"] = pd.to_datetime(t["scene_time"], format="mixed", utc=True, errors="coerce").dt.month
    text = ["## Snow over the burn", "",
            f"Snow-like: NDSI (B03, B11) above {SNOW_NDSI} and B03 above {SNOW_GREEN}, top-of-atmosphere. Own perimeter: the "
            "age raster's pixels dated to the tile's own fire.", ""]
    rows = []
    for window in ("fresh", "old"):
        part = t[(t["window"] == window) & t["own_snow"].notna()]
        rows.append({"Window": window, "Tiles": len(part), "Snow on >= 10 % of the burn": share(int((part["own_snow"] >= .1).sum()), len(part)),
                     "Snow on >= 50 %": share(int((part["own_snow"] >= .5).sum()), len(part)),
                     "Any snow in the tile >= 1 %": share(int((part["snow_tile_frac"] >= .01).sum()), len(part))})
    text += [md_table(pd.DataFrame(rows)), ""]
    old = t[(t["window"] == "old") & t["own_snow"].notna()]
    for column, title in (("month", "Old window, by scene month"), ("country", "Old window, by country")):
        grouped = old.groupby(column)["own_snow"]
        table_rows = [{title.split(", by ")[1].capitalize(): key, "Tiles": len(s), "Snow on >= 10 % of the burn": share(int((s >= .1).sum()), len(s)),
                       "Snow on >= 50 %": share(int((s >= .5).sum()), len(s))} for key, s in grouped]
        text += [f"### {title}", "", md_table(pd.DataFrame(table_rows)), ""]
    total = old["snow_own_px"].sum()
    if total:
        text += ["### What the mask says under the snow inside old-window burns", "",
                 ", ".join(f"{name}: {100 * old[f'snow_as_{name}'].sum() / total:.1f} %" for name in MASK_VALUE if name != "nodata"), ""]
    write_numbers(args.out, "snow", text)


# ------------------------------------------------------------------ spectra of fresh and old burn

def spectra_job(job):
    group, image_path, mask_path, value, pixels, seed = job
    try:
        rng = np.random.default_rng(seed)
        with rasterio.open(mask_path) as src:
            mask = src.read(1)
        with rasterio.open(image_path) as src:
            image = src.read().astype(np.float32) / 10000
        stored = image.max(axis=0) > 0
        rows, cols = np.nonzero((mask == value) & stored)
        if len(rows) < 25:
            return group, None, np.nan
        keep = rng.choice(len(rows), min(pixels, len(rows)), replace=False)
        nir, swir = image[BAND["B8A"], rows, cols], image[BAND["B12"], rows, cols]
        nbr = float(np.median((nir - swir) / np.maximum(nir + swir, 1e-6)))
        return group, image[:, rows[keep], cols[keep]].T, nbr
    except Exception:
        return group, None, np.nan


def run_spectra(frame, args):
    old = old_window_table(frame, args)
    fresh = frame[(frame["window"] == "fresh") & (frame["dnbr_flag"] == "burn") & clean(frame) & snow_free(frame)]
    confirmed = old[clean(old) & snow_free(old) & (old["fresh_flag"] == "burn")]
    visible, regrown = confirmed[confirmed["flag"] == "burn"], confirmed[confirmed["flag"] == "none"]
    groups = {"Fresh burn": (fresh, MASK_VALUE["fresh"]), "Old burn, stands out": (visible, MASK_VALUE["old"]),
              "Old burn, regrown": (regrown, MASK_VALUE["old"]),
              "Clear land around old burn": (pd.concat([visible, regrown]), MASK_VALUE["clear"])}
    jobs = []
    for g, (name, (part, value)) in enumerate(groups.items()):
        chosen = part.sample(min(args.tiles, len(part)), random_state=args.seed + g)
        jobs += [(name, str(Path(args.images_root) / r.image_path), str(Path(args.masks_root) / r.mask_path), value,
                  args.pixels, args.seed + i) for i, r in enumerate(chosen.itertuples())]
    print(f"Spectra: reading {len(jobs):,} tiles with {args.workers} workers", flush=True)
    pixels = {name: [] for name in groups}
    nbr = {name: [] for name in groups}
    with Pool(args.workers) as pool:
        for name, sample, tile_nbr in pool.imap_unordered(spectra_job, jobs, chunksize=8):
            if sample is not None:
                pixels[name].append(sample)
                nbr[name].append(tile_nbr)
    pixels = {name: np.concatenate(p) for name, p in pixels.items() if p}
    bands = sorted(BAND, key=BAND.get)
    x = np.arange(len(bands))
    fig = plt.figure(figsize=(WIDTH, 7.4 * CM))
    layout = GridSpec(1, 2, figure=fig, width_ratios=[2.2, 1], wspace=.32, left=.085, right=.985, top=.84, bottom=.2)
    ax, box = fig.add_subplot(layout[0]), fig.add_subplot(layout[1])
    for name, (colour, style) in SPECTRA.items():
        if name not in pixels:
            continue
        low, mid, high = np.percentile(pixels[name], [25, 50, 75], axis=0)
        ax.fill_between(x, low, high, color=colour, alpha=.13, linewidth=0, zorder=2)
        ax.plot(x, mid, color=colour, linestyle=style, linewidth=1.1, marker="o", markersize=1.8, label=name, zorder=3)
    ax.set_xticks(x, bands, rotation=90)
    ax.set_xlim(-.3, len(bands) - .7)
    ax.set_ylim(0, None)
    ax.set_ylabel("Top-of-atmosphere reflectance")
    ax.set_title("(a) Median and quartiles", loc="left")
    ax.legend(loc="lower left", bbox_to_anchor=(0, 1.08), ncol=2, borderaxespad=0, handlelength=2.6, handletextpad=.4,
              columnspacing=1, fontsize=6)
    grid(ax)
    names = [n for n in SPECTRA if nbr.get(n)]
    parts = box.boxplot([nbr[n] for n in names], widths=.55, showfliers=False, patch_artist=True,
                        medianprops={"color": "white", "linewidth": .9}, whiskerprops={"linewidth": .5},
                        capprops={"linewidth": .5}, boxprops={"linewidth": .4})
    for patch, n in zip(parts["boxes"], names):
        patch.set_facecolor(SPECTRA[n][0])
    short = {"Fresh burn": "fresh", "Old burn, stands out": "old,\nvisible", "Old burn, regrown": "old,\nregrown",
             "Clear land around old burn": "clear\naround"}
    box.set_xticks(range(1, len(names) + 1), [short[n] for n in names], fontsize=5.5)
    box.set_ylabel("NBR, median per tile")
    box.set_title("(b) NBR from fire to fire", loc="left")
    grid(box)
    save(fig, args.out, "fig_oldspectra")

    rows = []
    for name in SPECTRA:
        if name not in pixels:
            continue
        mid = np.percentile(pixels[name], 50, axis=0)
        values = np.array(nbr[name])
        q1, q2, q3 = np.percentile(values, [25, 50, 75])
        rows.append({"Group": name, "Tiles": len(values), "Pixels": len(pixels[name]),
                     **{b: f"{mid[BAND[b]]:.3f}" for b in ("B04", "B08", "B8A", "B11", "B12")},
                     "NBR per tile, median": f"{q2:.3f}", "NBR per tile, IQR": f"{q1:.3f} to {q3:.3f} ({q3 - q1:.3f})"})
    text = ["## Spectra of fresh and old burn", "",
            f"Scenes from April to October, no cloud or shadow above 1 %. Fresh burn: fresh-window tiles flagged burn. Old burn: "
            f"old-window tiles whose fire's first tile is flagged burn, split by the old-window flag (burn: stands out; none: "
            f"regrown). Up to {args.tiles} tiles per group and {args.pixels} pixels per tile, seed {args.seed}. "
            f"Candidates: fresh {len(fresh):,}, stands out {len(visible):,}, regrown {len(regrown):,}.", "",
            md_table(pd.DataFrame(rows)), ""]
    write_numbers(args.out, "spectra", text)


# ------------------------------------------------------------------ old burns: which stand out, and a gallery

def regrowth_tables(old):
    """Of the fires the first scene confirms, how many still stand out on the second scene, by group."""
    part = old[(old["fresh_flag"] == "burn") & old["flag"].isin(["burn", "weak", "none"])].copy()
    part["Fire season"] = pd.to_datetime(part["fire_date"], format="mixed", errors="coerce").dt.month.map(SEASON)
    part["Days to the second scene"] = pd.cut(part["age_days"], [119, 180, 240, 301], labels=["120-180", "180-240", "240-300"])
    part["Fire size (ha)"] = pd.cut(part["area_ha"], [5, 30, 100, 500, 2001], right=False, labels=["5-30", "30-100", "100-500", "500-2000"])
    part["Scene month"] = pd.to_datetime(part["scene_time"], format="mixed", utc=True, errors="coerce").dt.month
    text = ["## Old burns on the second scene", "",
            f"Fires flagged burn on their first scene and assessed on the second: {len(part):,}. Share flagged burn (stands "
            f"out), weak and none on the second scene, Wilson intervals.", ""]
    columns = [c for c in ("land_cover", "country", "Fire season", "Days to the second scene", "Fire size (ha)", "Scene month") if c in part]
    overall = {"Group": "all", "Fires": len(part), "Stands out": share(int((part["flag"] == "burn").sum()), len(part)),
               "Weak": share(int((part["flag"] == "weak").sum()), len(part)), "Regrown (none)": share(int((part["flag"] == "none").sum()), len(part))}
    text += [md_table(pd.DataFrame([overall])), ""]
    for column in columns:
        rows = [{"Group": key, "Fires": len(g), "Stands out": share(int((g["flag"] == "burn").sum()), len(g)),
                 "Weak": share(int((g["flag"] == "weak").sum()), len(g)), "Regrown (none)": share(int((g["flag"] == "none").sum()), len(g))}
                for key, g in part.groupby(column, observed=True) if len(g)]
        text += [f"### By {column.replace('_', ' ')}", "", md_table(pd.DataFrame(rows)), ""]
    return text


def gallery_picks(frame, old, args):
    fresh = frame[frame["window"] == "fresh"].drop_duplicates("fire_id").set_index("fire_id")
    cems = set()
    if Path(args.cems_tiles).exists():
        tiles = pd.read_csv(args.cems_tiles, usecols=["tile_id"])
        cems = set(frame.loc[frame["tile_id"].isin(tiles["tile_id"]), "fire_id"])
    first_clean = old["fire_id"].map(clean(fresh) & snow_free(fresh)).fillna(False).astype(bool)
    base = old[(old["fresh_flag"] == "burn") & first_clean & old["area_ha"].between(80, 2000) & old["own_frac"].between(.15, .65)]
    picks = []
    rules = [("still stands out", base[clean(base) & (base["flag"] == "burn") & (base["contrast"] >= .15)], 2),
             ("regrown", base[clean(base) & (base["flag"] == "none") & (base["contrast"] < .02)], 2)]
    snow_table = Path(args.work) / "snow_tiles.csv"
    if snow_table.exists():
        snow = pd.read_csv(snow_table)
        snow = snow[snow.get("own_px", 0) > 0]
        covered = set(snow.loc[snow["snow_own_px"] / snow["own_px"] >= .5, "tile_id"])
        rules.append(("under snow", base[base["tile_id"].isin(covered)], 1))
    else:
        print(f"No {snow_table}: run --only snow first for the snow example.", flush=True)
    for label, part, count in rules:
        preferred = part[part["fire_id"].isin(cems)]
        pool_ = preferred if len(preferred) >= count else part
        for _, row in pool_.sample(min(count, len(pool_)), random_state=args.pick_seed).iterrows():
            picks.append((label, row, fresh.loc[row["fire_id"]], row["fire_id"] in cems))
    return picks


def run_oldburn(frame, args):
    old = old_window_table(frame, args)
    text = regrowth_tables(old)
    picks = gallery_picks(frame, old, args)
    if not picks:
        print("No tiles for the old-burn gallery.", flush=True)
        write_numbers(args.out, "oldburn", text)
        return
    fresh_evidence = read_evidence(args.fresh_evidence)
    fresh_contrast = fresh_evidence.set_index("tile_id")["contrast"] if fresh_evidence is not None else pd.Series(dtype=float)
    outlines, outline_crs = own_perimeters(args.config, [row["fire_id"] for _, row, _, _ in picks])
    mask_cmap, mask_norm = mask_colours()
    fig, axes = plt.subplots(3, len(picks), figsize=(WIDTH, 9.4 * CM), squeeze=False)
    notes = []
    for letter, column, (label, row, first, in_cems) in zip("abcdefgh", axes.T, picks):
        with rasterio.open(Path(args.masks_root) / row["mask_path"]) as src:
            mask, tile_crs, transform = src.read(1), src.crs, src.transform
        top, left, size = window_of(row, mask.shape)
        crop = (slice(top, top + size), slice(left, left + size))
        column[0].imshow(read_rgb(Path(args.images_root) / first["image_path"], FALSE_COLOUR, BAND)[crop])
        column[1].imshow(read_rgb(Path(args.images_root) / row["image_path"], FALSE_COLOUR, BAND)[crop])
        column[2].imshow(mask[crop], cmap=mask_cmap, norm=mask_norm)
        geometry = outlines.get(row["fire_id"])
        for ax in column[:2]:
            if geometry is not None:
                draw_outline(ax, geometry, outline_crs, tile_crs, transform, offset=(top, left))
        for ax in column:
            ax.set_xlim(-.5, size - .5)
            ax.set_ylim(size - .5, -.5)
            frame_panel(ax)
        add_scale(column[0], size)
        c1 = fresh_contrast.get(first["tile_id"], np.nan)
        cover = str(row.get("land_cover", "")).replace("_", " ") if pd.notna(row.get("land_cover", np.nan)) else ""
        column[0].set_title(f"({letter}) {label}{' *' if in_cems else ''}\n{cover}", fontsize=5.8, linespacing=1.1, pad=2.5)
        column[0].text(.04, .96, f"day {first['age_days']:.0f}" + (f", {c1:+.2f}" if pd.notna(c1) else ""), transform=column[0].transAxes,
                       fontsize=5, color="white", va="top")
        column[1].text(.04, .96, f"day {row['age_days']:.0f}" + (f", {row['contrast']:+.2f}" if pd.notna(row["contrast"]) else ""),
                       transform=column[1].transAxes, fontsize=5, color="white", va="top")
        for ax in column[:2]:
            halo(ax)
        notes.append(f"- fig_oldburn ({letter}) {label}: fire {row['fire_id']}, first {first['tile_id']}, second {row['tile_id']}"
                     f"{', in a CEMS map' if in_cems else ''}")
    for ax, text_ in zip(axes[:, 0], ("First scene", "Second scene", "Mask, second")):
        ax.set_ylabel(text_, fontsize=6.5, labelpad=2)
    fig.subplots_adjust(left=.045, right=.995, top=.9, bottom=.1, wspace=.05, hspace=.05)
    legend_classes(fig, .005)
    save(fig, args.out, "fig_oldburn")
    write_numbers(args.out, "oldburn", text + ["### Gallery tiles", "", *notes, ""])


# ------------------------------------------------------------------ fires on earlier scars

def run_reburn(frame, args):
    from burn_labels import load_perimeters
    cfg = yaml.safe_load(Path(args.config).read_text())
    extra = list(args.effis_extra)
    perimeters = load_perimeters(cfg, extra)
    if perimeters["fire_date"].min() >= pd.Timestamp("2020-01-01") and not extra:
        # The config lists only the build export: look next to it for the export that reaches back to 2018.
        build = Path(cfg["effis_path"])
        for path in sorted(build.parent.glob("*")):
            if path.suffix.lower() not in {".zip", ".shp", ".gpkg", ".geojson", ".json"} or path.resolve() == build.resolve():
                continue
            try:
                trial = load_perimeters(cfg, [str(path)])
            except Exception:  # not an EFFIS export
                continue
            if trial["fire_date"].min() < pd.Timestamp("2020-01-01"):
                print(f"The config lists no export before 2020; using {path} for the earlier perimeters.", flush=True)
                perimeters, extra = trial, [str(path)]
                break
    earliest = perimeters["fire_date"].min()
    if earliest >= pd.Timestamp("2020-01-01"):
        print(f"WARNING: the earliest perimeter loaded is dated {earliest:%Y-%m-%d}, so fires of 2018 and 2019 are missing and "
              f"scars from them are not counted. Pass the 12 Nov export, e.g. --effis-extra effis/<its file>.", flush=True)
    perimeters = perimeters[perimeters["fire_date"] >= pd.Timestamp("2018-01-01")].to_crs(3035).reset_index(drop=True)
    perimeters["geometry"] = perimeters.geometry.buffer(0)
    kept = perimeters[perimeters["fire_id"].isin(set(frame["fire_id"]))].drop_duplicates("fire_id")
    index = perimeters.sindex
    gap = pd.Timedelta(days=args.reburn_gap)
    rows = []
    for fire in kept.itertuples():
        candidates = perimeters.iloc[index.query(fire.geometry, predicate="intersects")]
        earlier = candidates[(candidates["fire_date"] < fire.fire_date - gap) & (candidates["fire_id"] != fire.fire_id)]
        record = {"fire_id": fire.fire_id, "fire_date": fire.fire_date, "overlap": 0.0, "earlier_fire": "", "earlier_date": pd.NaT}
        if len(earlier):
            area = fire.geometry.area
            pieces = earlier.geometry.intersection(fire.geometry).area
            record["overlap"] = float(fire.geometry.intersection(unary_union(list(earlier.geometry))).area / area) if area else 0.0
            touching = earlier[pieces.to_numpy() >= .01 * area]
            if len(touching):
                latest = touching.sort_values("fire_date").iloc[-1]
                record["earlier_fire"], record["earlier_date"] = latest["fire_id"], latest["fire_date"]
        rows.append(record)
    table = pd.DataFrame(rows)
    table["gap_years"] = (table["fire_date"] - table["earlier_date"]).dt.days / 365.25
    work = Path(args.work)
    work.mkdir(parents=True, exist_ok=True)
    table.to_csv(work / "reburn_fires.csv", index=False)
    fresh = frame[frame["window"] == "fresh"].drop_duplicates("fire_id")
    t = table.merge(fresh[["fire_id", "tile_id", "country", "dnbr_flag", "area_ha", "image_path", "mask_path", "age_days",
                           "frac_cloud", "frac_shadow", "frac_nodata", "scene_time", "own_frac"]], on="fire_id")
    text = ["## Fires on earlier scars", "",
            f"EFFIS perimeters from 2018 on, ETRS89-LAEA. An earlier fire counts when its date is more than {args.reburn_gap} "
            f"days before the fire's, so that one event mapped twice is not counted. {len(t):,} released fires. Earliest "
            f"perimeter loaded: {earliest:%Y-%m-%d}; exports: {cfg['effis_path']}{''.join(', ' + e for e in extra)}.", ""]
    rows = []
    for name, part in [("all", t)] + [(c, g) for c, g in t.groupby("country")]:
        rows.append({"Fires": name, "n": len(part), "On an earlier scar, >= 10 % of the area": share(int((part["overlap"] >= .1).sum()), len(part)),
                     ">= 50 %": share(int((part["overlap"] >= .5).sum()), len(part)),
                     "Median years since, >= 10 %": f"{part.loc[part['overlap'] >= .1, 'gap_years'].median():.1f}"})
    text += [md_table(pd.DataFrame(rows)), ""]
    flags = []
    for name, part in (("on an earlier scar, >= 50 %", t[t["overlap"] >= .5]), ("no earlier scar, < 10 %", t[t["overlap"] < .1])):
        assessed = part[part["dnbr_flag"].isin(["burn", "weak", "none"])]
        flags.append({"Fires": name, "Assessed": len(assessed), "Burn": share(int((assessed["dnbr_flag"] == "burn").sum()), len(assessed)),
                      "Weak": share(int((assessed["dnbr_flag"] == "weak").sum()), len(assessed)),
                      "None": share(int((assessed["dnbr_flag"] == "none").sum()), len(assessed))})
    text += ["### Burn evidence of the first scene", "", md_table(pd.DataFrame(flags)), ""]

    candidates = t[(t["overlap"] >= .5) & t["gap_years"].between(1, 5) & t["area_ha"].between(30, 2000) & clean(t)
                   & t["own_frac"].between(.1, .7)]
    picks = candidates.sample(min(3, len(candidates)), random_state=args.pick_seed) if len(candidates) else candidates
    notes = []
    if len(picks):
        ids = list(picks["fire_id"]) + list(picks["earlier_fire"])
        part = perimeters[perimeters["fire_id"].isin(set(ids))].drop_duplicates("fire_id")
        outlines, outline_crs = dict(zip(part["fire_id"], part.geometry)), perimeters.crs
        mask_cmap, mask_norm = mask_colours()
        n = len(picks)
        panel = WIDTH / CM * .9 / (n + (n - 1) * .05)
        height = .95 + 2 * panel + .1 + .75
        fig = plt.figure(figsize=(WIDTH, height * CM))
        layout = GridSpec(2, n, figure=fig, left=.07, right=.97, top=1 - .95 / height, bottom=.75 / height, wspace=.05,
                          hspace=.1 / panel)
        axes = np.array([[fig.add_subplot(layout[r, c]) for c in range(n)] for r in range(2)])
        for letter, column, (_, row) in zip("abc", axes.T, picks.iterrows()):
            with rasterio.open(Path(args.masks_root) / row["mask_path"]) as src:
                mask, tile_crs, transform = src.read(1), src.crs, src.transform
            top, left, size = window_of(row, mask.shape)
            crop = (slice(top, top + size), slice(left, left + size))
            column[0].imshow(read_rgb(Path(args.images_root) / row["image_path"], FALSE_COLOUR, BAND)[crop])
            column[1].imshow(mask[crop], cmap=mask_cmap, norm=mask_norm)
            for fire_id, colour, style in ((row["earlier_fire"], "#f5d000", (0, (2, 1.4))), (row["fire_id"], "white", "-")):
                geometry = outlines.get(fire_id)
                if geometry is not None:
                    for ax in column:
                        draw_outline(ax, geometry, outline_crs, tile_crs, transform, offset=(top, left),
                                     colour=colour if ax is column[0] else INK, width=.6, style=style)
            for ax in column:
                ax.set_xlim(-.5, size - .5)
                ax.set_ylim(size - .5, -.5)
                frame_panel(ax)
            add_scale(column[0], size)
            halo(column[0])
            column[0].set_title(f"({letter}) {row['fire_date']:%Y} fire on a {row['earlier_date']:%Y} scar\n"
                                f"{100 * row['overlap']:.0f} % of its area", fontsize=5.8, linespacing=1.1, pad=2.5)
            notes.append(f"- fig_reburn ({letter}): fire {row['fire_id']} ({row['tile_id']}) on {row['earlier_fire']}, "
                         f"{100 * row['overlap']:.0f} %")
        axes[0, 0].set_ylabel("False colour", fontsize=6.5, labelpad=2)
        axes[1, 0].set_ylabel("Mask", fontsize=6.5, labelpad=2)
        fig.legend(handles=[Line2D([], [], color=INK, linewidth=.6, label="the fire's outline (white on the image)"),
                            Line2D([], [], color=INK, linewidth=.6, linestyle=(0, (2, 1.4)), label="the earlier fire's (yellow)")],
                   loc="lower center", bbox_to_anchor=(.5, .15 / height), ncol=2, fontsize=5.8, handlelength=1.8,
                   frameon=False)
        save(fig, args.out, "fig_reburn")
    write_numbers(args.out, "reburn", text + ["### Gallery tiles", "", *notes, ""])


# ------------------------------------------------------------------ the labels, CEMS and the dNBR on the same pixels

def confusion(burned, reference, keep):
    return {"tp": int((burned & reference & keep).sum()), "fp": int((burned & ~reference & keep).sum()),
            "fn": int((~burned & reference & keep).sum()), "tn": int((~burned & ~reference & keep).sum())}


def scores(tp, fp, fn, tn):
    return {"precision": tp / (tp + fp) if tp + fp else np.nan, "recall": tp / (tp + fn) if tp + fn else np.nan,
            "f1": 2 * tp / (2 * tp + fp + fn) if tp + fp + fn else np.nan, "iou": tp / (tp + fp + fn) if tp + fp + fn else np.nan}


def cems_arrays(job):
    """The tile's mask, the CEMS map on the tile's grid (2 burned, 1 not, 0 outside the CEMS area) and the dNBR."""
    with rasterio.open(job["mask_path"]) as src:
        mask, transform, crs = src.read(1), src.transform, src.crs
    with rasterio.open(job["dnbr_path"]) as src:
        raw = src.read(1)
    dnbr = np.where(raw == -32768, np.nan, raw / 1000.0)
    with rasterio.open(job["del_path"]) as src:
        data, ref_transform, ref_crs = src.read(1), src.transform, src.crs
    # In the wildfires-cems rasters 1 is burned, 0 not, 255 outside the image (as in compare_cems.py).
    source = np.where(data != 255, (data == 1).astype(np.uint8) + 1, 0).astype(np.uint8)
    target = np.zeros(mask.shape, np.uint8)
    reproject(source, target, src_transform=ref_transform, src_crs=ref_crs, dst_transform=transform, dst_crs=crs,
              src_nodata=0, dst_nodata=0, resampling=Resampling.nearest)
    return mask, target, dnbr


def cems_maps(mask, dnbr):
    with np.errstate(invalid="ignore"):
        return {"label": mask == MASK_VALUE["fresh"], "dnbr10": dnbr >= .10, "dnbr27": dnbr >= .27,
                "label_dnbr10": (mask == MASK_VALUE["fresh"]) & (dnbr >= .10)}


def cems_job(job):
    try:
        mask, target, dnbr = cems_arrays(job)
        keep = (target > 0) & np.isin(mask, KEEP_CLASSES) & np.isfinite(dnbr)
        reference = target == 2
        result = {"tile_id": job["tile_id"], "sample": job["sample"], "activation": job["activation"],
                  "pixels": int(keep.sum()), "cems_burned": int((reference & keep).sum())}
        maps = cems_maps(mask, dnbr)
        for name, burned in maps.items():
            result.update({f"{name}_{k}": v for k, v in confusion(burned, reference, keep).items()})
        result.update({f"label_vs_dnbr10_{k}": v for k, v in confusion(maps["label"], maps["dnbr10"], keep).items()})
        return result
    except Exception as error:  # a broken pair must not stop the others
        return {"tile_id": job["tile_id"], "sample": job["sample"], "error": f"{type(error).__name__}: {error}"[:200]}


def by_activation(table, name, reps, seed):
    """Pooled scores of one comparison, with 95 % intervals from resampling whole activations."""
    sums = table.groupby("activation")[[f"{name}_{k}" for k in COUNTS]].sum().to_numpy()
    estimate = scores(*sums.sum(0))
    rng = np.random.default_rng(seed)
    draws = pd.DataFrame([scores(*sums[rng.integers(len(sums), size=len(sums))].sum(0)) for _ in range(reps)])
    return {m: ci(estimate[m], *np.nanpercentile(draws[m], [2.5, 97.5])) for m in ("precision", "recall", "f1", "iou")}


def cems_jobs(frame, args):
    pairs = pd.read_csv(args.cems_tiles)
    evidence_path = Path(args.cems_dnbr) / "burn_evidence.csv"
    if not evidence_path.exists():
        raise SystemExit(f"No {evidence_path}. Make the dNBR of the CEMS tiles first:\n  python tools/burn_evidence.py "
                         f"--index {args.index} --masks-root {args.masks_root} --out {args.cems_dnbr} --tiles-from "
                         f"{args.cems_tiles} --save-dnbr --workers 16")
    evidence = pd.read_csv(evidence_path).drop_duplicates("tile_id", keep="last").set_index("tile_id")
    rasters = {path.name[:-len("_DEL.tif")]: path for path in Path(args.cems_root).rglob("*_DEL.tif")}
    tiles = frame.set_index("tile_id")
    jobs, skipped = [], Counter()
    for pair in pairs.itertuples():
        if pair.tile_id not in tiles.index:
            skipped["tile not in the index"] += 1
            continue
        tile = tiles.loc[pair.tile_id]
        dnbr_path = Path(args.cems_dnbr) / "dnbr" / Path(tile["mask_path"]).name.replace("_mask.tif", "_dnbr.tif")
        if not dnbr_path.exists():
            skipped["no dNBR raster (burn_evidence.py found no usable pre-fire or post-fire L2A)"] += 1
            continue
        if pair.sample not in rasters:
            skipped["no CEMS raster"] += 1
            continue
        scene = pd.to_datetime(tile["scene_time"], format="mixed", utc=True).tz_convert(None)
        cems_date = (scene - pd.Timedelta(days=float(pair.days_after_cems_date))).normalize()
        pre = pd.to_datetime(evidence["pre_date"].get(pair.tile_id), format="mixed", utc=True, errors="coerce")
        if pd.isna(pre) or pre.tz_convert(None).normalize() >= cems_date:
            # A pre-fire image taken after the CEMS fire would hide that fire's burn from the dNBR.
            skipped["pre-fire image not before the CEMS date"] += 1
            continue
        jobs.append({"tile_id": pair.tile_id, "sample": pair.sample, "activation": pair.activation,
                     "mask_path": str(Path(args.masks_root) / tile["mask_path"]), "dnbr_path": str(dnbr_path),
                     "del_path": str(rasters[pair.sample]), "image_path": str(Path(args.images_root) / tile["image_path"]),
                     "area_ha": tile["area_ha"], "country": tile.get("country", "")})
    return jobs, skipped, len(pairs)


def cems_picks(table, jobs, count=3):
    """A pair near the pooled agreement of the labels, the pair where the dNBR does best against the labels, and the
    pair where it does worst, among pairs with at least 10 ha of CEMS burn."""
    t = table[(table["cems_burned"] >= 1000) & (table["cems_burned"] >= .02 * table["pixels"])].copy()
    if t.empty:
        return []
    for name in ("label", "dnbr10"):
        t[f"{name}_f1"] = [scores(*r)["f1"] for r in t[[f"{name}_{k}" for k in COUNTS]].to_numpy()]
    pooled = scores(*table[[f"label_{k}" for k in COUNTS]].sum().to_numpy())["f1"]
    t["gain"] = t["dnbr10_f1"] - t["label_f1"]
    rankings = [t["label_f1"].sub(pooled).abs().sort_values().index, t["gain"].sort_values(ascending=False).index,
                t["gain"].sort_values().index]
    by_key = {(j["tile_id"], j["sample"]): j for j in jobs}
    picks, seen = [], set()
    for ranking in rankings[:count]:
        for position in ranking:      # the best of each ranking not shown already
            row = t.loc[position]
            if (row["tile_id"], row["sample"]) not in seen:
                seen.add((row["tile_id"], row["sample"]))
                picks.append((row, by_key[(row["tile_id"], row["sample"])]))
                break
    return picks


def figure_cems(picks, args):
    dnbr_cmap = plt.get_cmap("RdBu_r").copy()
    dnbr_cmap.set_bad("#d9d7d0")
    dnbr_norm = TwoSlopeNorm(vmin=-.3, vcenter=0, vmax=.6)
    agree = ListedColormap(["white", AGREE, CLASS_COLOUR["old"], BLUE, OUTSIDE])
    n = len(picks)
    panel = WIDTH / CM * .955 / (4 + 3 * .05)
    head, gap, foot = .9, .55, 1.25
    height = head + n * panel + (n - 1) * gap + foot
    fig = plt.figure(figsize=(WIDTH, height * CM))
    layout = GridSpec(n, 4, figure=fig, left=.04, right=.995, top=1 - head / height, bottom=foot / height, wspace=.05,
                      hspace=gap / panel)
    notes = []
    for k, (letter, (row, job)) in enumerate(zip("abc", picks)):
        mask, target, dnbr = cems_arrays(job)
        top, left, size = 0, 0, mask.shape[0]   # the whole tile: the F1 scores above the panels are computed over it
        crop = (slice(top, top + size), slice(left, left + size))
        keep = (target > 0) & np.isin(mask, KEEP_CLASSES) & np.isfinite(dnbr)
        reference = target == 2
        maps = cems_maps(mask, dnbr)
        axes = [fig.add_subplot(layout[k, j]) for j in range(4)]
        axes[0].imshow(read_rgb(job["image_path"], FALSE_COLOUR, BAND)[crop])
        axes[1].imshow(np.ma.masked_invalid(dnbr[crop]), cmap=dnbr_cmap, norm=dnbr_norm)
        for ax, burned in ((axes[2], maps["label"]), (axes[3], maps["dnbr10"])):
            picture = np.zeros(mask.shape, np.uint8)
            picture[burned & reference] = 1
            picture[burned & ~reference] = 2
            picture[~burned & reference] = 3
            picture[~keep] = 4
            ax.imshow(picture[crop], cmap=agree, vmin=-.5, vmax=4.5)
        for ax, colour, style in ((axes[0], CEMS_LINE, (0, (2.5, 1.2))), (axes[1], INK, "-")):
            ax.contour(reference[crop].astype(float), levels=[.5], colors=[colour], linewidths=.55, linestyles=[style])
        axes[0].contour(maps["label"][crop].astype(float), levels=[.5], colors=["white"], linewidths=.55)
        for ax in axes:
            ax.set_xlim(-.5, size - .5)
            ax.set_ylim(size - .5, -.5)
            frame_panel(ax)
        add_scale(axes[0], size)
        halo(axes[0])
        f1_label = scores(*[row[f"label_{c}"] for c in COUNTS])["f1"]
        f1_dnbr = scores(*[row[f"dnbr10_{c}"] for c in COUNTS])["f1"]
        axes[0].set_ylabel(f"({letter}) {job['sample'].split('_')[0]}", fontsize=6, labelpad=2)
        axes[2].set_title(f"F1 {100 * f1_label:.0f} %", fontsize=5.8, pad=1.5)
        axes[3].set_title(f"F1 {100 * f1_dnbr:.0f} %", fontsize=5.8, pad=1.5)
        if k == 0:
            for ax, text_ in zip(axes, ("False colour", "dNBR", "burn2scar against CEMS", "dNBR ≥ 0.10 against CEMS")):
                ax.text(.5, 1 + .55 / panel, text_, transform=ax.transAxes, ha="center", va="bottom", fontsize=6.3)
        notes.append(f"- fig_cems_dnbr ({letter}): tile {job['tile_id']}, CEMS {job['sample']}, F1 labels {100 * f1_label:.1f} %, "
                     f"dNBR >= 0.10 {100 * f1_dnbr:.1f} %")
    handles = [Patch(facecolor=AGREE, label="Both"), Patch(facecolor=CLASS_COLOUR["old"], label="burn2scar or dNBR only"),
               Patch(facecolor=BLUE, label="CEMS only"),
               Patch(facecolor=OUTSIDE, edgecolor="#8a8984", linewidth=.3, label="Not compared")]
    fig.legend(handles=handles, loc="lower left", bbox_to_anchor=(.03, .45 / height), ncol=4, handlelength=1,
               handletextpad=.35, columnspacing=.9, fontsize=5.8)
    lines = [Line2D([], [], color=INK2, linewidth=.6, label="fresh_burn (white on the image)"),
             Line2D([], [], color=CEMS_LINE, linewidth=.7, linestyle=(0, (2.5, 1.2)), label="CEMS (black on the dNBR)")]
    fig.legend(handles=lines, loc="lower left", bbox_to_anchor=(.03, .02 / height), ncol=2, handlelength=1.8,
               handletextpad=.4, columnspacing=1, fontsize=5.8)
    cax = fig.add_axes([.74, .38 / height, .23, .16 / height])
    bar = fig.colorbar(ScalarMappable(norm=dnbr_norm, cmap=dnbr_cmap), cax=cax, orientation="horizontal")
    bar.set_ticks([-.3, 0, .3, .6], labels=["−0.3", "0", "0.3", "0.6"])
    bar.ax.tick_params(labelsize=5.5, length=2, width=.4, pad=1)
    bar.outline.set_linewidth(.3)
    fig.text(.73, .46 / height, "dNBR", ha="right", va="center", fontsize=6)
    save(fig, args.out, "fig_cems_dnbr")
    return notes


def run_cems(frame, args):
    jobs, skipped, total = cems_jobs(frame, args)
    if not jobs:
        raise SystemExit(f"No tile and CEMS area pair to compare; left out: {dict(skipped)}")
    print(f"CEMS: comparing {len(jobs)} tile and area pairs with {args.workers} workers", flush=True)
    with Pool(args.workers) as pool:
        table = pd.DataFrame(pool.imap_unordered(cems_job, jobs, chunksize=4))
    work = Path(args.work)
    work.mkdir(parents=True, exist_ok=True)
    table.to_csv(work / "cems_dnbr_pairs.csv", index=False)
    if "error" in table:
        print(f"  {table['error'].notna().sum()} pairs could not be read; see {work / 'cems_dnbr_pairs.csv'}", flush=True)
        table = table[table["error"].isna()]
    table = table[table["pixels"] > 0]
    rows = []
    for name, title in CEMS_MAPS.items():
        result = by_activation(table, name, args.reps, args.seed)
        rows.append({"Map against CEMS": title, "Precision": result["precision"], "Recall": result["recall"],
                     "F1": result["f1"], "IoU": result["iou"]})
    against = by_activation(table, "label_vs_dnbr10", args.reps, args.seed)
    rows.append({"Map against CEMS": "burn2scar fresh_burn against dNBR ≥ 0.10 (dNBR as the reference)",
                 "Precision": against["precision"], "Recall": against["recall"], "F1": against["f1"], "IoU": against["iou"]})
    text = ["## The labels, CEMS and the dNBR on the same pixels", "",
            f"Tile and area pairs in {args.cems_tiles}: {total}. Compared: {len(table)} pairs from "
            f"{table['activation'].nunique()} activations, {table['pixels'].sum() / 100:,.0f} ha of pixels, "
            f"{table['cems_burned'].sum() / 100:,.0f} ha burned in CEMS.",
            *[f"- Left out, {reason}: {count}" for reason, count in skipped.items()], "",
            "Pixels compared: inside the CEMS area, of a mask class compare_cems.py compares (clear, fresh_burn, old_burn, "
            "water, older_burn), with a dNBR value. Percent, with 95 % intervals from resampling whole activations.", "",
            md_table(pd.DataFrame(rows)), ""]
    picks = cems_picks(table, jobs)
    notes = figure_cems(picks, args) if picks else []
    write_numbers(args.out, "cems", text + ["### Gallery pairs", "", *notes, ""])


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--index", default="data/v4/index.csv")
    parser.add_argument("--images-root", default="data/v1")
    parser.add_argument("--masks-root", default="data/v4")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--review", default="review2")
    parser.add_argument("--reviewers", nargs=2, default=["param", "silvia"])
    parser.add_argument("--judge", default="judge")
    parser.add_argument("--fresh-evidence", default="validation/dnbr/burn_evidence.csv")
    parser.add_argument("--old-evidence", default="validation/dnbr_old/burn_evidence.csv")
    parser.add_argument("--cems-tiles", default="validation/cems_tiles.csv")
    parser.add_argument("--cems-root", default="data/cems", help="the wildfires-cems rasters, as for compare_cems.py")
    parser.add_argument("--cems-dnbr", default="validation/dnbr_cems", help="burn_evidence.py --out for the CEMS tiles")
    parser.add_argument("--prefire-root", default="data/prefire", help="the pre-fire images of the review pictures")
    parser.add_argument("--effis-extra", nargs="*", default=[], help="further EFFIS exports for the re-burn part, if the "
                        "config does not list the 12 Nov export (2018 on)")
    parser.add_argument("--work", default="validation/labels", help="tables the parts share")
    parser.add_argument("--out", default="paper/figures")
    parser.add_argument("--only", choices=PARTS)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--tiles", type=int, default=400, help="tiles per group for the spectra")
    parser.add_argument("--pixels", type=int, default=200, help="pixels per tile for the spectra")
    parser.add_argument("--reps", type=int, default=2000, help="bootstrap repeats for the review")
    parser.add_argument("--reburn-gap", type=int, default=30, help="days; an earlier perimeter closer in time is the same event")
    parser.add_argument("--seed", type=int, default=20261119)
    parser.add_argument("--pick-seed", type=int, default=3, help="change it to draw other example tiles")
    args = parser.parse_args()

    warnings.filterwarnings("ignore", category=FutureWarning)
    warnings.filterwarnings("ignore", message=".*shapely.*")
    set_style()
    Path(args.out).mkdir(parents=True, exist_ok=True)
    parts = [args.only] if args.only else PARTS
    frame = None
    if set(parts) - {"review"}:
        index = pd.read_csv(args.index, low_memory=False)
        frame = index[index["split"].notna()].reset_index(drop=True)
    for part in parts:
        print(f"== {part}", flush=True)
        if part == "review":
            run_review(args)
        elif part == "snow":
            run_snow(frame, args)
        elif part == "spectra":
            run_spectra(frame, args)
        elif part == "oldburn":
            run_oldburn(frame, args)
        elif part == "reburn":
            run_reburn(frame, args)
        elif part == "cems":
            run_cems(frame, args)
    print(f"Done. Send {Path(args.out) / 'label_numbers.md'} and the new PDF files.")


if __name__ == "__main__":
    main()
