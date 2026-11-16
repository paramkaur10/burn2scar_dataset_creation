#!/usr/bin/env python3
"""Three more figures for the burn2scar paper, in the style of make_paper_figures.py.

  python tools/make_extra_figures.py --out paper/figures

fig_evidence.pdf         The burn evidence on six example tiles: false colour with the EFFIS outline, the mask and the
                         dNBR. The dNBR rasters come from tools/burn_evidence.py, which this script runs on the six tiles
                         (into validation/dnbr_fig) unless they are there already: about a minute, and it needs the internet.
fig_evidence_trends.pdf  The burn evidence by fire size and by days from the fire to the scene, from the index.
fig_predictions.pdf      SegFormer-B2 (spatial split, seed 0) on five test tiles. Needs results_v4 and torch; it uses the
                         GPU if there is one, else the CPU.

Make one figure alone with --only evidence, trends or predictions. Swap the examples with --evidence-tiles or
--prediction-tiles (tile IDs); the IDs used are written to examples_extra.md next to the figures.
"""
import argparse
import json
import subprocess
import sys
import warnings
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import geopandas as gpd  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import rasterio  # noqa: E402
from matplotlib.cm import ScalarMappable  # noqa: E402
from matplotlib.colors import TwoSlopeNorm  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402

TOOLS = Path(__file__).resolve().parent
sys.path.insert(0, str(TOOLS))
from make_paper_figures import (CLASS_COLOUR, CLASS_LABEL, CLASSES, CM, INK, INK2, SIZE_BINS, SIZE_LABELS,  # noqa: E402
                                WIDTH, draw_outline, frame_panel, grid, mask_colours, number, own_perimeters, read_rgb,
                                save, scale_bar, set_style)
from review_points import BAND, FALSE_COLOUR  # noqa: E402

ZOOM_HA, ZOOM_PX = 50, 96   # fires under 50 ha are shown in a 960 m window at the tile centre, which is the fire's
PRED_VALUE = {"clear": 0, "fresh_burn": 2, "old_burn": 3, "cloud": 4, "shadow": 5, "water": 6}
SCORED = [0, 2, 3, 6]       # fresh_burn is scored against clear, old_burn and water, as in the paper
DAY_BINS, DAY_LABELS = [0, 10, 20, 30, 45, 62], ["<10", "10–20", "20–30", "30–45", "45–60"]
SIZE_GROUPS = ([0, 30, 100, np.inf], ["under 30 ha", "30–100 ha", "100–2000 ha"])
NODATA = -32768


def one(part, seed):
    return part.sample(1, random_state=seed).iloc[0] if len(part) else None


def clean(frame):
    return (frame["frac_cloud"] < .01) & (frame["frac_shadow"] < .01) & (frame["frac_nodata"] < .01)


def window_of(row, shape):
    """(top, left, size): a 960 m window at the centre for small fires, else the whole tile."""
    if row["area_ha"] < ZOOM_HA and min(shape) > ZOOM_PX:
        return (shape[0] - ZOOM_PX) // 2, (shape[1] - ZOOM_PX) // 2, ZOOM_PX
    return 0, 0, shape[0]


def tiles_by_id(frame, ids):
    rows = []
    for tile in ids:
        match = frame[frame["tile_id"] == tile]
        if match.empty:
            raise SystemExit(f"Tile {tile} is not a released tile of the index.")
        rows.append(match.iloc[0])
    return rows


# ------------------------------------------------------------------ burn evidence on example tiles

def pick_evidence(frame, old_rows_path, seed):
    """Burn on a large and a small fire, weak, none, and two old-window tiles: one scar that stands out, one faded."""
    fresh = frame[(frame["window"] == "fresh") & clean(frame)]
    picks = []
    for rule in ((fresh["dnbr_flag"] == "burn") & (fresh["area_ha"] >= 300) & fresh["own_frac"].between(.2, .6),
                 (fresh["dnbr_flag"] == "burn") & fresh["area_ha"].between(5, 20),
                 (fresh["dnbr_flag"] == "weak") & fresh["area_ha"].between(30, 300),
                 (fresh["dnbr_flag"] == "none") & fresh["area_ha"].between(30, 300)):
        row = one(fresh[rule], seed)
        if row is not None:
            picks.append(row)
    old = frame[(frame["window"] == "old") & clean(frame) & (frame["area_ha"] >= 100) & frame["own_frac"].between(.15, .6)]
    if Path(old_rows_path).exists():
        rows = pd.read_csv(old_rows_path, usecols=["tile_id", "flag", "contrast"]).drop_duplicates("tile_id", keep="last")
        old = old.drop(columns=[c for c in ("flag", "contrast") if c in old.columns]).merge(rows, on="tile_id")
        for rule in (old["contrast"] >= .15, (old["contrast"] < .02) & (old["flag"] == "none")):
            row = one(old[rule], seed)
            if row is not None:
                picks.append(row)
    else:
        print(f"No {old_rows_path}, so the two old-window tiles are picked at random.", flush=True)
        picks += [row for _, row in old.sample(min(2, len(old)), random_state=seed).iterrows()]
    return picks


def dnbr_path(folder, row):
    return Path(folder) / "dnbr" / Path(row["mask_path"]).name.replace("_mask.tif", "_dnbr.tif")


def ensure_dnbr(rows, args):
    """Runs burn_evidence.py with --save-dnbr on the tiles whose dNBR raster is missing; returns its rows by tile."""
    folder = Path(args.dnbr_dir)
    rows_csv = folder / "burn_evidence.csv"
    missing = [row["tile_id"] for row in rows if not dnbr_path(folder, row).exists()]
    if missing:
        folder.mkdir(parents=True, exist_ok=True)
        if rows_csv.exists():   # a tile with a row but no raster would be skipped as done
            done = pd.read_csv(rows_csv)
            done[~done["tile_id"].isin(missing)].to_csv(rows_csv, index=False)
        wanted = folder / "figure_tiles.csv"
        pd.DataFrame({"tile_id": missing}).to_csv(wanted, index=False)
        command = [sys.executable, str(TOOLS / "burn_evidence.py"), "--index", args.index, "--masks-root", args.masks_root,
                   "--config", args.config, "--tiles-from", str(wanted), "--save-dnbr", "--out", str(folder),
                   "--workers", "6"]
        print("Running " + " ".join(command), flush=True)
        subprocess.run(command, check=True)
    if not rows_csv.exists():
        return pd.DataFrame(columns=["flag", "contrast"])
    return pd.read_csv(rows_csv).drop_duplicates("tile_id", keep="last").set_index("tile_id")


def read_dnbr(path):
    if not path.exists():
        return None
    with rasterio.open(path) as src:
        raw = src.read(1)
    return np.ma.masked_equal(raw, NODATA).astype(np.float32) / 1000


def evidence_title(letter, row, result):
    if row["window"] == "old":
        contrast = result.get("contrast") if result is not None else np.nan
        return f"({letter}) old window\n" + (f"{contrast:+.2f} above ring" if pd.notna(contrast) else f"day {row['age_days']:.0f}")
    flag = result.get("flag") if result is not None else row.get("dnbr_flag")
    return f"({letter}) {flag}\n{row['area_ha']:.0f} ha, day {row['age_days']:.0f}"


def add_scale(ax, size):
    scale_bar(ax, size, 200 if size <= ZOOM_PX else 1000)


def figure_evidence(frame, args, out):
    picks = tiles_by_id(frame, args.evidence_tiles) if args.evidence_tiles else pick_evidence(frame, args.old_evidence, args.pick_seed)
    if not picks:
        print("No tiles for the evidence figure.", flush=True)
        return []
    results = ensure_dnbr(picks, args)
    outlines, outline_crs = own_perimeters(args.config, [row["fire_id"] for row in picks])
    mask_cmap, mask_norm = mask_colours()
    dnbr_cmap = plt.get_cmap("RdBu_r").copy()
    dnbr_cmap.set_bad("#d9d7d0")
    dnbr_norm = TwoSlopeNorm(vmin=-.3, vcenter=0, vmax=.6)
    fig, axes = plt.subplots(3, len(picks), figsize=(WIDTH, 8.6 * CM), squeeze=False)
    for letter, column, row in zip("abcdefgh", axes.T, picks):
        with rasterio.open(Path(args.masks_root) / row["mask_path"]) as src:
            mask, tile_crs, transform = src.read(1), src.crs, src.transform
        top, left, size = window_of(row, mask.shape)
        crop = (slice(top, top + size), slice(left, left + size))
        column[0].imshow(read_rgb(Path(args.images_root) / row["image_path"], FALSE_COLOUR, BAND)[crop])
        column[1].imshow(mask[crop], cmap=mask_cmap, norm=mask_norm)
        dnbr = read_dnbr(dnbr_path(args.dnbr_dir, row))
        if dnbr is not None:
            column[2].imshow(dnbr[crop], cmap=dnbr_cmap, norm=dnbr_norm)
        geometry = outlines.get(row["fire_id"])
        if geometry is not None:
            draw_outline(column[0], geometry, outline_crs, tile_crs, transform, offset=(top, left))
            draw_outline(column[2], geometry, outline_crs, tile_crs, transform, offset=(top, left), colour=INK, width=.5)
            far = gpd.GeoSeries([geometry], crs=outline_crs).to_crs(tile_crs).buffer(300).iloc[0]
            draw_outline(column[2], far, tile_crs, tile_crs, transform, offset=(top, left), colour=INK, width=.4,
                         style=(0, (2, 1.5)))
        for ax in column:
            ax.set_xlim(-.5, size - .5)
            ax.set_ylim(size - .5, -.5)
            frame_panel(ax)
        add_scale(column[0], size)
        result = results.loc[row["tile_id"]] if row["tile_id"] in results.index else None
        column[0].set_title(evidence_title(letter, row, result), fontsize=5.8, linespacing=1.1, pad=2.5)
    for ax, label in zip(axes[:, 0], ("False colour", "Mask", "dNBR")):
        ax.set_ylabel(label, fontsize=6.5, labelpad=2)
    fig.subplots_adjust(left=.035, right=.995, top=.91, bottom=.17, wspace=.05, hspace=.05)
    classes = [Patch(facecolor=CLASS_COLOUR[name], edgecolor="#8a8984", linewidth=.3, label=CLASS_LABEL[name]) for name in CLASSES]
    fig.legend(handles=classes, loc="lower center", bbox_to_anchor=(.5, .075), ncol=len(classes), columnspacing=.7,
               handlelength=1, handletextpad=.3, fontsize=5.8)
    lines = [Line2D([], [], color=INK, linewidth=.6, label="EFFIS outline (white on the image)"),
             Line2D([], [], color=INK, linewidth=.5, linestyle=(0, (2, 1.5)), label="300 m outside it")]
    fig.legend(handles=lines, loc="lower left", bbox_to_anchor=(.03, .0), ncol=2, columnspacing=1, handlelength=1.8,
               handletextpad=.4, fontsize=5.8)
    cax = fig.add_axes([.72, .035, .25, .016])
    bar = fig.colorbar(ScalarMappable(norm=dnbr_norm, cmap=dnbr_cmap), cax=cax, orientation="horizontal")
    bar.set_ticks([-.3, 0, .3, .6], labels=["\u22120.3", "0", "0.3", "0.6"])
    bar.ax.tick_params(labelsize=5.5, length=2, width=.4, pad=1)
    bar.outline.set_linewidth(.3)
    fig.text(.71, .043, "dNBR", ha="right", va="center", fontsize=6)
    save(fig, out, "fig_evidence")
    return [f"fig_evidence: {row['tile_id']} ({row['window']}, {row['area_ha']:.0f} ha)" for row in picks]


# ------------------------------------------------------------------ burn evidence by size and by day

def trends_tables(frame, rows_path):
    fresh = frame[(frame["window"] == "fresh") & frame["dnbr_flag"].isin(["burn", "weak", "none"])].copy()
    fresh["size"] = pd.cut(fresh["area_ha"], SIZE_BINS, right=False, labels=SIZE_LABELS)
    flags = fresh.groupby("size", observed=True)["dnbr_flag"]
    by_size = pd.DataFrame({"fires": flags.size(), "burn": 100 * flags.apply(lambda s: (s == "burn").mean()),
                            "weak": 100 * flags.apply(lambda s: (s == "weak").mean())})
    days = pd.cut(fresh["age_days"], DAY_BINS, right=False, labels=DAY_LABELS)
    group = pd.cut(fresh["area_ha"], SIZE_GROUPS[0], right=False, labels=SIZE_GROUPS[1])
    by_day = 100 * (fresh["dnbr_flag"] == "burn").groupby([days, group], observed=True).mean().unstack()
    ring = np.nan
    if Path(rows_path).exists():
        rows = pd.read_csv(rows_path, usecols=["tile_id", "dnbr_ring_median"]).drop_duplicates("tile_id", keep="last")
        rings = rows.loc[rows["tile_id"].isin(fresh["tile_id"]), "dnbr_ring_median"].dropna()
        ring = 100 * (rings >= .10).mean() if len(rings) else np.nan
    return by_size, by_day, ring


def draw_trends(by_size, by_day, ring, out):
    fig, (left, right) = plt.subplots(1, 2, figsize=(WIDTH, 5.6 * CM), gridspec_kw={"width_ratios": [1.5, 1]})
    x = np.arange(len(by_size), dtype=float)
    left.bar(x, by_size["burn"], width=.62, color=CLASS_COLOUR["fresh"], label="burn", zorder=3)
    left.bar(x, by_size["weak"], width=.62, bottom=by_size["burn"], color=CLASS_COLOUR["older"], label="weak", zorder=3)
    for xi, burn in zip(x, by_size["burn"]):
        left.text(xi, burn / 2, f"{burn:.0f}", ha="center", va="center", color="white", fontsize=6, zorder=4)
    ticks = [f"{str(label).replace('-', '–')}\n{number(n)}" for label, n in zip(by_size.index, by_size["fires"])]
    if pd.notna(ring):   # the unburned rings, as the background any burn signal is read against
        x = np.append(x, len(x) + .3)
        left.bar(x[-1], ring, width=.62, color="#8a8984", zorder=3)
        left.text(x[-1], ring + 2, f"{ring:.1f}", ha="center", va="bottom", fontsize=6, color=INK2, zorder=4)
        ticks.append("clear\nrings")
    left.set_xticks(x, ticks)
    left.tick_params(axis="x", labelsize=5.4)
    left.set(ylim=(0, 100), ylabel="Flagged (%)", xlabel="Fire size (ha) and fires assessed")
    left.set_title("(a) By fire size")
    left.legend(loc="lower right", bbox_to_anchor=(1, .99), ncol=2, handlelength=1, handletextpad=.4, columnspacing=.8,
                borderaxespad=0)
    grid(left)
    styles = [(0, (1, 1.2)), (0, (4, 1.5)), "-"]
    markers = ["^", "s", "o"]
    colours = [CLASS_COLOUR["older"], CLASS_COLOUR["old"], CLASS_COLOUR["fresh"]]
    for (name, series), style, marker, colour in zip(by_day.items(), styles, markers, colours):
        right.plot(np.arange(len(series)), series.to_numpy(), linestyle=style, marker=marker, markersize=3, color=colour,
                   linewidth=1, label=str(name), zorder=3)
    right.set_xticks(np.arange(len(by_day.index)), [str(v) for v in by_day.index])
    right.set_xlim(-.35, len(by_day.index) - .65)
    right.set(ylim=(40, 100), ylabel="Flagged burn (%)", xlabel="Days from the fire date to the scene")
    right.set_title("(b) By days after the fire")
    right.legend(loc="lower left", handlelength=2.2, handletextpad=.4, borderaxespad=.2)
    grid(right)
    fig.subplots_adjust(left=.085, right=.985, top=.9, bottom=.24, wspace=.27)
    save(fig, out, "fig_evidence_trends")


# ------------------------------------------------------------------ model predictions

def pick_predictions(frame, seed):
    test = frame[frame["split"] == "test"]
    tidy, fresh = clean(test), test["window"] == "fresh"
    rules = [("large fire", fresh & (test["dnbr_flag"] == "burn") & (test["area_ha"] >= 300) & test["own_frac"].between(.2, .6) & tidy),
             ("small fire", fresh & (test["dnbr_flag"] == "burn") & test["area_ha"].between(5, 20) & tidy),
             ("both ages", (test["frac_fresh"] >= .05) & (test["frac_old"] >= .05) & tidy),
             ("cloud and shadow", fresh & (test["frac_cloud"] >= .05) & (test["frac_shadow"] >= .02) & (test["frac_fresh"] >= .05)),
             ("flagged none", fresh & (test["dnbr_flag"] == "none") & test["area_ha"].between(30, 300) & tidy)]
    picks = []
    for name, rule in rules:
        row = one(test[rule], seed)
        if row is not None:
            picks.append((name, row))
    return picks


def predict(rows, args):
    """{tile_id: mask values predicted by the model}, loaded as tools/extra_inference.py loads it."""
    import torch
    from train_baseline import ALL_BANDS, TASKS, build_model
    folder = Path(args.results) / f"{args.model}__split__multiclass__seed{args.seed}"
    metrics = json.loads((folder / "metrics.json").read_text())
    channels = [ALL_BANDS.index(b) for b in metrics.get("bands", ALL_BANDS)]
    if metrics["spec"].get("norm") == "reflectance":
        mean, std = np.zeros(13, np.float32), np.ones(13, np.float32)
    else:
        stats = json.loads((Path(args.results) / "band_stats.json").read_text())[metrics["split"]]
        mean, std = np.array(stats["mean"], np.float32), np.array(stats["std"], np.float32)
    classes = TASKS["multiclass"]["classes"]
    to_value = np.array([PRED_VALUE[name] for name in classes], np.uint8)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model({**metrics["spec"], "weights": None}, len(classes), len(channels)).to(device)
    model.load_state_dict(torch.load(folder / "best.pt", map_location=device, weights_only=True))
    model.eval()
    predicted = {}
    with torch.no_grad():
        for row in rows:
            with rasterio.open(Path(args.images_root) / row["image_path"]) as src:
                image = src.read().astype(np.float32)
            x = (np.clip(image[channels] / 10000.0, 0, 1.5) - mean[channels][:, None, None]) / std[channels][:, None, None]
            logits = model(torch.from_numpy(np.ascontiguousarray(x))[None].to(device))
            predicted[row["tile_id"]] = to_value[logits.argmax(1)[0].cpu().numpy()]
    return predicted


def fresh_f1(mask, predicted):
    scored = np.isin(mask, SCORED)
    truth, guess = (mask == 2) & scored, (predicted == 2) & scored
    tp, fp, fn = int((truth & guess).sum()), int((~truth & guess).sum()), int((truth & ~guess).sum())
    return 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) else np.nan


def figure_predictions(frame, args, out):
    if args.prediction_tiles:
        picks = [("", row) for row in tiles_by_id(frame, args.prediction_tiles)]
    else:
        picks = pick_predictions(frame, args.pick_seed)
    if not picks:
        print("No tiles for the predictions figure.", flush=True)
        return []
    try:
        predicted = predict([row for _, row in picks], args)
    except Exception as error:   # the other figures do not depend on this one
        print(f"No predictions figure: {type(error).__name__}: {error}", flush=True)
        return []
    outlines, outline_crs = own_perimeters(args.config, [row["fire_id"] for _, row in picks])
    mask_cmap, mask_norm = mask_colours()
    fig, axes = plt.subplots(3, len(picks), figsize=(WIDTH, 8.4 * CM), squeeze=False)
    for letter, column, (name, row) in zip("abcdefgh", axes.T, picks):
        with rasterio.open(Path(args.masks_root) / row["mask_path"]) as src:
            mask, tile_crs, transform = src.read(1), src.crs, src.transform
        guess = predicted[row["tile_id"]]
        top, left, size = window_of(row, mask.shape)
        crop = (slice(top, top + size), slice(left, left + size))
        column[0].imshow(read_rgb(Path(args.images_root) / row["image_path"], FALSE_COLOUR, BAND)[crop])
        column[1].imshow(mask[crop], cmap=mask_cmap, norm=mask_norm)
        column[2].imshow(guess[crop], cmap=mask_cmap, norm=mask_norm)
        if row["fire_id"] in outlines:
            draw_outline(column[0], outlines[row["fire_id"]], outline_crs, tile_crs, transform, offset=(top, left))
        for ax in column:
            ax.set_xlim(-.5, size - .5)
            ax.set_ylim(size - .5, -.5)
            frame_panel(ax)
        add_scale(column[0], size)
        f1 = fresh_f1(mask, guess)
        score = f"fresh F1 {f1:.2f}" if pd.notna(f1) else "no fresh_burn"
        column[0].set_title(f"({letter}) {name or row['window']}\n{score}", fontsize=5.8, linespacing=1.1, pad=2.5)
    for ax, label in zip(axes[:, 0], ("False colour", "Labels", "SegFormer-B2")):
        ax.set_ylabel(label, fontsize=6.5, labelpad=2)
    fig.subplots_adjust(left=.035, right=.995, top=.91, bottom=.1, wspace=.05, hspace=.05)
    classes = [Patch(facecolor=CLASS_COLOUR[name], edgecolor="#8a8984", linewidth=.3, label=CLASS_LABEL[name]) for name in CLASSES]
    fig.legend(handles=classes, loc="lower center", bbox_to_anchor=(.5, .0), ncol=len(classes), columnspacing=.7,
               handlelength=1, handletextpad=.3, fontsize=5.8)
    save(fig, out, "fig_predictions")
    return [f"fig_predictions: {row['tile_id']} ({name})" for name, row in picks]


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--index", default="data/v4/index.csv")
    parser.add_argument("--images-root", default="data/v1")
    parser.add_argument("--masks-root", default="data/v4")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--dnbr-dir", default="validation/dnbr_fig", help="where the example tiles' dNBR rasters go")
    parser.add_argument("--fresh-evidence", default="validation/dnbr/burn_evidence.csv", help="for the ring background")
    parser.add_argument("--old-evidence", default="validation/dnbr_old/burn_evidence.csv", help="to pick the old-window tiles")
    parser.add_argument("--results", default="results_v4")
    parser.add_argument("--model", default="segformer_b2")
    parser.add_argument("--seed", type=int, default=0, help="seed of the trained model")
    parser.add_argument("--pick-seed", type=int, default=3, help="change it to draw other example tiles")
    parser.add_argument("--evidence-tiles", nargs="+", help="tile IDs for fig_evidence")
    parser.add_argument("--prediction-tiles", nargs="+", help="tile IDs for fig_predictions")
    parser.add_argument("--only", choices=["evidence", "trends", "predictions"])
    parser.add_argument("--out", default="paper/figures")
    args = parser.parse_args()

    warnings.filterwarnings("ignore", category=FutureWarning)
    warnings.filterwarnings("ignore", message=".*shapely.*")
    set_style()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    index = pd.read_csv(args.index, low_memory=False)
    frame = index[index["split"].notna()].reset_index(drop=True)
    missing = {"dnbr_flag", "age_days", "area_ha", "own_frac"} - set(frame.columns)
    if missing:
        raise SystemExit(f"The index has no column {', '.join(sorted(missing))}; use data/v4/index.csv.")
    notes = []
    if args.only in (None, "trends"):
        draw_trends(*trends_tables(frame, args.fresh_evidence), out)
    if args.only in (None, "evidence"):
        notes += figure_evidence(frame, args, out)
    if args.only in (None, "predictions"):
        notes += figure_predictions(frame, args, out)
    if notes:
        (out / "examples_extra.md").write_text("# Example tiles\n\n" + "\n".join(f"- {note}" for note in notes) + "\n")
        print("\n".join(notes))
    print("Done. Send the PDF files; in Overleaf they go next to main.tex.")


if __name__ == "__main__":
    main()
