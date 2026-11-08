#!/usr/bin/env python3
"""Make the figures and the numbers for the burn2scar paper.

  python tools/make_paper_figures.py --out paper/figures

Reads data/v2/index.csv, the masks in data/v2, the images in data/v1 and, when they exist,
validation/cems_*.csv and data/cems. It uses no GPU and downloads nothing, and it can run while
models train. It writes fig01.pdf to fig08.pdf in --out, and paper_numbers.md next to them with
every number the manuscript takes from the data.

  python tools/make_paper_figures.py --only-spectra --out paper/figures

makes only Figure 7, the reflectance of each class, and spectra_numbers.md. It reads about 4000
tiles, takes a few minutes, and writes under 1 MB.

The figures follow the ESSD rules: one sans-serif font family, embedded; panels in one file,
labelled (a), (b); 300 dpi for raster parts; a legend inside the figure for every symbol; en dashes
in ranges; a space between number and unit; no borders on maps; colours checked for colour blindness.

The eight tiles left over from an early run have no split, and aren't part of the dataset, so
they are left out of everything here.

Swap an example with --example-tiles (six tile IDs) or --cems-examples (two sample names).
"""
import argparse
import sys
import warnings
from pathlib import Path

import geopandas as gpd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import matplotlib.ticker  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import rasterio  # noqa: E402
import yaml  # noqa: E402
from matplotlib.colors import BoundaryNorm, LinearSegmentedColormap, ListedColormap, LogNorm  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Patch, Rectangle  # noqa: E402

TOOLS = Path(__file__).resolve().parent
sys.path.insert(0, str(TOOLS))

CM = 1 / 2.54
WIDTH = 12 * CM  # The width Copernicus sets for a two-column figure.
INK, INK2, RULE = "#0b0b0b", "#52514e", "#d9d7d0"
BLUE, ORANGE, AQUA, YELLOW = "#2a78d6", "#eb6834", "#1baf7a", "#eda100"
AGREE, OUTSIDE = "#a8a69d", "#f1efe8"
COUNTRY_COLOUR = {"IT": BLUE, "ES": ORANGE, "FR": AQUA, "EL": YELLOW}
COUNTRY_NAME = {"IT": "Italy", "ES": "Spain", "FR": "France", "EL": "Greece"}
COUNTRY_LINE = {"IT": "-", "ES": (0, (4, 1.5)), "FR": (0, (1, 1.2)), "EL": (0, (4, 1.2, 1, 1.2))}  # So colour isn't the only clue.
# Burn age is ordered, so it is one hue: the more recent, the darker.
CLASS_COLOUR = {"fresh": "#7f2704", "old": "#d9480f", "older": "#f5a04a", "clear": "#f1efe8",
                "water": BLUE, "shadow": "#25272d", "cloud": "#b4c0ce", "nodata": "#000000"}
CLASS_LABEL = {"fresh": "fresh_burn", "old": "old_burn", "older": "older_burn", "clear": "clear",
               "water": "water", "shadow": "cloud shadow", "cloud": "cloud", "nodata": "no data"}
MASK_VALUE = {"clear": 0, "fresh": 2, "old": 3, "cloud": 4, "shadow": 5, "water": 6, "older": 7, "nodata": 255}
CLASSES = ["fresh", "old", "older", "clear", "water", "shadow", "cloud"]
SIZE_BINS = [5, 10, 30, 100, 500, 2001]
SIZE_LABELS = ["5-10", "10-30", "30-100", "100-500", "500-2000"]
# A thin pale line can't be seen on white, so clear is a neutral grey in line charts, not its map colour.
LINE_COLOUR = {**CLASS_COLOUR, "clear": "#8a8984"}
LINE_STYLE = {"fresh": "-", "water": (0, (4, 1.5)), "shadow": (0, (1, 1.2)),
              "old": (0, (4, 1.5)), "older": (0, (1, 1.2)), "clear": (0, (4, 1.2, 1, 1.2))}  # So colour isn't the only clue.
LOW_SUN_MONTHS = [11, 12, 1, 2]  # Water by scene month shows what these months add to the water class.


def number(value):
    """As ESSD prints numbers: 2000, but 13 895 with a thin space."""
    value = int(round(value))
    return f"{value:,}".replace(",", "\u2009") if abs(value) >= 10000 else str(value)


def number_ticks(axis):
    axis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: number(v)))


def set_style():
    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 7, "axes.titlesize": 7, "axes.labelsize": 7,
        "xtick.labelsize": 6.5, "ytick.labelsize": 6.5, "legend.fontsize": 6.5,
        "axes.spines.top": False, "axes.spines.right": False, "axes.linewidth": .5, "axes.edgecolor": "#8a8984",
        "xtick.major.width": .5, "ytick.major.width": .5, "xtick.major.size": 2.5, "ytick.major.size": 2.5,
        "xtick.color": INK2, "ytick.color": INK2, "axes.labelcolor": INK2, "text.color": INK, "axes.titlecolor": INK,
        "axes.titlelocation": "left", "legend.frameon": False, "pdf.fonttype": 42, "savefig.dpi": 300,
        "image.interpolation": "nearest",
    })


def save(fig, out, name):
    path = Path(out) / f"{name}.pdf"
    fig.savefig(path, facecolor="white")
    size = path.stat().st_size / 1e6
    print(f"Wrote {path} ({size:.2f} MB)", flush=True)
    if size > 2:
        # ESSD allows 2 MB for a PDF figure and 5 MB for other types.
        png = path.with_suffix(".png")
        fig.savefig(png, facecolor="white", dpi=300)
        print(f"OVER THE 2 MB LIMIT FOR A PDF FIGURE. Use {png} ({png.stat().st_size / 1e6:.2f} MB) instead.", flush=True)
    plt.close(fig)


def grid(ax, axis="y"):
    ax.grid(True, axis=axis, color=RULE, linewidth=.4, zorder=0)
    ax.set_axisbelow(True)


# ------------------------------------------------------------------ diagrams, no data needed

def figure_workflow(out):
    fig, ax = plt.subplots(figsize=(WIDTH, 8.6 * CM))
    ax.set(xlim=(0, 12), ylim=(0, 8.6))
    ax.axis("off")

    def block(x, y, w, h, title, text, fill):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0,rounding_size=.08", facecolor=fill, edgecolor="#8a8984", linewidth=.5))
        ax.text(x + .1, y + h - .1, title, fontsize=6.3, fontweight="bold", va="top", ha="left")
        ax.text(x + .1, y + h - .36, text, fontsize=6, va="top", ha="left", color=INK2, linespacing=1.15)

    def arrow(start, end):
        ax.add_patch(FancyArrowPatch(start, end, arrowstyle="-|>", mutation_scale=6, linewidth=.6, color="#6f6e69", shrinkA=0, shrinkB=0))

    columns = {"Sources": (0, 2.7), "Processing": (3.45, 4.85), "Released": (9.05, 2.95)}
    for title, (x, _) in columns.items():
        ax.text(x, 8.42, title, fontsize=7, fontweight="bold", va="center")

    x, w = columns["Sources"]
    sources = [("EFFIS perimeters", f"{number(25707)} fires in four\ncountries, 2020\u20132026"),
               ("Sentinel-2 L1C", "13 bands, for images\nand masks"),
               ("Sentinel-2 L2A", "scene cloud cover and\nscene classification"),
               ("OmniCloudMask", "cloud and shadow from\nthree L1C bands")]
    for i, (title, text) in enumerate(sources):
        block(x, 6.95 - i * 1.41, w, 1.23, title, text, "#f2f1ec")

    x, w = columns["Processing"]
    steps = [("1  Select fires", f"5\u20132000 ha; France south of 46° N\n{number(19443)} fires", 1.0),
             ("2  Share the budget by country", f"largest fires first\n{number(13895)} fires kept", 1.0),
             ("3  Search two windows per fire", f"5\u201360 and 120\u2013300 days after the fire date\nleast cloudy scene with burn at most\n50 % hidden; {number(26846)} tiles", 1.2),
             ("4  Build masks", "burn age from every EFFIS perimeter in the tile,\nthen water, shadow and cloud; winter water\nchecked on the fire's March–October scene", 1.2),
             ("5  Assign spatial blocks", "overlapping fires stay together\nmain, time and country splits", 1.0)]
    top = 8.18
    previous = None
    for title, text, h in steps:
        block(x, top - h, w, h, title, text, "white")
        if previous is not None:
            arrow((x + w / 2, previous), (x + w / 2, top))
        previous = top - h
        top -= h + .19

    x, w = columns["Released"]
    outputs = [("Tiles", "13-band image and mask,\nGeoTIFF, 2.56 km, 10 m"), ("Index", "one row per tile, with\nattributes and splits"), ("Reference files", "EFFIS snapshot and the\nreviewed points")]
    for i, (title, text) in enumerate(outputs):
        block(x, 6.95 - i * 1.41, w, 1.23, title, text, "#fdeee0")

    arrow((2.75, 4.95), (3.4, 4.95))
    arrow((8.35, 4.95), (9.0, 4.95))

    ax.text(0, 1.22, "Checks", fontsize=7, fontweight="bold", va="center")
    checks = [("Copernicus EMS maps", "made separately from EFFIS"), ("Reviewed points", "blind labels from the imagery"), ("Baseline models", "the labels can be learned")]
    for i, (title, text) in enumerate(checks):
        block(i * 4.1, 0, 3.8, 1.0, title, text, "#eaf2fb")
    arrow((10.5, 4.08), (10.5, 1.42))
    ax.plot([0, 12], [1.42, 1.42], color="#8a8984", linewidth=.4)
    fig.subplots_adjust(0, 0, 1, 1)
    save(fig, out, "fig01")


def figure_windows(out):
    fig, ax = plt.subplots(figsize=(WIDTH, 4.3 * CM))
    limit = 430
    ax.set(xlim=(0, limit), ylim=(0, 3.6))
    rows = {"Scene search": 2.55, "Class of a fire,\nby its age at\nthe scene date": .95}
    for label, y in rows.items():
        ax.text(-8, y + .3, label, ha="right", va="center", fontsize=6.5, color=INK2)

    def band(x0, x1, y, colour, text, ink="white"):
        ax.add_patch(Rectangle((x0, y), x1 - x0, .6, facecolor=colour, edgecolor="white", linewidth=.8))
        ax.text((x0 + x1) / 2, y + .3, text, ha="center", va="center", fontsize=6.5, color=ink)

    band(5, 60, 2.55, CLASS_COLOUR["fresh"], "fresh")
    band(120, 300, 2.55, CLASS_COLOUR["old"], "old")
    ax.text(5, 3.25, "day 5", ha="center", va="bottom", fontsize=6, color=INK2)
    band(0, 90, .95, CLASS_COLOUR["fresh"], "fresh_burn")
    band(90, 365, .95, CLASS_COLOUR["old"], "old_burn")
    band(365, limit, .95, CLASS_COLOUR["older"], "older_burn", ink=INK)
    for x in (60, 90, 120):
        ax.plot([x, x], [.55, 3.15], color="#8a8984", linewidth=.4, linestyle=(0, (2, 2)), zorder=0)
    ax.annotate("", xy=(60, 1.95), xytext=(90, 1.95), arrowprops=dict(arrowstyle="<->", linewidth=.5, color=INK2, shrinkA=0, shrinkB=0))
    ax.annotate("", xy=(90, 1.95), xytext=(120, 1.95), arrowprops=dict(arrowstyle="<->", linewidth=.5, color=INK2, shrinkA=0, shrinkB=0))
    ax.text(90, 2.16, "30-day margins", ha="center", va="center", fontsize=6, color=INK2, bbox=dict(facecolor="white", edgecolor="none", pad=.6))
    ax.set_xticks([0, 60, 90, 120, 300, 365])
    ax.set_yticks([])
    ax.spines["left"].set_visible(False)
    ax.set_xlabel("Days from the EFFIS fire date")
    fig.subplots_adjust(left=.17, right=.99, top=.97, bottom=.24)
    save(fig, out, "fig02")


# ------------------------------------------------------------------ figures from the index

def land():
    """Natural Earth land, public domain. Land only: ESSD asks authors to avoid drawing borders."""
    return gpd.read_file(TOOLS / "data" / "land_med_ne50m.geojson")


def figure_map(frame, out):
    fires = frame.drop_duplicates("fire_id")
    ground = land()
    colours = LinearSegmentedColormap.from_list("fires", ["#fbae72", "#e6550d", "#a63603", "#4a1602"])
    fig, ax = plt.subplots(figsize=(WIDTH, 5.9 * CM))

    def base(axis):
        ground.plot(ax=axis, color="#f4f3ee", edgecolor="#c9c7bf", linewidth=.25)
        axis.set_aspect(1.3)

    base(ax)
    main = fires[fires["lat"] >= 34]
    cells = ax.hexbin(main["lon"], main["lat"], gridsize=(78, 28), extent=(-10, 29, 34, 48), mincnt=1, cmap=colours,
                      linewidths=0, zorder=3, rasterized=True)
    norm = LogNorm(1, max(float(cells.get_array().max()), 10))
    cells.set_norm(norm)
    ax.set(xlim=(-10.5, 29), ylim=(34, 47.8))
    ax.set_xticks(range(-10, 30, 10))
    ax.set_yticks(range(35, 48, 5))
    ax.set_xticklabels([f"{abs(v)}° {'W' if v < 0 else 'E'}" if v else "0°" for v in range(-10, 30, 10)])
    ax.set_yticklabels([f"{v}° N" for v in range(35, 48, 5)])
    for code, lon, lat in (("ES", -7.6, 35.05), ("FR", 2.4, 46.85), ("IT", 12.6, 39.7), ("EL", 18.9, 36.4)):
        part = fires[fires["country"] == code]
        ax.text(lon, lat, f"{COUNTRY_NAME[code]}\n{number(len(part))} fires", fontsize=6.5, ha="center", va="center", zorder=5,
                bbox=dict(facecolor="white", edgecolor="none", alpha=.75, pad=1))
    bar = fig.colorbar(cells, ax=ax, fraction=.024, pad=.015, ticks=[1, 10, 100])
    bar.ax.set_yticklabels(["1", "10", "100"])
    bar.ax.minorticks_off()
    bar.set_label("Fires per cell", fontsize=6.5)
    bar.outline.set_linewidth(.3)
    bar.ax.tick_params(labelsize=6, length=2, width=.4)
    islands = fires[(fires["lat"] < 34) & (fires["lon"] < -10)]
    if len(islands):
        inset = ax.inset_axes([.27, .0, .2, .17])
        base(inset)
        inset.hexbin(islands["lon"], islands["lat"], gridsize=(14, 6), extent=(-18.5, -13, 27.5, 29.5), mincnt=1, cmap=colours, norm=norm,
                     linewidths=0, rasterized=True, zorder=3)
        inset.set(xlim=(-18.5, -13), ylim=(27.5, 29.6), xticks=[], yticks=[])
        for side in inset.spines.values():
            side.set_visible(True)
            side.set_linewidth(.4)
        inset.text(-13.2, 29.45, "Canary Islands", fontsize=5.5, ha="right", va="top", color=INK2)
    ax.text(.995, .012, "Made with Natural Earth", transform=ax.transAxes, fontsize=5.5, ha="right", va="bottom", color=INK2, zorder=6)
    fig.subplots_adjust(left=.075, right=.9, top=.985, bottom=.075)
    save(fig, out, "fig03")


def read_rgb(path, recipe, band_index):
    with rasterio.open(path) as src:
        image = src.read().astype(np.float32) / 10000
    return np.clip(np.stack([image[band_index[name]] / scale for name, scale in recipe], axis=-1), 0, 1) ** .9


def pick(frame, rule, seed):
    part = frame[rule]
    return part.sample(1, random_state=seed).iloc[0] if len(part) else None


def choose_examples(frame):
    """Six tiles: one fire at both dates, then water, a second old scar, older scars, and cloud with shadow."""
    clean = frame["frac_cloud"] < .01
    chosen = []
    fresh_it = pick(frame, (frame["country"] == "IT") & (frame["window"] == "fresh") & frame["frac_fresh"].between(.2, .55) & clean, 1)
    if fresh_it is not None:
        chosen.append(fresh_it)
        twin = frame[(frame["fire_id"] == fresh_it["fire_id"]) & (frame["window"] == "old")]
        if len(twin):
            chosen.append(twin.iloc[0])
    for rule in (((frame["country"] == "ES") & (frame["window"] == "fresh") & frame["frac_fresh"].between(.15, .6) & (frame["frac_water"] > .05) & clean),
                 ((frame["country"] == "EL") & (frame["window"] == "old") & frame["frac_old"].between(.15, .6) & clean),
                 ((frame["country"] == "FR") & (frame["window"] == "fresh") & frame["frac_fresh"].between(.1, .5) & (frame["frac_older"] > .15) & clean),
                 ((frame["window"] == "fresh") & frame["frac_fresh"].between(.1, .6) & (frame["frac_cloud"] > .1) & (frame["frac_shadow"] > .02))):
        row = pick(frame, rule, 2)
        if row is not None:
            chosen.append(row)
    seen = {row["tile_id"] for row in chosen}
    spare = frame[~frame["tile_id"].isin(seen)].sort_values("own_frac", ascending=False)
    for _, row in spare.iterrows():
        if len(chosen) >= 6:
            break
        chosen.append(row)
    return chosen[:6]


def figure_examples(frame, images_root, masks_root, out, tile_ids=None):
    from review_points import BAND, FALSE_COLOUR, TRUE_COLOUR
    rows = [frame[frame["tile_id"] == tile].iloc[0] for tile in tile_ids] if tile_ids else choose_examples(frame)
    values = sorted(MASK_VALUE.values())
    by_value = {value: name for name, value in MASK_VALUE.items()}
    colours = ListedColormap([CLASS_COLOUR[by_value[value]] for value in values])
    edges = [v - .5 for v in values] + [values[-1] + .5]
    norm = BoundaryNorm(edges, colours.N)
    fig, axes = plt.subplots(3, 6, figsize=(WIDTH, 7.7 * CM))
    any_nodata = False
    for letter, column, row in zip("abcdef", axes.T, rows):
        image_path = Path(images_root) / row["image_path"]
        with rasterio.open(Path(masks_root) / row["mask_path"]) as src:
            mask = src.read(1)
        any_nodata |= bool((mask == MASK_VALUE["nodata"]).any())
        column[0].imshow(read_rgb(image_path, TRUE_COLOUR, BAND))
        column[1].imshow(read_rgb(image_path, FALSE_COLOUR, BAND))
        column[2].imshow(mask, cmap=colours, norm=norm)
        column[0].set_title(f"({letter}) {COUNTRY_NAME.get(row['country'], row['country'])}, {row['window']}\nday {row['age_days']:.0f}", fontsize=6, linespacing=1.1, pad=2.5)
    for ax in axes.ravel():
        ax.set_xticks([])
        ax.set_yticks([])
        for side in ax.spines.values():
            side.set_visible(True)
            side.set_linewidth(.3)
    for ax, label in zip(axes[:, 0], ("True colour", "False colour", "Mask")):
        ax.set_ylabel(label, fontsize=6.5, labelpad=2)
    handles = [Patch(facecolor=CLASS_COLOUR[name], edgecolor="#8a8984", linewidth=.3, label=CLASS_LABEL[name]) for name in [*CLASSES, *(["nodata"] if any_nodata else [])]]
    fig.legend(handles=handles, loc="lower center", ncol=len(handles), columnspacing=.7, handlelength=1, handletextpad=.3, fontsize=5.8)
    fig.subplots_adjust(left=.035, right=.995, top=.925, bottom=.065, wspace=.04, hspace=.04)
    save(fig, out, "fig04")
    return [row["tile_id"] for row in rows]


def figure_composition(frame, out):
    fires = frame.drop_duplicates("fire_id")
    fig, axes = plt.subplots(2, 2, figsize=(WIDTH, 8.4 * CM))
    (year, month), (size, gap) = axes
    order = ["IT", "ES", "FR", "EL"]

    table = frame.groupby(["fire_year", "country"]).size().unstack(fill_value=0).reindex(columns=order, fill_value=0)
    bottom = np.zeros(len(table))
    for code in order:
        year.bar(table.index, table[code], bottom=bottom, width=.62, color=COUNTRY_COLOUR[code], edgecolor="white", linewidth=.5, label=COUNTRY_NAME[code], zorder=2)
        bottom += table[code].to_numpy()
    year.set_title("(a) Tiles by fire year")
    year.set_ylabel("Tiles")
    number_ticks(year.yaxis)
    year.set_xticks(table.index)
    year.tick_params(axis="x", length=0)
    year.legend(ncol=2, loc="upper left", columnspacing=.8, handlelength=1, handletextpad=.35, borderaxespad=0)
    year.set_ylim(0, bottom.max() * 1.32)
    grid(year)

    counts = fires["fire_date"].dt.month.value_counts().reindex(range(1, 13), fill_value=0)
    month.bar(counts.index, counts.values, width=.62, color=BLUE, zorder=2)
    month.set_title("(b) Fires by month of the fire date")
    month.set_ylabel("Fires")
    number_ticks(month.yaxis)
    month.set_xticks(range(1, 13))
    month.set_xticklabels(list("JFMAMJJASOND"))
    month.tick_params(axis="x", length=0)
    grid(month)

    for code in order:
        part = np.sort(fires.loc[fires["country"] == code, "area_ha"].to_numpy())
        if len(part):
            size.plot(part, np.arange(1, len(part) + 1) / len(part), color=COUNTRY_COLOUR[code], linewidth=1.1, linestyle=COUNTRY_LINE[code], label=COUNTRY_NAME[code])
    size.set_xscale("log")
    size.set_xlim(5, 2000)
    size.set_ylim(0, 1)
    size.set_xticks([5, 10, 30, 100, 300, 1000])
    size.set_xticklabels(["5", "10", "30", "100", "300", "1000"])
    size.minorticks_off()
    size.set_title("(c) Size of kept fires")
    size.set_xlabel("EFFIS area (ha)")
    size.set_ylabel("Share of fires at or below")
    size.legend(loc="lower right", handlelength=2.4, handletextpad=.4, borderaxespad=.2)
    grid(size, "both")

    bins = np.arange(0, 305, 5)
    for window, name in (("fresh", "Fresh scene"), ("old", "Old scene")):
        gap.hist(frame.loc[frame["window"] == window, "age_days"], bins=bins, color=CLASS_COLOUR[window], label=name, zorder=2, linewidth=0)
    gap.set_title("(d) Days from fire date to scene")
    gap.set_xlabel("Days")
    gap.set_ylabel("Tiles")
    number_ticks(gap.yaxis)
    gap.legend(loc="upper right", handlelength=1, handletextpad=.4, borderaxespad=0)
    grid(gap)
    fig.subplots_adjust(left=.115, right=.985, top=.95, bottom=.105, wspace=.32, hspace=.45)
    save(fig, out, "fig05")


def class_shares(part):
    valid = (1 - part["frac_nodata"]).sum()
    return {name: part[f"frac_{name}"].sum() / valid for name in CLASSES}


def figure_classes(frame, out):
    rows = [("All tiles", frame)]
    for code in ("IT", "ES", "FR", "EL"):
        for window in ("fresh", "old"):
            part = frame[(frame["country"] == code) & (frame["window"] == window)]
            if len(part):
                rows.append((f"{COUNTRY_NAME[code]}, {window} scene", part))
    fig, ax = plt.subplots(figsize=(WIDTH, 5.6 * CM))
    positions = np.arange(len(rows))[::-1]
    for y, (_, part) in zip(positions, rows):
        left = 0
        shares = class_shares(part)
        for name in CLASSES:
            ax.barh(y, shares[name] * 100, left=left, height=.66, color=CLASS_COLOUR[name], edgecolor="white", linewidth=.4)
            left += shares[name] * 100
    ax.set_yticks(positions)
    ax.set_yticklabels([label for label, _ in rows])
    ax.tick_params(axis="y", length=0)
    ax.set_xlim(0, 100)
    ax.set_xlabel("Share of valid pixels (%)")
    ax.spines["left"].set_visible(False)
    handles = [Patch(facecolor=CLASS_COLOUR[name], edgecolor="#8a8984", linewidth=.3, label=CLASS_LABEL[name]) for name in CLASSES]
    fig.legend(handles=handles, loc="upper center", ncol=7, columnspacing=.9, handlelength=1, handletextpad=.35, fontsize=6)
    fig.subplots_adjust(left=.215, right=.975, top=.915, bottom=.145)
    save(fig, out, "fig06")


# ------------------------------------------------------------------ reflectance by class

def sample_spectra(frame, images_root, masks_root, tiles_per_class, pixels_per_tile, seed):
    """Reflectance of sampled pixels per class. Each class draws its own random tiles from those that hold it,
    and takes a limited number of pixels from each, so no single tile decides a class's curve."""
    rng = np.random.default_rng(seed)
    wanted = {}
    for name in CLASSES:
        holding = frame.index[frame[f"frac_{name}"] > 0].to_numpy()
        for position in rng.choice(holding, min(tiles_per_class, len(holding)), replace=False):
            wanted.setdefault(int(position), []).append(name)
    parts = {name: [] for name in CLASSES}
    for done, (position, names) in enumerate(sorted(wanted.items()), 1):
        row = frame.loc[position]
        with rasterio.open(Path(masks_root) / row["mask_path"]) as src:
            mask = src.read(1)
        with rasterio.open(Path(images_root) / row["image_path"]) as src:
            image = src.read().astype(np.float32) / 10000
        stored = image.max(axis=0) > 0  # 0 in every band is no data. A single band can be 0 for real, as negative values are clamped.
        for name in names:
            rows, columns = np.nonzero((mask == MASK_VALUE[name]) & stored)
            if len(rows):
                keep = rng.choice(len(rows), min(pixels_per_tile, len(rows)), replace=False)
                parts[name].append(image[:, rows[keep], columns[keep]].T)
        if done % 500 == 0:
            print(f"  read {done:,} of {len(wanted):,} tiles", flush=True)
    return {name: np.concatenate(part) for name, part in parts.items() if part}, {name: len(part) for name, part in parts.items() if part}, len(wanted)


def spectra_tables(title, pixels, tiles, bands):
    medians, ranges = [], []
    for name in CLASSES:
        if name in pixels:
            low, mid, high = np.percentile(pixels[name], [25, 50, 75], axis=0)
            medians.append({"Class": CLASS_LABEL[name], "Tiles": f"{tiles[name]:,}", "Pixels": f"{len(pixels[name]):,}", **{b: f"{v:.3f}" for b, v in zip(bands, mid)}})
            ranges.append({"Class": CLASS_LABEL[name], **{b: f"{v:.3f}-{w:.3f}" for b, v, w in zip(bands, low, high)}})
    return [f"## {title}", "", "Median", "", md_table(pd.DataFrame(medians)), "", "25th to 75th percentile", "", md_table(pd.DataFrame(ranges)), ""]


def water_check(title, pixels, band_index):
    """How much of the water class is too bright for open water. Terrain shadow can't be told from these numbers alone."""
    green, swir = pixels[:, band_index["B03"]], pixels[:, band_index["B11"]]
    bright = green > .2
    snow = bright & ((green - swir) / np.maximum(green + swir, 1e-6) > .4)
    return [f"- {title}: {100 * bright.mean():.1f} % of water pixels have B03 above 0.20. Snow-like, also NDSI above 0.4: {100 * snow.mean():.1f} %"]


def figure_spectra(frame, images_root, masks_root, out, tiles_per_class=1000, pixels_per_tile=200, seed=20261105):
    from review_points import BAND
    bands = sorted(BAND, key=BAND.get)
    low_sun = pd.to_datetime(frame["scene_time"], format="mixed").dt.month.isin(LOW_SUN_MONTHS)
    pixels, tiles, read = sample_spectra(frame[~low_sun], images_root, masks_root, tiles_per_class, pixels_per_tile, seed)
    x = np.arange(len(bands))
    fig, (alike, ages) = plt.subplots(2, 1, figsize=(WIDTH, 9.6 * CM), sharex=True, sharey=True)

    def draw(ax, name):
        ax.plot(x, np.percentile(pixels[name], 50, axis=0), color=LINE_COLOUR[name], linestyle=LINE_STYLE[name], linewidth=1.1,
                marker="o", markersize=2, label=CLASS_LABEL[name], zorder=3)

    highest = 0
    for name in ("fresh", "water", "shadow"):
        if name in pixels:
            low, high = np.percentile(pixels[name], [25, 75], axis=0)
            alike.fill_between(x, low, high, color=LINE_COLOUR[name], alpha=.16, linewidth=0, zorder=2)
            highest = max(highest, high.max())
            draw(alike, name)
    for name in ("fresh", "old", "older", "clear"):
        if name in pixels:
            highest = max(highest, np.percentile(pixels[name], 50, axis=0).max())
            draw(ages, name)
    alike.set_title("(a) Median and quartiles")
    ages.set_title("(b) Median")
    for ax in (alike, ages):
        ax.set_ylabel("Top-of-atmosphere reflectance")
        ax.set_ylim(0, highest * 1.04)
        ax.set_xlim(-.3, len(bands) - .7)
        grid(ax)
        handles, labels = ax.get_legend_handles_labels()
        ax.legend(handles, labels, loc="lower right", bbox_to_anchor=(1, 1), ncol=len(labels), borderaxespad=0, handlelength=2.6, handletextpad=.4, columnspacing=1)
    ages.set_xticks(x)
    ages.set_xticklabels(bands)
    ages.set_xlabel("Sentinel-2 band, in order of wavelength")
    fig.subplots_adjust(left=.095, right=.985, top=.95, bottom=.095, hspace=.3)
    save(fig, out, "fig07")

    months = "March to October"
    lines = ["# Reflectance by mask class", "",
             "Top-of-atmosphere reflectance of the L1C images, by mask class. Each class draws up to "
             f"{tiles_per_class:,} random tiles that hold it, and up to {pixels_per_tile} pixels from each. Random seed {seed}. Not area-weighted.", "",
             f"Figure 7 uses scenes from {months}: {read:,} tiles read. The class cloud isn't drawn, but it is in the tables.", ""]
    lines += spectra_tables(f"Scenes from {months}", pixels, tiles, bands)
    checks = []
    if "water" in pixels:
        checks += water_check(months, pixels["water"], BAND)
    if low_sun.any():
        cold, cold_tiles, cold_read = sample_spectra(frame[low_sun], images_root, masks_root, tiles_per_class // 2, pixels_per_tile, seed + 1)
        lines += spectra_tables(f"Scenes from November to February: {cold_read:,} tiles read", cold, cold_tiles, bands)
        if "water" in cold:
            checks += water_check("November to February", cold["water"], BAND)
    lines += ["## Is the water class all water?", "", *checks, ""]
    target = Path(out) / "spectra_numbers.md"
    target.write_text("\n".join(lines))
    print(f"Wrote {target}", flush=True)


# ------------------------------------------------------------------ Copernicus EMS

def f1_of(frame):
    return 2 * frame["tp"] / (2 * frame["tp"] + frame["fp"] + frame["fn"])


def figure_cems(args, out, chosen=None):
    import compare_cems as cc
    from burn_labels import load_perimeters, to_raster
    frame = pd.read_csv(Path(args.validation) / "cems_perimeters.csv")
    frame["burned_ha"] = (frame["tp"] + frame["fn"]) * .01
    frame["f1"] = f1_of(frame)
    frame["precision"] = frame["tp"] / (frame["tp"] + frame["fp"]).replace(0, np.nan)
    frame["recall"] = frame["tp"] / (frame["tp"] + frame["fn"])
    fig = plt.figure(figsize=(WIDTH, 9.4 * CM))
    spec = fig.add_gridspec(2, 2, height_ratios=[1, 1.1], left=.1, right=.985, top=.95, bottom=.065, wspace=.27, hspace=.42)
    scatter, by_size = fig.add_subplot(spec[0, 0]), fig.add_subplot(spec[0, 1])
    sizes = 4 + 30 * np.sqrt(frame["burned_ha"] / frame["burned_ha"].max())
    touched = frame["touching_fires"] > 0
    scatter.scatter(frame.loc[touched, "recall"] * 100, frame.loc[touched, "precision"] * 100, s=sizes[touched], color=BLUE, alpha=.7, edgecolor="white", linewidth=.3, zorder=3)
    scatter.set(xlim=(0, 102), ylim=(0, 102), xlabel="Recall (%)", ylabel="Precision (%)")
    scatter.set_title("(a) Precision and recall per area")
    grid(scatter, "both")
    keys = [v for v in (100, 1000, 10000, 100000) if v <= frame["burned_ha"].max()] or [round(frame["burned_ha"].max())]
    scatter.legend(handles=[Line2D([], [], linestyle="", marker="o", color=BLUE, alpha=.7, markeredgecolor="white", markeredgewidth=.3,
                                   markersize=np.sqrt(4 + 30 * np.sqrt(v / frame["burned_ha"].max())), label=f"{number(v)} ha") for v in keys],
                   title="CEMS burned area", title_fontsize=6, loc="lower left", handletextpad=.2, borderaxespad=.3, labelspacing=.3, fontsize=6)
    by_size.scatter(frame["burned_ha"], frame["f1"] * 100, s=9, color=BLUE, alpha=.7, edgecolor="white", linewidth=.3, zorder=3)
    by_size.set_xscale("log")
    by_size.set(ylim=(0, 102), xlabel="CEMS burned area (ha)", ylabel="F1 (%)")
    number_ticks(by_size.xaxis)
    by_size.set_title("(b) F1 by burned area")
    grid(by_size, "both")

    cfg = yaml.safe_load(Path(args.config).read_text())
    table = cc.load_area_table(Path(args.cems_root) / "satelliteData.csv")
    samples = cc.load_samples(args.cems_root, table).set_index("sample", drop=False)
    perimeters = load_perimeters(cfg)
    big = frame[frame["burned_ha"] >= 200].dropna(subset=["f1"])
    if big.empty:
        big = frame.dropna(subset=["f1"])
    pooled = 2 * frame["tp"].sum() / (2 * frame["tp"].sum() + frame["fp"].sum() + frame["fn"].sum())
    if chosen:
        picks = chosen
    else:
        picks = [big.iloc[(big["f1"] - pooled).abs().argmin()]["sample"], big.sort_values("f1").iloc[0]["sample"]]
    before, after = cc.WINDOWS["main"]
    labels = ("c", "d")
    # Agreement is neutral, so the two kinds of disagreement stand out, also with colour blindness.
    colours = ListedColormap(["white", AGREE, CLASS_COLOUR["old"], BLUE, OUTSIDE])
    any_outside = False
    for place, name, letter in zip((spec[1, 0], spec[1, 1]), picks, labels):
        ax = fig.add_subplot(place)
        sample = samples.loc[name]
        row = frame[frame["sample"] == name].iloc[0]
        burned, valid, transform, crs = cc.read_reference(sample)
        nearby = perimeters.iloc[perimeters.sindex.query(sample["geometry"], predicate="intersects")]
        days = (nearby["fire_date"].dt.normalize() - sample["date"]).dt.days.to_numpy()
        geoms = nearby.geometry.to_crs(crs).to_numpy()[(days >= -before) & (days <= after)]
        effis = to_raster(cc.touching(geoms, burned & valid, transform), burned.shape, transform)
        picture = np.zeros(burned.shape, np.uint8)
        picture[effis & burned] = 1
        picture[effis & ~burned] = 2
        picture[~effis & burned] = 3
        picture[~valid] = 4
        any_outside |= bool((~valid).any())
        ax.imshow(picture, cmap=colours, vmin=-.5, vmax=4.5, interpolation="nearest")
        ax.set_xticks([])
        ax.set_yticks([])
        for side in ax.spines.values():
            side.set_visible(True)
            side.set_linewidth(.3)
        ax.set_title(f"({letter}) {name}, F1 {100 * row['f1']:.0f} %", fontsize=6.5)
    handles = [Patch(facecolor=AGREE, label="Both"), Patch(facecolor=CLASS_COLOUR["old"], label="EFFIS only"), Patch(facecolor=BLUE, label="CEMS only")]
    if any_outside:
        handles.append(Patch(facecolor=OUTSIDE, edgecolor="#8a8984", linewidth=.3, label="Outside the image"))
    fig.legend(handles=handles, loc="center", bbox_to_anchor=(.5, .025), ncol=len(handles), handlelength=1, handletextpad=.35, columnspacing=1)
    save(fig, out, "fig08")

    # How EFFIS dates sit against the CEMS date, for the fires that touch the CEMS burn.
    offsets = []
    for name in frame["sample"]:
        sample = samples.loc[name]
        burned, valid, transform, crs = cc.read_reference(sample)
        nearby = perimeters.iloc[perimeters.sindex.query(sample["geometry"], predicate="intersects")]
        days = (nearby["fire_date"].dt.normalize() - sample["date"]).dt.days.to_numpy()
        keep = (days >= -before) & (days <= after)
        for geom, day in zip(nearby.geometry.to_crs(crs).to_numpy()[keep], days[keep]):
            if to_raster([geom], burned.shape, transform)[burned & valid].any():
                offsets.append(int(day))
    return picks, offsets


# ------------------------------------------------------------------ numbers

def flag(series):
    """True or False from a CSV column, which can hold text if another row was empty."""
    return series.astype(str).str.lower().eq("true")


def md_table(frame):
    head = "| " + " | ".join(map(str, frame.columns)) + " |"
    rule = "|" + "---|" * len(frame.columns)
    body = ["| " + " | ".join(str(v) for v in row) + " |" for row in frame.itertuples(index=False)]
    return "\n".join([head, rule, *body])


def split_table(frame, column):
    rows = []
    for name, part in frame.groupby(column):
        share = class_shares(part)
        rows.append({column: name, "Tiles": f"{len(part):,}", "Fires": f"{part['fire_id'].nunique():,}", "IT": int((part["country"] == "IT").sum()),
                     "ES": int((part["country"] == "ES").sum()), "FR": int((part["country"] == "FR").sum()), "EL": int((part["country"] == "EL").sum()),
                     "Fresh tiles": int((part["window"] == "fresh").sum()), "Old tiles": int((part["window"] == "old").sum()),
                     "Fresh %": f"{100 * share['fresh']:.2f}", "Old %": f"{100 * share['old']:.2f}", "Older %": f"{100 * share['older']:.2f}"})
    return md_table(pd.DataFrame(rows))


def write_numbers(frame, args, out, examples, cems_examples, cems_offsets, all_rows):
    from build_index import covered_area
    fires = frame.drop_duplicates("fire_id").copy()
    two = frame.groupby("fire_id")["window"].nunique()
    lines = ["# Numbers for the burn2scar paper", "",
             f"Index rows: {all_rows:,}. Left out, no split: {all_rows - len(frame):,}.", "", "## Tiles and fires", "",
             f"- Tiles: {len(frame):,}", f"- Fires: {len(fires):,}", f"- Fires with two scenes: {int((two == 2).sum()):,}", f"- Fires with one scene: {int((two == 1).sum()):,}",
             f"- Fresh tiles: {int((frame['window'] == 'fresh').sum()):,}", f"- Old tiles: {int((frame['window'] == 'old').sum()):,}"]
    union, total = covered_area(fires)
    lines += [f"- Land under at least one tile: {union:,.0f} km2. Tiles add up to {total:,.0f} km2", f"- Overlap groups: {fires['group'].nunique():,}. Largest: {fires['group'].value_counts().iloc[0]:,} fires",
              f"- Spatial blocks: {frame['block'].nunique():,}", f"- Tiles where the fire's own scar is over 50 % hidden: {int((~flag(frame['scene_rule_pass'])).sum()):,}",
              f"- Tiles at least 95 % covered by their own fire: {int(flag(frame['tile_filled']).sum()):,}", ""]
    rows = []
    for code in ("IT", "ES", "FR", "EL"):
        part, kept = frame[frame["country"] == code], fires[fires["country"] == code]
        rows.append({"Country": code, "Fires": f"{len(kept):,}", "Tiles": f"{len(part):,}", "Fresh": f"{int((part['window'] == 'fresh').sum()):,}", "Old": f"{int((part['window'] == 'old').sum()):,}",
                     "Smallest ha": f"{kept['area_ha'].min():.1f}", "Median ha": f"{kept['area_ha'].median():.1f}", "Largest ha": f"{kept['area_ha'].max():.0f}"})
    lines += [md_table(pd.DataFrame(rows)), "", "## Fire size classes", ""]
    fires["size"] = pd.cut(fires["area_ha"], SIZE_BINS, right=False, labels=SIZE_LABELS)
    lines += [md_table(fires.groupby("size", observed=False).size().rename("Fires").reset_index()), "", "## Fire year and month", ""]
    lines += [md_table(fires["fire_year"].value_counts().sort_index().rename("Fires").rename_axis("Year").reset_index()), ""]
    lines += [md_table((fires["fire_date"].dt.month.value_counts(normalize=True).sort_index() * 100).round(1).rename("Share %").rename_axis("Month").reset_index()), "", "## Days from fire date to scene", ""]
    gap = frame.groupby("window")["age_days"].describe()[["min", "25%", "50%", "75%", "max"]].round(1).reset_index()
    lines += [md_table(gap), "", "## Class shares, % of valid pixels", "", f"No data: {100 * frame['frac_nodata'].mean():.2f} % of all pixels", ""]
    groups = [("All", frame)] + [(f"Window {w}", frame[frame["window"] == w]) for w in ("fresh", "old")] + [(c, frame[frame["country"] == c]) for c in ("IT", "ES", "FR", "EL")]
    lines += [md_table(pd.DataFrame([{"Group": name, **{CLASS_LABEL[k]: f"{100 * v:.2f}" for k, v in class_shares(part).items()}} for name, part in groups])), ""]
    scene_year = pd.to_datetime(frame["scene_time"], format="mixed").dt.year
    lines += ["## older_burn by scene year, % of valid pixels. The export starts in January 2020, so early scenes can hold little", "",
              md_table(pd.DataFrame([{"Scene year": year, "Tiles": f"{len(part):,}", "older_burn %": f"{100 * class_shares(part)['older']:.2f}"} for year, part in frame.groupby(scene_year)])), ""]
    # Terrain shadow under a low winter sun can read as water, so look at water by month.
    scene_month = pd.to_datetime(frame["scene_time"], format="mixed").dt.month
    rows = []
    for month, part in frame.groupby(scene_month):
        row = {"Scene month": month, "Tiles": f"{len(part):,}", "All %": f"{100 * class_shares(part)['water']:.2f}"}
        for code in ("IT", "ES", "FR", "EL"):
            country = part[part["country"] == code]
            row[f"{code} %"] = f"{100 * class_shares(country)['water']:.2f}" if len(country) else ""
        rows.append(row)
    lines += ["## Water by scene month, % of valid pixels", "", md_table(pd.DataFrame(rows)), ""]
    # Scars from before 2020 can't be labelled. Later tiles show how much burn from other fires a tile holds in a full year of history.
    scene = pd.to_datetime(frame["scene_time"], format="mixed")
    other = (frame["frac_fresh"] + frame["frac_old"] - frame["own_frac"]).clip(lower=0)
    full_history = scene >= "2021-01-01"
    rate = other[full_history].mean()
    start = pd.Timestamp(args.effis_start)
    lacking = (((start + pd.Timedelta(days=365)) - scene).dt.total_seconds() / 86400 / 365).clip(0, 1)
    hidden = (rate * lacking).sum() / (frame["frac_fresh"] + frame["frac_old"]).sum()
    lines += ["## Burn hidden because the export starts in 2020", "",
              f"- Tiles with a scene before 2021: {int((~full_history).sum()):,} ({100 * (~full_history).mean():.1f} %)",
              f"- Burn from other fires per tile with a full year of history: {100 * rate:.2f} % of the tile",
              f"- Estimated burn hidden in early tiles: {100 * hidden:.2f} % of all fresh and old burn pixels", ""]
    if "land_cover" in fires:
        lines += ["## Dominant land cover of the kept fires, % of fires", "",
                  md_table((pd.crosstab(fires["country"], fires["land_cover"], normalize="index") * 100).round(1).reset_index()), ""]
    if "map_source" in fires:
        lines += [f"Map source of the kept fires: {fires['map_source'].value_counts().to_dict()}", ""]
    if "fire_duration_days" in fires:
        lines += [f"Fire duration in days, median and 90th percentile: {fires['fire_duration_days'].median():.2f}, {fires['fire_duration_days'].quantile(.9):.2f}", ""]
    # How much of each kept fire lies near its outline: Table 8 of the manuscript.
    from burn_labels import load_perimeters
    cfg = yaml.safe_load(Path(args.config).read_text())
    kept = load_perimeters(cfg)
    kept = kept[kept["fire_id"].isin(set(fires["fire_id"]))]
    metric = kept.geometry.to_crs(3035)
    area = metric.area.to_numpy()
    edges = pd.DataFrame({"area_ha": kept["area_ha"].to_numpy()})
    for distance in (10, 20):
        inner = metric.buffer(-distance).area.to_numpy()
        edges[f"Within {distance} m %"] = 100 * (1 - inner / area)
        edges[f"IoU {distance} m"] = inner / metric.buffer(distance).area.to_numpy()
    edges["Size ha"] = pd.cut(edges["area_ha"], SIZE_BINS, right=False, labels=SIZE_LABELS)
    table = edges.groupby("Size ha", observed=False).agg(**{"Fires": ("area_ha", "size"), "Within 10 m %": ("Within 10 m %", "median"), "Within 20 m %": ("Within 20 m %", "median"),
                                                          "IoU 10 m": ("IoU 10 m", "median"), "IoU 20 m": ("IoU 20 m", "median")}).reset_index()
    lines += ["## Outline position, kept fires, median per fire", "", md_table(table.round({"Within 10 m %": 1, "Within 20 m %": 1, "IoU 10 m": 2, "IoU 20 m": 2})), ""]
    lines += ["## Splits", ""]
    for column in ["split", "split_time", "split_loco_IT", "split_loco_ES", "split_loco_FR", "split_loco_EL"]:
        lines += [f"### {column}", "", split_table(frame, column), ""]
    lines += ["## Example tiles in Figure 4", "", *[f"- {tile}" for tile in examples], ""]
    path = Path(args.validation) / "cems_perimeters.csv"
    if path.exists():
        cems = pd.read_csv(path)
        cems["burned_ha"] = (cems["tp"] + cems["fn"]) * .01
        cems["f1"] = f1_of(cems)
        lines += ["## Copernicus EMS, one row per area", "",
                  f"- Areas: {len(cems):,}. Activations: {cems['activation'].nunique():,}", f"- CEMS burned ha per area: median {cems['burned_ha'].median():,.0f}, 10th percentile {cems['burned_ha'].quantile(.1):,.0f}, 90th percentile {cems['burned_ha'].quantile(.9):,.0f}",
                  f"- F1 per area: median {100 * cems['f1'].median():.1f}, 10th percentile {100 * cems['f1'].quantile(.1):.1f}, 90th percentile {100 * cems['f1'].quantile(.9):.1f}",
                  f"- Fires in the box and window: {int(cems['effis_fires'].sum()):,}. Touching the CEMS burn: {int(cems['touching_fires'].sum()):,}", f"- Examples in Figure 8: {cems_examples}", ""]
        missing = cems[cems["touching_fires"] == 0]
        lines += [f"- Areas with no touching fire: {', '.join(f'{r.sample} ({r.burned_ha:.0f} ha)' for r in missing.itertuples())}", ""]
        if cems_offsets:
            offsets = np.array(cems_offsets)
            lines += [f"- EFFIS date minus CEMS date, touching fires, days: median {np.median(offsets):.0f}, 10th percentile {np.percentile(offsets, 10):.0f}, 90th percentile {np.percentile(offsets, 90):.0f}. "
                      f"Within 30 days: {100 * (np.abs(offsets) <= 30).mean():.1f} % of {len(offsets):,} fires", ""]
    path = Path(args.validation) / "cems_tiles.csv"
    if path.exists():
        tiles = pd.read_csv(path).merge(frame[["tile_id", "area_ha", "fire_id"]], on="tile_id", how="left")
        tiles["size"] = pd.cut(tiles["area_ha"], [5, 30, 100, 500, 2001], right=False, labels=["5-30", "30-100", "100-500", "500-2000"])
        rows = []
        for name, part in [("All", tiles)] + list(tiles.groupby("size", observed=False)):
            tp, fp, fn = part[["tp", "fp", "fn"]].sum()
            rows.append({"Fire size ha": name, "Pairs": len(part), "Fires": part["fire_id"].nunique(), "Precision": f"{100 * tp / max(tp + fp, 1):.1f}", "Recall": f"{100 * tp / max(tp + fn, 1):.1f}", "F1": f"{200 * tp / max(2 * tp + fp + fn, 1):.1f}"})
        lines += ["## Copernicus EMS, tile pairs by size of the tile's own fire", "", md_table(pd.DataFrame(rows)), ""]
    build = Path("requirements.build_env.txt")
    if build.exists():
        wanted = [line for line in build.read_text().splitlines() if any(line.lower().startswith(name) for name in ("omnicloudmask", "sentinelhub", "eo-learn", "eolearn", "rasterio"))]
        lines += ["## Packages in the environment that built the masks", "", *[f"- {line}" for line in wanted], ""]
    target = Path(out) / "paper_numbers.md"
    target.write_text("\n".join(lines))
    print(f"Wrote {target}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--index", default="data/v2/index.csv")
    parser.add_argument("--masks-root", default="data/v2")
    parser.add_argument("--images-root", default="data/v1")
    parser.add_argument("--cems-root", default="data/cems")
    parser.add_argument("--validation", default="validation")
    parser.add_argument("--out", default="paper/figures")
    parser.add_argument("--effis-start", default="2020-01-01", help="first date in the EFFIS export")
    parser.add_argument("--only-diagrams", action="store_true", help="only the figures that need no data")
    parser.add_argument("--only-spectra", action="store_true", help="only Figure 7, the reflectance of each class")
    parser.add_argument("--example-tiles", nargs=6, help="six tile IDs for Figure 4")
    parser.add_argument("--cems-examples", nargs=2, help="two CEMS sample names for Figure 8")
    args = parser.parse_args()

    warnings.filterwarnings("ignore", category=FutureWarning)
    warnings.filterwarnings("ignore", message=".*shapely.*")
    set_style()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    if not args.only_spectra:
        figure_workflow(out)
        figure_windows(out)
    if args.only_diagrams:
        return
    index = pd.read_csv(args.index, low_memory=False)
    frame = index[index["split"].notna()].reset_index(drop=True)
    frame["fire_date"] = pd.to_datetime(frame["fire_date"], format="mixed")
    frame["fire_year"] = frame["fire_year"].astype(int)
    if args.only_spectra:
        figure_spectra(frame, args.images_root, args.masks_root, out)
        print("All done. Send fig07.pdf and spectra_numbers.md.")
        return
    figure_map(frame, out)
    examples = figure_examples(frame, args.images_root, args.masks_root, out, args.example_tiles)
    figure_composition(frame, out)
    figure_classes(frame, out)
    figure_spectra(frame, args.images_root, args.masks_root, out)
    cems_examples, cems_offsets = None, None
    if (Path(args.validation) / "cems_perimeters.csv").exists() and Path(args.cems_root).exists():
        cems_examples, cems_offsets = figure_cems(args, out, args.cems_examples)
    else:
        print("No Copernicus EMS results yet, so no Figure 8.", flush=True)
    write_numbers(frame, args, out, examples, cems_examples, cems_offsets, len(index))
    print("All done. Send paper_numbers.md and the PDF files. In Overleaf they go next to main.tex, with no subfolder.")


if __name__ == "__main__":
    main()
