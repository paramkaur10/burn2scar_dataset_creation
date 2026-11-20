#!/usr/bin/env python3
"""The figure of the evaluation additions for the 5.0 paper, in the style of make_extra_figures.py. The CPU is enough.

  python tools/make_v5_figures.py --out paper/figures

fig_additions.pdf   SegFormer-B2 (spatial split, seed 0) on five tiles of the additions: a burn-free tile, the edge of a
                    fire above 2000 ha, scenes 61-90 and 91-120 days after a fire, and a first look 0-5 days after one.
                    The tile IDs used go to examples_v5.md next to the figure; swap them with --tiles (five IDs, in that
                    order, from data/additions/v4/additions_index.csv).
"""
import argparse
import os
import sys
from argparse import Namespace
from pathlib import Path

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")   # the GPUs stay free

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import rasterio  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402

TOOLS = Path(__file__).resolve().parent
sys.path.insert(0, str(TOOLS))
import make_extra_figures as mef  # noqa: E402
from make_paper_figures import (CLASS_COLOUR, CLASS_LABEL, CLASSES, CM, WIDTH, draw_outline, frame_panel,  # noqa: E402
                                mask_colours, own_perimeters, read_rgb, save, set_style)
from review_points import BAND, FALSE_COLOUR  # noqa: E402


def tidy(frame, cloud=.01):
    return (frame["frac_cloud"] < cloud) & (frame["frac_shadow"] < cloud) & (frame["frac_nodata"] < .01)


def pick(index, seed):
    """One tile per part: a clean one where there is, else the least obscured of the part."""
    sub = lambda name: index[index["subset"] == name]
    neg, large, boundary, first = sub("negative"), sub("large"), sub("boundary"), sub("firstlook")
    early, late = boundary[boundary["age_days"].between(61, 90)], boundary[boundary["age_days"].between(90.01, 121)]
    rules = [("burn-free", neg, tidy(neg)),
             ("large-fire edge", large, tidy(large, .02) & large["edge_burn_frac"].between(.35, .65)),
             ("boundary, 61\u201390 days", early, tidy(early) & (early["frac_fresh"] >= .05)),
             ("boundary, 91\u2013120 days", late, tidy(late) & (late["frac_old"] >= .05)),
             ("first look", first, first["frac_fresh"] >= .05)]
    picks = []
    for name, part, strict in rules:
        if strict.any():
            picks.append((name, part[strict].sample(1, random_state=seed).iloc[0]))
        elif len(part):
            hidden = part["frac_cloud"] + part["frac_shadow"] + part["frac_nodata"]
            picks.append((name, part.loc[hidden.idxmin()]))
        else:
            print(f"No tile for {name}", flush=True)
    return picks


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", default="paper/figures")
    parser.add_argument("--additions", default="data/additions")
    parser.add_argument("--results", default="results_v4")
    parser.add_argument("--model", default="segformer_b2")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--tiles", nargs="*", default=[])
    parser.add_argument("--pick-seed", type=int, default=5)
    args = parser.parse_args()
    set_style()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    root = Path(args.additions)
    index = pd.read_csv(root / "v4" / "additions_index.csv", low_memory=False)
    if args.tiles:
        names = ["burn-free", "large-fire edge", "boundary, 61–90 days", "boundary, 91–120 days", "first look"]
        picks = [(name, index[index["tile_id"] == tile].iloc[0]) for name, tile in zip(names, args.tiles)]
    else:
        picks = pick(index, args.pick_seed)
    predicted = mef.predict([row for _, row in picks], Namespace(results=args.results, model=args.model, seed=args.seed,
                                                                 images_root=root / "raw"))
    fires = [row["fire_id"] for _, row in picks if isinstance(row["fire_id"], str) and row["fire_id"]]
    outlines, outline_crs = own_perimeters(args.config, fires) if fires else ({}, None)
    mask_cmap, mask_norm = mask_colours()
    fig, axes = plt.subplots(3, len(picks), figsize=(WIDTH, 8.6 * CM), squeeze=False)
    for letter, column, (name, row) in zip("abcdefgh", axes.T, picks):
        with rasterio.open(root / "v4" / row["mask_path"]) as src:
            mask, tile_crs, transform = src.read(1), src.crs, src.transform
        guess = predicted[row["tile_id"]]
        top, left, size = mef.window_of(row, mask.shape)
        crop = (slice(top, top + size), slice(left, left + size))
        column[0].imshow(read_rgb(root / "raw" / row["image_path"], FALSE_COLOUR, BAND)[crop])
        column[1].imshow(mask[crop], cmap=mask_cmap, norm=mask_norm)
        column[2].imshow(guess[crop], cmap=mask_cmap, norm=mask_norm)
        if row["fire_id"] in outlines:
            draw_outline(column[0], outlines[row["fire_id"]], outline_crs, tile_crs, transform, offset=(top, left))
        for ax in column:
            ax.set_xlim(-.5, size - .5)
            ax.set_ylim(size - .5, -.5)
            frame_panel(ax)
        mef.add_scale(column[0], size)
        valid = mask != 255
        burn = np.isin(mask, [2, 3])
        if row["subset"] == "negative":
            score = f"fresh: {100 * ((guess == 2) & valid).sum() / max(valid.sum(), 1):.1f} % of tile"
        elif row["subset"] == "boundary" and burn.any():
            score = f"fresh: {100 * ((guess == 2) & burn).sum() / burn.sum():.0f} % of burn"
        else:
            f1 = mef.fresh_f1(mask, guess)
            score = f"fresh F1 {f1:.2f}" if pd.notna(f1) else "no fresh_burn"
        day = f", day {row['age_days']:.0f}" if pd.notna(row.get("age_days")) and row["subset"] in ("boundary", "firstlook") else ""
        column[0].set_title(f"({letter}) {name.split(',')[0]}{day}\n{score}", fontsize=5.8, linespacing=1.1, pad=2.5)
    for ax, label in zip(axes[:, 0], ("False colour", "Labels", "SegFormer-B2")):
        ax.set_ylabel(label, fontsize=6.5, labelpad=2)
    fig.subplots_adjust(left=.035, right=.995, top=.91, bottom=.1, wspace=.05, hspace=.05)
    classes = [Patch(facecolor=CLASS_COLOUR[c], edgecolor="#8a8984", linewidth=.3, label=CLASS_LABEL[c]) for c in CLASSES]
    fig.legend(handles=classes, loc="lower center", bbox_to_anchor=(.5, .0), ncol=len(classes), columnspacing=.7,
               handlelength=1, handletextpad=.3, fontsize=5.8)
    save(fig, out, "fig_additions")
    (out / "examples_v5.md").write_text("\n".join(f"fig_additions: {row['tile_id']} ({name})" for name, row in picks) + "\n")


if __name__ == "__main__":
    main()
