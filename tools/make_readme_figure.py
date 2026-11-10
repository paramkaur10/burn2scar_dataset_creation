"""The picture at the top of the README: burn2scar tiles in false colour with their burn outlined, over their masks.

  python tools/make_readme_figure.py --index data/v3/index.csv --images-root data/v1 --masks-root data/v3

Draws, with a fixed seed, one fresh scene (5 to 60 days after its fire) from each country and two old scenes
(120 to 300 days after), among tiles with at least 15 % burn of their window's class and under 2 % cloud,
shadow or no data. So the tiles are random, but chosen to show a burn. The README's caption says that.

Writes assets/burn2scar_examples.png (about 2,400 pixels wide) and assets/burn2scar_examples.json with the tiles.
"""
import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import rasterio  # noqa: E402
from matplotlib.colors import BoundaryNorm, ListedColormap  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402

BANDS = ["B01", "B02", "B03", "B04", "B05", "B06", "B07", "B08", "B8A", "B09", "B10", "B11", "B12"]
FALSE_COLOUR = (("B12", .40), ("B8A", .50), ("B04", .28))  # as in the paper's figures
# Mask value: (label, colour). Burn age is one hue, darker when more recent; the colours of the paper's figures.
CLASSES = {2: ("fresh_burn", "#7f2704"), 3: ("old_burn", "#d9480f"), 7: ("older_burn", "#f5a04a"),
           0: ("clear", "#f1efe8"), 6: ("water", "#2a78d6"), 5: ("cloud shadow", "#25272d"),
           4: ("cloud", "#b4c0ce"), 255: ("no data", "#000000")}
COUNTRY = {"IT": "Italy", "ES": "Spain", "FR": "France", "EL": "Greece"}
INK = "#13293d"


def choose(index, seed):
    index = index[index["split"].notna()]  # the released tiles only
    hidden = index["frac_cloud"] + index["frac_shadow"] + index["frac_nodata"]
    fresh = index[(index["window"] == "fresh") & (index["frac_fresh"] >= .15) & (hidden < .02)]
    old = index[(index["window"] == "old") & (index["frac_old"] >= .15) & (hidden < .02)]
    picks = [fresh[fresh["country"] == c].sample(1, random_state=seed) for c in COUNTRY if (fresh["country"] == c).any()]
    picks.append(old.sample(2, random_state=seed))
    return pd.concat(picks).reset_index(drop=True)


def false_colour(path):
    with rasterio.open(path) as src:
        image = src.read().astype(np.float32) / 10000
    rgb = np.stack([image[BANDS.index(band)] / scale for band, scale in FALSE_COLOUR], axis=-1)
    return np.clip(rgb, 0, 1) ** .9


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--index", default="data/v3/index.csv")
    parser.add_argument("--images-root", default="data/v1")
    parser.add_argument("--masks-root", default="data/v3")
    parser.add_argument("--out", default="assets")
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()

    tiles = choose(pd.read_csv(args.index, low_memory=False), args.seed)
    values = sorted(CLASSES)
    cmap = ListedColormap([CLASSES[v][1] for v in values])
    norm = BoundaryNorm([v - .5 for v in values] + [values[-1] + .5], cmap.N)

    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9, "text.color": INK})
    n = len(tiles)
    fig, axes = plt.subplots(2, n, figsize=(1.6 * n, 3.75), squeeze=False)
    seen = set()
    for column, tile in zip(axes.T, tiles.itertuples()):
        with rasterio.open(Path(args.masks_root) / tile.mask_path) as src:
            mask = src.read(1)
        seen |= set(np.unique(mask).tolist())
        burn = np.isin(mask, [2, 3]).astype(float)
        column[0].imshow(false_colour(Path(args.images_root) / tile.image_path), interpolation="bilinear")
        column[0].contour(burn, levels=[.5], colors="white", linewidths=.9)
        column[1].imshow(mask, cmap=cmap, norm=norm, interpolation="nearest")
        column[0].set_title(f"{COUNTRY.get(tile.country, tile.country)} · day {tile.age_days:.0f}", fontsize=9, pad=4)
        for ax in column:
            ax.set_xticks([])
            ax.set_yticks([])
            for side in ax.spines.values():
                side.set_visible(False)
    shown = [v for v in [2, 3, 7, 0, 6, 5, 4, 255] if v in seen]
    handles = [Patch(facecolor=CLASSES[v][1], edgecolor="#8a8984", linewidth=.4, label=CLASSES[v][0]) for v in shown]
    fig.legend(handles=handles, loc="lower center", ncol=len(handles), frameon=False, fontsize=8.5,
               handlelength=1.1, handletextpad=.4, columnspacing=1.1)
    fig.subplots_adjust(left=.005, right=.995, top=.93, bottom=.1, wspace=.04, hspace=.04)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    fig.savefig(out / "burn2scar_examples.png", dpi=200, facecolor="white")
    record = [{"tile_id": t.tile_id, "country": t.country, "window": t.window, "age_days": float(t.age_days),
               "split": t.split} for t in tiles.itertuples()]
    (out / "burn2scar_examples.json").write_text(json.dumps({"seed": args.seed, "tiles": record}, indent=1))
    print(f"Wrote {out / 'burn2scar_examples.png'} and burn2scar_examples.json: {n} tiles, seed {args.seed}.")


if __name__ == "__main__":
    main()
