"""Random test tiles with their image, their mask and a model's prediction, drawn by lot, not picked by hand.

  python tools/make_gallery.py --model release_models/burn2scar-unet-r34 --out paper/gallery
  python tools/make_gallery.py --model neet1797/burn2scar-unet-r34 --n 8 --seed 2

Tiles are drawn at random from the test set of the main split (column split of the index), with a fixed
seed, so the same command gives the same tiles. Without --model it draws image and mask only.

Writes gallery.png and gallery.pdf, and gallery.json with the tiles drawn and, for each, the share of
pixels where the prediction matches the mask (older_burn and no data left out, as in training).
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio

sys.path.insert(0, str(Path(__file__).resolve().parent))
from make_paper_figures import CLASS_COLOUR, CLASS_LABEL, CLASSES, COUNTRY_NAME, MASK_VALUE, read_rgb, set_style  # noqa: E402
from review_points import BAND, TRUE_COLOUR  # noqa: E402

import matplotlib.pyplot as plt  # noqa: E402  (make_paper_figures sets the backend)
from matplotlib.colors import BoundaryNorm, ListedColormap  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402

SCORED = [0, 2, 3, 4, 5, 6]


def draw(frame, n, seed):
    test = frame[frame["split"] == "test"]
    if len(test) < n:
        raise SystemExit(f"Only {len(test)} test tiles.")
    return test.sample(n, random_state=seed).reset_index(drop=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--index", default="data/v3/index.csv")
    parser.add_argument("--images-root", default="data/v1")
    parser.add_argument("--masks-root", default="data/v3")
    parser.add_argument("--model", default="", help="model folder or Hub repo for the prediction row")
    parser.add_argument("--n", type=int, default=8)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--out", default="paper/gallery")
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    tiles = draw(pd.read_csv(args.index), args.n, args.seed)
    net = config = None
    if args.model:
        from predict import load_model, predict_array
        net, config = load_model(args.model, args.device)
    values = sorted(MASK_VALUE.values())
    by_value = {value: name for name, value in MASK_VALUE.items()}
    colours = ListedColormap([CLASS_COLOUR[by_value[value]] for value in values])
    norm = BoundaryNorm([v - .5 for v in values] + [values[-1] + .5], colours.N)

    set_style()
    rows = 3 if net is not None else 2
    width = 17.0 / 2.54
    fig, axes = plt.subplots(rows, args.n, figsize=(width, width / args.n * rows + 0.75), squeeze=False)
    record, any_nodata = [], False
    for column, tile in zip(axes.T, tiles.itertuples()):
        image_path, mask_path = Path(args.images_root) / tile.image_path, Path(args.masks_root) / tile.mask_path
        with rasterio.open(mask_path) as src:
            mask = src.read(1)
        any_nodata |= bool((mask == MASK_VALUE["nodata"]).any())
        column[0].imshow(read_rgb(image_path, TRUE_COLOUR, BAND))
        column[1].imshow(mask, cmap=colours, norm=norm, interpolation="nearest")
        entry = {"tile_id": tile.tile_id, "country": tile.country, "window": tile.window, "age_days": float(tile.age_days)}
        if net is not None:
            with rasterio.open(image_path) as src:
                predicted = predict_array(net, config, src.read(), args.device)
            column[2].imshow(predicted, cmap=colours, norm=norm, interpolation="nearest")
            scored = np.isin(mask, SCORED)
            agreement = float((predicted[scored] == mask[scored]).mean()) if scored.any() else float("nan")
            entry["pixel_agreement"] = agreement
            column[2].set_xlabel(f"{agreement * 100:.0f} % agree", fontsize=6, labelpad=2)
        column[0].set_title(f"{COUNTRY_NAME.get(tile.country, tile.country)}, {tile.window}\nday {tile.age_days:.0f}",
                            fontsize=6, linespacing=1.1, pad=2.5)
        record.append(entry)
    for ax in axes.ravel():
        ax.set_xticks([])
        ax.set_yticks([])
        for side in ax.spines.values():
            side.set_linewidth(.3)
    for ax, label in zip(axes[:, 0], ("True colour", "Mask", "Prediction")):
        ax.set_ylabel(label, fontsize=6.5, labelpad=2)
    handles = [Patch(facecolor=CLASS_COLOUR[name], edgecolor="#8a8984", linewidth=.3, label=CLASS_LABEL[name]) for name in [*CLASSES, *(["nodata"] if any_nodata else [])]]
    fig.legend(handles=handles, loc="lower center", ncol=len(handles), columnspacing=.7, handlelength=1, handletextpad=.3, fontsize=5.8)
    fig.subplots_adjust(left=.035, right=.995, top=.9, bottom=.1 if net is None else .085, wspace=.04, hspace=.12)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    fig.savefig(out / "gallery.pdf", facecolor="white")
    fig.savefig(out / "gallery.png", dpi=300, facecolor="white")
    summary = {"seed": args.seed, "split": "test of column split", "model": args.model or None, "tiles": record}
    if net is not None:
        summary["median_pixel_agreement"] = float(np.nanmedian([t["pixel_agreement"] for t in record]))
    (out / "gallery.json").write_text(json.dumps(summary, indent=1))
    print(f"Wrote {out / 'gallery.png'}, gallery.pdf and gallery.json: {len(record)} random test tiles, seed {args.seed}.")
    if net is not None:
        print("Pixel agreement per tile: " + ", ".join(f"{t['pixel_agreement'] * 100:.0f} %" for t in record))
    print("Send gallery.png.")


if __name__ == "__main__":
    main()
