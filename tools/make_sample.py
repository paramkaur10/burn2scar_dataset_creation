#!/usr/bin/env python3
"""Build sample.zip: a small part of burn2scar for a first look without the full download.

Draws, with a fixed seed, --per-stratum tiles for each country and scene window (25 x 4 x 2 = 200 by default) so that
the sample holds two test fires seen in both windows (the pair that notebooks/01_explore_burn2scar.ipynb shows) and at
least --per-class tiles with each of old_burn, older_burn, water, cloud and cloud shadow on 5 % of the tile or more.
The tiles keep the release layout and their index rows, so code written for the sample runs on the full release.

Run from the repository root, with the DOI of the release the sample goes into:
    python tools/make_sample.py --version 4.1 --doi 10.5281/zenodo.XXXXXXXX
"""
import argparse
import shutil
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

CLASSES = {"old_burn": "frac_old", "older_burn": "frac_older", "water": "frac_water", "cloud": "frac_cloud",
           "cloud shadow": "frac_shadow"}
TOP = "burn2scar_sample"


def choose(index, per_stratum, per_class, pairs, min_share, seed):
    """Index rows of the sample, and the number of test fires it holds in both windows."""
    rng = np.random.default_rng(seed)
    stratum = index["country"].astype(str) + " " + index["window"].astype(str)
    counts = dict.fromkeys(stratum.unique(), 0)
    chosen = []

    def add(labels, need):
        added = 0
        for label in labels:
            if added >= need:
                break
            if label in chosen or counts[stratum[label]] >= per_stratum:
                continue
            chosen.append(label)
            counts[stratum[label]] += 1
            added += 1
        return added

    # Test fires seen in both windows, picked the way the explore notebook picks its pair.
    hidden = index["frac_cloud"] + index["frac_shadow"] + index["frac_nodata"]
    test = index[(index["split"] == "test") & (hidden < .02)]
    fresh = test[(test["window"] == "fresh") & (test["frac_fresh"] >= .2)]
    old = test[(test["window"] == "old") & (test["frac_old"] >= .15)]
    found = 0
    for fire in rng.permutation(sorted(set(fresh["fire_id"]) & set(old["fire_id"]))):
        if found >= pairs:
            break
        first, second = fresh.index[fresh["fire_id"] == fire][0], old.index[old["fire_id"] == fire][0]
        if counts[stratum[first]] < per_stratum and counts[stratum[second]] < per_stratum:
            add([first], 1)
            add([second], 1)
            found += 1
    # At least per_class tiles holding each of the rarer classes.
    for column in CLASSES.values():
        have = int(index.loc[chosen, column].ge(min_share).sum()) if chosen else 0
        add(rng.permutation(index.index[index[column] >= min_share]), per_class - have)
    # Fill every country and window.
    for key in sorted(counts):
        add(rng.permutation(index.index[stratum == key]), per_stratum - counts[key])
    return index.loc[sorted(chosen)], found


def readme(sample, total, per_stratum, least, pairs, args):
    return f"""# burn2scar sample

{len(sample)} of the {total:,} tiles of burn2scar {args.version} (https://doi.org/{args.doi}), for a first look without the
full download: {per_stratum} for each country (IT, ES, FR, EL) and scene window (fresh: 5-60 days after the fire;
old: 120-300 days), drawn at random with seed {args.seed}, with at least {least} tiles holding each of old_burn,
older_burn, water, cloud and cloud shadow, and {pairs} test fire{"" if pairs == 1 else "s"} seen in both windows. The layout and the index
columns are those of the full release, so code written for the sample runs on the full release unchanged.

Each acquisition sits under acquisitions/<fire_id>/<timestamp>/:

- *_s2.tif: 13 bands (B01 to B12, with B8A after B08), top-of-atmosphere reflectance x 10,000, uint16, 0 = no data,
  256 x 256 pixels at 10 m in the local UTM zone;
- *_mask.tif: 0 clear, 2 fresh_burn, 3 old_burn, 4 cloud, 5 cloud shadow, 6 water, 7 older_burn, 255 no data;
- *_age.tif: days since the fire, uint16, 0 where no perimeter applies.

```python
import pandas as pd
import rasterio

index = pd.read_csv("index.csv")
row = index.iloc[0]
with rasterio.open(row["image_path"]) as src:
    reflectance = src.read() / 10000
with rasterio.open(row["mask_path"]) as src:
    mask = src.read(1)
with rasterio.open(row["age_path"]) as src:
    age = src.read(1)
```

The notebook notebooks/01_explore_burn2scar.ipynb in https://github.com/paramkaur10/burn2scar_dataset_creation runs
on this folder: set BURN2SCAR_ROOT to its full path before starting Jupyter from the repository.

LICENSE.txt and NOTICE.txt give the terms and the credits to keep. Please cite the data set: https://doi.org/{args.doi}
"""


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--index", default="data/v4/index.csv")
    parser.add_argument("--images-root", default="data/v1")
    parser.add_argument("--masks-root", default="data/v4", help="holds the masks and the age rasters")
    parser.add_argument("--docs", default="release_docs", help="LICENSE.txt and NOTICE.txt are copied from here")
    parser.add_argument("--out", default="release/sample.zip")
    parser.add_argument("--per-stratum", type=int, default=25)
    parser.add_argument("--per-class", type=int, default=5)
    parser.add_argument("--pairs", type=int, default=2)
    parser.add_argument("--min-share", type=float, default=.05)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--version", default="4.1")
    parser.add_argument("--doi", required=True, help="DOI of the release the sample goes into")
    args = parser.parse_args()

    index = pd.read_csv(args.index, low_memory=False)
    if index["split"].isna().any():
        print(f"Left out {int(index['split'].isna().sum()):,} tiles with no split: they are not released.")
        index = index[index["split"].notna()]
    sample, pairs = choose(index, args.per_stratum, args.per_class, args.pairs, args.min_share, args.seed)

    out = Path(args.out)
    folder = out.parent / TOP
    if folder.exists():
        shutil.rmtree(folder)
    missing = []
    for _, row in sample.iterrows():
        for column, root in (("image_path", args.images_root), ("mask_path", args.masks_root),
                             ("age_path", args.masks_root)):
            if not isinstance(row.get(column), str):
                continue
            source = Path(root) / row[column]
            if not source.exists():
                missing.append(str(source))
                continue
            target = folder / row[column]
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, target)
    if missing:
        raise SystemExit(f"{len(missing)} files not found, for example {missing[0]}: check --images-root and --masks-root")
    sample.to_csv(folder / "index.csv", index=False)
    for name in ("LICENSE.txt", "NOTICE.txt"):
        if (Path(args.docs) / name).exists():
            shutil.copy2(Path(args.docs) / name, folder / name)
        else:
            print(f"Warning: {Path(args.docs) / name} not found, so the sample goes without it.")
    strata = sample.groupby(["country", "window"]).size().unstack(fill_value=0)
    sizes = sorted(set(strata.stack()))
    per_stratum = str(sizes[0]) if len(sizes) == 1 else f"{sizes[0]} to {sizes[-1]}"
    coverage = {name: int((sample[column] >= args.min_share).sum()) for name, column in CLASSES.items()}
    (folder / "README.md").write_text(readme(sample, len(index), per_stratum, min(coverage.values()), pairs, args))

    with zipfile.ZipFile(out, "w") as archive:
        for path in sorted(folder.rglob("*")):
            if path.is_file():
                kind = zipfile.ZIP_STORED if path.suffix == ".tif" else zipfile.ZIP_DEFLATED
                archive.write(path, Path(TOP) / path.relative_to(folder), compress_type=kind)

    print(f"{len(sample)} tiles from {sample['fire_id'].nunique()} fires, {out.stat().st_size / 1e6:.0f} MB: {out}")
    print(f"Unzipped copy for testing: {folder}")
    print(strata.to_string())
    print("Tiles with at least {:.0%} of the class: ".format(args.min_share)
          + ", ".join(f"{name} {count}" for name, count in coverage.items()))
    print(f"Test fires seen in both windows: {pairs}")
    planned = args.per_stratum * len(strata.stack())
    if len(sample) != planned or min(coverage.values()) < args.per_class:
        print("Note: the sample differs from the plan, so the paper's sentence on sample.zip (Sect. 4.1) needs updating.")
    if pairs == 0:
        print("Note: no test fire is in both windows, so part 4 of the explore notebook will not run on the sample.")


if __name__ == "__main__":
    main()
