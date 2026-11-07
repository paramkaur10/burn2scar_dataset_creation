#!/usr/bin/env python3
"""Unpack images and masks into plain NumPy files, so training spends its CPU on training.

Reading one compressed tile and its mask takes about 30 ms, which on a server with few cores
limits every training run. The same tile as NumPy files takes about 2 ms. This uses about 48 GB
of disk, takes a few minutes, and can be deleted at any time without losing anything.

  python tools/build_cache.py --index data/v2/index.csv --images-root data/v1 --masks-root data/v2 --out data/cache

Then add --cache-root data/cache to run_baselines.py and classic_baselines.py. Run this again after
rebuilding the masks: it replaces any file older than its source.
"""
import argparse
import os
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio


def unpack(job):
    source, target = job
    target = Path(target)
    if target.exists() and target.stat().st_mtime >= Path(source).stat().st_mtime:
        return 0
    target.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(source) as src:
        data = src.read()
    if data.shape[0] == 1:
        data = data[0]
    temp = target.with_name(f"{target.name}.{os.getpid()}.tmp.npy")
    np.save(temp, data)
    temp.replace(target)
    return data.nbytes


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--index", default="data/v2/index.csv")
    parser.add_argument("--images-root", default="data/v1")
    parser.add_argument("--masks-root", default="data/v2")
    parser.add_argument("--out", default="data/cache")
    parser.add_argument("--workers", type=int, default=os.cpu_count())
    args = parser.parse_args()

    index = pd.read_csv(args.index, low_memory=False)
    out = Path(args.out)
    jobs = [(str(Path(args.images_root) / p), str(out / f"{p}.npy")) for p in index["image_path"]]
    jobs += [(str(Path(args.masks_root) / p), str(out / f"{p}.npy")) for p in index["mask_path"]]
    with ProcessPoolExecutor(args.workers) as pool:
        written = list(pool.map(unpack, jobs, chunksize=64))
    print(f"{sum(1 for w in written if w):,} files written, {sum(1 for w in written if not w):,} already up to date, "
          f"{sum(written) / 1e9:.1f} GB new. Cache: {out}")


if __name__ == "__main__":
    main()
