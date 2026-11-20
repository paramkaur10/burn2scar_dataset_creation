#!/usr/bin/env python3
"""Download released image tiles from the Hugging Face mirror into the images root, only the ones a step needs.

  python tools/fetch_images.py --for points cems zeros --zero-sample 500
  python tools/fetch_images.py --tiles-from validation/referee/none_tiles.csv

  points  every tile of every fire with a reviewed point (both scenes), for score_models.py points and review_blind.py
  cems    the tiles paired with a CEMS area, for score_models.py cems
  zeros   the random tiles referee_checks.py misc reads (same sample: pass it the same --zero-sample)
  all     every released tile (about 30 GB)

Files land at <images-root>/<image_path>, as in the release, and files already there are skipped.
"""
import argparse
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]


def wanted(args, index):
    released = index[index["split"].notna()]
    tiles = set()
    for part in args.parts:
        if part == "all":
            tiles |= set(released["tile_id"])
        elif part == "points":
            points = pd.read_csv(Path(args.review) / "points.csv", low_memory=False)
            fires = set(index.loc[index["tile_id"].isin(points["tile_id"]), "fire_id"])
            tiles |= set(released.loc[released["fire_id"].isin(fires), "tile_id"])
        elif part == "cems":
            tiles |= set(pd.read_csv(args.cems_tiles)["tile_id"])
        elif part == "zeros":
            tiles |= set(released.sample(min(args.zero_sample, len(released)), random_state=0)["tile_id"])
    for path in args.tiles_from:
        tiles |= set(pd.read_csv(path)["tile_id"])
    return released[released["tile_id"].isin(tiles)]


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--for", dest="parts", nargs="*", default=[], choices=["points", "cems", "zeros", "all"])
    parser.add_argument("--tiles-from", nargs="*", default=[], help="CSV files with a tile_id column")
    parser.add_argument("--index", default="data/v4/index.csv")
    parser.add_argument("--images-root", default="data/v1")
    parser.add_argument("--review", default="review2")
    parser.add_argument("--cems-tiles", default="validation/cems_tiles.csv")
    parser.add_argument("--zero-sample", type=int, default=2000, help="as referee_checks.py misc --zero-sample")
    parser.add_argument("--repo", default="", help="Hugging Face data set; default hf_repo_id in config.yaml")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    from huggingface_hub import hf_hub_download

    repo = args.repo or (yaml.safe_load(Path(args.config).read_text()) or {}).get("hf_repo_id", "neet1797/burn2scar")
    index = pd.read_csv(args.index, low_memory=False)
    tiles = wanted(args, index)
    root = Path(args.images_root)
    todo = [p for p in tiles["image_path"] if not (root / p).exists()]
    print(f"{len(tiles):,} tiles wanted, {len(tiles) - len(todo):,} already there, {len(todo):,} to download from {repo}", flush=True)

    def fetch(path):
        for attempt in range(4):
            try:
                hf_hub_download(repo_id=repo, repo_type="dataset", filename=path, local_dir=str(root))
                return path, ""
            except Exception as exc:   # rate limits and dropped connections: wait and try again
                error = f"{type(exc).__name__}: {str(exc)[:160]}"
                import time
                time.sleep(10 * (attempt + 1))
        return path, error

    failed = []
    with ThreadPoolExecutor(args.workers) as pool:
        futures = [pool.submit(fetch, p) for p in todo]
        for done, future in enumerate(as_completed(futures), 1):
            path, error = future.result()
            if error:
                failed.append((path, error))
            if done % 100 == 0 or done == len(todo):
                print(f"  {done:,}/{len(todo):,} done, {len(failed)} failed", flush=True)
    if failed:
        print("Failed:", *[f"  {p}: {e}" for p, e in failed[:10]], sep="\n")
        sys.exit(1)
    print(f"All {len(tiles):,} images are in {root}")


if __name__ == "__main__":
    main()
