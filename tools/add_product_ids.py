#!/usr/bin/env python3
"""Record which Sentinel-2 L1C products each tile came from.

The pipeline kept only the acquisition time. To trace or reproduce a tile, a reader needs the source
product, so this looks it up in the public Copernicus Data Space Ecosystem catalogue (OData) and adds it
to the index. A catalogue search needs no account, downloads no imagery and uses none of the monthly
Sentinel Hub requests. Safe to stop and restart.

  python tools/add_product_ids.py --compare 20
  python tools/add_product_ids.py

--compare looks up 20 random tiles both here and in the Sentinel Hub catalogue the pipeline read from
(20 Sentinel Hub requests, out of 10,000 a month), prints both and changes nothing. Run it first: every
product Sentinel Hub names should also be found here.

The full run writes product_ids.csv next to the index, a line per tile as it goes. Once every tile has
a line, it adds two columns to the index:
  product_id           names of the L1C products over the tile from the satellite passes within --minutes
                       of the acquisition time, joined with ";"
  processing_baseline  their processing baselines, such as N0509
There is more than one product where granules overlap, and where two satellites passed minutes apart.
Both passes belong in the list: for each scene the pipeline asked Sentinel Hub for 3 hours either side of
the acquisition time (eo-learn's time_difference), and Sentinel Hub mosaics with the most recent on top,
so the later pass covers the earlier one wherever it has data.
A tile with nothing found gets an empty product_id. --redo-empty --minutes 60 looks those up again.
"""
import argparse
import csv
import os
import re
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import timedelta, timezone
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

CATALOGUE = "https://catalogue.dataspace.copernicus.eu/odata/v1/Products"
BASELINE = re.compile(r"_(N\d{4})_")
PASS = re.compile(r"^(S2[A-Z])_MSIL1C_(\d{8}T\d{6})_N\d{4}_(R\d{3})_")
MAX_WORKERS = 4    # The Data Space allows only a few connections at once per user.
SH_MINUTES = 5     # Sentinel Hub dates a tile the way the pipeline did, so a narrow window is enough there.
RETRY_CODES = {429, 500, 502, 503, 504}
STOPPING = threading.Event()  # Set on Ctrl-C, so no worker sits out a long wait.
_local = threading.local()


class Stop(Exception):
    """An answer that won't change if asked again, such as a query the catalogue rejects."""


def rest(seconds):
    if STOPPING.wait(seconds):
        raise Stop("stopped")


def http():
    if not hasattr(_local, "session"):
        import requests
        _local.session = requests.Session()
    return _local.session


def odata_url(row, minutes, base):
    # The catalogue may date a product by the start of the satellite's recording rather than the moment
    # this tile was seen, minutes apart, so the window is wide.
    moment = pd.Timestamp(row.scene_time).to_pydatetime()
    moment = moment.replace(tzinfo=timezone.utc) if moment.tzinfo is None else moment.astimezone(timezone.utc)
    start, end = ((moment + timedelta(minutes=m)).strftime("%Y-%m-%dT%H:%M:%S.000Z") for m in (-minutes, minutes))
    w, s, e, n = (f"{value:.6f}" for value in (row.lon_min, row.lat_min, row.lon_max, row.lat_max))
    polygon = f"POLYGON(({w} {s},{e} {s},{e} {n},{w} {n},{w} {s}))"
    query = (f"Collection/Name eq 'SENTINEL-2' and ContentDate/Start gt {start} and ContentDate/Start lt {end} "
             f"and OData.CSC.Intersects(area=geography'SRID=4326;{polygon}')")
    return f"{base}?$filter={query}&$top=100"


def retry_after(reply):
    try:
        return float(reply.headers.get("Retry-After", 0))
    except ValueError:
        return 0


def get_json(url, pause, retries):
    import requests

    wait, problem = 5, ""
    for attempt in range(retries):
        rest(pause)  # A steady pace, so the catalogue never has to push back.
        try:
            reply = http().get(url, timeout=60)
            if reply.status_code == 200:
                return reply.json()
            if reply.status_code not in RETRY_CODES:
                raise Stop(f"HTTP {reply.status_code}: {reply.text[:400]}")
            problem = f"HTTP {reply.status_code}"
            wait = max(wait, retry_after(reply))
        except (requests.RequestException, ValueError) as error:  # ValueError: a reply that isn't JSON.
            problem = f"{type(error).__name__}: {str(error)[:200]}"
        if attempt < retries - 1:
            rest(wait)
            wait = min(wait * 2, 300)
    raise RuntimeError(f"gave up after {retries} tries, last: {problem}")


def odata_ids(row, minutes, base, pause, retries=6):
    url, names = odata_url(row, minutes, base), []
    for _ in range(10):  # Pages. One is the norm.
        reply = get_json(url, pause, retries)
        names += [item.get("Name", "") for item in reply.get("value", [])]
        url = reply.get("@odata.nextLink")
        if not url:
            break
    return sorted({name for name in names if "_MSIL1C_" in name})  # The collection also holds L2A.


def baselines(names):
    return sorted({m.group(1) for m in map(BASELINE.search, names) if m})


def pass_of(name):
    """Satellite, start of its recording and relative orbit: the same for every granule of one pass."""
    found = PASS.match(name)
    return found.groups() if found else (name,)


def pass_count(value):
    return len({pass_of(name) for name in value.split(";") if name})


def bare(name):
    return name[:-5] if name.endswith(".SAFE") else name


def make_sh():
    from dotenv import load_dotenv

    from src.s2_pipeline import make_sh_config
    load_dotenv()
    return make_sh_config(os.environ["SH_CLIENT_ID"], os.environ["SH_CLIENT_SECRET"])


def sentinelhub_ids(row, sh):
    from sentinelhub import BBox, CRS, SentinelHubCatalog

    from src.s2_pipeline import S2L1C_COLLECTION

    moment = pd.Timestamp(row.scene_time)
    bbox = BBox((row.minx, row.miny, row.maxx, row.maxy), crs=CRS(int(row.epsg)))
    window = ((moment - timedelta(minutes=SH_MINUTES)).isoformat(), (moment + timedelta(minutes=SH_MINUTES)).isoformat())
    results = SentinelHubCatalog(sh).search(collection=S2L1C_COLLECTION, bbox=bbox, time=window,
                                            fields={"include": ["id"], "exclude": []})
    return sorted(result["id"] for result in results)


def run_compare(args, index, target):
    sh = make_sh()
    picks = index.sample(n=min(args.compare, len(index)), random_state=args.seed)
    rows = []
    for number, row in enumerate(picks.itertuples(), 1):
        found = odata_ids(row, args.minutes, args.catalogue, args.pause)
        theirs = sentinelhub_ids(row, sh)
        missing = sorted({bare(n) for n in theirs} - {bare(n) for n in found})
        rows.append({"tile_id": row.tile_id, "scene_time": row.scene_time, "catalogue": ";".join(found),
                     "sentinel_hub": ";".join(theirs), "sentinel_hub_not_found_here": ";".join(missing)})
        print(f"[{number}/{len(picks)}] {row.tile_id}\n  here:         {', '.join(found) or 'nothing'}\n"
              f"  Sentinel Hub: {', '.join(theirs) or 'nothing'}", flush=True)
    table = pd.DataFrame(rows)
    table.to_csv(target, index=False)
    named = table[table["sentinel_hub"] != ""]
    agree = int((named["sentinel_hub_not_found_here"] == "").sum())
    extra = sum(len({bare(n) for n in r.catalogue.split(";") if n} - {bare(n) for n in r.sentinel_hub.split(";") if n}) > 0
                for r in table.itertuples())
    print(f"\nSentinel Hub named a product for {len(named)} of {len(table)} tiles. "
          f"All of its products were also found here for {agree} of them.")
    print(f"Tiles where this lookup found more products than Sentinel Hub: {extra} "
          "(a second granule over the tile, a reprocessed copy, or a second satellite pass).")
    print(f"Wrote {target}. Nothing else changed.")
    if agree < len(named) or not len(named):
        print("Not the same: send this output before the full run.")
    else:
        print("Same products: the full run can go ahead.")


def read_done(path):
    if not path.exists():
        return pd.DataFrame(columns=["tile_id", "product_id", "processing_baseline"])
    return pd.read_csv(path, keep_default_na=False, dtype=str).drop_duplicates("tile_id", keep="last")


def merge_into_index(index_path, index, products):
    table = pd.DataFrame({"tile_id": list(products), "product_id": list(products.values())})
    table["processing_baseline"] = table["product_id"].map(lambda value: ";".join(baselines(value.split(";"))))
    merged = index.drop(columns=["product_id", "processing_baseline"], errors="ignore").merge(table, on="tile_id", how="left")
    temporary = Path(index_path).with_name(Path(index_path).name + ".tmp")
    merged.to_csv(temporary, index=False)
    temporary.replace(index_path)


def run_lookup(args, index, out):
    done = read_done(out)
    finished = set(done.loc[done["product_id"] != "", "tile_id"] if args.redo_empty else done["tile_id"])
    pending = index[~index["tile_id"].isin(finished)]
    if args.limit:
        pending = pending.head(args.limit)
    print(f"{len(pending):,} tiles to look up, {len(finished):,} already done.", flush=True)

    failed = 0
    if len(pending):
        head = next(pending.itertuples())
        try:  # One on its own first: if the catalogue rejects the query, every tile would fail the same way.
            first = odata_ids(head, args.minutes, args.catalogue, args.pause)
        except Stop as error:
            raise SystemExit(f"The catalogue rejected the query, so nothing was looked up:\n{error}")
        except RuntimeError as error:
            raise SystemExit(f"Couldn't reach the catalogue ({error}). Nothing is lost: run the same command later.")
        print(f"First tile: {', '.join(first) or 'nothing found'}", flush=True)
        new_file = not out.exists()
        with open(out, "a", newline="") as handle:
            writer = csv.writer(handle)
            if new_file:
                writer.writerow(["tile_id", "product_id", "processing_baseline"])
            writer.writerow([head.tile_id, ";".join(first), ";".join(baselines(first))])
            pool = ThreadPoolExecutor(args.workers)
            try:
                futures = {pool.submit(odata_ids, row, args.minutes, args.catalogue, args.pause): row.tile_id
                           for row in pending.iloc[1:].itertuples()}
                started = time.time()
                for number, future in enumerate(as_completed(futures), 1):
                    try:
                        names = future.result()
                    except (Stop, RuntimeError) as error:
                        failed += 1
                        print(f"{futures[future]}: failed: {str(error)[:300]}", flush=True)
                        continue
                    writer.writerow([futures[future], ";".join(names), ";".join(baselines(names))])
                    if number % 200 == 0 or number == len(futures):
                        handle.flush()
                        rate = number / max(time.time() - started, 1)
                        print(f"{number:,}/{len(futures):,}, about {(len(futures) - number) / rate / 60:.0f} min left", flush=True)
            except KeyboardInterrupt:
                STOPPING.set()
                pool.shutdown(wait=True, cancel_futures=True)
                handle.flush()
                raise SystemExit("Stopped. What was found so far is saved: run the same command to carry on.")
            pool.shutdown()

    done = read_done(out)
    left = int((~index["tile_id"].isin(done["tile_id"])).sum())
    products = done.set_index("tile_id")["product_id"].to_dict()
    values = pd.Series(products, dtype=str)
    counts = values.map(lambda value: len(value.split(";")) if value else 0)
    print(f"Lookups: {len(done):,} of {len(index):,} tiles. Products per tile: "
          + ", ".join(f"{k}: {v:,}" for k, v in counts.value_counts().sort_index().items()))
    print("Baselines:", values.map(lambda value: ";".join(baselines(value.split(";"))) or "none").value_counts().head(12).to_dict())
    if failed:
        print(f"{failed:,} failed this time. Run the same command again to retry them.")
    if left:
        print(f"{left:,} tiles still to look up. The index changes only once every tile has a lookup.")
        return
    merge_into_index(args.index, index, products)
    mixed = sum(pass_count(value) > 1 for value in products.values())
    print(f"Added product_id and processing_baseline to {args.index}. Nothing found for {int((counts == 0).sum()):,} tiles.")
    print(f"Tiles where two satellites passed within the window, both listed: {mixed:,}.")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--index", default="data/v3/index.csv")
    parser.add_argument("--out", help="default: product_ids.csv next to the index")
    parser.add_argument("--minutes", type=float, default=30, help="search this far either side of the acquisition time")
    parser.add_argument("--workers", type=int, default=3)
    parser.add_argument("--pause", type=float, default=0.5, help="seconds each worker waits before a request")
    parser.add_argument("--limit", type=int, default=0, help="only this many tiles, for a quick trial")
    parser.add_argument("--redo-empty", action="store_true", help="look up again the tiles where nothing was found")
    parser.add_argument("--compare", type=int, default=0, help="check this many random tiles against Sentinel Hub, change nothing")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--catalogue", default=CATALOGUE, help=argparse.SUPPRESS)  # For a test server.
    args = parser.parse_args()
    if not 1 <= args.workers <= MAX_WORKERS:
        parser.error(f"--workers must be 1 to {MAX_WORKERS}")

    index = pd.read_csv(args.index, low_memory=False)
    out = Path(args.out) if args.out else Path(args.index).with_name("product_ids.csv")
    if args.compare:
        run_compare(args, index, out.with_name(out.stem + "_compare.csv"))
    else:
        run_lookup(args, index, out)


if __name__ == "__main__":
    main()
