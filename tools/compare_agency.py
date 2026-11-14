#!/usr/bin/env python3
"""EFFIS perimeters and burn2scar labels against fire perimeters surveyed by a regional agency.

  python tools/compare_agency.py --agency data/agency/catalonia.zip --list-columns
  python tools/compare_agency.py --agency data/agency/catalonia.zip --name Catalonia --date-col DATA_INCEN \
      --regions Barcelona Girona Lleida Tarragona --index data/v4/index.csv --masks-root data/v4 --out validation

--agency is any vector file geopandas reads (shapefile, zipped shapefile, GeoPackage, GeoJSON); several files, such as
one per year, can be given at once. --date-col names the fire-date column; --list-columns prints the columns and a few
rows to find it. Areas are recomputed from the geometry in EPSG:3035, not taken from the file.

1. Detection: of the agency fires of at least --min-ha in the data set's period, the share an EFFIS perimeter matches
   (dated --days-before to --days-after around the agency date, and overlapping it), by size.
2. Outline: agency and EFFIS perimeters that match are grouped into events (one EFFIS perimeter can cover two agency
   fires, and the reverse), and each event's EFFIS area is compared with its agency area: precision, recall, F1 and
   IoU, pooled over events with bootstrap intervals over events, by size.
3. EFFIS fires with no agency perimeter: released burn2scar fires in --regions (the index's region column) and in the
   years the agency file covers that no agency perimeter matches. These are agency omissions, or burns agencies do not
   record, such as agricultural and prescribed burns.
4. Labels: on the released fresh-window tiles of matched fires, the fresh_burn class against every agency perimeter
   dated 0-90 days before the scene, on the 10 m grid, with cloud, cloud shadow and no data left out.
"""
import argparse
import sys
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
import yaml
from rasterio.features import rasterize
from shapely.geometry import box
from shapely.ops import unary_union

sys.path.insert(0, str(Path(__file__).resolve().parent))
from burn_labels import MASK_CLOUD, MASK_FRESH, MASK_NODATA, MASK_SHADOW, load_perimeters  # noqa: E402

SIZE_BINS = [0, 5, 10, 30, 100, 500, 2000, np.inf]
SIZE_LABELS = ["under 5 ha", "5-10 ha", "10-30 ha", "30-100 ha", "100-500 ha", "500-2000 ha", "over 2000 ha"]
EQUAL_AREA = 3035


def md(frame):
    head = "| " + " | ".join(map(str, frame.columns)) + " |"
    rule = "|" + "---|" * len(frame.columns)
    body = ["| " + " | ".join("" if pd.isna(v) else str(v) for v in row) + " |" for row in frame.itertuples(index=False)]
    return "\n".join([head, rule, *body])


def wilson(hits, total, z=1.96):
    if not total:
        return ""
    p = hits / total
    centre = (p + z * z / (2 * total)) / (1 + z * z / total)
    half = z * np.sqrt(p * (1 - p) / total + z * z / (4 * total * total)) / (1 + z * z / total)
    return f"{100 * p:.1f} ({100 * max(centre - half, 0):.1f}-{100 * min(centre + half, 1):.1f})"


def naive(values):
    dates = pd.to_datetime(values)
    return dates.dt.tz_localize(None) if dates.dt.tz is not None else dates


def read_agency(paths, date_col, dayfirst):
    parts = [gpd.read_file(p) for p in paths]
    frame = pd.concat([p.to_crs(EQUAL_AREA) for p in parts], ignore_index=True)
    frame = gpd.GeoDataFrame(frame, geometry="geometry", crs=EQUAL_AREA)
    raw = frame[date_col].astype(str)
    dates = pd.to_datetime(raw, errors="coerce", dayfirst=dayfirst, format="mixed")
    if dates.isna().mean() > .5:
        dates = pd.to_datetime(raw, errors="coerce", dayfirst=not dayfirst, format="mixed")
    frame["agency_date"] = naive(dates)
    frame["geometry"] = frame.geometry.make_valid()
    frame = frame[frame["agency_date"].notna() & ~frame.geometry.is_empty].reset_index(drop=True)
    frame["agency_id"] = np.arange(len(frame))
    frame["agency_ha"] = frame.geometry.area / 1e4
    return frame[["agency_id", "agency_date", "agency_ha", "geometry"]]


def match(agency, effis, days_before, days_after):
    """Pairs (agency_id, fire_id) that overlap, with the EFFIS date within the window around the agency date."""
    left, right = effis.sindex.query(agency.geometry.values, predicate="intersects")
    pairs = pd.DataFrame({"a": agency["agency_id"].to_numpy()[left], "e": right})
    if pairs.empty:
        return pd.DataFrame(columns=["agency_id", "fire_id"])
    pairs["days"] = ((effis["fire_date"].to_numpy()[pairs["e"]] - agency["agency_date"].to_numpy()[pairs["a"]])
                     / np.timedelta64(1, "D"))
    pairs = pairs[(pairs["days"] >= -days_before) & (pairs["days"] <= days_after)]
    # Overlap of more than a sliver, so a shared edge does not count.
    keep = [agency.geometry.iloc[a].intersection(effis.geometry.iloc[e]).area > 100 for a, e in zip(pairs["a"], pairs["e"])]
    pairs = pairs[np.array(keep, bool)] if len(pairs) else pairs
    return pd.DataFrame({"agency_id": pairs["a"].to_numpy(), "fire_id": effis["fire_id"].to_numpy()[pairs["e"]]})


def events(pairs):
    """Connected groups of agency fires and EFFIS perimeters that match each other."""
    parent = {}

    def find(x):
        while parent.setdefault(x, x) != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for a, e in zip(pairs["agency_id"], pairs["fire_id"]):
        parent[find(("a", a))] = find(("e", e))
    groups = {}
    for a, e in zip(pairs["agency_id"], pairs["fire_id"]):
        root = find(("a", a))
        groups.setdefault(root, (set(), set()))
        groups[root][0].add(a)
        groups[root][1].add(e)
    return list(groups.values())


def event_scores(agency, effis, pairs):
    by_agency = agency.set_index("agency_id").geometry
    by_effis = effis.set_index("fire_id").geometry
    rows = []
    for number, (agency_ids, fire_ids) in enumerate(events(pairs)):
        reference = unary_union(list(by_agency.loc[sorted(agency_ids)]))
        mapped = unary_union(list(by_effis.loc[sorted(fire_ids)]))
        both = reference.intersection(mapped).area
        rows.append({"event": number, "agency_fires": len(agency_ids), "effis_fires": len(fire_ids),
                     "agency_ha": reference.area / 1e4, "effis_ha": mapped.area / 1e4, "both_ha": both / 1e4})
    return pd.DataFrame(rows)


def pooled(frame, tp, fp, fn):
    t, f_p, f_n = frame[tp].sum(), frame[fp].sum(), frame[fn].sum()
    precision = t / (t + f_p) if t + f_p else np.nan
    recall = t / (t + f_n) if t + f_n else np.nan
    return {"precision": precision, "recall": recall, "f1": 2 * t / (2 * t + f_p + f_n) if t else np.nan,
            "iou": t / (t + f_p + f_n) if t else np.nan}


def with_interval(frame, tp, fp, fn, repeats, seed=0):
    point = pooled(frame, tp, fp, fn)
    rng = np.random.default_rng(seed)
    draws = [pooled(frame.iloc[rng.integers(len(frame), size=len(frame))], tp, fp, fn) for _ in range(repeats)] if len(frame) > 1 else []
    out = {}
    for key, value in point.items():
        if draws:
            values = [d[key] for d in draws if not pd.isna(d[key])]
            low, high = np.percentile(values, [2.5, 97.5])
            out[key] = f"{100 * value:.1f} ({100 * low:.1f}-{100 * high:.1f})"
        else:
            out[key] = "" if pd.isna(value) else f"{100 * value:.1f}"
    return out


def tile_scores(index, masks_root, agency, fire_ids):
    tiles = index[index["fire_id"].isin(fire_ids) & (index["window"] == "fresh") & index["split"].notna()]
    rows = []
    for record in tiles.itertuples():
        with rasterio.open(Path(masks_root) / record.mask_path) as src:
            mask, crs, transform, bounds = src.read(1), src.crs, src.transform, src.bounds
        scene = pd.Timestamp(record.scene_time)
        recent = agency[(agency["agency_date"] <= scene) & (agency["agency_date"] >= scene - pd.Timedelta(days=90))]
        footprint = gpd.GeoSeries([box(*bounds)], crs=crs).to_crs(EQUAL_AREA).iloc[0]
        recent = recent[recent.intersects(footprint)].to_crs(crs)
        reference = (rasterize([(g, 1) for g in recent.geometry if not g.is_empty], out_shape=mask.shape, transform=transform,
                               fill=0, dtype="uint8").astype(bool) if len(recent) else np.zeros(mask.shape, bool))
        valid = ~np.isin(mask, [MASK_CLOUD, MASK_SHADOW, MASK_NODATA])
        label = mask == MASK_FRESH
        rows.append({"tile_id": record.tile_id, "fire_id": record.fire_id, "area_ha": record.area_ha,
                     "tp": int((label & reference & valid).sum()), "fp": int((label & ~reference & valid).sum()),
                     "fn": int((~label & reference & valid).sum())})
    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--agency", nargs="+", required=True)
    parser.add_argument("--list-columns", action="store_true")
    parser.add_argument("--name", default="agency")
    parser.add_argument("--date-col", default="")
    parser.add_argument("--dayfirst", action="store_true", help="dates such as 03/07/2022 are day first")
    parser.add_argument("--regions", nargs="*", default=[], help="values of the index's region column the agency covers")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--effis-extra", nargs="*", default=[])
    parser.add_argument("--index", default="data/v4/index.csv")
    parser.add_argument("--masks-root", default="data/v4")
    parser.add_argument("--start", default="2020-01-01")
    parser.add_argument("--end", default="2026-08-03")
    parser.add_argument("--min-ha", type=float, default=5)
    parser.add_argument("--days-before", type=int, default=15, help="EFFIS date at most this many days before the agency date")
    parser.add_argument("--days-after", type=int, default=30, help="and at most this many after it")
    parser.add_argument("--bootstrap", type=int, default=1000)
    parser.add_argument("--out", default="validation")
    args = parser.parse_args()

    if args.list_columns:
        for path in args.agency:
            frame = gpd.read_file(path, rows=5)
            print(f"{path}: CRS {frame.crs}\n{frame.drop(columns='geometry').to_string()}\n")
        return
    if not args.date_col:
        raise SystemExit("Give --date-col; --list-columns shows the columns.")
    Path(args.out).mkdir(parents=True, exist_ok=True)
    agency = read_agency(args.agency, args.date_col, args.dayfirst)
    period = (agency["agency_date"] >= args.start) & (agency["agency_date"] <= args.end)
    agency = agency[period].reset_index(drop=True)
    agency["agency_id"] = np.arange(len(agency))
    if agency.empty:
        raise SystemExit("No agency perimeters in the period; check --date-col and --dayfirst.")
    lon_lat = agency.to_crs(4326).total_bounds
    cfg = yaml.safe_load(Path(args.config).read_text())
    effis = load_perimeters(cfg, args.effis_extra)
    effis = effis[effis.intersects(box(*lon_lat).buffer(.05))].drop_duplicates("fire_id").to_crs(EQUAL_AREA).reset_index(drop=True)
    effis["geometry"] = effis.geometry.make_valid()
    effis["fire_date"] = naive(effis["fire_date"])
    pairs = match(agency, effis, args.days_before, args.days_after)
    agency["matched"] = agency["agency_id"].isin(pairs["agency_id"])
    agency["size"] = pd.cut(agency["agency_ha"], SIZE_BINS, right=False, labels=SIZE_LABELS)
    first, last = agency["agency_date"].min(), agency["agency_date"].max()

    lines = [f"# EFFIS and burn2scar against {args.name} perimeters", "",
             f"- Agency perimeters {first.date()} to {last.date()}: {len(agency):,}; at least {args.min_ha:g} ha: {int((agency['agency_ha'] >= args.min_ha).sum()):,}",
             f"- EFFIS perimeters in the area: {len(effis):,}. Match: overlapping, EFFIS dated {args.days_before} days before to "
             f"{args.days_after} days after the agency date.", ""]

    rows = [{"Size": s, "Agency fires": len(part), "Matched by EFFIS %": wilson(int(part["matched"].sum()), len(part))}
            for s, part in agency.groupby("size", observed=True)]
    eligible = agency[agency["agency_ha"] >= args.min_ha]
    rows.append({"Size": f"all, at least {args.min_ha:g} ha", "Agency fires": len(eligible),
                 "Matched by EFFIS %": wilson(int(eligible["matched"].sum()), len(eligible))})
    lines += ["## 1. Detection: agency fires that EFFIS mapped", "", md(pd.DataFrame(rows)), ""]

    scored = event_scores(agency, effis, pairs)
    if len(scored):
        scored["tp"], scored["fp"], scored["fn"] = scored["both_ha"], scored["effis_ha"] - scored["both_ha"], scored["agency_ha"] - scored["both_ha"]
        scored["size"] = pd.cut(scored["agency_ha"], SIZE_BINS, right=False, labels=SIZE_LABELS)
        scored["event_iou"] = scored["both_ha"] / (scored["agency_ha"] + scored["effis_ha"] - scored["both_ha"])
        rows = [{"Size": "all", "Events": len(scored), **with_interval(scored, "tp", "fp", "fn", args.bootstrap),
                 "Median event IoU": f"{100 * scored['event_iou'].median():.1f}"}]
        for s, part in scored.groupby("size", observed=True):
            rows.append({"Size": s, "Events": len(part), **with_interval(part, "tp", "fp", "fn", args.bootstrap),
                         "Median event IoU": f"{100 * part['event_iou'].median():.1f}"})
        lines += ["## 2. Outline: EFFIS against the agency perimeter, matched events", "",
                  "Area-weighted precision, recall, F1 and IoU in percent, with bootstrap intervals over events; sizes are the agency area.", "",
                  md(pd.DataFrame(rows).rename(columns={"precision": "Precision", "recall": "Recall", "f1": "F1", "iou": "IoU"})), ""]
        scored.to_csv(Path(args.out) / f"agency_{args.name.lower()}_events.csv", index=False)

    index = pd.read_csv(args.index, low_memory=False)
    if args.regions:
        released = index[index["split"].notna() & index["region"].isin(args.regions)].drop_duplicates("fire_id")
        dates = pd.to_datetime(released["fire_date"])
        released = released[(dates >= pd.Timestamp(first.year, 1, 1)) & (dates < pd.Timestamp(last.year + 1, 1, 1))]   # whole years the file covers
        unmatched = released[~released["fire_id"].isin(pairs["fire_id"])]
        lines += ["## 3. burn2scar fires with no agency perimeter", "",
                  f"Released fires in {', '.join(args.regions)} in the years the agency file covers: {len(released):,}; "
                  f"with no agency perimeter: {wilson(len(unmatched), len(released))} %."]
        for column in ("land_cover", "agricultural", "winter_fire"):
            if column in released and len(released):
                share = unmatched.groupby(column).size() / released.groupby(column).size()
                lines.append(f"- By {column.replace('_', ' ')}: " + ", ".join(f"{k} {100 * v:.0f} %" for k, v in share.dropna().items()))
        lines.append("")

    tiles = tile_scores(index, args.masks_root, agency, set(pairs["fire_id"]))
    if len(tiles):
        tiles["size"] = pd.cut(tiles["area_ha"], SIZE_BINS, right=False, labels=SIZE_LABELS)
        rows = [{"Size": "all", "Tiles": len(tiles), **with_interval(tiles, "tp", "fp", "fn", args.bootstrap)}]
        for s, part in tiles.groupby("size", observed=True):
            rows.append({"Size": s, "Tiles": len(part), **with_interval(part, "tp", "fp", "fn", args.bootstrap)})
        lines += ["## 4. burn2scar fresh_burn labels against the agency perimeters", "",
                  "Released fresh-window tiles of matched fires; reference: agency perimeters dated 0-90 days before the scene. "
                  "Sizes are the EFFIS size of the tile's fire. Bootstrap intervals over tiles.", "",
                  md(pd.DataFrame(rows).rename(columns={"precision": "Precision", "recall": "Recall", "f1": "F1", "iou": "IoU"})), ""]
        tiles.to_csv(Path(args.out) / f"agency_{args.name.lower()}_tiles.csv", index=False)
    text = "\n".join(lines)
    Path(args.out).mkdir(parents=True, exist_ok=True)
    (Path(args.out) / f"agency_{args.name.lower()}.md").write_text(text)
    print(text)


if __name__ == "__main__":
    main()
