#!/usr/bin/env python3
"""The burn evidence as a result: threshold sensitivity, the background it is read against, size and country together,
season and land cover, and how long the scar stays detectable (old-window tiles).

  python tools/dnbr_analysis.py --index data/v4/index.csv --fresh validation/dnbr --old validation/dnbr_old \
      --out validation/dnbr_analysis.md

--fresh and --old are output folders of burn_evidence.py (fresh-window tiles, and --window old). Size bins are the
review's, so the two can be read side by side.
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

SIZE_BINS, SIZE_LABELS = [0, 10, 30, 100, 500, np.inf], ["5-10 ha", "10-30 ha", "30-100 ha", "100-500 ha", "500-2000 ha"]
SEASON = {12: "Dec-Feb", 1: "Dec-Feb", 2: "Dec-Feb", 3: "Mar-May", 4: "Mar-May", 5: "Mar-May",
          6: "Jun-Aug", 7: "Jun-Aug", 8: "Jun-Aug", 9: "Sep-Nov", 10: "Sep-Nov", 11: "Sep-Nov"}
THRESHOLDS = [0.05, 0.10, 0.15, 0.20, 0.27]


def md(frame):
    head = "| " + " | ".join(map(str, frame.columns)) + " |"
    rule = "|" + "---|" * len(frame.columns)
    body = ["| " + " | ".join("" if pd.isna(v) else str(v) for v in row) + " |" for row in frame.itertuples(index=False)]
    return "\n".join([head, rule, *body])


def load(folder, index):
    rows = pd.read_csv(Path(folder) / "burn_evidence.csv").drop_duplicates("tile_id", keep="last")
    keep = [c for c in ["tile_id", "country", "area_ha", "fire_date", "age_days", "land_cover"] if c in index]
    rows = rows.merge(index[keep], on="tile_id", how="left")
    rows["size"] = pd.cut(rows["area_ha"], SIZE_BINS, right=False, labels=SIZE_LABELS)
    rows["season"] = pd.to_datetime(rows["fire_date"], errors="coerce").dt.month.map(SEASON)
    return rows, rows[rows["flag"].isin(["burn", "weak", "none"])].copy()


def flag_at(frame, threshold):
    no_ring = frame["contrast"].isna()
    return (frame["dnbr_in_median"] >= threshold) & (no_ring | (frame["contrast"] >= threshold))


def table_by(frame, by):
    groups = frame.groupby(by, observed=True)
    out = pd.DataFrame({"Fires": groups.size(),
                        "Burn %": (100 * groups["flag"].apply(lambda s: (s == "burn").mean())).round(1),
                        "Burn or weak %": (100 * groups["flag"].apply(lambda s: s.isin(["burn", "weak"]).mean())).round(1),
                        "Median dNBR inside": groups["dnbr_in_median"].median().round(3),
                        "Median contrast": groups["contrast"].median().round(3)})
    return out.reset_index()


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--index", default="data/v4/index.csv")
    parser.add_argument("--fresh", default="validation/dnbr")
    parser.add_argument("--old", default="validation/dnbr_old")
    parser.add_argument("--out", default="validation/dnbr_analysis.md")
    args = parser.parse_args()
    index = pd.read_csv(args.index, low_memory=False)
    rows, fresh = load(args.fresh, index)
    lines = ["# Burn evidence: analysis", "", f"Fresh-window tiles assessed: {len(fresh):,} of {len(rows):,}.", ""]

    # 1. The background: what dNBR does on clear land next to the fires.
    ring = fresh["dnbr_ring_median"].dropna()
    q = ring.quantile([.5, .9, .95, .99])
    pooled = fresh["ring_010"].sum() / max(fresh["n_ring"].sum(), 1)
    lines += ["## The background: clear land 100-300 m outside each perimeter", "",
              f"- Tiles with a usable ring: {len(ring):,}",
              f"- Ring median dNBR, percentiles 50 / 90 / 95 / 99: {q[.5]:.3f} / {q[.9]:.3f} / {q[.95]:.3f} / {q[.99]:.3f}",
              f"- Ring pixels at or above 0.10, pooled: {100 * pooled:.1f} %",
              f"- Tiles whose ring median is at or above 0.10: {100 * (ring >= 0.10).mean():.1f} %", ""]

    # 2. Threshold sensitivity, overall and by size.
    rows_t = []
    for threshold in THRESHOLDS:
        hit = flag_at(fresh, threshold)
        row = {"Threshold": threshold, "All fires, burn %": round(100 * hit.mean(), 1),
               "Rings above it %": round(100 * (ring >= threshold).mean(), 1)}
        for label in SIZE_LABELS:
            part = fresh["size"] == label
            row[label] = round(100 * hit[part].mean(), 1) if part.any() else np.nan
        rows_t.append(row)
    lines += ["## Flag 'burn' at other thresholds", "",
              "Burn: median dNBR inside and contrast with the ring both at or above the threshold. 0.10 is the lower bound of "
              "Key and Benson's low severity; 0.27 of moderate-low. 'Rings above it' is how often clear land alone would pass.", "",
              md(pd.DataFrame(rows_t)), ""]

    # 3. Size, and size within country, which separates a country effect from a size effect.
    lines += ["## By fire size (the review's bins)", "", md(table_by(fresh, "size")), ""]
    by_country = fresh.groupby("country")
    mix = pd.DataFrame({"Fires": by_country.size(), "Median size ha": by_country["area_ha"].median().round(0),
                        "Under 30 ha %": (100 * by_country["area_ha"].apply(lambda s: (s < 30).mean())).round(1),
                        "Burn %": (100 * by_country["flag"].apply(lambda s: (s == "burn").mean())).round(1)}).reset_index()
    cross = (100 * fresh.assign(burn=fresh["flag"] == "burn").pivot_table(index="country", columns="size", values="burn",
                                                                         aggfunc="mean", observed=True)).round(1).reset_index()
    lines += ["## Country, and size within country", "", md(mix), "", "Burn % by country and size:", "", md(cross), ""]

    # 4. Season and land cover.
    for by, title in (("season", "Season of the fire"), ("land_cover", "Land cover")):
        if by in fresh and fresh[by].notna().any():
            lines += [f"## {title}", "", md(table_by(fresh, by)), ""]

    # 5. Old-window tiles: is the scar still there months later?
    old_path = Path(args.old) / "burn_evidence.csv"
    if old_path.exists():
        rows_o, old = load(args.old, index)
        old["months"] = pd.cut(old["age_days"], [0, 150, 225, 301, np.inf], right=False,
                               labels=["120-150 d", "150-225 d", "225-300 d", "over 300 d"])
        detectable = (old["dnbr_in_median"] >= 0.10)
        lines += ["## Old-window tiles: the same perimeters months later", "",
                  f"Assessed: {len(old):,} of {len(rows_o):,}. dNBR here is from the pre-fire image to the old-window scene, "
                  "inside the perimeter (old_burn pixels), so it measures whether the scar is still detectable.", "",
                  f"- Inside median at or above 0.10: {100 * detectable.mean():.1f} %; flagged burn (also above the ring): "
                  f"{100 * (old['flag'] == 'burn').mean():.1f} %", ""]
        lines += [md(table_by(old, "months")), "", md(table_by(old, "size")), ""]
        both = fresh[["fire_id", "flag"]].merge(old[["fire_id", "flag"]], on="fire_id", suffixes=("_fresh", "_old"))
        if len(both):
            lines += ["Fires assessed in both windows: flag in the fresh window (rows) against the old window (columns), fires:", "",
                      md(pd.crosstab(both["flag_fresh"], both["flag_old"]).reset_index()), ""]
    text = "\n".join(lines)
    Path(args.out).write_text(text)
    print(text)


if __name__ == "__main__":
    main()
