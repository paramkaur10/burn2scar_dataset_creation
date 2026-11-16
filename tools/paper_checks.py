#!/usr/bin/env python3
"""Numbers the paper still needs from the index and the EFFIS export. Reads only; changes nothing.

Run from the repo root:
  python tools/paper_checks.py --index data/v4/index.csv --config config.yaml --out validation/paper_checks.md

1. Coverage: the kept fires as a share of the EFFIS fires and burnt area of the build export, by country, and
   what the size limits leave out (paper Sect. 3.1).
2. Fresh scenes taken before the recorded end of their fire: age_days < fire_duration_days (Sect. 5.5).
3. Burn evidence by days from the fire to the fresh scene, from dnbr_flag (Sect. 5.5).
4. older_burn share of valid pixels by scene year (Sect. 5.5).
5. The core subset, fires with fire_dnbr_flag == burn: fires and tiles per split (Sect. 7).
"""
import argparse
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
warnings.filterwarnings("ignore", message=".*geographic CRS.*")

ASSESSED = ["burn", "weak", "none"]


def fmt(value):
    if isinstance(value, (float, np.floating)):
        return "" if np.isnan(value) else f"{value:,.1f}"
    if isinstance(value, (int, np.integer)):
        return f"{value:,}"
    return str(value)


def table(df):
    """A DataFrame as a markdown table (no tabulate needed)."""
    lines = ["| " + " | ".join(str(c) for c in df.columns) + " |", "|" + "---|" * len(df.columns)]
    lines += ["| " + " | ".join(fmt(v) for v in row) + " |" for row in df.itertuples(index=False)]
    return "\n".join(lines)


def needs(index, columns, out):
    missing = sorted(set(columns) - set(index.columns))
    if missing:
        out.append(f"Skipped: the index has no column {', '.join(missing)}.")
    return not missing


def coverage(index, cfg, out):
    from src.effis_loader import load_effis

    effis = load_effis(cfg["effis_path"])
    countries = {c.upper() for c in cfg["countries"]}
    start, end = pd.Timestamp(cfg["start_date"]), pd.Timestamp(cfg["end_date"])
    effis = effis[effis["country"].isin(countries) & effis["fire_date"].between(start, end)].copy()
    out.append(f"Build export {cfg['effis_path']}: {len(effis):,} perimeters in {sorted(countries)}, "
               f"{start:%Y-%m-%d} to {end:%Y-%m-%d} (the paper says 25,707).")

    low, high = cfg["min_area_ha"], cfg.get("max_area_ha", np.inf)
    area = effis["area_ha"].fillna(0)
    group = pd.Series("eligible, not in the release", index=effis.index)
    group[area < low] = f"under {low:g} ha"
    group[area > high] = f"over {high:g} ha"
    north = (effis["country"] == "FR") & (effis.geometry.centroid.y > cfg["southern_france_max_lat"])
    group[north & area.between(low, high)] = "France north of the limit"
    kept_ids = set(index["fire_id"])
    group[effis["fire_id"].isin(kept_ids)] = "kept"
    effis["group"] = group

    def summary(df):
        g = df.groupby("group").agg(fires=("fire_id", "size"), area_ha=("area_ha", "sum"))
        g["% of fires"] = 100 * g["fires"] / len(df)
        g["% of area"] = 100 * g["area_ha"] / df["area_ha"].sum()
        g.loc["all"] = [len(df), df["area_ha"].sum(), 100.0, 100.0]
        g["fires"] = g["fires"].astype(int)
        return g.reset_index()

    out.append("\nAll four countries:\n")
    out.append(table(summary(effis)))
    rows = []
    for country, part in effis.groupby("country"):
        total = part["area_ha"].sum()
        kept = part[part["group"] == "kept"]
        big = part[part["group"].str.startswith("over")]
        rows.append({"country": country, "perimeters": len(part), "burnt area (ha)": total,
                     "kept fires %": 100 * len(kept) / len(part), "kept area %": 100 * kept["area_ha"].sum() / total,
                     f"area in fires over {high:g} ha %": 100 * big["area_ha"].sum() / total})
    out.append("\nBy country:\n")
    out.append(table(pd.DataFrame(rows)))
    missing = len(kept_ids - set(effis["fire_id"]))
    if missing:
        out.append(f"\nCheck: {missing:,} fires of the index are not in this export; is effis_path the export of 4 August?")


def early_scenes(index, out):
    if not needs(index, ["window", "age_days", "fire_duration_days"], out):
        return
    fresh = index[index["window"] == "fresh"]
    age = pd.to_numeric(fresh["age_days"], errors="coerce")
    duration = pd.to_numeric(fresh["fire_duration_days"], errors="coerce")
    early = age < duration
    out.append(f"Fresh-window tiles: {len(fresh):,}, {duration.notna().sum():,} with a recorded duration.")
    out.append(f"Recorded duration, days: median {duration.median():.2f}, 90th percentile {duration.quantile(0.9):.2f}, "
               f"99th {duration.quantile(0.99):.2f}, maximum {duration.max():.1f}; over 5 days for "
               f"{(duration > 5).sum():,} tiles ({100 * (duration > 5).mean():.2f} %).")
    out.append(f"Scene earlier than the recorded end of its fire: {early.sum():,} tiles "
               f"({100 * early.mean():.2f} % of fresh-window tiles).")
    if early.any():
        out.append(f"Those tiles: median scene day {age[early].median():.1f}, "
                   f"median recorded duration {duration[early].median():.1f} days.")


def evidence_by_day(index, out):
    if not needs(index, ["window", "age_days", "dnbr_flag"], out):
        return
    fresh = index[(index["window"] == "fresh") & index["dnbr_flag"].isin(ASSESSED)].copy()
    age = pd.to_numeric(fresh["age_days"], errors="coerce")
    fresh["days"] = pd.cut(age, [0, 10, 20, 30, 45, 62], right=False,
                           labels=["under 10", "10-20", "20-30", "30-45", "45-61"])
    groups = fresh.groupby("days", observed=True)["dnbr_flag"]
    result = pd.DataFrame({"tiles": groups.size(),
                           "burn %": 100 * groups.apply(lambda s: (s == "burn").mean()),
                           "burn or weak %": 100 * groups.apply(lambda s: s.isin(["burn", "weak"]).mean())})
    out.append("Fresh-window tiles with a flag, by days from the EFFIS fire date to the scene:\n")
    out.append(table(result.reset_index()))
    if "area_ha" in fresh.columns:
        size = pd.cut(pd.to_numeric(fresh["area_ha"], errors="coerce"), [0, 30, 100, np.inf], right=False,
                      labels=["under 30 ha", "30-100 ha", "over 100 ha"])
        burn = (fresh["dnbr_flag"] == "burn").astype(float) * 100
        pivot = burn.groupby([fresh["days"], size], observed=True).mean().unstack()
        out.append("\nShare flagged burn (%), days by fire size, so that size does not explain the trend:\n")
        out.append(table(pivot.reset_index()))


def older_by_year(index, out):
    if not needs(index, ["scene_year", "frac_older", "frac_nodata"], out):
        return
    whole = 100.0 if index.get("frac_clear", pd.Series([0.0])).max() > 1.5 else 1.0   # shares or percentages
    valid = whole - pd.to_numeric(index["frac_nodata"], errors="coerce")
    frame = index.assign(valid=valid, older=pd.to_numeric(index["frac_older"], errors="coerce"))
    groups = frame.groupby(frame["scene_year"].astype(str))
    result = pd.DataFrame({"tiles": groups.size(),
                           "older_burn % of valid pixels": 100 * groups["older"].sum() / groups["valid"].sum()})
    out.append(table(result.reset_index()))
    overall = 100 * frame["older"].sum() / frame["valid"].sum()
    out.append(f"\nAll scenes: {overall:.2f} % (Table 3 of the paper says 8.24 %; if this differs, the frac columns "
               "are not shares of all pixels).")


def core_subset(index, out):
    if not needs(index, ["fire_dnbr_flag", "split"], out):
        return
    for name, flags in (("burn (the core subset)", ["burn"]), ("burn or weak", ["burn", "weak"])):
        chosen = index[index["fire_dnbr_flag"].isin(flags)]
        counts = chosen["split"].value_counts()
        out.append(f"fire_dnbr_flag {name}: {chosen['fire_id'].nunique():,} fires, {len(chosen):,} tiles; spatial "
                   "split " + ", ".join(f"{part} {counts.get(part, 0):,}" for part in ["train", "val", "test"]) + ".")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--index", default="data/v4/index.csv")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--out", default="validation/paper_checks.md")
    parser.add_argument("--skip-effis", action="store_true", help="skip part 1, which reads the EFFIS export")
    args = parser.parse_args()

    index = pd.read_csv(args.index, low_memory=False)
    index["fire_id"] = index["fire_id"].astype(str)
    out = [f"# Numbers for the paper\n\nIndex {args.index}: {len(index):,} tiles, {index['fire_id'].nunique():,} fires."]
    if "split" in index.columns and index["split"].isna().any():   # the early-run tiles that are not released
        unreleased = index["split"].isna()
        out.append(f"Left out {unreleased.sum():,} tiles of {index.loc[unreleased, 'fire_id'].nunique():,} fires that have "
                   f"no split and are not released; {(~unreleased).sum():,} tiles remain.")
        index = index[~unreleased].copy()
    parts = [("2. Fresh scenes before the recorded end of the fire (Sect. 5.5)", early_scenes),
             ("3. Burn evidence by days after the fire (Sect. 5.5)", evidence_by_day),
             ("4. older_burn by scene year (Sect. 5.5)", older_by_year),
             ("5. Core subset (Sect. 7)", core_subset)]
    if not args.skip_effis:
        cfg = yaml.safe_load(Path(args.config).read_text())
        parts.insert(0, ("1. Coverage of the EFFIS export (Sect. 3.1)", lambda idx, o: coverage(idx, cfg, o)))
    for title, run in parts:
        out.append(f"\n## {title}\n")
        try:
            run(index, out)
        except Exception as error:   # one failed part should not hide the others
            out.append(f"Failed: {type(error).__name__}: {error}")
    text = "\n".join(out) + "\n"
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(text)
    print(text)
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()