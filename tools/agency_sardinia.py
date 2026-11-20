#!/usr/bin/env python3
"""Download the fire perimeters of Sardinia's forest corps (CFVA, CC BY 4.0) and compare them with EFFIS and the labels.

  python tools/agency_sardinia.py

1. Downloads areeIncendiatePerim<year>.zip for 2020-2025 from the Sardinian geoportal into data/agency/sardinia
   (years that are not online are reported and skipped).
2. Finds each year's fire-date column (the column whose values read best as dates of that year), puts every year in one
   file, data/agency/sardinia/cfva_perimeters.gpkg, with the date in a column called cfva_date.
3. Runs compare_agency.py on it for the region Sardegna; the result is validation/agency_sardinia.md.
"""
import argparse
import subprocess
import sys
import urllib.request
from pathlib import Path

import geopandas as gpd
import pandas as pd

URL = "https://webgis.regione.sardegna.it/scaricocartografiaETL/cfva/areeIncendiatePerim{year}.zip"
ROOT = Path(__file__).resolve().parents[1]


def parse_dates(values, dayfirst):
    """Dates from a column: datetimes as they are; ISO strings (2022-07-03) always year-month-day, because pandas with
    dayfirst=True silently reads 2022-07-03 as 7 March; other strings day first or month first as asked."""
    if pd.api.types.is_datetime64_any_dtype(values):
        dates = pd.to_datetime(values)
        return dates.dt.tz_localize(None) if dates.dt.tz is not None else dates
    raw = values.astype(str).str.strip()
    iso = raw.str.match(r"^\d{4}-\d{2}-\d{2}")
    dates = pd.Series(pd.NaT, index=raw.index, dtype="datetime64[ns]")
    if iso.any():
        dates[iso] = pd.to_datetime(raw[iso].str[:10], errors="coerce", format="%Y-%m-%d")
    if (~iso).any():
        dates[~iso] = pd.to_datetime(raw[~iso], errors="coerce", dayfirst=dayfirst, format="mixed")
    return dates


def date_column(frame, year):
    """The column whose values parse as dates of that year most often, and whether they are day first. A column of years
    (such as ANNO), or one with only a few distinct days in the year, is not a fire date and is skipped; between columns
    that read equally well, the one with more distinct days wins. Day first, the Italian order, unless month first reads
    strictly better."""
    best, best_share, best_dayfirst, best_days = None, 0.0, True, 0
    for column in frame.columns:
        if column == "geometry":
            continue
        raw = frame[column].astype(str).str.strip()
        if raw.str.fullmatch(r"\d{4}(\.0+)?").mean() > .5:
            continue
        for dayfirst in (True, False):
            dates = parse_dates(frame[column], dayfirst)
            inside = dates.dt.year == year
            share = float(inside.mean())
            days = int(dates[inside].dt.dayofyear.nunique())
            if days < 10 or (not dayfirst and column == best and share <= best_share):
                continue
            if share > best_share or (share == best_share and days > best_days):
                best, best_share, best_dayfirst, best_days = column, share, dayfirst, days
    return best, best_share, best_dayfirst


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--years", nargs="+", type=int, default=list(range(2020, 2026)))
    parser.add_argument("--folder", default="data/agency/sardinia")
    parser.add_argument("--region", default="Sardegna", help="value of the index's region column")
    args = parser.parse_args()
    folder = Path(args.folder)
    folder.mkdir(parents=True, exist_ok=True)
    parts = []
    for year in args.years:
        path = folder / f"areeIncendiatePerim{year}.zip"
        if not path.exists():
            try:
                urllib.request.urlretrieve(URL.format(year=year), path.with_suffix(".part"))
                path.with_suffix(".part").replace(path)
            except Exception as error:
                print(f"{year}: not downloaded ({type(error).__name__}: {error})", flush=True)
                path.with_suffix(".part").unlink(missing_ok=True)
                continue
        frame = gpd.read_file(f"zip://{path}")
        column, share, dayfirst = date_column(frame, year)
        if column is not None:
            check = parse_dates(frame[column], dayfirst)
            span = f"{check.min():%Y-%m-%d} to {check.max():%Y-%m-%d}, {check.dt.dayofyear.nunique()} distinct days"
        else:
            span = "none"
        print(f"{year}: {len(frame):,} perimeters; columns {', '.join(c for c in frame.columns if c != 'geometry')}; "
              f"date column {column} ({100 * share:.0f} % read as {year}; {span}; day first {dayfirst})", flush=True)
        if column is None or share < .5:
            print(f"{year}: no date column found, left out. Send the line above.", flush=True)
            continue
        dates = parse_dates(frame[column], dayfirst)
        parts.append(gpd.GeoDataFrame({"cfva_date": dates.dt.strftime("%Y-%m-%d"), "cfva_year": year},
                                      geometry=frame.geometry.to_crs(3035), crs=3035))
    if not parts:
        raise SystemExit("No year could be read.")
    merged = pd.concat(parts, ignore_index=True)
    merged = gpd.GeoDataFrame(merged, geometry="geometry", crs=3035)
    target = folder / "cfva_perimeters.gpkg"
    merged.to_file(target, driver="GPKG")
    years = sorted(merged["cfva_year"].unique())
    print(f"Wrote {target}: {len(merged):,} perimeters, years {years[0]}-{years[-1]}", flush=True)
    end = f"{years[-1]}-12-31" if years[-1] < 2026 else "2026-08-03"
    subprocess.run([sys.executable, str(ROOT / "tools" / "compare_agency.py"), "--agency", str(target), "--name", "Sardinia",
                    "--date-col", "cfva_date", "--regions", args.region, "--start", f"{years[0]}-01-01", "--end", end,
                    "--out", "validation"], check=True)


if __name__ == "__main__":
    main()
