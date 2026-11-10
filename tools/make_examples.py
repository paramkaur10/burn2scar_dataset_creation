"""Copy one burn2scar fire into examples/, so the repository has real data to try right after cloning.

  python tools/make_examples.py --index data/v3/index.csv --images-root data/v1 --masks-root data/v3

Picks, with a fixed seed, one fire from the test set of the main split (column split), so the released model
never trained on it. The fire has both scenes: the fresh one with at least 15 % fresh burn, the old one with at
least 10 % old burn, and both with under 2 % cloud, shadow or no data. Copies the two images and two masks,
about 4 MB, and writes examples/index.csv (the fire's two rows of the release index, with paths into examples/)
and examples/README.md.

The tiles are part of the burn2scar release and keep its licence, CC BY 4.0 with the Copernicus and EFFIS
notices, not the code's Apache 2.0. examples/README.md says so.
"""
import argparse
import shutil
from pathlib import Path

import pandas as pd

COUNTRY = {"IT": "Italy", "ES": "Spain", "FR": "France", "EL": "Greece"}
DOI = "https://doi.org/10.5281/zenodo.23986827"


def choose(index, seed):
    test = index[index["split"] == "test"]
    hidden = test["frac_cloud"] + test["frac_shadow"] + test["frac_nodata"]
    fresh = test[(test["window"] == "fresh") & (test["frac_fresh"] >= .15) & (hidden < .02)]
    old = test[(test["window"] == "old") & (test["frac_old"] >= .10) & (hidden < .02)]
    fires = sorted(set(fresh["fire_id"]) & set(old["fire_id"]))
    if not fires:
        raise SystemExit("No test fire has both scenes clear enough. Loosen the thresholds in choose().")
    fire = pd.Series(fires).sample(1, random_state=seed).iloc[0]
    return pd.concat([fresh[fresh["fire_id"] == fire].iloc[:1], old[old["fire_id"] == fire].iloc[:1]])


def readme(rows):
    first = rows.iloc[0]
    lines = [
        "# Example tiles",
        "",
        f"One fire from the test set of burn2scar's main split, so the released model never saw it in training: "
        f"EFFIS fire `{first.fire_id}`, {COUNTRY.get(first.country, first.country)}, {first.area_ha:.0f} ha, "
        f"dated {str(first.fire_date)[:10]}. Two Sentinel-2 Level-1C scenes of it, each with its mask.",
        "",
        "| Image | Mask | Scene | Days after the fire | fresh_burn | old_burn | older_burn |",
        "|---|---|---|---|---|---|---|",
    ]
    for row in rows.itertuples():
        lines.append(f"| `{Path(row.image_path).name}` | `{Path(row.mask_path).name}` | {str(row.scene_time)[:10]} | "
                     f"{row.age_days:.0f} | {row.frac_fresh:.0%} | {row.frac_old:.0%} | {row.frac_older:.0%} |")
    image = Path(rows.iloc[0].image_path).name
    lines += [
        "",
        "Images: 13 bands (B01 to B12, with B8A after B08), top-of-atmosphere reflectance x 10,000, uint16, 256 x 256 "
        "pixels at 10 m. Masks: 0 clear, 2 fresh_burn, 3 old_burn, 4 cloud, 5 cloud shadow, 6 water, 7 older_burn, "
        "255 no data. `index.csv` holds the two rows of the release index.",
        "",
        "Run the baseline model on one of them, from the repository root:",
        "",
        "```bash",
        f"python tools/predict.py --image examples/{image} --out predictions/{image.replace('_s2.tif', '_pred.tif')}",
        "```",
        "",
        "## Licence",
        "",
        f"These files are part of [burn2scar]({DOI}) and are licensed under CC BY 4.0, not under the code's Apache "
        "2.0. Credit:",
        "",
        "```",
        "Contains modified Copernicus Sentinel data 2020-2026.",
        "Burn labels adapted from EFFIS, Copernicus Emergency Management Service,",
        "© European Union, 1995-2026, CC BY 4.0.",
        "burn2scar: Thind, P. K., Cadau, E., Del Prete, R., Ricolfi, S., Meoni, G., and Taramelli, A., 2026,",
        "https://doi.org/10.5281/zenodo.23986827",
        "```",
        "",
    ]
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--index", default="data/v3/index.csv")
    parser.add_argument("--images-root", default="data/v1")
    parser.add_argument("--masks-root", default="data/v3")
    parser.add_argument("--out", default="examples")
    parser.add_argument("--seed", type=int, default=3)
    args = parser.parse_args()

    rows = choose(pd.read_csv(args.index, low_memory=False), args.seed).copy()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    for name in out.glob("*.tif"):
        name.unlink()  # an earlier choice
    for i, row in rows.iterrows():
        for column, root in (("image_path", args.images_root), ("mask_path", args.masks_root)):
            source = Path(root) / row[column]
            shutil.copy(source, out / source.name)
            rows.at[i, column] = source.name
    rows.to_csv(out / "index.csv", index=False)
    (out / "README.md").write_text(readme(rows))
    size = sum(p.stat().st_size for p in out.glob("*.tif")) / 1e6
    print(f"Wrote {out}: fire {rows.iloc[0].fire_id} ({rows.iloc[0].country}), scenes at "
          + " and ".join(f"day {d:.0f}" for d in rows["age_days"]) + f", {size:.1f} MB of GeoTIFFs, index.csv, README.md.")


if __name__ == "__main__":
    main()
