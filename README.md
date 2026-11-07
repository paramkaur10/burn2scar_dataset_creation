# burn2scar

A Sentinel-2 burn-scar segmentation dataset built from real, unsimulated satellite
imagery paired with EFFIS-derived labels. This repository holds the code that made it.
The dataset itself is on Zenodo, [doi.org/10.5281/zenodo.23986827](https://doi.org/10.5281/zenodo.23986827),
with a mirror at [huggingface.co/datasets/neet1797/burn2scar](https://huggingface.co/datasets/neet1797/burn2scar).

---

## What this is

For each wildfire in the EFFIS (European Forest Fire Information System) record,
this pipeline finds the best available Sentinel-2 scene and produces:

- A **13-band Sentinel-2 L1C image** (top-of-atmosphere reflectance, all bands on a 10 m grid)
- A **segmentation mask** with seven classes plus no data, at the same resolution and extent

No sensor simulation is applied. Bands are Sentinel-2's own native radiometry,
resampled to a common 10 m grid (unavoidable, since S2's 13 bands natively span
10 m / 20 m / 60 m) and cropped to a fixed tile around each fire. No atmospheric
correction, denoising, or synthetic degradation of any kind is applied.

---

## Classes

| Mask value | Class | Definition |
|---|---|---|
| 0 | clear | No burn, cloud, shadow, or water detected |
| 2 | fresh burn | Burned area, ≤ 90 days since the EFFIS fire date |
| 3 | old burn | Burned area, > 90 days since the EFFIS fire date (up to 365 days) |
| 7 | older burn | Burned area, > 365 days since the EFFIS fire date; only where the pixel would otherwise be clear |
| 4 | cloud | Detected via [OmniCloudMask](https://github.com/DPIRD-DMA/OmniCloudMask) |
| 5 | cloud shadow | Detected via OmniCloudMask |
| 6 | water | NDWI > 0.05, or Sentinel-2 Scene Classification (SCL) value 6; in November to February scenes, kept only where the fire's March to October scene agrees |
| 255 | nodata | Outside the valid data mask (tile edge / sensor gap) |

The burn classes come from every EFFIS fire-perimeter polygon that touches the tile,
distinguished purely by the number of days between the fire's recorded date and the
satellite acquisition date. Version 1 on Hugging Face had no older burn class and took
burn only from the selected fires; `tools/burn_labels.py` and `tools/fix_winter_water.py`
turn version 1 masks into the version 3.0 masks on Zenodo.

---

## Label purity notes

`clear` is a residual class — anything not detected as burn, cloud, shadow, or
water — not a confirmed "vegetation" or "undisturbed land" label. Depending on
your use case this matters: Mediterranean terrain in particular can put bare
soil, exposed rock, and volcanic terrain (present in both Italy and Greece,
two of this dataset's source countries) into `clear`, since the pipeline has no
separate class for them. If your application needs a purer vegetation mask,
combine this dataset with a land-cover product (e.g. ESA WorldCover) and filter
`clear` further based on your own criteria.

Burn labels themselves are only as complete as EFFIS's own mapping. EFFIS states
no single minimum mapping unit: it maps fires of about 30 ha or larger from MODIS
and, since 2018, smaller fires as well from Sentinel-2. Fires EFFIS missed appear
as `clear` in this dataset. EFFIS dates may also differ from the actual ignition
date, which matters most near the 90-day line between `fresh_burn` and `old_burn`.

---

## Two scenes per fire

Each fire gets **two independently-acquired Sentinel-2 scenes**, searched in
separate time windows:

- **Fresh window**: 5–60 days after the fire's ignition date
- **Old window**: 120–300 days after ignition

Both windows are searched independently, starting from the least cloudy scene, for one
where burned pixels in the tile are at most 50% obscured by cloud or shadow. If no
candidate passes, the least obscured one is kept. The index marks tiles where the
fire's own scar is over that limit (`scene_rule_pass`). This means most tile pairs will show a fresh_burn-dominant mask
(early acquisition) and a separate old_burn-dominant mask (later acquisition) for
the same underlying fire — not necessarily both classes within a single image.

**Known limitation:** because both acquisition windows are strictly *post-fire*,
old_burn labels reflect either (a) the fire the tile is centered on, once enough
time has passed, or (b) incidental overlap with a different, genuinely older fire's
perimeter. There is no pre-fire baseline image in this dataset — every scene is
captured after the relevant fire(s) already occurred.

---

## Fire selection

Fires are drawn from an EFFIS export, filtered by:

| Filter | Value |
|---|---|
| Countries | Italy (IT), France (FR — restricted to south of 46°N latitude), Spain (ES), Greece (EL) |
| Date range | 2020-01-01 to 2026-08-04 |
| Minimum area | 5 hectares (excludes marginal / low-confidence micro-detections) |
| Maximum area | 2,000 hectares (excludes mega-fires large enough to make a single tile 100% burn scar, with no useful boundary signal) |

**19,443 fires** queued after filtering. Fires ran from the largest down, each country
within its share of the budget, so in Spain and France the smallest fires weren't reached.
The released dataset has **13,895 fires** and **26,846 scene/mask pairs**, about 31 GB.
(Eight tiles from an early run have no split and are left out of the release.)

---

## Fire population overview

![Fires overview](figures/fires_overview.png)

Characteristics of the underlying EFFIS-sourced fire population, after filtering
(before acquisition — this is about the *input* fire records, not the output
imagery). Left to right: fire count by source country, distribution of burn
area sizes within the 5–2,000 ha bounds, and seasonal distribution of fires
by quarter across the 2020–2026 date range — wildfire activity concentrates
heavily in the Mediterranean summer months, as expected.

---

## Class distribution

Computed across the 26,846 released masks, version 3.0:

| Class | % of valid pixels |
|---|---|
| clear | 77.26% |
| old_burn | 7.32% |
| fresh_burn | 7.19% |
| older_burn | 6.24% |
| water | 1.28% |
| cloud | 0.54% |
| shadow | 0.19% |

0.95% of all pixels are nodata (tile-edge artifacts from UTM reprojection),
excluded from the percentages above. Fresh and old burn are closely balanced —
a direct result of acquiring two independent time windows per fire (see
"Two scenes per fire" above) rather than relying on incidental overlap with
other fires' perimeters.

---

## Tiles

Each scene is a fixed **2.56 km × 2.56 km** tile (256 × 256 pixels at 10 m),
centered on the fire's centroid, reprojected into the local UTM zone.

---

## Bands

`B01, B02, B03, B04, B05, B06, B07, B08, B8A, B09, B10, B11, B12`

Stored as `uint16`, scale factor 10,000 (divide by 10,000 for TOA reflectance,
matching Sentinel-2's own L1C convention).

---

## Source data

- **Imagery**: Copernicus Sentinel-2, via the Copernicus Data Space Ecosystem
- **Burn labels**: EFFIS (European Forest Fire Information System), Copernicus
 Emergency Management Service
- **Cloud / shadow**: [OmniCloudMask](https://github.com/DPIRD-DMA/OmniCloudMask)
- **Water**: NDWI (green/NIR) + Sentinel-2 Scene Classification Layer (SCL)

---

## Pipeline

```
src/effis_loader.py   — loads and filters EFFIS fire perimeters
src/s2_pipeline.py     — acquisition, masking, GeoTIFF export
src/hf_upload.py       — streaming upload to Hugging Face (download → upload → delete, per fire)
run_extraction.py      — driver: multi-threaded, quota-capped, fully resume-safe
tools/                 — index and splits, label updates, label review, baselines, release
```

`tools/README.md` lists what each tool does and the order to run them in.

Each fire is downloaded, processed, and uploaded to Hugging Face individually, then
deleted locally — so total local disk usage stays roughly flat regardless of final
dataset size.

---

## Setup

```bash
git clone https://github.com/paramkaur10/burn2scar_dataset_creation.git
cd burn2scar_dataset_creation
pip install -r requirements.txt
cp .env.example .env   # fill in your own CDSE + Hugging Face credentials
```

## Run

```bash
python run_extraction.py --config config.yaml
```

Resumable — safe to stop and restart at any time. Already-uploaded fires are
tracked in `output/quota_state.json` and skipped on the next run.

---

## Data

This repository contains no data and no credentials — only the code that produced
the dataset. The dataset itself is at
[doi.org/10.5281/zenodo.23986827](https://doi.org/10.5281/zenodo.23986827), mirrored at
[huggingface.co/datasets/neet1797/burn2scar](https://huggingface.co/datasets/neet1797/burn2scar).

## Licence

The Apache License 2.0 in `LICENSE` covers the code in this repository only. The dataset
has its own terms, set out in the `LICENSE.txt` that comes with it.

## Attribution

```
Contains modified Copernicus Sentinel data 2020-2026.
Contains modified Copernicus Service information 2020-2026.
Burn labels adapted from EFFIS, Copernicus Emergency Management Service,
© European Union, 1995-2026, CC BY 4.0
(https://creativecommons.org/licenses/by/4.0/). Changes: perimeters rasterised
to a 10 m grid, split into fresh, old and older burn classes by date, and
merged with cloud, cloud shadow and water classes.
```

---

**Authors:** Parampuneet Kaur Thind, Enrico Cadau, Roberto Del Prete, Silvia Ricolfi, Gabriele Meoni, Andrea Taramelli