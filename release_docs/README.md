# burn2scar: a Sentinel-2 burn-scar segmentation data set

**Version 4.1** · DOI [10.5281/zenodo.24375951](https://doi.org/10.5281/zenodo.24375951) · version 4.0: [10.5281/zenodo.24221783](https://doi.org/10.5281/zenodo.24221783) · version 3.0: [10.5281/zenodo.23986827](https://doi.org/10.5281/zenodo.23986827)  
Authors: Parampuneet Kaur Thind, Enrico Cadau, Roberto Del Prete, Paola D'Aulerio, Silvia Ricolfi, Gabriele Meoni, Andrea Taramelli  
Licence: CC BY 4.0 for the authors' contributions. Third-party data keep their own terms; see `LICENSE.txt`.  
Code: [github.com/paramkaur10/burn2scar_dataset_creation](https://github.com/paramkaur10/burn2scar_dataset_creation) · Mirror: [huggingface.co/datasets/neet1797/burn2scar](https://huggingface.co/datasets/neet1797/burn2scar)

26,846 Sentinel-2 Level-1C tiles over 13,895 fires mapped by the European Forest Fire Information System (EFFIS)
in Italy, Spain, France south of 46° N and Greece, from 2020 to 2026. Each tile has all 13 bands on a 10 m grid,
256 × 256 pixels (2.56 km across), a mask with seven classes plus no data (clear, fresh burn, old burn, older burn,
water, cloud and cloud shadow) and, new in 4.0, a raster of the days since the fire. Most fires have two tiles of the
same place: one scene 5 to 60 days after the fire, and one 120 to 300 days after. Every fire carries a measure of
whether its burn shows in the imagery, and, new in 4.1, a review of 700 random points by two interpreters and an
adjudicator measures the labels against the imagery.

| Country | Fires | Two scenes | One scene | Tiles | Smallest fire (ha) | Median fire size (ha) |
|---|---:|---:|---:|---:|---:|---:|
| Italy | 8,014 | 7,351 | 663 | 15,365 | 5 | 24 |
| Spain | 2,442 | 2,275 | 167 | 4,717 | 27 | 66 |
| France, south of 46° N | 2,388 | 2,317 | 71 | 4,705 | 17 | 39 |
| Greece | 1,051 | 1,008 | 43 | 2,059 | 5 | 22 |
| All | 13,895 | 12,951 | 944 | 26,846 | | |

Fires by year: 2020 1,721; 2021 2,074; 2022 2,264; 2023 1,998; 2024 1,831; 2025 2,313; 2026 1,694 (to 4 August).

## Files

| File | Content |
|---|---|
| `images_<CC>_<YYYY>.zip` | Images of the fires in country `CC` (IT, ES, FR, EL) that EFFIS dates to year `YYYY`. Unchanged since 1.0 |
| `masks.zip` | All masks, version 4.0 |
| `ages.zip` | New: days from the fire to the scene, per pixel, beside every mask |
| `index.csv` | One row per tile: paths, dates, location, class shares, splits, source products, quality flags and burn evidence |
| `burn_evidence.csv` | New: the dNBR burn evidence of every fire, on its fresh-window tile |
| `burn_evidence_old_window.csv` | New: the same test on the old-window tiles, months after the fire |
| `effis_export_2026-08-04.zip` | The EFFIS export the fires were selected from, as downloaded, with its own readme |
| `effis_perimeters_2018-2026.zip` | The EFFIS export from 2018 that supplies the label history, as downloaded |
| `reviewed_points.csv` | New in 4.1: the 700 reviewed points and the 35 shown twice, with place, stratum, every answer, the dNBR at the point and the final label |
| `review_strata.csv` | New in 4.1: the area of each review stratum, to weight the points |
| `sample.zip` | New in 4.1: 200 tiles with their masks, age rasters and index rows, for a first look without the full download |
| `README.md`, `LICENSE.txt`, `NOTICE.txt` | This file, the terms for each part, and the notices to keep when redistributing |

Unzip every zip into the same folder. That gives

```
acquisitions/<fire_id>/<timestamp>/<fire_id>_<timestamp>_s2.tif     image
acquisitions/<fire_id>/<timestamp>/<fire_id>_<timestamp>_mask.tif   mask
acquisitions/<fire_id>/<timestamp>/<fire_id>_<timestamp>_age.tif    days since the fire
```

and `image_path`, `mask_path` and `age_path` in `index.csv` are relative to that folder. The whole set is about 31 GB.
`fire_id` is `effis_` followed by the EFFIS fire ID. The files open in QGIS, GDAL or rasterio:

```python
import numpy as np
import pandas as pd
import rasterio

index = pd.read_csv("index.csv")
tile = index[index["split"] == "train"].iloc[0]
with rasterio.open(tile["image_path"]) as src:
    reflectance = src.read().astype(np.float32) / 10000   # (13, 256, 256), top-of-atmosphere
with rasterio.open(tile["mask_path"]) as src:
    mask = src.read(1)                                      # (256, 256), values below
with rasterio.open(tile["age_path"]) as src:
    days = src.read(1)                                      # (256, 256), 0 where no perimeter
```

## Images

- GeoTIFF, 13 bands in the order B01, B02, B03, B04, B05, B06, B07, B08, B8A, B09, B10, B11, B12.
- `uint16`, top-of-atmosphere reflectance times 10,000. 0 is no data.
- The tile's local UTM zone (`epsg` in the index), 10 m pixels, deflate compression.
- Read as reflectance from the Sentinel-2 L1C collection of Sentinel Hub in the Copernicus Data Space Ecosystem, so
  values are on the same scale for every processing baseline. The 20 m and 60 m bands were upsampled to 10 m with
  bicubic interpolation.
- Each image is Sentinel Hub's mosaic of the acquisitions within 3 hours of `scene_time`, with the most recent on top.
  That is one satellite pass, or two where two satellites passed minutes apart (about 5 % of the tiles).
- Level-1C: no atmospheric correction, and nothing simulated or degraded.
- Sentinel Hub sets negative reflectance to 0 (its default), and reflectance below 0.00005 rounds to 0, so such pixels,
  for example very dark water in a SWIR band, read as no data.

## Masks

GeoTIFF, one `uint8` band on the image's grid, with a colour table. Shares are of all valid pixels in the 26,846
masks. No data is 0.95 % of all pixels.

| Value | Class | Rule | Share (%) |
|---:|---|---|---:|
| 0 | clear | None of the classes below | 75.23 |
| 2 | fresh_burn | Inside an EFFIS perimeter dated 0 to 90 days before the scene | 7.19 |
| 3 | old_burn | Inside an EFFIS perimeter dated more than 90 and up to 365 days before the scene | 7.34 |
| 7 | older_burn | Inside an EFFIS perimeter dated more than 365 days before the scene; only where the pixel would otherwise be clear | 8.24 |
| 6 | water | NDWI above 0.05, or Scene Classification value 6; checked in winter scenes, see below | 1.28 |
| 4 | cloud | OmniCloudMask, thick or thin cloud | 0.54 |
| 5 | cloud shadow | OmniCloudMask | 0.19 |
| 255 | no data | Outside the valid data of the scene | |

- **Burn age** is the time from the EFFIS fire date to the scene, worked out for every perimeter that touches the tile,
  not only the fire the tile is centred on, using the EFFIS perimeters from 2018 onwards. Perimeters dated after the
  scene are left out. Perimeters are rasterised by pixel centre. Scars older than 2018 are labelled clear.
- **Where classes overlap**, from lowest to highest: clear, water, old_burn, fresh_burn, cloud shadow, cloud, no data.
  So burn wins over water, and cloud and shadow win over burn. older_burn replaces clear only.
- **Water in winter.** In low winter sun, NDWI also marks mountain shadow and snow. In scenes from November to
  February, a water pixel stays water only if the same fire's March to October scene shows water there too, or can't
  see the ground (cloud, shadow or no data). 142 winter tiles have no such scene, keep their water unchecked, and
  have `water_checked` False.
- **Cloud and shadow** are predictions of OmniCloudMask, a model, and nobody checked them pixel by pixel.

## Age rasters

GeoTIFF, one `uint16` band on the mask's grid: for each pixel, the days from the most recent EFFIS perimeter covering
it to the scene, rounded up and at least 1; 0 where no perimeter applies. The mask's burn classes are this raster cut
at 90 and 365 days (1 to 90 fresh_burn, 91 to 365 old_burn, over 365 older_burn), so a different cut, a 30-day alert
class say, needs only the raster. Over the burn pixels, 48.0 % are 1 to 60 days old, 1.6 % 61 to 90, 0.9 % 91 to 120,
45.7 % 121 to 300 and 3.9 % 301 to 365, following the two scene windows.

## Index

`index.csv` has one row per tile.

| Columns | Content |
|---|---|
| `tile_id`, `fire_id`, `ts` | Tile (`<fire_id>/<ts>`), EFFIS fire, scene time stamp as used in the paths |
| `image_path`, `mask_path`, `age_path` | Paths after unzipping |
| `scene_time`, `scene_year`, `fire_date`, `fire_year`, `age_days` | Scene time (UTC), EFFIS fire date, and days between them |
| `window` | Search window of the scene: `fresh` (5 to 60 days) or `old` (120 to 300 days) |
| `epsg`, `minx`, `miny`, `maxx`, `maxy` | UTM zone and tile bounds in metres |
| `lat`, `lon`, `lon_min`, `lat_min`, `lon_max`, `lat_max` | Tile centre and bounds in degrees |
| `frac_clear`, `frac_fresh`, `frac_old`, `frac_older`, `frac_water`, `frac_cloud`, `frac_shadow`, `frac_nodata` | Share of the tile's pixels in each class |
| `other_values` | Any mask values outside the table above |
| `burn_age_median` | Median of the age raster over the tile's burn pixels, in days |
| `country`, `area_ha`, `region` | EFFIS country, burned area and region of the fire |
| `land_cover`, `forest_pct`, `shrubland_pct`, `agriculture_pct` | EFFIS land cover of the fire: the largest group, and three shares |
| `map_source`, `fire_duration_days` | EFFIS mapping source, and days from the fire's first to last date in EFFIS |
| `winter_fire`, `agricultural`, `modis_mapped` | Fire dated November to February; dominant land cover agriculture; perimeter mapped from MODIS. These are the fires most exposed to EFFIS's own limits |
| `own_frac`, `own_obscured_frac` | Share of the tile inside its own fire, and share of that part hidden by cloud or shadow |
| `scene_rule_pass` | Own fire at most 50 % hidden; False for 105 tiles, where no candidate scene passed |
| `tile_filled` | Tile at least 95 % inside its own fire |
| `water_checked` | False for the 142 winter tiles whose water couldn't be checked |
| `dnbr_flag`, `dnbr_in_median`, `dnbr_contrast` | Burn evidence of a fresh-window tile (see Label quality); empty on old-window tiles |
| `fire_dnbr_flag` | The fire's burn evidence, on all its tiles: `burn`, `weak`, `none`, `no_data` or `error` |
| `group`, `block`, `changed_px` | Overlap group, spatial block, and pixels changed from version 1 by the burn update |
| `split`, `split_time`, `split_loco_IT`, `split_loco_ES`, `split_loco_FR`, `split_loco_EL` | The tile's split in each design |
| `product_id`, `processing_baseline` | The Sentinel-2 L1C products over the tile from the passes in its image, joined with `;`, and their processing baselines |

`product_id` names more than one product where the tile lies where granules overlap, and where two satellites passed
minutes apart. The products were looked up in the Copernicus Data Space Ecosystem catalogue by time and place; every
tile has at least one.

## Splits

Use the split columns rather than a random split. 11,540 of the 13,895 fires (83.1 %) have a tile that overlaps another
fire's tile, so a random split puts the same ground in training and test. Fires whose tiles overlap form a group, each
group goes whole to one 0.5° block, and blocks go to train, validation or test.

| Column | Design | `train` | `val` | `test` | `unused` |
|---|---|---:|---:|---:|---:|
| `split` | Blocks, 60/20/20 | 16,183 | 5,243 | 5,420 | 0 |
| `split_time` | Same blocks; train fire year up to 2023, validation 2024, test 2025 and 2026 | 9,596 | 592 | 1,414 | 15,244 |
| `split_loco_IT` | Italy held out as test | 9,831 | 1,650 | 15,365 | 0 |
| `split_loco_ES` | Spain held out | 18,714 | 3,294 | 4,717 | 121 |
| `split_loco_FR` | France held out | 18,854 | 3,233 | 4,705 | 54 |
| `split_loco_EL` | Greece held out | 21,141 | 3,646 | 2,059 | 0 |

`unused` tiles are left out of that design to keep overlapping fires apart, or fall outside its years.

## How the data were made

1. **Fires.** EFFIS burnt-area export of 4 August 2026 for Italy, Spain, Greece and France, fires dated 2020-01-01 to
   2026-08-04: 25,707 perimeters. Kept: 5 to 2,000 ha, and in France south of 46° N, which left 19,443 fires. Fires
   were processed from the largest down: Italy and Greece to 5 ha, Spain to 27 ha and France to 17 ha, where their
   shares of the build were reached. The data set holds 13,895 of the 13,972 eligible EFFIS fires above those sizes;
   the build produced no tile for the other 77 (1 in Italy, 26 in Spain, 50 in France).
2. **Tiles.** A 2.56 km square in the local UTM zone, centred on the perimeter's centroid.
3. **Scenes.** In each window, Sentinel-2 L2A catalogue dates with under 70 % cloud, up to five, least cloudy first.
   For each candidate, OmniCloudMask finds cloud and shadow, and the first scene with at most 50 % of the burned pixels
   in the tile hidden is kept. If none passes, the least hidden is kept (`scene_rule_pass` False).
4. **Images** from the L1C collection at the chosen time; **masks** and **age rasters** from every EFFIS perimeter
   since 2018 that touches the tile.
5. **Index and splits** from the masks and the EFFIS attributes; **burn evidence** from the Sentinel-2 L2A archive.

Days from fire date to scene: fresh scenes median 30.4 (quartiles 16.9 and 46.0), old scenes median 238.4 (quartiles
186.9 and 281.9). There is no pre-fire image in the data set.

## Label quality

The burn labels are EFFIS's, and they were checked four ways. The first three use Sentinel-2 or another Copernicus
product, so they are not independent of the imagery EFFIS maps from; each tests something different.

### Burn evidence for every fire (`burn_evidence.csv`)

For each fire's fresh-window tile, the differenced Normalized Burn Ratio (dNBR, from bands B8A and B12 of Sentinel-2
L2A surface reflectance, Earth Search archive on AWS) between the most recent clear pre-fire image in the 90 days before
the EFFIS date and the tile's own acquisition, inside the fire's perimeter and in a ring 100 to 300 m outside it. A fire
is `burn` when the median inside is at least 0.10 and at least 0.10 above the ring, `weak` when one of the two holds or
the contrast is at least 0.05, and `none` otherwise. This tests whether the area inside the perimeter changed as a burn
does; it says nothing about burn outside the perimeter.

- 13,571 of the 13,895 fires could be assessed: 78.5 % `burn`, 9.5 % `weak`, 12.0 % `none`.
- 80.9 % of the fresh_burn pixels inside the perimeters reach a dNBR of 0.10, against 11.2 % of the clear pixels in
  the rings, whose median is 0.00.
- By fire size, `burn`: 67.7 % for 5 to 10 ha, 74.2 % for 10 to 30 ha, 80.8 % for 30 to 100 ha, 87.2 % for 100 to
  500 ha, 96.2 % for 500 to 2,000 ha.
- By country: Spain 88.0 %, Italy 78.4 %, France 73.1 %, Greece 69.7 % (Greek and Italian fires are the smallest).
- Months later, on the old-window tiles (`burn_evidence_old_window.csv`), 42.4 % of the perimeters still stand at
  least 0.05 above their ring, and 30.8 % at least 0.10.

`fire_dnbr_flag` in the index lets you keep the 10,660 fires with a clear signal, or the 11,947 with at least a weak one.

### Review of random points (`reviewed_points.csv`)

700 random points, drawn from every mask class and weighted by the area of their stratum, were judged from the
imagery alone by two interpreters independently, without the mask, the perimeter or the fire date. The question was
whether the place had burned at any time up to the scene date. A third person settled the 211 points where the two
answers differed and the 52 where the dNBR contradicted a shared answer; her answer is the final label there.

- The two interpreters give answers of the same class at 73.6 % of the points (Cohen's kappa 0.59).
- The final labels confirm 89.7 % of the fresh_burn area, 57.0 % of old_burn, 30.6 % of older_burn and 76.2 % of
  clear.
- Overall accuracy 71.9 % (95 % interval 66.4 to 77.6). Most of the rest is clear land in which the reviewers saw a
  scar already present on the earliest image shown: a burn from before 2018, one EFFIS did not map, or bare ground
  that looks like a scar. Not counting those scars as errors gives 86.9 % (83.8 to 89.5).

Columns: `point_id`; `repeat_of`, the point that a repeated showing repeats; the tile and fire (`tile_id`, `fire_id`,
`scene_time`, `country`, `window`, `fire_year`, `fire_date`, `age_days`, `area_ha`, `land_cover`); `stratum` and
`mask_value`; the place (`row` and `col` in the tile, `lon`, `lat`, and `edge_m`, the distance to the burn edge);
what was shown (`dates_shown`, `earliest_date`, `prefire_shown`); the dNBR at the point, on its scene (`dnbr`) and
on the fire's first scene (`dnbr_first_scene`); `adjudicated`; and the answers `answer_interpreter1`,
`answer_interpreter2`, `answer_adjudicator` and `answer_final`: burned; burned_before (burned, with the scar
already on the earliest image shown); not_burned; water; cloud; shadow; cant_tell. `review_strata.csv` gives the
pixels of each stratum, its share of the data set and its map class. The review published with version 3.0, by one
person, is withdrawn.

### Against Copernicus EMS

Where Copernicus Emergency Management Service rapid mapping covered the same fire (117 areas in 67 activations, 2020
to 2023), EFFIS perimeters agree with the EMS burned area at F1 93.3 % (95 % interval 91.7 to 94.6), with precision
91.0 % and recall 95.8 %. In 143 pairs of a fresh-window tile and an EMS map, fresh_burn agrees with the EMS burned
area at F1 88.6 % (85.5 to 93.2). EMS maps large events, so this says little about small fires.

### What this means for users

- **old_burn and older_burn are date classes.** They mark ground EFFIS mapped as burned 3 to 12 months, or over a year,
  before the scene; often the scar no longer shows.
- **The labels are weakest for small fires, at burn edges and in autumn and spring fires.** Use `fire_dnbr_flag` to keep
  the fires the imagery confirms.
- **clear means that no EFFIS perimeter since 2018 covers the pixel**, not that the land never burned: the review saw a
  scar in 23.8 % of the clear area, most of it already on the earliest image. An EFFIS perimeter can include small
  unburned patches.
  EFFIS does not separate wildfires from prescribed or agricultural burns; `agricultural` and `winter_fire` flag the
  fires most exposed to that.

## Using the data

- Leave out 255 in training and evaluation.
- Decide what older_burn should be for your task: keep it as a class, merge it with clear when only recent burns
  matter, or merge it with burn. Or cut the age raster where you need.
- `scene_rule_pass` and `water_checked` let you drop the tiles whose scene or water couldn't be made to meet the rules.
- To trace a tile to its source, search the Copernicus Data Space Ecosystem catalogue for the names in `product_id`.

## Versions

- **1.0** (Hugging Face, 2026): burn only from the fires that passed the selection, no older_burn class, winter water
  unchecked.
- **3.0**: burn from every EFFIS perimeter of the 2020 to 2026 export, and the older_burn class (15,482 masks changed);
  winter water checked; `water_checked`, `product_id` and `processing_baseline` in the index. Version 2 was an internal
  step.
- **4.0**: label history from 2018 (9,109 masks changed, almost all from clear to older_burn); age
  rasters; burn evidence for every fire; the one-person review of 3.0 withdrawn, and a two-reviewer review of 700 points to follow in 4.1; new index
  columns `age_path`, `burn_age_median`, `winter_fire`, `agricultural`, `modis_mapped`, `dnbr_flag`, `dnbr_in_median`,
  `dnbr_contrast` and `fire_dnbr_flag`. Images are unchanged.
- **4.1** (this record): the review of 700 random points (`reviewed_points.csv`, `review_strata.csv`) and a
  200-tile sample (`sample.zip`). Images, masks, age rasters and index are unchanged from 4.0.

## Citation

Thind, P. K., Cadau, E., Del Prete, R., D'Aulerio, P., Ricolfi, S., Meoni, G., and Taramelli, A.: burn2scar: a Sentinel-2
burn-scar segmentation data set, version 4.1, Zenodo [data set], https://doi.org/10.5281/zenodo.24375951, 2026.

```bibtex
@dataset{thind2026burn2scar,
  author    = {Thind, Parampuneet Kaur and Cadau, Enrico and Del Prete, Roberto and
               D'Aulerio, Paola and Ricolfi, Silvia and Meoni, Gabriele and Taramelli, Andrea},
  title     = {burn2scar: a Sentinel-2 burn-scar segmentation data set},
  year      = {2026},
  version   = {4.1},
  publisher = {Zenodo},
  doi       = {10.5281/zenodo.24375951}
}
```

A data description paper is in preparation.

## Licence and attribution

The authors' contributions are under CC BY 4.0. The Sentinel-2 imagery and the water class are under the Legal Notice
on the use of Copernicus Sentinel Data and Service Information. The burn classes and age rasters are adapted from EFFIS
(https://forest-fire.emergency.copernicus.eu/applications/data-and-services), © European Union, 1995-2026, CC BY 4.0. Cloud and shadow come from OmniCloudMask, MIT-licensed software. `LICENSE.txt`
has the details. Keep these notices when you redistribute the data:

```
Contains modified Copernicus Sentinel data 2020-2026.
Contains modified Copernicus Service information 2018-2026.
Burn labels adapted from EFFIS, Copernicus Emergency Management Service,
© European Union, 1995-2026, CC BY 4.0
(https://creativecommons.org/licenses/by/4.0/). Changes: perimeters rasterised
to a 10 m grid, split into fresh, old and older burn classes by date, merged
with cloud, cloud shadow and water classes, and turned into days-since-fire rasters.
```

Neither the European Union, ESA, the Joint Research Centre nor the EFFIS team has endorsed this data set.

Questions and error reports: [github.com/paramkaur10/burn2scar_dataset_creation/issues](https://github.com/paramkaur10/burn2scar_dataset_creation/issues)
