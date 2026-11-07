# burn2scar: a Sentinel-2 burn-scar segmentation data set

**Version 3.0** · DOI [10.5281/zenodo.23986827](https://doi.org/10.5281/zenodo.23986827)   
Authors: Parampuneet Kaur Thind, Enrico Cadau, Roberto Del Prete, Silvia Ricolfi, Gabriele Meoni, Andrea Taramelli  
Licence: CC BY 4.0 for the authors' contributions. Third-party data keep their own terms; see `LICENSE.txt`.  
Code: [github.com/paramkaur10/burn2scar_dataset_creation](https://github.com/paramkaur10/burn2scar_dataset_creation) · Mirror: [huggingface.co/datasets/neet1797/burn2scar](https://huggingface.co/datasets/neet1797/burn2scar)

26,846 Sentinel-2 Level-1C tiles over 13,895 fires mapped by the European Forest Fire Information System (EFFIS)
in Italy, Spain, France south of 46° N and Greece, from 2020 to 2026. Each tile has all 13 bands on a 10 m grid,
256 × 256 pixels (2.56 km across), and a mask with seven classes plus no data: clear, fresh burn, old burn,
older burn, water, cloud and cloud shadow. Most fires have two tiles of the same place: one scene 5 to 60 days
after the fire, and one 120 to 300 days after.

| Country | Fires | Two scenes | One scene | Tiles | Median fire size (ha) |
|---|---:|---:|---:|---:|---:|
| Italy | 8,014 | 7,351 | 663 | 15,365 | 24 |
| Spain | 2,442 | 2,275 | 167 | 4,717 | 66 |
| France, south of 46° N | 2,388 | 2,317 | 71 | 4,705 | 39 |
| Greece | 1,051 | 1,008 | 43 | 2,059 | 22 |
| All | 13,895 | 12,951 | 944 | 26,846 | |

Fires by year: 2020 1,721; 2021 2,074; 2022 2,264; 2023 1,998; 2024 1,831; 2025 2,313; 2026 1,694 (to 4 August).

## Files

| File | Content |
|---|---|
| `images_<CC>_<YYYY>.zip` | Images of the fires in country `CC` (IT, ES, FR, EL) that EFFIS dates to year `YYYY` |
| `masks.zip` | All masks |
| `index.csv` | One row per tile: paths, dates, location, class shares, splits and source products |
| `reviewed_points.csv`, `review_strata.csv` | The visual check of random pixels described under Label quality |
| `effis_export_2026-08-04.zip` | The EFFIS export the burn classes were made from, as downloaded, with its own readme |
| `README.md`, `LICENSE.txt`, `NOTICE.txt` | This file, the terms for each part, and the notices to keep when redistributing |

Unzip every zip into the same folder. That gives

```
acquisitions/<fire_id>/<timestamp>/<fire_id>_<timestamp>_s2.tif     image
acquisitions/<fire_id>/<timestamp>/<fire_id>_<timestamp>_mask.tif   mask
```

and `image_path` and `mask_path` in `index.csv` are relative to that folder. The whole set is about 31 GB.
`fire_id` is `effis_` followed by the EFFIS fire ID. Both files open in QGIS, GDAL or rasterio:

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
| 0 | clear | None of the classes below | 77.26 |
| 2 | fresh_burn | Inside an EFFIS perimeter dated 0 to 90 days before the scene | 7.19 |
| 3 | old_burn | Inside an EFFIS perimeter dated more than 90 and up to 365 days before the scene | 7.32 |
| 7 | older_burn | Inside an EFFIS perimeter dated more than 365 days before the scene; only where the pixel would otherwise be clear | 6.24 |
| 6 | water | NDWI above 0.05, or Scene Classification value 6; checked in winter scenes, see below | 1.28 |
| 4 | cloud | OmniCloudMask, thick or thin cloud | 0.54 |
| 5 | cloud shadow | OmniCloudMask | 0.19 |
| 255 | no data | Outside the valid data of the scene | |

- **Burn age** is the time from the EFFIS fire date to the scene, worked out for every perimeter that touches the tile,
  not only the fire the tile is centred on. Perimeters dated after the scene are left out. Perimeters are rasterised
  by pixel centre.
- **Where classes overlap**, from lowest to highest: clear, water, old_burn, fresh_burn, cloud shadow, cloud, no data.
  So burn wins over water, and cloud and shadow win over burn. older_burn replaces clear only.
- **Water in winter.** In low winter sun, NDWI also marks mountain shadow and snow. In scenes from November to
  February, a water pixel stays water only if the same fire's March to October scene shows water there too, or can't
  see the ground (cloud, shadow or no data). 142 winter tiles have no such scene, keep their water unchecked, and
  have `water_checked` False. After the check, water is 0.94 to 1.39 % of valid pixels in scenes from March to
  October, and 1.25 to 2.35 % in scenes from November to February, highest in January.
- **Cloud and shadow** are predictions of OmniCloudMask, a model, and nobody checked them pixel by pixel.

## Index

`index.csv` has one row per tile.

| Columns | Content |
|---|---|
| `tile_id`, `fire_id`, `ts` | Tile (`<fire_id>/<ts>`), EFFIS fire, scene time stamp as used in the paths |
| `image_path`, `mask_path` | Paths after unzipping |
| `scene_time`, `scene_year`, `fire_date`, `fire_year`, `age_days` | Scene time (UTC), EFFIS fire date, and days between them |
| `window` | Search window of the scene: `fresh` (5 to 60 days) or `old` (120 to 300 days) |
| `epsg`, `minx`, `miny`, `maxx`, `maxy` | UTM zone and tile bounds in metres |
| `lat`, `lon`, `lon_min`, `lat_min`, `lon_max`, `lat_max` | Tile centre and bounds in degrees |
| `frac_clear`, `frac_fresh`, `frac_old`, `frac_older`, `frac_water`, `frac_cloud`, `frac_shadow`, `frac_nodata` | Share of the tile's pixels in each class |
| `other_values` | Any mask values outside the table above |
| `country`, `area_ha`, `region` | EFFIS country, burned area and region of the fire |
| `land_cover`, `forest_pct`, `shrubland_pct`, `agriculture_pct` | EFFIS land cover of the fire: the largest group, and three shares |
| `map_source`, `fire_duration_days` | EFFIS mapping source, and days from the fire's first to last date in EFFIS |
| `own_frac`, `own_obscured_frac` | Share of the tile inside its own fire, and share of that part hidden by cloud or shadow |
| `scene_rule_pass` | Own fire at most 50 % hidden; False for 105 tiles, where no candidate scene passed |
| `tile_filled` | Tile at least 95 % inside its own fire |
| `water_checked` | False for the 142 winter tiles whose water couldn't be checked |
| `group`, `block`, `changed_px` | Overlap group, spatial block, and pixels changed from version 1 by the burn update |
| `split`, `split_time`, `split_loco_IT`, `split_loco_ES`, `split_loco_FR`, `split_loco_EL` | The tile's split in each design |
| `product_id`, `processing_baseline` | The Sentinel-2 L1C products over the tile from the passes in its image, joined with `;`, and their processing baselines |

`product_id` names more than one product where the tile lies where granules overlap, and where two satellites passed
minutes apart. In those tiles the later pass covers the earlier one wherever it has data, and `scene_time` is the
first acquisition time in Sentinel Hub's catalogue. The pipeline kept only the acquisition time, so the products were
looked up afterwards in the Copernicus Data Space Ecosystem catalogue, by time and place. Every tile has at least one.
On 20 random tiles, the lookup found every product that the Sentinel Hub catalogue lists at the acquisition time.

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
   2026-08-04: 25,707 perimeters. Kept: 5 to 2,000 ha, and in France south of 46° N, which left 19,443 fires. Each
   country had a share of a 40 GB budget (Italy 60 %, the others 13.3 % each), and fires ran from the largest down, so
   in Spain and France the smallest fires weren't reached. 13,895 fires have at least one tile. EFFIS mapped 13,892
   of them from Sentinel-2 and 3 from MODIS (`map_source`).
2. **Tiles.** A 2.56 km square in the local UTM zone, centred on the perimeter's centroid.
3. **Scenes.** In each window, Sentinel-2 L2A catalogue dates with under 70 % cloud, up to five, least cloudy first.
   For each candidate, OmniCloudMask finds cloud and shadow, and the first scene with at most 50 % of the burned pixels
   in the tile hidden is kept. If none passes, the least hidden is kept (`scene_rule_pass` False).
4. **Images** from the L1C collection at the chosen time, and **masks** as described above.
5. **Index and splits** from the masks and the EFFIS attributes.

Days from fire date to scene: fresh scenes median 30.4 (quartiles 16.9 and 46.0), old scenes median 238.4 (quartiles
186.9 and 281.9). There is no pre-fire image in the data set.

## Label quality

### Against Copernicus EMS

Where Copernicus Emergency Management Service rapid mapping covered the same fire (117 areas in 67 activations, 2020
to 2023), EFFIS perimeters agree with the EMS burned area at F1 93.3 % (95 % interval 91.7 to 94.6), with precision
91.0 % and recall 95.8 %. In 143 pairs of a fresh-window tile and an EMS map, fresh_burn agrees with the EMS burned
area at F1 88.6 % (85.5 to 93.2).

### Visual review of random pixels

The first author judged randomly chosen pixels from the imagery alone, without seeing the mask or EFFIS: burned, not burned,
water, cloud, shadow, or can't tell. Pixels were drawn separately from each mask class (clear split into near a burn,
within 300 m, and far from one), with every pixel of a class equally likely. Each picture showed the point on the
scene, on the fire's other scene, and, where one could be fetched, on a scene from before the fire. 1,050 points were
judged, and 52 of them were shown a second time: the same answer came back for 88.5 % (95 % interval 77.0 to 94.6).

| Mask class: expected answer | Points | Reviewer agrees (%) | 95 % interval |
|---|---:|---:|---|
| fresh_burn: burned | 180 | 65.6 | 58.4–72.1 |
| old_burn: burned | 180 | 24.2 | 18.5–30.9 |
| clear near a burn: not burned | 180 | 93.9 | 89.3–96.5 |
| clear far from a burn: not burned | 180 | 97.2 | 93.7–98.8 |
| water: water | 100 | 92.9 | 86.1–96.5 |
| cloud: cloud | 100 | 56.8 | 46.8–66.3 |
| cloud shadow: shadow | 100 | 68.1 | 58.0–76.8 |

"Can't tell" is left out of the percentages. Of the 30 older_burn points, 3 (10 %) were judged burned.

For the burned class, fresh_burn and old_burn count as burned, and clear and water as not burned. Each class is
weighted by its area, and the 816 points with an answer of burned, not burned or water are used.

| Burned class | Estimate (%) | 95 % interval |
|---|---:|---|
| Precision | 44.7 | 40.2–49.4 |
| Recall | 68.9 | 58.4–81.1 |
| F1 | 54.2 | 48.7–59.6 |
| IoU | 37.2 | 32.2–42.5 |
| Overall accuracy | 88.2 | 86.4–90.0 |
| Burned share of this area, from the review | 10.1 | 8.4–12.0 |
| Burned share of this area, in the masks | 15.6 | |

Intervals are Wilson intervals for each class, and a bootstrap within each class for the weighted measures.
`review_strata.csv` holds the pixel count of each class, so all of this can be recomputed from `reviewed_points.csv`.
By country, fresh and old burn points together were judged burned for 68.9 % of 90 points in Spain, 50.0 % of 30 in
Greece, 35.8 % of 173 in Italy and 33.8 % of 65 in France.

What this means for users:

- **old_burn is a date class.** It marks ground that EFFIS mapped as burned 3 to 12 months before the scene, and
  often the scar can no longer be seen. A check of the drop in NBR since before the fire agrees: against a threshold
  that 5 % of clear pixels far from a burn pass, 38.2 % of old_burn points pass on the fire's fresh scene, and 10.8 %
  on the old scene itself.
- **fresh_burn is weaker** for small fires (52.0 % confirmed for 5 to 30 ha, against 66 to 70 % above 30 ha), late in
  the window (54.2 % at 45 to 60 days, against 73.8 % at 5 to 15 days), for fires from December to February (50.0 %),
  and right at the edge of a burn (33.3 % within 15 m, from 12 points, against 73.9 % more than 100 m inside).
- **Burns that EFFIS didn't map are labelled clear**, and an EFFIS perimeter can include small unburned patches
  inside a fire. EFFIS maps fires of about 30 ha and up from MODIS and, since 2018, smaller ones as well from
  Sentinel-2.
- **The reviewer is the first author**, and one person's judgement from 10 m imagery is a check on visibility, not ground
  truth.

## Using the data

- Leave out 255 in training and evaluation.
- Decide what older_burn should be for your task: keep it as a class, merge it with clear when only recent burns
  matter, or merge it with burn.
- Treat old_burn as "burned 3 to 12 months ago by EFFIS", not as "visibly burned".
- `scene_rule_pass` and `water_checked` let you drop the tiles whose scene or water couldn't be made to meet the rules.
- To trace a tile to its source, search the Copernicus Data Space Ecosystem catalogue for the names in `product_id`.

## Versions

- **1.0** (Hugging Face, 2026): burn only from the fires that passed the selection, no older_burn class, winter water
  unchecked.
- **3.0** (this record): burn from every EFFIS perimeter in the tile, and the older_burn class (15,482 masks changed);
  winter water checked (water 2.32 % to 1.28 % of valid pixels); `water_checked`, `product_id` and
  `processing_baseline` in the index. Images are unchanged. Eight tiles from an early run have no split and are left
  out. Version 2 was an internal step.

## Citation

Thind, P. K., Cadau, E., Del Prete, R., Ricolfi, S., Meoni, G., and Taramelli, A.: burn2scar: a Sentinel-2
burn-scar segmentation data set, version 3.0, Zenodo [data set], https://doi.org/10.5281/zenodo.23986827, 2026.

```bibtex
@dataset{thind2026burn2scar,
  author    = {Thind, Parampuneet Kaur and Cadau, Enrico and Del Prete, Roberto and
               Ricolfi, Silvia and Meoni, Gabriele and Taramelli, Andrea},
  title     = {burn2scar: a Sentinel-2 burn-scar segmentation data set},
  year      = {2026},
  version   = {3.0},
  publisher = {Zenodo},
  doi       = {10.5281/zenodo.23986827}
}
```

A data description paper is in preparation.

## Licence and attribution

The authors' contributions are under CC BY 4.0. The Sentinel-2 imagery and the water class are under the Legal Notice
on the use of Copernicus Sentinel Data and Service Information. The burn classes are adapted from EFFIS, © European
Union, 1995-2026, CC BY 4.0. Cloud and shadow come from OmniCloudMask, MIT-licensed software. `LICENSE.txt` has the
details. Keep these notices when you redistribute the data:

```
Contains modified Copernicus Sentinel data 2020-2026.
Contains modified Copernicus Service information 2020-2026.
Burn labels adapted from EFFIS, Copernicus Emergency Management Service,
© European Union, 1995-2026, CC BY 4.0
(https://creativecommons.org/licenses/by/4.0/). Changes: perimeters rasterised
to a 10 m grid, split into fresh, old and older burn classes by date, and
merged with cloud, cloud shadow and water classes.
```

Neither the European Union, ESA, the Joint Research Centre nor the EFFIS team has endorsed this data set.

Questions and error reports: [github.com/paramkaur10/burn2scar_dataset_creation/issues](https://github.com/paramkaur10/burn2scar_dataset_creation/issues)
