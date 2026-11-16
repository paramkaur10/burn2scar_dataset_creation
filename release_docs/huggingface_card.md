---
license: cc-by-4.0
viewer: false
pretty_name: burn2scar
task_categories:
- image-segmentation
tags:
- remote-sensing
- earth-observation
- sentinel-2
- burn-scar
- burned-area
- wildfire
- cloud-mask
- segmentation
- effis
- geospatial
size_categories:
- 10K<n<100K
---

# burn2scar

Sentinel-2 Level-1C tiles with burn-scar segmentation masks for 13,895 fires mapped by the European Forest Fire
Information System (EFFIS) in Italy, Spain, France south of 46° N and Greece, 2020 to 2026. Real imagery: nothing is
simulated or degraded.

- **Citable copy, version 4.0:** [doi.org/10.5281/zenodo.24221783](https://doi.org/10.5281/zenodo.24221783) (version 3.0: [doi.org/10.5281/zenodo.23986827](https://doi.org/10.5281/zenodo.23986827)).
  Its README has the full documentation and the label-quality numbers with their intervals, and the record also holds
  the burn evidence of every fire and the EFFIS exports, which are not mirrored here.
- **Code:** [github.com/paramkaur10/burn2scar_dataset_creation](https://github.com/paramkaur10/burn2scar_dataset_creation)

## Summary

- **26,846 tiles** of 256 × 256 pixels (2.56 km across), all 13 bands on a 10 m grid, top-of-atmosphere reflectance.
- **Two scenes of most fires** (12,951 of 13,895): one 5 to 60 days after the fire, one 120 to 300 days after.
- **Masks with seven classes plus no data:** clear, fresh burn, old burn, older burn, water, cloud and cloud shadow.
- **Days since the fire, per pixel,** beside every mask (new in 4.0), so the age classes can be cut anywhere.
- **Burn evidence for every fire** in the index (`fire_dnbr_flag`), so you can keep the fires the imagery confirms.
- **Splits that keep overlapping fires apart**, by area, by year and by country.

| Country | Fires | Tiles |
|---|---:|---:|
| Italy | 8,014 | 15,365 |
| Spain | 2,442 | 4,717 |
| France, south of 46° N | 2,388 | 4,705 |
| Greece | 1,051 | 2,059 |

## Versions

The masks, age rasters and index here are version 4.0, the same as the Zenodo record. Earlier masks stay available at
the `v1` and `v3` tags of this repository. Version 4.0 takes the label history back to 2018, adds the age rasters, and
adds the burn evidence and quality flags to the index. Version 3.0 had taken burn from every EFFIS perimeter in the
tile, added the older_burn class and checked water in winter scenes. Images are unchanged since version 1.

## Files

```
index.csv                                                           one row per tile
acquisitions/<fire_id>/<timestamp>/<fire_id>_<timestamp>_s2.tif     image
acquisitions/<fire_id>/<timestamp>/<fire_id>_<timestamp>_mask.tif   mask
acquisitions/<fire_id>/<timestamp>/<fire_id>_<timestamp>_age.tif    days from the fire to the scene, 0 where none
LICENSE.txt, NOTICE.txt                                             terms and notices
```

**Images:** GeoTIFF, 13 bands in the order B01, B02, B03, B04, B05, B06, B07, B08, B8A, B09, B10, B11, B12; `uint16`,
reflectance times 10,000; 0 is no data; local UTM zone. Read as reflectance from the Sentinel-2 L1C collection of
Sentinel Hub in the Copernicus Data Space Ecosystem; the 20 m and 60 m bands were upsampled with bicubic interpolation.

**Masks:** GeoTIFF, one `uint8` band on the image's grid, with a colour table.

| Value | Class | Rule | Share of valid pixels (%) |
|---:|---|---|---:|
| 0 | clear | None of the classes below | 75.23 |
| 2 | fresh_burn | Inside an EFFIS perimeter dated 0 to 90 days before the scene | 7.19 |
| 3 | old_burn | Inside an EFFIS perimeter dated more than 90 and up to 365 days before the scene | 7.34 |
| 7 | older_burn | Inside an EFFIS perimeter (from 2018 on) dated more than 365 days before; only where the pixel would otherwise be clear | 8.24 |
| 6 | water | NDWI above 0.05, or Scene Classification value 6; in November to February scenes, kept only where the fire's March to October scene agrees | 1.28 |
| 4 | cloud | OmniCloudMask, thick or thin cloud | 0.54 |
| 5 | cloud shadow | OmniCloudMask | 0.19 |
| 255 | no data | Outside the valid data of the scene | |

Where classes overlap, cloud and shadow win over burn, and burn wins over water.

## How to load

```python
import numpy as np
import pandas as pd
import rasterio
from huggingface_hub import hf_hub_download

repo = "neet1797/burn2scar"
index = pd.read_csv(hf_hub_download(repo, "index.csv", repo_type="dataset"))
tile = index[index["split"] == "train"].iloc[0]

image_file = hf_hub_download(repo, tile["image_path"], repo_type="dataset")
mask_file = hf_hub_download(repo, tile["mask_path"], repo_type="dataset")
with rasterio.open(image_file) as src:
    reflectance = src.read().astype(np.float32) / 10000   # (13, 256, 256)
with rasterio.open(mask_file) as src:
    mask = src.read(1)                                      # (256, 256)
days = rasterio.open(hf_hub_download(repo, tile["age_path"], repo_type="dataset")).read(1)   # days since the fire
```

`snapshot_download(repo, repo_type="dataset")` fetches everything, about 31 GB. The Zenodo record has the same files
in zips by country and fire year.

## Splits

Use the split columns of `index.csv` rather than a random split: 83.1 % of the fires have a tile that overlaps another
fire's tile. Values are `train`, `val`, `test`, and `unused` where a tile is left out of that design.

| Column | Design | train | val | test | unused |
|---|---|---:|---:|---:|---:|
| `split` | 0.5° blocks, 60/20/20 | 16,183 | 5,243 | 5,420 | 0 |
| `split_time` | Same blocks; fire years up to 2023, 2024, then 2025 and 2026 | 9,596 | 592 | 1,414 | 15,244 |
| `split_loco_IT` | Italy held out | 9,831 | 1,650 | 15,365 | 0 |
| `split_loco_ES` | Spain held out | 18,714 | 3,294 | 4,717 | 121 |
| `split_loco_FR` | France held out | 18,854 | 3,233 | 4,705 | 54 |
| `split_loco_EL` | Greece held out | 21,141 | 3,646 | 2,059 | 0 |

## Label quality

- **Burn evidence for every fire.** The dNBR between a clear pre-fire Sentinel-2 L2A image and the tile's own scene,
  inside the fire's perimeter against a ring around it, shows a clear burn signal for 78.5 % of the 13,571 assessed
  fires and at least a weak one for 88.0 %: 67.7 % of fires of 5 to 10 ha, rising to 96.2 % at 500 to 2,000 ha.
  `fire_dnbr_flag` in the index carries it for every tile.
- **A review of 700 random points by two people**, from the imagery alone, is being completed; its results will be
  added to the Zenodo record in version 4.1.
- **Against Copernicus EMS rapid mapping** of the same fires (117 areas, 2020 to 2023), EFFIS perimeters agree at
  F1 93.3 %. In 143 tile and EMS map pairs, fresh_burn agrees at F1 88.6 %.

## Limitations

- **old_burn is a date class**: ground that EFFIS mapped as burned 3 to 12 months before the scene. Months later,
  42 % of the perimeters still stand out from the land around them in dNBR; the rest have faded.
- **fresh_burn is less reliable** for fires of 5 to 30 ha, for autumn and spring fires, and on the edge of a burn.
- **Burns that EFFIS didn't map are labelled clear**, and EFFIS perimeters can include small unburned patches.
- **`clear` is what is left**: anything not burn, cloud, shadow or water, including bare soil, rock and volcanic
  ground. Combine it with a land-cover product such as ESA WorldCover if you need a vegetation class.
- **Cloud and shadow are model predictions** (OmniCloudMask), not checked pixel by pixel.
- **No pre-fire image** is included.
- `scene_rule_pass` is False for 105 tiles where over half of the fire's own scar is hidden, and `water_checked` is
  False for 142 winter tiles whose water couldn't be checked.

## Licensing

The authors' own contributions are licensed under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).
Third-party data keep their own terms, set out in full in [LICENSE.txt](LICENSE.txt).

| Part | Terms | When you redistribute |
|---|---|---|
| Selection, labelling logic, index and splits, compilation, documentation | CC BY 4.0 | Credit the authors and state your changes |
| Sentinel-2 imagery and water class | [Copernicus Sentinel data legal notice](https://sentinels.copernicus.eu/documents/247904/690755/Sentinel_Data_Legal_Notice) | Include the Copernicus notice below |
| Fresh, old and older burn classes | [EFFIS data licence](https://forest-fire.emergency.copernicus.eu/about-effis/data-license) (CC BY 4.0) | Credit EFFIS, link the licence and state the changes |
| Cloud and cloud shadow classes | Predicted with [OmniCloudMask](https://github.com/DPIRD-DMA/OmniCloudMask) (MIT software) | Credit OmniCloudMask |

CC BY 4.0 has no non-commercial restriction, and ESA's Sentinel data principles make no distinction between commercial
and other uses. This is not legal advice. Check the terms for your own use case.

## Citation

```bibtex
@dataset{thind2026burn2scar,
  author    = {Thind, Parampuneet Kaur and Cadau, Enrico and Del Prete, Roberto and
               Ricolfi, Silvia and Meoni, Gabriele and Taramelli, Andrea},
  title     = {burn2scar: a Sentinel-2 burn-scar segmentation data set},
  year      = {2026},
  version   = {4.0},
  publisher = {Zenodo},
  doi       = {10.5281/zenodo.24221783}
}
```

Please also keep the notices from [NOTICE.txt](NOTICE.txt):

```
Contains modified Copernicus Sentinel data 2020-2026.
Contains modified Copernicus Service information 2018-2026.
Burn labels adapted from EFFIS, Copernicus Emergency Management Service,
© European Union, 1995-2026, CC BY 4.0
(https://creativecommons.org/licenses/by/4.0/). Changes: perimeters rasterised
to a 10 m grid, split into fresh, old and older burn classes by date, merged
with cloud, cloud shadow and water classes, and turned into days-since-fire rasters.
```

**Authors:** Parampuneet Kaur Thind, Enrico Cadau, Roberto Del Prete, Silvia Ricolfi, Gabriele Meoni, Andrea Taramelli
