# burn2scar baseline results

55 finished runs. Values are test-set percentages, as mean ± standard deviation across seeds.

Main task: fresh burn against everything else. Fresh burn is positive; clear, old burn and water are negative.
Deep models predict fresh and old burn separately, so their fresh scores come from the same runs. The classic
baselines are run once with fresh burn as their only target. Burnt metrics with fresh and old burn together
follow for reference. Cloud, shadow, older_burn and nodata labels are left out of both.

The FLOGA column is for reference only. FLOGA uses pre-fire and post-fire L2A images of Greece at 20 m,
so the two sets of results measure different tasks.

## Fresh burn against everything else, spatial split

| Method | Type | Seeds | Fresh P | Fresh R | Fresh F1 | Fresh IoU | Not fresh F1 | Not fresh IoU | Mean F1 | Mean IoU | FLOGA family | FLOGA burnt F1 (reference) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| NBR | index | 1 | 53.29 | 58.89 | 55.95 | 38.84 | 95.95 | 92.22 | 75.95 | 65.53 | dNBR | 45.14 |
| NBR2 | index | 1 | 43.83 | 64.21 | 52.10 | 35.22 | 94.76 | 90.05 | 73.43 | 62.64 | dNBR2 | 52.78 |
| BAIS2 | index | 1 | 43.08 | 61.05 | 50.51 | 33.79 | 94.71 | 89.95 | 72.61 | 61.87 | dBAIS2 | 31.04 |
| NBR+ | index | 1 | 45.74 | 51.32 | 48.37 | 31.90 | 95.21 | 90.87 | 71.79 | 61.38 | dNBR+ | 46.57 |
| CSI | index | 1 | 43.26 | 53.12 | 47.69 | 31.31 | 94.89 | 90.27 | 71.29 | 60.79 | dCSI | 40.31 |
| SAVI | index | 1 | 29.83 | 65.96 | 41.08 | 25.85 | 91.32 | 84.03 | 66.20 | 54.94 | dSAVI | 45.96 |
| EVI | index | 1 | 28.80 | 67.43 | 40.36 | 25.28 | 90.81 | 83.16 | 65.58 | 54.22 | dEVI | 30.89 |
| NDWI | index | 1 | 27.10 | 66.62 | 38.53 | 23.86 | 90.14 | 82.05 | 64.33 | 52.95 | dNDWI | 32.22 |
| MIRBI | index | 1 | 24.99 | 55.11 | 34.39 | 20.76 | 90.36 | 82.41 | 62.37 | 51.59 | dMIRBI | 64.06 |
| NDVI | index | 1 | 22.01 | 75.54 | 34.09 | 20.55 | 85.81 | 75.15 | 59.95 | 47.85 | dNDVI | 45.93 |
| RF | pixel | 5 | 33.66 ± 0.05 | 87.00 ± 0.05 | 48.54 ± 0.04 | 32.05 ± 0.04 | 91.39 ± 0.02 | 84.14 ± 0.04 | 69.96 ± 0.03 | 58.10 ± 0.04 | RF | 67.38 |
| segformer_b2 | deep | 5 | 77.58 ± 0.56 | 88.79 ± 0.16 | 82.81 ± 0.26 | 70.66 ± 0.37 | 98.39 ± 0.03 | 96.83 ± 0.06 | 90.60 ± 0.14 | 83.74 ± 0.22 | ChangeFormer | 85.65 |
| unet_r34 | deep | 5 | 74.26 ± 2.53 | 88.64 ± 1.05 | 80.78 ± 1.17 | 67.77 ± 1.63 | 98.15 ± 0.17 | 96.37 ± 0.32 | 89.47 ± 0.66 | 82.07 ± 0.97 | FC-EF-Conc | 72.18 |

## Fresh and old burn together against unburnt, spatial split

| Method | Type | Seeds | Burnt P | Burnt R | Burnt F1 | Burnt IoU | Unburnt F1 | Unburnt IoU | Mean F1 | Mean IoU | FLOGA family | FLOGA burnt F1 (reference) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| NBR2 | index | 1 | 41.72 | 52.72 | 46.58 | 30.36 | 88.19 | 78.87 | 67.39 | 54.62 | dNBR2 | 52.78 |
| NBR | index | 1 | 46.35 | 46.19 | 46.27 | 30.10 | 89.79 | 81.47 | 68.03 | 55.79 | dNBR | 45.14 |
| BAIS2 | index | 1 | 44.03 | 46.59 | 45.28 | 29.26 | 89.21 | 80.53 | 67.24 | 54.90 | dBAIS2 | 31.04 |
| NBR+ | index | 1 | 41.79 | 45.46 | 43.55 | 27.83 | 88.68 | 79.67 | 66.12 | 53.75 | dNBR+ | 46.57 |
| CSI | index | 1 | 37.15 | 46.73 | 41.39 | 26.10 | 87.08 | 77.12 | 64.24 | 51.61 | dCSI | 40.31 |
| SAVI | index | 1 | 29.05 | 54.09 | 37.80 | 23.30 | 81.54 | 68.83 | 59.67 | 46.07 | dSAVI | 45.96 |
| EVI | index | 1 | 28.53 | 55.19 | 37.62 | 23.16 | 80.87 | 67.88 | 59.24 | 45.52 | dEVI | 30.89 |
| MIRBI | index | 1 | 29.12 | 52.26 | 37.40 | 23.00 | 81.97 | 69.46 | 59.69 | 46.23 | dMIRBI | 64.06 |
| NDWI | index | 1 | 28.65 | 53.38 | 37.28 | 22.91 | 81.37 | 68.59 | 59.33 | 45.75 | dNDWI | 32.22 |
| NDVI | index | 1 | 25.79 | 59.03 | 35.90 | 21.87 | 77.11 | 62.75 | 56.50 | 42.31 | dNDVI | 45.93 |
| RF | pixel | 5 | 38.52 ± 0.03 | 72.57 ± 0.09 | 50.33 ± 0.01 | 33.62 ± 0.01 | 85.10 ± 0.03 | 74.07 ± 0.04 | 67.72 ± 0.01 | 53.85 ± 0.02 | RF | 67.38 |
| segformer_b2 | deep | 5 | 74.85 ± 0.43 | 82.02 ± 0.71 | 78.27 ± 0.16 | 64.30 ± 0.21 | 95.62 ± 0.03 | 91.61 ± 0.06 | 86.95 ± 0.08 | 77.96 ± 0.12 | ChangeFormer | 85.65 |
| unet_r34 | deep | 5 | 68.56 ± 2.29 | 82.49 ± 1.07 | 74.85 ± 1.06 | 59.82 ± 1.35 | 94.61 ± 0.38 | 89.77 ± 0.68 | 84.73 ± 0.72 | 74.80 ± 1.02 | FC-EF-Conc | 72.18 |

## All classes, spatial split

Each class against every other class, so these IoUs differ slightly from the two-class tables above.

| Method | Seeds | mIoU | clear IoU | fresh_burn IoU | old_burn IoU | cloud IoU | shadow IoU | water IoU | Fresh vs old accuracy |
|---|---|---|---|---|---|---|---|---|---|
| segformer_b2 | 5 | 74.89 ± 0.50 | 91.23 ± 0.06 | 70.53 ± 0.37 | 51.58 ± 0.82 | 77.50 ± 1.50 | 71.55 ± 0.92 | 86.95 ± 0.21 | 96.60 ± 0.34 |
| unet_r34 | 5 | 70.44 ± 1.37 | 89.29 ± 0.69 | 67.63 ± 1.63 | 45.85 ± 1.79 | 70.51 ± 1.46 | 64.48 ± 3.54 | 84.88 ± 0.41 | 95.89 ± 0.55 |

## Generalisation designs

| Design | Method | Seeds | Fresh P | Fresh R | Fresh F1 | Fresh IoU | Burnt F1 (fresh + old) | Burnt IoU (fresh + old) |
|---|---|---|---|---|---|---|---|---|
| split_loco_EL | unet_r34 | 3 | 76.10 ± 2.00 | 84.57 ± 2.36 | 80.08 ± 0.23 | 66.77 ± 0.31 | 67.06 ± 0.74 | 50.45 ± 0.84 |
| split_loco_ES | unet_r34 | 3 | 74.57 ± 4.95 | 91.86 ± 1.45 | 82.23 ± 2.64 | 69.87 ± 3.77 | 82.95 ± 0.48 | 70.87 ± 0.71 |
| split_loco_FR | unet_r34 | 3 | 71.68 ± 1.01 | 79.19 ± 0.42 | 75.25 ± 0.37 | 60.32 ± 0.48 | 68.89 ± 1.29 | 52.56 ± 1.51 |
| split_loco_IT | unet_r34 | 3 | 67.58 ± 2.77 | 77.29 ± 3.82 | 72.10 ± 3.07 | 56.43 ± 3.70 | 62.50 ± 1.51 | 45.46 ± 1.58 |
| split_time | unet_r34 | 3 | 79.13 ± 2.06 | 87.14 ± 0.66 | 82.93 ± 0.82 | 70.84 ± 1.20 | 76.16 ± 0.56 | 61.50 ± 0.73 |

## New years and countries, on the same test tiles: unet_r34

| Unseen in training | Test tiles | Fresh F1, trained on all | Fresh F1, trained without | Fresh F1, change | Burnt F1 (fresh + old), trained on all | Burnt F1 (fresh + old), trained without | Burnt F1 (fresh + old), change |
|---|---|---|---|---|---|---|---|
| country EL | 420 | 86.90 ± 1.60 | 86.16 ± 0.50 | -0.74 | 84.82 ± 1.44 | 81.27 ± 1.58 | -3.55 |
| country ES | 829 | 88.25 ± 1.24 | 83.11 ± 3.68 | -5.14 | 88.02 ± 0.77 | 87.37 ± 0.59 | -0.65 |
| country FR | 980 | 74.24 ± 0.97 | 68.17 ± 1.30 | -6.07 | 66.81 ± 1.63 | 63.96 ± 1.56 | -2.85 |
| country IT | 3191 | 77.96 ± 1.20 | 70.38 ± 2.71 | -7.57 | 69.77 ± 0.96 | 60.67 ± 1.48 | -9.1 |
| fires from 2025 on | 1414 | 83.73 ± 0.72 | 82.93 ± 0.82 | -0.8 | 77.19 ± 0.72 | 76.16 ± 0.56 | -1.02 |

## Best deep model by subgroup: segformer_b2

fresh_f1 and fresh_iou: fresh burn against the rest. burnt_f1 and burnt_iou: fresh and old burn together.

| stratum | value | tiles | fresh_f1 | fresh_iou | old_burn_iou | burnt_f1 | burnt_iou | fresh_px | burnt_px |
|---|---|---|---|---|---|---|---|---|---|
| country | EL | 420 | 88.54 | 79.46 | 65.35 | 86.99 | 76.98 | 2063114 | 3925611 |
| country | ES | 829 | 89.53 | 81.04 | 70.8 | 90.43 | 82.53 | 6919433 | 12956910 |
| country | FR | 980 | 76.05 | 61.35 | 42.35 | 71.02 | 55.07 | 3070740 | 6206654 |
| country | IT | 3191 | 80.3 | 67.08 | 45.53 | 73.36 | 57.93 | 14257875 | 29559321 |
| days_bin | [0, 30) | 1375 | 85.47 | 74.63 | 13.71 | 83.32 | 71.41 | 12282742 | 13320150 |
| days_bin | [150, 225) | 842 | 34.99 | 21.35 | 56.75 | 73.53 | 58.14 | 82531 | 8332035 |
| days_bin | [225, 400) | 1490 | 37.36 | 23.02 | 55.12 | 71.75 | 55.95 | 310012 | 14650599 |
| days_bin | [30, 60) | 1440 | 83.64 | 71.88 | 7.39 | 83.18 | 71.2 | 13402901 | 14250408 |
| days_bin | [60, 150) | 273 | 41.46 | 26.28 | 50.91 | 77.29 | 62.99 | 232976 | 2095304 |
| fire_season | Dec-Feb | 732 | 75.29 | 60.38 | 42.88 | 70.42 | 54.35 | 2215596 | 4682811 |
| fire_season | Jun-Aug | 3049 | 85.77 | 75.09 | 55.97 | 80.83 | 67.83 | 18705137 | 36840598 |
| fire_season | Mar-May | 849 | 77.89 | 63.8 | 41.53 | 74.26 | 59.06 | 2537012 | 5269634 |
| fire_season | Sep-Nov | 790 | 73.69 | 58.34 | 45.83 | 72.89 | 57.35 | 2853417 | 5855453 |
| fire_year | 2020 | 712 | 80.25 | 67.02 | 51.83 | 76.07 | 61.38 | 2993502 | 5916648 |
| fire_year | 2021 | 866 | 83.21 | 71.25 | 54.44 | 78.8 | 65.02 | 4292934 | 8996572 |
| fire_year | 2022 | 942 | 81.71 | 69.08 | 52.35 | 77.92 | 63.83 | 4221487 | 8830964 |
| fire_year | 2023 | 730 | 81.14 | 68.26 | 51.01 | 76.63 | 62.11 | 3279532 | 6971834 |
| fire_year | 2024 | 756 | 82.42 | 70.1 | 52.04 | 78.42 | 64.5 | 3139262 | 6631567 |
| fire_year | 2025 | 964 | 83.24 | 71.3 | 50.82 | 77.64 | 63.45 | 4823017 | 10610359 |
| fire_year | 2026 | 450 | 87.32 | 77.49 | 39.37 | 84.7 | 73.46 | 3561428 | 4690552 |
| land_cover | agriculture | 1123 | 83.28 | 71.36 | 49.67 | 76.6 | 62.08 | 6817956 | 13210423 |
| land_cover | artificial | 2 | 63.62 | 46.65 | 48.31 | 64.37 | 47.46 | 2128 | 4256 |
| land_cover | forest | 1166 | 84.17 | 72.66 | 56.29 | 81.03 | 68.12 | 4922526 | 9871448 |
| land_cover | other | 8 | 29.53 | 17.48 | 16.62 | 44.72 | 28.9 | 23257 | 55926 |
| land_cover | other_natural | 2193 | 81.4 | 68.63 | 49.5 | 76.87 | 62.43 | 10177615 | 20901418 |
| land_cover | shrubland | 928 | 84.07 | 72.53 | 55.01 | 81.31 | 68.51 | 4367680 | 8605025 |
| scene_season | Dec-Feb | 903 | 69.5 | 53.28 | 54.7 | 72.12 | 56.4 | 857678 | 9035326 |
| scene_season | Jun-Aug | 1778 | 85.2 | 74.21 | 40.92 | 79.65 | 66.18 | 12702425 | 17891173 |
| scene_season | Mar-May | 1381 | 77.08 | 62.71 | 60.75 | 78.34 | 64.4 | 3315560 | 12969352 |
| scene_season | Sep-Nov | 1358 | 82.96 | 70.88 | 37.33 | 80.69 | 67.64 | 9435499 | 12752645 |
| size_bin | 10-30 ha | 1775 | 70.95 | 54.98 | 35.01 | 63.85 | 46.89 | 3930249 | 8457081 |
| size_bin | 100-500 ha | 876 | 88.4 | 79.21 | 63.53 | 85.2 | 74.21 | 9567425 | 18410314 |
| size_bin | 30-100 ha | 1893 | 79.17 | 65.52 | 46.7 | 74.53 | 59.4 | 7308171 | 14684809 |
| size_bin | 5-10 ha | 697 | 67.83 | 51.33 | 29.1 | 58.53 | 41.37 | 1171625 | 2950542 |
| size_bin | over 500 ha | 179 | 92.89 | 86.73 | 78.52 | 93.87 | 88.44 | 4333692 | 8145750 |
| tile_filled | False | 5411 | 82.66 | 70.45 | 51.31 | 78.06 | 64.02 | 25992573 | 52074925 |
| tile_filled | True | 9 | 95.75 | 92.13 | 87.36 | 98.32 | 96.71 | 318589 | 573571 |
| window | fresh | 2825 | 84.49 | 73.15 | 10.19 | 83.26 | 71.32 | 25848498 | 27733655 |
| window | old | 2595 | 31.02 | 18.4 | 55.39 | 72.76 | 57.18 | 462664 | 24914841 |
