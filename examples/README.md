# Example tiles

One fire from the test set of burn2scar's main split, so the released model never saw it in training: EFFIS fire `effis_280331`, Italy, 46 ha, dated 2025-08-30. Two Sentinel-2 Level-1C scenes of it, each with its mask.

| Image | Mask | Scene | Days after the fire | fresh_burn | old_burn | older_burn |
|---|---|---|---|---|---|---|
| `effis_280331_2025-09-08T10-00-18_s2.tif` | `effis_280331_2025-09-08T10-00-18_mask.tif` | 2025-09-08 | 9 | 28% | 1% | 44% |
| `effis_280331_2026-04-11T10-00-17_s2.tif` | `effis_280331_2026-04-11T10-00-17_mask.tif` | 2026-04-11 | 224 | 0% | 29% | 44% |

Images: 13 bands (B01 to B12, with B8A after B08), top-of-atmosphere reflectance x 10,000, uint16, 256 x 256 pixels at 10 m. Masks: 0 clear, 2 fresh_burn, 3 old_burn, 4 cloud, 5 cloud shadow, 6 water, 7 older_burn, 255 no data. `index.csv` holds the two rows of the release index.

Run the baseline model on one of them, from the repository root:

```bash
python tools/predict.py --image examples/effis_280331_2025-09-08T10-00-18_s2.tif --out predictions/effis_280331_2025-09-08T10-00-18_pred.tif
```

## Licence

These files are part of [burn2scar](https://doi.org/10.5281/zenodo.23986827) and are licensed under CC BY 4.0, not under the code's Apache 2.0. Credit:

```
Contains modified Copernicus Sentinel data 2020-2026.
Burn labels adapted from EFFIS, Copernicus Emergency Management Service,
© European Union, 1995-2026, CC BY 4.0.
burn2scar: Thind, P. K., Cadau, E., Del Prete, R., Ricolfi, S., Meoni, G., and Taramelli, A., 2026,
https://doi.org/10.5281/zenodo.23986827
```
