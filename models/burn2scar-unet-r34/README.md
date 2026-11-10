# burn2scar U-Net (ResNet-34)

A baseline model trained on [burn2scar](https://doi.org/10.5281/zenodo.23986827) version 3.0. It takes one post-fire
Sentinel-2 Level-1C image and gives every pixel one of six classes: clear, fresh_burn (burned up to 90 days
before the image), old_burn (90 to 365 days before), cloud, cloud shadow and water.

It is one of the baselines of the burn2scar data paper, there to show that the labels can be learned. It was
not tuned to be the best possible model.

## Use

From the root of the code repository, after `pip install -r requirements.txt`:

```bash
python tools/predict.py --model models/burn2scar-unet-r34 --image tile_s2.tif --out tile_pred.tif
```

```python
import sys
sys.path.insert(0, "tools")
from predict import load_model, predict_array
net, config = load_model("models/burn2scar-unet-r34")
mask = predict_array(net, config, image)  # image: (13, height, width), reflectance x 10,000
```

The first use downloads the weights, `model.safetensors` (98 MB), from
[huggingface.co/neet1797/burn2scar-unet-r34](https://huggingface.co/neet1797/burn2scar-unet-r34) into this folder and checks them against the sha256 in
`config.json`. `notebooks/02_run_the_model.ipynb` runs the model on random test tiles and draws the results.

**Input:** the 13 Sentinel-2 Level-1C bands in the order B01, B02, B03, B04, B05, B06, B07, B08, B8A, B09, B10, B11, B12, as top-of-atmosphere
reflectance times 10,000 on a 10 m grid, the way burn2scar stores its images. Pixels where every band is 0
are no data. Any image size works; predict.py reads it in 256 x 256 windows, the size of a burn2scar tile.

**Output:** burn2scar mask values: 0 clear, 2 fresh_burn, 3 old_burn, 4 cloud, 5 cloud shadow, 6 water and
255 no data. The model never predicts 7 (older_burn), which was left out of training.

## Scores

Test set of burn2scar's spatial split, 5,420 tiles. This model is seed 1, the median of
5 seeds by fresh-burn F1. Percent.

| Measure | This model | 5 seeds, mean ± sd |
|---|---|---|
| Fresh burn F1 | 81.20 | 80.78 ± 1.17 |
| Fresh burn precision | 75.27 | 74.26 ± 2.53 |
| Fresh burn recall | 88.13 | 88.64 ± 1.05 |
| Fresh burn IoU | 68.34 | 67.77 ± 1.63 |
| Burnt F1, fresh and old together | 75.30 | 74.85 ± 1.06 |
| Fresh vs old accuracy | 95.85 | 95.89 ± 0.55 |
| Mean IoU, six classes | 71.02 | 70.44 ± 1.37 |

Fresh burn is scored against clear, old_burn and water. "Fresh and old together" scores both burn classes
against clear and water. Fresh vs old accuracy is the share of pixels labelled and predicted as burn that get
the right age. Cloud, shadow, older_burn and no-data labels are left out of all of these.

Per-class IoU of this model, each class against all others:

| Class | IoU |
|---|---|
| clear | 89.57 |
| fresh_burn | 68.21 |
| old_burn | 46.26 |
| cloud | 70.70 |
| shadow | 66.94 |
| water | 84.44 |

## Training

- Data: burn2scar version 3.0, column `split` of index.csv: 16,183 training and 5,243 validation tiles. The split keeps overlapping fires apart in 0.5 degree blocks.
- Input scaling: reflectance (value / 10,000) clipped to 0 to 1.5, then normalised per band with the mean and standard deviation of 1,000 training tiles (both in config.json).
- The encoder started from torchvision's ImageNet ResNet-34 weights (BSD-3-Clause), as distributed with segmentation-models-pytorch. The first layer takes 13 bands.
- Random 192 x 192 crops, flips and 90 degree turns. AdamW, learning rate 3e-4, weight decay 1e-4, batch 32, one warm-up epoch then cosine decay, at most 60 epochs, stopped after 15 without gain. Loss: cross entropy weighted by the inverse square root of class frequency, plus Dice. The kept epoch is the best by validation F1 of fresh and old burn together.
- This run: 60 epochs, best at epoch 52.

## Limits

- Trained on four Mediterranean countries (Italy, Spain, France south of 46° N, Greece), fires from 2020 to August 2026 of 5 to 2,000 ha. Scores fall on countries a model has not seen: in the data set's tests, leaving Spain, France or Italy out of training cost 5 to 8 points of fresh-burn F1.
- The labels follow EFFIS perimeters, so they share EFFIS's omissions and outline errors. old_burn is set by the date of the fire and is often no longer visible.
- Small fires and burn edges are harder for the model, as for any method.
- Level-1C only. Atmospherically corrected (Level-2A) images give different values and are not what it was trained on.

## Licence and credits

Model weights: Apache 2.0. Trained on burn2scar (CC BY 4.0), which contains modified Copernicus Sentinel
data 2020–2026 and burn classes adapted from EFFIS, © European Union, 1995–2026, CC BY 4.0.

Code that built the data and trained the model: https://github.com/paramkaur10/burn2scar_dataset_creation

## Citation

Thind, P. K., Cadau, E., Del Prete, R., Ricolfi, S., Meoni, G., and Taramelli, A.: burn2scar: a Sentinel-2 burn-scar segmentation data set, version 3.0, Zenodo [data set], https://doi.org/10.5281/zenodo.23986827, 2026.
