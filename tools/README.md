# Tools: how the release and the paper's results were made

These are the steps behind burn2scar 3.0 and its data paper, as they were run on our server. Run them in
order from the repo root. Everything is safe to stop and restart. Nothing
here changes the published dataset: new files go under `data/`, `results/`, `review/` and `validation/`.

Anything marked long belongs in the tmux `runs` window. Commands below use `--workers 8`, so change that to
suit your CPU count (`nproc`).

Disk needed: 30 GB for images, 48 GB for the training cache and about 15 GB for results. The independent maps
need 18 GB during download only, so step 4 removes those files before the cache is built.

## Setup

The environment lives in `.venv` in this folder. Install a PyTorch build that matches your GPU driver first,
then:

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -r environment/pipeline.txt -r environment/tools.txt jupyterlab
```

Start the workspace once, from an SSH shell on the server. It makes a tmux session called `b2s`, with
Jupyter Lab in one window and a shell for long runs in another. `.env` holds your Hugging Face and Copernicus
keys, and every terminal gets them.

```bash
cd ~/projects/burn2scar
ACTIVATE="source .venv/bin/activate"
START="$ACTIVATE && { set -a; source .env; set +a; }"
tmux new-session -d -s b2s -n jupyter
tmux send-keys -t b2s:jupyter "$START && unset TMUX TMUX_PANE && jupyter lab --no-browser --ip=127.0.0.1 --port=8882 --port-retries=0 --allow-root" C-m
tmux new-window -t b2s -n runs
tmux send-keys -t b2s:runs "$START" C-m
tmux select-window -t b2s:jupyter
tmux attach -t b2s
```

Jupyter listens only on the server itself, so reach it through SSH from your laptop. The second port is
for the label review page.

```bash
ssh -L 8882:localhost:8882 -L 8765:localhost:8765 you@your-server
```

Open the `http://127.0.0.1:8882/lab?token=...` address shown in the `jupyter` window. Upload the zip with the
file panel, open a terminal from the launcher, and run:

```bash
python -m zipfile -e burn2scar_v2_tools.zip .
git status --short
git checkout -b v2
git apply --3way pipeline_label_fix.patch
git status --short
```

Check the last `git status`: the changed files should now be listed with `M`. If `git apply` printed any `error:`
line, it applied nothing, even if it also printed "Applied patch" lines. When a file has local changes, leave it out
of the first command and apply it to the working copy alone, which keeps your edits:

```bash
git apply --3way --exclude=HF_DATASET_CARD.md pipeline_label_fix.patch
git apply --include=HF_DATASET_CARD.md pipeline_label_fix.patch
```

Short commands can run in any Jupyter terminal. For long runs, type `tmux attach -t b2s:runs` in a Jupyter
terminal first, so they keep going even if Jupyter restarts. Ctrl+b then d leaves the window running.

## 1. Audit the masks (2 minutes, no downloads)

```bash
python tools/burn_labels.py audit --config config.yaml --out audit
```

Read `audit/summary.md`. It warns if the export covers only the selected countries or dates. On the published
export that hides an estimated 0.2-0.5% of burn pixels, from 2019 fires, so the rest of this guide runs without a
wider one. To add one, follow these steps and rerun the steps below. Every tool and the pipeline read `effis_extra`,
so the masks and the code stay in agreement. The layer changes over time, so keep the file and its download date
with the release.

Download it in a browser, because the link below refused `wget` in November 2026, and save it as
`effis/effis_layer.zip`. If the browser is refused too, use the
[EFFIS Data Request Form](https://forest-fire.emergency.copernicus.eu/apps/data.request.form) with all countries and
the earliest dates it offers: https://forest-fire.emergency.copernicus.eu/downloads/data.request.form/effis_layer.zip

```bash
date -u +%F > effis/effis_layer.downloaded_on.txt
sed -i 's|^effis_extra:.*|effis_extra: [effis/effis_layer.zip]|' config.yaml
python tools/burn_labels.py audit --config config.yaml --out audit
```

## 2. Rebuild the masks (no imagery, `audit/summary.md` gives the download size)

```bash
python - <<'EOF'
from huggingface_hub import snapshot_download
snapshot_download("neet1797/burn2scar", repo_type="dataset", allow_patterns="*_mask.tif", local_dir="data/v1")
EOF
python tools/burn_labels.py patch --config config.yaml --masks-dir data/v1 --out-dir data/v2 --workers 8
```

If Hugging Face stops the download with a rate limit, run it again: it carries on from where it
stopped. Read `data/v2/patch_summary.md`. "Pixels that disagree with the export" should be near
zero. If it isn't, this EFFIS file isn't the one the masks were built from, so stop and check.

## 3. Build the index and splits (minutes)

```bash
python tools/build_index.py --config config.yaml --masks-dir data/v2 --out data/v2/index.csv \
    --patch-report data/v2/patch_report.csv --workers 8
```

Writes `data/v2/index.csv` and `data/v2/index_summary.md`. The summary has the split tables for the paper.

## 4. Comparison with independent maps (18 GB during download)

ESSD asks for validation against independent data. Copernicus EMS analysts map fires separately
from EFFIS, and `wildfires-cems` (Arnaudo et al. 2023, CC BY 4.0) holds those maps for 2017 to early 2023.
This needs the masks but not the images, and it keeps only the files the comparison reads.

```bash
python - <<'EOF'
from huggingface_hub import snapshot_download
snapshot_download("links-ads/wildfires-cems", repo_type="dataset", allow_patterns="data/*", local_dir="data/cems_raw")
EOF
mkdir -p data/cems && cd data/cems
for part in train val test; do cat ../cems_raw/data/$part/$part.tar.* | tar -xzf - -i --wildcards '*_DEL.tif' '*_S2L2A.json'; done
cd ../..
rm -rf data/cems_raw
python tools/compare_cems.py
```

Each area's activation date and country come from `satelliteData.csv`, the table kept by the authors of the
CEMS dataset. The tool fetches it once from a fixed commit (about 190 KB, no imagery). An EFFIS fire counts as
the same event if it is dated 60 days before to 15 days after that date, and the summary shows how much other
windows change the result. Results are in `validation/cems_summary.md`.

## 5. Download the images and build the cache (long)

This comes from your own Hugging Face dataset and costs no Copernicus processing units.

```bash
python - <<'EOF'
from huggingface_hub import snapshot_download
snapshot_download("neet1797/burn2scar", repo_type="dataset", allow_patterns="*_s2.tif", local_dir="data/v1")
EOF
python tools/build_cache.py --index data/v2/index.csv --images-root data/v1 --masks-root data/v2 --out data/cache --workers 8
```

The cache holds the same tiles as plain NumPy files. A tile reads about 7 times faster, and training gives identical
results. Run it again if the masks change. Once the baselines finish, `rm -rf data/cache` frees its 48 GB.

## 6. Baselines

Trial first. It takes under an hour, downloads all the pretrained weights and proves every model
fits your GPUs. If one runs out of memory, add `--batch-scale 0.5`.

```bash
python tools/run_baselines.py --gpus 0 1 --seeds 0 --skip-generalisation --limit 200 --epochs 2 --out results_trial --cache-root data/cache
python tools/run_baselines.py --gpus 0 1 --estimate results_trial
python tools/classic_baselines.py --limit 100 --seeds 0 --n-pixels 100000 --svm-pixels 20000 --out results_trial --workers 8 --cache-root data/cache
python tools/summarize_results.py --results results_trial
```

Then the full runs (long). The first command gives the runs the ESSD paper needs, including the time and country
tests. The second adds the rest and skips finished runs. On a server with few cores, run the classic baselines before
or after the GPU runs, not alongside them: they use every core. `--methods spectral RF` runs only what the paper needs.

```bash
python tools/run_baselines.py --gpus 0 1 --specs unet_r34 segformer_b2 --cache-root data/cache
python tools/run_baselines.py --gpus 0 1 --cache-root data/cache
python tools/classic_baselines.py --workers 8 --cache-root data/cache
python tools/classic_baselines.py --workers 8 --cache-root data/cache --target fresh
python tools/summarize_results.py --results results
```

The main task in the tables is fresh burn against everything else, old burn included, as for an alert
on new fires. The deep models give it from the same runs; the classic baselines need the run with
`--target fresh`. The tables also keep fresh and old burn together, for reference.

| What runs | Runs |
|---|---|
| 10 spectral indices with one threshold each | 10 |
| Random Forest, XGBoost, SVM, MLP, 5 seeds each | 20 |
| 8 deep models, 5 seeds each, on the spatial split | 40 |
| U-Net ResNet-34, 3 seeds each, on the time split and each held-out country | 15 |

The time and country runs show what a model loses on a year or country it never saw. `results/summary.md` compares
them with the main-split model on the same test tiles.

Each run saves its best weights, so allow about 15 GB under `results/`. The tables for the
paper are in `results/summary.md`, `comparison.csv`, `per_class.csv` and `stratified.csv`.

### How this compares with FLOGA

The protocol matches FLOGA: same family of methods, 60/20/20 split, burnt and unburnt precision,
recall, F1 and IoU, mean F1 and mean IoU, best epoch chosen by burnt F1 on validation, 5 repeats.

| FLOGA method | Here, with one image |
|---|---|
| dNBR, dNBR2, dNBR+, dMIRBI, dBAIS2, dNDVI, dNDWI, dEVI, dCSI, dSAVI | The same indices on one date. RdNBR and RBR need two dates, so they are left out. |
| Random Forest, XGBoost, SVM, MLP | The same four, on one date. SVM uses an approximated RBF kernel. |
| FC-EF-Conc, FC-EF-Diff | U-Net ResNet-18 from scratch, and U-Net ResNet-34 |
| SNUNet-CD | U-Net++ ResNet-34 |
| BAM-CD | U-Net++ ResNet-101 |
| ADHR-CDNet | U-Net with HRNet-W18 |
| ChangeFormer | SegFormer MiT-B2 |
| Not in FLOGA | DeepLabV3+ ResNet-50, and U-Net ResNet-50 with Sentinel-2 SSL4EO-S12 weights |

The two sets of numbers must not be ranked against each other. FLOGA sees before and after images
of Greece at 20 m. These models see one L1C image of four countries at 10 m. A true head-to-head
needs pre-fire images for every fire (see step 8). Leave the FLOGA column out of the ESSD paper, which
leaves comparisons of methods and datasets to review articles. It's there for the data card or a later article.

## 7. Label review

A person judges random pixels from the imagery alone, without seeing the masks. Pre-fire images make the
judgement much easier, so fetch them for the sampled fires. Check the fetch first: `--verify` downloads 3
images you already have and compares them with your files.

```bash
python tools/review_points.py sample --workers 8
python tools/fetch_prefire.py --verify 3
python tools/fetch_prefire.py --points review/points.csv --out data/prefire
python tools/review_points.py chips --prefire-root data/prefire --workers 8
python tools/review_points.py serve --reviewer param
```

The fetch costs about 1.08 PU per image, a few thousand PU in all. On your laptop run
`ssh -L 8765:localhost:8765 you@your-server`, then open http://localhost:8765. Keys 1 to 6 answer, and
Backspace goes back. Each answer saves at once.

The sample is about 1,630 points in random order, about 2¼ hours at 5 seconds each. You can stop at any
point and still get valid numbers, with wider intervals:

| Points answered | Time | Points per burn class | Interval at 90% agreement |
|---|---|---|---|
| 500 | 40 minutes | about 90 | ±6 |
| 1,000 | 1.4 hours | about 180 | ±4.3 |
| 1,630 | 2.3 hours | 300 | ±3.4 |

For the journal's uncertainty section, a second reviewer helps a lot. Each person uses `--part 1/2`
or `--part 2/2`, and they both answer the first 100 points, which measures how often two people agree.

```bash
python tools/review_points.py score
```

Writes `review/review_summary.md`: answers by class, area-weighted precision, recall, F1 and IoU with
95% intervals, confirmation rates by country, window and year, and reviewer consistency.

## 8. Pre-fire images for every fire (optional)

This makes burn2scar a before-and-after dataset, and allows a direct comparison with FLOGA's dNBR and change
detection models. It needs about 15,000 PU or more and at least 14,000 requests, and fires already fetched
for the review are skipped. The Copernicus general-user limit is 10,000 PU and 10,000 requests a month, so
check your quota first. `--max-pu` stops the run at that amount, and running it again continues.

```bash
python tools/fetch_prefire.py --all --out data/prefire --max-pu 5000
```

## 9. Product IDs (a few hours, no PU, no Sentinel Hub requests)

```bash
python tools/add_product_ids.py --compare 20
python tools/add_product_ids.py
```

Looks up each tile in the public Copernicus Data Space catalogue (OData), which needs no account. `--compare`
first checks 20 tiles against the Sentinel Hub catalogue the pipeline read from, using 20 Sentinel Hub requests.
The full run adds the source products and their processing baselines to the index once every tile has a lookup.
Where two satellites passed minutes apart, both passes are listed: the pipeline asked Sentinel Hub
for 3 hours either side of each scene, with the most recent acquisition on top.

To publish the label review with the data set:

```bash
python tools/review_points.py export --out review_final
```

## 10. Record the environment

```bash
pip freeze > environment/paper-gpu.lock.txt
```

The journal needs the exact versions, and the requirement files give only lower bounds. The lock files in
`environment/` start with a comment saying which machine and Python they come from.

## 11. Hugging Face copy (about an hour, safe to stop and restart)

```bash
python -c "from huggingface_hub import login; login()"
python tools/upload_hf.py plan
python tools/upload_hf.py run
python tools/upload_hf.py check
```

`run` first tags the repository as it is `v1`, so version 1 stays downloadable. It then uploads the version 3.0
masks that differ from the Hub, 1,000 to a commit, and in a last commit adds `index.csv`, the dataset card,
`LICENSE.txt` and `NOTICE.txt`, and removes the tiles left out of the release. `plan` changes nothing.

## 12. Release the baseline model (minutes, about 100 MB)

```bash
python tools/release_models.py build --spec unet_r34
python tools/release_models.py upload --spec unet_r34 --repo neet1797/burn2scar-unet-r34
python tools/release_models.py check --spec unet_r34 --repo neet1797/burn2scar-unet-r34
```

`build` takes the U-Net seed with the median fresh-burn F1 on the main split's test set, a typical run rather
than the best, and writes `release_models/burn2scar-unet-r34/`: the weights as `model.safetensors`, `config.json`
with the bands, scaling and class values, `metrics.json` with every seed's scores, the model card, `predict.py`
and the licence. Read the model card before `upload`, which makes a public model repo. `check` downloads it again,
compares every file and runs the model.

Only U-Net ResNet-34 is released, under Apache 2.0. SegFormer-B2 is not: its encoder starts from NVIDIA's MiT
ImageNet weights, whose licence allows only non-commercial use of anything derived from them.

Then put the model in the code repository, without its weights:

```bash
python tools/release_models.py export --spec unet_r34 --repo neet1797/burn2scar-unet-r34
```

`export` writes `models/burn2scar-unet-r34/`: `config.json` with the Hub repo and the weights' sha256 added,
`metrics.json`, and the model card with the repository's commands. `predict.py` fetches the weights into that
folder on first use and checks them.

`predict.py` runs a released model on any 13-band Level-1C GeoTIFF, of any size; `--model` defaults to
`models/burn2scar-unet-r34`:

```bash
python tools/predict.py --image tile_s2.tif --out tile_pred.tif
```

## 13. Random gallery (minutes, CPU is enough)

```bash
python tools/make_gallery.py --model release_models/burn2scar-unet-r34 --out paper/gallery
```

Eight test tiles drawn at random with a fixed seed, not chosen by hand: true colour, mask and prediction, with
the share of pixels where they agree. Writes `gallery.png`, `gallery.pdf` and `gallery.json`.

## 14. README picture, example fire and notebooks (minutes)

```bash
python tools/make_readme_figure.py
python tools/make_examples.py
jupyter nbconvert --to notebook --execute --inplace notebooks/01_explore_burn2scar.ipynb notebooks/02_run_the_model.ipynb
```

`make_readme_figure.py` draws `assets/burn2scar_examples.png`: one fresh scene per country and two old scenes,
at random among tiles with at least 15 % burn. `make_examples.py` copies one fire from the test set, both scenes
and their masks, into `examples/`, with its rows of the index and a README. Executing the notebooks saves their
figures in them, so GitHub shows them. They download their tiles from Hugging Face, or read an unzipped Zenodo
release when `BURN2SCAR_ROOT` points to it.

`results/` stays on the server and out of git: the paper's tables come from it, and the model card carries the
released model's scores.

## What each new mask class means

| Value | Class | Meaning |
|---|---|---|
| 0 | clear | None of the classes below |
| 2 | fresh_burn | EFFIS scar, 0 to 90 days before the scene |
| 3 | old_burn | EFFIS scar, over 90 and up to 365 days before the scene |
| 4 | cloud | OmniCloudMask |
| 5 | shadow | OmniCloudMask |
| 6 | water | NDWI or Scene Classification Layer |
| 7 | older_burn | **New.** EFFIS scar, over 365 days before the scene, where the mask would otherwise say clear |
| 255 | nodata | No data in the source image |

Training and all metrics ignore class 7 and nodata. Cloud and shadow are trained on, but left out of
the burnt-class metrics, since the ground under them can't be seen.

## Files

| File | Does |
|---|---|
| `burn_labels.py` | Audit and patch the masks |
| `build_index.py` | Per-tile index, overlap groups and splits |
| `build_cache.py` | Plain NumPy copies of the tiles, for faster training |
| `train_baseline.py`, `run_baselines.py` | Deep models |
| `classic_baselines.py` | Spectral indices and pixel classifiers |
| `summarize_results.py` | Tables for the paper |
| `review_points.py` | Label review: sample, pictures, page, score, and export for the release |
| `compare_cems.py` | Labels against Copernicus EMS maps |
| `make_paper_figures.py` | Figures and numbers for the manuscript |
| `make_release.py` | Zenodo draft with a reserved DOI, and the upload. It never publishes |
| `fix_winter_water.py` | Removes false water from November to February masks |
| `check_review.py` | Where a finished review confirmed burn, and a dNBR check at every point |
| `fetch_prefire.py` | Pre-fire images |
| `add_product_ids.py` | Source products, from the public catalogue |
| `upload_hf.py` | Brings the Hugging Face copy up to the release, after tagging the old one `v1` |
| `release_models.py` | The baseline model on Hugging Face, with its model card |
| `predict.py` | Runs a released model on Level-1C images |
| `make_gallery.py` | Random test tiles with mask and prediction |
| `make_readme_figure.py` | The picture at the top of the README |
| `make_examples.py` | One test fire, both scenes and masks, into `examples/` |
