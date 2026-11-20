# Tools: how the release and the paper's results were made

These are the steps behind burn2scar and its data paper, as they were run on our server: steps 1 to 14 made
version 3.0, and step 15 takes it to versions 4.0 and 4.1, whose numbers the paper reports. Run them in
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

Open the `http://127.0.0.1:8882/lab?token=...` address shown in the `jupyter` window and open a terminal from the
launcher.

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

The one-person review of version 3.0 is withdrawn. The review behind version 4.1, by two interpreters and an
adjudicator, is in step 15.

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

## 15. Versions 4.0 and 4.1

Every number in the paper comes from these steps, on the v4 masks. The images are those of step 5.

### Label history from 2018, and the age rasters

`effis_extra` in `config.yaml` points at the EFFIS export from 1 January 2018 (`effis/`), which adds the perimeters
before the build export. The masks are patched with every perimeter since 2018, the index is rebuilt from them as in
step 3, and the age raster goes beside every mask:

```bash
python tools/burn_labels.py patch --config config.yaml --masks-dir data/v3 --out-dir data/v4 --workers 16
python tools/make_age.py --config config.yaml --masks-dir data/v4 --out-dir data/v4 --workers 16 \
    --index data/v4/index.csv --carry-from data/v3/index.csv
```

### Burn evidence (no Copernicus quota: Sentinel-2 L2A from the Earth Search archive on AWS)

```bash
python tools/burn_evidence.py --index data/v4/index.csv --masks-root data/v4 --out validation/dnbr --workers 16
python tools/burn_evidence.py --index data/v4/index.csv --masks-root data/v4 --out validation/dnbr --summarize --write-index
python tools/burn_evidence.py --index data/v4/index.csv --masks-root data/v4 --out validation/dnbr_old --window old --workers 16
python tools/dnbr_analysis.py --index data/v4/index.csv --fresh validation/dnbr --old validation/dnbr_old \
    --out validation/dnbr_analysis.md
```

### Models

```bash
bash tools/final_v4.sh
python tools/summarize_results.py --results results_v4 --index data/v4/index.csv
python tools/extra_scores.py --results results_v4 --index data/v4/index.csv --out results_v4/extra_scores.md
python tools/extra_inference.py --results results_v4 --index data/v4/index.csv --images-root data/v1 --masks-root data/v4 \
    --out results_v4/extra_inference.md
```

### The point review

700 points: 500 drawn on the v3 masks, and a supplement of 200 on the v4 masks for older_burn and more fresh_burn.
Two interpreters answer every point on their own, from the pictures alone. The dNBR under each point then flags the
answers it contradicts, and an adjudicator settles those and every disagreement. `review_points.py` explains the
design.

```bash
python tools/review_points.py sample --index data/v3/index.csv --masks-root data/v3 --out review2
python tools/review_points.py sample --supplement --index data/v4/index.csv --masks-root data/v4 --out review2 --seed 2027 \
    --counts older=50 fresh_early=40 fresh_mid=40 fresh_late=40 cloud=15 shadow=15
python tools/fetch_prefire.py --index data/v4/index.csv --points review2/points.csv --out data/prefire
python tools/review_points.py chips --out review2 --index data/v4/index.csv --images-root data/v1 --prefire-root data/prefire

# Each interpreter, on a page served here or in a folder that opens in any browser
python tools/review_points.py serve --out review2 --reviewer param --port 8765
python tools/review_offline.py --out review2 --reviewer silvia

# The dNBR under every point, and the adjudication
python tools/burn_evidence.py --index data/v4/index.csv --masks-root data/v4 --out validation/dnbr_review \
    --tiles-from review2/points.csv --save-dnbr --workers 24
python tools/review_dnbr.py --out review2 --dnbr validation/dnbr_review/dnbr --reviewers param silvia --index data/v4/index.csv
python tools/review_adjudicate.py --out review2 --reviewers param silvia --judge judge

# Scores, the two rules that bound the overall accuracy, and the release files
python tools/review_points.py score --out review2 --reviewers param silvia --judge judge
python tools/review_bounds.py --out review2 --reviewers param silvia --judge judge
python tools/review_points.py export --out review2 --reviewers param silvia --judge judge
```

`export` writes `review2/release/reviewed_points.csv` and `review_strata.csv`, with the answers named by role:
`answer_interpreter1`, `answer_interpreter2`, `answer_adjudicator` and `answer_final`.

### Figures and numbers of the paper

```bash
python tools/compare_cems.py --index data/v4/index.csv --masks-root data/v4 --out validation
python tools/burn_evidence.py --index data/v4/index.csv --masks-root data/v4 --out validation/dnbr_cems \
    --tiles-from validation/cems_tiles.csv --save-dnbr --workers 16
python tools/make_label_figures.py --out paper/figures --workers 16
python tools/make_paper_figures.py --out paper/figures
python tools/make_extra_figures.py --out paper/figures
python tools/paper_checks.py --index data/v4/index.csv --config config.yaml --out validation/paper_checks.md
```

`make_label_figures.py` draws the figures of the label-quality section and writes the numbers its text quotes to
`paper/figures/label_numbers.md`.

### Release 4.1

A new version of the Zenodo record, with a reserved DOI, then the sample built with that DOI, then the new files. The
draft needs the files of 4.0 too: if `newversion` says it holds none, press "Import files" on the draft page first.

```bash
python tools/make_release.py newversion --version 4.1 --creator "D'Aulerio, Paola" \
    --affiliation "European Space Agency, ESRIN" --after "Del Prete, Roberto"
python tools/make_sample.py --version 4.1 --doi <reserved DOI>
python tools/make_release.py upload --files release_docs/README.md release_docs/NOTICE.txt \
    review2/release/reviewed_points.csv review2/release/review_strata.csv release/sample.zip
python tools/make_release.py status
```

Publish on the Zenodo page, then bring the Hugging Face card up to date (the masks and index are those of 4.0):

```bash
python tools/upload_hf.py plan --version 4.1 --tag v4.0
python tools/upload_hf.py run --version 4.1 --tag v4.0
```

## 16. Version 5.0

The evaluation additions, the referee's checks, the independent review and the comparison with Sardinia's fire
perimeters, then the release. The steps ran on one GPU server; `overnight_v5.sh` chains the long ones.

```bash
python tools/check_v5.py                                   # packages, inputs, effis_extra, trained runs
python tools/acquire_additions.py plan                     # 2,492 targets; then run, about 6,500 Sentinel Hub requests
python tools/acquire_additions.py run
python tools/finish_additions.py --workers 16 --zip        # labels, ages, additions_index.csv, additions.zip
bash tools/overnight_v5.sh                                 # referee_checks.py parts, score_models.py, the blind review page
python tools/review_blind.py score --reviewer <name>       # after the independent interpreter's answers are in review2/
python tools/agency_sardinia.py                            # CFVA perimeters 2020-2025 against EFFIS and the labels
python tools/score_models.py agency --models unet_r34 segformer_b2 --seeds 0 1 2 3 4
python tools/make_v5_figures.py                            # fig_additions.pdf
```

Release: make a new version of the data record and of the code record on the Zenodo pages, reserving a DOI for each
("Get a DOI now!"), and import the files of 4.1 into the data draft. Then:

```bash
python tools/release_v5.py prepare                         # data/release/v5/: index, new tables, README, licence, notice
python tools/release_v5.py adopt --id <data draft id>      # make_release.py now works on that draft
python tools/make_sample.py --version 5.0 --doi <data DOI>
python tools/make_release.py upload --index data/release/v5/index.csv --only index
python tools/make_release.py upload --index data/release/v5/index.csv --files data/additions/additions.zip \
    data/release/v5/burn_evidence_old_same_season.csv data/release/v5/reviewed_points_independent.csv \
    data/release/v5/README.md data/release/v5/NOTICE.txt release/sample.zip
python tools/make_release.py status --index data/release/v5/index.csv
python tools/upload_hf.py plan --version 5.0 --tag v4.1 --release-index data/release/index.csv
python tools/upload_hf.py run --version 5.0 --tag v4.1 --release-index data/release/index.csv
python tools/release_models.py build --spec unet_r34 --results results_v4
python tools/release_models.py upload --spec unet_r34 --repo neet1797/burn2scar-unet-r34
python tools/release_models.py export --spec unet_r34 --repo neet1797/burn2scar-unet-r34
```

The code record takes the archive of the `v5.0` tag, uploaded by hand to its draft; the tag is pushed without a
GitHub release, so the Zenodo integration does not make a second record.

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
| `make_age.py` | Days-since-fire raster beside every mask, and three index flags |
| `burn_evidence.py` | dNBR burn evidence of every fire, from Sentinel-2 L2A on Earth Search |
| `dnbr_analysis.py` | The burn evidence by threshold, size, country, season and time since the fire |
| `review_offline.py` | The point review as a folder that opens in any browser |
| `review_dnbr.py` | dNBR under every reviewed point, and the answers it contradicts |
| `review_adjudicate.py` | The adjudication, as a folder that opens in any browser |
| `review_bounds.py` | Overall accuracy of the review under the two rules that bound it |
| `make_label_figures.py` | Figures and numbers of the paper's label-quality section |
| `make_extra_figures.py` | Burn-evidence and prediction figures |
| `extra_scores.py`, `extra_inference.py` | Scores from the saved confusion matrices; the older_burn and off-centre tests |
| `age_confusion.py` | Image or calendar: the age accuracy against a month rule |
| `compare_agency.py` | EFFIS and the labels against a regional agency's fire perimeters |
| `paper_checks.py` | Numbers the paper needs from the index and the EFFIS export |
| `make_sample.py` | The 200-tile `sample.zip` of the release |
| `final_v4.sh`, `phase2.sh` | The final model runs on the v4 masks, and the steps queued after them |
| `check_v5.py` | Checks the server before the version 5.0 runs |
| `acquire_additions.py`, `finish_additions.py`, `fetch_images.py` | The evaluation additions: targets and download; labels, ages, index and zip; image fetch from the Hugging Face copy |
| `referee_checks.py` | The numbers the referee asked for: cloud, history, water, zeros, none flags, splits, managed burns, new index columns |
| `score_models.py` | The baselines on the additions, against CEMS, at the reviewed points and on Sardinia's agency-only burn |
| `review_blind.py`, `review_dates.py` | The independent date-blind review (pictures, page, score) and the review scored on dates |
| `agency_sardinia.py` | Downloads Sardinia's CFVA perimeters and runs `compare_agency.py` on them |
| `make_v5_figures.py` | The additions figure |
| `overnight_v5.sh` | The long version 5.0 steps in one run |
| `release_v5.py` | The files of release 5.0, and the link to its Zenodo draft |
