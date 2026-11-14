# Revision package 1 (11 November): the fixes that need no new imagery

Unzip in the repository root. Four files replace or join `tools/`:

| File | Replaces | What it does |
|---|---|---|
| `tools/review_points.py` | the old one | The point review, second edition: two reviewers, blind windows, error matrix |
| `tools/train_baseline.py` | the old one | Adds `--bands` for sensors without shortwave infrared |
| `tools/classic_baselines.py` | the old one | Pixel classifiers get a validation-tuned probability threshold |
| `tools/make_age.py` | new | Days-since-fire raster beside every mask, and three index flags |
| `tools/age_confusion.py` | new | Image or calendar: the age accuracy on mixed tiles against a month rule |

All tested here on a fake release (sample, chips, two simulated reviewers, a judge, score, export; the age raster on one tile
with synthetic perimeters; the age table on fake runs). Nothing touched the real data, so run each once with `--limit` first.

## 1. The point review (start this first: it is the only step that needs people)

The old review is dropped. Delete `review_final/` from the paper and from the next release; keep the folder on disk.

```bash
# 1. Sample: 500 points, strata fresh_early/mid/late 50 each, old 100, clear_near 75, clear_far 75, water 35, cloud 35, shadow 30.
python tools/review_points.py sample --index data/v3/index.csv --masks-root data/v3 --out review2
#    For 600 points: --counts fresh_early=60 fresh_mid=60 fresh_late=60 old=120 clear_near=90 clear_far=90 water=40 cloud=40 shadow=40
#    To include older_burn: --counts older=20

# 2. Pre-fire image for every sampled fire. About one Sentinel Hub request per fire, under 500 in all.
python tools/fetch_prefire.py --index data/v3/index.csv --points review2/points.csv --out data/prefire

# 3. Pictures. Each shows up to three dates in time order, captioned by date only, a 640 m window and a 1.9 km window, all centred on the point.
python tools/review_points.py chips --out review2 --index data/v3/index.csv --images-root data/v1 --prefire-root data/prefire

# 4. Two reviewers, independently, each on their own port and in their own random order. Do not discuss points while reviewing.
python tools/review_points.py serve --out review2 --reviewer param --port 8765
python tools/review_points.py serve --out review2 --reviewer silvia --port 8766
#    Laptop: ssh -L 8765:localhost:8765 you@server  (and 8766 for the second), then open http://localhost:8765

# 5. Adjudicate the disagreements: a third person, or the two of you together, on a third port.
python tools/review_points.py serve --out review2 --reviewer judge --disagreements param silvia --port 8767

# 6. Score and export. The release gets both reviewers' answers and the final label per point.
python tools/review_points.py score --out review2 --reviewers param silvia --judge judge
python tools/review_points.py export --out review2 --reviewers param silvia --judge judge
```

The answers have seven keys: 1 burned; 2 burned, and the scar is already on the earliest date shown; 3 not burned; 4 water;
5 cloud; 6 shadow; 7 can't tell. Key 2 is the date check: at a fresh_burn point it means the scar predates the pre-fire image.

`review2/review_summary.md` holds, for the final labels and for each reviewer: the per-stratum table, the area-weighted error
matrix (map class against what the reviewer saw), overall, user's and producer's accuracy with bootstrap intervals, burned
precision, recall, F1 and IoU, the groups (window, country, fire size, days, edge distance, season, year, land cover) with Wilson
intervals, agreement between the two reviewers with Cohen's kappa, and each reviewer's repeat consistency.

Expect 30 to 40 seconds a point with the pre-fire image, so 4 to 6 hours each for 500 points. The old review ran at 5 seconds
a point; a referee will compare.

## 2. The free fixes to the baselines

```bash
# A6: pixel classifiers with a tuned threshold. Delete the old RF runs first, or use a new --out.
rm -rf results/RF__split__*            # the five random-forest runs
python tools/classic_baselines.py --index data/v3/index.csv --images-root data/v1 --masks-root data/v3 --out results --methods RF --target fresh
python tools/classic_baselines.py --index data/v3/index.csv --images-root data/v1 --masks-root data/v3 --out results --methods RF --target burn
#    metrics.json now holds "threshold" and the scores at 0.5, so the paper can give both.

# A5: bands without shortwave infrared. Three seeds, two models, two band sets = 12 runs. Give me HEO's band list for a third set.
for seed in 0 1 2; do
  for spec in unet_r34 segformer_b2; do
    for bands in vnir rgbnir; do
      python tools/train_baseline.py --spec $spec --seed $seed --bands $bands --index data/v3/index.csv --images-root data/v1 --masks-root data/v3 --out results
    done
  done
done
#    The runs are named unet_r34_vnir etc., so summarize_results.py lists them as their own methods.

# A4: image or calendar. Needs only the finished runs.
python tools/age_confusion.py --results results --index data/v3/index.csv --runs segformer_b2 unet_r34 --out results/age_confusion.md
```

## 3. The age raster and the longer label history (free of imagery)

First request a new EFFIS export covering 1 January 2018 (or the start of the record) to 3 August 2026 for Italy, Spain,
France and Greece (the text is in the Revision plan tab). Then:

```bash
# Regenerate the masks with the full history: burn from every perimeter, older_burn where a scar predates the window.
# effis_extra in config.yaml, or --effis-extra, points at the new export. Water, cloud, shadow and nodata stay as in v3.
python tools/burn_labels.py patch --config config.yaml --masks-dir data/v3 --out-dir data/v4 --effis-extra effis/<new export>.zip

# The age raster beside every mask, plus age_path, burn_age_median, winter_fire, agricultural and modis_mapped in the index.
cp data/v3/index.csv data/v4/index.csv
python tools/make_age.py --config config.yaml --masks-dir data/v4 --out-dir data/v4 --effis-extra effis/<new export>.zip --index data/v4/index.csv
```

Then `build_index.py` for the class shares of v4, `summarize_results.py` after the new runs, and `make_paper_figures.py` as before.
