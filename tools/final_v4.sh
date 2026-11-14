#!/usr/bin/env bash
# The final model runs, all on the v4 masks: every model number in the paper comes from results_v4.
#   nohup bash tools/final_v4.sh > logs/final_v4.log 2>&1 &
# 25 GPU runs from run_baselines (U-Net and SegFormer x 5 seeds on the spatial split; U-Net x 3 seeds on split_time
# and each leave-one-country-out split), then 6 HEO-band runs (B02-B08, no SWIR), plus the pixel classifiers on the CPU.
# Rerunning the script skips every run that already has metrics.json.
set -uo pipefail
cd "$(dirname "$0")/.."
mkdir -p logs results_v4
COMMON="--index data/v4/index.csv --images-root data/v1 --masks-root data/v4 --out results_v4"
HEO=B02,B03,B04,B05,B06,B07,B08

# Spectral indices and the random forest, both targets, at low CPU priority so the GPU data loaders come first.
( for target in fresh burn; do
    nice -n 19 python tools/classic_baselines.py $COMMON --methods spectral RF --target $target
  done ) > logs/classic_v4.log 2>&1 &

python tools/run_baselines.py $COMMON --specs unet_r34 segformer_b2 --gpus 0 0

for job in "0 unet_r34" "0 segformer_b2"; do
  set -- $job
  ( for seed in 0 1 2; do
      CUDA_VISIBLE_DEVICES=$1 python tools/train_baseline.py --spec $2 --seed $seed --bands $HEO $COMMON
    done ) > logs/heo_$2.log 2>&1 &
done
wait
echo "final_v4 finished $(date)"
