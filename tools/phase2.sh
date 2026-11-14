#!/usr/bin/env bash
# Every technical step left for the revision, in one queue.
#   nohup bash tools/phase2.sh > logs/phase2.log 2>&1 &
# Part A starts now (CPU and network, no GPU). Part B waits until final_v4.sh has finished, then needs the GPU for minutes.
# Safe to start again: finished steps are skipped or redone the same way, and the review supplement is added only once.
set -uo pipefail
cd "$(dirname "$0")/.."
mkdir -p logs validation
IDX=data/v4/index.csv
step() { echo; echo "== $(date '+%a %d %b %H:%M') $*"; }

step "A1 dNBR: retry the failed reads of the fresh-window run, then rewrite its summary and the index flags"
python tools/burn_evidence.py --index $IDX --masks-root data/v4 --out validation/dnbr --workers 16 > logs/p2_dnbr_retry.log 2>&1
python tools/burn_evidence.py --index $IDX --masks-root data/v4 --out validation/dnbr --summarize --write-index > /dev/null

step "A2 dNBR on the old-window tiles: is the scar still detectable months later"
python tools/burn_evidence.py --index $IDX --masks-root data/v4 --out validation/dnbr_old --window old --workers 16 > logs/p2_dnbr_old.log 2>&1

step "A3 dNBR analysis"
python tools/dnbr_analysis.py --index $IDX --fresh validation/dnbr --old validation/dnbr_old --out validation/dnbr_analysis.md > /dev/null

step "A4 review supplement: about 230 more points, pre-fire images for new fires only (Sentinel Hub), pictures, Silvia's zip"
if python -c "import pandas as pd, sys; p = pd.read_csv('review2/points.csv'); sys.exit(0 if 'batch' in p and (p['batch'] >= 2).any() else 1)"; then
  echo "supplement already drawn"
else
  python tools/review_points.py sample --supplement --index $IDX --masks-root data/v4 --out review2 --seed 2027 \
      --counts older=50 fresh_early=40 fresh_mid=40 fresh_late=40 cloud=15 shadow=15
fi
python tools/fetch_prefire.py --index $IDX --points review2/points.csv --out data/prefire > logs/p2_prefire.log 2>&1
tail -2 logs/p2_prefire.log
python tools/review_points.py chips --out review2 --index $IDX --images-root data/v1 --prefire-root data/prefire
python tools/review_offline.py --out review2 --reviewer silvia --batch 2

step "B0 waiting for the final runs (checks every 10 minutes)"
while pgrep -f "tools/final_v4.sh" > /dev/null; do sleep 600; done

step "B1 summaries, extra scores, and the inference tests (older_burn, off-centre windows)"
python tools/summarize_results.py --results results_v4 --index $IDX > logs/p2_summarize.log 2>&1
python tools/extra_scores.py --results results_v4 --index $IDX --out results_v4/extra_scores.md > /dev/null
python tools/extra_inference.py --results results_v4 --index $IDX --images-root data/v1 --masks-root data/v4 \
    --out results_v4/extra_inference.md > logs/p2_inference.log 2>&1
step "phase 2 finished. Send: validation/dnbr/burn_evidence.md, validation/dnbr_analysis.md, results_v4/extra_scores.md, results_v4/extra_inference.md, and the summarize_results output"
