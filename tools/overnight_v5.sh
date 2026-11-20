#!/usr/bin/env bash
# Everything for the 5.0 revision that can run unattended, on the CPU, with no Copernicus quota.
#   tmux new -s night
#   cd ~/projects/burn2scar && source .venv/bin/activate && bash tools/overnight_v5.sh
# Two lanes run side by side: lane 1 works on the files on disk, lane 2 reads the open Sentinel-2 archive
# (Earth Search). Each step writes logs/overnight/<step>.log; a failed step is noted and the next one runs.
# logs/overnight/status.txt says what ran, and overnight_results.zip at the end holds every result to send back.
cd "$(dirname "$0")/.."
export PYTHONWARNINGS=ignore
LOG=logs/overnight
mkdir -p "$LOG"
STATUS=$LOG/status.txt
echo "Started $(date)" > "$STATUS"

step() {   # step <name> <command ...>
    local name=$1; shift
    local start; start=$(date +%s)
    echo "$(date '+%H:%M') start  $name" | tee -a "$STATUS"
    if [ -n "${DRY:-}" ]; then echo "  $*" | tee -a "$STATUS"; return; fi
    if "$@" > "$LOG/$name.log" 2>&1; then
        echo "$(date '+%H:%M') ok     $name ($(( ($(date +%s) - start) / 60 )) min)" | tee -a "$STATUS"
    else
        echo "$(date '+%H:%M') FAILED $name: see $LOG/$name.log" | tee -a "$STATUS"
    fi
}

lane1() {
    step check        python tools/check_v5.py
    step finish       python tools/finish_additions.py --workers 8 --zip
    step matrices     python tools/score_models.py matrices
    step review_dates python tools/review_dates.py --out review2 --reviewers param silvia --judge judge
    step cloud        python tools/referee_checks.py cloud --workers 8
    step history      python tools/referee_checks.py history --workers 8
    step burns        python tools/referee_checks.py burns
    step water        python tools/referee_checks.py water --workers 8
    step misc         python tools/referee_checks.py misc --workers 8
    step columns      python tools/referee_checks.py columns
    step additions    python tools/score_models.py additions --seeds 0 1 2 3 4
    step points       python tools/score_models.py points --seeds 0 1 2 3 4
    step blind_chips  python tools/review_blind.py chips --out review2 --index data/v4/index.csv --images-root data/v1 \
                          --prefire-root data/prefire --sample 350
    step blind_page   python tools/review_blind.py page --out review2 --reviewer colleague
}

lane2() {
    step none_list    python tools/referee_checks.py none --make-list
    step none_dnbr    python tools/burn_evidence.py --tiles-from validation/referee/none_tiles.csv --out validation/dnbr_none --workers 16
    step none         python tools/referee_checks.py none
    if [ ! -f validation/dnbr_cems/burn_evidence.csv ]; then
        step cems_dnbr python tools/burn_evidence.py --tiles-from validation/cems_tiles.csv --out validation/dnbr_cems --save-dnbr --workers 16
    fi
    step anniversary  python tools/burn_evidence.py --window old --reference anniversary --out validation/dnbr_old_anniversary --workers 16
    if [ -f validation/dnbr_old/burn_evidence.csv ]; then
        step old      python tools/referee_checks.py old
    else
        step old      python tools/referee_checks.py old --old-dnbr data/release/burn_evidence_old_window.csv
    fi
}

lane1 &
lane2 &
wait
step cems python tools/score_models.py cems --seeds 0 1 2 3 4

python - <<'EOF'
import glob, os, zipfile
patterns = ["validation/referee/*.md", "validation/referee/*.csv", "review2/review_dates.md", "review2/*.csv",
            "results_v4/model_scores_*.md", "data/additions/v4/additions_summary.md", "data/additions/v4/additions_index.csv",
            "data/additions/v4/negatives_dropped.csv", "data/additions/plan.md", "data/additions/targets.csv",
            "data/additions/acquired.csv", "validation/dnbr_none/burn_evidence.csv",
            "validation/dnbr_old_anniversary/burn_evidence.csv", "validation/dnbr/burn_evidence.csv",
            "validation/dnbr_old/burn_evidence.csv", "data/v4/index.csv", "logs/overnight/*"]
with zipfile.ZipFile("overnight_results.zip", "w", zipfile.ZIP_DEFLATED) as archive:
    for pattern in patterns:
        for name in sorted(glob.glob(pattern)):
            archive.write(name)
print(f"Wrote overnight_results.zip, {os.path.getsize('overnight_results.zip') / 1e6:.1f} MB")
EOF
echo "Finished $(date)" | tee -a "$STATUS"
