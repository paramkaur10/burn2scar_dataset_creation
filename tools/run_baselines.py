#!/usr/bin/env python3
"""Run all deep learning baselines across your GPUs. Safe to stop and restart.

  tmux new -s baselines
  python tools/run_baselines.py --gpus 0 1

Finished runs are skipped on restart. Logs go to results/logs/. Add --dry-run to see
the job list first. For a quick trial of every model, and an estimate of the full run:

  python tools/run_baselines.py --gpus 0 1 --seeds 0 --skip-generalisation --limit 200 --epochs 2 --out results_trial
  python tools/run_baselines.py --gpus 0 1 --estimate results_trial
"""
import argparse
import json
import os
import queue
import subprocess
import sys
import threading
import time
from pathlib import Path

import pandas as pd

MAIN_SPECS = ["unet_r34", "unet_r18_scratch", "unetpp_r34", "deeplabv3p_r50", "unet_r50_ssl4eo",
              "segformer_b2", "hrnet_w18", "unetpp_r101"]
GENERALISATION_SPEC = "unet_r34"


def build_jobs(args):
    index = pd.read_csv(args.index, nrows=5, low_memory=False)
    loco = sorted(c for c in index.columns if c.startswith("split_loco_"))
    jobs = []
    # All models run seed 0 first, so you get a full comparison early.
    for seed in args.seeds:
        for spec in args.specs:
            jobs.append((spec, "split", seed))
    # How well the data generalises to new years and new countries.
    if not args.skip_generalisation:
        for seed in args.seeds[:3]:
            for column in ["split_time", *loco]:
                jobs.append((GENERALISATION_SPEC, column, seed))
    return jobs


def worker(gpu, jobs, args, tool, done):
    while True:
        try:
            spec, column, seed = jobs.get_nowait()
        except queue.Empty:
            return
        name = f"{spec}__{column}__multiclass__seed{seed}"
        log_path = Path(args.out) / "logs" / f"{name}.log"
        log_path.parent.mkdir(parents=True, exist_ok=True)
        command = [sys.executable, str(tool), "--spec", spec, "--seed", str(seed), "--split-col", column,
                   "--index", args.index, "--images-root", args.images_root, "--masks-root", args.masks_root,
                   "--out", args.out, "--epochs", str(args.epochs), "--patience", str(args.patience),
                   "--workers", str(args.workers), "--limit", str(args.limit), "--batch-scale", str(args.batch_scale)]
        if args.cache_root:
            command += ["--cache-root", args.cache_root]
        environment = None
        if gpu == "cpu":
            command += ["--device", "cpu"]
        else:
            environment = {**os.environ, "CUDA_VISIBLE_DEVICES": str(gpu)}
        started = time.time()
        print(f"[gpu {gpu}] start {name}", flush=True)
        with open(log_path, "a") as log:
            code = subprocess.call(command, stdout=log, stderr=subprocess.STDOUT, env=environment)
        minutes = (time.time() - started) / 60
        done.append((name, code))
        status = "done" if code == 0 else f"FAILED with code {code}, see {log_path}"
        print(f"[gpu {gpu}] {status}: {name} in {minutes:.1f} min ({len(done)}/{args.total})", flush=True)


def estimate(trial, jobs, args):
    """Scale timings from a short trial up to the full job list."""
    index = pd.read_csv(args.index, low_memory=False)
    minutes_per_tile = {}
    for path in Path(trial).glob("*/metrics.json"):
        run = json.loads(path.read_text())
        history = run.get("history", [])
        if run.get("kind") == "deep" and len(history) >= 2:
            # The second epoch: the first one includes starting the data loaders.
            epoch = history[-1]["minutes"] - history[-2]["minutes"]
            minutes_per_tile[run["name"]] = epoch / (run["tiles"]["train"] + run["tiles"]["val"])
    if not minutes_per_tile:
        raise SystemExit(f"No finished trial runs with two or more epochs under {trial}")
    rows, missing = {}, set()
    for spec, column, _ in jobs:
        if spec not in minutes_per_tile:
            missing.add(spec)
            continue
        tiles = int(index[column].isin(["train", "val"]).sum())
        rows.setdefault(spec, []).append(minutes_per_tile[spec] * tiles * args.epochs / 60)
    total = 0.0
    for spec, hours in sorted(rows.items(), key=lambda kv: -sum(kv[1])):
        total += sum(hours)
        print(f"{spec:20s} {len(hours):3d} runs, up to {max(hours):5.1f} hours each, {sum(hours):6.1f} hours in all")
    gpus = len(args.gpus)
    print(f"\nAt most about {total / gpus:.0f} hours ({total / gpus / 24:.1f} days) on {gpus} GPUs, if every run goes the full "
          f"{args.epochs} epochs. Early stopping usually cuts this. Data loading speed in a short trial is only a rough guide.")
    if missing:
        print("No timing for:", ", ".join(sorted(missing)))


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--gpus", nargs="+", default=["0", "1"], help="GPU numbers, or cpu")
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2, 3, 4], help="FLOGA repeats each model 5 times")
    parser.add_argument("--specs", nargs="+", default=MAIN_SPECS)
    parser.add_argument("--skip-generalisation", action="store_true")
    parser.add_argument("--index", default="data/v2/index.csv")
    parser.add_argument("--images-root", default="data/v1")
    parser.add_argument("--masks-root", default="data/v2")
    parser.add_argument("--out", default="results")
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument("--workers", type=int, default=-1, help="data loading processes per run; by default the CPU cores are shared between GPUs")
    parser.add_argument("--cache-root", default="", help="NumPy files from build_cache.py, which are much quicker to read")
    parser.add_argument("--limit", type=int, default=0, help="tiles per split, for a quick trial")
    parser.add_argument("--batch-scale", type=float, default=1.0, help="multiply batch sizes, e.g. 0.5 for a small GPU")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--estimate", metavar="TRIAL_DIR", help="estimate the full run from a finished trial, then stop")
    args = parser.parse_args()
    if args.workers < 0:
        args.workers = max((os.cpu_count() or 4) // len(args.gpus) - 1, 1)

    if args.estimate:
        estimate(args.estimate, build_jobs(args), args)
        return
    tool = Path(__file__).resolve().parent / "train_baseline.py"
    pending = [job for job in build_jobs(args)
               if not (Path(args.out) / f"{job[0]}__{job[1]}__multiclass__seed{job[2]}" / "metrics.json").exists()]
    args.total = len(pending)
    print(f"{len(pending)} runs to do on GPUs {args.gpus}, {args.workers} data workers per run", flush=True)
    if args.dry_run:
        for job in pending:
            print(job)
        return
    jobs = queue.Queue()
    for job in pending:
        jobs.put(job)
    done = []
    threads = [threading.Thread(target=worker, args=(gpu, jobs, args, tool, done)) for gpu in args.gpus]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    failed = [name for name, code in done if code]
    print(f"Finished {len(done) - len(failed)} of {len(done)} runs.")
    if failed:
        print("Failed:\n  " + "\n  ".join(failed))
        sys.exit(1)


if __name__ == "__main__":
    main()
