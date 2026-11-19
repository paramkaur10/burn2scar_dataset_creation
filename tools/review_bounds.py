#!/usr/bin/env python3
"""Overall accuracy of the point review under two rules, for each interpreter and for the final labels.

  python tools/review_bounds.py --out review2 --reviewers param silvia --judge judge

strict    every burn answer at a clear point counts as an error (the rule of review_points.py score).
earliest  a scar already on the earliest image shown (burned_before) at a clear point is not counted as an error:
          clear means that no EFFIS perimeter since 2018 covers the pixel, so such a scar may be older than the label
          history, or bare ground taken for a scar. The two rules bound the accuracy; the review cannot say where in
          between the true value lies.

The estimate is area-weighted over the strata, with can't tell left out, as in review_points.py; the interval is a
bootstrap that resamples the answers within each stratum. Writes <out>/review_bounds.md.
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from review_points import MAP_CLASS, expected, final_labels, latest_answers, reference_class  # noqa: E402

ANSWERS = ["burned", "burned_before", "not_burned", "water", "cloud", "shadow", "cant_tell"]


def hits(points, answers, rule):
    """Per stratum, a 0/1 vector of the usable answers, in a fixed order so that the bootstrap is reproducible."""
    out = {}
    first = points[points["repeat_of"] < 0]
    for stratum, part in first.groupby("stratum"):
        given = [answers.get(pid) for pid in part["point_id"]]
        counts = {a: sum(g == a for g in given) for a in ANSWERS}
        want = expected(MAP_CLASS[stratum])
        vector = []
        for answer in ANSWERS[:-1]:
            ok = reference_class(answer) == want
            if rule == "earliest" and MAP_CLASS[stratum] == "not_burned" and answer == "burned_before":
                ok = True
            vector += [ok] * counts[answer]
        out[stratum] = np.array(vector, float)
    return out


def bounded(points, weights, answers, rule, reps, seed):
    per = {s: h for s, h in hits(points, answers, rule).items() if len(h)}
    w = np.array([weights[s] for s in per])
    w = w / w.sum()
    estimate = float(w @ np.array([h.mean() for h in per.values()]))
    rng = np.random.default_rng(seed)
    boot = np.array([[rng.choice(h, len(h)).mean() for h in per.values()] for _ in range(reps)]) @ w
    low, high = np.percentile(boot, [2.5, 97.5])
    return f"{100 * estimate:.1f} ({100 * low:.1f}-{100 * high:.1f})"


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", default="review2")
    parser.add_argument("--reviewers", nargs=2, default=["param", "silvia"])
    parser.add_argument("--judge", default="judge")
    parser.add_argument("--reps", type=int, default=4000)
    parser.add_argument("--seed", type=int, default=20261119)
    args = parser.parse_args()
    out = Path(args.out)
    points = pd.read_csv(out / "points.csv", low_memory=False)
    area = pd.read_csv(out / "strata_area.csv").set_index("stratum")["pixels"].astype(float)
    weights = (area / area.sum()).to_dict()
    sets = {f"Interpreter {i}": latest_answers(out / f"answers_{name}.csv") for i, name in enumerate(args.reviewers, 1)}
    if args.judge and (out / f"answers_{args.judge}.csv").exists():
        sets["Final"] = final_labels(points, out, args.reviewers, args.judge)
    rows = [{"Rule": rule, **{key: bounded(points, weights, answers, rule, args.reps, args.seed)
                              for key, answers in sets.items()}} for rule in ("strict", "earliest")]
    table = pd.DataFrame(rows)
    text = ["# Overall accuracy of the point review under two rules", "",
            "strict: every burn answer at a clear point is an error. earliest: a scar already on the earliest image "
            "(burned_before) at a clear point is not counted as an error. Percent, bootstrap 95 % intervals.", "",
            "| " + " | ".join(table.columns) + " |", "|" + "---|" * len(table.columns),
            *["| " + " | ".join(map(str, r)) + " |" for r in table.itertuples(index=False)], ""]
    (out / "review_bounds.md").write_text("\n".join(text))
    print("\n".join(text))


if __name__ == "__main__":
    main()
