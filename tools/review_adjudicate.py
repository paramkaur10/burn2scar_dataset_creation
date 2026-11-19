#!/usr/bin/env python3
"""The adjudication of the point review, as a folder that opens in any browser, like review_offline.py.

  python tools/review_adjudicate.py --out review2 --reviewers param silvia --judge judge

Picks the points to settle: every point where the two interpreters gave different answers, and every answer that
the dNBR contradicts (the "look again" flags of review_dnbr_points.csv, from review_dnbr.py). The adjudicator sees
the same pictures as the interpreters, plus one line: both earlier answers, in random order and without names, and
the dNBR at the point. She never sees the mask. Writes review2/offline_judge.zip. She unzips it, double-clicks
index.html, answers with the same keys and presses "Save answers (CSV)"; the file, answers_judge.csv, goes into
review2/ as it is. Then her answer is the final label wherever she gave one:
  python tools/review_points.py score --out review2 --reviewers param silvia --judge judge
"""
import argparse
import json
import shutil
import sys
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from review_offline import offline_page  # noqa: E402
from review_points import latest_answers, reviewer_order  # noqa: E402

SAID = {"burned": "burned", "burned_before": "burned, already on the earliest date", "not_burned": "not burned",
        "water": "water", "cloud": "cloud", "shadow": "shadow", "cant_tell": "can't tell"}
HELP = ("<p><b>Adjudication.</b> These are the points where two interpreters gave different answers, or where the "
        "dNBR does not fit an answer. The orange line gives both earlier answers, in random order, and the dNBR at the "
        "point: about 0.27 or more is a clear burn signal, below 0.05 none. On a later scene it also gives the value on "
        "the fire's first scene, which shows whether the place burned even if the scar has gone. Decide from the "
        "pictures; the numbers are a hint, not a rule.</p>\n")


def dnbr_text(value):
    return "no value" if pd.isna(value) else f"{value:+.2f}"


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", default="review2")
    parser.add_argument("--reviewers", nargs=2, default=["param", "silvia"])
    parser.add_argument("--judge", default="judge")
    parser.add_argument("--seed", type=int, default=7, help="for the order in which the two answers are shown")
    args = parser.parse_args()
    out = Path(args.out)

    points = pd.read_csv(out / "points.csv", low_memory=False)
    first = points.loc[points["repeat_of"] < 0, "point_id"]
    one, two = (latest_answers(out / f"answers_{name}.csv") for name in args.reviewers)
    flags_path = out / "review_dnbr_points.csv"
    if not flags_path.exists():
        raise SystemExit(f"No {flags_path}. Run tools/review_dnbr.py with --reviewers {' '.join(args.reviewers)} first.")
    flags = pd.read_csv(flags_path).drop_duplicates("point_id").set_index("point_id")
    checks = [f"{name}_check" for name in args.reviewers if f"{name}_check" in flags.columns]
    flagged = set(flags.index[(flags[checks] == "look again").any(axis=1)]) if checks else set()

    answered = [pid for pid in first if pid in one and pid in two]
    if len(answered) < len(first):
        print(f"Warning: {len(first) - len(answered)} points lack an answer from one interpreter; they are left out.")
    disagree = {pid for pid in answered if one[pid] != two[pid]}
    contradicted = {pid for pid in answered if pid in flagged} - disagree
    chosen = disagree | contradicted
    ids = [pid for pid in reviewer_order(points, args.judge) if pid in chosen]

    rng = np.random.default_rng(args.seed)
    info = {}
    for pid in ids:
        pair = [SAID.get(one[pid], one[pid]), SAID.get(two[pid], two[pid])]
        rng.shuffle(pair)
        row = flags.loc[pid] if pid in flags.index else None
        this = row["dnbr"] if row is not None else np.nan
        text = f"Earlier answers: {pair[0]}  |  {pair[1]}.    dNBR at the point: {dnbr_text(this)} on THIS DATE"
        if row is not None and row.get("window") != "fresh" and "dnbr_fresh" in row:
            text += f", {dnbr_text(row['dnbr_fresh'])} on the fire's first scene"
        info[pid] = text + "."

    page = offline_page(ids, args.judge)
    for old, new in [('<span id="progress"></span></header>',
                      '<span id="progress"></span></header>\n<div id="info" style="padding:2px 14px 8px;color:#e0a030;'
                      'font-variant-numeric:tabular-nums"></div>'),
                     ("<script>\n", "<script>\nconst INFO = " + json.dumps(info) + ";\n"),
                     ("  shownAt = performance.now();\n",
                      '  shownAt = performance.now();\n  document.getElementById("info").textContent = INFO[ids[at]] || "";\n'),
                     ("<p>Backspace goes back one point.", HELP + "<p>Backspace goes back one point.")]:
        if page.count(old) != 1:
            raise SystemExit(f"review_points.py has changed; cannot find: {old[:60]}")
        page = page.replace(old, new)

    folder = out / f"offline_{args.judge}"
    if folder.exists():
        shutil.rmtree(folder)
    (folder / "chips").mkdir(parents=True)
    for pid in ids:
        shutil.copy2(out / "chips" / f"{pid}.jpg", folder / "chips" / f"{pid}.jpg")
    (folder / "index.html").write_text(page, encoding="utf-8")
    target = out / f"offline_{args.judge}.zip"
    with zipfile.ZipFile(target, "w", zipfile.ZIP_STORED) as archive:
        for path in sorted(folder.rglob("*")):
            archive.write(path, Path(f"burn2scar_review_{args.judge}") / path.relative_to(folder))
    print(f"{len(ids)} points to settle: {len(disagree)} where the two answers differ, and {len(contradicted)} more "
          f"where they agree but the dNBR contradicts the answer.")
    print(f"Wrote {target} ({target.stat().st_size / 1e6:.0f} MB). The answers come back as answers_{args.judge}.csv; "
          f"put it in {out}/ and run:\n  python tools/review_points.py score --out {out} "
          f"--reviewers {' '.join(args.reviewers)} --judge {args.judge}")


if __name__ == "__main__":
    main()
