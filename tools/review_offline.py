#!/usr/bin/env python3
"""The point review as a folder that opens in any browser: no server, no code, no internet.

  python tools/review_offline.py --out review2 --reviewer silvia
  python tools/review_offline.py --out review2 --reviewer silvia --batch 2     # only the points of a supplement

Writes review2/offline_silvia.zip: index.html and the reviewer's pictures, in her own random order (the same order
`serve` would give her). She unzips it, double-clicks index.html and answers with the same keys. Answers stay in her
browser between sessions; the "Save answers" button downloads answers_silvia.csv, which goes into review2/ as it is.
The mask classes are not in the folder.
"""
import argparse
import json
import shutil
import sys
import zipfile
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from review_points import PAGE, reviewer_order  # noqa: E402

SAVE_BLOCK = """  try {
    const response = await fetch("/answer", {method: "POST", headers: {"Content-Type": "application/json"}, body: JSON.stringify({id, answer: value, ms})});
    if (!response.ok) throw new Error(response.status);
  } catch (error) {
    progress.textContent = "Could not save the last answer. Check the server, then reload this page.";
  }"""
STATE_LINE = """fetch("/state").then(r => r.json()).then(state => { ids = state.ids; answers = state.answers; at = firstOpen(); show(); });"""
OFFLINE_START = """const REVIEWER = %s, KEY = "burn2scar_review_" + %s;
let log = [];
function store() {
  try { localStorage.setItem(KEY, JSON.stringify(log)); }
  catch (error) { progress.textContent = "This browser is not keeping answers: press Save answers often."; }
}
function download() {
  const rows = ["point_id,answer,ms,saved_at", ...log.map(r => r.join(","))];
  const link = document.createElement("a");
  link.href = URL.createObjectURL(new Blob([rows.join("\\n") + "\\n"], {type: "text/csv"}));
  link.download = FILE;
  link.click();
}
ids = %s;
const FILE = %s;
try { log = JSON.parse(localStorage.getItem(KEY) || "[]"); } catch (error) { log = []; }
for (const [id, value] of log) answers[id] = value;
const save = document.createElement("button");
save.innerHTML = "Save answers (CSV)";
save.onclick = download;
keys.append(save);
at = firstOpen(); show();"""


def offline_page(ids, reviewer, batch=0):
    suffix = f"_batch{batch}" if batch else ""
    page = PAGE
    for old, new in [(SAVE_BLOCK, "  log.push([id, value, ms, Math.floor(Date.now() / 1000)]);\n  store();"),
                     (STATE_LINE, OFFLINE_START % (json.dumps(reviewer), json.dumps(reviewer + suffix), json.dumps(ids),
                                                   json.dumps(f"answers_{reviewer}{suffix}.csv"))),
                     ("`/chips/", "`chips/"),
                     ("All points answered. You can close this tab.",
                      "All points answered. Press Save answers (CSV) and send the file to Param."),
                     ("Your answers save on every key press.",
                      "Your answers stay in this browser if you close it; press Save answers (CSV) at the end, "
                      "or any time, and send the downloaded file to Param."),
                     ("Is the pixel in the yellow square burned on THIS DATE, or was it burned earlier?",
                      "Had the spot in the yellow square burned at any time up to THIS DATE? Ignore pictures after THIS DATE.")]:
        if old not in page and not old.startswith("Is the pixel"):
            raise SystemExit(f"review_points.py has changed; cannot find: {old[:60]}")
        page = page.replace(old, new)
    return page


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", default="review2")
    parser.add_argument("--reviewer", required=True)
    parser.add_argument("--batch", type=int, default=0, help="only this batch of the sample (review_points.py sample --supplement)")
    args = parser.parse_args()
    out = Path(args.out)
    points = pd.read_csv(out / "points.csv", low_memory=False)
    ids = reviewer_order(points, args.reviewer)
    if args.batch:
        ids = [i for i in ids if i in set(points.loc[points.get("batch", 1) == args.batch, "point_id"])]
    name = f"{args.reviewer}_batch{args.batch}" if args.batch else args.reviewer
    folder = out / f"offline_{name}"
    if folder.exists():
        shutil.rmtree(folder)
    (folder / "chips").mkdir(parents=True)
    for point_id in ids:
        shutil.copy2(out / "chips" / f"{point_id}.jpg", folder / "chips" / f"{point_id}.jpg")
    (folder / "index.html").write_text(offline_page(ids, args.reviewer, args.batch), encoding="utf-8")
    target = out / f"offline_{name}.zip"
    with zipfile.ZipFile(target, "w", zipfile.ZIP_STORED) as archive:
        for path in sorted(folder.rglob("*")):
            archive.write(path, Path(f"burn2scar_review_{name}") / path.relative_to(folder))
    print(f"{len(ids):,} points for {args.reviewer}: {target} ({target.stat().st_size / 1e6:.0f} MB). "
          f"Unzip, double-click index.html. The answers file she sends back: answers_{name}.csv"
          + (f"; add its rows to {out}/answers_{args.reviewer}.csv with: tail -n +2 answers_{name}.csv >> {out}/answers_{args.reviewer}.csv"
             if args.batch else f", saved as {out}/answers_{args.reviewer}.csv."))


if __name__ == "__main__":
    main()
