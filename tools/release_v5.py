#!/usr/bin/env python3
"""Version 5.0 of the release: the files that change, and the Zenodo draft made with "New version" on the web page.

  python tools/release_v5.py prepare               # writes data/release/v5/, changes nothing on Zenodo
  python tools/release_v5.py adopt --id 24397115   # points make_release.py at the draft, sets version and description

prepare writes into data/release/v5/:
  index.csv                            the release index with the six columns new in 5.0 (from referee_checks.py columns)
  burn_evidence_old_same_season.csv    the old-window burn evidence against a pre-fire image of the same season
  reviewed_points_independent.csv      the independent date-blind review: every answer on every date shown
  README.md, LICENSE.txt, NOTICE.txt   from release_docs/
and checks that additions.zip exists. Then upload with make_release.py (see tools/README.md, version 5.0).

adopt keeps the details of the earlier record as data/release/zenodo_<id>.json and writes data/release/zenodo.json for
the new draft, with its file bucket and reserved DOI, so that make_release.py upload and status work on it. It sets
the version (5.0), the publication date and the description, and never publishes.
"""
import argparse
import json
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

TOOLS = Path(__file__).resolve().parent
sys.path.insert(0, str(TOOLS))

NEW_COLUMNS = ["history_days", "older_2y_frac", "two_pass", "cold_season_fire", "likely_managed_burn", "reburn_frac"]


def say(text):
    print(text, flush=True)


def independent_review(review):
    """One row per point and date shown: the answer of the independent interpreter, the last one where he corrected it."""
    points = pd.read_csv(review / "points.csv", low_memory=False)
    first = points[points["repeat_of"] < 0].set_index("point_id")
    shown = pd.read_csv(review / "chips_blind_shown.csv").set_index("point_id")
    answers_path = next(iter(sorted(review.glob("answers_blind_*.csv"))), None)
    if answers_path is None:
        raise SystemExit(f"No answers_blind_<name>.csv in {review}.")
    log = pd.read_csv(answers_path)
    latest = log.drop_duplicates(["point_id", "date_index"], keep="last").set_index(["point_id", "date_index"])["answer"]
    rows = []
    for point_id, record in shown.iterrows():
        dates = str(record["dates"]).split(";")
        for k, date in enumerate(dates, 1):
            rows.append({"point_id": point_id, "stratum": first.at[point_id, "stratum"],
                         "mask_value": first.at[point_id, "mask_value"], "tile_id": first.at[point_id, "tile_id"],
                         "scene_time": first.at[point_id, "scene_time"], "date_index": k, "date": date,
                         "is_scene": k == int(record["scene_position"]),
                         "answer": latest.get((point_id, k), "")})
    return pd.DataFrame(rows)


def prepare(args):
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    index = pd.read_csv(args.index, low_memory=False)
    columns = pd.read_csv(args.columns, low_memory=False)
    missing = [c for c in NEW_COLUMNS if c not in columns]
    if missing:
        raise SystemExit(f"{args.columns} lacks {missing}: run python tools/referee_checks.py columns first.")
    index = index.drop(columns=[c for c in NEW_COLUMNS if c in index]).merge(columns[["tile_id", *NEW_COLUMNS]],
                                                                             on="tile_id", how="left")
    released = index["split"].notna()
    gaps = index.loc[released, NEW_COLUMNS].isna().all(axis=1).sum()
    say(f"index.csv: {released.sum():,} released tiles, {len(index.columns)} columns; released tiles without the new columns: {gaps}")
    index.to_csv(out / "index.csv", index=False)

    same = pd.read_csv(args.same_season, low_memory=False)
    same.to_csv(out / "burn_evidence_old_same_season.csv", index=False)
    say(f"burn_evidence_old_same_season.csv: {len(same):,} tiles, flags {same['flag'].value_counts().to_dict()}")

    review = independent_review(Path(args.review))
    review.to_csv(out / "reviewed_points_independent.csv", index=False)
    say(f"reviewed_points_independent.csv: {review['point_id'].nunique()} points, {len(review):,} answers, "
        f"{int((review['answer'] == '').sum())} dates unanswered")

    for name in ("README.md", "LICENSE.txt", "NOTICE.txt"):
        shutil.copy(Path(args.docs) / name, out / name)
    readme = (out / "README.md").read_text()
    if "DATA_DOI" in readme:
        raise SystemExit("README.md still holds a DOI placeholder.")
    additions = Path(args.additions)
    say(f"additions.zip: {'found, %.2f GB' % (additions.stat().st_size / 1e9) if additions.exists() else 'MISSING: python tools/finish_additions.py --workers 16 --zip'}")
    say(f"Wrote {out}/. Nothing was sent to Zenodo.")


def adopt(args):
    from make_release import Zenodo, description, load_index
    parent = Path(args.out_parent)
    path = parent / "zenodo.json"
    if path.exists():
        old = json.loads(path.read_text())
        if str(old.get("id")) == str(args.id):
            say(f"zenodo.json already points at draft {args.id}.")
        else:
            path.rename(parent / f"zenodo_{old.get('id')}.json")
            say(f"Kept the earlier record's details as zenodo_{old.get('id')}.json")
    api = Zenodo(args.base, args.id)
    reply = api.get()
    if reply.get("submitted"):
        raise SystemExit(f"Record {args.id} is already published; give the id of the new-version draft.")
    bucket = reply.get("links", {}).get("bucket")
    doi = (reply.get("metadata", {}).get("prereserve_doi") or {}).get("doi") or reply.get("metadata", {}).get("doi")
    frame, _ = load_index(args.index)
    metadata = {k: v for k, v in reply.get("metadata", {}).items() if k not in ("doi", "prereserve_doi")}
    metadata.update(version=args.version, publication_date=datetime.now(timezone.utc).date().isoformat(),
                    description=description(frame))
    answer = api.set_metadata(metadata)
    if answer.status_code >= 400:
        say(f"Zenodo didn't take the version and description (HTTP {answer.status_code}); set them on the page.\n{answer.text[:400]}")
    state = {"base": api.base, "id": int(args.id), "bucket": bucket, "version": args.version, "doi": doi,
             "created": datetime.now(timezone.utc).isoformat(timespec="seconds")}
    path.write_text(json.dumps(state, indent=2))
    say(f"Draft: {api.base}/uploads/{args.id}\nReserved DOI: {doi}\nFile bucket: {'found' if bucket else 'MISSING'}")
    say(f"It holds {len(api.files())} files. Nothing is public until you press Publish on that page.")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("command", choices=["prepare", "adopt"])
    parser.add_argument("--index", default="data/v4/index.csv")
    parser.add_argument("--columns", default="validation/referee/index_columns.csv")
    parser.add_argument("--same-season", default="validation/dnbr_old_anniversary/burn_evidence.csv")
    parser.add_argument("--review", default="review2")
    parser.add_argument("--docs", default="release_docs")
    parser.add_argument("--additions", default="data/additions/additions.zip")
    parser.add_argument("--out", default="data/release/v5")
    parser.add_argument("--out-parent", default="data/release")
    parser.add_argument("--base", default="https://zenodo.org")
    parser.add_argument("--id", help="adopt: the id of the new-version draft, from its page address")
    parser.add_argument("--version", default="5.0")
    args = parser.parse_args()
    if args.command == "adopt" and not args.id:
        raise SystemExit("Give --id, the number at the end of the draft's address (zenodo.org/uploads/<id>).")
    {"prepare": prepare, "adopt": adopt}[args.command](args)


if __name__ == "__main__":
    main()
