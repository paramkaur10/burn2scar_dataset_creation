#!/usr/bin/env python3
"""Bring the Hugging Face copy of burn2scar up to version 3.0. Safe to stop and restart.

  python tools/upload_hf.py plan
  python tools/upload_hf.py run
  python tools/upload_hf.py check

Log in once on this machine first, with a write token from huggingface.co/settings/tokens. The token
stays on this machine:
  python -c "from huggingface_hub import login; login()"

plan changes nothing and says what run will do.
run, in this order:
  1. tags the repository as it is now "v1", so the version 1 masks stay downloadable
  2. uploads every version 3.0 mask that differs from the one on the Hub, 1,000 to a commit
  3. one last commit: index.csv (the release index, with product IDs), README.md (the dataset card),
     LICENSE.txt and NOTICE.txt, and removes the images and masks of the tiles left out of the release
check compares the Hub with the release: which images and masks are there, and the checksum of every
mask and of the four files.
"""
import argparse
import hashlib
from pathlib import Path

import pandas as pd

REPO = "neet1797/burn2scar"
TAG = "v1"
INFO_CHUNK = 500  # Paths per question to the Hub.


def say(text):
    print(text, flush=True)


def hashes(path):
    data = Path(path).read_bytes()
    return hashlib.sha256(data).hexdigest(), hashlib.sha1(b"blob %d\0" % len(data) + data).hexdigest()


def same(remote, path):
    """Whether the file on the Hub has the same content as the local one."""
    if remote is None:
        return False
    sha256, blob = hashes(path)
    lfs = getattr(remote, "lfs", None)
    return lfs.sha256 == sha256 if lfs is not None else remote.blob_id == blob


def remote_info(api, repo, paths):
    found = {}
    paths = list(paths)
    for start in range(0, len(paths), INFO_CHUNK):
        for item in api.get_paths_info(repo, paths[start:start + INFO_CHUNK], repo_type="dataset"):
            if hasattr(item, "blob_id"):  # A file, not a folder.
                found[item.path] = item
    return found


def read_indexes(args):
    release = pd.read_csv(args.release_index, low_memory=False)
    full = pd.read_csv(args.full_index, low_memory=False)
    left = full[~full["tile_id"].isin(release["tile_id"])]
    left_paths = sorted([*left["image_path"], *left["mask_path"]])
    missing = [p for p in release["mask_path"] if not (Path(args.masks_root) / p).is_file()]
    if missing:
        raise SystemExit(f"{len(missing):,} masks of the release aren't in {args.masks_root}, such as {missing[0]}")
    for name, path in extra_files(args).items():
        if not Path(path).is_file():
            raise SystemExit(f"No such file for {name}: {path}")
    return release, left_paths


def extra_files(args):
    return {"index.csv": args.release_index, "README.md": args.card, "LICENSE.txt": args.licence, "NOTICE.txt": args.notice}


def connect():
    from huggingface_hub import HfApi
    api = HfApi()
    try:
        user = api.whoami()
    except Exception:  # No token, or one that has expired.
        raise SystemExit('Not logged in. Run: python -c "from huggingface_hub import login; login()"')
    say(f"Logged in as {user.get('name', '?')}")
    return api


def has_tag(api, repo):
    return TAG in [tag.name for tag in api.list_repo_refs(repo, repo_type="dataset").tags]


def differing_masks(api, args, release):
    paths = sorted(release["mask_path"])
    remote = remote_info(api, args.repo, paths)
    return [p for p in paths if not same(remote.get(p), Path(args.masks_root) / p)]


def final_operations(api, args, left_paths):
    from huggingface_hub import CommitOperationAdd, CommitOperationDelete

    files = extra_files(args)
    remote = remote_info(api, args.repo, [*files, *left_paths])
    adds = [name for name, path in files.items() if not same(remote.get(name), path)]
    deletes = [p for p in left_paths if p in remote]
    operations = [CommitOperationAdd(path_in_repo=name, path_or_fileobj=str(files[name])) for name in adds]
    operations += [CommitOperationDelete(path_in_repo=p) for p in deletes]
    return operations, adds, deletes


def plan(args):
    api = connect()
    release, left_paths = read_indexes(args)
    say(f"Repository: {args.repo}. Tag {TAG}: {'already there' if has_tag(api, args.repo) else 'will be made first'}")
    todo = differing_masks(api, args, release)
    commits = -(-len(todo) // args.batch)
    say(f"Masks that differ from the Hub: {len(todo):,} of {len(release):,}, in {commits} commit{'' if commits == 1 else 's'}")
    _, adds, deletes = final_operations(api, args, left_paths)
    say(f"Then one commit. Files to add or replace: {', '.join(adds) or 'none'}. Files to remove: {len(deletes)}")
    say("Nothing was changed.")


def run(args):
    from huggingface_hub import CommitOperationAdd

    api = connect()
    release, left_paths = read_indexes(args)
    if has_tag(api, args.repo):
        say(f"Tag {TAG} is already there.")
    else:
        api.create_tag(args.repo, tag=TAG, tag_message="burn2scar version 1, before the version 3.0 masks", repo_type="dataset")
        say(f"Tagged the repository as it was: {TAG}")
    todo = differing_masks(api, args, release)
    say(f"Masks that differ from the Hub: {len(todo):,} of {len(release):,}")
    batches = [todo[i:i + args.batch] for i in range(0, len(todo), args.batch)]
    for number, batch in enumerate(batches, 1):
        operations = [CommitOperationAdd(path_in_repo=p, path_or_fileobj=str(Path(args.masks_root) / p)) for p in batch]
        api.create_commit(args.repo, operations, commit_message=f"burn2scar 3.0 masks, part {number} of {len(batches)}",
                          repo_type="dataset")
        say(f"  part {number} of {len(batches)}: {len(batch):,} masks")
    operations, adds, deletes = final_operations(api, args, left_paths)
    if operations:
        api.create_commit(args.repo, operations, repo_type="dataset",
                          commit_message="burn2scar 3.0: index, dataset card, licence and notice; tiles left out of the release removed")
        say(f"Added or replaced: {', '.join(adds) or 'none'}. Removed: {len(deletes)} files.")
    say("Done. Now run: python tools/upload_hf.py check")


def check(args):
    api = connect()
    release, left_paths = read_indexes(args)
    files = set(api.list_repo_files(args.repo, repo_type="dataset"))
    problems = 0
    for kind, column in (("images", "image_path"), ("masks", "mask_path")):
        on_hub = {p for p in files if p.startswith("acquisitions/") and p.endswith("_s2.tif" if kind == "images" else "_mask.tif")}
        wanted = set(release[column])
        missing, extra = wanted - on_hub, on_hub - wanted
        problems += len(missing) + len(extra)
        say(f"{kind.capitalize()} on the Hub: {len(on_hub):,}. Release: {len(wanted):,}. Missing: {len(missing):,}. Not in the release: {len(extra):,}.")
    todo = differing_masks(api, args, release)
    problems += len(todo)
    say(f"Masks whose content differs from version 3.0: {len(todo):,}")
    _, adds, deletes = final_operations(api, args, left_paths)
    problems += len(adds) + len(deletes)
    say(f"Of index.csv, README.md, LICENSE.txt and NOTICE.txt, out of date: {', '.join(adds) or 'none'}")
    tagged = has_tag(api, args.repo)
    problems += not tagged
    say(f"Tag {TAG}: {'there' if tagged else 'missing'}")
    say("Everything matches the release." if not problems else "Not everything matches: send this output.")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("command", choices=["plan", "run", "check"])
    parser.add_argument("--repo", default=REPO)
    parser.add_argument("--release-index", default="data/release/index.csv", help="the index uploaded to Zenodo")
    parser.add_argument("--full-index", default="data/v3/index.csv", help="to find the tiles left out of the release")
    parser.add_argument("--masks-root", default="data/v3")
    parser.add_argument("--card", default="release_docs/huggingface_card.md")
    parser.add_argument("--licence", default="release_docs/LICENSE.txt")
    parser.add_argument("--notice", default="release_docs/NOTICE.txt")
    parser.add_argument("--batch", type=int, default=1000, help="masks per commit")
    args = parser.parse_args()
    try:
        {"plan": plan, "run": run, "check": check}[args.command](args)
    except (SystemExit, KeyboardInterrupt):
        raise
    except Exception as error:  # Mostly network trouble, which a rerun gets past.
        raise SystemExit(f"Couldn't finish ({type(error).__name__}: {str(error)[:300]}).\n"
                         "Nothing is lost: run the same command again in a while. If it keeps failing, send this output.")


if __name__ == "__main__":
    main()
