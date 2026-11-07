#!/usr/bin/env python3
"""Put the data set in a Zenodo draft with a reserved DOI. It never publishes.

Publishing makes the DOI permanent, so you do that yourself on the Zenodo page once the data are
final. A token with only the deposit:write scope can't publish, so create it that way.

  python tools/make_release.py plan
  python tools/make_release.py draft --creator "Family, Given"
  python tools/make_release.py upload --only masks
  python tools/make_release.py upload --only images
  python tools/make_release.py upload --only index
  python tools/make_release.py upload --files README.md
  python tools/make_release.py status

Images go in one zip per country and fire year, and masks in one zip, so labels can change without
moving the images. Unzip everything into one folder to get the paths in index.csv:
acquisitions/<fire_id>/<timestamp>/<fire_id>_<timestamp>_s2.tif and _mask.tif.

Each zip is built, uploaded, checked against Zenodo's checksum, then deleted, so only one is on
disk at a time. Safe to stop and restart. Finished image zips are skipped. Masks and index are
rebuilt each time and only uploaded if they changed, replacing the old file in the draft.
Tiles with no split are left out. The token comes from ZENODO_TOKEN or ~/.zenodo_token.
"""
import argparse
import hashlib
import json
import os
import shutil
import sys
import time
import zipfile
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

GB = 1000 ** 3
MAX_FILES = 100  # Zenodo's limits per record.
MAX_BYTES = 50 * GB
ZIP_DATE = (2026, 1, 1, 0, 0, 0)  # The same files always give the same zip, and the same checksum.
TITLE = "burn2scar: a Sentinel-2 burn-scar segmentation data set"
KEYWORDS = ["burned area", "burn scar", "Sentinel-2", "EFFIS", "semantic segmentation", "cloud mask", "Mediterranean"]
COUNTRY_NAMES = {"IT": "Italy", "ES": "Spain", "FR": "France south of 46 N", "EL": "Greece"}


class TryAgain(Exception):
    """A problem on Zenodo's side that may pass."""


def say(text):
    print(text, flush=True)


def load_index(path):
    frame = pd.read_csv(path, low_memory=False)
    kept = frame[frame["split"].notna()].copy()
    kept["fire_year"] = kept["fire_year"].astype(int)
    return kept, len(frame) - len(kept)


def image_bundles(frame):
    """One zip per country and fire year: name -> paths from the index."""
    return {f"images_{country}_{year}.zip": sorted(group["image_path"])
            for (country, year), group in frame.groupby(["country", "fire_year"])}


def description(frame):
    first, last = frame["fire_year"].min(), frame["fire_year"].max()
    names = ", ".join(COUNTRY_NAMES.get(code, code) for code in COUNTRY_NAMES if code in set(frame["country"]))
    return (f"Sentinel-2 Level-1C imagery (13 bands on a 10 m grid, 256 by 256 pixels) paired with burn-scar segmentation masks "
            f"for {frame['fire_id'].nunique():,} EFFIS fires in {names}, {first} to {last}: {len(frame):,} tiles. "
            "Mask classes: clear, fresh burn, old burn, older burn, cloud, cloud shadow, water, no data. "
            "Unzip all files into one folder. index.csv lists every tile. "
            f"Contains modified Copernicus Sentinel data {first}-{last}. Contains modified Copernicus Service information {first}-{last}. "
            f"Burn labels adapted from EFFIS, Copernicus Emergency Management Service, (c) European Union, 1995-{last}, CC BY 4.0.")


def sizes(paths, root):
    total, missing = 0, []
    for relative in paths:
        try:
            total += os.path.getsize(Path(root) / relative)
        except OSError:
            missing.append(relative)
    return total, missing


def plan(args):
    frame, dropped = load_index(args.index)
    say(f"Tiles: {len(frame):,}. Fires: {frame['fire_id'].nunique():,}. Left out, no split: {dropped}.")
    rows, total, missing = [], 0, []
    for name, paths in image_bundles(frame).items():
        size, lost = sizes(paths, args.images_root)
        rows.append((name, len(paths), size))
        total += size
        missing += lost
    masks = sorted(frame["mask_path"])
    size, lost = sizes(masks, args.masks_root)
    rows.append(("masks.zip", len(masks), size))
    total += size
    missing += lost
    for name, count, size in rows:
        say(f"  {name:24s} {count:6,} files {size / GB:7.2f} GB")
    say(f"Zips: {len(rows)}, {total / GB:.2f} GB. Zenodo allows {MAX_FILES} files and {MAX_BYTES / GB:.0f} GB per record.")
    say(f"Largest zip: {max(row[2] for row in rows) / GB:.2f} GB. Free on this disk: {shutil.disk_usage(args.out_parent).free / GB:.0f} GB.")
    say(f"Files listed in the index but not on disk: {len(missing)}")
    for relative in missing[:5]:
        say(f"  {relative}")
    say(f"Product IDs in the index: {'yes' if 'product_id' in frame.columns else 'not yet'}")
    if missing or len(rows) + 5 > MAX_FILES or total > MAX_BYTES * .98:
        raise SystemExit("Not ready to upload, see above.")
    say("Ready to upload.")


class Zenodo:
    def __init__(self, base, depo=None, bucket=None):
        import requests

        self.requests = requests
        self.base = base.rstrip("/")
        self.depo = depo
        self.bucket = bucket
        token = os.environ.get("ZENODO_TOKEN", "").strip()
        keyfile = Path.home() / ".zenodo_token"
        if not token and keyfile.exists():
            token = keyfile.read_text().strip()
        if not token:
            raise SystemExit("No token. Put it in ~/.zenodo_token or ZENODO_TOKEN.")
        self.session = requests.Session()
        self.session.headers["Authorization"] = f"Bearer {token}"  # In a header, so it never appears in a URL or a log.

    def check(self, response, what):
        if response.status_code >= 500 or response.status_code == 429:
            raise TryAgain(f"{what} failed: HTTP {response.status_code}")
        if response.status_code >= 400:
            raise SystemExit(f"{what} failed: HTTP {response.status_code}\n{response.text[:800]}")
        return response

    def create(self):
        reply = self.check(self.session.post(f"{self.base}/api/deposit/depositions", json={}, timeout=120), "Creating the draft").json()
        self.depo = reply["id"]
        self.bucket = reply.get("links", {}).get("bucket")
        return reply

    def get(self):
        url = f"{self.base}/api/deposit/depositions/{self.depo}"
        return self.check(self.session.get(url, timeout=120), "Reading the draft").json()

    def set_metadata(self, metadata):
        url = f"{self.base}/api/deposit/depositions/{self.depo}"
        return self.session.put(url, json={"metadata": metadata}, timeout=120)

    def files(self):
        """Name -> (md5, size, id) of the files now in the draft."""
        url = f"{self.base}/api/deposit/depositions/{self.depo}/files"
        data = self.check(self.session.get(url, timeout=120), "Listing the draft's files").json()
        entries = data if isinstance(data, list) else data.get("entries") or data.get("hits", {}).get("hits", [])
        found = {}
        for entry in entries:
            name = entry.get("filename") or entry.get("key")
            found[name] = (str(entry.get("checksum", "")).replace("md5:", ""), entry.get("filesize") or entry.get("size"), entry.get("id"))
        return found

    def delete(self, name, file_id):
        reply = self.session.delete(f"{self.base}/api/deposit/depositions/{self.depo}/files/{file_id}", timeout=120)
        if reply.status_code >= 400:
            reply = self.session.delete(f"{self.bucket}/{name}", timeout=120)
        self.check(reply, f"Replacing {name}")

    def put(self, path, name):
        """Send one file. Returns the md5 Zenodo reports, if it reports one."""
        with open(path, "rb") as handle:
            reply = self.session.put(f"{self.bucket}/{name}", data=Progress(handle, path.stat().st_size, name), timeout=(60, 7200))
        self.check(reply, f"Uploading {name}")
        return str(reply.json().get("checksum", "")).replace("md5:", "")


class Progress:
    """A file that reports how much has been read from it, so a long upload shows signs of life."""

    def __init__(self, handle, size, name):
        self.handle, self.size, self.name = handle, size, name
        self.sent, self.next_report = 0, .25

    def __len__(self):
        return self.size

    def read(self, count=-1):
        chunk = self.handle.read(count)
        self.sent += len(chunk)
        if self.size and self.sent / self.size >= self.next_report and self.next_report < 1:
            self.next_report += .25
            try:
                say(f"    {self.name}: {self.next_report - .25:.0%} sent")
            except (OSError, ValueError):  # A lost terminal must not stop an upload, and urllib3 hides this error if it comes from here.
                pass
        return chunk


def md5_of(path):
    digest = hashlib.md5()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_zip(target, paths, root):
    """GeoTIFFs are already compressed, so the zip only stores them."""
    temporary = target.with_name(target.name + ".part")
    with zipfile.ZipFile(temporary, "w", zipfile.ZIP_STORED, allowZip64=True) as archive:
        for relative in paths:
            info = zipfile.ZipInfo(relative, date_time=ZIP_DATE)
            info.external_attr = 0o644 << 16
            archive.writestr(info, (Path(root) / relative).read_bytes())
    temporary.replace(target)


def send(api, path, name, md5, retries, wait):
    """Upload, replacing any older copy, and confirm Zenodo holds the same bytes."""
    last = None
    for attempt in range(1, retries + 1):
        try:
            remote = api.files()
            if name in remote:
                api.delete(name, remote[name][2])
            reported = api.put(path, name)
            held = api.files().get(name)
            if reported and reported != md5:
                raise ValueError(f"Zenodo reports checksum {reported}, this file is {md5}")
            if not held or (held[0] and held[0] != md5):
                raise ValueError(f"The draft's copy doesn't match: {held}")
            return
        except (OSError, ValueError, api.requests.RequestException, KeyError, TryAgain) as error:
            last = error
            say(f"    {name}: attempt {attempt} of {retries} failed: {str(error)[:200]}")
            if attempt < retries:
                time.sleep(wait * attempt)
    raise SystemExit(f"Stopped at {name}: {last}. Run the same command again to carry on.")


def load_state(args):
    path = Path(args.out_parent) / "zenodo.json"
    if not path.exists():
        raise SystemExit("No draft yet. Run: python tools/make_release.py draft --creator \"Family, Given\"")
    return json.loads(path.read_text()), path


def draft(args):
    path = Path(args.out_parent) / "zenodo.json"
    if path.exists() and not args.new:
        state = json.loads(path.read_text())
        say(f"There is already a draft: {state['base']}/uploads/{state['id']}\nReserved DOI: {state.get('doi')}")
        return
    frame, _ = load_index(args.index)
    api = Zenodo(args.base)
    reply = api.create()
    creator = {"name": args.creator}
    if args.affiliation:
        creator["affiliation"] = args.affiliation
    if args.orcid:
        creator["orcid"] = args.orcid
    metadata = {"upload_type": "dataset", "title": TITLE, "creators": [creator], "description": description(frame), "access_right": "open",
                "license": "cc-by-4.0", "version": args.version, "keywords": KEYWORDS, "prereserve_doi": True}
    answer = api.set_metadata(metadata)
    if answer.status_code >= 400:
        say(f"Zenodo didn't accept the description (HTTP {answer.status_code}). The draft exists, so fill this in on its page:\n{answer.text[:800]}")
    else:
        reply = answer.json()
    if not api.bucket:
        api.bucket = api.get().get("links", {}).get("bucket")
    doi = (reply.get("metadata", {}).get("prereserve_doi") or {}).get("doi")
    state = {"base": api.base, "id": api.depo, "bucket": api.bucket, "doi": doi, "created": datetime.now(timezone.utc).isoformat(timespec="seconds")}
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(state, indent=2))
    say(f"Draft: {api.base}/uploads/{api.depo}")
    say(f"Reserved DOI: {doi or 'none yet, use the button on the draft page'}. It only starts working when you publish.")
    say("Nothing is public until you press Publish on that page.")


def upload(args):
    state, _ = load_state(args)
    try:
        os.nice(10)  # Training keeps priority.
    except OSError:
        pass
    out = Path(args.out_parent)
    frame, _ = load_index(args.index)
    api = Zenodo(state["base"], state["id"], state["bucket"])
    manifest_path = out / "manifest.json"
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
    remote = api.files()

    def remember(name, md5, size, count):
        manifest[name] = {"md5": md5, "bytes": size, "files": count}
        manifest_path.write_text(json.dumps(manifest, indent=1))

    def zip_and_send(name, paths, root, final):
        known = manifest.get(name)
        if final and known and name in remote and remote[name][0] == known["md5"]:
            say(f"  {name}: already there")
            return
        size, missing = sizes(paths, root)
        if missing:
            raise SystemExit(f"{name}: {len(missing)} files are missing, such as {missing[0]}")
        free = shutil.disk_usage(out).free
        if free - size < args.min_free_gb * GB:
            raise SystemExit(f"{name}: needs {size / GB:.1f} GB, and {free / GB:.1f} GB is free. Keeping {args.min_free_gb} GB free, so stopping.")
        target = out / name
        started = time.time()
        try:
            build_zip(target, paths, root)
            md5 = md5_of(target)
            if name in remote and remote[name][0] == md5:
                say(f"  {name}: unchanged, already there")
            else:
                say(f"  {name}: {target.stat().st_size / GB:.2f} GB, uploading")
                send(api, target, name, md5, args.retries, args.wait)
                say(f"  {name}: done and checked in {(time.time() - started) / 60:.1f} min")
            remember(name, md5, target.stat().st_size, len(paths))
        finally:  # Whatever happened, no zip stays on the disk.
            target.unlink(missing_ok=True)
            target.with_name(target.name + ".part").unlink(missing_ok=True)

    def send_file(path, name=None):
        name = name or path.name
        md5 = md5_of(path)
        if name in remote and remote[name][0] == md5:
            say(f"  {name}: unchanged, already there")
            return
        say(f"  {name}: {path.stat().st_size / GB:.3f} GB, uploading")
        send(api, path, name, md5, args.retries, args.wait)
        remember(name, md5, path.stat().st_size, 1)
        say(f"  {name}: done and checked")

    if args.only == "images":
        bundles = image_bundles(frame)
        for number, (name, paths) in enumerate(bundles.items(), 1):
            say(f"[{number}/{len(bundles)}] {name}")
            zip_and_send(name, paths, args.images_root, final=True)
    elif args.only == "masks":
        zip_and_send("masks.zip", sorted(frame["mask_path"]), args.masks_root, final=False)
    elif args.only == "index":
        target = out / "index.csv"
        frame.to_csv(target, index=False)
        send_file(target)
    for extra in args.files or []:
        path = Path(extra)
        if not path.is_file():
            raise SystemExit(f"No such file: {extra}")
        send_file(path)
    say("Done. Nothing is public until you press Publish on the draft page.")
    say(f"{state['base']}/uploads/{state['id']}")


def status(args):
    state, _ = load_state(args)
    frame, _ = load_index(args.index)
    remote = Zenodo(state["base"], state["id"], state["bucket"]).files()
    wanted = [*image_bundles(frame), "masks.zip", "index.csv"]
    total = sum(entry[1] or 0 for entry in remote.values())
    say(f"Draft: {state['base']}/uploads/{state['id']}\nReserved DOI: {state.get('doi')}")
    say(f"In the draft: {len(remote)} files, {total / GB:.2f} GB")
    missing = [name for name in wanted if name not in remote]
    say(f"Still to upload: {len(missing)}")
    for name in missing:
        say(f"  {name}")
    others = sorted(set(remote) - set(wanted))
    if others:
        say("Also in the draft: " + ", ".join(others))


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("command", choices=["plan", "draft", "upload", "status"])
    parser.add_argument("--index", default="data/v2/index.csv")
    parser.add_argument("--images-root", default="data/v1")
    parser.add_argument("--masks-root", default="data/v2")
    parser.add_argument("--out-parent", default="data/release", help="the draft's details and, briefly, each zip")
    parser.add_argument("--base", default="https://zenodo.org", help="https://sandbox.zenodo.org to try things out, with its own token")
    parser.add_argument("--creator", help='as "Family, Given"')
    parser.add_argument("--affiliation", default="")
    parser.add_argument("--orcid", default="")
    parser.add_argument("--version", default="2.0")
    parser.add_argument("--new", action="store_true", help="make another draft even though one exists")
    parser.add_argument("--only", choices=["images", "masks", "index"])
    parser.add_argument("--files", nargs="+", help="small files to add as they are, such as README.md")
    parser.add_argument("--min-free-gb", type=float, default=15)
    parser.add_argument("--retries", type=int, default=5)
    parser.add_argument("--wait", type=float, default=60, help="seconds before a retry, more on each one")
    args = parser.parse_args()
    Path(args.out_parent).mkdir(parents=True, exist_ok=True)
    if args.command == "draft" and not args.creator and not (Path(args.out_parent) / "zenodo.json").exists():
        parser.error('draft needs --creator "Family, Given"')
    if args.command == "upload" and not (args.only or args.files):
        parser.error("upload needs --only images, masks or index, or --files")
    try:
        {"plan": plan, "draft": draft, "upload": upload, "status": status}[args.command](args)
    except (TryAgain, OSError) as error:  # Network errors from requests are OSErrors too.
        raise SystemExit(f"{str(error)[:300]}\nCouldn't finish. Nothing is lost: run the same command again in a while.")


if __name__ == "__main__":
    sys.exit(main())
