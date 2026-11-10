"""Put trained baseline models on the Hugging Face Hub, each with a model card, its settings and predict.py.

The released weights are the seed with the median fresh-burn F1 on the test set of the main split, so
they are a typical run, not the luckiest one. The model card gives the scores of every seed.

  python tools/release_models.py build --spec unet_r34
  python tools/release_models.py upload --spec unet_r34 --repo neet1797/burn2scar-unet-r34
  python tools/release_models.py check --spec unet_r34 --repo neet1797/burn2scar-unet-r34
  python tools/release_models.py export --spec unet_r34 --repo neet1797/burn2scar-unet-r34

build reads results/ (metrics.json, test_tile_confusion.npz and best.pt of each seed, and band_stats.json)
and writes release_models/burn2scar-<spec>/: model.safetensors, config.json, metrics.json, README.md (the
model card), predict.py and LICENSE. It needs no network.
upload creates the model repo if it is missing, as a public repo, and uploads the folder in one commit.
check downloads what is on the Hub, compares it byte for byte with the folder, and runs the model once.
export writes models/burn2scar-<spec>/ for the code repository from the built folder: config.json, with the Hub
repo and the weights' sha256 added so tools/predict.py can fetch and check them on first use; metrics.json; and
the model card, without the Hub's metadata header and with the repository's commands. The weights stay on the Hub.

Only specs listed in RELEASE can be released. SegFormer-B2 is not among them: its encoder starts from
NVIDIA's MiT ImageNet weights, whose licence allows only non-commercial use of derived works.
"""
import argparse
import hashlib
import json
import shutil
import sys
import tempfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from summarize_results import burnt_from_matrix, fresh_from_matrix, pm  # noqa: E402

DATASET_DOI = "10.5281/zenodo.23986827"
DATASET_REPO = "neet1797/burn2scar"
CODE_URL = "https://github.com/paramkaur10/burn2scar_dataset_creation"
CITATION = ("Thind, P. K., Cadau, E., Del Prete, R., Ricolfi, S., Meoni, G., and Taramelli, A.: burn2scar: a Sentinel-2 "
            "burn-scar segmentation data set, version 3.0, Zenodo [data set], https://doi.org/10.5281/zenodo.23986827, 2026.")
BANDS = ["B01", "B02", "B03", "B04", "B05", "B06", "B07", "B08", "B8A", "B09", "B10", "B11", "B12"]
CLASSES = ["clear", "fresh_burn", "old_burn", "cloud", "shadow", "water"]
MASK_VALUES = [0, 2, 3, 4, 5, 6]  # mask value of each model class, as in train_baseline.TASKS["multiclass"]
RELEASE = {
    "unet_r34": {"title": "U-Net (ResNet-34)", "architecture": "Unet", "encoder": "resnet34", "licence": "apache-2.0", "licence_name": "Apache 2.0",
                 "encoder_credit": "The encoder started from torchvision's ImageNet ResNet-34 weights (BSD-3-Clause), "
                                   "as distributed with segmentation-models-pytorch.",
                 "training": "AdamW, learning rate 3e-4, weight decay 1e-4, batch 32"},
}
FILES = ["model.safetensors", "config.json", "metrics.json", "README.md", "predict.py", "LICENSE"]


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def folder_for(args):
    return Path(args.out) / f"burn2scar-{args.spec.replace('_', '-')}"


def seed_scores(results, spec, split):
    """One entry per finished seed: its metrics.json, and fresh and burnt scores from its saved test matrices."""
    runs = []
    for path in sorted(Path(results).glob(f"{spec}__{split}__multiclass__seed*/metrics.json")):
        run = json.loads(path.read_text())
        data = np.load(path.parent / "test_tile_confusion.npz", allow_pickle=False)
        matrix, classes = data["matrices"].sum(0), data["classes"]
        if list(classes) != CLASSES:
            raise SystemExit(f"{path.parent}: classes {list(classes)}, expected {CLASSES}")
        runs.append({"dir": path.parent, "run": run, "fresh": fresh_from_matrix(matrix, classes),
                     "burnt": burnt_from_matrix(matrix, classes)})
    return runs


def median_run(runs):
    ordered = sorted(runs, key=lambda r: (r["fresh"]["burnt_f1"], r["run"]["seed"]))
    return ordered[(len(ordered) - 1) // 2]


def model_card(spec, info, chosen, runs, config):
    run, test = chosen["run"], chosen["run"]["test"]
    seeds = len(runs)

    def both(get):
        return f"{get(chosen) * 100:.2f}", pm([get(r) for r in runs])
    rows = [
        ("Fresh burn F1", both(lambda r: r["fresh"]["burnt_f1"])),
        ("Fresh burn precision", both(lambda r: r["fresh"]["burnt_precision"])),
        ("Fresh burn recall", both(lambda r: r["fresh"]["burnt_recall"])),
        ("Fresh burn IoU", both(lambda r: r["fresh"]["burnt_iou"])),
        ("Burnt F1, fresh and old together", both(lambda r: r["burnt"]["burnt_f1"])),
        ("Fresh vs old accuracy", both(lambda r: r["run"]["test"]["fresh_old_accuracy"])),
        ("Mean IoU, six classes", both(lambda r: r["run"]["test"]["class_miou"])),
    ]
    table = "\n".join(f"| {name} | {mine} | {spread} |" for name, (mine, spread) in rows)
    per_class = "\n".join(f"| {name} | {test['per_class'][name]['iou'] * 100:.2f} |" for name in CLASSES)
    tiles = run["tiles"]
    return f"""---
license: {info['licence']}
library_name: segmentation-models-pytorch
pipeline_tag: image-segmentation
datasets:
- {DATASET_REPO}
tags:
- burned-area
- burn-scar
- wildfire
- sentinel-2
- earth-observation
- remote-sensing
- semantic-segmentation
---

# burn2scar {info['title']}

A baseline model trained on [burn2scar](https://doi.org/{DATASET_DOI}) version 3.0. It takes one post-fire
Sentinel-2 Level-1C image and gives every pixel one of six classes: clear, fresh_burn (burned up to 90 days
before the image), old_burn (90 to 365 days before), cloud, cloud shadow and water.

It is one of the baselines of the burn2scar data paper, there to show that the labels can be learned. It was
not tuned to be the best possible model.

## Use

```bash
pip install segmentation-models-pytorch safetensors rasterio huggingface_hub
python -c "from huggingface_hub import hf_hub_download; hf_hub_download('{'{repo}'}', 'predict.py', local_dir='.')"
python predict.py --model {'{repo}'} --image tile_s2.tif --out tile_pred.tif
```

```python
from predict import load_model, predict_array
net, config = load_model("{'{repo}'}")
mask = predict_array(net, config, image)  # image: (13, height, width), reflectance x 10,000
```

**Input:** the 13 Sentinel-2 Level-1C bands in the order {', '.join(BANDS)}, as top-of-atmosphere
reflectance times 10,000 on a 10 m grid, the way burn2scar stores its images. Pixels where every band is 0
are no data. Any image size works; predict.py reads it in 256 x 256 windows, the size of a burn2scar tile.

**Output:** burn2scar mask values: 0 clear, 2 fresh_burn, 3 old_burn, 4 cloud, 5 cloud shadow, 6 water and
255 no data. The model never predicts 7 (older_burn), which was left out of training.

## Scores

Test set of burn2scar's spatial split, {tiles['test']:,} tiles. This model is seed {run['seed']}, the median of
{seeds} seeds by fresh-burn F1. Percent.

| Measure | This model | {seeds} seeds, mean ± sd |
|---|---|---|
{table}

Fresh burn is scored against clear, old_burn and water. "Fresh and old together" scores both burn classes
against clear and water. Fresh vs old accuracy is the share of pixels labelled and predicted as burn that get
the right age. Cloud, shadow, older_burn and no-data labels are left out of all of these.

Per-class IoU of this model, each class against all others:

| Class | IoU |
|---|---|
{per_class}

## Training

- Data: burn2scar version 3.0, column `split` of index.csv: {tiles['train']:,} training and {tiles['val']:,} validation tiles. The split keeps overlapping fires apart in 0.5 degree blocks.
- Input scaling: reflectance (value / 10,000) clipped to 0 to 1.5, then normalised per band with the mean and standard deviation of 1,000 training tiles (both in config.json).
- {info['encoder_credit']} The first layer takes 13 bands.
- Random 192 x 192 crops, flips and 90 degree turns. {info['training']}, one warm-up epoch then cosine decay, at most 60 epochs, stopped after 15 without gain. Loss: cross entropy weighted by the inverse square root of class frequency, plus Dice. The kept epoch is the best by validation F1 of fresh and old burn together.
- This run: {run['epochs_run']} epochs, best at epoch {run['best_epoch']}.

## Limits

- Trained on four Mediterranean countries (Italy, Spain, France south of 46° N, Greece), fires from 2020 to August 2026 of 5 to 2,000 ha. Scores fall on countries a model has not seen: in the data set's tests, leaving Spain, France or Italy out of training cost 5 to 8 points of fresh-burn F1.
- The labels follow EFFIS perimeters, so they share EFFIS's omissions and outline errors. old_burn is set by the date of the fire and is often no longer visible.
- Small fires and burn edges are harder for the model, as for any method.
- Level-1C only. Atmospherically corrected (Level-2A) images give different values and are not what it was trained on.

## Licence and credits

Model weights: {info['licence_name']}. Trained on burn2scar (CC BY 4.0), which contains modified Copernicus Sentinel
data 2020–2026 and burn classes adapted from EFFIS, © European Union, 1995–2026, CC BY 4.0.

Code that built the data and trained the model: {CODE_URL}

## Citation

{CITATION}
"""


def run_build(args):
    import torch
    from safetensors.torch import save_file
    if args.spec not in RELEASE:
        raise SystemExit(f"{args.spec} is not in RELEASE. Add it with its licence first; SegFormer's encoder weights "
                         "allow only non-commercial use.")
    info = RELEASE[args.spec]
    runs = seed_scores(args.results, args.spec, args.split)
    if not runs:
        raise SystemExit(f"No finished {args.spec} runs on {args.split} in {args.results}.")
    chosen = median_run(runs)
    print(f"{len(runs)} seeds; fresh-burn F1: " + ", ".join(f"seed {r['run']['seed']} {r['fresh']['burnt_f1'] * 100:.2f}" for r in runs))
    print(f"Releasing seed {chosen['run']['seed']} ({chosen['dir']}), the median.")
    stats = json.loads((Path(args.results) / "band_stats.json").read_text())[args.split]
    folder = folder_for(args)
    folder.mkdir(parents=True, exist_ok=True)
    state = torch.load(chosen["dir"] / "best.pt", map_location="cpu", weights_only=True)
    save_file({key: value.contiguous() for key, value in state.items()}, str(folder / "model.safetensors"))
    config = {"architecture": info["architecture"], "encoder": info["encoder"], "in_channels": len(BANDS),
              "classes": CLASSES, "mask_values": MASK_VALUES, "bands": BANDS, "scale": 10000, "clip": [0.0, 1.5],
              "mean": stats["mean"], "std": stats["std"], "tile_size": 256, "nodata_value": 255,
              "training": {"data": f"burn2scar 3.0, https://doi.org/{DATASET_DOI}", "split_column": args.split,
                           "seed": chosen["run"]["seed"], "run": chosen["dir"].name}}
    (folder / "config.json").write_text(json.dumps(config, indent=1))
    metrics = {"model_seed": chosen["run"]["seed"], "test_tiles": chosen["run"]["tiles"]["test"],
               "seeds": [{"seed": r["run"]["seed"], "fresh": r["fresh"], "burnt": r["burnt"],
                          "fresh_old_accuracy": r["run"]["test"]["fresh_old_accuracy"],
                          "per_class": r["run"]["test"]["per_class"], "class_miou": r["run"]["test"]["class_miou"]} for r in runs]}
    (folder / "metrics.json").write_text(json.dumps(metrics, indent=1, default=float))
    (folder / "README.md").write_text(model_card(args.spec, info, chosen, runs, config).replace("{repo}", args.repo))
    shutil.copy(Path(__file__).resolve().parent / "predict.py", folder / "predict.py")
    shutil.copy(Path(__file__).resolve().parent.parent / "LICENSE", folder / "LICENSE")
    # The folder must load and run before it goes anywhere.
    from predict import load_model, predict_array
    net, loaded = load_model(folder)
    mask = predict_array(net, loaded, np.full((len(BANDS), 256, 256), 1000, np.uint16))
    print(f"Loads and runs: a flat test image gives mask values {sorted(set(np.unique(mask).tolist()))}.")
    print(f"Wrote {folder}: " + ", ".join(f"{name} ({(folder / name).stat().st_size / 1e6:.1f} MB)" for name in FILES))
    print("Read README.md before you upload it.")


def run_upload(args):
    from huggingface_hub import HfApi
    folder = folder_for(args)
    missing = [name for name in FILES if not (folder / name).exists()]
    if missing:
        raise SystemExit(f"{folder} lacks {', '.join(missing)}: run build first.")
    api = HfApi()
    print(f"Logged in as {api.whoami()['name']}")
    api.create_repo(repo_id=args.repo, repo_type="model", exist_ok=True)
    api.upload_folder(folder_path=str(folder), repo_id=args.repo, repo_type="model", allow_patterns=FILES,
                      commit_message=f"burn2scar {RELEASE[args.spec]['title']} baseline, data set version 3.0")
    print(f"Uploaded to https://huggingface.co/{args.repo}. Now run check.")


def run_check(args):
    from huggingface_hub import hf_hub_download
    folder = folder_for(args)
    problems = 0
    with tempfile.TemporaryDirectory() as temp:
        for name in FILES:
            remote = Path(hf_hub_download(repo_id=args.repo, filename=name, local_dir=temp))
            same = sha256(remote) == sha256(folder / name)
            problems += not same
            print(f"{name}: {'same as the folder' if same else 'DIFFERS from the folder'}")
        from predict import load_model, predict_array
        net, config = load_model(temp)
        predict_array(net, config, np.full((len(BANDS), 256, 256), 1000, np.uint16))
        print("The downloaded model loads and runs.")
    print("Everything matches." if not problems else f"{problems} file{'s differ' if problems > 1 else ' differs'}: run upload again.")
    raise SystemExit(1 if problems else 0)


GITHUB_USE = """## Use

From the root of the code repository, after `pip install -r requirements.txt`:

```bash
python tools/predict.py --model models/{name} --image tile_s2.tif --out tile_pred.tif
```

```python
import sys
sys.path.insert(0, "tools")
from predict import load_model, predict_array
net, config = load_model("models/{name}")
mask = predict_array(net, config, image)  # image: (13, height, width), reflectance x 10,000
```

The first use downloads the weights, `model.safetensors` ({size:.0f} MB), from
[huggingface.co/{repo}](https://huggingface.co/{repo}) into this folder and checks them against the sha256 in
`config.json`. `notebooks/02_run_the_model.ipynb` runs the model on random test tiles and draws the results.

"""


def run_export(args):
    folder = folder_for(args)
    missing = [name for name in ("model.safetensors", "config.json", "metrics.json", "README.md") if not (folder / name).exists()]
    if missing:
        raise SystemExit(f"{folder} lacks {', '.join(missing)}: run build first.")
    target = Path(args.to) / folder.name
    target.mkdir(parents=True, exist_ok=True)
    config = json.loads((folder / "config.json").read_text())
    config["hub_repo"] = args.repo
    config["weights_sha256"] = sha256(folder / "model.safetensors")
    (target / "config.json").write_text(json.dumps(config, indent=1))
    shutil.copy(folder / "metrics.json", target / "metrics.json")
    card = (folder / "README.md").read_text()
    if card.startswith("---\n"):
        card = card[card.index("\n---\n", 4) + 5:].lstrip("\n")
    start, end = card.index("## Use\n"), card.index("**Input:**")
    size = (folder / "model.safetensors").stat().st_size / 1e6
    card = card[:start] + GITHUB_USE.format(name=folder.name, repo=args.repo, size=size) + card[end:]
    (target / "README.md").write_text(card)
    stray = target / "model.safetensors"
    print(f"Wrote {target}: config.json, metrics.json and README.md. The weights stay on the Hub"
          + (f"; {stray} is a local copy, kept out of git by .gitignore." if stray.exists() else "."))


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("build", "upload", "check", "export"):
        command = sub.add_parser(name)
        command.add_argument("--spec", default="unet_r34", choices=sorted(RELEASE))
        command.add_argument("--repo", default="neet1797/burn2scar-unet-r34", help="Hugging Face model repo")
        command.add_argument("--out", default="release_models", help="folder that holds the model folders")
        if name == "build":
            command.add_argument("--results", default="results")
            command.add_argument("--split", default="split")
        if name == "export":
            command.add_argument("--to", default="models", help="folder of the code repository that holds the models")
    args = parser.parse_args()
    {"build": run_build, "upload": run_upload, "check": run_check, "export": run_export}[args.command](args)


if __name__ == "__main__":
    main()
