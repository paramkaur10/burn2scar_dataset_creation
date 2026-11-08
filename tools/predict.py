"""Run a burn2scar baseline model on Sentinel-2 Level-1C images and write burn2scar-style masks.

  python predict.py --model neet1797/burn2scar-unet-r34 --image tile_s2.tif --out tile_pred.tif
  python predict.py --model ./burn2scar-unet-r34 --image a_s2.tif b_s2.tif --out-dir predictions

--model is a folder holding config.json and model.safetensors, or a Hugging Face model repo, which is
downloaded once into the Hugging Face cache.

Input: a GeoTIFF with the 13 Sentinel-2 Level-1C bands in the order B01, B02, B03, B04, B05, B06, B07,
B08, B8A, B09, B10, B11, B12, as top-of-atmosphere reflectance times 10,000 on a 10 m grid, the way
burn2scar stores its images. Pixels where every band is 0 are no data. The image can be any size: it is
read in 256 x 256 windows, the size of a burn2scar tile.

Output: one band of uint8 with the burn2scar mask values, 0 clear, 2 fresh_burn, 3 old_burn, 4 cloud,
5 cloud shadow, 6 water and 255 no data, with the burn2scar colour table. The models never predict
7 (older_burn): it was left out of training.

Needs numpy, torch, segmentation-models-pytorch, safetensors and rasterio; huggingface_hub too when
--model names a Hub repo.
"""
import argparse
import json
from pathlib import Path

import numpy as np

NODATA = 255
# The colour table of the burn2scar masks (src/s2_pipeline.py and tools/burn_labels.py).
MASK_COLORMAP = {0: (200, 200, 200, 255), 2: (220, 60, 20, 255), 3: (150, 40, 20, 255), 4: (255, 255, 255, 255),
                 5: (60, 60, 90, 255), 6: (30, 90, 200, 255), 7: (120, 95, 70, 255), 255: (0, 0, 0, 0)}


def resolve_model(model):
    """A local folder as it is, or a Hub repo downloaded into the Hugging Face cache."""
    if Path(model).is_dir():
        return Path(model)
    from huggingface_hub import snapshot_download
    return Path(snapshot_download(repo_id=model, allow_patterns=["config.json", "model.safetensors"]))


def load_model(model, device="cpu"):
    """The model, in evaluation mode on the device, and its config.json."""
    import segmentation_models_pytorch as smp
    import torch
    from safetensors.torch import load_file
    folder = resolve_model(model)
    config = json.loads((folder / "config.json").read_text())
    architecture = getattr(smp, config["architecture"])
    # No pretrained weights to fetch: every weight comes from model.safetensors.
    net = architecture(encoder_name=config["encoder"], encoder_weights=None,
                       in_channels=config["in_channels"], classes=len(config["classes"]))
    net.load_state_dict(load_file(folder / "model.safetensors"), strict=True)
    return net.to(torch.device(device)).eval(), config


def predict_array(net, config, image, device="cpu", batch=8):
    """Mask values for an image of shape (13, height, width) in burn2scar's scale (reflectance x 10,000)."""
    import torch
    image = np.asarray(image)
    bands, height, width = image.shape
    if bands != config["in_channels"]:
        raise ValueError(f"The model needs {config['in_channels']} bands in the order {', '.join(config['bands'])}; the image has {bands}.")
    nodata = (image == 0).all(0)
    size = config["tile_size"]
    pad_h, pad_w = -height % size, -width % size
    padded = np.pad(image, ((0, 0), (0, pad_h), (0, pad_w)))
    low, high = config["clip"]
    data = np.clip(padded.astype(np.float32) / config["scale"], low, high)
    mean = np.asarray(config["mean"], np.float32)[:, None, None]
    std = np.asarray(config["std"], np.float32)[:, None, None]
    data = (data - mean) / std
    corners = [(top, left) for top in range(0, data.shape[1], size) for left in range(0, data.shape[2], size)]
    classes = np.zeros(data.shape[1:], np.int64)
    with torch.no_grad():
        for start in range(0, len(corners), batch):
            chunk = corners[start:start + batch]
            windows = np.stack([data[:, top:top + size, left:left + size] for top, left in chunk])
            logits = net(torch.from_numpy(windows).to(torch.device(device)))
            for (top, left), labels in zip(chunk, logits.argmax(1).cpu().numpy()):
                classes[top:top + size, left:left + size] = labels
    values = np.asarray(config["mask_values"], np.uint8)[classes[:height, :width]]
    values[nodata] = NODATA
    return values


def predict_file(net, config, image_path, out_path, device="cpu", batch=8):
    import rasterio
    with rasterio.open(image_path) as src:
        image, profile = src.read(), src.profile
    mask = predict_array(net, config, image, device, batch)
    profile.update(count=1, dtype="uint8", nodata=NODATA, compress="deflate")
    for key in ("photometric", "interleave", "blockxsize", "blockysize", "tiled"):
        profile.pop(key, None)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(out_path, "w", **profile) as dst:
        dst.write(mask, 1)
        dst.write_colormap(1, MASK_COLORMAP)
    return mask


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", required=True, help="folder with config.json and model.safetensors, or a Hub repo id")
    parser.add_argument("--image", required=True, nargs="+", help="one or more 13-band Level-1C GeoTIFFs")
    parser.add_argument("--out", help="output file, with one --image")
    parser.add_argument("--out-dir", help="output folder, with several --image; files are named <image>_pred.tif")
    parser.add_argument("--device", default="", help="cpu or cuda; the default takes cuda when there is one")
    parser.add_argument("--batch", type=int, default=8, help="256 x 256 windows per forward pass")
    args = parser.parse_args()
    if bool(args.out) == bool(args.out_dir) or (args.out and len(args.image) > 1):
        parser.error("give --out with one --image, or --out-dir")
    import torch
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    net, config = load_model(args.model, device)
    for image in args.image:
        out = Path(args.out) if args.out else Path(args.out_dir) / f"{Path(image).stem}_pred.tif"
        mask = predict_file(net, config, image, out, device, args.batch)
        valid = mask != NODATA
        shares = {name: (mask[valid] == value).mean() * 100 if valid.any() else 0.0
                  for name, value in zip(config["classes"], config["mask_values"])}
        print(f"{out}: " + ", ".join(f"{name} {share:.1f} %" for name, share in shares.items()), flush=True)


if __name__ == "__main__":
    main()
