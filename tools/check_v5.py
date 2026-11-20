"""Check that the project files and packages the version 5.0 tools use are there, before a long run.

  python tools/check_v5.py

Prints every missing function, option, package or input it finds, or "All present".
"""
import importlib
import os
import subprocess
import sys
from pathlib import Path

os.environ["CUDA_VISIBLE_DEVICES"] = ""
ROOT = Path(__file__).resolve().parents[1]
TOOLS = ROOT / "tools"
sys.path[:0] = [str(TOOLS), str(ROOT)]

NAMES = {
    "burn_labels": ["MASK_CLEAR", "MASK_CLOUD", "MASK_FRESH", "MASK_NODATA", "MASK_OLD", "MASK_SHADOW", "MASK_WATER",
                    "load_perimeters"],
    "build_index": ["fire_attributes"],
    "extra_inference": ["TestTiles"],
    "make_label_figures": ["cems_jobs", "cems_arrays", "by_activation"],
    "review_points": ["MAP_CLASS", "final_labels", "latest_answers", "reference_class"],
    "train_baseline": ["ALL_BANDS", "build_model"],
    "src.effis_loader": ["load_all_perimeters", "load_effis"],
    "src.s2_pipeline": ["S2L1C_COLLECTION", "S2L2A_COLLECTION", "make_sh_config"],
}
PACKAGES = ["PIL", "dotenv", "eolearn", "geopandas", "omnicloudmask", "pyproj", "rasterio", "sentinelhub", "shapely",
            "torch", "yaml"]
OPTIONS = {("burn_labels.py", "patch"): ["--config", "--masks-dir", "--out-dir", "--workers"],
           ("make_age.py",): ["--config", "--masks-dir", "--out-dir", "--workers"]}
INPUTS = ["config.yaml", ".env", "data/v4/index.csv", "data/v4", "data/v1", "results_v4/band_stats.json",
          "validation/cems_tiles.csv", "review2"]

missing = []
for package in PACKAGES:
    try:
        importlib.import_module(package)
    except Exception as error:
        missing.append(f"package {package} ({type(error).__name__}: {error})")
for module, names in NAMES.items():
    try:
        loaded = importlib.import_module(module)
    except Exception as error:
        missing.append(f"{module} cannot be imported ({type(error).__name__}: {error})")
        continue
    missing += [f"{module}.{name}" for name in names if not hasattr(loaded, name)]
for command, options in OPTIONS.items():
    shown = subprocess.run([sys.executable, str(TOOLS / command[0]), *command[1:], "--help"],
                           capture_output=True, text=True, cwd=ROOT)
    missing += [f"{' '.join(command)} {option}" for option in options if option not in shown.stdout]
missing += [f"input {path}" for path in INPUTS if not (ROOT / path).exists()]
runs = sorted((ROOT / "results_v4").glob("*__split__multiclass__seed*/best.pt"))
if not runs:
    missing.append("results_v4/*__split__multiclass__seed*/best.pt (the trained baselines)")
try:
    import yaml
    config = yaml.safe_load((ROOT / "config.yaml").read_text()) or {}
    extra = config.get("effis_extra") or []
    if not extra:
        missing.append("effis_extra in config.yaml (the export with perimeters from 2018)")
    missing += [f"effis_extra file {path}" for path in extra if not (ROOT / path).exists() and not Path(path).exists()]
except Exception as error:
    missing.append(f"config.yaml ({type(error).__name__}: {error})")

print(f"{len(runs)} trained baseline runs on the spatial split")
print("All present" if not missing else "Missing:\n  " + "\n  ".join(missing))
