import sys, os, json, numpy as np, torch
from pathlib import Path
ROOT = Path(os.environ.get("T1_ROOT", Path.home() / "t1"))
sys.path.insert(0, str(ROOT / "starter"))
from wmloc.data import LocalisationDataset
from wmloc.utils import downsample_mask, miou_image, score, to_tensor
from wmloc.watermarks import load_watermark
DATA = ROOT / "release"
MODELS = ROOT / "starter" / "models"
torch.set_num_threads(int(os.environ.get("T1_THREADS", "4")))

def ds(split, method=None):
    return LocalisationDataset(DATA, split, methods=[method] if method else None)

def wm(name):
    return load_watermark(name, MODELS, "cpu")
