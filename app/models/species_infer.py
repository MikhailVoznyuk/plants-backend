# app/models/species_infer.py
import os, json
import torch
import torchvision.transforms as T
from PIL import Image
from typing import List, Dict, Any, Optional
from app import config

_TS = None           # torchscript module or False if unavailable
_CLASSES = None      # list[str] or None
_RU = {}             # dict latin->russian

_IMG = int(os.getenv("SPECIES_IMG_SIZE", "384"))
_TF = T.Compose([T.Resize((_IMG, _IMG)), T.ToTensor()])

def _lazy_load() -> None:
    """Load TorchScript + label maps once, only on first call."""
    global _TS, _CLASSES, _RU
    if _TS is not None:
        return
    ts = os.getenv("SPECIES_TS", "").strip()
    if not ts or not os.path.exists(ts):
        _TS = False
        _CLASSES = None
        _RU = {}
        return
    _TS = torch.jit.load(ts, map_location=config.DEVICE).eval()

    cj = os.getenv("SPECIES_CLASSES", "").strip()
    rj = os.getenv("SPECIES_RU_MAP", "").strip()

    _CLASSES = None
    if cj and os.path.exists(cj):
        with open(cj, "r", encoding="utf-8") as f:
            _CLASSES = json.load(f)
    _RU = {}
    if rj and os.path.exists(rj):
        with open(rj, "r", encoding="utf-8") as f:
            _RU = json.load(f)

@torch.inference_mode()
def predict_species(pil_image: Image.Image, topk: int = 3) -> List[Dict[str, Any]]:
    """Return top-k predictions; [] if model/labels are missing."""
    _lazy_load()
    if _TS is False:
        return []
    x = _TF(pil_image.convert("RGB")).unsqueeze(0).to(config.DEVICE)
    logits = _TS(x)
    prob = torch.softmax(logits, dim=1).squeeze(0).float().cpu().numpy()
    idx = prob.argsort()[::-1][:topk]
    out: List[Dict[str, Any]] = []
    for i in idx:
        latin = _CLASSES[i] if _CLASSES and i < len(_CLASSES) else f"class_{i}"
        out.append({
            "latin": latin,
            "russian": _RU.get(latin, latin),
            "prob": float(prob[i]),
        })
    return out