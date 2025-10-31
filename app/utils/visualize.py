from typing import List, Tuple, Optional
import cv2, numpy as np
from .enc import rle_encode

def hex_to_bgr(hexstr: str) -> Tuple[int,int,int]:
    hexstr = hexstr.strip().lstrip("#")
    if len(hexstr) != 6:
        return (0, 255, 0)
    r = int(hexstr[0:2], 16)
    g = int(hexstr[2:4], 16)
    b = int(hexstr[4:6], 16)
    return (b, g, r)

def overlay_instances(image: np.ndarray, plants, defects, plant_colors=None, defect_colors=None):
    out = image.copy()
    # Draw plants
    for i, p in enumerate(plants):
        col = hex_to_bgr(plant_colors[i % len(plant_colors)]) if plant_colors else (0, 200, 0)
        if p.get("mask") is not None:
            mask = p["mask"].astype(bool)
            color = np.array(col, dtype=np.uint8)
            out[mask] = (0.6 * out[mask] + 0.4 * color).astype(np.uint8)
        x1,y1,x2,y2 = p["bbox"]
        cv2.rectangle(out, (x1,y1), (x2,y2), col, 2)
        cv2.putText(out, f"P{p['id']}:{p['cls']} {p['conf']:.2f}", (x1, max(15, y1-5)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, col, 2)
    # Draw defects
    for i, d in enumerate(defects):
        col = hex_to_bgr(defect_colors[i % len(defect_colors)]) if defect_colors else (0, 0, 255)
        if d.get("mask") is not None:
            mask = d["mask"].astype(bool)
            color = np.array(col, dtype=np.uint8)
            out[mask] = (0.6 * out[mask] + 0.4 * color).astype(np.uint8)
        x1,y1,x2,y2 = d["bbox"]
        cv2.rectangle(out, (x1,y1), (x2,y2), col, 2)
        tag = f"D{d['id']}:{d['cls']} {d['conf']:.2f}"
        if d.get("plant_id") is not None:
            tag += f" -> P{d['plant_id']}"
        cv2.putText(out, tag, (x1, y2 + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, col, 2)
    return out

def compute_tilt_deg(mask: np.ndarray) -> float:
    # Angle of main axis relative to vertical
    ys, xs = np.where(mask > 0)
    if len(xs) < 10:
        return 0.0
    pts = np.vstack([xs, ys]).T.astype(np.float32)
    rect = cv2.minAreaRect(pts)
    angle = rect[-1]
    # Convert to tilt from vertical in degrees [0..90]
    if angle < -45:
        angle = angle + 90
    tilt_from_vertical = abs(90 - abs(angle))
    return float(round(tilt_from_vertical, 2))

def compute_dry_ratio(image_bgr: np.ndarray, mask: np.ndarray) -> float:
    # Naive dryness: ratio of low-saturation or brown-ish pixels in mask
    hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    m = mask.astype(bool)
    if m.sum() == 0:
        return 0.0
    # dry if saturation low or hue in brown range (10..25) with low V
    dry = ((s < 50) | (((h >= 10) & (h <= 25)) & (v < 180)))[m]
    return float(round(dry.mean(), 3))
