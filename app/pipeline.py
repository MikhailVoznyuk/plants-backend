# app/pipeline.py

import os
import uuid
import csv
from typing import Dict, Any, List, Tuple, Optional

import cv2
import numpy as np
import pandas as pd
import torch
from PIL import Image

from app import config
from app.models.loaders import LazyModels
from app.utils.visualize import overlay_instances, compute_tilt_deg, compute_dry_ratio
from app.models.species_infer import predict_species

# rle_encode: безопасный фоллбек
try:
    from app.utils.enc import rle_encode
except Exception:
    def rle_encode(mask):
        return None


# -------------------------- helpers (вне класса) --------------------------

def _auto_thresholds(defects: List[Dict[str, Any]]) -> Dict[str, Dict[str, float]]:
    """Расчёт median/p85 по area_ratio по классам для auto-severity."""
    by_cls: Dict[str, List[float]] = {}
    for d in defects:
        ar = d.get("area_ratio", None)
        if ar is None:
            continue
        by_cls.setdefault(d["cls"], []).append(float(ar))

    out: Dict[str, Dict[str, float]] = {}
    for cls, vals in by_cls.items():
        arr = np.array(vals, dtype=float)
        if arr.size >= 3:
            med = float(np.percentile(arr, 50))
            p85 = float(np.percentile(arr, 85))
        else:
            med, p85 = 0.03, 0.10  # холодный старт
        out[cls] = {"median": med, "p85": p85, "n": float(arr.size)}
    return out


def _load_severity_override() -> Optional[Dict[str, str]]:
    path = getattr(config, "SEVERITY_RULES_CSV", "") or ""
    if not path or not os.path.exists(path):
        return None
    try:
        df = pd.read_csv(path)
        m: Dict[str, str] = {}
        for _, row in df.iterrows():
            sev = str(row["severity"]).strip().lower()
            if sev in {"low", "medium", "high"}:
                m[str(row["defect"]).strip()] = sev
        return m
    except Exception:
        return None


def _bump(sev: str) -> str:
    order = ["low", "medium", "high"]
    try:
        i = order.index(sev)
        return order[min(i + 1, len(order) - 1)]
    except ValueError:
        return sev


def _grade_from_score(score: float) -> str:
    if score >= 80:
        return "good"
    if score >= 60:
        return "fair"
    if score >= 40:
        return "poor"
    return "critical"


def _bbox_to_dict(b) -> Dict[str, int]:
    if isinstance(b, dict):
        return {"x1": int(b["x1"]), "y1": int(b["y1"]), "x2": int(b["x2"]), "y2": int(b["y2"])}
    if isinstance(b, (list, tuple)) and len(b) == 4:
        return {"x1": int(b[0]), "y1": int(b[1]), "x2": int(b[2]), "y2": int(b[3])}
    return {"x1": 0, "y1": 0, "x2": 0, "y2": 0}


def _green_veg_mask(image_bgr: np.ndarray) -> np.ndarray:
    hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    mask = (
        (h >= config.VEG_H_MIN) & (h <= config.VEG_H_MAX)
        & (s >= config.VEG_S_MIN) & (v >= config.VEG_V_MIN)
    ).astype(np.uint8) * 255
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k, iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k, iterations=1)
    return mask


def _brown_patch_mask(image_bgr: np.ndarray) -> np.ndarray:
    hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    mask = (
        (h >= config.BRN_H_MIN) & (h <= config.BRN_H_MAX)
        & (s >= config.BRN_S_MIN) & (v <= config.BRN_V_MAX)
    )
    return (mask.astype(np.uint8) * 255)


def _instances_from_mask(mask: np.ndarray, cls_name: str, min_area: int = 5000) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    num, lbl = cv2.connectedComponents(mask)
    idx = 1
    for i in range(1, num):
        comp = (lbl == i).astype(np.uint8)
        area = int(comp.sum())
        if area < min_area:
            continue
        ys, xs = np.where(comp)
        x1, y1, x2, y2 = int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())
        out.append({
            "id": idx,
            "cls": cls_name,
            "conf": 0.30,
            "bbox": [x1, y1, x2, y2],
            "area": area,
            "mask": comp
        })
        idx += 1
    return out


def _iou_bool(a: np.ndarray, b: np.ndarray) -> float:
    inter = np.logical_and(a, b).sum()
    if inter == 0:
        return 0.0
    union = a.sum() + b.sum() - inter
    return float(inter) / float(union + 1e-6)


def _median_depth(mask: np.ndarray, depth: Optional[np.ndarray]) -> Optional[float]:
    if depth is None:
        return None
    m = (mask > 0)
    if not m.any():
        return None
    return float(np.median(depth[m]))


# -------------------------- core class --------------------------

class Pipeline:
    def __init__(self) -> None:
        self.m = LazyModels()

    # ---------- public entry ----------
    def run(self, image_bgr: np.ndarray) -> Dict[str, Any]:
        req_id = str(uuid.uuid4())
        H, W = image_bgr.shape[:2]

        # [D] Depth (оффлайн/None)
        depth = self._depth_safe(image_bgr)

        # [A] Растения (YOLO-Seg)
        plants_pred = self._yolo_seg(self.m.plant_seg(), image_bgr, config.THRESH_PLANT)

        # Фоллбек зелени (опционально; в проде держи PLANT_FALLBACK=0)
        if len(plants_pred) == 0 and config.PLANT_FALLBACK:
            veg = _green_veg_mask(image_bgr)
            plants_pred = _instances_from_mask(veg, "tree", min_area=5000)

        # Если совсем нет растений — всё равно пишем артефакты
        if len(plants_pred) == 0:
            out_dir = os.path.join(config.OUT_DIR, req_id)
            os.makedirs(out_dir, exist_ok=True)
            overlay_path = os.path.join(out_dir, "overlay.jpg")
            cv2.imwrite(overlay_path, image_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), 92])

            report = {
                "request_id": req_id,
                "plants": [],
                "defects": []
            }
            report_json_path = os.path.join(out_dir, "report.json")
            with open(report_json_path, "w", encoding="utf-8") as f:
                import json
                json.dump(report, f, ensure_ascii=False, indent=2)

            # пустые CSV с заголовками
            self._save_csv(out_dir, [], [])

            return {
                "request_id": req_id,
                "status": "NO_PLANTS",
                "overlay_path": overlay_path,
                "report_json_path": report_json_path,
                "plants": [],
                "defects": []
            }

        # Приводим маски растений к (H, W) на всякий
        for p in plants_pred:
            m = p["mask"]
            if m.shape[:2] != (H, W):
                p["mask"] = cv2.resize(m.astype(np.uint8), (W, H), interpolation=cv2.INTER_NEAREST).astype(np.uint8)

        # [C] Вид (TorchScript, без ONNX)
        if os.getenv("SPECIES_TS", "").strip():
            for p in plants_pred:
                x1, y1, x2, y2 = p["bbox"]
                pad = 8
                x1 = max(0, x1 - pad); y1 = max(0, y1 - pad)
                x2 = min(W, x2 + pad); y2 = min(H, y2 + pad)
                crop = image_bgr[y1:y2, x1:x2, ::-1]  # BGR->RGB
                if crop.size == 0:
                    continue
                preds = predict_species(Image.fromarray(crop), topk=3)
                if preds:
                    top1 = preds[0]
                    p["species"] = top1.get("russian", top1["latin"])  # совместимость
                    p["species_latin"] = top1["latin"]
                    p["species_russian"] = top1.get("russian", top1["latin"])
                    p["species_conf"] = float(top1["prob"])
                    p["species_topk"] = preds

        # [B] Дефекты (YOLO-Seg)
        defects_pred = self._yolo_seg(self.m.defect_seg(), image_bgr, config.THRESH_DEFECT)

        # Приводим маски дефектов к (H, W)
        for d in defects_pred:
            m = d["mask"]
            if m.shape[:2] != (H, W):
                d["mask"] = cv2.resize(m.astype(np.uint8), (W, H), interpolation=cv2.INTER_NEAREST).astype(np.uint8)

        # ROI-клип дефектов маской растений
        union_plants = np.zeros((H, W), dtype=np.uint8)
        for p in plants_pred:
            union_plants |= (p["mask"] > 0).astype(np.uint8)

        if config.PLANT_FALLBACK:
            # сделаем маску плотнее: закроем дыры и чуть расширим
            k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (21, 21))
            filled = cv2.morphologyEx((union_plants * 255), cv2.MORPH_CLOSE, k, iterations=1)
            # лёгкая дилатация
            filled = cv2.dilate(filled, k, iterations=1)
            union_plants = (filled > 0).astype(np.uint8)

        clipped_defects: List[Dict[str, Any]] = []
        for d in defects_pred:
            m_bin = (d["mask"] > 0) & (union_plants > 0)
            if not m_bin.any():
                continue
            d2 = d.copy()
            d2["mask"] = (m_bin.astype(np.uint8))
            d2["area"] = int(m_bin.sum())
            ys, xs = np.where(m_bin)
            d2["bbox"] = [int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())]
            clipped_defects.append(d2)

        # Дефект-фоллбек (опционально)
        if len(clipped_defects) == 0 and config.DEFECT_FALLBACK:
            # ВАЖНО: используем заполненный union_plants
            union_plants_mask = (union_plants * 255).astype(np.uint8)
            brn = _brown_patch_mask(image_bgr)
            pseudo = cv2.bitwise_and(brn, brn, mask=union_plants_mask)
            clipped_defects = _instances_from_mask(pseudo, "fungus", min_area=800)

        # Линковка дефект→растение (IoU + tiebreak по глубине)
        self._link_defects_to_plants(clipped_defects, plants_pred, depth)

        # Эвристики на растениях
        for p in plants_pred:
            p["tilt_deg"] = float(compute_tilt_deg(p["mask"]))
            p["dry_ratio"] = float(compute_dry_ratio(image_bgr, p["mask"]))

        # Правила: severity/score/grade
        self._apply_rules(plants_pred, clipped_defects)

        # Визуализация и экспорт
        overlay = overlay_instances(
            image_bgr, plants_pred, clipped_defects,
            plant_colors=config.PLANT_COLORS, defect_colors=config.DEFECT_COLORS
        )
        out_dir = os.path.join(config.OUT_DIR, req_id)
        os.makedirs(out_dir, exist_ok=True)
        overlay_path = os.path.join(out_dir, "overlay.jpg")
        cv2.imwrite(overlay_path, overlay, [int(cv2.IMWRITE_JPEG_QUALITY), 92])

        report = {
            "request_id": req_id,
            "plants": self._serialize_plants(plants_pred),
            "defects": self._serialize_defects(clipped_defects)
        }
        report_json_path = os.path.join(out_dir, "report.json")
        with open(report_json_path, "w", encoding="utf-8") as f:
            import json
            json.dump(report, f, ensure_ascii=False, indent=2)

        self._save_csv(out_dir, plants_pred, clipped_defects)

        return {
            "request_id": req_id,
            "status": "OK",
            "overlay_path": overlay_path,
            "report_json_path": report_json_path,
            "plants": report["plants"],
            "defects": report["defects"]
        }

    # ---------- internals ----------
    def _yolo_seg(self, model, image_bgr, conf_thr) -> List[Dict[str, Any]]:
        H, W = image_bgr.shape[:2]
        results = model.predict(
            source=image_bgr[..., ::-1],  # BGR->RGB
            verbose=False,
            conf=conf_thr,
            device=config.DEVICE
        )
        items: List[Dict[str, Any]] = []
        if len(results) == 0:
            return items

        r = results[0]
        names = r.names
        if r.masks is None:
            return items

        masks_small = r.masks.data.cpu().numpy()  # (N, mh, mw) — НЕ размер исходника
        boxes = r.boxes.xyxy.cpu().numpy()
        confs = r.boxes.conf.cpu().numpy()
        cls_ids = r.boxes.cls.cpu().numpy().astype(int)

        for i in range(masks_small.shape[0]):
            # Приводим маску к (H, W)
            mask = cv2.resize(masks_small[i], (W, H), interpolation=cv2.INTER_NEAREST)
            mask = (mask > 0.5).astype(np.uint8)
            area = int(mask.sum())
            x1, y1, x2, y2 = boxes[i].astype(int).tolist()
            items.append({
                "id": i + 1,
                "cls": names.get(cls_ids[i], str(cls_ids[i])),
                "conf": float(confs[i]),
                "bbox": [x1, y1, x2, y2],
                "area": area,
                "mask": mask
            })
        return items

    def _depth_safe(self, image_bgr: np.ndarray) -> Optional[np.ndarray]:
        """Пытаемся получить depth через LazyModels.depth_map_for(); если нет — None."""
        try:
            if hasattr(self.m, "depth_map_for"):
                d = self.m.depth_map_for(image_bgr)
                if d is None:
                    return None
                H, W = image_bgr.shape[:2]
                if d.shape != (H, W):
                    d = cv2.resize(d, (W, H), interpolation=cv2.INTER_CUBIC)
                d = (d - d.min()) / (d.max() - d.min() + 1e-6)
                return d.astype(np.float32)
        except Exception:
            pass
        return None

    def _link_defects_to_plants(
        self,
        defects: List[Dict[str, Any]],
        plants: List[Dict[str, Any]],
        depth: Optional[np.ndarray]
    ) -> None:
        """Линковка по IoU; при равенстве IoU — выбираем по близости медианы глубины."""
        # заранее медианы глубины растений
        for p in plants:
            p["depth_med"] = _median_depth(p["mask"], depth)

        for d in defects:
            d_mask = d["mask"].astype(bool)
            if not d_mask.any():
                d["plant_id"] = None
                continue

            ious = []
            for p in plants:
                ious.append(_iou_bool(d_mask, p["mask"].astype(bool)))

            if not ious:
                d["plant_id"] = None
                continue

            best = int(np.argmax(ious))
            # tiebreak
            tied = [i for i, v in enumerate(ious) if abs(v - ious[best]) <= 1e-6]
            if len(tied) > 1 and depth is not None:
                dd = _median_depth(d["mask"], depth)
                best = min(tied, key=lambda i: abs((plants[i]["depth_med"] or 0.5) - (dd or 0.5)))

            d["plant_id"] = int(plants[best]["id"])

    # ---------- rules & scoring ----------
    def _apply_rules(self, plants: List[Dict[str, Any]], defects: List[Dict[str, Any]],
                     debug: bool = False, out_dir: Optional[str] = None) -> None:
        """area_ratio, severity (override/auto+mods), health-score, auto-thresholds dump (debug)."""
        # area_ratio
        plant_area = {int(p["id"]): float(p["mask"].sum()) for p in plants}
        for d in defects:
            pid = d.get("plant_id", None)
            p_area = plant_area.get(int(pid), 0.0) if pid is not None else 0.0
            d["area_ratio"] = float(d["area"] / max(1.0, p_area))

        # severity
        override = _load_severity_override()
        thresholds = None
        if override:
            for d in defects:
                d["severity"] = override.get(d["cls"], d.get("severity") or "low")
        else:
            thresholds = _auto_thresholds(defects)
            for d in defects:
                t = thresholds.get(d["cls"], {"median": 0.03, "p85": 0.10})
                ar = float(d.get("area_ratio", 0.0))
                sev = "low" if ar < t["median"] else ("medium" if ar < t["p85"] else "high")
                d["severity"] = sev

        # модификаторы от dry/tilt
        plant_by_id = {int(p["id"]): p for p in plants}
        for d in defects:
            pid = d.get("plant_id", None)
            pl = plant_by_id.get(int(pid), {}) if pid is not None else {}
            tilt = float(pl.get("tilt_deg", 0.0))
            dry = float(pl.get("dry_ratio", 0.0))
            sev = d["severity"]
            cls = d["cls"]
            if cls in {"fungus", "pests"} and dry > 0.45:
                sev = _bump(sev)
            if cls in {"crack", "cavity", "mech_damage"} and tilt > 20.0:
                sev = _bump(sev)
            d["severity"] = sev

        # health-score
        weight = {"cavity": 60.0, "crack": 45.0, "mech_damage": 35.0, "fungus": 30.0, "pests": 25.0}
        mult = {"low": 0.7, "medium": 1.0, "high": 1.5}
        acc: Dict[int, float] = {int(p["id"]): 0.0 for p in plants}
        for d in defects:
            pid = d.get("plant_id", None)
            if pid is None:
                continue
            w = float(weight.get(d["cls"], 25.0))
            m = float(mult.get(d.get("severity") or "low", 1.0))
            ar = float(d.get("area_ratio", 0.0))
            acc[int(pid)] = acc.get(int(pid), 0.0) + w * ar * m

        for p in plants:
            pid = int(p["id"])
            base = float(acc.get(pid, 0.0))
            base += 0.30 * float(p.get("dry_ratio", 0.0)) * 100.0
            base += 0.8 * max(0.0, float(p.get("tilt_deg", 0.0)) - 10.0)
            score = max(0.0, 100.0 - base)
            p["health_score"] = float(round(score, 1))
            p["health_grade"] = _grade_from_score(score)

        # debug dump порогов для авто-режима
        if debug and out_dir and thresholds:
            df = pd.DataFrame(
                [{"defect": k, "median": v["median"], "p85": v["p85"], "n": v["n"]} for k, v in thresholds.items()]
            ).sort_values("defect")
            df.to_csv(os.path.join(out_dir, "auto_rule_thresholds.csv"), index=False)

    # ---------- serialization & CSV ----------
    def _serialize_plants(self, plants: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        out: List[Dict[str, Any]] = []
        for p in plants:
            bb = _bbox_to_dict(p.get("bbox"))
            out.append({
                "id": int(p["id"]),
                "cls": p["cls"],
                "conf": float(p.get("conf", 0.0)),
                "bbox": bb,
                "area": int(p.get("area", 0)),
                "tilt_deg": float(p.get("tilt_deg", 0.0)),
                "dry_ratio": float(p.get("dry_ratio", 0.0)),
                "species": p.get("species"),
                "mask_rle": rle_encode(p.get("mask")) if p.get("mask") is not None else None,
                "health_score": float(p.get("health_score", 100.0)),
                "health_grade": p.get("health_grade"),
                "species_latin": p.get("species_latin"),
                "species_russian": p.get("species_russian"),
                "species_conf": float(p.get("species_conf", 0.0)),
                "species_topk": p.get("species_topk", []),
            })
        return out

    def _serialize_defects(self, defects: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        out: List[Dict[str, Any]] = []
        for d in defects:
            bb = _bbox_to_dict(d.get("bbox"))
            out.append({
                "id": int(d["id"]),
                "cls": d["cls"],
                "conf": float(d.get("conf", 0.0)),
                "bbox": bb,
                "area": int(d.get("area", 0)),
                "plant_id": int(d.get("plant_id")) if d.get("plant_id") is not None else None,
                "mask_rle": rle_encode(d.get("mask")) if d.get("mask") is not None else None,
                "area_ratio": float(d.get("area_ratio", 0.0)),
                "severity": d.get("severity"),
            })
        return out

    def _save_csv(self, out_dir: str, plants: List[Dict[str, Any]], defects: List[Dict[str, Any]]) -> None:
        os.makedirs(out_dir, exist_ok=True)
        # plants.csv
        with open(os.path.join(out_dir, "plants.csv"), "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["id", "cls", "conf", "x1", "y1", "x2", "y2", "area",
                        "tilt_deg", "dry_ratio", "species", "health_score", "health_grade"])
            for p in plants:
                bb = _bbox_to_dict(p.get("bbox"))
                w.writerow([
                    int(p["id"]), p["cls"], float(p.get("conf", 0.0)),
                    bb["x1"], bb["y1"], bb["x2"], bb["y2"],
                    int(p.get("area", 0)),
                    float(p.get("tilt_deg", 0.0)),
                    float(p.get("dry_ratio", 0.0)),
                    p.get("species"),
                    float(p.get("health_score", 100.0)),
                    p.get("health_grade")
                ])
        # defects.csv
        with open(os.path.join(out_dir, "defects.csv"), "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["id", "cls", "conf", "x1", "y1", "x2", "y2", "area",
                        "plant_id", "area_ratio", "severity"])
            for d in defects:
                bb = _bbox_to_dict(d.get("bbox"))
                w.writerow([
                    int(d["id"]), d["cls"], float(d.get("conf", 0.0)),
                    bb["x1"], bb["y1"], bb["x2"], bb["y2"],
                    int(d.get("area", 0)),
                    int(d.get("plant_id")) if d.get("plant_id") is not None else None,
                    float(d.get("area_ratio", 0.0)),
                    d.get("severity")
                ])