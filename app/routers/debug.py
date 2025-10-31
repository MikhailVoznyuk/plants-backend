from fastapi import APIRouter, UploadFile, File, HTTPException
import uuid, os, cv2, numpy as np, torch, json
from app import config
from app.utils.visualize import compute_tilt_deg, compute_dry_ratio, overlay_instances
from app.pipeline import Pipeline

router = APIRouter(prefix="/debug", tags=["debug"])

def _make_outdir():
    rid = str(uuid.uuid4())[:8]
    d = os.path.join(config.OUT_DIR, rid)
    os.makedirs(d, exist_ok=True)
    return rid, d

@router.post("/depth")
async def depth(file: UploadFile = File(...)):
    try:
        data = await file.read()
        img = cv2.imdecode(np.frombuffer(data,np.uint8), cv2.IMREAD_COLOR)
        if img is None:
            raise ValueError("Not an image")
        rid, d = _make_outdir()

        midas = torch.hub.load("intel-isl/MiDaS", "DPT_Hybrid"); midas.eval()
        tr = torch.hub.load("intel-isl/MiDaS", "transforms")
        inp = tr.dpt_transform(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        with torch.no_grad():
            dep = midas(inp)
            dep = torch.nn.functional.interpolate(dep.unsqueeze(1), size=img.shape[:2],
                    mode="bicubic", align_corners=False).squeeze().cpu().numpy()
        dep = (dep - dep.min()) / (dep.max() - dep.min() + 1e-9)
        p = os.path.join(d, "D_depth.png"); cv2.imwrite(p, (dep*255).astype(np.uint8))
        return {"request_id": rid, "status": "OK", "depth_png_path": p}
    except Exception as e:
        raise HTTPException(500, str(e))

@router.post("/heuristics")
async def heuristics(file: UploadFile = File(...)):
    try:
        data = await file.read()
        img = cv2.imdecode(np.frombuffer(data,np.uint8), cv2.IMREAD_COLOR)
        if img is None:
            raise ValueError("Not an image")
        rid, d = _make_outdir()

        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV); h,s,v = cv2.split(hsv)
        veg = ((h>=30)&(h<=90)&(s>=40)&(v>=40)).astype(np.uint8)*255  # только для теста
        mask_path = os.path.join(d, "veg_mask.png"); cv2.imwrite(mask_path, veg)

        tilt = float(compute_tilt_deg(veg))
        dry  = float(compute_dry_ratio(img, veg))
        return {"request_id": rid, "status": "OK",
                "mask_png_path": mask_path, "tilt_deg": tilt, "dry_ratio": dry}
    except Exception as e:
        raise HTTPException(500, str(e))

@router.post("/export-smoke")
async def export_smoke(file: UploadFile = File(...)):
    try:
        data = await file.read()
        img = cv2.imdecode(np.frombuffer(data,np.uint8), cv2.IMREAD_COLOR)
        if img is None:
            raise ValueError("Not an image")
        H, W = img.shape[:2]
        rid, d = _make_outdir()

        # синтетическое растение
        plant_mask = np.zeros((H,W), np.uint8)
        cv2.ellipse(plant_mask, (W//2,H//2), (W//4,H//3), 0, 0, 360, 255, -1)
        ys,xs = np.where(plant_mask>0)
        plant = {"id":1,"cls":"tree","conf":0.9,
                 "bbox":[int(xs.min()),int(ys.min()),int(xs.max()),int(ys.max())],
                 "area": int(plant_mask.sum()),
                 "mask": (plant_mask>0).astype(np.uint8),
                 "tilt_deg": compute_tilt_deg(plant_mask),
                 "dry_ratio": compute_dry_ratio(img, plant_mask),
                 "species": config.SPECIES_STUB_LABEL}

        # синтетический дефект внутри кроны
        defect_mask = np.zeros((H,W), np.uint8)
        x1=W//2; y1=H//2; x2=x1+W//8; y2=y1+H//8
        cv2.rectangle(defect_mask,(x1,y1),(x2,y2),255,-1)
        dmask = ((defect_mask>0) & (plant_mask>0)).astype(np.uint8)
        ys,xs = np.where(dmask>0)
        defect = {"id":1,"cls":"crack","conf":0.8,
                  "bbox":[int(xs.min()),int(ys.min()),int(xs.max()),int(ys.max())],
                  "area": int(dmask.sum()),
                  "mask": dmask,
                  "plant_id": 1}

        # виз и экспорт через пайплайн-хелперы
        pipe = Pipeline()
        ov = overlay_instances(img, [plant], [defect], [], [])
        ov_path = os.path.join(d,"overlay.png"); cv2.imwrite(ov_path, ov)
        pipe._apply_rules([plant], [defect], debug=True, out_dir=d)

        report = {"request_id": rid,
                  "plants": pipe._serialize_plants([plant]),
                  "defects": pipe._serialize_defects([defect])}
        rp = os.path.join(d,"report.json")
        with open(rp,"w",encoding="utf-8") as f: json.dump(report, f, ensure_ascii=False, indent=2)

        pipe._save_csv(d, [plant], [defect])
        return {"request_id": rid, "status": "OK",
                "overlay_path": ov_path, "report_json_path": rp}
    except Exception as e:
        raise HTTPException(500, str(e))