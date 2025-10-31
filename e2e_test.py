# tools/e2e_test.py
import os, sys, time, json, argparse
from urllib.parse import urljoin
try:
    import requests
except Exception:
    print("ERROR: requests not installed. Run: py -m pip install -U requests", file=sys.stderr); sys.exit(2)
try:
    from PIL import Image, ImageDraw
except Exception:
    print("ERROR: pillow not installed. Run: py -m pip install -U pillow", file=sys.stderr); sys.exit(2)

def wait_health(base_url, timeout=45):
    url = urljoin(base_url, "/health")
    t0 = time.time()
    while time.time() - t0 < timeout:
        try:
            r = requests.get(url, timeout=3)
            if r.ok and r.json().get("status") == "ok":
                return True
        except Exception:
            pass
        time.sleep(1)
    return False

def gen_green_image(path, w=640, h=480):
    img = Image.new("RGB", (w, h), (20, 160, 60))
    d = ImageDraw.Draw(img)
    d.rectangle([50, 50, w-50, h-50], outline=(0, 100, 0), width=8)
    img.save(path, "JPEG")
    return path

def assert_keys(obj, keys, ctx):
    missing = [k for k in keys if k not in obj]
    if missing: raise AssertionError(f"{ctx}: missing keys: {missing}")

def run(base_url, sample_path=None, report_path=None):
    report = {"base_url": base_url, "ts": time.time(), "steps": []}

    if not wait_health(base_url, 45):
        raise SystemExit("Healthcheck failed")

    if not sample_path or not os.path.exists(sample_path):
        os.makedirs("data/smoke", exist_ok=True)
        sample_path = os.path.join("data/smoke", "sample.jpg")
        gen_green_image(sample_path)

    url = urljoin(base_url, "/infer")
    with open(sample_path, "rb") as f:
        files = {"file": ("sample.jpg", f, "image/jpeg")}
        r = requests.post(url, files=files, timeout=120)
    if not r.ok:
        raise SystemExit(f"/infer failed: {r.status_code} {r.text[:200]}")
    j = r.json()

    assert_keys(j, ["request_id", "plants", "defects", "overlay_path", "report_json_path"], "response")
    assert isinstance(j["plants"], list) and isinstance(j["defects"], list)

    species_env = os.environ.get("SPECIES_TS") or ""
    species_present = False
    for p in j["plants"]:
        if p.get("species") or p.get("species_russian") or p.get("species_latin"):
            species_present = True; break

    report["species_expected"] = bool(species_env)
    report["species_found"] = species_present
    report["ok"] = True

    if report_path:
        with open(report_path, "w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
    print("E2E OK")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://localhost:8000")
    ap.add_argument("--sample", default="")
    ap.add_argument("--report", default="smoke_report.json")
    args = ap.parse_args()
    run(args.url, args.sample, args.report)
