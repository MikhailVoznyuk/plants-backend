# tools/validate_report.py
import os, sys, json

def fail(msg):
    print("FAIL:", msg)
    sys.exit(1)

def ok(msg):
    print("OK:", msg)

def load(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def main(report_path):
    if not os.path.exists(report_path):
        fail(f"report.json not found: {report_path}")
    j = load(report_path)

    # top-level
    for k in ["request_id", "plants", "defects"]:
        if k not in j:
            fail(f"missing top-level key: {k}")
    ok("top-level keys present")

    # plants
    for p in j["plants"]:
        for k in ["id","cls","bbox","area"]:
            if k not in p:
                fail(f"plant missing key: {k}")
        bb = p["bbox"]
        for k in ["x1","y1","x2","y2"]:
            if k not in bb:
                fail(f"bbox missing key: {k}")

        if not (0.0 <= float(p.get("dry_ratio",0.0)) <= 1.0):
            fail("dry_ratio out of [0,1]")

        # tilt sanity: -90..+90 should be enough
        if not (-90.0 <= float(p.get("tilt_deg",0.0)) <= 90.0):
            fail("tilt_deg outside plausible range")

    ok("plants validated")

    # defects
    for d in j["defects"]:
        for k in ["id","cls","bbox","area","plant_id"]:
            if k not in d:
                fail(f"defect missing key: {k}")
        if d["plant_id"] is None:
            fail("defect has no plant_id after ROI/link")
    ok("defects validated")

    print("VALIDATION: PASS")

if __name__ == "__main__":
    rp = sys.argv[1] if len(sys.argv) > 1 else "report.json"
    main(rp)
