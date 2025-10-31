from fastapi import FastAPI
from app.routers.infer import router as infer_router
from app.routers.debug import router as debug_router
from fastapi.staticfiles import StaticFiles
from app import config

app = FastAPI(title="tree-health-infer-service", version="0.1.0")

app.mount("/artifacts", StaticFiles(directory=config.OUT_DIR), name="artifacts")


@app.get("/health")
def health():
    return {"status": "ok"}

@app.on_event("startup")
async def _prewarm_depth():
    try:
        import torch
        torch.hub.load("intel-isl/MiDaS", "DPT_Hybrid", trust_repo=True)
        torch.hub.load("intel-isl/MiDaS", "transforms", trust_repo=True)
    except Exception as e:
        import logging; logging.getLogger("prewarm").warning(f"midas prewarm failed: {e}")

app.include_router(infer_router)
app.include_router(debug_router)
