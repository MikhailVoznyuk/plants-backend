from fastapi import APIRouter, UploadFile, File, HTTPException
from app.pipeline import Pipeline
from app.schemas import InferResponse
import numpy as np, cv2

router = APIRouter(tags=["inference"])
pipe = Pipeline()

@router.post("/infer", response_model=InferResponse)
async def infer(file: UploadFile = File(...)):
    try:
        data = await file.read()
        image = cv2.imdecode(np.frombuffer(data, dtype=np.uint8), cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError("Not an image")
        out = pipe.run(image)
        return InferResponse(**out)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
