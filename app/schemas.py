from pydantic import BaseModel, Field
from typing import List, Optional, Dict

class BBox(BaseModel):
    x1: int
    y1: int
    x2: int
    y2: int

class PlantItem(BaseModel):
    id: int
    cls: str
    conf: float
    bbox: BBox
    area: int
    tilt_deg: float
    dry_ratio: float
    species: Optional[str] = None
    mask_rle: Optional[str] = None
    health_score: float = 100.0
    health_grade: Optional[str] = None

class DefectItem(BaseModel):
    id: int
    cls: str
    conf: float
    bbox: BBox
    area: int
    plant_id: Optional[int] = None
    mask_rle: Optional[str] = None
    area_ratio: float = 0.0
    severity: Optional[str] = None

class InferResponse(BaseModel):
    request_id: str
    status: str = Field(..., description="OK | NO_PLANTS | ERROR")
    overlay_path: Optional[str] = None
    report_json_path: Optional[str] = None
    plants: List[PlantItem] = []
    defects: List[DefectItem] = []
    extras: Dict[str, str] = {}