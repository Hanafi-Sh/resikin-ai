from pydantic import BaseModel
from typing import Optional, Dict

class PredictRequest(BaseModel):
    image: str  # base64 encoded image

class PredictResponse(BaseModel):
    success: bool
    is_waste: bool
    confidence: float
    suggested_category: Optional[str] = None
    detail: Dict
