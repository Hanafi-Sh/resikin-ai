"""
ResikIn Waste Classifier - FastAPI Inference Server
Main entry point for the REST API.
"""

import io
import base64
import logging
from fastapi import FastAPI, HTTPException, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image

from .schemas import PredictRequest, PredictResponse
from .inference import get_prediction, model

app = FastAPI(
    title="ResikIn Waste Classifier API",
    description="API untuk klasifikasi gambar sampah menggunakan fine-tuned CLIP ViT-B/32",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

logger = logging.getLogger(__name__)

@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": model is not None}

# --- PRIMARY ENDPOINTS (BOT COMPATIBILITY) ---

@app.post("/api/ai/validate-image", response_model=PredictResponse)
async def validate_image(request: PredictRequest):
    """Endpoint utama yang digunakan oleh Telegram Bot (Base64)."""
    try:
        b64 = request.image
        if "," in b64:
            b64 = b64.split(",")[1]
        image = Image.open(io.BytesIO(base64.b64decode(b64))).convert("RGB")
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid base64 image")

    return get_prediction(image)

@app.post("/api/ai/validate-image-file", response_model=PredictResponse)
async def validate_image_file(file: UploadFile = File(...)):
    """Endpoint untuk upload file gambar langsung."""
    try:
        image_bytes = await file.read()
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid image file")

    return get_prediction(image)

# --- ASSIGNMENT COMPLIANCE ALIASES ---

@app.post("/predict", response_model=PredictResponse)
async def predict_alias(request: PredictRequest):
    """Alias untuk mematuhi format penugasan DSAI (Base64)."""
    return await validate_image(request)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.main:app", host="0.0.0.0", port=8001, reload=True)
