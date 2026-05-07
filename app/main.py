"""
ResikIn Waste Classifier - FastAPI Inference Server
Load fine-tuned CLIP model dan serve predictions via REST API.
"""

import io
import os
import base64
import logging
import torch
from fastapi import FastAPI, HTTPException, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from PIL import Image
from transformers import CLIPModel, CLIPProcessor

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

# ── Load Model ──
MODEL_PATH = os.environ.get("MODEL_PATH", "models/clip_waste_classifier.pth")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

logger.info("Loading fine-tuned CLIP model...")
try:
    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
    if os.path.exists(MODEL_PATH):
        model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
        logger.info(f"Fine-tuned weights loaded from {MODEL_PATH}")
    else:
        logger.warning(f"No fine-tuned weights at {MODEL_PATH}, using base CLIP")
    model.to(DEVICE).eval()
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    logger.info("Model ready!")
except Exception as e:
    logger.error(f"Failed to load model: {e}")
    model = None
    processor = None

# ── Text labels for classification ──
WASTE_LABELS = [
    "a photo of garbage, trash, or waste",
    "a photo that is not garbage or waste",
]

# Sub-category labels untuk detail klasifikasi
SUBCATEGORY_LABELS = [
    "an overflowing public waste bin or dumpster",       # tps_penuh
    "illegal dumping of trash on the street",             # sampah_liar
    "uncollected garbage bags on the sidewalk",           # tidak_terangkut
    "a clean environment with no waste",                  # bukan_sampah
]

SUBCATEGORY_MAP = {
    "an overflowing public waste bin or dumpster": "tps_penuh",
    "illegal dumping of trash on the street": "sampah_liar",
    "uncollected garbage bags on the sidewalk": "tidak_terangkut",
    "a clean environment with no waste": "bukan_sampah",
}


class PredictRequest(BaseModel):
    image: str  # base64 encoded image


class PredictResponse(BaseModel):
    success: bool
    is_waste: bool
    confidence: float
    suggested_category: str | None
    detail: dict


def _get_prediction(image: Image.Image) -> PredictResponse:
    if not model or not processor:
        raise HTTPException(status_code=503, detail="Model not available")

    # Step 1: Binary classification (waste vs not_waste)
    with torch.no_grad():
        inputs = processor(
            text=WASTE_LABELS, images=image, return_tensors="pt", padding=True
        ).to(DEVICE)
        outputs = model(**inputs)
        probs = outputs.logits_per_image.softmax(dim=1)[0]

    is_waste = probs[0].item() > probs[1].item()
    confidence = probs[0].item() if is_waste else probs[1].item()

    # Step 2: Sub-category if waste
    suggested_category = None
    sub_detail = {}
    if is_waste:
        with torch.no_grad():
            inputs2 = processor(
                text=SUBCATEGORY_LABELS, images=image, return_tensors="pt", padding=True
            ).to(DEVICE)
            outputs2 = model(**inputs2)
            sub_probs = outputs2.logits_per_image.softmax(dim=1)[0]

        for i, label in enumerate(SUBCATEGORY_LABELS):
            cat = SUBCATEGORY_MAP[label]
            sub_detail[cat] = round(sub_probs[i].item(), 4)

        top_sub_idx = sub_probs.argmax().item()
        top_sub_label = SUBCATEGORY_LABELS[top_sub_idx]
        suggested_category = SUBCATEGORY_MAP[top_sub_label]

    return PredictResponse(
        success=True,
        is_waste=is_waste,
        confidence=round(confidence, 4),
        suggested_category=suggested_category,
        detail={
            "waste_prob": round(probs[0].item(), 4),
            "not_waste_prob": round(probs[1].item(), 4),
            "subcategories": sub_detail,
        },
    )


@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": model is not None}


@app.post("/predict", response_model=PredictResponse)
async def predict(request: PredictRequest):
    # Decode base64 image
    try:
        b64 = request.image
        if "," in b64:
            b64 = b64.split(",")[1]
        image = Image.open(io.BytesIO(base64.b64decode(b64))).convert("RGB")
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid base64 image")

    return _get_prediction(image)


@app.post("/predict-file", response_model=PredictResponse)
async def predict_file(file: UploadFile = File(...)):
    # Load image from file upload
    try:
        image_bytes = await file.read()
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid image file")

    return _get_prediction(image)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.main:app", host="0.0.0.0", port=8001, reload=True)
