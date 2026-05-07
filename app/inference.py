import os
import torch
import logging
from PIL import Image
from transformers import CLIPModel, CLIPProcessor
from fastapi import HTTPException
from .schemas import PredictResponse

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

def get_prediction(image: Image.Image) -> PredictResponse:
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

    if is_waste:
        message = "Foto terdeteksi sebagai sampah. Silakan lanjutkan pelaporan."
    else:
        message = "Sistem mendeteksi bahwa foto ini kemungkinan bukan sampah. Apakah Anda tetap ingin melanjutkan pelaporan?"

    return PredictResponse(
        success=True,
        is_waste=is_waste,
        confidence=round(confidence, 4),
        message=message,
        suggested_category=suggested_category,
        detail={
            "waste_prob": round(probs[0].item(), 4),
            "not_waste_prob": round(probs[1].item(), 4),
            "subcategories": sub_detail,
        },
    )
