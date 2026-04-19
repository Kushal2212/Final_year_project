import json
import numpy as np
import tensorflow as tf
from pathlib import Path
import cv2

# ── Paths ───────────────────────────────────────────────
BASE_DIR   = Path(__file__).parent.parent
MODEL_DIR  = BASE_DIR / "models"

EFFNET_PATH    = MODEL_DIR / "model_efficientnet.keras"
MOBILENET_PATH = MODEL_DIR / "model_mobilenet.keras"
LABELS_PATH    = MODEL_DIR / "class_labels.json"

IMG_SIZE = 224

# ── Thresholds ──────────────────────────────────────────
REJECT_THRESHOLD = 78.0
WARN_THRESHOLD   = 88.0

# ── Disease Info ────────────────────────────────────────
DISEASE_INFO = {
    "healthy": {
        "nepali": "स्वस्थ",
        "description": "Plant is healthy.",
        "recommendation": "Continue proper care.",
        "severity": "None",
    },
    "chhirke": {
        "nepali": "छिर्के रोग",
        "description": "Viral disease spread by aphids.",
        "recommendation": "Remove infected plants. Use neem oil.",
        "severity": "High",
    },
    "leaf_blight": {
        "nepali": "पात झुल्सा रोग",
        "description": "Fungal disease causing brown spots.",
        "recommendation": "Apply fungicide.",
        "severity": "Medium",
    },
}

# ── Cache ───────────────────────────────────────────────
_effnet = None
_mobilenet = None
_index_to_class = None


def _load_resources():
    global _effnet, _mobilenet, _index_to_class

    if _effnet is not None:
        return _effnet, _mobilenet, _index_to_class

    if not EFFNET_PATH.exists() or not MOBILENET_PATH.exists():
        raise FileNotFoundError("Model files not found. Upload /models folder to server.")

    _effnet = tf.keras.models.load_model(EFFNET_PATH)
    _mobilenet = tf.keras.models.load_model(MOBILENET_PATH)

    with open(LABELS_PATH) as f:
        class_indices = json.load(f)

    # reverse mapping
    _index_to_class = {v: k for k, v in class_indices.items()}

    print("✅ Models loaded")

    return _effnet, _mobilenet, _index_to_class


# ── Prediction ──────────────────────────────────────────
def predict(img_path: str) -> dict:
    try:
        effnet, mobilenet, index_to_class = _load_resources()

        # Load image
        img = tf.keras.utils.load_img(img_path, target_size=(IMG_SIZE, IMG_SIZE))
        arr = tf.keras.utils.img_to_array(img)

        # Basic validation
        gray = cv2.cvtColor(arr.astype("uint8"), cv2.COLOR_RGB2GRAY)

        if np.std(gray) < 10:
            return {"error": "Image too plain"}

        if np.mean(gray) < 20:
            return {"error": "Image too dark"}

        if np.mean(gray) > 240:
            return {"error": "Image too bright"}

        arr = np.expand_dims(arr, axis=0).astype("float32")

        # Predictions
        p1 = effnet.predict(arr, verbose=0)[0]
        p2 = mobilenet.predict(arr, verbose=0)[0]

        preds = (p1 + p2) / 2

        top_idx = int(np.argmax(preds))
        confidence = float(preds[top_idx]) * 100

        # Confidence gap check
        sorted_preds = sorted(preds, reverse=True)
        gap = (sorted_preds[0] - sorted_preds[1]) * 100

        if confidence < REJECT_THRESHOLD or gap < 15:
            return {
                "error": "not_cardamom",
                "confidence": round(confidence, 2),
                "message": "Not confident this is a cardamom leaf."
            }

        disease = index_to_class[top_idx]
        info = DISEASE_INFO.get(disease, {})

        return {
            "disease": disease,
            "confidence": round(confidence, 2),
            "nepali": info.get("nepali"),
            "description": info.get("description"),
            "recommendation": info.get("recommendation"),
            "severity": info.get("severity"),
        }

    except Exception as e:
        return {
            "error": "server_error",
            "message": str(e)
        }