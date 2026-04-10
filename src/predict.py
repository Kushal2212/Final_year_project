"""
predict.py
──────────
Predict cardamom disease from a leaf image.
Honest confidence reporting — no fake 100% results shown to farmers.
"""

import os
import json
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model

BASE_DIR    = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH  = os.path.join(BASE_DIR, "models", "cardamom_disease_model.keras")
LABELS_PATH = os.path.join(BASE_DIR, "models", "class_labels.json")
IMG_SIZE    = 224

# ── Confidence thresholds ──────────────────────────────────────────────────
# Below REJECT_THRESHOLD  → image is probably not cardamom → reject completely
# Below WARN_THRESHOLD    → show result but with a warning to farmer
# Above WARN_THRESHOLD    → show result normally
REJECT_THRESHOLD = 65.0   # below this → not a cardamom leaf
WARN_THRESHOLD   = 80.0   # below this → show but warn farmer

# ── Disease information ────────────────────────────────────────────────────
DISEASE_INFO = {
    "healthy": {
        "nepali":         "स्वस्थ",
        "description":    "Plant is healthy. No disease detected.",
        "recommendation": "Continue regular care: proper shade, watering, and organic fertiliser.",
        "severity":       "None",
    },
    "chhirke": {
        "nepali":         "छिर्के रोग",
        "description":    "Viral disease spread by aphids. Causes mosaic yellowing and stunted growth.",
        "recommendation": "Remove infected plants. Spray neem oil or imidacloprid to control aphids. Use disease-free planting material.",
        "severity":       "High",
    },
    "leaf_blight": {
        "nepali":         "पात झुल्सा रोग",
        "description":    "Fungal disease causing brown water-soaked lesions on leaves, leading to leaf fall.",
        "recommendation": "Apply mancozeb or copper-based fungicide. Improve drainage and air circulation.",
        "severity":       "Medium",
    },
}

# ── Cache model in memory ──────────────────────────────────────────────────
_model          = None
_index_to_class = None


def _load_resources():
    global _model, _index_to_class
    if _model is None:
        _model = load_model(MODEL_PATH)
        with open(LABELS_PATH) as f:
            class_indices = json.load(f)
        _index_to_class = {int(k): v for k, v in class_indices.items()}
    return _model, _index_to_class  # ← must be here, same level as if


def predict(img_path: str) -> dict:
    model, index_to_class = _load_resources()

    # ── Preprocess ────────────────────────────────────────────────────────
    img = tf.keras.utils.load_img(img_path, target_size=(IMG_SIZE, IMG_SIZE))
    arr = tf.keras.utils.img_to_array(img) 
    arr = np.expand_dims(arr, axis=0)

    preds      = model.predict(arr, verbose=0)[0]
    top_idx    = int(np.argmax(preds))
    confidence = float(preds[top_idx]) * 100

    # ── All predictions sorted ────────────────────────────────────────────
    all_predictions = sorted(
        [
            {
                "class":      index_to_class[i],
                "confidence": round(float(preds[i]) * 100, 2)
            }
            for i in range(len(preds))
        ],
        key=lambda x: x["confidence"],
        reverse=True,
    )

    # ── Rejection check ───────────────────────────────────────────────────
    # If all classes score roughly equally → image is probably not cardamom
    # Example: [34%, 33%, 33%] means model has no idea what it is looking at
    top_confidence    = float(preds[top_idx]) * 100
    second_confidence = sorted([float(p) for p in preds], reverse=True)[1] * 100
    confidence_gap    = top_confidence - second_confidence

    if confidence < REJECT_THRESHOLD or confidence_gap < 20:
        return {
            "error":   "not_cardamom",
            "message": (
                f"This does not look like a cardamom leaf. "
                f"The model is not confident enough ({confidence:.1f}%). "
                f"Please upload a clear, well-lit photo of a cardamom leaf only."
            ),
            "confidence":      round(confidence, 2),
            "all_predictions": all_predictions,
        }

    # ── Get disease info ──────────────────────────────────────────────────
    disease_key = index_to_class[top_idx]
    info = DISEASE_INFO.get(disease_key, {
        "nepali":         disease_key,
        "description":    "Unknown disease.",
        "recommendation": "Consult an agricultural expert.",
        "severity":       "Unknown",
    })

    # ── Warning level ─────────────────────────────────────────────────────
    # Honest confidence label shown to farmer
    if confidence >= 90:
        confidence_level = "High"
        confidence_label = "Very confident"
    elif confidence >= WARN_THRESHOLD:
        confidence_level = "Medium"
        confidence_label = "Fairly confident"
    else:
        confidence_level = "Low"
        confidence_label = "Not very confident — please verify with an expert"

    return {
        "disease":          disease_key,
        "confidence":       round(confidence, 2),
        "confidence_level": confidence_level,
        "confidence_label": confidence_label,
        "low_confidence":   confidence < WARN_THRESHOLD,
        "nepali":           info["nepali"],
        "description":      info["description"],
        "recommendation":   info["recommendation"],
        "severity":         info["severity"],
        "all_predictions":  all_predictions,
        "image_path":       img_path,
    }


# ── CLI ────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python src/predict.py <image_path>")
        sys.exit(1)

    r = predict(sys.argv[1])

    if r.get("error") == "not_cardamom":
        print(f"\n⚠️  {r['message']}")
        sys.exit(0)

    print("\n" + "=" * 52)
    print("  CARDAMOM DISEASE PREDICTION")
    print("=" * 52)
    print(f"  Disease           : {r['disease'].upper().replace('_', ' ')}")
    print(f"  Nepali            : {r['nepali']}")
    print(f"  Confidence        : {r['confidence']}%")
    print(f"  Confidence level  : {r['confidence_level']} ({r['confidence_label']})")
    print(f"  Severity          : {r['severity']}")
    if r['low_confidence']:
        print(f"\n  ⚠️  WARNING: Low confidence. Please verify with an expert.")
    print(f"\n  Description :\n    {r['description']}")
    print(f"\n  Recommendation :\n    {r['recommendation']}")
    print("=" * 52)