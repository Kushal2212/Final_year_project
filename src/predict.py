import os
import json
import cv2
from matplotlib.pyplot import gray
import numpy as np
import tensorflow as tf
from pathlib import Path

BASE_DIR      = Path(__file__).parent.parent
MODEL_DIR     = BASE_DIR / "models"
EFFNET_PATH   = MODEL_DIR / "model_efficientnet.keras"
MOBILENET_PATH= MODEL_DIR / "model_mobilenet.keras"
LABELS_PATH   = MODEL_DIR / "class_labels.json"
IMG_SIZE      = 224

# ── Confidence thresholds ──────────────────────────────────────────────────
REJECT_THRESHOLD = 78.0   # below this → not a cardamom leaf
WARN_THRESHOLD   = 88.0   # below this → show warning to farmer

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
        "recommendation": "Remove infected plants. Spray neem oil or imidacloprid to control aphids.",
        "severity":       "High",
    },
    "leaf_blight": {
        "nepali":         "पात झुल्सा रोग",
        "description":    "Fungal disease causing brown water-soaked lesions on leaves.",
        "recommendation": "Apply mancozeb or copper-based fungicide. Improve drainage.",
        "severity":       "Medium",
    },
}

# ── Cache models in memory ─────────────────────────────────────────────────
_effnet_model    = None
_mobilenet_model = None
_index_to_class  = None


def _load_resources():
    global _effnet_model, _mobilenet_model, _index_to_class

    if _effnet_model is None:
        # Check both model files exist
        if not EFFNET_PATH.exists():
            raise FileNotFoundError(
                f"EfficientNetB0 model not found at:\n  {EFFNET_PATH}\n\n"
                "Train first: python main.py train"
            )
        if not MOBILENET_PATH.exists():
            raise FileNotFoundError(
                f"MobileNetV2 model not found at:\n  {MOBILENET_PATH}\n\n"
                "Train first: python main.py train"
            )

        print("Loading EfficientNetB0 …")
        _effnet_model    = tf.keras.models.load_model(str(EFFNET_PATH))
        print("Loading MobileNetV2 …")
        _mobilenet_model = tf.keras.models.load_model(str(MOBILENET_PATH))

        with open(LABELS_PATH) as f:
            class_indices = json.load(f)

        # Handle both formats
        first_key = list(class_indices.keys())[0]
        if first_key.isdigit():
            _index_to_class = {int(k): v for k, v in class_indices.items()}
        else:
            _index_to_class = {v: k for k, v in class_indices.items()}

        print("✅ Ensemble models loaded (EfficientNetB0 + MobileNetV2)")

    return _effnet_model, _mobilenet_model, _index_to_class


def predict(img_path: str) -> dict:
    """
    Predict disease using soft voting ensemble.

    Flow:
        Image
          ↓
        EfficientNetB0  →  [0.1, 0.8, 0.1]
        MobileNetV2     →  [0.2, 0.7, 0.1]
                             ↓
                    Average: [0.15, 0.75, 0.1]
                             ↓
                    Final:   healthy (75%)
    """
    effnet, mobilenet, index_to_class = _load_resources()

    # ── Load image — raw [0, 255], NO division ────────────────────────────
    img = tf.keras.utils.load_img(img_path, target_size=(IMG_SIZE, IMG_SIZE))
    arr = tf.keras.utils.img_to_array(img)   # [0,255]
    gray = cv2.cvtColor(arr.astype("uint8"), cv2.COLOR_RGB2GRAY)

    if np.std(gray) < 10:
        return {
            "error": "Image too plain or blank. Please upload a clear leaf image."
    }
    
    if np.mean(gray) < 20:
        return {"error": "Image too dark"}

    if np.mean(gray) > 240:
        return {"error": "Image too bright"}    
        
    arr_batch = np.expand_dims(arr, axis=0).astype("float32")

    # ── Get predictions from both models ──────────────────────────────────
    preds_eff = effnet.predict(arr_batch,    verbose=0)[0]
    preds_mob = mobilenet.predict(arr_batch, verbose=0)[0]

    # ── Soft voting: average probabilities ───────────────────────────────
    # This is the key step — combining both models
    preds_ensemble = (preds_eff + preds_mob) / 2.0

    top_idx    = int(np.argmax(preds_ensemble))
    confidence = float(preds_ensemble[top_idx]) * 100

    # ── All predictions sorted ────────────────────────────────────────────
    all_predictions = sorted(
        [
            {
                "class":          index_to_class[i],
                "confidence":     round(float(preds_ensemble[i]) * 100, 2),
                "conf_effnet":    round(float(preds_eff[i]) * 100, 2),
                "conf_mobilenet": round(float(preds_mob[i]) * 100, 2),
            }
            for i in range(len(preds_ensemble))
        ],
        key=lambda x: x["confidence"],
        reverse=True,
    )

    # ── Rejection check ───────────────────────────────────────────────────
    sorted_preds  = sorted(preds_ensemble, reverse=True)
    top_conf      = float(sorted_preds[0]) * 100
    second_conf   = float(sorted_preds[1]) * 100
    confidence_gap = top_conf - second_conf

    if confidence < REJECT_THRESHOLD or confidence_gap < 15:
        return {
            "error":           "not_cardamom",
            "message":         (
                f"This does not look like a cardamom leaf. "
                f"Both models are not confident enough ({confidence:.1f}%). "
                f"Please upload a clear, well-lit cardamom leaf image."
            ),
            "confidence":      round(confidence, 2),
            "all_predictions": all_predictions,
        }

    # ── Disease info ──────────────────────────────────────────────────────
    disease_key = index_to_class[top_idx]
    info = DISEASE_INFO.get(disease_key, {
        "nepali":         disease_key,
        "description":    "Unknown disease.",
        "recommendation": "Consult an agricultural expert.",
        "severity":       "Unknown",
    })

    # ── Honest confidence label ───────────────────────────────────────────
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
        "model_details": {
            "efficientnet_top":  round(float(preds_eff[top_idx])  * 100, 2),
            "mobilenet_top":     round(float(preds_mob[top_idx])  * 100, 2),
            "ensemble_top":      round(confidence, 2),
        },
        "image_path": img_path,
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

    print("\n" + "=" * 55)
    print("  SOFT VOTING ENSEMBLE PREDICTION")
    print("=" * 55)
    print(f"  Disease           : {r['disease'].upper().replace('_', ' ')}")
    print(f"  Nepali            : {r['nepali']}")
    print(f"  Ensemble conf.    : {r['confidence']}%")
    print(f"  EfficientNet conf.: {r['model_details']['efficientnet_top']}%")
    print(f"  MobileNet conf.   : {r['model_details']['mobilenet_top']}%")
    print(f"  Confidence level  : {r['confidence_level']} ({r['confidence_label']})")
    print(f"  Severity          : {r['severity']}")
    if r['low_confidence']:
        print(f"\n  ⚠️  Low confidence. Verify with an agricultural expert.")
    print(f"\n  Description :\n    {r['description']}")
    print(f"\n  Recommendation :\n    {r['recommendation']}")
    print("=" * 55)