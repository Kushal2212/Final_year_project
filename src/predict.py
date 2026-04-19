import os
import json
import cv2
import numpy as np
import tensorflow as tf
from pathlib import Path

BASE_DIR       = Path(__file__).resolve().parent.parent
MODEL_DIR      = BASE_DIR / "models"
EFFNET_PATH    = MODEL_DIR / "model_efficientnet.keras"
MOBILENET_PATH = MODEL_DIR / "model_mobilenet.keras"
LABELS_PATH    = MODEL_DIR / "class_labels.json"
IMG_SIZE       = 224

# ── Confidence thresholds ──────────────────────────────────────────────────
REJECT_THRESHOLD = 75.0   # below → not a cardamom leaf 
WARN_THRESHOLD   = 85.0   # below → show low-confidence warning

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
    "chirke": {
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

# ── Model cache ────────────────────────────────────────────────────────────
_effnet_model    = None
_mobilenet_model = None
_index_to_class  = None


def _load_resources():
    global _effnet_model, _mobilenet_model, _index_to_class

    if _effnet_model is not None:
        return _effnet_model, _mobilenet_model, _index_to_class

    # ── Validate model files exist ─────────────────────────────────────────
    missing = []
    for p in (EFFNET_PATH, MOBILENET_PATH, LABELS_PATH):
        if not p.exists():
            missing.append(str(p))
    if missing:
        raise FileNotFoundError(
            f"Missing model files:\n" + "\n".join(f"  {m}" for m in missing)
        )

    print("Loading EfficientNetB0 …")
    _effnet_model    = tf.keras.models.load_model(str(EFFNET_PATH))

    print("Loading MobileNetV2 …")
    _mobilenet_model = tf.keras.models.load_model(str(MOBILENET_PATH))

    # ── Load class labels — handle BOTH formats ────────────────────────────
    #
    #  main.py produces:  {"0": "chirke", "1": "healthy", "2": "leaf_blight"}
    #  old format was:    {"chirke": 0,   "healthy": 1,   "leaf_blight": 2  }
    #
    #  We need: {0: "chirke", 1: "healthy", 2: "leaf_blight"}  (int → name)
    #
    with open(LABELS_PATH) as f:
        raw = json.load(f)

    first_key = next(iter(raw))
    if str(first_key).isdigit():
        # Format from main.py: {"0": "chirke", ...}  → {0: "chirke", ...}
        _index_to_class = {int(k): v for k, v in raw.items()}
    else:
        # Old format: {"chirke": 0, ...}  → {0: "chirke", ...}
        _index_to_class = {int(v): k for k, v in raw.items()}

    print(f"✅ Ensemble models loaded — classes: {list(_index_to_class.values())}")
    return _effnet_model, _mobilenet_model, _index_to_class


def predict(img_path: str) -> dict:
    """
    Predict disease using soft-voting ensemble of EfficientNetB0 + MobileNetV2.
    Returns a result dict or an error dict.
    """
    try:
        effnet, mobilenet, index_to_class = _load_resources()
    except FileNotFoundError as e:
        return {"error": str(e)}
    except Exception as e:
        return {"error": f"Model load failed: {e}"}

    # ── Load image ─────────────────────────────────────────────────────────
    try:
        img = tf.keras.utils.load_img(img_path, target_size=(IMG_SIZE, IMG_SIZE))
        arr = tf.keras.utils.img_to_array(img)   # raw [0, 255] — no rescaling
    except Exception as e:
        return {"error": f"Could not read image: {e}"}

    # ── Basic quality checks ───────────────────────────────────────────────
    gray = cv2.cvtColor(arr.astype("uint8"), cv2.COLOR_RGB2GRAY)
    std  = float(np.std(gray))
    mean = float(np.mean(gray))

    if std < 10:
        return {"error": "Image is too plain or blank. Please upload a clear leaf photo."}
    if mean < 20:
        return {"error": "Image is too dark. Improve lighting and try again."}
    if mean > 240:
        return {"error": "Image is overexposed. Reduce lighting and try again."}

    # ── Run both models ────────────────────────────────────────────────────
    arr_batch = np.expand_dims(arr, axis=0).astype("float32")

    try:
        preds_eff = effnet.predict(arr_batch,    verbose=0)[0]
        preds_mob = mobilenet.predict(arr_batch, verbose=0)[0]
    except Exception as e:
        return {"error": f"Prediction failed: {e}"}

    # ── Soft voting: average probabilities ────────────────────────────────
    preds_ensemble = (preds_eff + preds_mob) / 2.0

    top_idx    = int(np.argmax(preds_ensemble))
    confidence = float(preds_ensemble[top_idx]) * 100

    # ── Build all-predictions list ────────────────────────────────────────
    all_predictions = sorted(
        [
            {
                "class":          index_to_class[i],
                "confidence":     round(float(preds_ensemble[i]) * 100, 2),
                "conf_effnet":    round(float(preds_eff[i])       * 100, 2),
                "conf_mobilenet": round(float(preds_mob[i])       * 100, 2),
            }
            for i in range(len(preds_ensemble))
        ],
        key=lambda x: x["confidence"],
        reverse=True,
    )

    # ── Confidence gap between top-1 and top-2 ────────────────────────────
    sorted_confs   = sorted(preds_ensemble, reverse=True)
    confidence_gap = (float(sorted_confs[0]) - float(sorted_confs[1])) * 100

    # ── Rejection: too uncertain to trust ─────────────────────────────────
    if confidence < REJECT_THRESHOLD or confidence_gap < 10:
        return {
            "error":           "not_cardamom",
            "message":         (
                f"This does not look like a cardamom leaf. "
                f"The model is not confident enough ({confidence:.1f}%). "
                f"Please upload a clear, well-lit photo of a cardamom leaf."
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

    # ── Confidence level label ─────────────────────────────────────────────
    if confidence >= 90:
        conf_level = "High"
        conf_label = "Very confident"
        low_conf   = False
    elif confidence >= WARN_THRESHOLD:
        conf_level = "Medium"
        conf_label = "Fairly confident — result is likely correct"
        low_conf   = False
    else:
        conf_level = "Low"
        conf_label = "Low confidence — please verify with an agricultural expert"
        low_conf   = True

    return {
        "disease":          disease_key,
        "confidence":       round(confidence, 2),
        "confidence_level": conf_level,
        "confidence_label": conf_label,
        "low_confidence":   low_conf,
        "nepali":           info["nepali"],
        "description":      info["description"],
        "recommendation":   info["recommendation"],
        "severity":         info["severity"],
        "all_predictions":  all_predictions,
        "model_details": {
            "efficientnet_top": round(float(preds_eff[top_idx])  * 100, 2),
            "mobilenet_top":    round(float(preds_mob[top_idx])  * 100, 2),
            "ensemble_top":     round(confidence, 2),
        },
    }


# ── CLI ────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python src/predict.py <image_path>")
        sys.exit(1)

    r = predict(sys.argv[1])

    if "error" in r:
        msg = r.get("message", r["error"])
        print(f"\n⚠️  {msg}")
        if r.get("all_predictions"):
            print("\nAll scores:")
            for p in r["all_predictions"]:
                print(f"  {p['class']:15s}: {p['confidence']:.1f}%")
        sys.exit(0)

    print("\n" + "=" * 55)
    print("  ENSEMBLE PREDICTION RESULT")
    print("=" * 55)
    print(f"  Disease      : {r['disease'].upper().replace('_', ' ')}")
    print(f"  Nepali       : {r['nepali']}")
    print(f"  Confidence   : {r['confidence']}% ({r['confidence_level']})")
    print(f"  EfficientNet : {r['model_details']['efficientnet_top']}%")
    print(f"  MobileNetV2  : {r['model_details']['mobilenet_top']}%")
    print(f"  Severity     : {r['severity']}")
    if r['low_confidence']:
        print(f"\n  ⚠️  {r['confidence_label']}")
    print(f"\n  Description:\n    {r['description']}")
    print(f"\n  Recommendation:\n    {r['recommendation']}")
    print("\n  All scores:")
    for p in r['all_predictions']:
        bar = '█' * int(p['confidence'] / 5)
        print(f"  {p['class']:15s}: {p['confidence']:5.1f}%  {bar}")
    print("=" * 55)