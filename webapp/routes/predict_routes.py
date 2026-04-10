"""
predict_routes.py
─────────────────
Prediction REST API endpoints.
"""

import os
import sys
import uuid

from flask import Blueprint, request, jsonify, current_app
from flask_jwt_extended import jwt_required, get_jwt_identity
from werkzeug.utils import secure_filename

from webapp.extensions import db
from webapp.models import Prediction

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))
from predict import predict as ml_predict  # noqa: E402

predict_bp = Blueprint("predict", __name__, url_prefix="/api")
ALLOWED_EXT = {"jpg", "jpeg", "png", "bmp", "webp"}


def allowed(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXT


@predict_bp.route("/predict", methods=["POST"])
@jwt_required()
def predict():
    user_id = int(get_jwt_identity())

    if "file" not in request.files:
        return jsonify({"error": "No image file uploaded"}), 400

    file = request.files["file"]
    if file.filename == "" or not allowed(file.filename):
        return jsonify({"error": "Invalid file. Use JPG, PNG or BMP"}), 400

    # Save uploaded image
    ext = file.filename.rsplit(".", 1)[1].lower()
    filename = f"{uuid.uuid4().hex}.{ext}"
    upload_dir = os.path.join(current_app.root_path, "static", "uploads")
    os.makedirs(upload_dir, exist_ok=True)
    filepath = os.path.join(upload_dir, secure_filename(filename))
    file.save(filepath)

    # Run ML model
    try:
        result = ml_predict(filepath)
    except FileNotFoundError as e:
        return jsonify({"error": str(e)}), 503
    except Exception as e:
        import traceback
        traceback.print_exc()   # ← prints full error in terminal
        return jsonify({"error": f"Prediction failed: {e}"}), 500

    # Not a cardamom leaf → return error, do NOT save to database
    if result.get("error") == "not_cardamom":
        return jsonify(result), 200

    # Valid prediction → save to database
    record = Prediction(
        user_id=user_id,
        disease=result["disease"],
        confidence=result["confidence"],
        severity=result["severity"],
        nepali_name=result["nepali"],
        recommendation=result["recommendation"],
        image_filename=filename,
    )
    db.session.add(record)
    db.session.commit()

    return jsonify({
        "prediction_id":    record.id,
        "disease":          result["disease"],
        "confidence":       result["confidence"],
        "confidence_level": result.get("confidence_level", "Medium"),
        "confidence_label": result.get("confidence_label", ""),
        "low_confidence":   result.get("low_confidence", False),
        "severity":         result["severity"],
        "nepali":           result["nepali"],
        "description":      result["description"],
        "recommendation":   result["recommendation"],
        "all_predictions":  result["all_predictions"],
        "image_url":        f"/static/uploads/{filename}",
        "saved":            True,
    }), 200


@predict_bp.route("/history", methods=["GET"])
@jwt_required()
def history():
    user_id = int(get_jwt_identity())
    page = request.args.get("page",    1,    type=int)
    per_page = request.args.get("limit",   10,   type=int)
    disease = request.args.get("disease", None)

    query = Prediction.query.filter_by(user_id=user_id).order_by(
        Prediction.created_at.desc()
    )
    if disease:
        query = query.filter_by(disease=disease)

    paginated = query.paginate(page=page, per_page=per_page, error_out=False)

    return jsonify({
        "predictions": [p.to_dict() for p in paginated.items],
        "total":       paginated.total,
        "page":        page,
        "pages":       paginated.pages,
    }), 200


@predict_bp.route("/history/<int:pred_id>", methods=["GET"])
@jwt_required()
def history_detail(pred_id):
    user_id = int(get_jwt_identity())
    record = Prediction.query.filter_by(
        id=pred_id, user_id=user_id).first_or_404()
    return jsonify({"prediction": record.to_dict()}), 200


@predict_bp.route("/history/<int:pred_id>", methods=["DELETE"])
@jwt_required()
def delete_prediction(pred_id):
    user_id = int(get_jwt_identity())
    record = Prediction.query.filter_by(
        id=pred_id, user_id=user_id).first_or_404()

    if record.image_filename:
        img_path = os.path.join(
            current_app.root_path, "static", "uploads", record.image_filename
        )
        if os.path.exists(img_path):
            os.remove(img_path)

    db.session.delete(record)
    db.session.commit()
    return jsonify({"message": "Prediction deleted"}), 200


@predict_bp.route("/stats", methods=["GET"])
@jwt_required()
def stats():
    user_id = int(get_jwt_identity())
    records = Prediction.query.filter_by(user_id=user_id).all()

    counts = {}
    for r in records:
        counts[r.disease] = counts.get(r.disease, 0) + 1

    return jsonify({
        "total_predictions": len(records),
        "disease_counts":    counts,
    }), 200
