from flask import Blueprint, jsonify, request
from flask_jwt_extended import jwt_required, get_jwt_identity
from webapp.extensions import db
from webapp.models import User, Prediction, ContactMessage
from sqlalchemy import func
from datetime import datetime, timedelta
import os

admin_bp = Blueprint("admin", __name__, url_prefix="/api/admin")


def check_admin():
    """Return the admin user object, or None if not admin."""
    user = User.query.get(int(get_jwt_identity()))
    return user if (user and user.is_admin) else None


# ════════════════════════════════════════════════════════
#  STATS
# ════════════════════════════════════════════════════════
@admin_bp.route("/stats", methods=["GET"])
@jwt_required()
def stats():
    if not check_admin():
        return jsonify({"error": "Admin access required"}), 403

    total_users       = User.query.count()
    total_predictions = Prediction.query.count()

    # Disease breakdown
    disease_counts = {}
    for disease, count in db.session.query(
        Prediction.disease, func.count(Prediction.id)
    ).group_by(Prediction.disease).all():
        disease_counts[disease] = count

    # Last 7 days daily counts
    daily = []
    for i in range(6, -1, -1):
        day   = datetime.utcnow() - timedelta(days=i)
        start = day.replace(hour=0, minute=0, second=0, microsecond=0)
        end   = start + timedelta(days=1)
        count = Prediction.query.filter(
            Prediction.created_at >= start,
            Prediction.created_at <  end
        ).count()
        daily.append({"date": start.strftime("%b %d"), "count": count})

    week_ago  = datetime.utcnow() - timedelta(days=7)
    new_users = User.query.filter(User.created_at >= week_ago).count()
    new_preds = Prediction.query.filter(Prediction.created_at >= week_ago).count()
    avg_conf  = db.session.query(func.avg(Prediction.confidence)).scalar()

    return jsonify({
        "total_users":       total_users,
        "total_predictions": total_predictions,
        "new_users_week":    new_users,
        "new_preds_week":    new_preds,
        "avg_confidence":    round(avg_conf or 0, 1),
        "disease_counts":    disease_counts,
        "daily_predictions": daily,
    }), 200


# ════════════════════════════════════════════════════════
#  USERS
# ════════════════════════════════════════════════════════
@admin_bp.route("/users", methods=["GET"])
@jwt_required()
def users():
    if not check_admin():
        return jsonify({"error": "Admin access required"}), 403

    page     = request.args.get("page",   1,  type=int)
    per_page = request.args.get("limit",  8,  type=int)
    search   = request.args.get("search", "")

    query = User.query.order_by(User.created_at.desc())
    if search:
        query = query.filter(
            User.name.ilike(f"%{search}%") | User.email.ilike(f"%{search}%")
        )

    paginated  = query.paginate(page=page, per_page=per_page, error_out=False)
    users_data = []
    for u in paginated.items:
        d = u.to_dict()
        d["prediction_count"] = Prediction.query.filter_by(user_id=u.id).count()
        users_data.append(d)

    return jsonify({
        "users": users_data,
        "total": paginated.total,
        "page":  page,
        "pages": paginated.pages,
    }), 200


@admin_bp.route("/users/<int:uid>", methods=["DELETE"])
@jwt_required()
def delete_user(uid):
    admin = check_admin()
    if not admin:
        return jsonify({"error": "Admin access required"}), 403

    # Prevent admin from deleting themselves
    if uid == admin.id:
        return jsonify({"error": "You cannot delete your own account"}), 400

    user = User.query.get_or_404(uid)

    if user.is_admin:
        return jsonify({"error": "Cannot delete another admin account"}), 400

    # Delete user's uploaded images from disk
    predictions = Prediction.query.filter_by(user_id=uid).all()
    for p in predictions:
        if p.image_filename:
            try:
                img_path = os.path.join("webapp", "static", "uploads", p.image_filename)
                if os.path.exists(img_path):
                    os.remove(img_path)
            except Exception:
                pass  # don't block deletion if file missing

    # Delete predictions then user
    Prediction.query.filter_by(user_id=uid).delete()
    db.session.delete(user)
    db.session.commit()

    return jsonify({"message": f"User '{user.name}' and their data deleted"}), 200


@admin_bp.route("/users/<int:uid>/make-admin", methods=["POST"])
@jwt_required()
def make_admin(uid):
    if not check_admin():
        return jsonify({"error": "Admin access required"}), 403

    user = User.query.get_or_404(uid)

    if user.is_admin:
        return jsonify({"message": f"{user.name} is already an admin"}), 200

    user.is_admin = True
    db.session.commit()
    return jsonify({"message": f"{user.name} is now an admin"}), 200


# ── NEW: revoke admin ────────────────────────────────────
@admin_bp.route("/users/<int:uid>/revoke-admin", methods=["POST"])
@jwt_required()
def revoke_admin(uid):
    admin = check_admin()
    if not admin:
        return jsonify({"error": "Admin access required"}), 403

    # Prevent revoking own admin
    if uid == admin.id:
        return jsonify({"error": "You cannot revoke your own admin role"}), 400

    user = User.query.get_or_404(uid)
    if not user.is_admin:
        return jsonify({"message": f"{user.name} is not an admin"}), 200

    user.is_admin = False
    db.session.commit()
    return jsonify({"message": f"{user.name}'s admin role revoked"}), 200


# ════════════════════════════════════════════════════════
#  PREDICTIONS
# ════════════════════════════════════════════════════════
@admin_bp.route("/predictions", methods=["GET"])
@jwt_required()
def predictions():
    if not check_admin():
        return jsonify({"error": "Admin access required"}), 403

    page     = request.args.get("page",    1,    type=int)
    per_page = request.args.get("limit",   10,   type=int)
    disease  = request.args.get("disease", None)

    query = Prediction.query.order_by(Prediction.created_at.desc())
    if disease:
        query = query.filter_by(disease=disease)

    paginated = query.paginate(page=page, per_page=per_page, error_out=False)
    data = []
    for p in paginated.items:
        row               = p.to_dict()
        u                 = User.query.get(p.user_id)
        row["user_name"]  = u.name  if u else "—"
        row["user_email"] = u.email if u else "—"
        data.append(row)

    return jsonify({
        "predictions": data,
        "total":       paginated.total,
        "page":        page,
        "pages":       paginated.pages,
    }), 200


# ── NEW: delete a single prediction ─────────────────────
@admin_bp.route("/predictions/<int:pid>", methods=["DELETE"])
@jwt_required()
def delete_prediction(pid):
    if not check_admin():
        return jsonify({"error": "Admin access required"}), 403

    record = Prediction.query.get_or_404(pid)

    # Remove image file from disk
    if record.image_filename:
        try:
            img_path = os.path.join("webapp", "static", "uploads", record.image_filename)
            if os.path.exists(img_path):
                os.remove(img_path)
        except Exception:
            pass

    db.session.delete(record)
    db.session.commit()
    return jsonify({"message": "Prediction deleted"}), 200

# ── Save contact message ───────────────────────────────────────────────────
@admin_bp.route("/contact", methods=["POST"])
def save_contact():
    data = request.get_json()
    if not data:
        return jsonify({"error": "No data provided"}), 400

    name    = data.get("name",    "").strip()
    email   = data.get("email",   "").strip()
    subject = data.get("subject", "").strip()
    message = data.get("message", "").strip()

    if not name or not email or not message:
        return jsonify({"error": "Name, email and message are required"}), 400

    msg = ContactMessage(
        name=name, email=email,
        subject=subject or "No subject",
        message=message
    )
    db.session.add(msg)
    db.session.commit()
    return jsonify({"message": "Message received! We will reply within 24 hours."}), 201


# ── Get all messages (admin only) ─────────────────────────────────────────
@admin_bp.route("/messages", methods=["GET"])
@jwt_required()
def get_messages():
    if not check_admin():
        return jsonify({"error": "Admin access required"}), 403

    messages = ContactMessage.query.order_by(
        ContactMessage.created_at.desc()
    ).all()
    return jsonify({
        "messages": [m.to_dict() for m in messages],
        "unread":   ContactMessage.query.filter_by(is_read=False).count(),
    }), 200


# ── Mark message as read ───────────────────────────────────────────────────
@admin_bp.route("/messages/<int:msg_id>/read", methods=["POST"])
@jwt_required()
def mark_read(msg_id):
    if not check_admin():
        return jsonify({"error": "Admin access required"}), 403
    msg = ContactMessage.query.get_or_404(msg_id)
    msg.is_read = True
    db.session.commit()
    return jsonify({"message": "Marked as read"}), 200