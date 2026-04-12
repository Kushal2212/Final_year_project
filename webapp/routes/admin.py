from flask import Blueprint, jsonify, request
from flask_jwt_extended import jwt_required, get_jwt_identity
from webapp.extensions import db
from webapp.models import User, Prediction
from sqlalchemy import func
from datetime import datetime, timedelta

admin_bp = Blueprint("admin", __name__, url_prefix="/api/admin")


def check_admin():
    user = User.query.get(int(get_jwt_identity()))
    return user and user.is_admin


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
    paginated   = query.paginate(page=page, per_page=per_page, error_out=False)
    users_data  = []
    for u in paginated.items:
        d = u.to_dict()
        d["predictions"] = Prediction.query.filter_by(user_id=u.id).count()
        users_data.append(d)

    return jsonify({
        "users": users_data,
        "total": paginated.total,
        "page":  page,
        "pages": paginated.pages,
    }), 200


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
        row            = p.to_dict()
        u              = User.query.get(p.user_id)
        row["user_name"]  = u.name  if u else "—"
        row["user_email"] = u.email if u else "—"
        data.append(row)

    return jsonify({
        "predictions": data,
        "total":       paginated.total,
        "page":        page,
        "pages":       paginated.pages,
    }), 200


@admin_bp.route("/users/<int:uid>", methods=["DELETE"])
@jwt_required()
def delete_user(uid):
    if not check_admin():
        return jsonify({"error": "Admin access required"}), 403
    user = User.query.get_or_404(uid)
    if user.is_admin:
        return jsonify({"error": "Cannot delete admin"}), 400
    Prediction.query.filter_by(user_id=uid).delete()
    db.session.delete(user)
    db.session.commit()
    return jsonify({"message": f"User {user.name} deleted"}), 200


@admin_bp.route("/users/<int:uid>/make-admin", methods=["POST"])
@jwt_required()
def make_admin(uid):
    if not check_admin():
        return jsonify({"error": "Admin access required"}), 403
    user          = User.query.get_or_404(uid)
    user.is_admin = True
    db.session.commit()
    return jsonify({"message": f"{user.name} is now admin"}), 200