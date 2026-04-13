from flask import Blueprint, request, jsonify
from webapp.models import ContactMessage
from webapp.extensions import db

contact_bp = Blueprint("contact", __name__, url_prefix="/api")

@contact_bp.route("/contact", methods=["POST"])
def contact():
    data = request.get_json()

    msg = ContactMessage(
        name=data.get("name"),
        email=data.get("email"),
        subject=data.get("subject"),
        message=data.get("message"),
        is_read=False
    )

    db.session.add(msg)
    db.session.commit()

    return jsonify({"success": True})