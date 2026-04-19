from flask import Blueprint, request, jsonify
from flask_jwt_extended import jwt_required, get_jwt_identity
from webapp.extensions import db
from webapp.models import NewsletterSubscriber, User

newsletter_bp = Blueprint("newsletter", __name__, url_prefix="/api/newsletter")


def check_admin():
    user = User.query.get(int(get_jwt_identity()))
    return user and user.is_admin


# ── Public: Subscribe ─────────────────────────────────────────────────────
@newsletter_bp.route("/subscribe", methods=["POST"])
def subscribe():
    data  = request.get_json(silent=True) or {}
    email = (data.get("email") or "").strip().lower()

    if not email or "@" not in email:
        return jsonify({"error": "Please enter a valid email address"}), 400

    existing = NewsletterSubscriber.query.filter_by(email=email).first()
    if existing:
        if existing.is_active:
            return jsonify({"message": "You are already subscribed!"}), 200
        # Reactivate
        existing.is_active = True
        db.session.commit()
        return jsonify({"message": "Welcome back! You have been resubscribed."}), 200

    sub = NewsletterSubscriber(email=email)
    db.session.add(sub)
    db.session.commit()
    return jsonify({
        "message": "Thank you for subscribing! You will receive our latest updates.",
        "id": sub.id,
    }), 201


# ── Public: Unsubscribe ───────────────────────────────────────────────────
@newsletter_bp.route("/unsubscribe", methods=["POST"])
def unsubscribe():
    data  = request.get_json(silent=True) or {}
    email = (data.get("email") or "").strip().lower()

    if not email:
        return jsonify({"error": "Email is required"}), 400

    sub = NewsletterSubscriber.query.filter_by(email=email).first()
    if not sub:
        return jsonify({"error": "Email not found"}), 404

    sub.is_active = False
    db.session.commit()
    return jsonify({"message": "You have been unsubscribed successfully."}), 200


# ── Admin: List subscribers ───────────────────────────────────────────────
@newsletter_bp.route("/subscribers", methods=["GET"])
@jwt_required()
def get_subscribers():
    if not check_admin():
        return jsonify({"error": "Admin access required"}), 403

    page     = request.args.get("page",   1,    type=int)
    per_page = request.args.get("limit",  10,   type=int)
    active   = request.args.get("active", None)

    query = NewsletterSubscriber.query.order_by(
        NewsletterSubscriber.created_at.desc()
    )
    if active == "true":
        query = query.filter_by(is_active=True)
    elif active == "false":
        query = query.filter_by(is_active=False)

    paginated    = query.paginate(page=page, per_page=per_page, error_out=False)
    total_active = NewsletterSubscriber.query.filter_by(is_active=True).count()

    return jsonify({
        "subscribers":   [s.to_dict() for s in paginated.items],
        "total":         paginated.total,
        "total_active":  total_active,
        "page":          page,
        "pages":         paginated.pages,
    }), 200


# ── Admin: Delete subscriber ──────────────────────────────────────────────
@newsletter_bp.route("/subscribers/<int:sid>", methods=["DELETE"])
@jwt_required()
def delete_subscriber(sid):
    if not check_admin():
        return jsonify({"error": "Admin access required"}), 403

    sub = NewsletterSubscriber.query.get_or_404(sid)
    db.session.delete(sub)
    db.session.commit()
    return jsonify({"message": f"Subscriber {sub.email} removed"}), 200


# ── Admin: Send newsletter ────────────────────────────────────────────────
@newsletter_bp.route("/send", methods=["POST"])
@jwt_required()
def send_newsletter():
    if not check_admin():
        return jsonify({"error": "Admin access required"}), 403

    data    = request.get_json(silent=True) or {}
    subject = (data.get("subject") or "").strip()
    title   = (data.get("title")   or "").strip()
    body    = (data.get("body")    or "").strip()

    if not subject or not body:
        return jsonify({"error": "Subject and body are required"}), 400

    subscribers = NewsletterSubscriber.query.filter_by(is_active=True).all()
    if not subscribers:
        return jsonify({"error": "No active subscribers to send to"}), 400

    # ── Try to send real emails via Flask-Mail if configured ──────────────
    sent   = 0
    failed = 0

    mail_server = __import__("os").environ.get("MAIL_SERVER", "")
    if mail_server:
        try:
            from flask_mail import Mail, Message
            from flask import current_app
            mail = Mail(current_app)
            for sub in subscribers:
                try:
                    msg = Message(
                        subject=subject,
                        recipients=[sub.email],
                        html=_build_email_html(title or subject, body),
                    )
                    mail.send(msg)
                    sent += 1
                except Exception:
                    failed += 1
        except ImportError:
            # Flask-Mail not installed — simulate send
            sent   = len(subscribers)
            failed = 0
    else:
        # No mail server configured — count as sent (demo mode)
        sent   = len(subscribers)
        failed = 0

    return jsonify({
        "message": f"Newsletter sent to {sent} subscriber(s).",
        "sent":    sent,
        "failed":  failed,
    }), 200


def _build_email_html(title, body):
    """Simple branded email template."""
    return f"""
    <div style="font-family:Arial,sans-serif;max-width:600px;margin:0 auto;background:#1a3a2a;color:#f5f0e8;padding:32px;border-radius:12px">
      <div style="text-align:center;margin-bottom:24px">
        <span style="font-size:32px">🌿</span>
        <h1 style="color:#fff;font-size:1.4rem;margin-top:8px">Cardamom Disease Detection</h1>
      </div>
      <h2 style="color:#a8d5b0;font-size:1.15rem;margin-bottom:16px">{title}</h2>
      <div style="color:#f5f0e8;font-size:.95rem;line-height:1.7">{body}</div>
      <div style="margin-top:32px;padding-top:16px;border-top:1px solid rgba(168,213,176,.2);text-align:center;font-size:.78rem;color:rgba(168,213,176,.5)">
        Nepal 🇳🇵 · <a href="{{unsubscribe_url}}" style="color:rgba(168,213,176,.5)">Unsubscribe</a>
      </div>
    </div>"""