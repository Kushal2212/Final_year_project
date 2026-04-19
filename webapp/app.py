# import os
# from flask import Flask, jsonify, render_template, Blueprint, request
# from datetime import timedelta

# from flask_jwt_extended import get_jwt_identity, jwt_required

# from webapp.extensions import db, bcrypt, jwt
# from webapp.routes.auth import auth_bp
# from webapp.routes.predict_routes import predict_bp
# from webapp.routes.contact_routes import contact_bp
# from webapp.sms_alert_system import sms_bp, start_scheduler

# BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# # ─────────────────────────────────────────────────────────────
# # ✅ Newsletter Blueprint (DEFINE FIRST)
# # ─────────────────────────────────────────────────────────────
# newsletter_bp = Blueprint("newsletter", __name__, url_prefix="/api/newsletter")


# @newsletter_bp.route("/subscribers")
# def get_subscribers():
#     page = request.args.get("page", 1)
#     limit = request.args.get("limit", 10)

#     return jsonify({
#         "success": True,
#         "subscribers": [],   # ✅ FIXED
#         "pages": 1           # ✅ frontend needs this
#     })


# @newsletter_bp.route("/send", methods=["POST"])
# def send_newsletter():
#     data = request.get_json()

#     if not data:
#         return jsonify({"error": "No data provided"}), 400

#     subject = data.get("subject")
#     message = data.get("message")

#     if not subject or not message:
#         return jsonify({"error": "Missing subject or message"}), 400

#     return jsonify({
#         "success": True,
#         "message": "Newsletter sent (dummy)"
#     })


# # ─────────────────────────────────────────────────────────────
# # ✅ App Factory
# # ─────────────────────────────────────────────────────────────
# def create_app():
#     app = Flask(__name__, template_folder="templates", static_folder="static")

#     app.config["SECRET_KEY"] = os.environ.get(
#         "SECRET_KEY", "cardamom-secret-key-change-in-production")
#     app.config["JWT_SECRET_KEY"] = os.environ.get(
#         "JWT_SECRET_KEY", "jwt-cardamom-secret-change-in-production")
#     app.config["JWT_ACCESS_TOKEN_EXPIRES"] = timedelta(days=7)
#     app.config["SQLALCHEMY_DATABASE_URI"] = f"sqlite:///{os.path.join(BASE_DIR, '..', 'database', 'cardamom.db')}"
#     app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False
#     app.config["MAX_CONTENT_LENGTH"] = 10 * 1024 * 1024

#     # Extensions
#     db.init_app(app)
#     bcrypt.init_app(app)
#     jwt.init_app(app)

#     # Blueprints
#     app.register_blueprint(auth_bp)
#     app.register_blueprint(predict_bp)
#     app.register_blueprint(contact_bp)

#     from webapp.routes.admin import admin_bp
#     from webapp.routes.weather import weather_bp

#     app.register_blueprint(admin_bp)
#     app.register_blueprint(weather_bp)
#     app.register_blueprint(sms_bp)
#     app.register_blueprint(newsletter_bp)   # ✅ NOW WORKS

#     start_scheduler(app)

#     # Routes
#     @app.route("/")
#     def index():
#         return render_template("index.html")

#     @app.route("/contact")
#     def contact():
#         return render_template("contact.html")

#     @app.route("/about")
#     def about():
#         return render_template("about.html")

#     @app.route("/blog")
#     def blog():
#         return render_template("blog.html")

#     @app.route("/app")
#     def app_page():
#         return render_template("app.html")

#     @app.route("/login")
#     def login():
#         return render_template("login.html")

#     @app.route("/register")
#     def register():
#         return render_template("register.html")

#     @app.route("/api/test-token", methods=["GET"])
#     @jwt_required()
#     def test_token():
#         user_id = get_jwt_identity()
#         return jsonify({"user_id": user_id, "message": "Token works!"})

#     @app.route("/api/predict", methods=["POST"])
#     def predict():
#         return {"message": "working"}

#     @app.route("/admin")
#     def admin_page():
#         return render_template("admin.html")

#     # DB init
#     with app.app_context():
#         os.makedirs(os.path.join(BASE_DIR, "..", "database"), exist_ok=True)
#         db.create_all()
#         print("✅ Database tables ready")

#     return app


# # ─────────────────────────────────────────────────────────────
# # ✅ Run App
# # ─────────────────────────────────────────────────────────────
# if __name__ == "__main__":
#     app = create_app()
#     print("\nCardamom Disease Prediction System")
#     print("   Web:  http://127.0.0.1:5000")
#     print("   API:  http://127.0.0.1:5000/api\n")
#     app.run(debug=True, host="0.0.0.0", port=5000)



import os
from flask import Flask, jsonify, render_template
from datetime import timedelta
from flask_jwt_extended import get_jwt_identity, jwt_required

from webapp.extensions import db, bcrypt, jwt

BASE_DIR = os.path.dirname(os.path.abspath(__file__))


def create_app():
    app = Flask(__name__, template_folder="templates", static_folder="static")

    # ── Config ────────────────────────────────────────────────────────────
    app.config["SECRET_KEY"]                  = os.environ.get("SECRET_KEY", "a8bcf6bc5ffd994ac6dd154b54298185dfa72c8a37f072610ca5bf4c17b89a40")
    app.config["JWT_SECRET_KEY"]              = os.environ.get("JWT_SECRET_KEY", "7e5668b66e65f66f9eab2331962d72e45e8448ebc249a457a9f122d5218c2de5")
    app.config["JWT_ACCESS_TOKEN_EXPIRES"]    = timedelta(days=7)
    app.config["SQLALCHEMY_DATABASE_URI"]     = f"sqlite:///{os.path.join(BASE_DIR, '..', 'database', 'cardamom.db')}"
    app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False
    app.config["MAX_CONTENT_LENGTH"]          = 10 * 1024 * 1024  # 10 MB

    # Optional mail config (for real newsletter sending)
    app.config["MAIL_SERVER"]   = os.environ.get("MAIL_SERVER", "")
    app.config["MAIL_PORT"]     = int(os.environ.get("MAIL_PORT", 587))
    app.config["MAIL_USE_TLS"]  = True
    app.config["MAIL_USERNAME"] = os.environ.get("MAIL_USERNAME", "")
    app.config["MAIL_PASSWORD"] = os.environ.get("MAIL_PASSWORD", "")
    app.config["MAIL_DEFAULT_SENDER"] = os.environ.get("MAIL_USERNAME", "noreply@cardamomdx.com")

    # ── Extensions ────────────────────────────────────────────────────────
    db.init_app(app)
    bcrypt.init_app(app)
    jwt.init_app(app)

    # ── Blueprints ────────────────────────────────────────────────────────
    from webapp.routes.auth            import auth_bp
    from webapp.routes.predict_routes  import predict_bp
    from webapp.routes.admin           import admin_bp
    from webapp.routes.weather         import weather_bp
    from webapp.routes.newsletter_routes import newsletter_bp

    app.register_blueprint(auth_bp)
    app.register_blueprint(predict_bp)
    app.register_blueprint(admin_bp)
    app.register_blueprint(weather_bp)
    app.register_blueprint(newsletter_bp)

    # Optional: contact routes
    try:
        from webapp.routes.contact_routes import contact_bp
        app.register_blueprint(contact_bp)
    except ImportError:
        pass

    # Optional: SMS alert system
    try:
        from webapp.sms_alert_system import sms_bp, start_scheduler
        app.register_blueprint(sms_bp)
        start_scheduler(app)
    except ImportError:
        print("⚠️  SMS system not loaded (apscheduler not installed)")
    except Exception as e:
        print(f"⚠️  SMS system error: {e}")

    # ── Page routes ───────────────────────────────────────────────────────
    @app.route("/")
    def index():      return render_template("index.html")

    @app.route("/contact")
    def contact():    return render_template("contact.html")

    @app.route("/about")
    def about():      return render_template("about.html")

    @app.route("/blog")
    def blog():       return render_template("blog.html")

    @app.route("/app")
    def app_page():   return render_template("app.html")

    @app.route("/login")
    def login():      return render_template("login.html")

    @app.route("/register")
    def register():   return render_template("register.html")

    @app.route("/admin")
    def admin_page(): return render_template("admin.html")

    # ── Test token route ──────────────────────────────────────────────────
    @app.route("/api/test-token", methods=["GET"])
    @jwt_required()
    def test_token():
        user_id = get_jwt_identity()
        return jsonify({"user_id": user_id, "message": "Token works!"})

    # ── Create DB tables ──────────────────────────────────────────────────
    with app.app_context():
        os.makedirs(os.path.join(BASE_DIR, "..", "database"), exist_ok=True)
        db.create_all()
        print("✅ Database tables ready")

    return app


if __name__ == "__main__":
    app = create_app()
    print("\n🌿 Cardamom Disease Detection System")
    print("   Web:  http://127.0.0.1:5000")
    print("   API:  http://127.0.0.1:5000/api\n")
    app.run(debug=True, host="0.0.0.0", port=5000)