"""
app.py
──────
Flask application factory.

Run:
    python app/app.py
Open: http://127.0.0.1:5000
"""

import os
import sys
from flask import Flask, jsonify, render_template
from datetime import timedelta

from flask_jwt_extended import get_jwt_identity, jwt_required

from webapp.extensions import db, bcrypt, jwt
from webapp.routes.auth import auth_bp
from webapp.routes.predict_routes import predict_bp

BASE_DIR = os.path.dirname(os.path.abspath(__file__))


def create_app():
    app = Flask(__name__, template_folder="templates", static_folder="static")

    # ── Config ────────────────────────────────────────────────────────────────
    app.config["SECRET_KEY"] = os.environ.get(
        "SECRET_KEY", "cardamom-secret-key-change-in-production")
    app.config["JWT_SECRET_KEY"] = os.environ.get(
        "JWT_SECRET_KEY", "jwt-cardamom-secret-change-in-production")
    app.config["JWT_ACCESS_TOKEN_EXPIRES"] = timedelta(days=7)
    app.config["SQLALCHEMY_DATABASE_URI"] = f"sqlite:///{os.path.join(BASE_DIR, '..', 'database', 'cardamom.db')}"
    app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False
    app.config["MAX_CONTENT_LENGTH"] = 10 * 1024 * 1024   # 10 MB

    # ── Extensions ────────────────────────────────────────────────────────────
    db.init_app(app)
    bcrypt.init_app(app)
    jwt.init_app(app)

    # ── Blueprints (API routes) ───────────────────────────────────────────────
    app.register_blueprint(auth_bp)
    app.register_blueprint(predict_bp)
    from webapp.routes.admin   import admin_bp
    from webapp.routes.weather import weather_bp

    app.register_blueprint(admin_bp)
    app.register_blueprint(weather_bp)

    # ── Web page route ────────────────────────────────────────────────────────
    @app.route("/")
    def index():
        return render_template("index.html")
    
    @app.route('/contact')
    def contact():
        return render_template("contact.html")
    
    @app.route('/about')
    def about():
        return render_template("about.html")
    
    @app.route("/blog")
    def blog():
        return render_template("blog.html")
    
    @app.route("/app")
    def app_page():
        return render_template("app.html")
    
    @app.route("/login")
    def login():
        return render_template("login.html")
    
    @app.route("/register")
    def register():
        return render_template("register.html")

    @app.route("/api/test-token", methods=["GET"])
    @jwt_required()
    def test_token():
            user_id = get_jwt_identity()
            return jsonify({"user_id": user_id, "message": "Token works!"})
        
    @app.route("/api/predict", methods=["POST"])
    def predict():
        return {"message": "working"}
    
    @app.route("/admin")
    def admin_page():
        return render_template("admin.html")

    # ── Create DB tables ──────────────────────────────────────────────────────
    with app.app_context():
        os.makedirs(os.path.join(BASE_DIR, "..", "database"), exist_ok=True)
        db.create_all()
        print("✅ Database tables ready")

    return app


if __name__ == "__main__":
    app = create_app()
    print("\nCardamom Disease Prediction System")
    print("   Web:  http://127.0.0.1:5000")
    print("   API:  http://127.0.0.1:5000/api\n")
    app.run(debug=True, host="0.0.0.0", port=5000)