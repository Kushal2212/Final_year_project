import email
from xml.parsers.expat import errors

from flask import Blueprint, request, jsonify
from flask_jwt_extended import create_access_token, jwt_required, get_jwt_identity

from webapp.extensions import db, bcrypt
from webapp.models import User
import validators
from email_validator import validate_email, EmailNotValidError
from password_strength import PasswordPolicy


auth_bp = Blueprint("auth", __name__, url_prefix="/api/auth")

policy = PasswordPolicy.from_names(
    length=8,
    uppercase=1,
    numbers=1,
    special=1
)


# ── Register ──────────────────────────────────────────────────────────────────
@auth_bp.route("/register", methods=["POST"])
def register():
    data = request.get_json()
    if not data:
        return jsonify({"error": "No data provided"}), 400

    name     = data.get("name", "").strip()
    def validate_name(name):
        if len(name) < 3:
            return False
        if not name.replace(" ", "").isalpha():
            return False
        return True
    
    
    password = data.get("password", "")

    def validate_password(password):
        errors = policy.test(password)
        return len(errors) == 0
    
    
    
    email    = data.get("email", "").strip().lower()
    def check_email(email):
        try:
            validate_email(email)
            return True
        except EmailNotValidError:
            return False
    
    # Validation
    # Name validation
    if not validate_name(name):
        return jsonify({"error": "Name must contain only letters and be at least 3 characters"}), 400

# Email validation
    if not check_email(email):
        return jsonify({"error": "Invalid email address"}), 400

# Password validation
    if not validate_password(password):
        return jsonify({
        "error": "Password must contain 8 characters, uppercase letter, number and special character"
    }), 400

    hashed = bcrypt.generate_password_hash(password).decode("utf-8")
    user   = User(name=name, email=email, password=hashed)
    db.session.add(user)
    db.session.commit()

    token = create_access_token(identity=str(user.id))
    return jsonify({
        "message": "Account created successfully",
        "token":   token,
        "user":    user.to_dict(),
    }), 201


# ── Login ─────────────────────────────────────────────────────────────────────
@auth_bp.route("/login", methods=["POST"])
def login():
    data = request.get_json()
    if not data:
        return jsonify({"error": "No data provided"}), 400

    email    = data.get("email", "").strip().lower()
    password = data.get("password", "")

    user = User.query.filter_by(email=email).first()
    if not user or not bcrypt.check_password_hash(user.password, password):
        return jsonify({"error": "Invalid email or password"}), 401

    token = create_access_token(identity=str(user.id))
    return jsonify({
        "message": "Login successful",
        "token":   token,
        "user":    user.to_dict(),
    }), 200


# ── Profile ───────────────────────────────────────────────────────────────────
@auth_bp.route("/profile", methods=["GET"])
@jwt_required()
def profile():
    user_id = get_jwt_identity()
    user    = User.query.get_or_404(user_id)
    return jsonify({"user": user.to_dict()}), 200


# ── Logout (hint to client to delete token) ───────────────────────────────────
@auth_bp.route("/logout", methods=["POST"])
@jwt_required()
def logout():
    return jsonify({"message": "Logged out. Please delete your token on the client."}), 200