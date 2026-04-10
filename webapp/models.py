from datetime import datetime
from webapp.extensions import db


class User(db.Model):
    __tablename__ = "users"

    id         = db.Column(db.Integer, primary_key=True)
    name       = db.Column(db.String(100), nullable=False)
    email      = db.Column(db.String(150), unique=True, nullable=False)
    password   = db.Column(db.String(256), nullable=False)   # bcrypt hash
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

    # One user → many predictions
    predictions = db.relationship("Prediction", backref="user", lazy=True)

    def to_dict(self):
        return {
            "id":         self.id,
            "name":       self.name,
            "email":      self.email,
            "created_at": self.created_at.isoformat(),
        }


class Prediction(db.Model):
    __tablename__ = "predictions"

    id             = db.Column(db.Integer, primary_key=True)
    user_id        = db.Column(db.Integer, db.ForeignKey("users.id"), nullable=False)
    disease        = db.Column(db.String(50),  nullable=False)
    confidence     = db.Column(db.Float,       nullable=False)
    severity       = db.Column(db.String(20),  nullable=False)
    nepali_name    = db.Column(db.String(100), nullable=False)
    recommendation = db.Column(db.Text,        nullable=False)
    image_filename = db.Column(db.String(200), nullable=True)
    created_at     = db.Column(db.DateTime,    default=datetime.utcnow)

    def to_dict(self):
        return {
            "id":             self.id,
            "user_id":        self.user_id,
            "disease":        self.disease,
            "confidence":     self.confidence,
            "severity":       self.severity,
            "nepali_name":    self.nepali_name,
            "recommendation": self.recommendation,
            "image_filename": self.image_filename,
            "created_at":     self.created_at.isoformat(),
        }