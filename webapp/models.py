from datetime import datetime
from webapp.extensions import db


class User(db.Model):
    __tablename__ = "users"

    id         = db.Column(db.Integer,     primary_key=True)
    name       = db.Column(db.String(100), nullable=False)
    email      = db.Column(db.String(150), unique=True, nullable=False)
    password   = db.Column(db.String(256), nullable=False)
    is_admin   = db.Column(db.Boolean,     default=False, nullable=False,
                           server_default="0")
    created_at = db.Column(db.DateTime,    default=datetime.utcnow)

    predictions = db.relationship("Prediction", backref="user", lazy=True)

    def to_dict(self):
        return {
            "id":         self.id,
            "name":       self.name,
            "email":      self.email,
            "is_admin":   self.is_admin,
            "created_at": self.created_at.isoformat(),
        }


class Prediction(db.Model):
    __tablename__ = "predictions"

    id             = db.Column(db.Integer,     primary_key=True)
    user_id        = db.Column(db.Integer,     db.ForeignKey("users.id"), nullable=False)
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
        
class ContactMessage(db.Model):
    __tablename__ = "contact_messages"

    id         = db.Column(db.Integer,     primary_key=True)
    name       = db.Column(db.String(100), nullable=False)
    email      = db.Column(db.String(150), nullable=False)
    subject    = db.Column(db.String(200), nullable=False)
    message    = db.Column(db.Text,        nullable=False)
    is_read    = db.Column(db.Boolean,     default=False)
    created_at = db.Column(db.DateTime,    default=datetime.utcnow)

    def to_dict(self):
        return {
            "id":         self.id,
            "name":       self.name,
            "email":      self.email,
            "subject":    self.subject,
            "message":    self.message,
            "is_read":    self.is_read,
            "created_at": self.created_at.isoformat(),
        }