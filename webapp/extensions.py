from flask_sqlalchemy import SQLAlchemy
from flask_bcrypt import Bcrypt
from flask_jwt_extended import JWTManager
from apscheduler.schedulers.background import BackgroundScheduler
 
db     = SQLAlchemy()
bcrypt = Bcrypt()
jwt    = JWTManager()