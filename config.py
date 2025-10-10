import os
from datetime import timedelta

class Config:
    SECRET_KEY = os.environ.get('SECRET_KEY') or 'your-secret-key-here'
    SQLALCHEMY_DATABASE_URI = os.environ.get('DATABASE_URL') or 'sqlite:///finance.db'
    SQLALCHEMY_TRACK_MODIFICATIONS = False
    SESSION_COOKIE_SECURE = True
    REMEMBER_COOKIE_DURATION = timedelta(days=1)