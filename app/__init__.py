# app/__init__.py
from flask import Flask, redirect, url_for
from flask_sqlalchemy import SQLAlchemy
from flask_login import LoginManager
from flask_bcrypt import Bcrypt
import os

# Initialize extensions
db = SQLAlchemy()
bcrypt = Bcrypt()
login_manager = LoginManager()
login_manager.login_view = 'auth.login'
login_manager.login_message_category = 'info'

def create_app():
    app = Flask(__name__)
    
    # Configuration
    app.config['SECRET_KEY'] = os.environ.get('SECRET_KEY') or 'dev-secret-key-change-in-production'
    app.config['SQLALCHEMY_DATABASE_URI'] = os.environ.get('DATABASE_URL') or 'sqlite:///finance.db'
    app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
    
    # Initialize extensions with app
    db.init_app(app)
    bcrypt.init_app(app)
    login_manager.init_app(app)
    
    # Import models here to avoid circular imports
    from core.database import User, Transaction, Household
    
    @login_manager.user_loader
    def load_user(user_id):
        return User.query.get(int(user_id))
    
    # Import and register blueprints
    from app.routes.auth import auth_bp
    from app.routes.transactions import transactions_bp
    from app.routes.charts import charts_bp
    from app.routes.insights import insights_bp
    from app.routes.budget import budget_bp
    from app.routes.export import export_bp
    from app.routes.mcp_api import mcp_bp
    from app.routes.main import main_bp

    app.register_blueprint(main_bp)  # Register main blueprint first
    app.register_blueprint(auth_bp)
    app.register_blueprint(transactions_bp)
    app.register_blueprint(charts_bp)
    app.register_blueprint(insights_bp)
    app.register_blueprint(budget_bp)
    app.register_blueprint(export_bp)
    app.register_blueprint(mcp_bp)
    
    # Define the default route
    @app.route('/')
    def index():
        return redirect(url_for('main.home'))
    
    # Create tables
    with app.app_context():
        db.create_all()
        
        # Initialize simplified agents
        try:
            from agents.architect_agent import ArchitectAgent
            app.architect_agent = ArchitectAgent()
        except Exception as e:
            print(f"Warning: Could not initialize agents: {e}")
            # Create a simple mock agent
            class MockArchitectAgent:
                def __init__(self):
                    from agents.data_collector import DataCollectorAgent
                    from agents.chart_creator import ChartCreatorAgent
                    from agents.insight_generator import InsightGeneratorAgent
                    self.data_agent = DataCollectorAgent()
                    self.chart_agent = ChartCreatorAgent(self.data_agent)
                    self.insight_agent = InsightGeneratorAgent(self.data_agent)
                
                def trigger_chart_generation(self, user_id):
                    print(f"Mock: Trigger chart generation for user {user_id}")
                
                def trigger_insight_generation(self, user_id):
                    print(f"Mock: Trigger insight generation for user {user_id}")
            
            app.architect_agent = MockArchitectAgent()

    return app