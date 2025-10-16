from flask import Blueprint, render_template

main_bp = Blueprint('main', __name__)

@main_bp.route('/')
@main_bp.route('/home')
def home():
    """Home page route"""
    return render_template('home.html', title='Home - Household Finance AI')

@main_bp.route('/about')
def about():
    """About page route"""
    return render_template('about.html', title='About Us')

@main_bp.route('/features')
def features():
    """Features page route"""
    return render_template('features.html', title='Features')