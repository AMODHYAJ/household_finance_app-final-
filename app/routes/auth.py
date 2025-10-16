# app/routes/auth.py (clean version without test routes)
from flask import Blueprint, render_template, redirect, url_for, flash, request
from flask_login import login_user, logout_user, login_required, current_user
from app import db, bcrypt
from app.models.forms import LoginForm, RegistrationForm
from core.database import User, Household
from sqlalchemy.exc import IntegrityError

auth_bp = Blueprint('auth', __name__)

@auth_bp.route('/login', methods=['GET', 'POST'])
def login():
    if current_user.is_authenticated:
        return redirect(url_for('transactions.dashboard'))
    
    form = LoginForm()
    if form.validate_on_submit():
        user = User.query.filter_by(username=form.username.data).first()
        if user and bcrypt.check_password_hash(user.password_hash, form.password.data):
            login_user(user, remember=form.remember.data)
            next_page = request.args.get('next')
            flash('Login successful!', 'success')
            return redirect(next_page) if next_page else redirect(url_for('transactions.dashboard'))
        else:
            flash('Login unsuccessful. Please check username and password', 'danger')
    
    return render_template('login.html', title='Login', form=form)

@auth_bp.route('/register', methods=['GET', 'POST'])
def register():
    if current_user.is_authenticated:
        return redirect(url_for('transactions.dashboard'))
    
    form = RegistrationForm()
    
    # Populate household choices
    households = Household.query.all()
    form.household_id.choices = [(0, 'Create New Household')] + [(h.id, h.name) for h in households]
    
    if form.validate_on_submit():
        hashed_password = bcrypt.generate_password_hash(form.password.data).decode('utf-8')
        
        household = None
        if form.create_household.data and form.household_name.data:
            household = Household(name=form.household_name.data)
            db.session.add(household)
            db.session.flush()
        elif form.household_id.data != 0:
            household = Household.query.get(form.household_id.data)
        
        user = User(
            username=form.username.data,
            password_hash=hashed_password,
            household=household,
            role=form.role.data
        )
        
        try:
            db.session.add(user)
            db.session.commit()
            flash('Your account has been created! You can now log in.', 'success')
            return redirect(url_for('auth.login'))
        except IntegrityError:
            db.session.rollback()
            flash('Username already exists. Please choose a different one.', 'danger')
    
    return render_template('register.html', title='Register', form=form)

@auth_bp.route('/logout')
@login_required
def logout():
    logout_user()
    flash('You have been logged out.', 'info')
    return redirect(url_for('main.home'))  # Changed from auth.login to main.home