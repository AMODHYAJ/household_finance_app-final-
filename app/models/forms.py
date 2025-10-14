from flask_wtf import FlaskForm
from wtforms import StringField, PasswordField, BooleanField, SelectField, \
                   DateField, FloatField, TextAreaField, SubmitField
from wtforms.validators import DataRequired, Length, EqualTo, ValidationError, Optional, NumberRange
from app import db
from core.database import User
from flask_wtf.file import FileField, FileAllowed
from utils.category_standardizer import category_standardizer

class LoginForm(FlaskForm):
    username = StringField('Username', validators=[DataRequired()])
    password = PasswordField('Password', validators=[DataRequired()])
    remember = BooleanField('Remember Me')
    submit = SubmitField('Login')

class RegistrationForm(FlaskForm):
    username = StringField('Username', 
                          validators=[DataRequired(), Length(min=3, max=20)])
    password = PasswordField('Password', 
                            validators=[DataRequired(), Length(min=6)])
    confirm_password = PasswordField('Confirm Password',
                                    validators=[DataRequired(), EqualTo('password')])
    role = SelectField('Role', choices=[('user', 'User'), ('admin', 'Admin')], default='user')
    create_household = BooleanField('Create New Household')
    household_name = StringField('Household Name')
    household_id = SelectField('Select Household', coerce=int, default=0)
    submit = SubmitField('Register')
    
    def validate_username(self, username):
        user = User.query.filter_by(username=username.data).first()
        if user:
            raise ValidationError('That username is taken. Please choose a different one.')

class TransactionForm(FlaskForm):
    type = SelectField('Type', 
                      choices=[('income', 'Income'), ('expense', 'Expense')],
                      validators=[DataRequired()])
    
    category = SelectField('Category', 
                          choices=[],  # Will be populated dynamically
                          validators=[Optional()])
    
    amount = FloatField('Amount', validators=[DataRequired(), NumberRange(min=0.01)])
    date = DateField('Date', validators=[DataRequired()])
    note = TextAreaField('Note', validators=[Optional()])
    receipt = FileField('Upload Receipt', validators=[
        FileAllowed(['jpg', 'jpeg', 'png', 'pdf'], 'Images or PDFs only!')
    ])
    submit = SubmitField('Add Transaction')
    
    def __init__(self, *args, **kwargs):
        super(TransactionForm, self).__init__(*args, **kwargs)
        # Populate with standardized categories
        self.category.choices = [('', 'Auto-categorize')] + [
            (cat, cat) for cat in category_standardizer.get_standard_categories()
        ]

    def validate_amount(self, amount):
        if amount.data <= 0:
            raise ValidationError('Amount must be greater than 0.')
        if amount.data > 1000000:
            raise ValidationError('Amount seems unusually high. Please verify.')

class SearchForm(FlaskForm):
    query = StringField('Search', validators=[Optional()])
    category = SelectField('Category', choices=[], validators=[Optional()])
    type = SelectField('Type',  # FIXED: t_type → type
                      choices=[('', 'All'), ('income', 'Income'), ('expense', 'Expense')],
                      validators=[Optional()])
    date_from = DateField('From Date', validators=[Optional()])
    date_to = DateField('To Date', validators=[Optional()])
    submit = SubmitField('Search')