from flask_wtf import FlaskForm
from wtforms import StringField, PasswordField, BooleanField, SelectField, \
                   DateField, FloatField, TextAreaField, SubmitField
from wtforms.validators import DataRequired, Length, EqualTo, ValidationError
from app import db
from core.database import User

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
    t_type = SelectField('Type', 
                        choices=[('income', 'Income'), ('expense', 'Expense')],
                        validators=[DataRequired()])
    category = StringField('Category', validators=[DataRequired()])
    amount = FloatField('Amount', validators=[DataRequired()])
    date = DateField('Date', validators=[DataRequired()])
    note = TextAreaField('Note')
    submit = SubmitField('Add Transaction')