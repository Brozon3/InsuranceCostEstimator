import decimal

from flask_wtf import FlaskForm
from wtforms import SubmitField
from wtforms.fields import SelectField, IntegerField, DecimalField
from wtforms.validators import DataRequired


class InformationForm(FlaskForm):
    userAge = IntegerField("Age:", validators=[DataRequired()])
    userSex = SelectField("Sex:", choices=["male", "female"], validators=[DataRequired()])
    userBMI = DecimalField("BMI:", places=2, rounding=decimal.ROUND_UP, validators=[DataRequired()])
    userSmoker = SelectField("Smoker:", choices=["yes", "no"], validators=[DataRequired()])

    submit = SubmitField("Submit")
