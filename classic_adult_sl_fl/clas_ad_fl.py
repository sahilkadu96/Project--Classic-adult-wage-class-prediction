from flask import Flask, render_template, redirect, session, request
from flask_wtf import FlaskForm
from wtforms import IntegerField, StringField, SubmitField, SelectField
from wtforms.validators import DataRequired
import pickle
import numpy as np
import pandas as pd

app = Flask(__name__)
app.config['SECRET_KEY'] = 'My_secret_key'

with open(r'C:\Users\Sahil\.spyder-py3\classic_adult_sl_fl\clad_xgbc.pkl', 'rb') as f:
    model = pickle.load(f)

columns = ['Federal-gov', 'Local-gov', 'NA wc', 'Never-worked', 'Private',
       'Self-emp-inc', 'Self-emp-not-inc', 'State-gov', 'Without-pay',
       'Divorced', 'Married-AF-spouse', 'Married-civ-spouse',
       'Married-spouse-absent', 'Never-married', 'Separated', 'Widowed',
       'Adm-clerical', 'Armed-Forces', 'Craft-repair', 'Exec-managerial',
       'Farming-fishing', 'Handlers-cleaners', 'Machine-op-inspct', 'NA o',
       'Other-service', 'Priv-house-serv', 'Prof-specialty', 'Protective-serv',
       'Sales', 'Tech-support', 'Transport-moving', 'Husband', 'Not-in-family',
       'Other-relative', 'Own-child', 'Unmarried', 'Wife',
       'Amer-Indian-Eskimo', 'Asian-Pac-Islander', 'Black', 'Other', 'White',
       'Cambodia', 'Canada', 'China', 'Columbia', 'Cuba', 'Dominican-Republic',
       'Ecuador', 'El-Salvador', 'England', 'France', 'Germany', 'Greece',
       'Guatemala', 'Haiti', 'Holand-Netherlands', 'Honduras', 'Hong',
       'Hungary', 'India', 'Iran', 'Ireland', 'Italy', 'Jamaica', 'Japan',
       'Laos', 'Mexico', 'NA nc', 'Nicaragua', 'Outlying-US(Guam-USVI-etc)',
       'Peru', 'Philippines', 'Poland', 'Portugal', 'Puerto-Rico', 'Scotland',
       'South', 'Taiwan', 'Thailand', 'Trinadad&Tobago', 'United-States',
       'Vietnam', 'Yugoslavia', 'age', 'fnlwgt', 'education_num', 'sex',
       'capital_gain', 'capital_loss', 'hours_per_week']


class ClassicAdult(FlaskForm):
    wc = SelectField('Enter working class', choices=['Federal-gov', 'Local-gov', 'NA wc', 'Never-worked', 'Private',
       'Self-emp-inc', 'Self-emp-not-inc', 'State-gov', 'Without-pay'], validators=[DataRequired()])
    ms = SelectField('Enter marital status', choices=['Divorced', 'Married-AF-spouse', 'Married-civ-spouse',
       'Married-spouse-absent', 'Never-married', 'Separated', 'Widowed'], validators=[DataRequired()])
    oc = SelectField('Enter occupation', choices=['Adm-clerical', 'Armed-Forces', 'Craft-repair', 'Exec-managerial',
       'Farming-fishing', 'Handlers-cleaners', 'Machine-op-inspct', 'NA o',
       'Other-service', 'Priv-house-serv', 'Prof-specialty', 'Protective-serv',
       'Sales', 'Tech-support', 'Transport-moving'], validators=[DataRequired()])
    rel = SelectField('Enter relationship', choices=['Husband', 'Not-in-family',
       'Other-relative', 'Own-child', 'Unmarried', 'Wife'], validators=[DataRequired()])
    ra = SelectField('Enter race', choices=['Amer-Indian-Eskimo', 'Asian-Pac-Islander', 'Black', 'Other', 'White'], validators=[DataRequired()])
    co = SelectField('Enter country', choices=['Cambodia', 'Canada', 'China', 'Columbia', 'Cuba', 'Dominican-Republic',
       'Ecuador', 'El-Salvador', 'England', 'France', 'Germany', 'Greece',
       'Guatemala', 'Haiti', 'Holand-Netherlands', 'Honduras', 'Hong',
       'Hungary', 'India', 'Iran', 'Ireland', 'Italy', 'Jamaica', 'Japan',
       'Laos', 'Mexico', 'NA nc', 'Nicaragua', 'Outlying-US(Guam-USVI-etc)',
       'Peru', 'Philippines', 'Poland', 'Portugal', 'Puerto-Rico', 'Scotland',
       'South', 'Taiwan', 'Thailand', 'Trinadad&Tobago', 'United-States',
       'Vietnam', 'Yugoslavia'], validators=[DataRequired()])
    age = IntegerField('Enter age', validators=[DataRequired()])
    fnlwgt = IntegerField('Enter fnlwgt', validators=[DataRequired()])
    education_num = IntegerField('Enter education_num', validators=[DataRequired()])
    sex = IntegerField('Enter sex', validators=[DataRequired()])
    capital_gain = IntegerField('Enter capital_gain', validators=[DataRequired()])
    capital_loss = IntegerField('Enter capital_loss', validators=[DataRequired()])
    hours_per_week = IntegerField('Enter hours_per_week', validators=[DataRequired()])
    submit = SubmitField()


@app.route('/', methods = ['GET', 'POST'])
def index():
    form = ClassicAdult()
    if form.validate_on_submit():
        session['wc'] = form.wc.data
        session['ms'] = form.ms.data
        session['oc'] = form.oc.data
        session['rel'] = form.rel.data
        session['ra'] = form.ra.data
        session['co'] = form.co.data
        session['age'] = form.age.data
        session['fnlwgt'] = form.fnlwgt.data
        session['education_num'] = form.education_num.data
        session['sex'] = form.sex.data
        session['capital_gain'] = form.capital_gain.data
        session['capital_loss'] = form.capital_loss.data
        session['hours_per_week'] = form.hours_per_week.data
        return redirect('predict_wage_class')
    return render_template('home.html', form = form)

@app.route('/predict_wage_class', methods = ['GET', 'POST'])
def predict_wage_class():
    a = np.zeros(91, dtype=int)
    cat_list = [session['wc'], session['ms'], session['oc'], session['rel'],  session['ra'], session['co']]

    for i in range(0, len(columns)):
        for j in cat_list:
            if columns[i] == j:
                 a[i] = 1

    for i in range(0, len(columns)):
        if columns[i] == 'age':
            a[i] = session['age']
        if columns[i] == 'fnlwgt':
            a[i] = session['fnlwgt']
        if columns[i] == 'education_num':
            a[i] = session['education_num']
        if columns[i] == 'sex':
            a[i] = session['sex']
        if columns[i] == 'capital_gain':
            a[i] = session['capital_gain']
        if columns[i] == 'capital_loss':
            a[i] = session['capital_loss']
        if columns[i] == 'hour_per_week':
            a[i] = session['hours_per_week']

    b = np.expand_dims(a, axis=0)
    res = model.predict(b)
    if res[0] == 0:
        wage_class = '<=50K'
    else:
        wage_class = '>50K'
    return render_template('result.html', wage_class = wage_class)


if __name__ == '__main__':
    app.run()