import numpy as np
import pandas as pd
import pickle
import streamlit as st


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

wc = st.selectbox('Enter working class', ['Federal-gov', 'Local-gov', 'NA wc', 'Never-worked', 'Private',
       'Self-emp-inc', 'Self-emp-not-inc', 'State-gov', 'Without-pay'])

ms = st.selectbox('Enter marital status', ['Divorced', 'Married-AF-spouse', 'Married-civ-spouse',
       'Married-spouse-absent', 'Never-married', 'Separated', 'Widowed'])

oc = st.selectbox('Enter occupation', ['Adm-clerical', 'Armed-Forces', 'Craft-repair', 'Exec-managerial',
       'Farming-fishing', 'Handlers-cleaners', 'Machine-op-inspct', 'NA o',
       'Other-service', 'Priv-house-serv', 'Prof-specialty', 'Protective-serv',
       'Sales', 'Tech-support', 'Transport-moving'])

rel = st.selectbox('Enter relationship', ['Husband', 'Not-in-family',
       'Other-relative', 'Own-child', 'Unmarried', 'Wife'])

ra = st.selectbox('Enter race', ['Amer-Indian-Eskimo', 'Asian-Pac-Islander', 'Black', 'Other', 'White'])

co = st.selectbox('Enter country', ['Cambodia', 'Canada', 'China', 'Columbia', 'Cuba', 'Dominican-Republic',
       'Ecuador', 'El-Salvador', 'England', 'France', 'Germany', 'Greece',
       'Guatemala', 'Haiti', 'Holand-Netherlands', 'Honduras', 'Hong',
       'Hungary', 'India', 'Iran', 'Ireland', 'Italy', 'Jamaica', 'Japan',
       'Laos', 'Mexico', 'NA nc', 'Nicaragua', 'Outlying-US(Guam-USVI-etc)',
       'Peru', 'Philippines', 'Poland', 'Portugal', 'Puerto-Rico', 'Scotland',
       'South', 'Taiwan', 'Thailand', 'Trinadad&Tobago', 'United-States',
       'Vietnam', 'Yugoslavia'])


age = st.number_input('Enter age', min_value=17, max_value=90, step=1)
fnlwgt = st.number_input('Enter fnlwgt', min_value=10000, max_value=1000000, step = 50)
education_num = st.number_input('Enter eductaion number', min_value=1, max_value=16, step=1)
sex = st.number_input('Enter sex', min_value=0, max_value=1, step=1)
capital_gain = st.number_input('Enter captial gain', min_value=0, max_value=99999, step=100)
capital_loss = st.number_input('Enter captial loss', min_value=0, max_value=4500, step=100)
hours_per_week = st.number_input('Enter hours per week', min_value=1, max_value=99)

a = np.zeros(91, dtype=int)

cat_list = [wc, ms, oc, rel, ra, co]
num_list = [age, fnlwgt, education_num, sex, capital_gain, capital_loss, hours_per_week]

for i in range(0, len(columns)):
    for j in cat_list:
        if columns[i] == j:
            a[i] = 1


for i in range(0, len(columns)):
    if columns[i] == 'age':
        a[i] = age
    if columns[i] == 'fnlwgt':
        a[i] = fnlwgt
    if columns[i] == 'education_num':
        a[i] = education_num
    if columns[i] == 'sex':
        a[i] = sex
    if columns[i] == 'capital_gain':
        a[i] = capital_gain
    if columns[i] == 'capital_loss':
        a[i] = capital_loss
    if columns[i] == 'hour_per_week':
        a[i] = hours_per_week

if st.button('Predict'):
    b = np.expand_dims(a, axis=0)
    res = model.predict(b)
    if res[0] == 0:
        wage_class = '<=50K'
    else:
        wage_class = '>50K'
    st.write(f' The predicted wage class is {wage_class}')
