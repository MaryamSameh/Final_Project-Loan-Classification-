
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import streamlit as st
import joblib


def getInput():
    
    person_age = st.slider('what is your age'.title() ,  min_value=20, max_value=70, step=1)
    person_gender =  st.selectbox('select gender'.title() , ['male', 'female'])
    person_education = st.selectbox('What type of your education'.title(),['Master' ,'High School' ,'Bachelor' ,'Associate' ,'Doctorate'])
    person_income =  st.slider('what is your income'.title() , min_value=8000.000 , max_value=2448661.0 , step=1000.0)
    person_emp_exp = st.slider('what is your years of Exp.'.title() , min_value=0 , max_value=50 , step=1)
    loan_amnt = st.slider('what is your Loan Amount'.title() , min_value=500.0 , max_value=35000.0 , step=100.0)
    person_home_ownership = st.selectbox('what is your home ownership'.title() ,['RENT' ,'OWN', 'MORTGAGE' ,'OTHER'])
    loan_intent = st.selectbox('what is your intention'.title() , ['PERSONAL' ,'EDUCATION' ,'MEDICAL' ,'VENTURE', 'HOMEIMPROVEMENT',
                                                                     'DEBTCONSOLIDATION'])
    loan_int_rate = st.slider('what is the rate of interest'.title(),min_value=5.42 , max_value=20.0 , step=0.1)
    loan_percent_income = st.slider('what is the percentile from income'.title(),min_value=0.0,max_value=0.62 , step=0.01)
    cb_person_cred_hist_length = st.slider('what is your Credit history years'.title(),min_value=2.0 ,max_value=30.0 , step=1.0)
    credit_score = st.slider('what is your Credit Score'.title(), min_value=418 ,max_value=772 , step=5)
    previous_loan_defaults_on_file = st.selectbox('Previous Loan Accepted or Denied'.title() , [0 ,1])


    return pd.DataFrame(
        data=[ 
            [person_age, person_gender, person_education, person_income, person_emp_exp,loan_amnt,
       person_home_ownership, loan_intent,loan_int_rate,
       loan_percent_income, cb_person_cred_hist_length, credit_score,previous_loan_defaults_on_file]
        ] , 
                 columns=['person_age', 'person_gender', 'person_education', 'person_income', 'person_emp_exp',
       'loan_amnt', 'person_home_ownership', 'loan_intent',
       'loan_int_rate', 'loan_percent_income', 'cb_person_cred_hist_length' , 'credit_score' , 'previous_loan_defaults_on_file'])

test = getInput()
st.dataframe(test)
model = joblib.load('model.h5')

st.write('Accepted' if model.predict(test) == 1 else 'Not Accepted')
