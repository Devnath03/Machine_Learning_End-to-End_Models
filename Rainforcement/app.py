#Import libraries
import streamlit as st
import pandas as pd
import numpy as np
import joblib

#Load the imported model
model = joblib.load('model.joblib')

#Add title
st.title('I-Phone Purchase Prediction')
#Add description
st.write('This is a machine learning model that predicts the likelihood of a customer purchasing an iPhone based on various features.')

#Create Sidebar
st.sidebar.header('User Input Features')

#Create function to get inputs
def user_input():
    age = st.sidebar.slider('Age', 18, 70, 30)
    salary = st.sidebar.slider('Salary', 30000, 150000, 60000)

    return pd.DataFrame({
        'age': [age],
        'salary': [salary]
    })

user_input()