#Import libraries
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

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

    return pd.DataFrame([[age, salary]], columns=['Age', 'Salary'])

df = user_input()

#Display user input
st.write('User Input:')
st.write(df)

# Make prediction
prediction = model.predict(df)
prediction_proba = model.predict_proba(df)

st.subheader('Predictions')
purchase = np.array([['Not Purchased','Purchased']])
st.write(purchase[prediction[0]])

st.subheader('Prediction Probability')
st.write(f'Purchased Probability: {prediction_proba[0][1]*100:.2f}')
st.write(f'Not Purchased Probability: {prediction_proba[0][0]*100:.2f}')