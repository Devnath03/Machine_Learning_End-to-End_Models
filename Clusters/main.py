#Import Libraries
import streamlit as st
import pandas as pd
import pickle

#Import Model
with open('model.pickle', 'rb') as file:
    model = pickle.load(file)

with open('scaler.pickle', 'rb') as file:
    scaler = pickle.load(file)

# Streamlit App
st.title('Customer Segmentation')

st.markdown('This app segments customers based on their purchasing behavior.')

age = st.slider('Age', 18, 100, 30)
income = st.slider('Income', 1000, 100000, 50000)
spending = st.slider('Spending Score', 1, 100, 50)
st.write('Input values:', age, income, spending)

# Make prediction
input_data = [[age, income, spending]]
input_scaled = scaler.transform(input_data)
cluster = model.predict(input_scaled)[0]
st.write('Predicted cluster:', cluster)

cluster_label = {
    0:"budget",
    1:"premium",
    2:"Moderate"
}

st.subheader(cluster_label[cluster])
