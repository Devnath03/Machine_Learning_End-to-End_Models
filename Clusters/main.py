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