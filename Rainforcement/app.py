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