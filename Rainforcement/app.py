# ==========================================
# iPhone Purchase Prediction Web App
# ==========================================

# Import libraries
import streamlit as st
import pandas as pd
import numpy as np
import joblib

# ------------------------------------------
# Load the trained model
# ------------------------------------------
# Make sure 'model.joblib' was trained with the same feature names: ['Age', 'Salary']
model = joblib.load('model.joblib')

# ------------------------------------------
# App title and description
# ------------------------------------------
st.title('📱 iPhone Purchase Prediction')
st.write(
    'This machine learning model predicts the likelihood of a customer purchasing '
    'an iPhone based on their **Age** and **Estimated Salary**.'
)

# ------------------------------------------
# Sidebar - User Inputs
# ------------------------------------------
st.sidebar.header('User Input Features')

def user_input():
    """
    Collects user input for Age and Salary from sliders in the sidebar
    and returns a DataFrame with the correct column names.
    """
    age = st.sidebar.slider('Age', 18, 70, 30)
    salary = st.sidebar.slider('Salary', 30000, 150000, 60000)
    
    # Create dataframe with the same structure as training data
    input_df = pd.DataFrame([[age, salary]], columns=['Age', 'Salary'])
    
    # Reorder columns to match the model's training features
    if hasattr(model, "feature_names_in_"):
        input_df = input_df[model.feature_names_in_]
    
    return input_df

# Get user data
df = user_input()

# Display the entered data
st.subheader('User Input')
st.write(df)

# ------------------------------------------
# Make Predictions
# ------------------------------------------
prediction = model.predict(df)            # Returns class label (0 or 1)
prediction_proba = model.predict_proba(df)  # Returns probabilities

# Define label mapping
purchase_labels = ['Not Purchased', 'Purchased']

# ------------------------------------------
# Display Prediction
# ------------------------------------------
st.subheader('Prediction')
st.write(f"**{purchase_labels[int(prediction[0])]}**")

# ------------------------------------------
# Display Prediction Probabilities
# ------------------------------------------
st.subheader('Prediction Probability')
st.write(f'Purchased Probability: {prediction_proba[0][1] * 100:.2f}%')
st.write(f'Not Purchased Probability: {prediction_proba[0][0] * 100:.2f}%')



