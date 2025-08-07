import streamlit as st
import numpy as np
import pickle

# Load the pre-trained model
model = pickle.load(open('model.pickle', 'rb'))

#Page Configuration
st.set_page_config(page_title=" GCE A/L - Z Score Prediction", page_icon="🌸",layout='centered')

# Title of the app
st.title("GCE A/L - Z Score Prediction 🌸" )
st.markdown("This app predicts the Z Score based on the GCE A/L exam results. Please enter your marks below:")

# Input fields for user marks
st.markdown("### Enter your marks for each subject:")

stream_map = {
    'ARTS': 1,
    'BIOLOGICAL SCIENCE': 2,
    'BIOSYSTEMS TECHNOLOGY': 3,
    'COMMERCE': 4,
    'ENGINEERING TECHNOLOGY': 5,
    'NON': 6,
    'PHYSICAL SCIENCE': 7
}

syllabus_map = {
    'new': 0,
    'old': 1
}

gender_map = {
    'Major error': 0,
    'Unknown': 1,
    'female': 2,
    'male': 3
}

grade_map = {
    'A+': 0,
    'Absent': 1,
    'B+': 2,
    'C': 3,
    'F': 4,
    'S': 5,
    'Withheld': 6
}



sub_map = {
    'ACCOUNTING': 0,
    'AGRICULTURAL SCIENCE': 1,
    'AGRO TECHNOLOGY': 2,
    'ART': 3,
    'BIO SYSTEMS TECHNOLOGY': 4,
    'BIO-RESOURCE TECHNOLOGY': 5,
    'BIOLOGY': 6,
    'BUDDHIST CIVILIZATION': 7,
    'BUSINESS STATISTICS': 8,
    'BUSINESS STUDIES': 10,
    'CARNATIC MUSIC': 11,
    'CHEMISTRY': 12,
    'CHRISTIAN CIVILIZATION': 13,
    'CHRISTIANITY': 14,
    'CIVIL TECHNOLOGY': 15,
    'COMBINED MATHEMATICS': 16,
    'COMMUNICATION & MEDIA STUDIES': 17,
    'DANCING(BHARATHA)': 18,
    'DANCING(INDIGENOUS)': 19,
    'DRAMA AND THEATRE (SINHALA)': 20,
    'ECONOMICS': 21,
    'ELECTRICAL,ELECTRONIC AND IT': 22,
    'ENGINEERING TECHNOLOGY': 23,
    'ENGLISH': 24,
    'FOOD TECHNOLOGY': 25,
    'GEOGRAPHY': 26,
    'GREEK & ROMAN CIVILIZATION': 27,
    'HIGHER MATHEMATICS': 28,
    'HINDU CIVILIZATION': 29,
    'HINDUISM': 30,
    'HISTORY OF EUROPE': 31,
    'HISTORY OF INDIA': 32,
    'HISTORY OF MODERN WORLD': 33,
    'HISTORY OF SRI LANKA & EUROPE': 34,
    'HISTORY OF SRI LANKA & INDIA': 35,
    'HISTORY OF SRI LANKA & MODERN WORLD': 36,
    'HOME ECONOMICS': 37,
    'INFORMATION & COMMUNICATION TECHNOLOGY': 38,
    'ISLAM': 39,
    'ISLAMIC CIVILIZATION': 40,
    'LOGIC & SCIENTIFIC METHOD': 41,
    'MATHEMATICS': 42,
    'MECHANICAL TECHNOLOGY': 43,
    'ORIENTAL MUSIC': 44,
    'PALI': 45,
    'PHYSICS': 46,
    'POLITICAL SCIENCE': 47,
    'SINHALA': 48,
    'TAMIL': 49,
    'WESTERN MUSIC': 50
}


with st.form("Prediction Form"):
    stream = st.selectbox("Select your stream:", list(stream_map.keys()))

    sub1 = st.selectbox("Select your first subject:", list(sub_map.keys()))
    sub1_r = st.selectbox("Select your subject 01 grade:", list(grade_map.keys()))

    sub2 = st.selectbox("Select your second subject:", list(sub_map.keys()))
    sub2_r = st.selectbox("Select your subject 02 grade:", list(grade_map.keys()))

    sub3 = st.selectbox("Select your third subject:", list(sub_map.keys()))
    sub3_r = st.selectbox("Select your subject 03 grade:", list(grade_map.keys()))

    syllabus = st.selectbox("Select your syllabus:", list(syllabus_map.keys()))

    gender = st.selectbox("Select your gender:", list(gender_map.keys()))
  

    # Submit button
    submitted = st.form_submit_button("Predict Z Score ")

    if submitted:
        # Prepare the input data for prediction
       inp = np.array([[
            stream_map[stream],
            sub_map[sub1],
            grade_map[sub1_r],
            sub_map[sub2],
            grade_map[sub2_r],
            sub_map[sub3],
            grade_map[sub3_r],
            syllabus_map[syllabus],
            gender_map[gender]
        ]])
try:
    prediction = model.predict(inp)
    st.success(f"Your predicted Z Score is: {prediction[0]:.4f} 🎉")
except Exception as e:
    st.error(f"Error occurred: {e}")
