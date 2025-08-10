#Import Libraries
import pandas as pd
import streamlit as st
import joblib
import numpy as np

#Load the trained model
@st.cache_resource
def load_model():
    data = joblib.load('Book_Recommendation_Pipeline.joblib')
    return data['pipeline'],data['data']

pipeline, data = load_model()

title = [
       'Adult Children of Emotionally Immature Parents: How to Heal from Distant, Rejecting, or Self-Involved Parents',
       'From Strength to Strength: Finding Success, Happiness, and Deep Purpose in the Second Half of Life',
       'Good Inside: A Guide to Becoming the Parent You Want to Be',
       "The Seven Principles for Making Marriage Work: A Practical Guide from the Country's Foremost Relationship Expert",
       'The Glass Castle: A Memoir',
       'What Happened to You?: Conversations on Trauma, Resilience, and Healing',
       'Vax-Unvax: Let the Science Speak (Children’s Health Defense)',
       'Happy-Go-Lucky',
       'Habits of the Household: Practicing the Story of God in Everyday Family Rhythms',
       'Anne Of Green Gables Complete 8 Book Set',
       'The Classic Fairy Tales (Second Edition) (Norton Critical Editions)',
       'Touched Out: Motherhood, Misogyny, Consent, and Control',
       'Child Development and Education',
       'Autism Spectrum Disorders from Theory to Practice: Assessment and Intervention Tools Across the Lifespan',
       "Broken Faith: Inside one of America's Most Dangerous Cults",
       'Saturdays at Noon: An uplifting, emotional and unpredictable page-turner to make you smile',
       'Raising Girls Who Like Themselves', 'Tweens',
       'Child, Family, School, Community: Socialization and Support',
       'Attainable Sustainable: The Lost Art of Self-Reliant Living',
       "From the Pilot's Seat: Kiwi Adventurers in the Sky",
       'The Patron Saint of Used Cars and Second Chances: A Memoir',
       "The Business Writer's Companion",
       'Understanding Dental Insurance: A Guide for Dentists and their Teams',
       'Draw and animate your manga characters: The Complete Guide by @ZESENSEI_DRAWS',
       'Parisian Chic - Look Book: What should I wear today ?',
       'Field Guide to Trees of Southern Africa: An African Perspective (Field Guide To... (Struik Publishers))',
       'Ravenor: The Omnibus',
       'Ghettoside: A True Story of Murder in America',
       'Accidental Archaeologists: True Stories of Unexpected Discoveries',
       'Reign of the Seven Spellblades, Vol. 9 (light novel) (Reign of the Seven Spellblades (novel))',
       "What's Best Next: How the Gospel Transforms the Way You Get Things Done",
       'Relational Aesthetics',
       'Harry Potter and the Cursed Child: The Journey: Behind the Scenes of the Award-Winning Stage Production',
]

st.title("Book Recommendation System")

selected_title = st.selectbox("Select a book title", title)

if st.button("Recommend"):
    # Get recommendations
    recommendations = pipeline.predict([selected_title])
    st.write("Recommended books:")
    for rec in recommendations:
        st.write(f"- {rec}")

