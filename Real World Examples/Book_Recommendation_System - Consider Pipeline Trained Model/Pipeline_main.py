# Import Libraries
import pandas as pd
import streamlit as st
import joblib
import numpy as np

# Load the trained model and data
@st.cache_resource
def load_model():
    data = joblib.load('Book_Recommendation_Pipeline.joblib')
    # The joblib file contains a dict with keys 'pipeline' and 'data'
    return data['pipeline'], data['data']

pipeline, data = load_model()

# List of book titles for selection (you can load this from data too)
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

# Select book title
selected_title = st.selectbox("Select a book title", title)

# Find the index of the selected title in your data
book_index = data[data['title'] == selected_title].index[0]

# DEBUG: show pipeline step names to confirm
# st.write("Pipeline step names:", list(pipeline.named_steps.keys()))

# Transform the book titles to features using the proper pipeline step
# Replace 'tfidf' with the correct transformer step name if different
features = pipeline.named_steps['tfidf'].transform(data['title'])

# Extract the feature vector for the selected book
query_vec = features[book_index]

# If sparse matrix, convert to dense
if hasattr(query_vec, "toarray"):
    query_vec = query_vec.toarray()

query_vec = query_vec.reshape(1, -1)

# Find neighbors using the model step (replace 'knn' if your step name is different)
distances, indices = pipeline.named_steps['knn'].kneighbors(query_vec)

st.subheader("Recommendations:")

# Display the recommended books (skipping the first one, which is the query itself)
for i, (idx, dist) in enumerate(zip(indices[0][1:], distances[0][1:])):
    book = data.iloc[idx]
    st.write(f"{i+1}. {book['title']} (Distance: {dist:.4f})")
