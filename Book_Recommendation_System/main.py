# =========================
# Import Required Libraries
# =========================
import streamlit as st  # Streamlit for creating the web app interface
import pickle           # For loading pre-trained models and objects
import pandas as pd     # For handling data
from sklearn.feature_extraction.text import TfidfVectorizer  # For text feature extraction
from scipy.sparse import hstack, csc_matrix                  # For combining sparse matrices

# =========================
# Function to Load Data and Models
# =========================
@st.cache_data()  # Cache the function output so it's not reloaded every time the app refreshes
def load_model():
    # Load the dataset containing book information
    df = pd.read_pickle("Book_Recommend_Dataset.pickle")

    # Load the trained KNN model
    model = pickle.load(open('knn_model.pickle', 'rb'))
    
    # Load the pre-trained TF-IDF vectorizer
    tfidf = pickle.load(open('tfidf.pickle', 'rb'))
    
    # Load the pre-fitted scaler for numerical features
    scaler = pickle.load(open('scaler.pickle', 'rb'))
    
    return df, model, tfidf, scaler

# =========================
# Load the Data and Models
# =========================
df, model, tfidf, scaler = load_model()

# Extract book titles from the dataset
book_title = df['title'].values

# Create a dictionary mapping book titles to their index positions
title_to_idx = {title: idx for idx, title in enumerate(book_title)}

# =========================
# Streamlit UI
# =========================
st.title("📚 Book Recommendation System")  # Page title

# Dropdown menu to select a book
book_title = st.selectbox("Select a book", book_title)

# =========================
# Generate Recommendations
# =========================
if book_title:
    # Find the index of the selected book
    idx = title_to_idx[book_title]

    # Extract the text description of the selected book
    text_feature = df.iloc[idx]['text']

    # Transform the text into TF-IDF vector format
    tfidf_feature = tfidf.transform([text_feature])

    # Extract and scale the publication year feature
    year = scaler.transform([[df.iloc[idx]['year']]])

    # Combine text features and numerical year feature into a single sparse matrix
    combined = hstack([tfidf_feature, csc_matrix(year)])

    # Use the trained KNN model to find nearest neighbors (similar books)
    distances, indices = model.kneighbors(combined)

    # =========================
    # Display Recommendations
    # =========================
    st.header("📖 Recommended Books")
    recommendations = []

    # Create a list of (rank, title, distance) tuples
    for i, (dist, idx) in enumerate(zip(distances[0], indices[0])):
        recommendations.append((i + 1, df.iloc[idx]['title'], dist))

    # Display each recommended book
    for rank, title, dist in recommendations:
        st.write(f"{rank}. {title} (Similarity Distance: {dist:.4f})")
