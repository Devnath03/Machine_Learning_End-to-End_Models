#Import Libraries
import streamlit as st
import pickle
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse import hstack, csc_matrix

@st.cache_data()
def load_model():
    df = pd.read_pickle("Book_Recommend_Dataset.pickle")

#Save the Model
    model = pickle.load(open('knn_model.pickle', 'rb'))
    tfidf = pickle.load(open('tfidf.pickle', 'rb'))
    scaler = pickle.load(open('scaler.pickle', 'rb'))
    return df, model, tfidf, scaler

#Call the Function
df, model, tfidf, scaler = load_model()

book_title =  df['title'].values
title_to_idx = {title: idx for idx, title in enumerate (book_title)}

st.title("Book Recommendation System")

book_title = st.selectbox("Select a book", book_title)

if book_title:
    idx =  title_to_idx[book_title]

text_feature = df.iloc[idx]['text']

tfidf_feature  = tfidf.transform([text_feature])
year = scaler.transform([[df.iloc[idx]['year']]])
combined = hstack([tfidf_feature, csc_matrix(year)])