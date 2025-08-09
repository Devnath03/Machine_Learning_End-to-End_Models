#Import Libraries
import streamlit as st
import pickle
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse import hstack, csc_matrix

@st.cache_data()
def load_model():
    df = pd.read_pickle("Book_Recommend_Dataset.pkl")

#Save the Model
    model = pickle.load(open('knn_model.pickle', 'rb'))
    tfidf = pickle.load(open('tfidf.pickle', 'rb'))
    scaler = pickle.load(open('scaler.pickle', 'rb'))
    return df, model, tfidf, scaler

#Call the Function
df, model, tfidf, scaler = load_model()