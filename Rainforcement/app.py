#Import libraries
import streamlit as st
import pandas as pd
import numpy as np
import joblib


model = joblib.load('model.joblib')
