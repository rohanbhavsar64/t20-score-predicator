import pandas as pd
import streamlit as st
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder

# Load the CSV file
df = pd.read_csv('pornstars_0.csv')

# Create a selectbox for names
a = st.selectbox('Name', df['0'].unique())

# Get the image path for the selected name
image_path = df[df['0'] == a]['5'].values[0]  # Use .values[0] to get the first element
image_path1 = 'https://www.google.com/search?sca_esv=396d2339e225b75b&rlz=1C1VDKB_enIN1075IN1075&biw=1536&bih=738&sxsrf=ADLYWIJxV5c9EDF88ALrKScnRpmusxsvPw:1734756359891&q=lana+rhoades&udm=7&fbs=AEQNm0Aa4sjWe7Rqy32pFwRj0UkWd8nbOJfsBGGB5IQQO6L3JyJJclJuzBPl12qJyPx7ESIJGrw-8wOtrT6OJapeZNCMiHjzNLCgfihuLPeo-Kz42W_mc-Lo66-CC0XfhB2-gcvS1ImaFbRkFYvxNeH1dZ16HNAN6FvlRDlum52F-GKVg_YBgNCQg4hKfKJmHfrrysdhDhB2&sa=X&ved=2ahUKEwi5sumEh7iKAxWy1wIHHbQ2CI0QtKgLegQIDBAB#fpstate=ive&vld=cid:27e4d7d5,vid:iFSA9dshSJI,st:0'  # Use .values[0] to get the first element

# Display the image
st.image(image_path, caption=f'Image of {a}', use_container_width=True)
st.image(image_path1, caption=f'Image of {a}', use_container_width=True)
st.video(image_path1, format="video/mp4", start_time=0, subtitles=None, end_time=1, loop=False, autoplay=False, muted=False)
# Display the description
desc = df[df['0'] == a]['8'].values[0]
st.write(desc)

# Prepare the data for model training
x = df[['8']]  # Features (make sure it's a DataFrame)
y = df['0']    # Target variable


