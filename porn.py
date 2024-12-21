import pandas as pd
import streamlit as st
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
url='https://www.sexvid.pro/pornstars/'
# Load the CSV file
df = pd.read_csv('pornstars_0.csv')

# Create a selectbox for names
a = st.selectbox('Name', df['0'].unique())

# Get the image path for the selected name
image_path = df[df['0'] == a]['5'].values[0]  # Use .values[0] to get the first element
image_path1 = f'https://www.sexvid.pro/pornstars/{a}/'  # Use .values[0] to get the first element

# Display the image
st.image(image_path, caption=f'Image of {a}', use_container_width=True)
st.image(image_path1, caption=f'Image of {a}', use_container_width=True)
st.video(image_path1, format="mp4", start_time=0, subtitles=None, end_time=, loop=False, autoplay=False, muted=False)
# Display the description
desc = df[df['0'] == a]['8'].values[0]
st.write(desc)

# Prepare the data for model training
x = df[['8']]  # Features (make sure it's a DataFrame)
y = df['0']    # Target variable


