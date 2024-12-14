import pandas as pd
import streamlit as st

# Load the CSV file
df = pd.read_csv('pornstars_0.csv')

# Assuming the image path is in the column named '5' and the first row
image_path = df['5'][5]

# Display the image with the updated parameter
st.image(image_path, caption='Image from DataFrame', use_container_width=True)
