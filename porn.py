import pandas as pd
import streamlit as st

# Load the CSV file
df = pd.read_csv('pornstars_0.csv')

# Assuming the image path is in the column named '5' and the first row
image_path = df['5'][0]

# Display the image
st.image(image_path, caption='Image from DataFrame', use_column_width=True)
