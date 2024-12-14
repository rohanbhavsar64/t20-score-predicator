import pandas as pd
import streamlit as st

# Load the CSV file
df = pd.read_csv('pornstars_0.csv')

# Create a selectbox for names
a = st.selectbox('Name', df['0'].unique())

# Get the image path for the selected name
image_path = df[df['0'] == a]['5'].values[0]  # Use .values[0] to get the first element

# Display the image
st.image(image_path, caption=f'Image of {a}', use_container_width=True)
