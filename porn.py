import pandas as pd
import streamlit as st

# Load the CSV file
df = pd.read_csv('pornstars_0.csv')
a=st.selectbox('Name',df['0'].unique())
# Assuming the image path is in the column named '5' and the first row
image_path = df[df['0']==a]['5'].unique()
# Display the image with the updated parameter
#st.image(image_path, caption='Image from DataFrame', use_container_width=True)
st.write(image_path)
