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


# Display the image
st.image(image_path, caption=f'Image of {a}', use_container_width=True)
desc = df[df['0'] == a]['8'].values[0]
st.write(desc)

# Prepare the data for model training
x = df[['8']]  # Features (make sure it's a DataFrame)
y = df['0']    # Target variable

# Title of the app
from bs4 import BeautifulSoup
import requests 
url='https://www.sexvid.pro/pornstars/mia-khalifa/'
r = requests.get(url)
b=BeautifulSoup(r.text,'html')
# Video URL
from bs4 import BeautifulSoup

# Sample HTML snippet (replace this with your actual HTML content)
html_content =str(b.find_all(class_='thumbs')[0])

# Function to scrape video URLs
def scrape_video_urls(html):
    if not isinstance(html, str):
        raise TypeError("Expected a string for HTML content.")
    
    soup = BeautifulSoup(html, 'html.parser')
    video_urls = []

    # Find all anchor tags with the data-preview attribute
    for a_tag in soup.find_all('a', attrs={'data-preview': True}):
        video_url = a_tag['data-preview']
        video_urls.append(video_url)

    return video_urls

# Parse the HTML content
soup = BeautifulSoup(html_content,'html.parser')

# Find the specific section with class 'thumbs'
thumbs_section = soup.find(class_='thumbs')

# Replace html_content with the string representation of the thumbs section
html_content = str(thumbs_section)

# Scrape the video URLs from the new html_content
video_urls = scrape_video_urls(html_content)

# Print the extracted video URLs
for url in video_urls:
    print(url)
video_url = "https://pr1.sexvid.pro/contents/videos/9000/9400/9400_short_preview.mp4"  # Replace with actual video URL

# Display the video
st.video(video_url, format="mp4", start_time=0,subtitles=None, end_time='358s', loop=False, autoplay=False, muted=False)
