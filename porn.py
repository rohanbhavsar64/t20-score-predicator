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

# Display the image
st.image(image_path, caption=f'Image of {a}', use_container_width=True)

# Display the description
desc = df[df['0'] == a]['8'].values[0]
st.write(desc)

# Prepare the data for model training
x = df[['8']]  # Features (make sure it's a DataFrame)
y = df['0']    # Target variable
import os
import requests
import pandas as pd
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split

# Load the CSV file
data = pd.read_csv('pornstars_0.csv')

# Prepare the data
data['5'] = data['5'].astype(str)  # Convert to string
data = data[data['5'].notna()]  # Remove rows with NaN in column 5

image_paths = data['5'].tolist()
labels = data['0'].tolist()

# Encode labels
label_to_index = {label: index for index, label in enumerate(set(labels))}
index_to_label = {index: label for label, index in label_to_index.items()}
y = np.array([label_to_index[label] for label in labels])

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(image_paths, y, test_size=0.2, random_state=42)

# Create a directory for downloaded images
download_dir = "images"
os.makedirs(download_dir, exist_ok=True)

# Function to download images
def download_images(image_urls, output_dir):
    local_paths = []
    total_images = len(image_urls)
    
    # Print the progress using simple print statements
    for i, url in enumerate(image_urls):
        try:
            filename = os.path.join(output_dir, os.path.basename(url.split("?")[0]))
            response = requests.get(url, timeout=10)
            response.raise_for_status()
            with open(filename, "wb") as f:
                f.write(response.content)
            local_paths.append(filename)
        except Exception as e:
            print(f"Failed to download {url}: {e}")
            local_paths.append(None)  # Mark as None if download fails
        
        # Print progress after every 10th image (or you can adjust as needed)
        if (i + 1) % 10 == 0 or i == total_images - 1:
            print(f"Downloaded {i + 1}/{total_images} images.")
    
    return local_paths

# Download training and testing images
X_train = download_images(X_train, download_dir)
X_test = download_images(X_test, download_dir)

# Remove failed downloads
train_df = pd.DataFrame({'filename': X_train, 'class': [index_to_label[i] for i in y_train]})
test_df = pd.DataFrame({'filename': X_test, 'class': [index_to_label[i] for i in y_test]})

train_df = train_df[train_df['filename'].notna()]
test_df = test_df[test_df['filename'].notna()]

# Image data generator for preprocessing
train_datagen = ImageDataGenerator(rescale=1./255)
test_datagen = ImageDataGenerator(rescale=1./255)

# Create generators
train_generator = train_datagen.flow_from_dataframe(
    dataframe=train_df,
    x_col='filename',
    y_col='class',
    target_size=(150, 150),
    batch_size=32,
    class_mode='categorical'
)

test_generator = test_datagen.flow_from_dataframe(
    dataframe=test_df,
    x_col='filename',
    y_col='class',
    target_size=(150, 150),
    batch_size=32,
    class_mode='categorical'
)

# Get the number of classes based on the train_generator
num_classes = len(train_generator.class_indices)
print(f"Number of unique classes: {num_classes}")

# Build the model
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(150, 150, 3)),
    MaxPooling2D(pool_size=(2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),
    Flatten(),
    Dense(128, activation='relu'),
    Dense(num_classes, activation='softmax')  # Adjust output layer to match number of classes
])

# Compile the model
model.compile(optimizer=Adam(), loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(train_generator, epochs=10)

# Save the model
model.save('image_detection_model.h5')
