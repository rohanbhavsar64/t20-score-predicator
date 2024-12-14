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

# One-hot encode the target variable
encoder = OneHotEncoder(sparse=False)  # Use sparse=False to get a dense array
y_encoded = encoder.fit_transform(y.values.reshape(-1, 1))  # Reshape y to 2D

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(x, y_encoded, test_size=0.2, random_state=1)

# Build the model
model = keras.models.Sequential()
model.add(keras.layers.Flatten(input_shape=[1]))  # Input shape should match the number of features
model.add(keras.layers.Dense(300, activation="relu"))
model.add(keras.layers.Dense(100, activation="relu"))
model.add(keras.layers.Dense(y_encoded.shape[1], activation="softmax"))  # Output layer size matches number of classes

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model (this should ideally be done outside of the Streamlit app)
history = model.fit(X_train, y_train, epochs=30, validation_split=0.2)  # Added validation split for monitoring

# Evaluate the model on the test set
test_loss, test_accuracy = model.evaluate(X_test, y_test)
st.write(f'Test accuracy: {test_accuracy:.4f}')

# Plot training history
plt.figure(figsize=(15, 8))
pd.DataFrame(history.history).plot()
plt.grid(True)
plt.gca().set_ylim(0, 1)
st.pyplot(plt)  # Use Streamlit's function to display the plot




