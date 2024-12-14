import pandas as pd
import streamlit as st
import tensorflow as tf
from tensorflow import keras

# Load the CSV file
df = pd.read_csv('pornstars_0.csv')

# Create a selectbox for names
a = st.selectbox('Name', df['0'].unique())

# Get the image path for the selected name
image_path = df[df['0'] == a]['5'].values[0]  # Use .values[0] to get the first element
#image_path1 = df[df['0'] == a]['1'].values[0]  # Use .values[0] to get the first element


# Display the image
st.image(image_path, caption=f'Image of {a}', use_container_width=True)

desc=df[df['0'] == a]['8'].values[0]
st.write(desc)

# Build the model
model = keras.models.Sequential()
model.add(keras.layers.Flatten(input_shape=[28, 28]))
model.add(keras.layers.Dense(300, activation="relu"))
model.add(keras.layers.Dense(100, activation="relu"))
model.add(keras.layers.Dense(10, activation="softmax"))

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train the modelfrom sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import r2_score,mean_absolute_error
x=df['8']
y=df['0']
encoder = OneHotEncoder()
y = encoder.fit_transform(df[['0']])
x = encoder.fit_transform(df[['8']])
# Split the dataset into training and testing sets
X_train, X_val, y_train, y_val = train_test_split(x, y, test_size=0.2, random_state=1)


# Evaluate the model on the test set
test_loss, test_accuracy = model.evaluate(X_test.astype('float32') / 255.0, encoder.transform(y_test.reshape(-1, 1)))
print(f'Test accuracy: {test_accuracy:.4f}')
import pandas as pd

pd.DataFrame(history.history).plot(figsize=(15,8))
plt.grid(True)
plt.gca().set_ylim(0,1)
st.write(plt.show())




