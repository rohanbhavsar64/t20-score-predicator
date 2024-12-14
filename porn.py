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
x=df['8']
y=df['0']
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=1)
model = keras.models.Sequential()
model.add(keras.layers.Flatten(input_shape=[28,28]))
model.add(keras.layers.Dense(300, activation = "relu"))
model.add(keras.layers.Dense(100, activation = "relu"))
model.add(keras.layers.Dense(10, activation = "softmax"))
history = model.fit(X_train, y_train, epochs=30, batch_size=32)
import pandas as pd

pd.DataFrame(history.history).plot(figsize=(15,8))
plt.grid(True)
plt.gca().set_ylim(0,1)
st.write(plt.show())




