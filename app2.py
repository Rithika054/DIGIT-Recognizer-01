import os 
import random
import streamlit as st
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from keras.models import load_model
from sklearn.preprocessing import LabelBinarizer
from keras.preprocessing.image import img_to_array, load_img

# Load your trained model
model = load_model('model.h5')

# Streamlit app
st.title("Digit Recognition")

# File upload
st.write("Upload an image of a digit:")
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

# Function to preprocess and predict an uploaded image
def predict_image(image):
    image = img_to_array(image)
    image = np.expand_dims(image, axis=0)
    image = image / 255.0  # Normalize
    prediction = model.predict(image)
    label = np.argmax(prediction) + 1  # Add 1 to match the label encoding
    return label

if uploaded_file is not None:
    # Display the uploaded image
    image = load_img(uploaded_file, target_size=(28, 28), color_mode="grayscale")
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Make a prediction
    label = predict_image(image)
    st.write(f"Predicted Label: {label}")

# Add more content to your Streamlit app, e.g., explanations, charts, etc.
st.write("This Streamlit app allows you to predict digit  using a trained model.")
