# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import streamlit as st
import numpy as np
from tensorflow import keras
from PIL import Image

# Load the saved Keras model for Braille alphabet prediction
model = keras.models.load_model('E:/Deo Bolivar/Downloads/EMTECH/Final_Model.h5')

# Function to preprocess the image
def preprocess_image(image):
    # Resize the image to the required input shape of the model
    image = image.resize((28, 28))
    # Convert the image to grayscale
    image = image.convert('L')
    # Convert the image to numpy array
    image_array = np.array(image)
    # Normalize the image array
    normalized_image_array = (image_array.astype(np.float32) / 255.0)
    # Expand dimensions to match the model input shape
    preprocessed_image = np.expand_dims(normalized_image_array, axis=0)
    preprocessed_image = np.expand_dims(preprocessed_image, axis=-1)  # Add channel dimension
    return preprocessed_image

# Mapping from class index to letter
class_to_letter = {
    0: 'A',
    1: 'B',
    2: 'C',
    3: 'D',
    4: 'E',
    5: 'F',
    6: 'G',
    7: 'H',
    8: 'I',
    9: 'J',
    10: 'K',
    11: 'L',
    12: 'M',
    13: 'N',
    14: 'O',
    15: 'P',
    16: 'Q',
    17: 'R',
    18: 'S',
    19: 'T',
    20: 'U',
    21: 'V',
    22: 'W',
    23: 'X',
    24: 'Y',
    25: 'Z'
}

# Streamlit app
def main():
    st.title("Image Classification")
    st.write("Upload an image and the model will predict its letter.")

    # File uploader widget
    uploaded_file = st.file_uploader("Choose an image file", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        # Display the uploaded image
        image = Image.open(uploaded_file)
        st.image(image, caption='Uploaded Image', use_column_width=True)

        # Preprocess the image
        preprocessed_image = preprocess_image(image)

        # Make predictions
        predictions = model.predict(preprocessed_image)
        predicted_class = np.argmax(predictions)
        predicted_letter = class_to_letter.get(predicted_class, 'Unknown')

        # Display the predicted letter
        st.write(f"Predicted Letter: {predicted_letter}")

if __name__ == '__main__':
    main()


