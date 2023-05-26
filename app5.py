# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import streamlit as st
import numpy as np
from tensorflow import keras
import requests
from PIL import Image
from io import BytesIO

# Load the saved Keras model
model = keras.models.load_model('E:/Deo Bolivar/Downloads/EMTECH/Final_Model.h5')

# Function to predict image classification
def predict_image(image):
    image = image.resize((28, 28))  # Resize the image to the input shape expected by the model
    image = np.array(image)  # Convert image to numpy array
    image = image[:, :, 0]  # Keep only the first channel (remove the extra dimension)
    image = image / 255.0  # Normalize the image pixel values
    image = np.expand_dims(image, axis=0)  # Add batch dimension

    # Make predictions using the loaded model
    prediction = model.predict(image)
    predicted_class = np.argmax(prediction)  # Get the index of the predicted class
    return predicted_class

# Streamlit app code
def main():
    st.title("Image Classifier")
    st.write("Upload an image or provide an image URL to get the predicted class.")

    # File upload or URL input
    option = st.radio("Select Input Option", ("Upload Image", "Use Image URL"))

    if option == "Upload Image":
        # File upload
        uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

        if uploaded_file is not None:
            # Display the uploaded image
            image = Image.open(uploaded_file)
            st.image(image, caption="Uploaded Image", use_column_width=True)

            # Make prediction on the uploaded image
            if st.button("Classify"):
                prediction = predict_image(image)

                # Map predicted class index to letter
                letters = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L", "M",
                           "N", "O", "P", "Q", "R", "S", "T", "U", "V", "W", "X", "Y", "Z"]
                predicted_letter = letters[prediction]

                st.write(f"Prediction: {predicted_letter}")

    elif option == "Use Image URL":
        # Image URL input
        image_url = st.text_input("Enter the image URL:")

        if st.button("Classify"):
            try:
                response = requests.get(image_url)
                image = Image.open(BytesIO(response.content))
                st.image(image, caption="Image from URL", use_column_width=True)
                prediction = predict_image(image)

                # Map predicted class index to letter
                letters = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L", "M",
                           "N", "O", "P", "Q", "R", "S", "T", "U", "V", "W", "X", "Y", "Z"]
                predicted_letter = letters[prediction]

                st.write(f"Prediction: {predicted_letter}")
            except Exception as e:
                st.write("Error occurred while processing the image. Please check the URL and try again.")
                st.write(f"Error Details: {str(e)}")

if __name__ == '__main__':
    main()

