import streamlit as st
import tensorflow as tf
from PIL import Image
import cv2
import numpy as np
import requests
from io import BytesIO

model_url = 'https://github.com/keithskadi24/test_cpe019/blob/c997cfd9ec5980e68f914196b1304c910d807158/braille_finalmodel.h5'
response = requests.get(model_url)
model_path = BytesIO(response.content)
model = tf.keras.models.load_model(model_path)

def main():
    st.title("Braille Character Recognition")
    st.write("Upload an image for prediction.")

    # Create a file uploader
    uploaded_file = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        # Preprocess the uploaded image
        image = Image.open(uploaded_file)
        image = image.convert('L')  # Convert to grayscale
        image = image.resize((28, 28))  # Resize to model input size
        img_array = np.array(image) / 255.0  # Normalize pixel values
        img_array = img_array[..., np.newaxis]  # Add channel dimension

        # Make prediction
        prediction = model.predict(np.array([img_array]))
        predicted_class = np.argmax(prediction)
        predicted_label = chr(predicted_class + 97)

        # Display the uploaded image and prediction
        st.image(image, caption=f"Predicted Label: {predicted_label}", use_column_width=True)

if __name__ == "__main__":
    main()

