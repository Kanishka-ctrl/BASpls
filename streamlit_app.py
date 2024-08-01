import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np
import os

# Debugging: Check current directory and list files
st.write("Current Directory:", os.getcwd())
st.write("Files in Directory:", os.listdir('.'))

# Load your trained model (update the path to your model file)
model_path = 'path/to/your/model.h5'
if os.path.exists(model_path):
    model = tf.keras.models.load_model(model_path)
else:
    st.error(f"Model file not found at {model_path}")
    st.stop()

@st.cache
def preprocess_image(image):
    """Preprocess the uploaded image for prediction."""
    try:
        img = image.convert('RGB')  # Ensure image is in RGB mode
        img = img.resize((224, 224))  # Adjust size according to your model's input size
        img_array = np.array(img) / 255.0  # Normalize image data
        img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
        return img_array
    except Exception as e:
        st.error(f"Error preprocessing image: {e}")
        return None

@st.cache
def predict_image(img_array):
    """Predict the class of the image."""
    try:
        predictions = model.predict(img_array)
        return predictions
    except Exception as e:
        st.error(f"Error making prediction: {e}")
        return None

st.title("Tomato Disease Prediction")
st.write("Upload an image of a tomato plant leaf to get the disease prediction.")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    try:
        image = Image.open(uploaded_file)
        st.image(image, caption='Uploaded Image', use_column_width=True)
        
        # Preprocess the image and make prediction
        img_array = preprocess_image(image)
        if img_array is not None:
            predictions = predict_image(img_array)
            
            # Display predictions
            if predictions is not None:
                st.write("Predictions:")
                st.write(predictions)
    except Exception as e:
        st.error(f"Error loading image: {e}")
