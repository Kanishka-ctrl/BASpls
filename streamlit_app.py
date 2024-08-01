import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np

# Load your trained model (update the path to your model file)
model = tf.keras.models.load_model('path/to/your/model.h5')

@st.cache
def preprocess_image(image):
    """Preprocess the uploaded image for prediction."""
    img = image.resize((224, 224))  # Adjust size according to your model's input size
    img_array = np.array(img) / 255.0  # Normalize image data
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    return img_array

@st.cache
def predict_image(img_array):
    """Predict the class of the image."""
    predictions = model.predict(img_array)
    return predictions

st.title("Tomato Disease Prediction")
st.write("Upload an image of a tomato plant leaf to get the disease prediction.")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image', use_column_width=True)
    
    # Preprocess the image and make prediction
    img_array = preprocess_image(image)
    predictions = predict_image(img_array)
    
    # Display predictions
    st.write("Predictions:")
    st.write(predictions)
