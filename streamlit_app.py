import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np

# Dictionary to map class indices to disease names
class_dict = {
    0: 'Tomato Bacterial spot',
    1: 'Tomato Early blight',
    2: 'Tomato Late blight',
    3: 'Tomato Leaf Mold',
    4: 'Tomato Septoria leaf spot',
    5: 'Tomato Spider mites Two-spotted spider mite',
    6: 'Tomato Target Spot',
    7: 'Tomato Tomato Yellow Leaf Curl Virus',
    8: 'Tomato Tomato mosaic virus',
    9: 'Tomato healthy'
}

# Function to preprocess the image
def prepare_image(image):
    image = image / 255.0
    return np.expand_dims(image, axis=0)

# Function to predict the class of the tomato leaf
def predict_class(image, model):
    prediction = model.predict(image)
    return class_dict[np.argmax(prediction)]

# Load and preprocess the image
@st.cache_data
def load_image(image_file):
    img = Image.open(image_file)
    img = img.resize((128, 128))
    img = tf.keras.preprocessing.image.img_to_array(img)
    return img

# Main function to run the Streamlit app
def main():
    st.set_page_config(page_title="Tomato Disease Prediction", page_icon="🍅", layout="wide")
    st.image("./img2.jpg", use_column_width=True)
    st.title("Tomato Disease Prediction")
    st.subheader("Upload a Tomato Leaf Image")

    image_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

    if image_file is not None:
        img = load_image(image_file)
        st.image(img, caption="Uploaded Image", use_column_width=True)
        st.write("")
        st.write("Classifying...")

        with st.spinner('Loading model...'):
            model = tf.keras.models.load_model("model_vgg19.h5")

        image = prepare_image(img)
        prediction = predict_class(image, model)
        st.success(f"The tomato leaf is classified as: **{prediction}**")

if __name__ == "__main__":
    main()
