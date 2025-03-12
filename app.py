# Importing Necessary Libraries
import streamlit as st
from PIL import Image
import io
import numpy as np
import tensorflow as tf
import os
from utils import clean_image, get_prediction, make_results

# Loading the Model and saving to cache
@st.cache_resource
def load_model(path):
    if not os.path.exists(path):
        st.error(f"Model file '{path}' not found. Please make sure the file exists in the project directory.")
        return None

    # Xception Model
    xception_model = tf.keras.models.Sequential([
        tf.keras.applications.xception.Xception(include_top=False, weights='imagenet', input_shape=(512, 512, 3)),
        tf.keras.layers.GlobalAveragePooling2D(),
        tf.keras.layers.Dense(4, activation='softmax')
    ])

    # DenseNet Model
    densenet_model = tf.keras.models.Sequential([
        tf.keras.applications.densenet.DenseNet121(include_top=False, weights='imagenet', input_shape=(512, 512, 3)),
        tf.keras.layers.GlobalAveragePooling2D(),
        tf.keras.layers.Dense(4, activation='softmax')
    ])

    # Ensembling the Models
    inputs = tf.keras.Input(shape=(512, 512, 3))
    xception_output = xception_model(inputs)
    densenet_output = densenet_model(inputs)
    outputs = tf.keras.layers.average([densenet_output, xception_output])
    model = tf.keras.Model(inputs=inputs, outputs=outputs)

    # Loading the Weights of Model
    model.load_weights(path)
    return model


# Removing Streamlit Menu and Footer
hide_streamlit_style = """
    <style>
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    </style>
"""
st.markdown(hide_streamlit_style, unsafe_allow_html=True)

# Load Model
model = load_model('model.weights.h5')


# Title and Description
st.title('Plant Disease Detection')
st.write("Upload a plant leaf image to check if the plant is healthy or diseased.")

# File Upload
uploaded_file = st.file_uploader("Choose an image file", type=["png", "jpg", "jpeg"])

# If a file is uploaded, process it
if uploaded_file is not None:
    # Show Progress
    progress = st.text("Processing image...")
    my_bar = st.progress(0)

    # Load and display the image
    image = Image.open(io.BytesIO(uploaded_file.read()))
    resized_image = np.array(image.resize((700, 400), Image.LANCZOS))
    st.image(resized_image, caption="Uploaded Image", use_column_width=True)
    my_bar.progress(40)

    # Preprocess the image
    image = clean_image(image)

    # Check if model is loaded
    if model is not None:
        # Make predictions
        predictions, predictions_arr = get_prediction(model, image)
        my_bar.progress(70)

        # Generate results
        result = make_results(predictions, predictions_arr)
        my_bar.progress(100)

        # Clear progress bar
        progress.empty()
        my_bar.empty()

        # Display results
        st.write(f"The plant is **{result['status']}** with a prediction confidence of **{result['prediction']}**.")
    else:
        st.error("Model not loaded. Please check if 'model.h5' exists in the project folder.")
