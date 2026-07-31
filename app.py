"""
Acne vs Eczema Skin Disease Classifier
--------------------------------------
Author : Your Name
Course : GET 324 Mini Project

This application uses AI to predict whether
an uploaded skin image contains Acne or Eczema.

Educational purposes only.
"""

import io
import json
import logging
import os

import numpy as np
import streamlit as st
from PIL import Image
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.models import load_model

# ----------------------------------------------------
# Configuration
# ----------------------------------------------------

APP_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_PATH = os.path.join(APP_DIR, "acne_vs_eczema_model.keras")
CLASS_INDEX_PATH = os.path.join(APP_DIR, "class_indices.json")

IMAGE_SIZE = (224, 224)
CONFIDENCE_THRESHOLD = 0.60

logging.basicConfig(level=logging.INFO)

# ----------------------------------------------------
# Page Configuration
# ----------------------------------------------------

st.set_page_config(
    page_title="Acne vs Eczema Classifier",
    page_icon="🩺",
    layout="centered",
)

# ----------------------------------------------------
# Load Model
# ----------------------------------------------------


@st.cache_resource
def load_classifier():

    if not os.path.exists(MODEL_PATH):
        st.error(f"Model not found:\n{MODEL_PATH}")
        st.stop()

    model = load_model(MODEL_PATH)

    logging.info("Model loaded successfully.")

    return model


model = load_classifier()

# ----------------------------------------------------
# Load Class Names
# ----------------------------------------------------

if os.path.exists(CLASS_INDEX_PATH):

    with open(CLASS_INDEX_PATH, "r") as file:

        class_indices = json.load(file)

else:

    class_indices = {
        "acne": 0,
        "eczema": 1,
    }

index_to_class = {v: k for k, v in class_indices.items()}

# ----------------------------------------------------
# Image Preprocessing
# ----------------------------------------------------


def preprocess_image(image_bytes):

    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

    image = image.resize(IMAGE_SIZE)

    image = np.asarray(image, dtype=np.float32)

    image = preprocess_input(image)

    image = np.expand_dims(image, axis=0)

    return image


# ----------------------------------------------------
# Prediction
# ----------------------------------------------------


def predict_image(image_bytes):

    processed = preprocess_image(image_bytes)

    probability = float(model.predict(processed, verbose=0)[0][0])

    acne_label = index_to_class.get(0, "Acne").title()
    eczema_label = index_to_class.get(1, "Eczema").title()

    if probability >= 0.5:

        label = eczema_label
        confidence = probability

    else:

        label = acne_label
        confidence = 1 - probability

    if confidence < CONFIDENCE_THRESHOLD:

        label = "Uncertain"

    return {
        "label": label,
        "confidence": confidence,
        "probability": probability,
    }


# ----------------------------------------------------
# User Interface
# ----------------------------------------------------

st.title("🩺 Acne vs Eczema Skin Disease Classifier")

st.info(
    """
This AI application predicts whether a skin image is more likely to
contain **Acne** or **Eczema**.
"""
)

uploaded_file = st.file_uploader(
    "Upload a skin image",
    type=["jpg", "jpeg", "png"],
)

if uploaded_file is not None:

    image = Image.open(uploaded_file)

    st.image(
        image,
        caption="Uploaded Image",
        use_column_width=True,
    )

    if st.button("Predict"):

        with st.spinner("Analyzing image..."):

            try:

                image_bytes = uploaded_file.getvalue()

                result = predict_image(image_bytes)

                if result["label"] == "Acne":

                    st.success("Acne Detected")

                elif result["label"] == "Eczema":

                    st.success("Eczema Detected")

                else:

                    st.info(
                        "The uploaded image does not appear to be Acne or Eczema, "
                        "or the model is not confident enough to classify it."
                    )

                st.metric(
                    "Confidence",
                    f"{result['confidence']}%"
                )

            except Exception as e:

                logging.exception(e)

                st.error(
                    "An error occurred while processing the uploaded image."
                )

# ----------------------------------------------------
# Footer
# ----------------------------------------------------

st.markdown("---")
st.caption(
    "Developed using TensorFlow, MobileNetV2, and Streamlit."
)