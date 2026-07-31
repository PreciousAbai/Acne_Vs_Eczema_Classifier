"""
Acne vs Eczema Skin Disease Classifier
--------------------------------------
Author : Your Name
Course : GET 324 Mini Project

This application uses AI to predicts whether
an uploaded skin image contains Acne or Eczema.
"""

import io
import json
import logging
import os

import numpy as np
from flask import Flask, jsonify, render_template_string, request
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

# ----------------------------------------------------
# Flask App
# ----------------------------------------------------

app = Flask(__name__)

logging.basicConfig(level=logging.INFO)

# ----------------------------------------------------
# Load Model
# ----------------------------------------------------

if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(
        f"Model not found:\n{MODEL_PATH}\n"
        "Please train the model before running the application."
    )

model = load_model(MODEL_PATH)
logging.info("Model loaded successfully.")

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

index_to_class = {value: key for key, value in class_indices.items()}

# ----------------------------------------------------
# HTML Template
# ----------------------------------------------------

INDEX_HTML = """
<!DOCTYPE html>

<html>
<head>

<title>Acne vs Eczema Classifier</title>

<style>

body{
font-family:Arial;
max-width:600px;
margin:40px auto;
padding:20px;
background:#f7f7f7;
}

.card{
background:white;
padding:25px;
border-radius:10px;
box-shadow:0 2px 8px rgba(0,0,0,.15);
}

.result{
margin-top:20px;
padding:15px;
background:#eef;
border-radius:8px;
}

.warning{
color:#d97706;
font-size:14px;
}

button{
padding:10px 18px;
margin-top:10px;
cursor:pointer;
}

</style>

</head>

<body>

<div class="card">

<h2>Acne vs Eczema Classifier</h2>

<p class="warning">
Educational purpose only. This application is NOT a medical diagnosis tool.
Consult a qualified dermatologist for medical advice.
</p>

<form action="/predict" method="POST" enctype="multipart/form-data">

<input type="file" name="image" accept="image/*" required>

<br><br>

<button type="submit">
Predict
</button>

</form>

{% if result %}

<div class="result">

<h3>Prediction Result</h3>

<p><strong>Prediction:</strong> {{ result.label }}</p>

</div>

{% endif %}

</div>

</body>

</html>

"""

# ----------------------------------------------------
# Image Preprocessing
# ----------------------------------------------------


def preprocess_image(image_bytes):
    """
    Converts an uploaded image into a MobileNetV2-compatible tensor.
    """

    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    image = image.resize(IMAGE_SIZE)

    image_array = np.asarray(image, dtype=np.float32)

    image_array = preprocess_input(image_array)

    image_array = np.expand_dims(image_array, axis=0)

    return image_array


# ----------------------------------------------------
# Prediction Function
# ----------------------------------------------------


def predict_image(image_bytes):
    """
    Runs inference on the uploaded image.
    """

    processed_image = preprocess_image(image_bytes)

    probability = float(model.predict(processed_image, verbose=0)[0][0])

    acne_label = index_to_class.get(0, "Acne")
    eczema_label = index_to_class.get(1, "Eczema")

    if probability >= 0.5:
        predicted_label = eczema_label
    else:
        predicted_label = acne_label
        confidence = 1 - probability

    if confidence < CONFIDENCE_THRESHOLD:
        predicted_label = "Uncertain"

    return {
        "label": predicted_label,
    }


# ----------------------------------------------------
# Routes
# ----------------------------------------------------


@app.route("/")
def home():
    return render_template_string(INDEX_HTML, result=None)


@app.route("/predict", methods=["POST"])
def predict():

    if "image" not in request.files:
        return jsonify({"error": "Image file is required."}), 400

    uploaded_file = request.files["image"]

    if uploaded_file.filename == "":
        return jsonify({"error": "No image selected."}), 400

    try:

        image_bytes = uploaded_file.read()

        prediction = predict_image(image_bytes)

    except Exception as error:

        logging.exception(error)

        return jsonify({
            "error": "Unable to process the uploaded image."
        }), 500

    if request.accept_mimetypes.accept_html and not request.is_json:
        return render_template_string(
            INDEX_HTML,
            result=prediction
        )

    return jsonify(prediction)


@app.route("/health")
def health():
    """
    Health check endpoint.
    """

    return jsonify({
        "status": "OK",
        "model_loaded": True
    })


# ----------------------------------------------------
# Main
# ----------------------------------------------------

if __name__ == "__main__":

    app.run(
        host="0.0.0.0",
        port=5000,
        debug=True,
    )