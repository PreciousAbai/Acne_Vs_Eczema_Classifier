# Acne vs Eczema Classification

## Project Overview

This project uses Artificial Intelligence to classify skin images as either **Acne** or **Eczema**. The model was developed with TensorFlow/Keras and deployed using Streamlit.

## Features

- Upload skin images.
- Predict Acne or Eczema.
- Displays confidence score.
- Rejects images that are not confidently identified.

## Technologies Used

- Python
- TensorFlow/Keras
- Streamlit
- NumPy
- Pillow

## Project Structure

Acne_vs_Eczema/
│── app.py
│── acne_vs_eczema_model.keras
│── requirements.txt
│── README.md

## How to Run

Install dependencies:

pip install -r requirements.txt

Run:

streamlit run app.py

Upload a skin image for prediction.

## Dataset

The model was trained using the MSC-6 Skin Condition Dataset from Kaggle, using only the Acne and Eczema image classes.

## Future Improvements

- Improve accuracy with more training data.
- Support additional skin diseases.
- Enhance confidence threshold handling.
