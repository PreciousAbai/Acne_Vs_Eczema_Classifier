"""
Train Acne vs Eczema Classifier

Author: Your Name
Course: GET 324 Mini Project

This script trains a MobileNetV2 transfer learning model to classify
skin images as either Acne or Eczema.
"""

import json
import os

import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.callbacks import (
    EarlyStopping,
    ModelCheckpoint,
    ReduceLROnPlateau,
)
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# -------------------------------------------------
# Configuration
# -------------------------------------------------

DATA_DIR = "data"

IMAGE_SIZE = (224, 224)
BATCH_SIZE = 32

HEAD_EPOCHS = 10
FINE_TUNE_EPOCHS = 5

CLASS_NAMES = ["Acne", "Eczema"]

MODEL_NAME = "acne_vs_eczema_model.keras"

# -------------------------------------------------
# Data Generators
# -------------------------------------------------

train_datagen = ImageDataGenerator(
    preprocessing_function=preprocess_input,
    validation_split=0.2,      # 80% training, 20% validation
    rotation_range=20,
    zoom_range=0.15,
    width_shift_range=0.1,
    height_shift_range=0.1,
    horizontal_flip=True,
)

train_generator = train_datagen.flow_from_directory(
    DATA_DIR,
    target_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="binary",
    classes=CLASS_NAMES,
    subset="training",
    shuffle=True,
)

validation_generator = train_datagen.flow_from_directory(
    DATA_DIR,
    target_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="binary",
    classes=CLASS_NAMES,
    subset="validation",
    shuffle=False,
)

# -------------------------------------------------
# Save Class Labels
# -------------------------------------------------

with open("class_indices.json", "w") as file:
    json.dump(train_generator.class_indices, file)

print("Class Mapping")
print(train_generator.class_indices)

# -------------------------------------------------
# Build Model
# -------------------------------------------------

base_model = MobileNetV2(
    weights="imagenet",
    include_top=False,
    input_shape=(224, 224, 3),
)

base_model.trainable = False

x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dropout(0.3)(x)
x = Dense(128, activation="relu")(x)
x = Dropout(0.2)(x)

output = Dense(1, activation="sigmoid")(x)

model = Model(inputs=base_model.input, outputs=output)

# -------------------------------------------------
# Compile
# -------------------------------------------------

model.compile(
    optimizer=Adam(1e-3),
    loss="binary_crossentropy",
    metrics=[
        "accuracy",
        tf.keras.metrics.AUC(name="auc"),
    ],
)

callbacks = [

    ModelCheckpoint(
        MODEL_NAME,
        monitor="val_auc",
        save_best_only=True,
        mode="max",
    ),

    EarlyStopping(
        monitor="val_loss",
        patience=5,
        restore_best_weights=True,
    ),

    ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.5,
        patience=2,
    ),
]

# -------------------------------------------------
# Stage 1 Training
# -------------------------------------------------

print("\nTraining Classification Head...\n")

model.fit(
    train_generator,
    validation_data=validation_generator,
    epochs=HEAD_EPOCHS,
    callbacks=callbacks,
)

# -------------------------------------------------
# Fine Tuning
# -------------------------------------------------

base_model.trainable = True

for layer in base_model.layers[:-30]:
    layer.trainable = False

model.compile(
    optimizer=Adam(1e-5),
    loss="binary_crossentropy",
    metrics=[
        "accuracy",
        tf.keras.metrics.AUC(name="auc"),
    ],
)

print("\nFine-Tuning Model...\n")

model.fit(
    train_generator,
    validation_data=validation_generator,
    epochs=FINE_TUNE_EPOCHS,
    callbacks=callbacks,
)

# -------------------------------------------------
# Save Final Model
# -------------------------------------------------

model.save(MODEL_NAME)

print("\nTraining Completed Successfully.")

# -------------------------------------------------
# Evaluation
# -------------------------------------------------

loss, accuracy, auc = model.evaluate(validation_generator)

print("\nValidation Results")
print(f"Loss      : {loss:.4f}")
print(f"Accuracy  : {accuracy:.4f}")
print(f"AUC Score : {auc:.4f}")

print(f"\nModel saved as {MODEL_NAME}")