import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
import os

# Paths
DATA_DIR = "data"
MODEL_DIR = "model"
MODEL_KERAS = os.path.join(MODEL_DIR, "skymind.keras")
MODEL_TFLITE = os.path.join(MODEL_DIR, "skymind.tflite")

# Make sure data exists
if not os.path.exists(DATA_DIR):
    raise FileNotFoundError("❌ No data/ folder found. Create folders like data/orion_nebula/, data/jupiter/, etc.")

# Load dataset (TensorFlow will treat each folder name as a label)
img_height, img_width = 224, 224
train_ds = tf.keras.utils.image_dataset_from_directory(
    DATA_DIR,
    image_size=(img_height, img_width),
    batch_size=8
)

class_names = train_ds.class_names
print("Classes:", class_names)

# Normalize pixel values
train_ds = train_ds.map(lambda x, y: (x / 255.0, y))

# Use a small efficient model (MobileNetV2)
base_model = tf.keras.applications.MobileNetV2(
    input_shape=(img_height, img_width, 3),
    include_top=False,
    weights='imagenet'
)
base_model.trainable = False  # freeze feature extractor

model = keras.Sequential([
    base_model,
    layers.GlobalAveragePooling2D(),
    layers.Dense(len(class_names), activation='softmax')
])

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
print("Training started...")
model.fit(train_ds, epochs=5)

# Save keras model
os.makedirs(MODEL_DIR, exist_ok=True)
model.save(MODEL_KERAS)
print(f"✅ Saved Keras model to {MODEL_KERAS}")

# Convert to TensorFlow Lite
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()
with open(MODEL_TFLITE, "wb") as f:
    f.write(tflite_model)
print(f"✅ Converted and saved TensorFlow Lite model at {MODEL_TFLITE}")

print("Training complete! You can now run app.py to test images.")
