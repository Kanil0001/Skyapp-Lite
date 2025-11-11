import tensorflow as tf
from tensorflow.keras import layers, models
import os

# === Configuration ===
IMG_SIZE = (224, 224)
BATCH_SIZE = 8
DATA_DIR = "data/"

# === Load Dataset ===
train_ds = tf.keras.utils.image_dataset_from_directory(
    DATA_DIR,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE
)

class_names = train_ds.class_names
print("Classes:", class_names)

# === Base Model ===
base_model = tf.keras.applications.MobileNetV2(
    input_shape=IMG_SIZE + (3,),
    include_top=False,
    weights="imagenet"
)
base_model.trainable = False

# === Build Model ===
model = models.Sequential([
    base_model,
    layers.GlobalAveragePooling2D(),
    layers.Dense(128, activation="relu"),
    layers.Dense(len(class_names), activation="softmax")
])

model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

# === Train Model ===
model.fit(train_ds, epochs=5)
os.makedirs("model", exist_ok=True)
model.save("model/skymind.h5")

# === Export to TensorFlow Lite (Arm compatible) ===
converter = tf.lite.TFLiteConverter.from_saved_model("model")
tflite_model = converter.convert()

with open("model/skymind.tflite", "wb") as f:
    f.write(tflite_model)

print("✅ Model trained and exported to model/skymind.tflite")
