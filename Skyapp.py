# ...existing code...
import gradio as gr
import tensorflow as tf
import numpy as np
import cv2
import os

MODEL_PATH = "model/skymind.tflite"
DATA_DIR = "data/"

if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError("❌ No model found! Train it first using train_model.py")

# === Load TensorFlow Lite Model ===
interpreter = tf.lite.Interpreter(model_path=MODEL_PATH)
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# determine expected input size and dtype
_in_shape = input_details[0].get("shape", [1, 224, 224, 3])
try:
    IN_H = int(_in_shape[1])
    IN_W = int(_in_shape[2])
except Exception:
    IN_H, IN_W = 224, 224
IN_DTYPE = np.dtype(input_details[0].get("dtype", np.float32))

# === Get Labels from Folder Names ===
labels = sorted([d for d in os.listdir(DATA_DIR) if os.path.isdir(os.path.join(DATA_DIR, d))])
if not labels:
    labels = []

def _preprocess(img):
    if img is None:
        raise ValueError("No image provided")

    # Ensure numpy array
    img = np.asarray(img)

    # Handle grayscale or alpha channels
    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    elif img.shape[-1] == 4:
        img = cv2.cvtColor(img, cv2.COLOR_RGBA2RGB)

    # Resize to model expected size (cv2.resize expects (width, height) as (cols, rows) tuple)
    img_resized = cv2.resize(img, (IN_W, IN_H), interpolation=cv2.INTER_AREA)

    # Prepare tensor according to model input dtype
    if IN_DTYPE == np.float32:
        input_data = (img_resized.astype(np.float32) / 255.0)[np.newaxis, ...]
    else:
        # for uint8 or other integer types
        input_data = img_resized.astype(IN_DTYPE)[np.newaxis, ...]

    return input_data

def _dequantize_outputs(raw_preds, out_detail):
    # handle quantized outputs if quantization params present
    q_scale, q_zero = out_detail.get("quantization", (0.0, 0))
    if q_scale and q_zero:
        return q_scale * (raw_preds.astype(np.float32) - q_zero)
    return raw_preds.astype(np.float32)

def analyze(img):
    try:
        if not labels:
            return "No labels found in data/ — create class subfolders first."

        input_data = _preprocess(img)

        interpreter.set_tensor(input_details[0]["index"], input_data)
        interpreter.invoke()
        raw_preds = interpreter.get_tensor(output_details[0]["index"])[0]
        preds = _dequantize_outputs(raw_preds, output_details[0])

        idx = int(np.argmax(preds))
        confidence = float(preds[idx])
        # If model outputs are logits or probabilities in [0,1], show percent; otherwise still show numeric percent
        if confidence <= 1.0:
            confidence *= 100.0

        return f"🌌 Detected: {labels[idx]} — Confidence: {confidence:.2f}%"
    except Exception as e:
        return f"Error during analysis: {str(e)}"

gr.Interface(
    fn=analyze,
    inputs=gr.Image(type="numpy", label="Upload Night Sky Image"),
    outputs=gr.Textbox(label="Result"),
    title="SkyMind Lite 🌠",
    description="An AI app that identifies constellations in night-sky images. Runs locally and is Arm-ready via TensorFlow Lite."
).launch()
# ...existing code...