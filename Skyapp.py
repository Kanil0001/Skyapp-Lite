import gradio as gr
import tensorflow as tf
import numpy as np
import cv2
import os

MODEL_PATH = "model/skymind.tflite"
DATA_DIR = "data/"

# === Load Model ===
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError("❌ No model found! Train it first using train_model.py")

interpreter = tf.lite.Interpreter(model_path=MODEL_PATH)
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# determine model input shape + dtype
_in_shape = input_details[0].get("shape", [1, 224, 224, 3])
IN_H, IN_W = int(_in_shape[1]), int(_in_shape[2])
IN_DTYPE = np.dtype(input_details[0].get("dtype", np.float32))

# === Load Label Names from Folders ===
labels = sorted([d for d in os.listdir(DATA_DIR) if os.path.isdir(os.path.join(DATA_DIR, d))])
if not labels:
    raise RuntimeError("❌ No label folders found! Create subfolders in data/ for each object (e.g., orion_nebula, jupiter).")

def _preprocess(img):
    img = np.asarray(img)
    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    elif img.shape[-1] == 4:
        img = cv2.cvtColor(img, cv2.COLOR_RGBA2RGB)
    img_resized = cv2.resize(img, (IN_W, IN_H), interpolation=cv2.INTER_AREA)
    if IN_DTYPE == np.float32:
        input_data = (img_resized.astype(np.float32) / 255.0)[np.newaxis, ...]
    else:
        input_data = img_resized.astype(IN_DTYPE)[np.newaxis, ...]
    return input_data

def _dequantize(raw_preds, detail):
    q_scale, q_zero = detail.get("quantization", (0.0, 0))
    if q_scale and q_zero:
        return q_scale * (raw_preds.astype(np.float32) - q_zero)
    return raw_preds.astype(np.float32)

def analyze(img):
    try:
        input_data = _preprocess(img)
        interpreter.set_tensor(input_details[0]["index"], input_data)
        interpreter.invoke()
        raw_preds = interpreter.get_tensor(output_details[0]["index"])[0]
        preds = _dequantize(raw_preds, output_details[0])

        # normalize if not probabilities
        preds = np.exp(preds) / np.sum(np.exp(preds))

        # prepare results
        results = {labels[i].replace("_", " ").title(): float(preds[i] * 100) for i in range(len(labels))}
        return results
    except Exception as e:
        return {"Error": str(e)}

# === Gradio Interface ===
gr.Interface(
    fn=analyze,
    inputs=gr.Image(type="numpy", label="Upload Space Image"),
    outputs=gr.Label(num_top_classes=5),
    title="SkyMind Lite — Object Analyzer",
    description="Upload an image of the night sky to identify specific celestial objects like Orion Nebula, Comet Lemmon, or Jupiter using TensorFlow Lite."
).launch()

# ...existing code...
