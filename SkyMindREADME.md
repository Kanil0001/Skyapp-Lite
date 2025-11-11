# SkyMind Lite — AI Sky Analyzer

SkyMind Lite is a lightweight AI app that identifies constellations in night-sky photos.  
It’s optimized for **Arm-powered devices** using **TensorFlow Lite**, but it can be run and tested entirely on a PC.

---

## Features
- Upload a night-sky image and get instant AI analysis.
- Model trained on sample constellation data.
- Runs fully offline, no internet or API calls needed.
- Exported as `.tflite` model — Arm-ready.

---

## Setup
```bash
pip install -r requirements.txt
python train_model.py
python app.py

## How to Use
1. Create your `data/` folder with 2–3 subfolders of sky image classes (like `orion`, `cassiopeia`, `ursa_major`).
2. Train:
   ```bash
   python train_model.py