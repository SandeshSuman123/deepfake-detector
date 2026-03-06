# DeepScan — Real vs Deepfake Image Detector

A web app that detects whether a face image is **real or AI-generated (deepfake)** using a fine-tuned Vision Transformer (ViT) model, served via a Flask backend.

---

## Demo

Upload any face image → get an instant verdict with confidence score.

---

## Model

- **Architecture:** Vision Transformer (`ViT-Base-Patch16-224`)
- **Base model:** `dima806/deepfake_vs_real_image_detection`
- **Fine-tuned on:** Custom dataset of real and AI-generated faces
- **Classes:** `Real`, `Fake`
- **Saved format:** SafeTensors

---

##  Project Structure

```
project/
├── app.py                  # Flask backend
├── templates/
│   └── index.html          # Frontend UI
└── vit_deepfake_final/     # Saved model
    ├── config.json
    ├── model.safetensors
    └── preprocessor_config.json
``

---

## Setup & Run

**1. Install dependencies**
```bash
pip install flask flask-cors torch transformers pillow
```

**2. Run the app**
```bash
python app.py
```

**3. Open in browser**
```
http://localhost:5000
```

---

##  API

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/` | Serves the frontend |
| `GET` | `/health` | Health check |
| `POST` | `/predict` | Predict real vs fake |

**Predict request:**
```
POST /predict
Content-Type: multipart/form-data
Body: file=<image>
```

**Response:**
```json
{
  "label": "Real",
  "confidence": 0.9741,
  "scores": {
    "Real": 0.9741,
    "Fake": 0.0259
  }
}
```

---

##  Built With

- [PyTorch](https://pytorch.org/)
- [Hugging Face Transformers](https://huggingface.co/transformers/)
- [Flask](https://flask.palletsprojects.com/)
- Vision Transformer (ViT)

---

##  Disclaimer

This tool is intended for **educational and research purposes only**. No detection system is 100% accurate.
