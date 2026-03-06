from flask import Flask, request, jsonify, render_template
from flask_cors import CORS

import torch
from transformers import ViTForImageClassification, ViTImageProcessor
from PIL import Image
import io

# -----------------------------
# App initialization
# -----------------------------
app = Flask(__name__)
CORS(app)

# -----------------------------
# Config
# -----------------------------
MODEL_PATH = "vit_deepfake_final"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(f"Using device: {device}")

# -----------------------------
# Load model & processor
# -----------------------------
try:
    model = ViTForImageClassification.from_pretrained(MODEL_PATH)
    processor = ViTImageProcessor.from_pretrained(MODEL_PATH)

    model.to(device)
    model.eval()
    print(" Model loaded successfully")

except Exception as e:
    print(f" Model loading failed: {e}")
    raise e

# -----------------------------
# Serve frontend
# -----------------------------
@app.route("/", methods=["GET"])
def home():
    return render_template("index.html")

# -----------------------------
# Health check
# -----------------------------
@app.route("/health", methods=["GET"])
def health():
    return jsonify({
        "status": "ok",
        "device": str(device)
    })

# -----------------------------
# Prediction route
# -----------------------------
@app.route("/predict", methods=["POST"])
def predict():
    if "file" not in request.files:
        return jsonify({"error": "No image file provided"}), 400

    image_file = request.files["file"]

    try:
        # Read image
        image = Image.open(io.BytesIO(image_file.read())).convert("RGB")

        # Preprocess
        inputs = processor(images=image, return_tensors="pt")
        inputs = {k: v.to(device) for k, v in inputs.items()}

        # Inference
        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits
            probs = torch.softmax(logits, dim=1)

            predicted_class = torch.argmax(probs, dim=1).item()
            confidence = probs[0][predicted_class].item()

        label_map = model.config.id2label

        return jsonify({
            "label": label_map[predicted_class],
            "confidence": round(confidence, 4),
            "scores": {
                label_map[0]: round(probs[0][0].item(), 4),
                label_map[1]: round(probs[0][1].item(), 4)
            }
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


# -----------------------------
# Run server
# -----------------------------
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)