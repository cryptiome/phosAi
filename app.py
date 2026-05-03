import os

import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image

from flask import Flask, request, jsonify


app = Flask(__name__)

app.config["MAX_CONTENT_LENGTH"] = 10 * 1024 * 1024

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

MODEL_PATH = os.environ.get(
    "MODEL_PATH",
    "checkpoints/best_model.pth"
)

ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg", "webp"}

transform = transforms.Compose([
    transforms.Resize((512, 512)),
    transforms.ToTensor()
])


def allowed_file(filename):
    return (
        "." in filename
        and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS
    )


def build_model():
    model = models.resnet50(weights=None)

    model.fc = nn.Sequential(
        nn.Linear(model.fc.in_features, 256),
        nn.ReLU(),
        nn.Dropout(0.3),
        nn.Linear(256, 1)
    )

    state_dict = torch.load(MODEL_PATH, map_location=DEVICE)
    model.load_state_dict(state_dict)

    model.to(DEVICE)
    model.eval()

    return model


model = build_model()


@app.route("/", methods=["GET"])
def home():
    return jsonify({
        "message": "Phosphate AI API is running",
        "model": "ResNet50 Image Regression",
        "status": "ok"
    })


@app.route("/health", methods=["GET"])
def health():
    return jsonify({
        "status": "healthy",
        "device": str(DEVICE),
        "model_loaded": model is not None
    })


@app.route("/predict", methods=["POST"])
def predict():
    try:
        if "image" not in request.files:
            return jsonify({
                "success": False,
                "error": "No image file provided. Use form-data key: image"
            }), 400

        image_file = request.files["image"]

        if image_file.filename == "":
            return jsonify({
                "success": False,
                "error": "Empty image filename"
            }), 400

        if not allowed_file(image_file.filename):
            return jsonify({
                "success": False,
                "error": "Invalid file type. Allowed: png, jpg, jpeg, webp"
            }), 400

        image = Image.open(image_file.stream).convert("RGB")
        image_tensor = transform(image).unsqueeze(0).to(DEVICE)

        with torch.no_grad():
            prediction = model(image_tensor)

        value = float(prediction.item())

        return jsonify({
            "success": True,
            "predicted_value": value
        })

    except Exception as e:
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(
        host="0.0.0.0",
        port=port,
        debug=False
    )