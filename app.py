import os

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

import gc
import time

import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image, ImageOps

from flask import Flask, request, jsonify


torch.set_num_threads(1)
torch.set_num_interop_threads(1)

Image.MAX_IMAGE_PIXELS = 20000000

app = Flask(__name__)

app.config["MAX_CONTENT_LENGTH"] = 8 * 1024 * 1024

DEVICE = torch.device("cpu")

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
    print("Loading model...", flush=True)

    model = models.resnet50(weights=None)

    model.fc = nn.Sequential(
        nn.Linear(model.fc.in_features, 256),
        nn.ReLU(),
        nn.Dropout(0.3),
        nn.Linear(256, 1)
    )

    state_dict = torch.load(
        MODEL_PATH,
        map_location=DEVICE,
        weights_only=True
    )

    model.load_state_dict(state_dict)
    model.to(DEVICE)
    model.eval()

    print("Model loaded successfully", flush=True)

    return model


def preprocess_image(image_stream):
    with Image.open(image_stream) as image:
        image = ImageOps.exif_transpose(image)
        image = image.convert("RGB")

        if image.width > 2048 or image.height > 2048:
            image.thumbnail((2048, 2048))

        image_tensor = transform(image).unsqueeze(0).to(DEVICE)

    return image_tensor


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


@app.errorhandler(413)
def file_too_large(error):
    return jsonify({
        "success": False,
        "error": "Image file is too large. Please upload an image smaller than 8MB."
    }), 413


@app.route("/predict", methods=["POST"])
def predict():
    start_time = time.time()

    try:
        print("Prediction request received", flush=True)

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

        print(f"Processing image: {image_file.filename}", flush=True)

        image_tensor = preprocess_image(image_file.stream)

        with torch.inference_mode():
            prediction = model(image_tensor)

        value = float(prediction.item())

        elapsed_time = round(time.time() - start_time, 3)

        print(
            f"Prediction completed. Value: {value}, Time: {elapsed_time}s",
            flush=True
        )

        del image_tensor
        del prediction
        gc.collect()

        return jsonify({
            "success": True,
            "predicted_value": value,
            "processing_time_seconds": elapsed_time
        })

    except Exception as e:
        gc.collect()

        print(f"Prediction error: {str(e)}", flush=True)

        return jsonify({
            "success": False,
            "error": str(e)
        }), 500