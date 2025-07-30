# app.py (with MediaPipe Silhouette Generation)

from flask import Flask, request, render_template
from bmi_model import BMIRegressor
from PIL import Image
import torch
import torchvision.transforms as transforms
import os
import cv2
import numpy as np
import tempfile
import mediapipe as mp

app = Flask(__name__)
UPLOAD_FOLDER = 'static/uploaded'
SIL_FOLDER = 'static/silhouettes'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(SIL_FOLDER, exist_ok=True)

# Load the trained model
model = BMIRegressor()
model.load_state_dict(torch.load("bmi_model.pth", map_location=torch.device("cpu")))
model.eval()

# Initialize MediaPipe segmentation
mp_selfie_segmentation = mp.solutions.selfie_segmentation
segment = mp_selfie_segmentation.SelfieSegmentation(model_selection=1)

def to_silhouette_mediapipe(pil_image):
    # Convert PIL to OpenCV format
    img = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    result = segment.process(img_rgb)
    mask = result.segmentation_mask > 0.1
    silhouette = (mask * 255).astype("uint8")
    silhouette_img = cv2.merge([silhouette, silhouette, silhouette])
    return Image.fromarray(silhouette_img).convert("L")

def classify_bmi(bmi):
    if bmi < 18.5:
        return "Underweight"
    elif 18.5 <= bmi < 25:
        return "Normal Weight"
    elif 25 <= bmi < 30:
        return "Overweight"
    else:
        return "Obese"

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])
])

def predict_bmi(front_img, side_img):
    front_sil = to_silhouette_mediapipe(front_img)
    side_sil = to_silhouette_mediapipe(side_img)

    front_tensor = transform(front_sil)
    side_tensor = transform(side_sil)
    combined = torch.cat((front_tensor, side_tensor), dim=0).unsqueeze(0)

    with torch.no_grad():
        pred = model(combined)
    return round(pred.item(), 2), front_sil, side_sil

@app.route("/", methods=["GET", "POST"])
def index():
    bmi = None
    classified_bmi = None
    front_sil_path = side_sil_path = None

    if request.method == "POST":
        front_file = request.files["front"]
        side_file = request.files["side"]

        front_path = os.path.join(UPLOAD_FOLDER, front_file.filename)
        side_path = os.path.join(UPLOAD_FOLDER, side_file.filename)
        front_file.save(front_path)
        side_file.save(side_path)

        pred_bmi, front_sil, side_sil = predict_bmi(Image.open(front_path), Image.open(side_path))
        bmi = pred_bmi
        classified_bmi = classify_bmi(pred_bmi)

        # Save silhouette previews
        front_sil_path = os.path.join(SIL_FOLDER, "front_sil.png")
        side_sil_path = os.path.join(SIL_FOLDER, "side_sil.png")
        front_sil.save(front_sil_path)
        side_sil.save(side_sil_path)

    return render_template("index.html", bmi=bmi, classified_bmi=classified_bmi,front_sil=front_sil_path, side_sil=side_sil_path)

if __name__ == "__main__":
    app.run(debug=True)