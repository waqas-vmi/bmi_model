import cv2
import mediapipe as mp
import os

# Input and output directories
input_dir = "dataset/images/dataset"
output_dir = "dataset/images/silhouettes/dataset"
os.makedirs(output_dir, exist_ok=True)

# Initialize MediaPipe segmentation
mp_selfie_segmentation = mp.solutions.selfie_segmentation
segment = mp_selfie_segmentation.SelfieSegmentation(model_selection=1)

# Loop through all images
for filename in os.listdir(input_dir):
    if not filename.lower().endswith(('.jpg', '.png')):
        continue

    img_path = os.path.join(input_dir, filename)
    img = cv2.imread(img_path)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    result = segment.process(img_rgb)

    mask = result.segmentation_mask > 0.1  # threshold the body
    silhouette = (mask * 255).astype("uint8")

    # Create a 3-channel grayscale silhouette
    silhouette_img = cv2.merge([silhouette, silhouette, silhouette])
    out_path = os.path.join(output_dir, filename)
    cv2.imwrite(out_path, silhouette_img)

print("✅ Silhouettes saved to:", output_dir)
