# train_bmi_model.py

import os
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms
from PIL import Image
import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt
from bmi_model import BMIRegressor  # make sure your model is defined here


# ---------- Custom Dataset ----------
class BMIDataset(Dataset):
    def __init__(self, csv_file, root_dir, transform=None):
        self.annotations = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):
        front_path = os.path.join(self.root_dir, self.annotations.iloc[idx]["filename_front"])
        side_path = os.path.join(self.root_dir, self.annotations.iloc[idx]["filename_side"])

        height_cm = self.annotations.iloc[idx]["height_cm"]
        weight_kg = self.annotations.iloc[idx]["weight_kg"]
        height_m = height_cm / 100.0
        bmi = weight_kg / (height_m ** 2)
        bmi = torch.tensor([bmi], dtype=torch.float32)

        front_img = Image.open(front_path).convert("L")
        side_img = Image.open(side_path).convert("L")

        if self.transform:
            front_img = self.transform(front_img)
            side_img = self.transform(side_img)

        combined = torch.cat((front_img, side_img), dim=0)  # shape: (2, 224, 224)
        return combined, bmi


# ---------- Training Function ----------
def train_model():
    torch.manual_seed(42)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5])
    ])

    dataset = BMIDataset(
        csv_file="dataset/bmi_data_new.csv",
        root_dir="dataset/images/silhouettes/training",
        transform=transform
    )

    # Split into 80% train / 20% test
    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=8)

    model = BMIRegressor().to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # ---------- Training Loop ----------
    model.train()
    for epoch in range(1, 201):
        total_loss = 0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            preds = model(x)
            loss = criterion(preds, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch} - Train Loss: {avg_loss:.4f}")

    # ---------- Evaluation ----------
    model.eval()
    y_true = []
    y_pred = []

    with torch.no_grad():
        for x, y in test_loader:
            x, y = x.to(device), y.to(device)
            outputs = model(x)
            y_true.extend(y.cpu().numpy())
            y_pred.extend(outputs.cpu().numpy())

    y_true = [val[0] for val in y_true]
    y_pred = [val[0] for val in y_pred]

    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)

    print("\n--- Evaluation Metrics ---")
    print(f"MAE: {mae:.2f}")
    print(f"RMSE: {rmse:.2f}")
    print(f"R² Score: {r2:.2f}")

    # ---------- Optional Visualization ----------
    plt.figure(figsize=(6, 6))
    plt.scatter(y_true, y_pred, alpha=0.7)
    plt.plot([min(y_true), max(y_true)], [min(y_true), max(y_true)], 'r--')
    plt.xlabel("Actual BMI")
    plt.ylabel("Predicted BMI")
    plt.title("Actual vs Predicted BMI")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("bmi_prediction_plot.png")
    plt.close()

    # Save the model
    torch.save(model.state_dict(), "bmi_model.pth")
    print("Model saved to bmi_model.pth")


# ---------- Main ----------
if __name__ == "__main__":
    train_model()
