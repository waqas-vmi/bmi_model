# train.py
import os
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import pandas as pd
from bmi_model import BMIRegressor

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

        combined = torch.cat((front_img, side_img), dim=0)
        return combined, bmi

def train_model():
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5])
    ])

    # dataset = BMIDataset(csv_file="dataset/bmi_data.csv", root_dir="dataset/images/training", transform=transform)
    # dataset = BMIDataset(csv_file="dataset/bmi_data.csv", root_dir="dataset/images/silhouettes", transform=transform)
    dataset = BMIDataset(csv_file="dataset/bmi_data_new.csv", root_dir="dataset/images/silhouettes/training", transform=transform)
    loader = DataLoader(dataset, batch_size=8, shuffle=True)

    model = BMIRegressor()
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    model.train()
    for epoch in range(1, 101):
        total_loss = 0
        for x, y in loader:
            preds = model(x)
            loss = criterion(preds, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch+1} - Loss: {total_loss:.4f}")

    torch.save(model.state_dict(), "bmi_model.pth")

if __name__ == "__main__":
    train_model()
