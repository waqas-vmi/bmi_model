from bmi_model import BMIRegressor
from PIL import Image
import torch
import torchvision.transforms as transforms

model = BMIRegressor()
model.load_state_dict(torch.load("bmi_model.pth"))
model.eval()

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])
])

def predict_bmi(front_path, side_path):
    front = Image.open(front_path).convert("L")
    side = Image.open(side_path).convert("L")
    front = transform(front)
    side = transform(side)
    combined = torch.cat((front, side), dim=0).unsqueeze(0)
    with torch.no_grad():
        pred = model(combined)
    return pred.item()

# Example
print(predict_bmi("dataset/images/silhouettes/testing/img5_front.png", "dataset/images/silhouettes/testing/img5_side.png"))
