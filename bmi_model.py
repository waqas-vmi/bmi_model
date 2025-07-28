# bmi_model.py
import torch.nn as nn
from torchvision.models import resnet18
from torchvision.models import ResNet18_Weights

class BMIRegressor(nn.Module):
    def __init__(self):
        super(BMIRegressor, self).__init__()
        self.base_model = resnet18(weights=ResNet18_Weights.DEFAULT)
        self.base_model.conv1 = nn.Conv2d(2, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.base_model.fc = nn.Linear(self.base_model.fc.in_features, 1)

    def forward(self, x):
        return self.base_model(x)
