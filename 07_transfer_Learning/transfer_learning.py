import os
import certifi
import sys

import torch
from torchvision import models
from torch import nn

import matplotlib.pyplot as plt

sys.path.append("06_Brain_Tumor_MRI_Classification")

from dataset import train_path, test_path

os.environ["SSL_CERT_FILE"] = certifi.where()

model = models.resnet18(
    weights = models.ResNet18_Weights.DEFAULT
)
model.fc = nn.Linear(512,2)

for param in model.parameters():
    param.requires_grad = False

for param in model.fc.parameters():
    param.requires_grad = True

for name, param in model.named_parameters():
    if param.requires_grad:
        print(name)
    


weights = models.ResNet18_Weights.DEFAULT
transform = weights.transforms()
print(transform)
