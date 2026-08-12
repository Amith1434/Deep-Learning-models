import os
import certifi
import sys

import torch
from torchvision import models
from torch import nn
from torchvision import datasets
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt


device = "cuda" if torch.cuda.is_available() else "cpu"
print("Device : ",device)
sys.path.append("06_Brain_Tumor_MRI_Classification")

from dataset import train_path, test_path

os.environ["SSL_CERT_FILE"] = certifi.where()



weights = models.ResNet18_Weights.DEFAULT
model = models.resnet18(
    weights = None
)
model.fc = nn.Linear(512,4)

for param in model.parameters():
    param.requires_grad = True

transform = weights.transforms()


train_dataset = datasets.ImageFolder(
    root = train_path,
    transform=transform
)
test_dataset = datasets.ImageFolder(
    root = test_path,
    transform = transform
)

print("Classes : ",train_dataset.classes)
print("Number of training images : ", len(train_dataset))
print("Number of testing images : ",len(test_dataset))

train_dataloader = DataLoader(
    dataset = train_dataset,
    batch_size = 32,
    shuffle = True
)

test_dataloader = DataLoader(
    dataset = test_dataset,
    shuffle = False,
    batch_size = 32
)

model = model.to(device)

loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(
    model.fc.parameters(),
    lr = 0.001
)

epochs = 3
for epoch in range(epochs):
    model.train()
    train_loss = 0
    for X, y in train_dataloader:
        X = X.to(device)
        y= y.to(device)
        logit = model(X)
        loss = loss_fn(logit,y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        train_loss += loss.item()

    train_loss /= len(train_dataloader)
    print(f"Epoch : {epoch+1} | Loss : {train_loss :.4f}")

model.eval()
correct = 0
total = 0
with torch.no_grad():
    for X,y in test_dataloader:
        X = X.to(device)
        y = y.to(device)
        logits = model(X)
        predictions = torch.argmax(logits,dim=1)
        total += y.size(0)
        correct += (predictions == y).sum().item()
    accuracy = correct /total
    print(f"Test Accuracy : {accuracy * 100:.2f}%")
