import torch
from torchvision import datasets
from torchvision.transforms import ToTensor

from torch.utils.data import DataLoader

from torch import nn

import matplotlib.pyplot as plt

device = "cuda" if torch.cuda.is_available() else "cpu"
print("Device : ", device)


train_data = datasets.FashionMNIST(
    root = "root",
    train = True,
    download = True,
    transform = ToTensor()
)

test_data = datasets.FashionMNIST(
    root = "root",
    train = False,
    download = True,
    transform = ToTensor()
)

train_dataloader = DataLoader(
    dataset = train_data,
    batch_size = 32,
    shuffle = True
)

test_dataloader = DataLoader(
    dataset = test_data,
    shuffle = False,
    batch_size = 32
)


class FashionMNISTModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(
                in_channels = 1,
                out_channels = 32,
                kernel_size = 3,
                padding =1
            ),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size = 2),
            nn.Conv2d(
                    in_channels = 32,
                    out_channels = 64,
                    kernel_size = 3,
                    padding =1
                ),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size = 2),
            nn.Flatten(),
            nn.Linear(in_features = 3136 , out_features = 10)    #3136 = 64 * 7 * 7
        )

    def forward(self,x):
        return self.layers(x)

model = FashionMNISTModel().to(device)
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(
    model.parameters(),
    lr = 0.01
)

epochs = 2
for epoch in range(epochs):
    model.train()
    train_loss = 0
    correct = 0
    total = 0
    for X,y in train_dataloader:
        X = X.to(device)
        y = y.to(device)
        
        logits = model(X)
        loss = loss_fn(logits,y)

        train_loss += loss.item()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        prediction = torch.argmax(logits,dim=1)
        correct += (prediction == y).sum().item()
        total += y.size(0)
    train_loss /= len(train_dataloader)
    train_accuracy = correct / total *100
    print(f"Epoch: {epoch +1} | Loss : {train_loss:.4f} | Accuracy : {train_accuracy:.2f}")

def evaluate(model, dataloader):

    model.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.inference_mode():
        for X,y in dataloader:
            X = X.to(device)
            y = y.to(device)

            logits = model(X)

            loss = loss_fn(logits,y)
            test_loss += loss.item()

            predictions = torch.argmax(logits,dim=1)

            correct += (predictions == y).sum().item()
            total += y.size(0)
        test_loss /= len(dataloader)
        test_accuracy = correct / total *100
        return test_loss, test_accuracy
    

test_loss , test_accuracy = evaluate(model,test_dataloader)
print(f"Test Loss : { test_loss:.4f} | Test Accuracy : {test_accuracy:.2f}")
