import torch
from dataset import train_dataset, test_dataset
from torch.utils.data import DataLoader
from torch import nn

torch.manual_seed(42)

device = "cuda" if torch.cuda.is_available() else "cpu"
print("Device : ", device)


train_dataloader = DataLoader(
    dataset = train_dataset,
    batch_size = 64,
    shuffle = True)

test_dataloader = DataLoader(
    dataset = test_dataset,
    shuffle = False,
    batch_size = 64
)

class BrainTumorCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_block = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size = 3, padding = 1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),

            nn.Conv2d(32,64,kernel_size=3,padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 16 * 16,4)
        )

    def forward(self,x):
        x = self.conv_block(x)
        x = self.classifier(x)
        return x

model = BrainTumorCNN().to(device)
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(
    model.parameters(),
    lr = 0.001
)
epochs = 5

for epoch in range(epochs):
    model.train()
    train_loss = 0
    for X, y  in train_dataloader:
        X = X.to(device)
        y = y.to(device)
        logits = model(X)
        loss = loss_fn(logits, y)
        train_loss += loss.item()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    train_loss /=len(train_dataloader)
    print(f"Epoch : {epoch} | Loss : {train_loss}")

correct = 0
total = 0
model.eval()
with torch.inference_mode():
    for X, y in test_dataloader:
        X = X.to(device)
        y = y.to(device)

        logits = model(X)
        predictions = logits.argmax(dim=1)

        correct += (predictions == y).sum().item()
        total += y.size(0)

accuracy = correct / total

print(f"Test Accuracy: {accuracy:.2%}")