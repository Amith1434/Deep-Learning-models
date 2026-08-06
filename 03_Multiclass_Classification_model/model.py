import torch 
from torch import nn

from torchvision import datasets
from torchvision.transforms import ToTensor

from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

device = "cuda" if torch.cuda.is_available() else "cpu"
print("Device  : ", device)
print(torch.cuda.get_device_name(0))
torch.manual_seed(42)

train_data = datasets.FashionMNIST(
    root = "root",
    train =True,
    download = True,
    transform=ToTensor()
)

test_data = datasets.FashionMNIST(
    root = "root",
    train = False,
    download = True,
    transform =ToTensor()
)

BATCH_SIZE = 64

train_dataloader =DataLoader(
    dataset = train_data,
    batch_size = BATCH_SIZE,
    shuffle=True
)

test_dataloader = DataLoader(
    dataset = test_data,
    batch_size = BATCH_SIZE,
    shuffle = False
)

class FashionMNISTModel(nn.Module):
    def __init__(self):
        super().__init__()

        self.layers = nn.Sequential(
            nn.Flatten(),
            nn.Linear(784,128),
            nn.ReLU(),
            nn.Linear(128,10)
        )

    def forward(self,x):
        return self.layers(x)

model = FashionMNISTModel().to(device)

loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(
    model.parameters(),
    lr = 0.001
)

epochs = 5

for epoch in range(epochs):
    model.train()
    train_loss = 0
    correct = 0
    total = 0

    for X, y in train_dataloader:
        X = X.to(device)
        y = y.to(device)
        logits = model(X)
        loss = loss_fn(logits, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        prediction = torch.argmax(logits,dim=1)
        correct += (prediction == y).sum().item()
        total += y.size(0)

    train_loss /= len(train_dataloader)
    train_accuracy = correct / total *100
    print(f"Epoch: {epoch +1} | Loss : {train_loss:.4f} | Accuracy : {train_accuracy:.2f}")


model.eval()
test_loss = 0
correct = 0
total = 0
with torch.inference_mode():
    for X,y in test_dataloader:
        X = X.to(device)
        y = y.to(device)

        logits = model(X)

        loss = loss_fn(logits,y)
        test_loss += loss.item()

        predictions = torch.argmax(logits,dim=1)

        correct += (predictions == y).sum().item()
        total += y.size(0)
    test_loss /= len(test_dataloader)
    test_accuracy = correct / total *100
    print(f"Test Loss : { test_loss:.4f} | Test Accuracy : {test_accuracy:.2f}")


class_names = train_data.classes
print(class_names)

image, label = test_data[0]
print("Image shape",image.shape)
print("Actual : ",class_names[label])

model.eval()
with torch.inference_mode():
    logits = model(image.unsqueeze(0).to(device))
    prediction = torch.argmax(logits,dim = 1).item()

print("predicted : ",class_names[prediction])

plt.imshow(image.squeeze(),cmap ="gray")
plt.title(f"Actual : {class_names[label]} | Predicted : {class_names[prediction]}")
plt.axis("off")
plt.show()

torch.save(model.state_dict(),"fashion_mnist_model.pth")
