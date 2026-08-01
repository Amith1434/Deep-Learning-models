import torch
from torch import nn

from dataset import X, y

from sklearn.model_selection import train_test_split

torch.manual_seed(42)
X_tensor = torch.tensor(X, dtype = torch.float32)
y_tensor = torch.tensor(y, dtype = torch.float32)
print(type(X))

train_x, test_x,train_y, test_y = train_test_split(
    X_tensor, y_tensor, random_state = 42, test_size =.3
)

device = "cuda" if torch.cuda.is_available() else "cpu"

print(device)
train_x = train_x.to(device)
test_x = test_x.to(device)
train_y = train_y.to(device)
test_y = test_y.to(device)

train_y = train_y.unsqueeze(1)
test_y = test_y.unsqueeze(1)

class BinaryClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1 = nn.Linear(2,8)
        self.layer2 = nn.Linear(8,1)

    def forward(self,x):
        x = self.layer1(x)
        x = torch.relu(x)
        x = self.layer2(x)
        return x

model = BinaryClassifier().to(device)

loss_fn = nn.BCEWithLogitsLoss()

optimizer = torch.optim.Adam(
    model.parameters(), lr = .01
)

epochs = 200
for epoch in range(1, epochs+1):
    logits = model(train_x)
    loss = loss_fn(logits, train_y)
    optimizer.zero_grad()

    loss.backward()
    optimizer.step()

    if epoch%10 == 0:
        print(f"Epoch : {epoch} | Loss : {loss}")


model.eval()
with torch.no_grad():
    test_logits = model(test_x)
    test_prob = torch.sigmoid(test_logits)
    test_preds = (test_prob >= 0.5).float()
    accuracy = (test_preds == test_y).float().mean()
    print(f"Test Accuracy : {accuracy.item() * 100 : .2f}%")

