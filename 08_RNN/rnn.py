import torch
from torch import nn

X = torch.tensor([
    [[1.0], [2.0], [3.0], [4.0], [5.0]],
    [[5.0], [5.0], [5.0], [5.0], [5.0]],
    [[2.0], [2.0], [2.0], [2.0], [2.0]],
    [[4.0], [5.0], [4.0], [5.0], [4.0]]
])

y = torch.tensor([0, 1, 0, 1])
rnn = nn.RNN(
    input_size = 1,
    hidden_size=16,
    batch_first=True
)

class SequenceClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.rnn = nn.RNN(
            input_size=1,
            hidden_size=16,batch_first=True
        )
        self.fc=nn.Linear(16,1)
    def forward(self,x):
        output,hidden = self.rnn(x)
        hidden = hidden.squeeze(0)
        output = self.fc(hidden)
        return output


model = SequenceClassifier()
predictions = model(X)
print(predictions.shape)

loss_fn = nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(model.parameters(),lr=0.01)

y = y.float().unsqueeze(1)

epochs = 100
for epoch in range(epochs):
    predictions = model(X)
    
    loss = loss_fn(predictions,y)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    if(epoch + 1) % 100 == 0:
        print(f"Epoch {epoch+1}, Loss : {loss.item():.4f}")



model.eval()
with torch.no_grad():
    logits = model(X)
    probabilities = torch.sigmoid(logits)
    predictions = (probabilities >= 0.5).float()

print("Probabilities:")
print(probabilities)

print("Predictions:")
print(predictions)

print("Actual:")
print(y)