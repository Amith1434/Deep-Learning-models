import pandas as pd
import numpy as np
import torch
from torch import nn
from sklearn.preprocessing import MinMaxScaler


# Load dataset
df = pd.read_csv(
    "09_LSTM/household_power_consumption (1).txt",
    sep=";",
    na_values="?"
)

# Use only the column we need
data = df["Global_active_power"].dropna().values

# Use a small portion
data = data[:5000]

# Normalize
scaler = MinMaxScaler()
data = scaler.fit_transform(data.reshape(-1, 1))


# Create sequences
sequence_length = 24

X = []
y = []

for i in range(len(data) - sequence_length):
    X.append(data[i:i + sequence_length])
    y.append(data[i + sequence_length])

X = torch.tensor(np.array(X), dtype=torch.float32)
y = torch.tensor(np.array(y), dtype=torch.float32)


# LSTM model
class LSTMModel(nn.Module):

    def __init__(self):
        super().__init__()

        self.lstm = nn.LSTM(
            input_size=1,
            hidden_size=32,
            batch_first=True
        )

        self.fc = nn.Linear(32, 1)

    def forward(self, x):

        output, (hidden, cell) = self.lstm(x)

        last_output = output[:, -1, :]

        return self.fc(last_output)


# Create model
model = LSTMModel()

loss_function = nn.MSELoss()

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=0.001
)


# Training
epochs = 5

for epoch in range(epochs):

    prediction = model(X)

    loss = loss_function(
        prediction,
        y
    )

    optimizer.zero_grad()

    loss.backward()

    optimizer.step()

    print(
        f"Epoch {epoch + 1}/{epochs}, "
        f"Loss: {loss.item():.6f}"
    )


# Test one prediction
model.eval()

with torch.no_grad():

    prediction = model(X[-1].unsqueeze(0))

print("\nPredicted:", prediction.item())
print("Actual:", y[-1].item())