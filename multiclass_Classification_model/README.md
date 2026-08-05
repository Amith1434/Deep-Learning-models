# Day 3 — Multiclass Classification

## Dataset & DataLoader

`Dataset` stores/accesses the data.

`DataLoader` gives the data to the model in batches.

```python
train_loader = DataLoader(
    train_data,
    batch_size=64,
    shuffle=True
)
```

One batch of Fashion-MNIST:

```text
X → [64, 1, 28, 28]
y → [64]
```

- 64 = batch size
- 1 = grayscale channel
- 28 × 28 = image size

---

## iter() and next()

```python
X, y = next(iter(train_loader))
```

- `iter()` → creates an iterator
- `next()` → gets the next batch

During training, the `for` loop handles this automatically:

```python
for X, y in train_loader:
```

---

## Flatten

A Linear layer needs features as a vector.

```text
[64, 1, 28, 28]
        ↓
    nn.Flatten()
        ↓
    [64, 784]
```

Because:

```text
28 × 28 = 784
```

---

## Multiclass Model

```python
self.layers = nn.Sequential(
    nn.Flatten(),
    nn.Linear(784, 128),
    nn.ReLU(),
    nn.Linear(128, 10)
)
```

10 outputs because Fashion-MNIST has **10 classes**.

---

## Logits

The final layer outputs raw numbers called **logits**.

```text
Model output → [64, 10]
```

Meaning:

```text
64 images × 10 class scores
```

To get the predicted class:

```python
predictions = torch.argmax(logits, dim=1)
```

`argmax` returns the index of the largest logit.

---

## CrossEntropyLoss

For multiclass classification:

```python
loss_fn = nn.CrossEntropyLoss()
```

Do **NOT** add Softmax to the model before `CrossEntropyLoss`.

```text
Model → raw logits → CrossEntropyLoss
```

CrossEntropyLoss handles the required log-softmax internally.

---

## Training Cycle

```text
Forward pass
    ↓
Calculate loss
    ↓
optimizer.zero_grad()
    ↓
loss.backward()
    ↓
optimizer.step()
```

```python
logits = model(X)
loss = loss_fn(logits, y)

optimizer.zero_grad()
loss.backward()
optimizer.step()
```

---

## Training Accuracy

```python
predictions = torch.argmax(logits, dim=1)

correct += (predictions == y).sum().item()
total += y.size(0)

accuracy = correct / total * 100
```

---

## train() vs eval()

Training:

```python
model.train()
```

Evaluation:

```python
model.eval()
```

During evaluation:

```python
with torch.inference_mode():
```

No gradients or weight updates are required.

---

## CPU / GPU

```python
device = "cuda" if torch.cuda.is_available() else "cpu"

model = model.to(device)
X = X.to(device)
y = y.to(device)
```

Dataset and DataLoader are **not** moved to the GPU.

Move each batch to the GPU during training.

---

## Single Image Prediction

A single image has:

```text
[1, 28, 28]
```

The model expects a batch, so:

```python
image = image.unsqueeze(0)
```

changes:

```text
[1, 28, 28]
      ↓
[1, 1, 28, 28]
```

Then:

```python
logits = model(image)
prediction = torch.argmax(logits, dim=1)
```

---

## Saving the Model

```python
torch.save(model.state_dict(), "fashion_mnist_model.pth")
```

Load:

```python
model.load_state_dict(
    torch.load("fashion_mnist_model.pth")
)
```

---

## Day 3 Mental Model

```text
Dataset
   ↓
DataLoader
   ↓
Batch [64,1,28,28]
   ↓
Flatten
   ↓
[64,784]
   ↓
Linear → ReLU → Linear
   ↓
Logits [64,10]
   ↓
CrossEntropyLoss
   ↓
Backpropagation
   ↓
Optimizer
   ↓
Prediction with argmax
```
