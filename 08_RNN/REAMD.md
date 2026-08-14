# RNN and LSTM Sequence Classification

## 📌 Overview

This project explores **Recurrent Neural Networks (RNNs)** and **Long Short-Term Memory (LSTM)** networks using PyTorch.

The goal was to understand how neural networks process **sequential data**, how hidden states carry information between timesteps, and how LSTMs improve upon traditional RNNs by maintaining an additional cell state.

The project uses a small synthetic sequence-classification dataset to focus on understanding the architecture and data flow rather than dataset complexity.

---

## 🧠 Concepts Learned

### 1. Sequence Data

Sequential data is represented using the following format:

```text
[batch_size, sequence_length, features]
```

For this project:

```text
[4, 5, 1]
```

means:

* `4` → number of samples
* `5` → timesteps per sample
* `1` → feature at each timestep

---

### 2. Recurrent Neural Networks

An RNN processes a sequence one timestep at a time while carrying a **hidden state** from one timestep to the next.

Conceptually:

```text
x₁ → RNN → h₁
           ↓
x₂ → RNN → h₂
           ↓
x₃ → RNN → h₃
           ↓
x₄ → RNN → h₄
           ↓
x₅ → RNN → h₅
```

The hidden state acts as the network's memory of previous inputs.

In PyTorch:

```python
nn.RNN(
    input_size=1,
    hidden_size=16,
    batch_first=True
)
```

---

### 3. RNN Output and Hidden State

For an input with shape:

```text
[4, 5, 1]
```

and:

```text
hidden_size = 16
```

the RNN produces:

```text
Output: [4, 5, 16]
Hidden: [1, 4, 16]
```

The output contains a hidden representation for **every timestep**.

The hidden state contains the **final hidden representation** for each sample.

---

### 4. LSTM

LSTM stands for **Long Short-Term Memory**.

LSTMs were designed to handle long-term dependencies better than traditional RNNs.

Unlike a basic RNN, an LSTM maintains two states:

```text
Hidden State
+
Cell State
```

The LSTM uses gates to control information flow:

* **Forget Gate** — determines what information should be discarded
* **Input Gate** — determines what new information should be stored
* **Output Gate** — determines what information should be exposed

In PyTorch:

```python
nn.LSTM(
    input_size=1,
    hidden_size=16,
    batch_first=True
)
```

---

## 📊 Dataset

A small synthetic dataset was created for binary sequence classification.

The model learns whether the sum of a sequence is greater than `15`.

Examples:

```text
[1, 2, 3, 4, 5] → 0
[5, 5, 5, 5, 5] → 1
[2, 2, 2, 2, 2] → 0
[4, 5, 4, 5, 4] → 1
```

The labels are:

```python
y = [0, 1, 0, 1]
```

This simple dataset was intentionally chosen so that the focus remained on understanding RNN and LSTM mechanics.

---

## 🏗️ Model Architecture

### RNN Classifier

```text
Input
  ↓
RNN
  ↓
Final Hidden State
  ↓
Linear Layer
  ↓
Binary Output
```

The RNN uses:

```text
Input Size  = 1
Hidden Size = 16
Output Size = 1
```

The final hidden state has 16 features and is passed into:

```python
nn.Linear(16, 1)
```

---

### LSTM Classifier

```text
Input
  ↓
LSTM
  ↓
Final Hidden State
  ↓
Linear Layer
  ↓
Binary Output
```

The LSTM maintains:

```text
Hidden State
Cell State
```

while processing the sequence.

---

## ⚙️ Training

The model was trained using:

```python
nn.BCEWithLogitsLoss()
```

This loss function is suitable for binary classification and internally combines:

```text
Sigmoid
+
Binary Cross Entropy
```

The optimizer used was:

```python
torch.optim.Adam
```

with a learning rate of:

```text
0.01
```

The training process follows:

```text
Input
  ↓
RNN / LSTM
  ↓
Logits
  ↓
BCEWithLogitsLoss
  ↓
Backpropagation
  ↓
Adam Optimizer
  ↓
Updated Weights
```

---

## 🔍 Prediction

During evaluation, the model produces logits.

These logits are converted into probabilities using:

```python
torch.sigmoid(logits)
```

A threshold of `0.5` is then used to obtain the final class:

```python
predictions = (probabilities >= 0.5).float()
```

The final predictions successfully matched the training labels for the simple dataset.

---

## 📐 Important Tensor Shapes

### Input

```text
[4, 5, 1]
```

Meaning:

```text
4 samples
5 timesteps
1 feature
```

### RNN/LSTM Output

```text
[4, 5, 16]
```

Meaning:

```text
4 samples
5 timesteps
16 hidden features
```

### Hidden State

```text
[1, 4, 16]
```

Meaning:

```text
1 layer
4 samples
16 hidden features
```

After:

```python
hidden.squeeze(0)
```

the shape becomes:

```text
[4, 16]
```

This can then be passed into:

```python
nn.Linear(16, 1)
```

producing:

```text
[4, 1]
```

---

## 🆚 RNN vs LSTM

| Feature                | RNN            | LSTM      |
| ---------------------- | -------------- | --------- |
| Hidden State           | ✅              | ✅         |
| Cell State             | ❌              | ✅         |
| Handles sequences      | ✅              | ✅         |
| Long-term dependencies | More difficult | Better    |
| Gating mechanism       | ❌              | ✅         |
| PyTorch layer          | `nn.RNN`       | `nn.LSTM` |

The main conceptual difference is that an LSTM has an additional **cell state** and uses gates to control what information is remembered or forgotten.

---

## 🛠️ Technologies Used

* Python
* PyTorch
* Tensor operations
* `torch.nn`
* `torch.optim`
* RNN
* LSTM

---

## 📁 Project Structure

```text
08_RNN/
│
├── rnn.py
└── README.md
```

---

## 🎯 Key Takeaways

Through this project, I learned:

* How sequential data is represented in PyTorch
* The meaning of `[batch, sequence, features]`
* How an RNN processes data timestep by timestep
* What a hidden state represents
* The difference between RNN output and final hidden state
* How `hidden_size` affects tensor dimensions
* How to build an RNN classifier using PyTorch
* How to train an RNN using backpropagation
* How `BCEWithLogitsLoss` is used for binary classification
* How to convert logits into probabilities
* Why traditional RNNs can struggle with long-term dependencies
* How LSTMs use hidden and cell states
* The basic role of LSTM gates
* The difference between RNNs and LSTMs

---

## 🚀 Future Learning

Possible extensions of this project include:

* GRU networks
* Bidirectional RNNs and LSTMs
* Stacked LSTMs
* Word embeddings
* NLP sequence classification
* Attention mechanisms
* Transformers

The next major step is understanding **attention and Transformer architectures**, which are fundamental to modern deep learning and NLP systems.
