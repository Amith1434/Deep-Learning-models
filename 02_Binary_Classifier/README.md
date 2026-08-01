# Day 2 — Binary Classification with PyTorch

A simple binary classification neural network built with PyTorch to understand the complete training and evaluation workflow for classification problems.

## Project Overview

This project trains a small neural network to classify points from the `make_circles` dataset into one of two classes.

The main goal of this project was not model complexity, but understanding the PyTorch binary classification pipeline:

- Preparing data for PyTorch
- Creating train/test splits
- Moving tensors and models to GPU
- Building a neural network with `nn.Module`
- Understanding logits
- Using `BCEWithLogitsLoss`
- Training using backpropagation
- Converting logits into class predictions
- Evaluating the model using accuracy

## Dataset

The dataset was generated using:

`sklearn.datasets.make_circles`

It contains two numerical input features and a binary target:

- Class `0`
- Class `1`

The circular structure makes the problem non-linear, allowing the project to demonstrate why a non-linear activation function such as ReLU is useful.

## Model Architecture

The neural network uses the following architecture:

Input (2 features)
→ Linear(2, 8)
→ ReLU
→ Linear(8, 1)
→ Raw Logit

The final layer does not use Sigmoid because `BCEWithLogitsLoss` expects raw logits.

## Loss Function

The project uses:

`nn.BCEWithLogitsLoss()`

This combines the mathematical operations of sigmoid and binary cross-entropy loss in a numerically stable implementation.

During training:

Model → Logits → BCEWithLogitsLoss

During evaluation:

Model → Logits → Sigmoid → Threshold → Prediction

## Optimizer

Adam optimizer was used:

- Optimizer: Adam
- Learning rate: 0.01
- Epochs: 200

## Training Process

Each training iteration follows the standard PyTorch workflow:

1. Perform a forward pass
2. Calculate the loss
3. Clear previous gradients
4. Perform backpropagation
5. Update model parameters

```python
logits = model(train_x)
loss = loss_fn(logits, train_y)

optimizer.zero_grad()
loss.backward()
optimizer.step()