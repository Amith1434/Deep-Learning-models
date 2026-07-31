# Project 1 - Linear Regression from Scratch (PyTorch)

## Objective
Build a Linear Regression model from scratch using PyTorch to understand the complete training process, including autograd, gradient descent, and optimization.

## Dataset
A synthetic dataset was generated using:

y = 3x + 2 + noise

where random Gaussian noise was added to simulate real-world data.

## Concepts Learned
- Tensors and GPU (CUDA)
- `requires_grad`
- Computational Graph
- Forward Pass
- Mean Squared Error (MSE)
- Backpropagation (`loss.backward()`)
- Manual Gradient Descent
- `torch.optim.SGD`
- Gradient Accumulation
- `optimizer.zero_grad()`

## Training Pipeline
1. Forward Pass
2. Compute MSE Loss
3. Backpropagation
4. Update Parameters
5. Clear Gradients
6. Repeat

## Results
- Successfully trained a Linear Regression model.
- Learned parameters close to the original equation (`y = 3x + 2`).
- Training loss decreased significantly over epochs.

## Files
- `dataset.py` – Generates the dataset.
- `handwritten_model.py` – Implements Linear Regression from scratch.
- `README.md` – Project documentation.

## Key Takeaways
- Understood how PyTorch builds a computational graph.
- Learned how gradients are computed automatically.
- Implemented gradient descent manually before using `torch.optim.SGD`.
- Trained the model on the GPU using CUDA.

## Next Project
**Logistic Regression** – Sigmoid, Binary Cross Entropy Loss, Binary Classification, and `nn.Module`.