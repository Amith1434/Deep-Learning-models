# 05 - CNN Evaluation

## Overview

In this project, I evaluated the CNN built for Fashion-MNIST classification.

The main focus was understanding:
- `model.eval()`
- `torch.inference_mode()`
- Test loss
- Test accuracy
- Comparing training and test performance

## Model

The CNN consists of:

```text
Conv2d → ReLU → MaxPool
→ Conv2d → ReLU → MaxPool
→ Flatten → Linear