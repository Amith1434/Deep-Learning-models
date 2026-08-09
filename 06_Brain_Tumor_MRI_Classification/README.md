# 06 - Brain Tumor MRI Classification

## Overview

In this project, I built a CNN using PyTorch to classify brain MRI images into four classes.

The main concepts covered were:

- Loading datasets with `ImageFolder`
- Image preprocessing with `transforms`
- Batching with `DataLoader`
- CNN architecture
- Convolution kernels and channels
- ReLU activation
- MaxPooling
- Flattening
- Cross-Entropy Loss
- Adam optimizer
- GPU training
- Training and evaluation loops

## Dataset

The dataset contains four classes:

```text
Glioma
Meningioma
No Tumor
Pituitary