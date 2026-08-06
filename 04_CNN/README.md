# Day 4 — Convolutional Neural Network (CNN)

Built a CNN using PyTorch on the **Fashion-MNIST** dataset.

## CNN Flow

```
Image → Conv2d → ReLU → MaxPool
      → Conv2d → ReLU → MaxPool
      → Flatten → Linear → Prediction
```

## Conv2d

```python
nn.Conv2d(
    in_channels=1,
    out_channels=32,
    kernel_size=3,
    padding=1
)
```

- **in_channels** → channels entering the layer (`1` for grayscale).
- **out_channels** → number of filters / feature maps produced.
- **kernel_size** → size of the sliding filter (`3` = 3×3).
- **padding** → adds pixels around the image. `padding=1` keeps the spatial size unchanged for a 3×3 kernel.

Filters are learned automatically during training through backpropagation.

### Channel Rule

The output channels of one convolution become the input channels of the next.

```python
Conv2d(1, 32, ...)
Conv2d(32, 64, ...)
```

## ReLU

```python
nn.ReLU()
```

Introduces non-linearity:

```
ReLU(x) = max(0, x)
```

It changes values but not the tensor shape.

## MaxPool2d

```python
nn.MaxPool2d(kernel_size=2)
```

Keeps the maximum value from each 2×2 region and reduces spatial size:

```
28×28 → 14×14 → 7×7
```

Default:

```
stride = kernel_size
padding = 0
```

Padding can be used with MaxPool, but isn't needed here because we intentionally want to reduce the dimensions.

## Flatten + Linear

After the convolution blocks:

```
[64, 7, 7]
     ↓ Flatten
3136 features
     ↓ Linear
10 logits
```

```python
nn.Flatten()
nn.Linear(64 * 7 * 7, 10)
```

The 10 logits represent the scores for the **10 Fashion-MNIST classes**.

## Model Shape

```
[1, 28, 28]
      ↓ Conv + ReLU + Pool
[32, 14, 14]
      ↓ Conv + ReLU + Pool
[64, 7, 7]
      ↓ Flatten
[3136]
      ↓ Linear
[10]
```

## Key Takeaways

- CNN filters learn image features automatically.
- `padding` can preserve spatial dimensions.
- Pooling reduces spatial dimensions.
- Previous `out_channels` = next `in_channels`.
- `Flatten` prepares CNN features for a Linear layer.
- Final output contains one logit per class.