07 - Transfer Learning with ResNet18

Overview

In this project, I used a pretrained ResNet18 model for brain tumor MRI classification.

The model classifies images into four classes:

Glioma

Meningioma

No Tumor

Pituitary

The main concepts covered were:

Transfer Learning

Pretrained models

ResNet18

Replacing the final layer

weights

weights.transforms()

ImageFolder

DataLoader

Freezing parameters

Unfreezing parameters

Feature Extraction

Fine-Tuning

CrossEntropyLoss

Adam optimizer

GPU training

Training loop

Evaluation loop

model.train()

model.eval()

torch.no_grad()

torch.argmax()

Dataset

Training images: 5600
Testing images: 1600

Classes:
0 → glioma
1 → meningioma
2 → notumor
3 → pituitary

Dataset structure:

train/
├── glioma/
├── meningioma/
├── notumor/
└── pituitary/

test/
├── glioma/
├── meningioma/
├── notumor/
└── pituitary/

1. Importing Libraries

import os
import certifi
import sys

import torch
from torch import nn
from torchvision import models
from torchvision import datasets
from torch.utils.data import DataLoader

2. Selecting the Device

device = "cuda" if torch.cuda.is_available() else "cpu"

print("Device:", device)

Move the model to the device:

model = model.to(device)

Move a batch to the device:

X = X.to(device)
y = y.to(device)

3. Loading a Pretrained ResNet18

weights = models.ResNet18_Weights.DEFAULT

model = models.resnet18(
    weights=weights
)

weights=... loads pretrained ImageNet weights.

Without pretrained weights:

model = models.resnet18(
    weights=None
)

4. Replacing the Final Layer

Original ResNet18:

512 features → 1000 ImageNet classes

Our model:

512 features → 4 MRI classes

Syntax:

model.fc = nn.Linear(512, 4)

The four outputs represent:

0 → glioma
1 → meningioma
2 → notumor
3 → pituitary

The new FC layer is randomly initialized and must learn the new classification task.

5. Why Can We Use a Model Trained on Another Dataset?

The pretrained ResNet18 was trained on ImageNet.

It does NOT know what a glioma or meningioma is.

However, CNNs learn useful visual features such as:

Edges
 ↓
Textures
 ↓
Shapes
 ↓
Complex patterns

Many of these features can be reused for another image problem.

Therefore:

ImageNet
   ↓
Pretrained ResNet18
   ↓
Reuse learned visual features
   ↓
Adapt to MRI dataset

This is Transfer Learning.

6. Pretrained Model Preprocessing

Use the preprocessing associated with the selected weights:

transform = weights.transforms()

Then:

Image
 ↓
Resize / Crop
 ↓
Tensor
 ↓
Normalize
 ↓
ResNet18

7. Loading the Dataset

train_dataset = datasets.ImageFolder(
    root=train_path,
    transform=transform
)

test_dataset = datasets.ImageFolder(
    root=test_path,
    transform=transform
)

Check classes:

print(train_dataset.classes)

Check class-to-index mapping:

print(train_dataset.class_to_idx)

Check dataset size:

print(len(train_dataset))
print(len(test_dataset))

8. Creating DataLoaders

train_dataloader = DataLoader(
    train_dataset,
    batch_size=32,
    shuffle=True
)

test_dataloader = DataLoader(
    test_dataset,
    batch_size=32,
    shuffle=False
)

Important

shuffle=True

for training.

shuffle=False

for testing.

9. Freezing Parameters

To freeze the entire pretrained model:

for param in model.parameters():
    param.requires_grad = False

A frozen parameter does not get updated during training.

10. Unfreezing the FC Layer

To train only the classifier:

for param in model.fc.parameters():
    param.requires_grad = True

Check trainable parameters:

for name, param in model.named_parameters():
    if param.requires_grad:
        print(name)

Expected output:

fc.weight
fc.bias

11. Feature Extraction

Feature Extraction means:

Pretrained backbone → Frozen
New FC layer       → Trainable

Code:

for param in model.parameters():
    param.requires_grad = False

for param in model.fc.parameters():
    param.requires_grad = True

Architecture:

ResNet18
├── Early layers → Frozen
├── Layer1       → Frozen
├── Layer2       → Frozen
├── Layer3       → Frozen
├── Layer4       → Frozen
└── FC           → Trainable

12. Feature Extraction Optimizer

Only the FC layer is trainable:

optimizer = torch.optim.Adam(
    model.fc.parameters(),
    lr=0.001
)

Result:

Feature Extraction Accuracy = 81.19%

This is our baseline.

13. Fine-Tuning

Fine-Tuning means allowing some pretrained layers to update.

In this project:

Early layers → Frozen
Layer4       → Trainable
FC           → Trainable

14. Freezing and Unfreezing for Fine-Tuning

Freeze everything:

for param in model.parameters():
    param.requires_grad = False

Unfreeze Layer4:

for param in model.layer4.parameters():
    param.requires_grad = True

Unfreeze FC:

for param in model.fc.parameters():
    param.requires_grad = True

Check trainable parameters:

for name, param in model.named_parameters():
    if param.requires_grad:
        print(name)

Expected trainable sections:

layer4
fc.weight
fc.bias

15. Why Fine-Tune Layer4?

A simplified view of CNN layers:

Early layers
→ Edges
→ Corners
→ Simple textures

Middle layers
→ More complex patterns

Deep layers
→ More task-specific features

MRI images are different from ImageNet images.

Therefore, allowing deeper layers to adapt can help the model learn MRI-specific features.

16. Fine-Tuning Optimizer

optimizer = torch.optim.Adam(
    filter(
        lambda p: p.requires_grad,
        model.parameters()
    ),
    lr=0.0001
)

We use a smaller learning rate because pretrained weights already contain useful knowledge.

The goal is:

Preserve useful pretrained knowledge
+
Adapt it to MRI

17. Loss Function

This is a four-class classification problem.

Use:

loss_fn = nn.CrossEntropyLoss()

The model produces four logits:

[glioma_score,
 meningioma_score,
 notumor_score,
 pituitary_score]

Example:

[2.4, -0.8, 0.5, 1.2]

The largest value is 2.4.

Therefore:

Prediction → glioma

18. Training Loop

epochs = 3

for epoch in range(epochs):

    model.train()

    train_loss = 0.0

    for X, y in train_dataloader:

        X = X.to(device)
        y = y.to(device)

        logits = model(X)

        loss = loss_fn(logits, y)

        optimizer.zero_grad()

        loss.backward()

        optimizer.step()

        train_loss += loss.item()

    train_loss /= len(train_dataloader)

    print(
        f"Epoch: {epoch + 1} | "
        f"Loss: {train_loss:.4f}"
    )

19. Training Loop Explained

Forward Pass

logits = model(X)

Images go through the model and predictions are produced.

Calculate Loss

loss = loss_fn(logits, y)

Measures how wrong the predictions are.

Clear Old Gradients

optimizer.zero_grad()

PyTorch accumulates gradients, so old gradients must be cleared.

Backpropagation

loss.backward()

Calculates gradients for trainable parameters.

Update Weights

optimizer.step()

Updates the trainable parameters.

20. Evaluation

Switch to evaluation mode:

model.eval()

Disable gradient calculation:

with torch.no_grad():

Complete evaluation:

model.eval()

correct = 0
total = 0

with torch.no_grad():

    for X, y in test_dataloader:

        X = X.to(device)
        y = y.to(device)

        logits = model(X)

        predictions = torch.argmax(
            logits,
            dim=1
        )

        total += y.size(0)

        correct += (
            predictions == y
        ).sum().item()

accuracy = correct / total

print(
    f"Test Accuracy: {accuracy * 100:.2f}%"
)

21. Important Evaluation Syntax

Evaluation mode

model.eval()

Used when testing the model.

Disable gradients

with torch.no_grad():

Used because we are not training.

Get predicted class

torch.argmax(
    logits,
    dim=1
)

Number of samples in a batch

y.size(0)

Do NOT use:

y.size()

when you need an integer count.

22. Feature Extraction vs Fine-Tuning

Feature Extraction

Backbone → Frozen
FC       → Trainable

Result:

81.19%

Fine-Tuning

Early layers → Frozen
Layer4       → Trainable
FC           → Trainable

Result:

92.75%

Comparison:

Method

Trainable Parts

Learning Rate

Accuracy

Feature Extraction

FC

0.001

81.19%

Fine-Tuning

Layer4 + FC

0.0001

92.75%

Improvement:

92.75 - 81.19 = 11.56 percentage points

23. Why Did Fine-Tuning Improve the Result?

Feature Extraction:

MRI
 ↓
Pretrained ResNet
 ↓
Frozen features
 ↓
FC
 ↓
Prediction

The feature extractor cannot change.

Fine-Tuning:

MRI
 ↓
Pretrained ResNet
 ↓
Layer4 adapts to MRI
 ↓
FC
 ↓
Prediction

Layer4 can now learn features that are more suitable for brain MRI images.

Therefore, accuracy improved:

81.19%
   ↓
Fine-Tuning
   ↓
92.75%

24. Transfer Learning vs Fine-Tuning

Transfer Learning is the broad concept:

Reuse knowledge from a pretrained model
for a new task.

Feature Extraction:

Freeze pretrained layers
+
Train new classifier

Fine-Tuning:

Freeze some pretrained layers
+
Train selected pretrained layers
+
Train new classifier

Conceptually:

Transfer Learning
├── Feature Extraction
└── Fine-Tuning

25. Important Questions to Remember

Was ResNet18 trained on brain tumors?

No. It was pretrained on ImageNet.

Does the original ResNet18 know the four tumor classes?

No.

Then why use it?

Because it learned useful visual features that can be reused.

Where does the knowledge about the four classes come from?

Our MRI training dataset.

What does the new FC layer learn?

It learns:

Visual features
      ↓
Glioma
Meningioma
No Tumor
Pituitary

What is Feature Extraction?

Freeze the pretrained backbone and train only the new classifier.

What is Fine-Tuning?

Allow some pretrained layers to update along with the new classifier.

Why use a smaller learning rate for Fine-Tuning?

To make small changes to useful pretrained weights instead of destroying them.

26. Main Mental Model

The pretrained model does NOT know brain tumors.

Instead:

              ImageNet Knowledge
                     ↓
MRI Image → Pretrained ResNet18
                     ↓
               Visual Features
                     ↓
                New FC Layer
                     ↓
                  4 Classes
          ┌────────┬────────┬────────┬────────┐
          ↓        ↓        ↓        ↓
       Glioma  Meningioma No Tumor Pituitary

Feature Extraction:

ResNet → Frozen
FC     → Trainable

Fine-Tuning:

Early ResNet layers → Frozen
Layer4              → Trainable
FC                  → Trainable

27. Key Syntax Cheat Sheet

Load pretrained model

weights = models.ResNet18_Weights.DEFAULT

model = models.resnet18(
    weights=weights
)

Replace final layer

model.fc = nn.Linear(512, 4)

Get preprocessing

transform = weights.transforms()

Load dataset

datasets.ImageFolder(
    root=path,
    transform=transform
)

Create DataLoader

DataLoader(
    dataset,
    batch_size=32,
    shuffle=True
)

Freeze

param.requires_grad = False

Unfreeze

param.requires_grad = True

Check trainable parameters

for name, param in model.named_parameters():
    if param.requires_grad:
        print(name)

Loss

nn.CrossEntropyLoss()

Adam

torch.optim.Adam(
    parameters,
    lr=0.001
)

GPU

device = "cuda" if torch.cuda.is_available() else "cpu"

Move to GPU

model.to(device)
X.to(device)
y.to(device)

Training mode

model.train()

Evaluation mode

model.eval()

Disable gradients

with torch.no_grad():

Prediction

torch.argmax(
    logits,
    dim=1
)

Loss value as Python number

loss.item()

28. Final Result

Feature Extraction
        ↓
81.19% Accuracy

Fine-Tuning Layer4 + FC
        ↓
92.75% Accuracy

Improvement:

11.56 percentage points

The main lesson:

A pretrained model does not need to be trained on exactly the same problem to be useful.

We reuse its learned visual representations and adapt them to our new task.