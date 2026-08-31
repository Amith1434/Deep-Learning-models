import torch
from torch import nn
import matplotlib.pyplot as plt

num_classes = 4
class CNNBackbone(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(
            in_channels = 3,
            out_channels = 16,
            kernel_size = 3,
            padding =1
        )
        self.pool = nn.MaxPool2d(2)
        self.box_head = nn.Linear(
            16*112*112,
            4
        )
        self.conf_head = nn.Linear(
            16*112*112,
            1
        )
        self.class_head = nn.Linear(
            16*112*112,num_classes
        )

    def forward(self,x):
        x = self.conv1(x)
        x = self.pool(x)

        x= x.flatten(1)

        box = self.box_head(x)
        confidence = self.conf_head(x)
        class_logits = self.class_head(x)

        return box,confidence,class_logits


if __name__ == "__main__":
    model = CNNBackbone()
    x = torch.zeros(1,3,224,224)
    x[:,:,50:150,60:160] = 1.0
    true_box = torch.tensor([
        [60.0,50.0,160.0,150.0]
    ])
    box,confidence,class_logits = model(x)
    print("Input : ",x.shape)
    print("Box : ",box.shape)
    print("Confidence : ",confidence.shape)
    print("Class : ", class_logits.shape)


box_loss_fn = nn.MSELoss()
box_loss = box_loss_fn(box,true_box)
print("Box Loss:", box_loss.item())


print("Before training:")
print("Predicted box:", box)
print("Loss:", box_loss.item())

optimizer = torch.optim.Adam(model.parameters(), lr = 0.001)
optimizer.zero_grad()
box_loss.backward()
optimizer.step()
box, confidence, class_logits = model(x)

loss_after = box_loss_fn(
    box,
    true_box
)

print("\nAfter one update:")
print("Predicted box:", box)
print("Loss:", loss_after.item())