import torch
from dataset import x,y, device
w = torch.randn(1,device=device, requires_grad = True)
b = torch.randn(1,device = device, requires_grad = True)

epochs = 100
lr = 0.01
optimizer = torch.optim.SGD([w,b],lr = 0.01)
for epoch in range(epochs):
    y_pred = w*x +b
    loss = ((y_pred - y) ** 2).mean()
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
    
    if epoch %10 == 0:
        print(f"Epoch {epoch} : Loss = {loss.item() : .4f}")

print("Fianl Weight : ", w.item())
print("Final Weight : ", b.item())