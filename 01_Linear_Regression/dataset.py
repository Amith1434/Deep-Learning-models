import torch
torch.manual_seed(42)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
x = torch.arange(1,11,dtype=torch.float32).view(-1,1).to(device)
noise = torch.randn_like(x) *2
y = 3 *x +2 +noise
