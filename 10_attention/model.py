import torch
import torch.nn.functional as F

X= torch.tensor([
    [0.1,0.0,1.0,0.0],
    [0.0,1.0,0.0,1.0],
    [1.0,1.0,0.0,0.0]
])

W_Q = torch.randn(4,4)
W_K = torch.randn(4,4)
W_V = torch.randn(4,4)

Q = X @ W_Q
K = X @ W_K
V = X @ W_V

print("Q:\n",Q)
print("K:\n",K)
print("V:\n,",V)

scores = Q @ K.T

print("\nAttention scores: ",scores)

attention_weights = F.softmax(scores,dim=-1)

print("\nAttention weights: ",attention_weights)

output = attention_weights @ V
print("\nAttention output: ",output)

