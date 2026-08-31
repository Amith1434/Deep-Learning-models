import torch
from torch import nn

class SelfAttentionModel(nn.Module):
    def __init__(self,vocab_size,embedding_dim):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size,embedding_dim)
        self.query = nn.Linear(embedding_dim,embedding_dim)
        self.key = nn.Linear(embedding_dim,embedding_dim)
        self.value = nn.Linear(embedding_dim,embedding_dim)
    def forward(self,x):
        x=self.embedding(x)

        Q = self.query(x)
        K = self.key(x)
        V = self.value(x)

        print("Embedding :",x.shape)
        print("Q:", Q.shape)
        print("K:",K.shape)
        print("V:",V.shape)
        scores = torch.matmul(Q,K.transpose(-2,-1))
        print("Scores:",scores.shape)
        return Q, K, V


if __name__ == "__main__":
    vocab_size = 10000
    embedding_dim = 128
    model = SelfAttentionModel(vocab_size,embedding_dim)
    x=  torch.randint(0,vocab_size,(4,20))
    Q, K, V = model(x)
    print("Input shape :",x.shape)