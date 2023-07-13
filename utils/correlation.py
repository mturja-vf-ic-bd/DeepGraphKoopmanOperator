import torch


def pearson2(X):
    m = torch.mean(X, dim=-1, keepdim=True)
    s = torch.std(X, dim=-1, keepdim=True)
    X = X - m
    return torch.matmul(X, X.transpose(-1, -2)) \
           / torch.matmul(s, s.transpose(-1, -2)) \
           / (X.shape[-1] - 1)


def pearson(A):
    eps = 1e-5
    m = A.mean(dim=-1, keepdim=True)
    s = A.std(dim=-1, keepdim=True)
    N = A.size(-1)
    A = A - m
    cov = (A @ A.transpose(-1, -2)) / (N - 1)
    corr = cov / (s @ s.transpose(-1, -2) + eps)
    return corr - torch.diag_embed(torch.diagonal(corr, dim1=-2, dim2=-1))