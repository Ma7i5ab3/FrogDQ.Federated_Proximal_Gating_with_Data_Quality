import torch
from sklearn.metrics import accuracy_score

def accuracy(
    X: torch.Tensor,
    y: torch.Tensor,
    w,
    g: torch.Tensor = None,
) -> float:
    # Logistic regression 
    if g is None:
        g = torch.ones_like(w[1:]) 
    preds = torch.sigmoid((X * g) @ w[1:] + w[0]) > 0.5
    return accuracy_score(y.numpy(), preds.numpy())