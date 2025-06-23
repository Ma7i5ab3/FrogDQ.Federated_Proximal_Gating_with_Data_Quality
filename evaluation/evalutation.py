import torch
from sklearn.metrics import accuracy_score

def accuracy(
    X: torch.Tensor,
    y: torch.Tensor,
    w: torch.Tensor,
    g: torch.Tensor = None,
) -> float:
    if g is None:
        g = torch.ones_like(w)
    preds = torch.sigmoid((X * g) @ w) > 0.5
    return accuracy_score(y.numpy(), preds.numpy())