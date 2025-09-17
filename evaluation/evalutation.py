import torch
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, roc_auc_score, balanced_accuracy_score
from typing import Tuple

def eval(
    X: torch.Tensor,
    y: torch.Tensor,
    w,
    g: torch.Tensor = None,
) -> Tuple[float, float]:
    # Logistic regression
    if g is None:
        g = torch.ones_like(w[1:])

    logits = (X * g) @ w[1:] + w[0]
    probs = torch.sigmoid(logits)
    preds = probs > 0.5

    acc = accuracy_score(y.numpy(), preds.numpy())
    loss = F.binary_cross_entropy(probs, y.float())
    auc_roc = roc_auc_score(y_true=y.numpy(), y_score=probs.numpy())
    balanced_accuracy = balanced_accuracy_score(y.numpy(), preds.numpy())

    return auc_roc, acc, balanced_accuracy, loss.item()