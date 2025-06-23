import torch
from torch import nn, optim
from typing import Tuple

def local_train_fedavg(
    X: torch.Tensor,
    y: torch.Tensor,
    w_global: torch.Tensor,
    lr: float = 0.01,
    epochs: int = 1,
) -> torch.Tensor:
    
    """Single-client FedAvg update (logistic regression, SGD)."""

    w = w_global.clone().detach().requires_grad_(True)
    opt = optim.SGD([w], lr=lr)
    loss_fn = nn.BCEWithLogitsLoss()

    for _ in range(epochs):
        opt.zero_grad()
        logits = X @ w
        loss = loss_fn(logits, y)
        loss.backward()
        opt.step()

    return w.detach()


def local_train_frogdq(
    X: torch.Tensor,
    y: torch.Tensor,
    w_global: torch.Tensor,
    g_global: torch.Tensor,
    q: torch.Tensor,
    mu: float = 0.1,
    lr: float = 0.01,
    epochs: int = 1,
) -> Tuple[torch.Tensor, torch.Tensor]:
    
    """Single-client FroG-DQ update (prox only on g)."""

    # clone → local copies that will be sent back
    w = w_global.clone().detach().requires_grad_(True)
    g = g_global.clone().detach().requires_grad_(True)
    opt = optim.SGD([w, g], lr=lr)
    loss_fn = nn.BCEWithLogitsLoss()
    inv_q = (1.0 - q)  # quality factor

    for _ in range(epochs):
        opt.zero_grad()
        logits = (X * g) @ w
        data_loss = loss_fn(logits, y)
        prox = 0.5 * mu * torch.sum(inv_q * (g - g_global) ** 2)
        loss = data_loss + prox
        loss.backward()
        opt.step()

    return w.detach(), g.detach()