import torch
from torch import nn, optim
from typing import Tuple
from federated.model import SimpleNN

def local_train_fedavg(
    X: torch.Tensor,
    y: torch.Tensor,
    model,
    lr: float = 0.01,
    epochs: int = 1,
) -> object:
    """Single-client FedAvg update (logistic regression or neural network, SGD)."""
    if isinstance(model, torch.Tensor):
         # Logistic regression (FedAvg original, w[0] is intercept)
        w = model.clone().detach().requires_grad_(True)
        opt = optim.SGD([w], lr=lr)
        loss_fn = nn.BCEWithLogitsLoss()
        for _ in range(epochs):
            opt.zero_grad()
            logits = w[0] + X @ w[1:]
            loss = loss_fn(logits, y)
            loss.backward()
            opt.step()
        return w.detach()


def local_train_frogdq(
    X: torch.Tensor,
    y: torch.Tensor,
    model,
    g_global: torch.Tensor,
    q: torch.Tensor,
    mu: float = 0.1,
    lr: float = 0.01,
    epochs: int = 1,
) -> tuple:
    """Single-client FroG-DQ update (prox only on g, supports logistic regression or neural network)."""
    inv_q = (1.0 - q)
    if isinstance(model, torch.Tensor):
        # Logistic regression with intercept (w[0] is intercept, w[1:] are feature weights)
        w = model.clone().detach().requires_grad_(True)
        g = g_global.clone().detach().requires_grad_(True)
        #opt = optim.SGD([w, g], lr=lr)
        loss_fn = nn.BCEWithLogitsLoss()
        for i in range(epochs):
            # Forward pass
            logits = (X * g) @ w[1:] + w[0]   # g is gating mask
            data_loss = loss_fn(logits, y)
            prox = 0.5 * mu * torch.sum(inv_q * (g - g_global) ** 2)
            loss = data_loss + prox

            # Backward pass
            loss.backward()

            # Manual SGD update
            with torch.no_grad():
                w -= lr * w.grad
                g -= lr * g.grad

            # Zero gradients manually
            w.grad.zero_()
            g.grad.zero_()
        return w.detach(), g.detach()
    


def local_train_fedprox(
    X: torch.Tensor,
    y: torch.Tensor,
    model,
    mu: float = 0.1,
    lr: float = 0.01,
    epochs: int = 1,
) -> object:
    """Single-client FedProx update (proximal term on model parameters, supports logistic regression or neural network). Assumes model is initialized as a clone of the global model."""
    if isinstance(model, torch.Tensor):
        # Logistic regression (FedProx)
        w = model.clone().detach().requires_grad_(True)
        w_global = model.clone().detach()
        opt = optim.SGD([w], lr=lr)
        loss_fn = nn.BCEWithLogitsLoss()
        for _ in range(epochs):
            opt.zero_grad()
            logits = w[0] + X @ w[1:]
            data_loss = loss_fn(logits, y)
            prox = 0.5 * mu * torch.sum((w - w_global) ** 2)
            loss = data_loss + prox
            loss.backward()
            opt.step()
        return w.detach()
    
 