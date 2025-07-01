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
         # Logistic regression (FedAvg original)
        w = model.clone().detach().requires_grad_(True)
        opt = optim.SGD([w], lr=lr)
        loss_fn = nn.BCEWithLogitsLoss()
        for _ in range(epochs):
            opt.zero_grad()
            logits = X @ w
            loss = loss_fn(logits, y)
            loss.backward()
            opt.step()
        return w.detach()
    elif isinstance(model, nn.Module):
        # Neural network
        model = model.train()
        opt = optim.SGD(model.parameters(), lr=lr)
        # Use CrossEntropyLoss for multi-class, BCE for binary
        if y.ndim == 1 or y.shape[1] == 1:
            loss_fn = nn.CrossEntropyLoss() if model.fc3.out_features > 1 else nn.BCELoss()
        else:
            loss_fn = nn.BCELoss()
        for _ in range(epochs):
            opt.zero_grad()
            outputs = model(X)
            y_input = y
            if isinstance(loss_fn, nn.CrossEntropyLoss):
                y_input = y_input.long()
                if y_input.ndim > 1:
                    y_input = y_input.squeeze()
            loss = loss_fn(outputs, y_input)
            loss.backward()
            opt.step()
        return model
    else:
        raise ValueError("Model must be either a torch.Tensor (logistic regression) or nn.Module (neural network)")


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
        # Logistic regression
        w = model.clone().detach().requires_grad_(True)
        g = g_global.clone().detach().requires_grad_(True)
        opt = optim.SGD([w, g], lr=lr)
        loss_fn = nn.BCEWithLogitsLoss()
        for _ in range(epochs):
            opt.zero_grad()
            logits = (X * g) @ w
            data_loss = loss_fn(logits, y)
            prox = 0.5 * mu * torch.sum(inv_q * (g - g_global) ** 2)
            loss = data_loss + prox
            loss.backward()
            opt.step()
        return w.detach(), g.detach()
    elif isinstance(model, nn.Module):
        # Neural network
        model = model.train()
        g = g_global.clone().detach().requires_grad_(True)
        opt = optim.SGD(list(model.parameters()) + [g], lr=lr)
        if y.ndim == 1 or y.shape[1] == 1:
            loss_fn = nn.CrossEntropyLoss() if model.fc3.out_features > 1 else nn.BCELoss()
        else:
            loss_fn = nn.BCELoss()
        for _ in range(epochs):
            opt.zero_grad()
            X_mod = X * g
            outputs = model(X_mod)
            y_input = y
            if isinstance(loss_fn, nn.CrossEntropyLoss):
                y_input = y_input.long()
                if y_input.ndim > 1:
                    y_input = y_input.squeeze()
            data_loss = loss_fn(outputs, y_input)
            prox = 0.5 * mu * torch.sum(inv_q * (g - g_global) ** 2)
            loss = data_loss + prox
            loss.backward()
            opt.step()
        return model, g.detach()
    else:
        raise ValueError("Model must be either a torch.Tensor (logistic regression) or nn.Module (neural network)")
