import numpy as np
import torch
from typing import List, Tuple

def make_synthetic_data(
    n_clients: int = 10,
    samples_per_client: int = 1_000,
    d: int = 20,
    noise_std: float = 0.1,
    samples_test: int = 5_000,
) -> Tuple[List[torch.Tensor], List[torch.Tensor], torch.Tensor, torch.Tensor]:
    """Generate logistic-regression data with a ground-truth weight."""

    # Generate ground-truth weight
    w_true = torch.randn(d) / np.sqrt(d) #It creates a random weight vector of size d normalized ot maintain stable the variance (d, )
    X_all, y_all = [], []
    for _ in range(n_clients):
        x = torch.randn(samples_per_client, d) #It creates a random feature matrix (samples_per_client, d)
        logit = (x @ w_true) + noise_std * torch.randn(samples_per_client) #It creates a logit vector of size (samples per client, d) adding gaussian noise
        y = (torch.sigmoid(logit) > 0.5).float()
        X_all.append(x)
        y_all.append(y)

    # Global test set (hold-out)
    X_test = torch.randn(samples_test, d)
    y_test = (torch.sigmoid(X_test @ w_true) > 0.5).float()

    return X_all, y_all, X_test, y_test