import torch
from typing import List, Tuple
import numpy as np
import pandas as pd


def feature_poisoning(
    X_clients: List[torch.Tensor],
    n_corrupt: int = 5,
    noise_std: float = 0.5,
) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
    """Add heavy Gaussian noise to n_corrupt random columns per client. If a column is binary (only 0 and 1), flip the values instead of adding noise."""
    d = X_clients[0].shape[1]
    q_clients = []
    X_corr = []
    for X in X_clients:
        # Determine which columns are binary (only 0 and 1)
        is_binary = []
        for col in range(d):
            unique_vals = torch.unique(X[:, col])
            if len(unique_vals) == 2 and set(unique_vals.tolist()) == {0, 1}:
                is_binary.append(True)
            else:
                is_binary.append(False)
        is_binary = np.array(is_binary)
        # Randomly select columns to corrupt
        cols = np.random.choice(d, n_corrupt, replace=False)
        Xc = X.clone()
        for col in cols:
            if is_binary[col]:
                # Flip 0 <-> 1
                Xc[:, col] = 1.0 - Xc[:, col]
            else:
                Xc[:, col] += noise_std * torch.randn_like(Xc[:, col])
        X_corr.append(Xc)
        q = torch.ones(d)
        q[cols] = 0.0  # trust = 0 for corrupted
        q_clients.append(q)
    return X_corr, q_clients