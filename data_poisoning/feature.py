import torch
from typing import List, Tuple
import numpy as np
import pandas as pd


def feature_poisoning(
    X_clients: List[torch.Tensor],
    n_corrupt: int = 5,
    noise_std: float = 0.1,
    rnd_per_client: bool = False,
    indexes_to_corrupt: list = [],
    missing_value: bool = False
) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
    """Add heavy Gaussian noise to n_corrupt random columns per client."""
    d = X_clients[0].shape[1]
    q_clients = []
    X_corr = []

    if not rnd_per_client:
        if not indexes_to_corrupt:
            cols = np.random.choice(d, n_corrupt, replace=False)
        else:
            cols = indexes_to_corrupt
    
    print(f"Columns poisoned: {cols} ")

    for X in X_clients:
        
        # Randomly select columns to corrupt
        if rnd_per_client:
            cols = np.random.choice(d, n_corrupt, replace=False)

        Xc = X.clone()
        for col in cols:
            if missing_value:
                Xc[:, col] = 0
            else:
                Xc[:, col] += noise_std * torch.randn_like(Xc[:, col])
        X_corr.append(Xc)
        q = torch.ones(d)
        q[cols] = 0.0  # trust = 0 for corrupted
        q_clients.append(q)
        
    return X_corr, q_clients