import torch
from typing import List, Tuple
import numpy as np

def feature_poisoning(
    X_clients: List[torch.Tensor],
    n_corrupt: int = 5,
    noise_std: float = 5.0,
) -> Tuple[List[torch.Tensor], List[torch.Tensor], List[torch.Tensor]]:
    
    """Add heavy Gaussian noise to n_corrupt random columns per client."""
    
    d = X_clients[0].shape[1]
    q_clients = []
    X_corr = []
    for X in X_clients:
        cols = np.random.choice(d, n_corrupt, replace=False)
        Xc = X.clone()
        Xc[:, cols] += noise_std * torch.randn_like(Xc[:, cols])
        X_corr.append(Xc)
        q = torch.ones(d)
        q[cols] = 0.0  # trust = 0 for corrupted
        q_clients.append(q)
    return X_corr, q_clients