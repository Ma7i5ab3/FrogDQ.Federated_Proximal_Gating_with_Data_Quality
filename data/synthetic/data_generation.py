import numpy as np
import torch
from typing import List, Tuple

def make_synthetic_data(
    n_clients: int = 10,
    samples_per_client: int = 1_000,
    d: int = 20,
    noise_std: float = 0.1,
    samples_test: int = 5000,
    samples_validation: int = 1000
) -> Tuple[List[torch.Tensor], List[torch.Tensor], torch.Tensor, torch.Tensor]:
    """Generate logistic-regression data with a ground-truth weight.

    Parameters
    ----------
    n_clients : int, optional
        Number of clients (datasets) to generate.
    samples_per_client : int, optional
        Number of samples per client. 
    d : int, optional
        Number of features (dimensionality) for each sample. 
    noise_std : float, optional
        Standard deviation of Gaussian noise added to the logits.
    samples_test : int, optional
        Number of samples in the global test set. 

    Returns
    -------
    Tuple[List[torch.Tensor], List[torch.Tensor], torch.Tensor, torch.Tensor]
        Tuple containing the training data (features and labels) for each client, and the global test set (features and labels).
    """

    # Generate ground-truth weight and intercept (bias)
    w_true = torch.randn(d) / np.sqrt(d) #It creates a random weight vector of size d normalized to maintain stable the variance (d, )
    w_0 = torch.randn(1).item()  # Intercept (bias term)
    X_all, y_all = [], []
    for _ in range(n_clients):
        x = torch.randn(samples_per_client, d) #It creates a random feature matrix (samples_per_client, d)
        logit = w_0 + (x @ w_true) + noise_std * torch.randn(samples_per_client) 
        y = (torch.sigmoid(logit) > 0.5).float()
        X_all.append(x)
        y_all.append(y)

    # Global test set (hold-out)
    X_test = torch.randn(samples_test, d)
    y_test = (torch.sigmoid(w_0 + X_test @ w_true) > 0.5).float()

    # Validation test set
    X_val = torch.randn(samples_validation, d)
    y_val = (torch.sigmoid(w_0 + X_val @ w_true) > 0.5).float()


    return X_all, y_all, X_test, y_test, X_val, y_val, w_true