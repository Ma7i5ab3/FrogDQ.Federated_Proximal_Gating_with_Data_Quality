import torch
from typing import List, Tuple
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split


def homogeneous_splitting(
    df: pd.DataFrame,
    n_clients: int,
    label_col: str,
    test_size: float = 0.2,
    random_state: int = 42,
) -> Tuple[List[torch.Tensor], List[torch.Tensor], torch.Tensor, torch.Tensor]:
    """
    Split a DataFrame into N clients and a test set, returning lists of torch tensors.
    Assumes the label column is specified by label_col.

    Returns
    -------
    Tuple[List[torch.Tensor], List[torch.Tensor], torch.Tensor, torch.Tensor]
        Tuple containing the training data (features and labels) for each client, and the global test set (features and labels).
    
    """
    # Split into train/test
    train_df, test_df = train_test_split(df, test_size=test_size, random_state=random_state, shuffle=True)
    
    # Separate features and labels
    X_train = train_df.drop(columns=[label_col]).values
    y_train = train_df[label_col].values
    X_test = test_df.drop(columns=[label_col]).values
    y_test = test_df[label_col].values

    # Split train into N clients (as evenly as possible)
    X_clients = np.array_split(X_train, n_clients)
    y_clients = np.array_split(y_train, n_clients)

    # Convert to torch tensors
    X_clients = [torch.tensor(x, dtype=torch.float32) for x in X_clients]
    y_clients = [torch.tensor(y, dtype=torch.float32) for y in y_clients]
    X_test = torch.tensor(X_test, dtype=torch.float32)
    y_test = torch.tensor(y_test, dtype=torch.float32)

    return X_clients, y_clients, X_test, y_test