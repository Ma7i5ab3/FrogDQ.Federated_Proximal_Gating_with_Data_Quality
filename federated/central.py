import torch
from typing import List, Tuple
from .local import local_train_fedavg, local_train_frogdq

def run_federated_training(
    X_clients: List[torch.Tensor],
    y_clients: List[torch.Tensor],
    use_frogdq: bool = False,
    n_rounds: int = 100,
    local_epochs: int = 1,
    lr: float = 0.01,
    mu: float = 0.1,
    q_clients: List[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Return final (w,g).  For vanilla FedAvg, g=ones."""
    d = X_clients[0].shape[1]
    w_global = torch.zeros(d)
    g_global = torch.ones(d)
    n_clients = len(X_clients)
    n_samples = torch.tensor([len(x) for x in X_clients], dtype=torch.float32)

    for rnd in range(n_rounds):
        w_updates, g_updates = [], []
        for k in range(n_clients):
            if use_frogdq:
                w_k, g_k = local_train_frogdq(
                    X_clients[k],
                    y_clients[k],
                    w_global,
                    g_global,
                    q_clients[k],
                    mu=mu,
                    lr=lr,
                    epochs=local_epochs,
                )
                w_updates.append(w_k)
                g_updates.append(g_k)
            else:  # plain FedAvg
                w_k = local_train_fedavg(
                    X_clients[k], y_clients[k], w_global, lr=lr, epochs=local_epochs
                )
                w_updates.append(w_k)

        # ---- server aggregation (simple weighted average) ----
        stacked_w = torch.stack(w_updates)
        w_global = (stacked_w.T @ n_samples / n_samples.sum()).T
        if use_frogdq:
            stacked_g = torch.stack(g_updates)
            g_global = (stacked_g.T @ n_samples / n_samples.sum()).T

    return w_global, g_global