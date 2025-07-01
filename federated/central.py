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
    model=None,  # new argument: can be tensor (logreg) or nn.Module (NN)
) -> Tuple[object, object]:
    """Return final (w,g). For vanilla FedAvg, g=ones or None. Model can be tensor or nn.Module."""
    d = X_clients[0].shape[1]
    n_clients = len(X_clients)
    n_samples = torch.tensor([len(x) for x in X_clients], dtype=torch.float32)

    # Model initialization for each client
    if model is None:
        # Default: logistic regression
        w_global = torch.zeros(d)
        g_global = torch.ones(d)
    elif isinstance(model, torch.Tensor):
        w_global = model.clone().detach()
        g_global = torch.ones(d)
    else:
        # Assume nn.Module
        w_global = model
        g_global = torch.ones(d)

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
            else:
                w_k = local_train_fedavg(
                    X_clients[k],
                    y_clients[k],
                    w_global,
                    lr=lr,
                    epochs=local_epochs,
                )
                w_updates.append(w_k)

        # ---- server aggregation (simple weighted average for tensors, or averaging state_dict for NN) ----
        if isinstance(w_updates[0], torch.Tensor):
            stacked_w = torch.stack(w_updates)
            w_global = (stacked_w.T @ n_samples / n_samples.sum()).T
        else:
            # For neural networks, average parameters
            new_state_dict = {}
            for key in w_updates[0].state_dict().keys():
                avg_param = torch.stack([w.state_dict()[key].float() for w in w_updates], dim=0).mean(dim=0)
                new_state_dict[key] = avg_param
            w_global.load_state_dict(new_state_dict)

        if use_frogdq:
            stacked_g = torch.stack(g_updates)
            g_global = (stacked_g.T @ n_samples / n_samples.sum()).T

    return w_global, g_global