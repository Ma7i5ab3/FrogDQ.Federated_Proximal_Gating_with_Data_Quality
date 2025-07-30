import torch
from typing import List, Tuple
from .local import local_train_fedavg, local_train_frogdq, local_train_fedprox
from evaluation.evalutation import accuracy
from data_statistics.Stats import Stats
import numpy as np

def run_federated_training(
    X_clients: List[torch.Tensor],
    y_clients: List[torch.Tensor],
    X_val: torch.Tensor,
    y_val: torch.Tensor,
    stats_client: Stats,
    fl_algorithm: str,
    dq_type: str,
    w_global: torch.Tensor,
    n_rounds: int = 100,
    local_epochs: int = 1,
    lr: float = 0.01,
    mu: float = 0.1,
    q_clients: List[torch.Tensor] = None,
) -> Tuple[object, object]:
   
    d = X_clients[0].shape[1]
    n_clients = len(X_clients)
    n_samples = torch.tensor([len(x) for x in X_clients], dtype=torch.float32)
    #w_global = torch.randn(d+1) / torch.sqrt(torch.tensor(d + 1.0)) #init values
    #w_global = torch.zeros(d+1)
    w_global = w_global
    g_global = torch.ones(d) #init values
    w_global_lst = [w_global]
    g_global_lst = [g_global]

    for rnd in range(n_rounds):
        w_updates, g_updates = [], []
        for k in range(n_clients):
            if fl_algorithm == 'frog':
                w_k, g_k = local_train_frogdq(
                    X=X_clients[k],
                    y=y_clients[k],
                    model=w_global,
                    g_global=g_global,
                    q=q_clients[k],
                    mu=mu,
                    lr=lr,
                    epochs=local_epochs,
                )
                w_updates.append(w_k)
                g_updates.append(g_k)
            elif fl_algorithm == 'fedavg':
                w_k = local_train_fedavg(
                    X=X_clients[k],
                    y=y_clients[k],
                    model=w_global,
                    lr=lr,
                    epochs=local_epochs,
                )
                w_updates.append(w_k)
            elif fl_algorithm == 'fedprox':
                w_k = local_train_fedprox(
                    X=X_clients[k],
                    y=y_clients[k],
                    model=w_global,
                    mu=mu,
                    lr=lr,
                    epochs=local_epochs,
                )
                w_updates.append(w_k)

        # ---- server aggregation (simple weighted average for tensors, or averaging state_dict for NN) ----
        if isinstance(w_updates[0], torch.Tensor):
            stacked_w = torch.stack(w_updates)
            w_global = (stacked_w.T @ n_samples / n_samples.sum()).T
            # Save w global aggregated values after the end of each round
            w_global_lst.append(w_global)

        if fl_algorithm == 'frog':
            stacked_g = torch.stack(g_updates)
            g_global = (stacked_g.T @ n_samples / n_samples.sum()).T
            # Save g global aggregated values after the end of each round
            g_global_lst.append(g_global)

        # Compute Accuracy on Validation test
        accuracy_val = accuracy(X=X_val, y=y_val, w=w_global, g=g_global if fl_algorithm == 'frog' else None)
        stats_client.set_accuracy_val(value=accuracy_val, algorithm=fl_algorithm, dq_type=dq_type)
        
    #Update stats_client
    stats_client.set_gate_updates(value=g_global_lst, dq_type=dq_type, algorithm=fl_algorithm)
    stats_client.set_weights_updates(value=w_global_lst, dq_type=dq_type, algorithm=fl_algorithm)

    return w_global, g_global