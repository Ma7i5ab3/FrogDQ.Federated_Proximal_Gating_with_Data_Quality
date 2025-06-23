from data_poisoning.feature import feature_poisoning
from data.synthetic.data_generation import make_synthetic_data
from federated.central import run_federated_training
from evaluation.evalutation import accuracy
import torch

if __name__ == "__main__":
    # ---------------- data ----------------
    K, N, D = 10, 1000, 20
    X_clients, y_clients, X_test, y_test = make_synthetic_data(
        n_clients=K, samples_per_client=N, d=D
    )

    # also make a corrupted copy + trust vectors
    X_corr_clients, q_clients = feature_poisoning(X_clients, n_corrupt=5)

    # ---------------- experiments ----------------
    print("===> CLEAN DATA")
    w_fedavg, _ = run_federated_training(
        X_clients, y_clients, use_frogdq=False, n_rounds=10, local_epochs=1, lr=0.2
    )
    acc_fedavg_clean = accuracy(X_test, y_test, w_fedavg)
    print(f"FedAvg  accuracy: {acc_fedavg_clean:6.3f}")

    w_frogdq, g_frogdq = run_federated_training(
        X_clients,
        y_clients,
        use_frogdq=True,
        n_rounds=10,
        local_epochs=1,
        lr=0.2,
        mu=1.0,
        q_clients=[torch.ones(D) for _ in range(K)],  # all features trusted
    )
    acc_frog_clean = accuracy(X_test, y_test, w_frogdq, g_frogdq)
    print(f"FroG-DQ accuracy: {acc_frog_clean:6.3f}")

    print("\n===> CORRUPTED DATA")
    # FedAvg on noisy clients
    w_fedavg_noise, _ = run_federated_training(
        X_corr_clients,
        y_clients,
        use_frogdq=False,
        n_rounds=100,
        local_epochs=1,
        lr=0.01,
    )
    acc_fedavg_noise = accuracy(X_test, y_test, w_fedavg_noise)
    print(f"FedAvg  accuracy: {acc_fedavg_noise:6.3f}")

    # FroG-DQ with trust scores
    w_frog_noise, g_frog_noise = run_federated_training(
        X_corr_clients,
        y_clients,
        use_frogdq=True,
        n_rounds=100,
        local_epochs=1,
        lr=0.01,
        mu=0.1,
        q_clients=q_clients,
    )
    acc_frog_noise = accuracy(X_test, y_test, w_frog_noise, g_frog_noise)
    print(f"FroG-DQ accuracy: {acc_frog_noise:6.3f}")