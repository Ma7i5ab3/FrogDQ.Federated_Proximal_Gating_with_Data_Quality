from data_distribution.distribution import homogeneous_splitting
from data_poisoning.feature import feature_poisoning
from data.synthetic.data_generation import make_synthetic_data
from federated.central import run_federated_training
from evaluation.evalutation import accuracy
from federated.model import SimpleNN
import torch
import argparse
import os
import yaml
import pandas as pd
import random
import numpy as np

if __name__ == "__main__":
    # Check for config.yaml
    config = {}
    if os.path.exists("config.yaml"):
        with open("config.yaml", "r") as f:
            config = yaml.safe_load(f)
    

    def get_cfg(key, default):
        return config.get(key, default)
    
    def set_seed(seed: int = 42):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        # For deterministic behavior (optional, but recommended for reproducibility)
        torch.use_deterministic_algorithms(True)

    # Example usage:
    set_seed() 

    parser = argparse.ArgumentParser(description="Federated Proximal Gating with Data Quality")
    parser.add_argument('--K', type=int, default=get_cfg('K', 10), help='Number of clients')
    parser.add_argument('--N', type=int, default=get_cfg('N', 1000), help='Samples per client')
    parser.add_argument('--D', type=int, default=get_cfg('D', 20), help='Number of features')
    parser.add_argument('--n_rounds', type=int, default=get_cfg('n_rounds', 10), help='Number of rounds')
    parser.add_argument('--local_epochs', type=int, default=get_cfg('local_epochs', 1), help='Local epochs')
    parser.add_argument('--lr', type=float, default=get_cfg('lr', 0.2), help='Learning rate')
    parser.add_argument('--mu', type=float, default=get_cfg('mu', 1.0), help='Proximal regularization term')
    parser.add_argument('--n_corrupt', type=int, default=get_cfg('n_corrupt', 5), help='Number of corrupted features per client')
    parser.add_argument('--synthetic_data', type=bool, default=get_cfg('synthetic_data', False), help='Whether to use synthetic data')
    parser.add_argument('--dataset_path', type=str, default=get_cfg('dataset_path', ''), help='Path to experimental dataset')
    parser.add_argument('--target_column', type=str, default=get_cfg('target_column', ''), help='Target Column Name')
    parser.add_argument('--model_type', type=str, default=get_cfg('model_type', 'logreg'), choices=['logreg', 'nn'], help='Model type: logreg or nn')
    parser.add_argument('--num_classes', type=int, default=get_cfg('num_classes', 2), help='Number of classes')
    args = parser.parse_args()

    if args.synthetic_data:
        print("--- Creating Synthetic Data ---")
        X_clients, y_clients, X_test, y_test = make_synthetic_data(
            n_clients=args.K, samples_per_client=args.N, d=args.D
        )
    else:
        print("--- Loading Real Data ---")
        df = pd.read_csv(args.dataset_path)
        print(f"Loaded {len(df)} rows and {len(df.columns)} columns")
        print(df.head(10))
        X_clients, y_clients, X_test, y_test = homogeneous_splitting(df=df,
                                                                     n_clients=args.K,
                                                                     label_col=args.target_column)

    # Feature Data Poisoning
    X_corr_clients, q_clients = feature_poisoning(X_clients, n_corrupt=args.n_corrupt)

    # Model selection
    if args.model_type == 'logreg':
        model = torch.zeros(args.D)
    else:
        model = SimpleNN(args.D, args.num_classes)

    # ---------------- experiments ----------------
    print("===> CLEAN DATA")
    w_fedavg, _ = run_federated_training(
        X_clients,
        y_clients,
        use_frogdq=False,
        n_rounds=args.n_rounds,
        local_epochs=args.local_epochs,
        lr=args.lr,
        model=model
    )
    acc_fedavg_clean = accuracy(X_test, y_test, w_fedavg)
    print(f"FedAvg  accuracy: {acc_fedavg_clean:6.3f}")

    w_frogdq, g_frogdq = run_federated_training(
        X_clients,
        y_clients,
        use_frogdq=True,
        n_rounds=args.n_rounds,
        local_epochs=args.local_epochs,
        lr=args.lr,
        mu=args.mu,
        q_clients=[torch.ones(args.D) for _ in range(args.K)],  # all features trusted
        model=model
    )
    acc_frog_clean = accuracy(X_test, y_test, w_frogdq, g_frogdq)
    print(f"FroG-DQ accuracy: {acc_frog_clean:6.3f}")

    print("\n===> CORRUPTED DATA")
    # FedAvg on noisy clients
    w_fedavg_noise, _ = run_federated_training(
        X_corr_clients,
        y_clients,
        use_frogdq=False,
        n_rounds=args.n_rounds,
        local_epochs=args.local_epochs,
        lr=args.lr,
        model=model
    )
    acc_fedavg_noise = accuracy(X_test, y_test, w_fedavg_noise)
    print(f"FedAvg  accuracy: {acc_fedavg_noise:6.3f}")

    # FroG-DQ with trust scores
    w_frog_noise, g_frog_noise = run_federated_training(
        X_corr_clients,
        y_clients,
        use_frogdq=True,
        n_rounds=args.n_rounds,
        local_epochs=args.local_epochs,
        lr=args.lr,
        mu=args.mu,
        q_clients=q_clients,
        model=model
    )
    acc_frog_noise = accuracy(X_test, y_test, w_frog_noise, g_frog_noise)
    print(f"FroG-DQ accuracy: {acc_frog_noise:6.3f}")