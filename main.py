from data_distribution.distribution import homogeneous_splitting
from data_poisoning.feature import feature_poisoning
from data.synthetic.data_generation import make_synthetic_data
from data.real.DataPreparation import DataPreparation
from federated.central import run_federated_training
from evaluation.evalutation import accuracy
from federated.model import SimpleNN
from data_statistics.Stats import Stats
from data_visualization.plot import (
    plot_weights_over_rounds,
    plot_compare_weights_over_rounds,
    plot_accuracy_test_comparison,
    plot_accuracy_val_over_rounds,
    plot_compare_weights_original_model,
)
from utils import *
import torch
import os
import yaml
import pandas as pd
import random
import numpy as np
import sys
import random

import warnings

warnings.filterwarnings("ignore")

if __name__ == "__main__":
    # Load all experiment setups from config.yaml
    config = {}
    if os.path.exists("config.yaml"):
        with open("config.yaml", "r") as f:
            config = yaml.safe_load(f)

    experiments = config.get("experiments", [])
    n_repeat_exps = 10
    if not experiments:
        print("No experiments found in config.yaml under 'experiments'. Exiting.")
        sys.exit(1)

    def set_seed(seed: int = 42):
        print(f"Seed: {seed}")
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.use_deterministic_algorithms(True)

    # Init Stats Clients
    stats_client = Stats()

    for exp in experiments:
        print(f"\n===== Running Experiment: {exp.get('name', 'Unnamed')} =====")
        if exp.get('name', 'Unnamed') == 'Experiment Real':
            # Set seed for reproducibility (optional: allow per-experiment seed)
            # set_seed(exp.get('seed', 42))

            # Reset Statistics Client for each new experiment type
            stats_client.reset()

            # Extract parameters for this experiment
            experiment_name = exp.get("name", "Unnamed")
            K = exp.get("K", 10)
            N = exp.get("N", 1000)
            D = exp.get("D", 20)
            n_rounds = exp.get("n_rounds", 10)
            local_epochs = exp.get("local_epochs", 1)
            lr = exp.get("lr", 0.2)
            mu = exp.get("mu", 1.0)
            n_corrupt = exp.get("n_corrupt", 5)
            synthetic_data = exp.get("synthetic_data", False)
            dataset_path = exp.get("dataset_path", "")
            target_column = exp.get("target_column", "")
            model_type = exp.get("model_type", "logreg")
            num_classes = exp.get("num_classes", 2)
            n_feature_show = exp.get("n_feature_show", 5)
            noise_std_syn = exp.get("noise_std_syn", 0.2)
            noise_std_poisoning = exp.get("noise_std_poisoning", 0.2)
            missing = exp.get("missing", False)

            # Set fixed seeds for reproducibility per experiment
            seeds = [42, 56, 221, 800, 1258, 2, 121, 4200, 444, 500]
            if len(seeds) < n_repeat_exps:
                raise ValueError(
                    f"Not enough seeds for {n_repeat_exps} repetitions. Please provide at least {n_repeat_exps} seeds."
                )

            for exp_iteration in range(n_repeat_exps):
                print(f"\n===== Loop {exp_iteration+1} of {n_repeat_exps} =====")
                set_seed(seed=seeds[exp_iteration])
                w_global = torch.randn(D+1) / torch.sqrt(torch.tensor(D+1))

                if synthetic_data:
                    print("--- Creating Synthetic Data ---")
                    X_clients, y_clients, X_test, y_test, X_val, y_val, w_true = (
                        make_synthetic_data(
                            n_clients=K, samples_per_client=N, d=D, noise_std=noise_std_syn
                        )
                    )
                else:
                    data_prep = DataPreparation(dataset_name='adult')
                    data_prep.load()
                    sys.exit()


                # Define the topk_features to show in Stats
                topk_features = get_top_k_indexes(weights=w_true, k=n_feature_show)
                stats_client.set_topk_features(
                    topk_features
                )

                # Feature Data Poisoning: if largest==False it takes the less important features
                corr_features = get_top_k_indexes(
                    weights=w_true, k=n_corrupt, largest=False
                )
                print(f"Weights of corrupted features: {w_true[topk_features]}")
                X_corr_clients, q_clients = feature_poisoning(
                    X_clients,
                    n_corrupt=n_corrupt,
                    rnd_per_client=False,
                    indexes_to_corrupt=corr_features,
                    noise_std=noise_std_poisoning,
                    missing_value=missing
                )

                # ---------------- Run Experiments ----------------
                for sim_info in stats_client.data.keys():
                    w, g = run_federated_training(
                        X_clients=(
                            X_clients if sim_info[1] == "clean" else X_corr_clients
                        ),  # if simulation is on dq_type == 'dirty' choose corrupted X data.
                        y_clients=y_clients,
                        X_val=X_val,
                        y_val=y_val,
                        dq_type=sim_info[1],
                        fl_algorithm=sim_info[0],
                        n_rounds=n_rounds,
                        local_epochs=local_epochs,
                        lr=lr,
                        q_clients=q_clients,
                        stats_client=stats_client,
                        w_global=w_global,
                    )
                    accuracy_test = accuracy(
                        X=X_test, y=y_test, w=w, g=g if sim_info[0] == "frog" else None
                    )
                    stats_client.set_accuracy_test(
                        value=accuracy_test, algorithm=sim_info[0], dq_type=sim_info[1]
                    )
                    print(
                        f"{sim_info[0]} ({sim_info[1]} Data) accuracy: {accuracy_test:6.3f}"
                    )

                # --- Create Statistics ----
                plot_weights_over_rounds(
                    stats_client=stats_client,
                    fl_algorithm="frog",
                    dq_type="clean",
                    type="gate",
                    experiment_name=experiment_name,
                    iteration=exp_iteration + 1,
                )
                plot_weights_over_rounds(
                    stats_client=stats_client,
                    fl_algorithm="frog",
                    dq_type="dirty",
                    type="gate",
                    experiment_name=experiment_name,
                    iteration=exp_iteration + 1,
                )
                plot_weights_over_rounds(
                    stats_client=stats_client,
                    fl_algorithm="fedavg",
                    dq_type="clean",
                    type="weight",
                    experiment_name=experiment_name,
                    iteration=exp_iteration + 1,
                )
                plot_weights_over_rounds(
                    stats_client=stats_client,
                    fl_algorithm="fedavg",
                    dq_type="dirty",
                    type="weight",
                    experiment_name=experiment_name,
                    iteration=exp_iteration + 1,
                )
                plot_compare_weights_over_rounds(
                    stats_client=stats_client,
                    corr_features=corr_features,
                    experiment_name=experiment_name,
                    iteration=exp_iteration + 1,
                )
                plot_accuracy_val_over_rounds(
                    stats_client=stats_client,
                    experiment_name=experiment_name,
                    iteration=exp_iteration + 1,
                )
                plot_compare_weights_original_model(
                    stats_client=stats_client,
                    w_true=w_true,
                    dq_type='dirty',
                    feature=topk_features[0],
                    corrupted=True if topk_features[0] in corr_features else False,
                    experiment_name=experiment_name,
                    iteration=exp_iteration + 1
                )
                plot_compare_weights_original_model(
                    stats_client=stats_client,
                    w_true=w_true,
                    dq_type='dirty',
                    feature=corr_features[0],
                    corrupted=True,
                    experiment_name=experiment_name,
                    iteration=exp_iteration + 1
                )
                stats_client.reset(stats_field="accuracy_val")

            plot_accuracy_test_comparison(
                stats_client=stats_client,
                experiment_name=experiment_name,
            )
