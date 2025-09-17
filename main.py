from data_poisoning.data_poisoning import *
from DataPreparation import DataPreparation
from federated.central import run_federated_training
from data_statistics.Stats import Stats
from utils import *
import torch
import os
import yaml
import pandas as pd
import random
import numpy as np
import sys
import random
from loguru import logger

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

    for exp in experiments:
        logger.info(f"\n===== Running Experiment for Dataset: {exp.get('dataset', 'Unnamed')} =====")

        # Extract parameters for this experiment
        experiment_name = exp.get("dataset", "Unnamed")
        lr = exp.get("lr", 0.2)
        mu = exp.get("mu", 1.0)
        dataset_path = exp.get("dataset_path", "")
        label_col = exp.get("target_column", "")

        # Set fixed seeds for reproducibility per experiment
        seeds = [42, 56, 221, 800, 1258, 2, 121, 4200, 444, 500]
        if len(seeds) < n_repeat_exps:
            raise ValueError(
                f"Not enough seeds for {n_repeat_exps} repetitions. Please provide at least {n_repeat_exps} seeds."
            )

        for exp_iteration in range(n_repeat_exps):
            logger.info(f"\n===== Loop {exp_iteration+1} of {n_repeat_exps} =====")
            set_seed(seed=seeds[exp_iteration])

            data_prep = DataPreparation(dataset_name='diabates')
            data_prep.load(label_col=label_col)
            X_train, y_clients, X_test, y_test, X_val, y_val = data_prep.run_preprocessing(label_col=label_col)
                
            # ---------------- Run Experiments ----------------
            '''w_global = torch.randn(X_clients[0].size(1)+1) / torch.sqrt(torch.tensor(X_clients[0].size(1)+1))
            for sim_info in stats_client.data.keys():
                if sim_info[0] in ['fedavg', 'frog', 'frog_new']:
                    w, g = run_federated_training(
                        X_clients=(
                            X_clients if sim_info[1] == "clean"
                            else X_corr_clients if sim_info[0] != "fedavg_no_corr_feat" and sim_info[1] == "dirty"
                            else X_clients_no_corr if sim_info[0] == "fedavg_no_corr_feat" and sim_info[1] == "dirty"
                            else None
                        ),  
                        y_clients=y_clients,
                        X_val=X_val if sim_info[0] != "fedavg_no_corr_feat" else X_val_no_corr,
                        y_val=y_val,
                        dq_type=sim_info[1],
                        fl_algorithm=sim_info[0],
                        n_rounds=n_rounds,
                        local_epochs=local_epochs,
                        lr=lr,
                        mu=mu,
                        q_clients=q_clients,
                        stats_client=stats_client,
                        w_global=w_global if sim_info[0] != "fedavg_no_corr_feat" else torch.randn(X_clients_no_corr[0].size(1)+1) / torch.sqrt(torch.tensor(X_clients_no_corr[0].size(1)+1)),
                    )
                    roc_auc_test, accuracy_test, balanced_accuracy, loss_test = eval(
                        X=X_test if sim_info[0] != "fedavg_no_corr_feat" else X_test_no_corr, y=y_test, w=w, g=g if sim_info[0] == "frog" else None
                    )
                    stats_client.set_accuracy_test(
                        value=balanced_accuracy, algorithm=sim_info[0], dq_type=sim_info[1]
                    )
                    stats_client.set_eval_metric(
                        value=roc_auc_test, algorithm=sim_info[0], dq_type=sim_info[1], metric="roc_auc_test"
                    )
                    print(
                        f"{sim_info[0]} ({sim_info[1]} Data) accuracy: {accuracy_test:.3f} / roc_auc: {roc_auc_test:.3f} / balanced accuracy: {balanced_accuracy:.3f}"
                    )'''
        
