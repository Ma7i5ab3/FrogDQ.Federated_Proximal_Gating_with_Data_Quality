from data_poisoning import *
from DataPreparation import DataPreparation
import torch
import os
import yaml
import pandas as pd
import random
import numpy as np
import sys
import random
from loguru import logger
from model import *
from itertools import product
import json
from datetime import datetime

import warnings

warnings.filterwarnings("ignore")

def convert_numpy_types(obj):
    """
    Convert numpy types to native Python types for JSON serialization.
    
    Parameters
    ----------
    obj : any
        Object that may contain numpy types
        
    Returns
    -------
    any
        Object with numpy types converted to Python types
    """
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, dict):
        return {key: convert_numpy_types(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_types(item) for item in obj]
    else:
        return obj

def save_experiment_results(experiment_data):
    """
    Save experiment results to a JSON file, appending new results to existing data.
    
    Parameters
    ----------
    experiment_data : dict
        Dictionary containing all experiment results and metadata
    """
    json_filename = "experiment_results.json"
    
    # Convert numpy types to JSON-serializable types
    experiment_data = convert_numpy_types(experiment_data)
    
    # Load existing results if file exists
    if os.path.exists(json_filename):
        try:
            with open(json_filename, 'r') as f:
                all_results = json.load(f)
        except (json.JSONDecodeError, FileNotFoundError):
            # If file is corrupted or doesn't exist, start fresh
            all_results = {"experiments": [], "metadata": {"created": datetime.now().isoformat(), "last_updated": None}}
    else:
        # Create new structure if file doesn't exist
        all_results = {"experiments": [], "metadata": {"created": datetime.now().isoformat(), "last_updated": None}}
    
    # Add new experiment data
    all_results["experiments"].append(experiment_data)
    all_results["metadata"]["last_updated"] = datetime.now().isoformat()
    all_results["metadata"]["total_experiments"] = len(all_results["experiments"])
    
    # Save updated results
    try:
        with open(json_filename, 'w') as f:
            json.dump(all_results, f, indent=2, ensure_ascii=False)
        logger.info(f"Experiment results saved to {json_filename}")
    except Exception as e:
        logger.error(f"Failed to save experiment results: {e}")

if __name__ == "__main__":
    # Load all experiment setups from config.yaml
    config = {}
    if os.path.exists("config.yaml"):
        with open("config.yaml", "r") as f:
            config = yaml.safe_load(f)

    #Get Configuration values
    experiments = config.get("experiments", [])
    data_poisoning_methods = config.get("data_poisoning").get("methods")
    features_percentage = config.get("data_poisoning").get("features_percentage")
    poisoning_percentage = config.get("data_poisoning").get("poisoning_percentage")
    seeds = config.get("seeds")
    models = config.get("models")
    frogdq_modes = config.get("frogdq_modes")

    n_repeat_exps = len(seeds)
    if not experiments:
        print("No experiments found in config.yaml under 'experiments'. Exiting.")
        sys.exit(1)

    def set_seed(seed: int = 42):
        print(f"Seed: {seed}")
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.use_deterministic_algorithms(True)

    #Running Experiment for each Dataset define in the config.yaml
    for exp in experiments:
        logger.info(
            f"\n===== Running Experiment for Dataset: {exp.get('dataset', 'Unnamed')} ====="
        )

        # Extract hyperparameters for the current dataset
        dataset = exp.get("dataset", "Unnamed")
        lr = exp.get("lr", 0.2)
        lambda_prox = exp.get("lambda_prox", 1.0)
        frog_temp_tau = exp.get("frog_temp_tau", 1.0)
        epochs = exp.get("local_epochs", 200)
        label_col = exp.get("target_column", "")
        splitting_perc_train_test = exp.get("splitting_perc_train_test", "")
        splitting_perc_test_val = exp.get("splitting_perc_test_val", "")


        for exp_iteration in range(n_repeat_exps):
            logger.info(f"\n===== Loop {exp_iteration} of {n_repeat_exps} / Seed: {seeds[exp_iteration]} =====")
            set_seed(seed=seeds[exp_iteration])

            # Iterate over each combination of features_percentage and poisoning_percentage
            for feat_pct, pois_pct in product(features_percentage, poisoning_percentage):
                logger.info(f"----- Features percentage: {feat_pct}, Poisoning Percentage={pois_pct} -----")

                data_prep = DataPreparation(dataset_name=dataset)
                data_prep.load()
                data_dct = data_prep.run_preprocessing(
                    splitting_perc_train_test = splitting_perc_train_test,
                    splitting_perc_test_val = splitting_perc_test_val,
                    features_percentage = feat_pct,
                    poisoning_percentage = pois_pct
                )

                for model_type, poisonong_type, frogdq_mode in product(models, data_poisoning_methods, frogdq_modes):
                    logger.info(f"----- Start Training ----- Arch: {model_type}/Poisoning Type: {poisonong_type}/FrogDQ Mode: {frogdq_mode}")
                
                    # Load Model
                    model = build_model(
                        input_dim=data_dct[poisonong_type]['X_train'].shape[1],
                        output_dim=torch.unique(data_dct['y_train']).numel(),
                        random_state=seeds[exp_iteration],
                        arch=model_type,
                        use_frogdq=True if frogdq_mode != "none" else False
                    )

                    #Train Model
                    history = train(
                        model=model,
                        X_train=data_dct[poisonong_type]['X_train'],
                        y_train=data_dct['y_train'],
                        X_val=data_dct[poisonong_type]['X_val'],
                        y_val=data_dct['y_val'],
                        q_vec=data_dct[poisonong_type]['q'],
                        frogdq_mode=frogdq_mode,
                        epochs=epochs,
                        lr=lr,
                        lambda_prox=lambda_prox,
                        frog_temp_tau=frog_temp_tau,
                        verbose=True
                    )

                    #Test Model
                    test_metrics = evaluate(
                        model=model,
                        X=data_dct[poisonong_type]['X_test'],
                        y=data_dct['y_test'],
                        random_state=seeds[exp_iteration]
                    )

                    logger.info(
                        f"Test: loss={test_metrics['loss']:.4f}, acc={test_metrics['accuracy']:.4f}, "
                        f"bal_acc={test_metrics['balanced_accuracy']:.4f}, "
                        f"f1={test_metrics['f1']:.4f}, auc={test_metrics['auc_roc']:.4f}"
                    )

                    #Save Test results in history
                    history["test_loss"] = test_metrics['loss']
                    history["test_acc"] = test_metrics["accuracy"]
                    history["test_bal_acc"] = test_metrics["balanced_accuracy"]
                    history["test_f1"] = test_metrics["f1"]
                    history["test_auc"] = test_metrics["auc_roc"]

                    # Prepare experiment data for JSON saving
                    experiment_data = {
                        "experiment_id": f"{dataset}_{exp_iteration+1}_{feat_pct}_{pois_pct}_{model_type}_{poisonong_type}_{frogdq_mode}_{seeds[exp_iteration]}",
                        "timestamp": datetime.now().isoformat(),
                        "dataset": dataset,
                        "seed": seeds[exp_iteration],
                        "iteration": exp_iteration,
                        "model_type": model_type,
                        "poisoning_method": poisonong_type,
                        "frogdq_mode": frogdq_mode,
                        "features_percentage": feat_pct,
                        "poisoning_percentage": pois_pct,
                        "hyperparameters": {
                            "learning_rate": lr,
                            "lambda_prox": lambda_prox,
                            "frog_temp_tau": frog_temp_tau,
                            "epochs": epochs,
                        },
                        "history": history
                    }

                    # Save experiment results to JSON file
                    save_experiment_results(experiment_data)
                    

            
