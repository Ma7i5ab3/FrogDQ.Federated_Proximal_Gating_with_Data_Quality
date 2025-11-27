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
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

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
    
def load_checkpoint(checkpoint_file="experiment_checkpoint.json"):
    """
    Load checkpoint data to track completed experiments.
    
    Parameters
    ----------
    checkpoint_file : str
        Path to the checkpoint file (can include dataset-specific directories)
        
    Returns
    -------
    set
        Set of experiment IDs that have been completed
    """
    if os.path.exists(checkpoint_file):
        try:
            with open(checkpoint_file, 'r') as f:
                checkpoint_data = json.load(f)
                return set(checkpoint_data.get("completed_experiments", []))
        except (json.JSONDecodeError, FileNotFoundError):
            logger.warning(f"Could not load checkpoint file {checkpoint_file}. Starting fresh.")
            return set()
    return set()

def save_checkpoint(experiment_id, checkpoint_file="experiment_checkpoint.json"):
    """
    Save experiment ID to checkpoint file.
    
    Parameters
    ----------
    experiment_id : str
        Unique identifier for the completed experiment
    checkpoint_file : str
        Path to the checkpoint file (can include dataset-specific directories)
    """
    # Load existing checkpoint data
    checkpoint_dir = os.path.dirname(checkpoint_file)
    if checkpoint_dir:
        os.makedirs(checkpoint_dir, exist_ok=True)

    if os.path.exists(checkpoint_file):
        try:
            with open(checkpoint_file, 'r') as f:
                checkpoint_data = json.load(f)
        except (json.JSONDecodeError, FileNotFoundError):
            checkpoint_data = {"completed_experiments": [], "metadata": {"created": datetime.now().isoformat()}}
    else:
        checkpoint_data = {"completed_experiments": [], "metadata": {"created": datetime.now().isoformat()}}
    
    # Add new experiment ID if not already present
    if experiment_id not in checkpoint_data["completed_experiments"]:
        checkpoint_data["completed_experiments"].append(experiment_id)
        checkpoint_data["metadata"]["last_updated"] = datetime.now().isoformat()
        checkpoint_data["metadata"]["total_completed"] = len(checkpoint_data["completed_experiments"])
        
        # Save updated checkpoint
        try:
            with open(checkpoint_file, 'w') as f:
                json.dump(checkpoint_data, f, indent=2, ensure_ascii=False)
            logger.info(f"Checkpoint updated: {experiment_id}")
        except Exception as e:
            logger.error(f"Failed to save checkpoint: {e}")

def is_experiment_completed(experiment_id, completed_experiments):
    """
    Check if an experiment has already been completed.
    
    Parameters
    ----------
    experiment_id : str
        Unique identifier for the experiment
    completed_experiments : set
        Set of completed experiment IDs
        
    Returns
    -------
    bool
        True if experiment is completed, False otherwise
    """
    return experiment_id in completed_experiments

def save_experiment_results(experiment_data, json_filename="experiment_results.json"):
    """
    Save experiment results to a JSON file, appending new results to existing data.
    
    Parameters
    ----------
    experiment_data : dict
        Dictionary containing all experiment results and metadata
    json_filename : str
        Path to the results file (can include dataset-specific directories)
    """
    
    # Convert numpy types to JSON-serializable types
    experiment_data = convert_numpy_types(experiment_data)
    
    # Load existing results if file exists
    results_dir = os.path.dirname(json_filename)
    if results_dir:
        os.makedirs(results_dir, exist_ok=True)

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
    # Set device
    if torch.cuda.is_available():
        device = torch.device("cuda")
        logger.info(f"Using CUDA device: {torch.cuda.get_device_name()}")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
        logger.info("Using Apple Silicon GPU (MPS)")
    else:
        device = torch.device("cpu")
        logger.info("Falling back to CPU")


    completed_experiments_by_dataset = {}

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
        dataset_dir = dataset
        os.makedirs(dataset_dir, exist_ok=True)
        checkpoint_file = os.path.join(dataset_dir, "experiment_checkpoint.json")
        results_file = os.path.join(dataset_dir, "experiment_results.json")

        completed_experiments = load_checkpoint(checkpoint_file)
        completed_experiments_by_dataset[dataset] = completed_experiments
        logger.info(
            f"Loaded checkpoint for '{dataset}': {len(completed_experiments)} experiments already completed"
        )

        lr = exp.get("lr", 0.2)
        optimizer = exp.get("optimizer", "sgd")
        lambda_prox = exp.get("lambda_prox", 1.0)
        frog_temp_tau = exp.get("frog_temp_tau", 1.0)
        frog_temp_eta = exp.get("frog_temp_eta", 1.01)
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
                    poisoning_percentage = pois_pct,
                    random_state=seeds[exp_iteration]
                )

                for model_type, poisoning_type, frogdq_mode_raw in product(models, data_poisoning_methods, frogdq_modes):
                    parsed_mode = parse_frogdq_mode(frogdq_mode_raw)
                    frogdq_mode = parsed_mode.canonical
                    requested_modules = set(parsed_mode.requested)
                    active_modules = set(parsed_mode.active)
                    logger.info(
                        f"----- Start Training ----- Arch: {model_type}/Poisoning Type: {poisoning_type}/"
                        f"FrogDQ Mode: {frogdq_mode} (requested={parsed_mode.requested})"
                    )

                    # Create unique experiment ID for checkpoint tracking
                    experiment_id = f"{dataset}_{exp_iteration+1}_{feat_pct}_{pois_pct}_{model_type}_{poisoning_type}_{frogdq_mode}_{seeds[exp_iteration]}"
                    
                    # Check if this experiment has already been completed
                    if is_experiment_completed(experiment_id, completed_experiments):
                        logger.info(f"----- Skipping completed experiment: {experiment_id} -----")
                        continue
                
                    # Load Model
                    model = build_model(
                        input_dim=data_dct[poisoning_type]['X_train'].shape[1],
                        output_dim=torch.unique(data_dct['y_train']).numel(),
                        random_state=seeds[exp_iteration],
                        arch=model_type,
                        use_frogdq="inertia" in active_modules,
                        gate_init_vector=data_dct[poisoning_type]['q'] if "q" in requested_modules else None
                    )

                    #Train Model
                    history = train(
                        model=model,
                        X_train=data_dct[poisoning_type]['X_train'],
                        y_train=data_dct['y_train'],
                        X_val=data_dct[poisoning_type]['X_val'],
                        y_val=data_dct['y_val'],
                        q_vec=data_dct[poisoning_type]['q'],
                        r_vec=data_dct[poisoning_type]['r'] if "samplewise" in active_modules else None,
                        fetch_g_every=10,
                        frogdq_mode=frogdq_mode,
                        epochs=epochs,
                        lr=lr,
                        lr_scheduler="plateau",
                        weight_decay=1e-3 if model_type == "mlp" else 0.0,
                        optimizer=optimizer,
                        lambda_prox=lambda_prox,
                        frog_temp_tau=frog_temp_tau,
                        frog_temp_eta=frog_temp_eta,
                        verbose=True,
                        random_state=seeds[exp_iteration],
                        device=device
                    )

                    #Test Model
                    test_metrics = evaluate(
                        model=model,
                        X=data_dct[poisoning_type]['X_test'],
                        y=data_dct['y_test'],
                        random_state=seeds[exp_iteration],
                        device=device
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
                        "experiment_id": f"{dataset}_{exp_iteration+1}_{feat_pct}_{pois_pct}_{model_type}_{poisoning_type}_{frogdq_mode}_{seeds[exp_iteration]}",
                        "timestamp": datetime.now().isoformat(),
                        "dataset": dataset,
                        "seed": seeds[exp_iteration],
                        "iteration": exp_iteration,
                        "model_type": model_type,
                        "poisoning_method": poisoning_type,
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
                    save_experiment_results(experiment_data, json_filename=results_file)

                    # Save checkpoint to mark this experiment as completed
                    save_checkpoint(experiment_id, checkpoint_file=checkpoint_file)
                    completed_experiments.add(experiment_id)

                    logger.info(f"----- Experiment completed and checkpointed: {experiment_id} -----")


# Final summary
total_experiments = len(experiments) * n_repeat_exps * len(features_percentage) * len(poisoning_percentage) * len(models) * len(data_poisoning_methods) * len(frogdq_modes)
completed_count = sum(len(exp_set) for exp_set in completed_experiments_by_dataset.values())
logger.info(f"\n===== EXPERIMENT SUMMARY =====")
logger.info(f"Total experiments configured: {total_experiments}")
logger.info(f"Completed experiments: {completed_count}")
logger.info(f"Remaining experiments: {total_experiments - completed_count}")
if completed_experiments_by_dataset:
    logger.info("Checkpoint files per dataset:")
    for dataset_name in completed_experiments_by_dataset.keys():
        logger.info(f"  - {dataset_name}: {os.path.join(dataset_name, 'experiment_checkpoint.json')}")
    logger.info("Results files per dataset:")
    for dataset_name in completed_experiments_by_dataset.keys():
        logger.info(f"  - {dataset_name}: {os.path.join(dataset_name, 'experiment_results.json')}")
logger.info("================================\n")
                

            
