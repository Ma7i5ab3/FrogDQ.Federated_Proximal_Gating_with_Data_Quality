"""
Hyperparameter optimization with Optuna for FrogDQ experiments.

This module provides a comprehensive pipeline for hyperparameter tuning across
different model architectures (linear, MLP), curriculum learning settings, and
gate configurations using Optuna with multi-seed evaluation.
"""

import time
import warnings
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Tuple

import numpy as np
import optuna
import pandas as pd
import yaml
from joblib import Parallel, delayed
from optuna.samplers import TPESampler
from optuna.study import Study

from frogdq.catboost_model import fit_catboost
from frogdq.data import get_datasets, load_autogluon_data, load_baseline_zero_data, load_cp_data, load_data, load_knn_data, load_raw_data, load_saga_data
from frogdq.nn import build_model
from frogdq.training import fit, set_seed


def compute_convergence_metrics(history: Dict[str, List], primary_metric: str,
                                  percentiles: List[float] = [0.5, 0.75, 0.9, 0.95, 0.99]) -> Dict[str, Any]:
    """
    Compute convergence metrics from training history.

    Parameters
    ----------
    history : Dict[str, List]
        Training history containing metric values over epochs
    primary_metric : str
        Primary metric to track (e.g., 'f1' or 'rmse')
    percentiles : List[float]
        Percentiles of final performance to track (e.g., 0.9 = 90% of final performance)

    Returns
    -------
    Dict[str, Any]
        Dictionary containing convergence metrics:
        - best_epoch: Epoch with best validation performance
        - epochs_to_X_pct: Epochs needed to reach X% of final val performance
        - final_gate_sparsity: Final average gate weight (if gates used)
        - convergence_stability: Std dev of last 10% of epochs
    """
    val_metric_key = f'val_{primary_metric}'

    if val_metric_key not in history or len(history[val_metric_key]) == 0:
        return {}

    val_metric = np.array(history[val_metric_key])

    # For RMSE/MAE, lower is better; for accuracy/F1, higher is better
    minimize = primary_metric in ['rmse', 'mae', 'loss']

    # Best epoch (based on validation metric)
    if minimize:
        best_epoch = int(np.argmin(val_metric))
        final_performance = val_metric[best_epoch]
    else:
        best_epoch = int(np.argmax(val_metric))
        final_performance = val_metric[best_epoch]

    metrics = {
        'best_epoch': best_epoch,
        'best_val_metric': float(final_performance),
    }

    # Compute epochs to reach percentiles of final performance
    for pct in percentiles:
        if minimize:
            # For minimization: reach pct above best (worse performance)
            # E.g., 90% means 10% worse than best
            threshold = final_performance * (1 + (1 - pct))
            epochs_to_pct = np.where(val_metric <= threshold)[0]
        else:
            # For maximization: reach pct of best
            threshold = final_performance * pct
            epochs_to_pct = np.where(val_metric >= threshold)[0]

        if len(epochs_to_pct) > 0:
            metrics[f'epochs_to_{int(pct*100)}pct'] = int(epochs_to_pct[0])
        else:
            metrics[f'epochs_to_{int(pct*100)}pct'] = len(val_metric)  # Never reached

    # Convergence stability (lower = more stable in late training)
    last_10pct = int(len(val_metric) * 0.1)
    if last_10pct > 1:
        metrics['convergence_stability'] = float(np.std(val_metric[-last_10pct:]))

    # Gate metrics if available
    if 'gate_weights' in history and len(history['gate_weights']) > 0:
        final_gate_weights = history['gate_weights'][-1]
        metrics['final_gate_sparsity'] = float(np.mean(final_gate_weights))
        metrics['final_gate_std'] = float(np.std(final_gate_weights))

        # Track gate evolution
        all_gates = np.array(history['gate_weights'])
        metrics['gate_change_rate'] = float(np.mean(np.abs(np.diff(all_gates, axis=0))))

    return metrics


def _get_storage_url(output_dir: Path) -> str:
    """
    Get the Optuna storage URL for persisting studies.

    Parameters
    ----------
    output_dir : Path
        Output directory for results

    Returns
    -------
    str
        SQLite storage URL
    """
    db_path = output_dir / "optuna_studies.db"
    return f"sqlite:///{db_path}"


class OptunaExperiment:
    """
    Optuna-based hyperparameter optimization for FrogDQ experiments.

    This class manages hyperparameter optimization across different configurations:
    - Model type: linear (no hidden layers) or MLP
    - Curriculum learning: on or off
    - Gate layer: on or off
    - Data quality modes: clean, AR (At Random), NAR (Not At Random)

    The optimization:
    - Samples hyperparameters from predefined ranges
    - Evaluates each trial across multiple random seeds
    - Maximizes average validation metric across seeds
    - Logs top-M runs with train/val/test metrics
    - Supports parallel seed evaluation within trials for reproducibility

    Example
    -------
    >>> config = {
    ...     'datasets': ['iris', 'wine'],
    ...     'data_modes': ['clean', 'ar'],
    ...     'model_types': ['linear', 'mlp'],
    ...     'curriculum_settings': [True, False],
    ...     'gate_settings': [True, False],
    ...     'n_trials': 50,
    ...     'n_seeds': 5,
    ...     'seed_start': 42,
    ...     'top_m': 5,
    ...     'n_jobs_seeds': 4,
    ... }
    >>> experiment = OptunaExperiment(config)
    >>> results = experiment.run_all_experiments()
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the Optuna experiment with configuration.

        Parameters
        ----------
        config : dict
            Configuration dictionary with keys:
            - datasets: List of dataset names to run on
            - data_modes: List of data modes ('clean', 'ar', 'nar')
            - model_types: List of model types ('linear', 'mlp')
            - curriculum_settings: List of curriculum on/off (True/False)
            - gate_settings: List of gate on/off (True/False)
            - n_trials: Number of Optuna trials per configuration
            - n_seeds: Number of random seeds to evaluate per trial
            - seed_start: Starting seed (incremental from this)
            - top_m: Number of top runs to log
            - n_jobs_seeds: Number of parallel jobs for seed evaluation
            - clean_val: Whether to use clean validation data
            - clean_test: Whether to use clean test data
            - output_dir: Directory to save results
            - save_npz_histories: Whether to save training histories as .npz files (default: True)
            - verbose: Verbosity level (0=silent, 1=progress, 2=detailed)
            - optuna_sampler_seed: Random seed for Optuna sampler
            - hyperparameter_ranges: Dict of hyperparameter ranges (optional)
            - warm_start: Whether to resume from previous incomplete runs (default: False)
            - reuse_params: Whether to reuse parameters from runs for AR/NAR (default: False)
            - reuse_top_n: Number of top trials to sample from (default: 5)
            - run_catboost: Whether to include CatBoost as an additional model
              baseline (default: False). Runs once per dataset/data_mode,
              independent of model_types. No preprocessing is applied —
              CatBoost is trained directly on raw data (NaN preserved,
              categorical features passed natively).
            - catboost_thread_count: CPU threads for CatBoost (default: -1,
              i.e. use all available cores)
        """
        self.config = config

        # Extract configuration
        self.datasets = config.get('datasets', [])
        self.data_modes = config.get('data_modes', ['clean'])
        self.model_types = config.get('model_types', ['linear', 'mlp'])
        self.curriculum_settings = config.get('curriculum_settings', [False, True])
        self.gate_settings = config.get('gate_settings', [False, True])

        self.n_trials = config.get('n_trials', 50)
        self.n_seeds = config.get('n_seeds', 5)
        self.seed_start = config.get('seed_start', 42)
        self.top_m = config.get('top_m', 5)
        self.n_jobs_seeds = config.get('n_jobs_seeds', 1)
        self.n_jobs_optuna = config.get('n_jobs_optuna', 1)

        self.clean_val = config.get('clean_val', False)
        self.clean_test = config.get('clean_test', True)

        self.output_dir = Path(config.get('output_dir', 'results'))
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.verbose = config.get('verbose', 1)
        self.save_npz_histories = config.get('save_npz_histories', True)
        self.optuna_sampler_seed = config.get('optuna_sampler_seed', 42)
        self.warm_start = config.get('warm_start', False)

        # Parameter reuse from previous runs
        self.reuse_params = config.get('reuse_params', False)
        self.reuse_top_n = config.get('reuse_top_n', 5)

        # AutoGluon benchmark settings
        self.run_autogluon = config.get('run_autogluon', False)
        self.autogluon_data_dir = config.get('autogluon_data_dir', 'data_autogluon')

        # Saga benchmark settings
        self.run_saga = config.get('run_saga', False)
        self.saga_data_dir = config.get('saga_data_dir', 'data_cleaned_saga')

        # Custom pipeline (CP) benchmark settings
        self.run_cp = config.get('run_cp', False)
        self.cp_data_dir = config.get('cp_data_dir', 'data_cleaned_cp')

        # Baseline 0 (zero/random imputation) benchmark settings
        self.run_baseline_zero = config.get('run_baseline_zero', False)
        self.baseline_zero_data_dir = config.get('baseline_zero_data_dir', 'data_baseline_zero')

        # KNN imputation benchmark settings
        self.run_knn = config.get('run_knn', False)
        self.knn_data_dir = config.get('knn_data_dir', 'data_knn')

        # CatBoost benchmark settings.
        # Unlike the other benchmarks above, this is a model baseline (not a
        # data-preparation baseline): it runs once per dataset/data_mode,
        # independent of model_types. It also does not need a data dir since
        # nothing is precomputed — CatBoost is trained directly on raw data
        # (NaN preserved, categorical features passed natively; no
        # imputation/one-hot encoding/scaling).
        self.run_catboost = config.get('run_catboost', False)
        self.catboost_thread_count = config.get('catboost_thread_count', -1)

        # Base data directories (relative to CWD or absolute)
        self.data_dir = config.get('data_dir', 'data')
        self.poisoned_dir = config.get('poisoned_dir', 'data_poisoned')

        # Optuna storage for persistence
        self.storage_url = _get_storage_url(self.output_dir)

        # Hyperparameter ranges (can be customized via config)
        self.hp_ranges = config.get('hyperparameter_ranges', self._default_hp_ranges())

        # Results storage
        self.all_results = []

        # Cache for clean run parameters
        self._clean_params_cache = {}

    def _default_hp_ranges(self) -> Dict[str, Any]:
        """
        Get default hyperparameter ranges for optimization.

        Returns
        -------
        dict
            Dictionary of hyperparameter ranges with keys:
            - Common ranges (for both linear and MLP)
            - MLP-specific ranges
            - Curriculum-specific ranges
            - Gate-specific ranges
        """
        return {
            # Common hyperparameters
            'epochs': (50, 300),
            'batch_size_choices': [16, 32, 64, 128, 256],
            'learning_rate': (1e-4, 1e-1, 'log'),
            'optimizer_choices': ['adam', 'adamw', 'sgd'],
            'weight_decay': (1e-6, 1e-2, 'log'),
            'early_stopping_patience': (10, 50),
            'lr_scheduler_choices': ['none', 'plateau', 'cosine', 'step'],
            'dropout': (0.0, 0.5),

            # MLP-specific
            'hidden_neurons_choices': [32, 64, 128, 256, 512],
            'num_layers': (1, 4),
            'activation_choices': ['relu', 'gelu', 'elu'],
            'use_batch_norm': [True, False],

            # Curriculum-specific
            'curriculum_strategy_choices': ['linear', 'exponential', 'step'],

            # Gate-specific
            'gate_init_choices': ['random', 'ones', 'quality'],
            'gate_loss_weight': (1e-4, 1e-1, 'log'),
            'gate_anchor_interval': (1, 20),
            'gate_quality_weighting_choices': ['linear', 'quadratic', 'exp', 'inv_exp'],
            'gate_loss_scheduler_choices': ['none', 'decay', 'cosine'],

            # CatBoost-specific (trained on raw data: NaN + native categoricals, no scaling)
            'catboost_iterations': (1000, 1000),
            'catboost_early_stopping_rounds': (50, 50),
            'catboost_learning_rate': (0.01, 0.3, 'log'),
            'catboost_depth': (4, 10),
            'catboost_l2_leaf_reg': (1.0, 10.0, 'log'),
            'catboost_random_strength': (1e-9, 10.0, 'log'),
            'catboost_bagging_temperature': (0.0, 1.0),
            'catboost_border_count': (32, 255),
            'catboost_min_data_in_leaf': (1, 100),
            'catboost_grow_policy_choices': ['SymmetricTree', 'Depthwise', 'Lossguide'],
        }

    def _suggest_hyperparameters(
        self,
        trial: optuna.Trial,
        model_type: str,
        use_curriculum: bool,
        use_gate: bool,
        task: str,
    ) -> Dict[str, Any]:
        """
        Suggest hyperparameters for a trial based on configuration.

        Parameters
        ----------
        trial : optuna.Trial
            Optuna trial object
        model_type : str
            'linear' or 'mlp'
        use_curriculum : bool
            Whether curriculum learning is enabled
        use_gate : bool
            Whether gate layer is enabled
        task : str
            'classification' or 'regression'

        Returns
        -------
        dict
            Dictionary of suggested hyperparameters
        """
        hp = {}

        # Common hyperparameters
        hp['epochs'] = trial.suggest_int('epochs', *self.hp_ranges['epochs'])
        hp['batch_size'] = trial.suggest_categorical('batch_size', self.hp_ranges['batch_size_choices'])
        hp['learning_rate'] = trial.suggest_float('learning_rate', *self.hp_ranges['learning_rate'][:2],
                                                    log=(self.hp_ranges['learning_rate'][2] == 'log'))
        hp['optimizer'] = trial.suggest_categorical('optimizer', self.hp_ranges['optimizer_choices'])
        hp['weight_decay'] = trial.suggest_float('weight_decay', *self.hp_ranges['weight_decay'][:2],
                                                   log=(self.hp_ranges['weight_decay'][2] == 'log'))
        hp['early_stopping_patience'] = trial.suggest_int('early_stopping_patience',
                                                           *self.hp_ranges['early_stopping_patience'])
        hp['lr_scheduler'] = trial.suggest_categorical('lr_scheduler', self.hp_ranges['lr_scheduler_choices'])

        # Model architecture
        if model_type == 'linear':
            hp['hidden_neurons'] = 0
            hp['num_layers'] = 0
        else:  # mlp
            hp['hidden_neurons'] = trial.suggest_categorical('hidden_neurons',
                                                              self.hp_ranges['hidden_neurons_choices'])
            hp['num_layers'] = trial.suggest_int('num_layers', *self.hp_ranges['num_layers'])
            hp['dropout'] = trial.suggest_float('dropout', *self.hp_ranges['dropout'])
            hp['activation'] = trial.suggest_categorical('activation', self.hp_ranges['activation_choices'])
            hp['use_batch_norm'] = trial.suggest_categorical('use_batch_norm',
                                                              self.hp_ranges['use_batch_norm'])

        # Curriculum learning
        if use_curriculum:
            hp['curriculum_strategy'] = trial.suggest_categorical('curriculum_strategy',
                                                                   self.hp_ranges['curriculum_strategy_choices'])

        # Gate layer
        if use_gate:
            hp['gate_init'] = trial.suggest_categorical('gate_init', self.hp_ranges['gate_init_choices'])
            hp['gate_loss_weight'] = trial.suggest_float('gate_loss_weight',
                                                          *self.hp_ranges['gate_loss_weight'][:2],
                                                          log=(self.hp_ranges['gate_loss_weight'][2] == 'log'))
            hp['gate_anchor_interval'] = trial.suggest_int('gate_anchor_interval',
                                                            *self.hp_ranges['gate_anchor_interval'])
            hp['gate_quality_weighting'] = trial.suggest_categorical('gate_quality_weighting',
                                                                      self.hp_ranges['gate_quality_weighting_choices'])
            hp['gate_loss_scheduler'] = trial.suggest_categorical('gate_loss_scheduler',
                                                                   self.hp_ranges['gate_loss_scheduler_choices'])

        return hp

    def _suggest_catboost_hyperparameters(self, trial: optuna.Trial) -> Dict[str, Any]:
        """
        Suggest CatBoost hyperparameters for a trial.

        Parameters
        ----------
        trial : optuna.Trial
            Optuna trial object

        Returns
        -------
        dict
            Dictionary of suggested CatBoost hyperparameters, matching the
            keyword arguments of ``frogdq.catboost_model.fit_catboost``.
        """
        hp = {}
        hp['iterations'] = trial.suggest_int('catboost_iterations', *self.hp_ranges['catboost_iterations'])
        hp['early_stopping_rounds'] = trial.suggest_int(
            'catboost_early_stopping_rounds', *self.hp_ranges['catboost_early_stopping_rounds']
        )
        hp['learning_rate'] = trial.suggest_float(
            'catboost_learning_rate', *self.hp_ranges['catboost_learning_rate'][:2],
            log=(self.hp_ranges['catboost_learning_rate'][2] == 'log')
        )
        hp['depth'] = trial.suggest_int('catboost_depth', *self.hp_ranges['catboost_depth'])
        hp['l2_leaf_reg'] = trial.suggest_float(
            'catboost_l2_leaf_reg', *self.hp_ranges['catboost_l2_leaf_reg'][:2],
            log=(self.hp_ranges['catboost_l2_leaf_reg'][2] == 'log')
        )
        hp['random_strength'] = trial.suggest_float(
            'catboost_random_strength', *self.hp_ranges['catboost_random_strength'][:2],
            log=(self.hp_ranges['catboost_random_strength'][2] == 'log')
        )
        hp['bagging_temperature'] = trial.suggest_float(
            'catboost_bagging_temperature', *self.hp_ranges['catboost_bagging_temperature']
        )
        hp['border_count'] = trial.suggest_int('catboost_border_count', *self.hp_ranges['catboost_border_count'])
        hp['min_data_in_leaf'] = trial.suggest_int(
            'catboost_min_data_in_leaf', *self.hp_ranges['catboost_min_data_in_leaf']
        )
        hp['grow_policy'] = trial.suggest_categorical(
            'catboost_grow_policy', self.hp_ranges['catboost_grow_policy_choices']
        )
        return hp

    def _evaluate_single_seed(
        self,
        dataset_name: str,
        data_mode: str,
        model_type: str,
        use_curriculum: bool,
        use_gate: bool,
        hyperparams: Dict[str, Any],
        seed: int,
        preparation: str = 'standard',
    ) -> Dict[str, Any]:
        """
        Evaluate a single seed for a given configuration.

        Parameters
        ----------
        dataset_name : str
            Name of the dataset
        data_mode : str
            Data quality mode ('clean', 'ar', 'nar')
        model_type : str
            'linear' or 'mlp'
        use_curriculum : bool
            Whether to use curriculum learning
        use_gate : bool
            Whether to use gate layer
        hyperparams : dict
            Dictionary of hyperparameters
        seed : int
            Random seed for this evaluation
        preparation : str, default='standard'
            Data preparation method: 'standard' (TabularPreprocessor) or
            'autogluon' (pre-computed AutoGluon features, no further preprocessing).

        Returns
        -------
        dict
            Results dictionary with metrics
        """
        # Set random seed
        set_seed(seed)

        # Load data — branch on preparation method
        preproc_wall_t0 = time.perf_counter()
        preproc_cpu_t0  = time.process_time()

        if preparation == 'autogluon':
            (X_train, X_val, X_test), (y_train, y_val, y_test), preprocessor, metadata = (
                load_autogluon_data(
                    dataset_name=dataset_name,
                    mode=data_mode,
                    model_type=model_type,
                    seed=seed,
                    autogluon_dir=self.autogluon_data_dir,
                    clean_val=self.clean_val,
                    clean_test=self.clean_test,
                )
            )
        elif preparation == 'saga':
            (X_train, X_val, X_test), (y_train, y_val, y_test), preprocessor, metadata = (
                load_saga_data(
                    dataset_name=dataset_name,
                    mode=data_mode,
                    seed=seed,
                    saga_dir=self.saga_data_dir,
                    clean_val=self.clean_val,
                    clean_test=self.clean_test,
                    data_dir=self.data_dir,
                    poisoned_dir=self.poisoned_dir,
                )
            )
        elif preparation == 'cp':
            (X_train, X_val, X_test), (y_train, y_val, y_test), preprocessor, metadata = (
                load_cp_data(
                    dataset_name=dataset_name,
                    mode=data_mode,
                    seed=seed,
                    cp_dir=self.cp_data_dir,
                    clean_val=self.clean_val,
                    clean_test=self.clean_test,
                    data_dir=self.data_dir,
                    poisoned_dir=self.poisoned_dir,
                )
            )
        elif preparation == 'baseline_zero':
            (X_train, X_val, X_test), (y_train, y_val, y_test), preprocessor, metadata = (
                load_baseline_zero_data(
                    dataset_name=dataset_name,
                    mode=data_mode,
                    seed=seed,
                    baseline_zero_dir=self.baseline_zero_data_dir,
                    clean_val=self.clean_val,
                    clean_test=self.clean_test,
                    data_dir=self.data_dir,
                    poisoned_dir=self.poisoned_dir,
                )
            )
        elif preparation == 'knn':
            (X_train, X_val, X_test), (y_train, y_val, y_test), preprocessor, metadata = (
                load_knn_data(
                    dataset_name=dataset_name,
                    mode=data_mode,
                    seed=seed,
                    knn_dir=self.knn_data_dir,
                    clean_val=self.clean_val,
                    clean_test=self.clean_test,
                    data_dir=self.data_dir,
                    poisoned_dir=self.poisoned_dir,
                )
            )
        elif preparation == 'catboost':
            # No TabularPreprocessor here on purpose: CatBoost handles NaN and
            # categorical features natively, so raw data is used as-is (no
            # imputation, one-hot encoding, or scaling).
            (X_train, X_val, X_test), (y_train, y_val, y_test), preprocessor, metadata = (
                load_raw_data(
                    dataset_name=dataset_name,
                    mode=data_mode,
                    seed=seed,
                    clean_val=self.clean_val,
                    clean_test=self.clean_test,
                    data_dir=self.data_dir,
                    poisoned_dir=self.poisoned_dir,
                    test_dir=self.poisoned_dir,
                )
            )
        else:
            (X_train, X_val, X_test), (y_train, y_val, y_test), preprocessor, metadata = load_data(
                dataset_name=dataset_name,
                mode=data_mode,
                seed=seed,
                clean_val=self.clean_val,
                clean_test=self.clean_test,
                data_dir=self.data_dir,
                poisoned_dir=self.poisoned_dir,
                test_dir=self.poisoned_dir,
            )

        preproc_wall_time_s = time.perf_counter() - preproc_wall_t0
        preproc_cpu_time_s  = time.process_time()  - preproc_cpu_t0

        # Determine task type
        task = metadata.get('task_type', 'classification')
        if task not in ['classification', 'regression']:
            # Infer from label
            n_unique = len(np.unique(y_train))
            task = 'classification' if n_unique < 50 else 'regression'

        # Handle label encoding for classification with non-numeric labels
        label_encoder = None
        if task == 'classification':
            # Check if labels are non-numeric (object dtype or string)
            if y_train.dtype == object or np.issubdtype(y_train.dtype, np.str_) or np.issubdtype(y_train.dtype, np.str_):
                from sklearn.preprocessing import LabelEncoder
                label_encoder = LabelEncoder()
                # Fit on all labels (train + val + test) to ensure all classes are known
                all_labels = np.concatenate([y_train, y_val, y_test])
                label_encoder.fit(all_labels)
                y_train = label_encoder.transform(y_train)
                y_val = label_encoder.transform(y_val)
                y_test = label_encoder.transform(y_test)
            else:
                # Even for numeric labels, we need to ensure they are in range [0, num_classes-1]
                # Example: wine dataset has labels [1, 2, 3] which need to be mapped to [0, 1, 2]
                from sklearn.preprocessing import LabelEncoder
                label_encoder = LabelEncoder()
                all_labels = np.concatenate([y_train, y_val, y_test])
                label_encoder.fit(all_labels)
                y_train = label_encoder.transform(y_train)
                y_val = label_encoder.transform(y_val)
                y_test = label_encoder.transform(y_test)

        # Get output dimension - use all splits to avoid "target out of bounds" errors
        if task == 'classification':
            all_labels = [y_train, y_val, y_test]
            output_dim = len(np.unique(np.concatenate(all_labels)))
        else:
            output_dim = 1
            # Ensure float dtype for regression
            y_train = y_train.astype(np.float64)
            y_val = y_val.astype(np.float64)
            y_test = y_test.astype(np.float64)

        # Build model (skipped for CatBoost, which is not a torch nn.Module
        # trained via frogdq.training.fit — see the 'catboost' branch below).
        if preparation != 'catboost':
            model_kwargs = {
                'input_dim': X_train.shape[1],
                'output_dim': output_dim,
                'task': task,
            }

            if model_type == 'mlp':
                model_kwargs.update({
                    'hidden_neurons': hyperparams['hidden_neurons'],
                    'num_layers': hyperparams['num_layers'],
                    'dropout': hyperparams['dropout'],
                    'activation': hyperparams['activation'],
                    'use_batch_norm': str(hyperparams['use_batch_norm']).lower(),
                })
            else:
                # Linear model (no hidden layers)
                model_kwargs['hidden_neurons'] = hyperparams.get('hidden_neurons', 0)

            model = build_model(**model_kwargs)

            # Prepare training kwargs
            train_kwargs = {
                'model': model,
                'X_train': X_train,
                'y_train': y_train,
                'X_val': X_val,
                'y_val': y_val,
                'X_test': X_test,
                'y_test': y_test,
                'task': task,
                'epochs': hyperparams['epochs'],
                'batch_size': hyperparams['batch_size'],
                'learning_rate': hyperparams['learning_rate'],
                'optimizer': hyperparams['optimizer'],
                'weight_decay': hyperparams['weight_decay'],
                'early_stopping_patience': hyperparams['early_stopping_patience'],
                'lr_scheduler': hyperparams['lr_scheduler'],
                'random_seed': seed,
                'verbose': 0,  # Suppress training output
            }

            # Add curriculum learning parameters
            if use_curriculum:
                train_kwargs.update({
                    'use_curriculum': 'true',
                    'sample_quality': metadata['sample_quality_train'],
                    'curriculum_strategy': hyperparams['curriculum_strategy'],
                })
            else:
                train_kwargs['use_curriculum'] = 'false'

            # Add gate layer parameters
            if use_gate:
                # Convert feature quality dict to array aligned with preprocessed features
                feature_names = preprocessor.get_feature_names_out()
                feature_quality_array = np.zeros(len(feature_names))

                # Map feature quality from metadata
                for i, feat_name in enumerate(feature_names):
                    if feat_name in metadata['feature_quality']:
                        feature_quality_array[i] = metadata['feature_quality'][feat_name] / 100.0
                    else:
                        # Default to 1.0 (perfect quality) if not found
                        feature_quality_array[i] = 1.0

                train_kwargs.update({
                    'use_gate': 'true',
                    'gate_init': hyperparams['gate_init'],
                    'feature_quality': feature_quality_array,
                    'gate_loss_weight': hyperparams['gate_loss_weight'],
                    'gate_anchor_interval': hyperparams['gate_anchor_interval'],
                    'gate_quality_weighting': hyperparams['gate_quality_weighting'],
                    'gate_loss_scheduler': hyperparams['gate_loss_scheduler'],
                })
            else:
                train_kwargs['use_gate'] = 'false'

        # Train model
        try:
            train_wall_t0 = time.perf_counter()
            train_cpu_t0 = time.process_time()

            if preparation == 'catboost':
                trained_model, history = fit_catboost(
                    X_train=X_train,
                    y_train=y_train,
                    X_val=X_val,
                    y_val=y_val,
                    X_test=X_test,
                    y_test=y_test,
                    cat_features=metadata.get('categorical_features', []),
                    task=task,
                    random_seed=seed,
                    thread_count=self.catboost_thread_count,
                    verbose=0,
                    **hyperparams,
                )
            else:
                trained_model, history = fit(**train_kwargs)

            train_wall_time_s = time.perf_counter() - train_wall_t0
            train_cpu_time_s = time.process_time() - train_cpu_t0

            # Extract final metrics
            primary_metric = 'f1' if task == 'classification' else 'r2'

            # Use best-epoch index (matching the model state restored by early stopping)
            val_hist = history[f'val_{primary_metric}']
            best_epoch_idx = int(np.argmax(val_hist)) if primary_metric != 'rmse' else int(np.argmin(val_hist))

            result = {
                'seed': seed,
                'train_loss': history['train_loss'][best_epoch_idx],
                'val_loss': history['val_loss'][best_epoch_idx],
                'test_loss': history['test_loss'][best_epoch_idx] if 'test_loss' in history else None,
                f'train_{primary_metric}': history[f'train_{primary_metric}'][best_epoch_idx],
                f'val_{primary_metric}': history[f'val_{primary_metric}'][best_epoch_idx],
                f'test_{primary_metric}': history[f'test_{primary_metric}'][best_epoch_idx] if f'test_{primary_metric}' in history else None,
                'n_epochs_trained': len(history['train_loss']),
                'preproc_cpu_time': round(preproc_cpu_time_s, 4),
                'preproc_wall_time': round(preproc_wall_time_s, 4),
                'cpu_time': round(train_cpu_time_s, 4),
                'wall_time': round(train_wall_time_s, 4),
                'history': history,  # Store full history
            }

            # Add convergence metrics
            convergence_metrics = compute_convergence_metrics(history, primary_metric)
            result.update(convergence_metrics)

            # Add all other metrics
            for key in history.keys():
                if key not in ['train_loss', 'val_loss', 'test_loss', f'train_{primary_metric}',
                               f'val_{primary_metric}', f'test_{primary_metric}', 'learning_rate',
                               'gate_weights', 'gate_loss_weight']:
                    result[key] = history[key][-1]

            return result

        except Exception as e:
            if self.verbose > 0:
                print(f"  Error in seed {seed}: {e}")
            return {
                'seed': seed,
                'error': str(e),
                'train_loss': float('inf'),
                'val_loss': float('inf'),
                'test_loss': float('inf'),
            }

    def _save_trial_histories(self, study_name: str, trial_number: int, results: List[Dict[str, Any]]) -> None:
        """
        Save trial histories to disk immediately after trial completion.

        Parameters
        ----------
        study_name : str
            Name of the study
        trial_number : int
            Trial number
        results : List[Dict[str, Any]]
            List of result dictionaries containing 'history' key
        """
        if not self.save_npz_histories:
            return

        histories_to_save = {}

        for seed_result in results:
            if 'error' not in seed_result and 'history' in seed_result:
                history = seed_result['history']
                seed = seed_result['seed']

                # Convert histories to numpy arrays for storage
                for metric_name, metric_values in history.items():
                    if metric_name == 'gate_weights':
                        # Gate weights are 2D arrays (epochs x features)
                        histories_to_save[f"trial{trial_number}_seed{seed}_{metric_name}"] = np.array(metric_values)
                    else:
                        # Other metrics are 1D arrays
                        histories_to_save[f"trial{trial_number}_seed{seed}_{metric_name}"] = np.array(metric_values)

        if histories_to_save:
            # Save to individual trial history file
            trial_history_path = self.output_dir / f"{study_name}_trial{trial_number}_histories.npz"
            np.savez_compressed(trial_history_path, **histories_to_save)

    def _load_baseline_params(self, dataset_name: str, data_mode: str, model_type: str) -> Optional[List[Dict[str, Any]]]:
        """
        Load top-N hyperparameter configurations from baseline run (no curriculum, no gates).

        For AR/NAR modes, loads from the corresponding AR/NAR baseline run, not clean.
        This makes sense because AR/NAR baseline is already optimized for degraded data.

        Parameters
        ----------
        dataset_name : str
            Name of the dataset
        data_mode : str
            Data mode ('ar' or 'nar')
        model_type : str
            'linear' or 'mlp'

        Returns
        -------
        Optional[List[Dict[str, Any]]]
            List of top-N parameter configurations, or None if not available
        """
        cache_key = f"{dataset_name}_{data_mode}_{model_type}_baseline"

        # Check cache first
        if cache_key in self._clean_params_cache:
            return self._clean_params_cache[cache_key]

        # Load from baseline study (no curriculum, no gates)
        study_name = f"{dataset_name}_{data_mode}_{model_type}_curr0_gate0"

        try:
            # Load existing study
            study = optuna.load_study(
                study_name=study_name,
                storage=self.storage_url,
                sampler=TPESampler(seed=self.optuna_sampler_seed),
            )

            # Get top-N trials
            top_trials = sorted(
                [t for t in study.trials if t.value is not None],
                key=lambda t: t.value,
                reverse=True
            )[:self.reuse_top_n]

            if not top_trials:
                if self.verbose > 0:
                    print(f"  No completed baseline trials found for {cache_key}")
                return None

            # Extract parameters
            params_list = [trial.params for trial in top_trials]

            # Cache for future use
            self._clean_params_cache[cache_key] = params_list

            if self.verbose > 0:
                print(f"  Loaded {len(params_list)} baseline parameter sets for {cache_key}")

            return params_list

        except KeyError:
            if self.verbose > 0:
                print(f"  Baseline study not found: {study_name}")
            return None

    def _get_study_name(
        self,
        dataset_name: str,
        data_mode: str,
        model_type: str,
        use_curriculum: bool,
        use_gate: bool,
        preparation: str,
    ) -> str:
        if preparation == 'autogluon':
            return f"{dataset_name}_{data_mode}_{model_type}_ag"
        elif preparation == 'saga':
            return f"{dataset_name}_{data_mode}_{model_type}_saga"
        elif preparation == 'cp':
            return f"{dataset_name}_{data_mode}_{model_type}_cp"
        elif preparation == 'baseline_zero':
            return f"{dataset_name}_{data_mode}_{model_type}_baseline_zero"
        elif preparation == 'knn':
            return f"{dataset_name}_{data_mode}_{model_type}_knn"
        elif preparation == 'catboost':
            return f"{dataset_name}_{data_mode}_catboost"
        else:
            return f"{dataset_name}_{data_mode}_{model_type}_curr{int(use_curriculum)}_gate{int(use_gate)}"

    def _objective(
        self,
        trial: optuna.Trial,
        dataset_name: str,
        data_mode: str,
        model_type: str,
        use_curriculum: bool,
        use_gate: bool,
        preparation: str = 'standard',
    ) -> float:
        """
        Objective function for Optuna optimization.

        This function is called by Optuna for each trial. It:
        1. Suggests hyperparameters
        2. Evaluates the configuration across multiple seeds (in parallel)
        3. Returns the average validation metric

        Parameters
        ----------
        trial : optuna.Trial
            Optuna trial object
        dataset_name : str
            Name of the dataset
        data_mode : str
            Data quality mode
        model_type : str
            'linear' or 'mlp'
        use_curriculum : bool
            Whether to use curriculum learning
        use_gate : bool
            Whether to use gate layer
        preparation : str, default='standard'
            Data preparation method ('standard' or 'autogluon').

        Returns
        -------
        float
            Average validation metric across seeds (to maximize)
        """
        # Load a sample to determine task type
        try:
            if preparation == 'autogluon':
                _, (y_train, _, _), _, metadata = load_autogluon_data(
                    dataset_name=dataset_name,
                    mode=data_mode,
                    model_type=model_type,
                    seed=self.seed_start,
                    autogluon_dir=self.autogluon_data_dir,
                    clean_val=self.clean_val,
                    clean_test=self.clean_test,
                )
            elif preparation == 'saga':
                _, (y_train, _, _), _, metadata = load_saga_data(
                    dataset_name=dataset_name,
                    mode=data_mode,
                    seed=self.seed_start,
                    saga_dir=self.saga_data_dir,
                    clean_val=self.clean_val,
                    clean_test=self.clean_test,
                    data_dir=self.data_dir,
                    poisoned_dir=self.poisoned_dir,
                )
            elif preparation == 'cp':
                _, (y_train, _, _), _, metadata = load_cp_data(
                    dataset_name=dataset_name,
                    mode=data_mode,
                    seed=self.seed_start,
                    cp_dir=self.cp_data_dir,
                    clean_val=self.clean_val,
                    clean_test=self.clean_test,
                    data_dir=self.data_dir,
                    poisoned_dir=self.poisoned_dir,
                )
            elif preparation == 'baseline_zero':
                _, (y_train, _, _), _, metadata = load_baseline_zero_data(
                    dataset_name=dataset_name,
                    mode=data_mode,
                    seed=self.seed_start,
                    baseline_zero_dir=self.baseline_zero_data_dir,
                    clean_val=self.clean_val,
                    clean_test=self.clean_test,
                    data_dir=self.data_dir,
                    poisoned_dir=self.poisoned_dir,
                )
            elif preparation == 'knn':
                _, (y_train, _, _), _, metadata = load_knn_data(
                    dataset_name=dataset_name,
                    mode=data_mode,
                    seed=self.seed_start,
                    knn_dir=self.knn_data_dir,
                    clean_val=self.clean_val,
                    clean_test=self.clean_test,
                    data_dir=self.data_dir,
                    poisoned_dir=self.poisoned_dir,
                )
            elif preparation == 'catboost':
                _, (y_train, _, _), _, metadata = load_raw_data(
                    dataset_name=dataset_name,
                    mode=data_mode,
                    seed=self.seed_start,
                    clean_val=self.clean_val,
                    clean_test=self.clean_test,
                    data_dir=self.data_dir,
                    poisoned_dir=self.poisoned_dir,
                    test_dir=self.poisoned_dir,
                )
            else:
                _, (y_train, _, _), _, metadata = load_data(
                    dataset_name=dataset_name,
                    mode=data_mode,
                    seed=self.seed_start,
                    clean_val=self.clean_val,
                    clean_test=self.clean_test,
                    data_dir=self.data_dir,
                    poisoned_dir=self.poisoned_dir,
                    test_dir=self.poisoned_dir,
                )
            task = metadata.get('task_type', 'classification')
            if task not in ['classification', 'regression']:
                n_unique = len(np.unique(y_train))
                task = 'classification' if n_unique < 50 else 'regression'
        except Exception as e:
            if self.verbose > 0:
                print(f"  Error loading dataset {dataset_name}: {e}")
            raise optuna.TrialPruned()

        # Suggest hyperparameters (or reuse from baseline run if applicable)
        # Only reuse parameters for MLP models with curriculum/gates (not baseline)
        if self.reuse_params and data_mode in ['ar', 'nar'] and (use_curriculum or use_gate):
            # Try to load baseline parameters from the same data mode
            baseline_params_list = self._load_baseline_params(dataset_name, data_mode, model_type)

            if baseline_params_list:
                # Sample one configuration from top-N baseline runs (shallow copy to
                # avoid mutating the cached dict when curriculum/gate keys are added below)
                rng = np.random.RandomState(trial.number + self.optuna_sampler_seed)
                hyperparams = dict(rng.choice(baseline_params_list))

                # Now suggest only the curriculum/gate specific parameters
                if use_curriculum:
                    hyperparams['curriculum_strategy'] = trial.suggest_categorical(
                        'curriculum_strategy', self.hp_ranges['curriculum_strategy_choices']
                    )

                if use_gate:
                    hyperparams['gate_init'] = trial.suggest_categorical(
                        'gate_init', self.hp_ranges['gate_init_choices']
                    )
                    hyperparams['gate_loss_weight'] = trial.suggest_float(
                        'gate_loss_weight', *self.hp_ranges['gate_loss_weight'][:2],
                        log=(self.hp_ranges['gate_loss_weight'][2] == 'log')
                    )
                    hyperparams['gate_anchor_interval'] = trial.suggest_int(
                        'gate_anchor_interval', *self.hp_ranges['gate_anchor_interval']
                    )
                    hyperparams['gate_quality_weighting'] = trial.suggest_categorical(
                        'gate_quality_weighting', self.hp_ranges['gate_quality_weighting_choices']
                    )
                    hyperparams['gate_loss_scheduler'] = trial.suggest_categorical(
                        'gate_loss_scheduler', self.hp_ranges['gate_loss_scheduler_choices']
                    )

                # Log which parameters we're reusing
                trial.set_user_attr('reused_baseline_params', True)

                if self.verbose > 1:
                    print(f"  Trial {trial.number}: Reusing baseline params, optimizing curriculum/gate only")
            else:
                # Fall back to normal optimization
                hyperparams = self._suggest_hyperparameters(trial, model_type, use_curriculum, use_gate, task)
                if self.verbose > 1:
                    print(f"  Trial {trial.number}: Baseline params not available, full optimization")
        elif preparation == 'catboost':
            hyperparams = self._suggest_catboost_hyperparameters(trial)
        else:
            # Normal hyperparameter optimization
            hyperparams = self._suggest_hyperparameters(trial, model_type, use_curriculum, use_gate, task)

        # Generate seeds
        seeds = [self.seed_start + i for i in range(self.n_seeds)]

        # Evaluate across seeds in parallel
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")

            results = Parallel(n_jobs=self.n_jobs_seeds, backend='loky')(
                delayed(self._evaluate_single_seed)(
                    dataset_name, data_mode, model_type, use_curriculum, use_gate,
                    hyperparams, seed, preparation
                )
                for seed in seeds
            )

        # Compute average validation and test metrics across seeds.
        # Use best_val_metric (computed by compute_convergence_metrics) so that
        # Optuna ranks trials by the performance of the restored best-epoch model,
        # not by the last-epoch metric (which is lower when early stopping triggers).
        primary_metric = 'f1' if task == 'classification' else 'r2'
        valid_results = [r for r in results if 'error' not in r]
        val_metrics = [r.get('best_val_metric', r.get(f'val_{primary_metric}', float('-inf'))) for r in valid_results]
        test_metrics = [
            r[f'test_{primary_metric}'] for r in valid_results
            if r.get(f'test_{primary_metric}') is not None
        ]

        if not val_metrics:
            raise optuna.TrialPruned()

        avg_val_metric = np.mean(val_metrics)
        avg_test_metric = float(np.mean(test_metrics)) if test_metrics else float('-inf')

        # Extract histories and create clean results for Optuna storage
        # (histories contain numpy arrays which are not JSON serializable)
        results_for_optuna = []

        for r in results:
            if 'error' not in r:
                # Create a copy without history for Optuna storage
                r_copy = {k: v for k, v in r.items() if k != 'history'}
                results_for_optuna.append(r_copy)
            else:
                results_for_optuna.append(r)

        # Store trial results (without histories to avoid JSON serialization issues)
        trial.set_user_attr('seed_results', results_for_optuna)
        trial.set_user_attr('avg_val_metric', float(avg_val_metric))
        trial.set_user_attr('std_val_metric', float(np.std(val_metrics)))
        trial.set_user_attr('avg_test_metric', avg_test_metric)
        trial.set_user_attr('std_test_metric', float(np.std(test_metrics)) if test_metrics else 0.0)

        # Histories are persisted to disk by _save_trial_histories below.
        # trial.user_attrs returns a copy in Optuna, so direct assignment is a no-op;
        # use set_user_attr if in-memory recovery is ever needed in the future.

        # Save histories to disk immediately (so they're available even after warm_start)
        study_name = self._get_study_name(
            dataset_name, data_mode, model_type, use_curriculum, use_gate, preparation
        )
        self._save_trial_histories(study_name, trial.number, results)

        # For both classification (F1) and regression (R2), higher is better
        return avg_val_metric

    def run_experiment(
        self,
        dataset_name: str,
        data_mode: str,
        model_type: str,
        use_curriculum: bool,
        use_gate: bool,
        preparation: str = 'standard',
    ) -> Tuple[Study, pd.DataFrame]:
        """
        Run a single Optuna experiment for a given configuration.

        With warm_start enabled, this method will:
        - Load existing study from database if it exists
        - Skip if already completed (n_trials reached)
        - Resume from last checkpoint if incomplete
        - Create new study if not found

        Parameters
        ----------
        dataset_name : str
            Name of the dataset
        data_mode : str
            Data quality mode ('clean', 'ar', 'nar')
        model_type : str
            'linear' or 'mlp'
        use_curriculum : bool
            Whether to use curriculum learning
        use_gate : bool
            Whether to use gate layer
        preparation : str, default='standard'
            Data preparation method ('standard' or 'autogluon').
            AutoGluon experiments use pre-computed features; no further
            preprocessing is applied inside the Optuna objective.

        Returns
        -------
        study : optuna.Study
            Completed Optuna study
        top_results : pd.DataFrame
            DataFrame with top-M trial results
        """
        study_name = self._get_study_name(
            dataset_name, data_mode, model_type, use_curriculum, use_gate, preparation
        )

        # Check if study exists and determine trials to run
        existing_study = None
        trials_completed = 0

        if self.warm_start:
            try:
                # Try to load existing study
                existing_study = optuna.load_study(
                    study_name=study_name,
                    storage=self.storage_url,
                )
                trials_completed = len([t for t in existing_study.trials if t.state == optuna.trial.TrialState.COMPLETE])

                # Curriculum-only on AR/NAR always uses 3 trials (3 strategies)
                target_trials = self.n_trials
                if data_mode in ['ar', 'nar'] and use_curriculum and not use_gate:
                    target_trials = 3

                if trials_completed >= target_trials:
                    if self.verbose > 0:
                        print(f"\n{'='*80}")
                        print(f"Study already complete: {study_name}")
                        print(f"Completed trials: {trials_completed}/{target_trials}")
                        print(f"Skipping...")
                        print(f"{'='*80}")
                    study = existing_study
                else:
                    if self.verbose > 0:
                        print(f"\n{'='*80}")
                        print(f"Resuming experiment: {study_name}")
                        print(f"Completed trials: {trials_completed}/{target_trials}")
                        print(f"Remaining trials: {target_trials - trials_completed}")
                        print(f"{'='*80}")
                    existing_study.optimize(
                        lambda trial: self._objective(trial, dataset_name, data_mode, model_type,
                                                      use_curriculum, use_gate, preparation),
                        n_trials=target_trials - trials_completed,
                        n_jobs=1,
                        show_progress_bar=(self.verbose > 0),
                    )
                    study = existing_study

            except KeyError:
                # Study doesn't exist yet, will create new one
                if self.verbose > 0:
                    print(f"\n{'='*80}")
                    print(f"Starting new experiment: {study_name}")
                    print(f"{'='*80}")
                existing_study = None
        else:
            if self.verbose > 0:
                print(f"\n{'='*80}")
                print(f"Running experiment: {study_name}")
                print(f"{'='*80}")

        # Create new study if not resuming
        if existing_study is None:
            sampler = TPESampler(seed=self.optuna_sampler_seed)
            study = optuna.create_study(
                direction='maximize',  # Maximize F1 (classification) or R2 (regression)
                sampler=sampler,
                study_name=study_name,
                storage=self.storage_url if self.warm_start else None,
                load_if_exists=False,
            )

            # Determine number of trials
            # Curriculum-only on AR/NAR: only 3 strategies to try
            n_trials = self.n_trials
            if data_mode in ['ar', 'nar'] and use_curriculum and not use_gate:
                n_trials = 3
                if self.verbose > 0:
                    print(f"Using {n_trials} trials for curriculum-only experiment (3 strategies to test)")
            elif self.verbose > 0 and n_trials != self.n_trials:
                print(f"Using {n_trials} trials")

            # Run optimization
            study.optimize(
                lambda trial: self._objective(trial, dataset_name, data_mode, model_type,
                                              use_curriculum, use_gate, preparation),
                n_trials=n_trials,
                n_jobs=self.n_jobs_optuna,
                show_progress_bar=(self.verbose > 0),
            )

        # Extract ALL trial results (not just top-M)
        # Sort by value descending (best first)
        all_trials = sorted(
            [t for t in study.trials if t.value is not None],
            key=lambda t: t.value,
            reverse=True
        )

        results_list = []
        histories_dict = {}  # Store histories separately

        # Process all trials and assign ranks
        for rank, trial in enumerate(all_trials, 1):
            # Try to get full results with histories (only available for current run)
            full_results = trial.user_attrs.get('_full_results_with_histories', None)

            # Fall back to stored results without histories (for warm-started runs)
            if full_results is None:
                full_results = trial.user_attrs.get('seed_results', [])

            for seed_result in full_results:
                if 'error' not in seed_result:
                    # Extract history (may be None if loading from DB)
                    history = seed_result.get('history', None)

                    # Create result dict without history for CSV
                    result_dict = {
                        'dataset': dataset_name,
                        'data_mode': data_mode,
                        'model_type': model_type,
                        'use_curriculum': use_curriculum,
                        'use_gate': use_gate,
                        'preparation': preparation,
                        'rank': rank,
                        'trial_number': trial.number,
                        'trial_value': trial.value,
                        **trial.params,
                        **{k: v for k, v in seed_result.items() if k != 'history'},
                    }
                    results_list.append(result_dict)

                    # Store history with unique key (if available, and only for top-M for space efficiency)
                    if history is not None and rank <= self.top_m:
                        history_key = f"rank{rank}_trial{trial.number}_seed{seed_result['seed']}"
                        histories_dict[history_key] = history

        top_results = pd.DataFrame(results_list)

        # Save results CSV
        results_path = self.output_dir / f"{study_name}_results.csv"
        top_results.to_csv(results_path, index=False)

        # Consolidate individual trial histories into a single comprehensive npz file for top-M trials
        # Individual trial npz files are saved during optimization (see _save_trial_histories)
        # Here we create a consolidated file for the top-M trials for easier analysis
        if self.save_npz_histories:
            top_m_trials = all_trials[:self.top_m]
            histories_to_save = {}

            for rank, trial in enumerate(top_m_trials, 1):
                # Try to load from individual trial history file
                trial_history_path = self.output_dir / f"{study_name}_trial{trial.number}_histories.npz"

                if trial_history_path.exists():
                    # Load and add to consolidated dict with rank prefix
                    trial_histories = np.load(trial_history_path)
                    for key in trial_histories.keys():
                        # Rename key to use rank instead of trial number for easier interpretation
                        # Original: trial123_seed42_train_loss
                        # New: rank1_trial123_seed42_train_loss
                        new_key = f"rank{rank}_{key}"
                        histories_to_save[new_key] = trial_histories[key]
                elif rank <= self.top_m:
                    # For current run, histories might be in memory
                    full_results = trial.user_attrs.get('_full_results_with_histories', None)
                    if full_results:
                        for seed_result in full_results:
                            if 'error' not in seed_result and 'history' in seed_result:
                                history = seed_result['history']
                                seed = seed_result['seed']
                                for metric_name, metric_values in history.items():
                                    key = f"rank{rank}_trial{trial.number}_seed{seed}_{metric_name}"
                                    histories_to_save[key] = np.array(metric_values)

            if histories_to_save:
                histories_path = self.output_dir / f"{study_name}_top{self.top_m}_histories.npz"
                np.savez_compressed(histories_path, **histories_to_save)
                if self.verbose > 0:
                    print(f"Top-{self.top_m} histories saved to: {histories_path}")

        if self.verbose > 0:
            print(f"\nBest trial value: {study.best_trial.value:.4f}")
            print(f"Results saved to: {results_path}")
            if self.save_npz_histories:
                print(f"Individual trial histories: {study_name}_trial*_histories.npz")

        return study, top_results

    def _build_experiment_list(self) -> List[Dict[str, Any]]:
        experiments = []
        ar_nar_modes = [m for m in self.data_modes if m in ('ar', 'nar')]

        for dataset in self.datasets:
            for model_type in self.model_types:
                if 'clean' in self.data_modes:
                    experiments.append({
                        'dataset': dataset, 'data_mode': 'clean', 'model_type': model_type,
                        'use_curriculum': False, 'use_gate': False, 'preparation': 'standard',
                    })

                for data_mode in ar_nar_modes:
                    experiments.append({
                        'dataset': dataset, 'data_mode': data_mode, 'model_type': model_type,
                        'use_curriculum': False, 'use_gate': False, 'preparation': 'standard',
                    })

                    if self.run_autogluon:
                        experiments.append({
                            'dataset': dataset, 'data_mode': data_mode, 'model_type': model_type,
                            'use_curriculum': False, 'use_gate': False, 'preparation': 'autogluon',
                        })

                    if self.run_cp:
                        experiments.append({
                            'dataset': dataset, 'data_mode': data_mode, 'model_type': model_type,
                            'use_curriculum': False, 'use_gate': False, 'preparation': 'cp',
                        })

                    if self.run_saga:
                        experiments.append({
                            'dataset': dataset, 'data_mode': data_mode, 'model_type': model_type,
                            'use_curriculum': False, 'use_gate': False, 'preparation': 'saga',
                        })

                    if self.run_baseline_zero:
                        experiments.append({
                            'dataset': dataset, 'data_mode': data_mode, 'model_type': model_type,
                            'use_curriculum': False, 'use_gate': False, 'preparation': 'baseline_zero',
                        })

                    if self.run_knn:
                        experiments.append({
                            'dataset': dataset, 'data_mode': data_mode, 'model_type': model_type,
                            'use_curriculum': False, 'use_gate': False, 'preparation': 'knn',
                        })

                    experiments.append({
                        'dataset': dataset, 'data_mode': data_mode, 'model_type': model_type,
                        'use_curriculum': True, 'use_gate': False, 'preparation': 'standard',
                    })

                    experiments.append({
                        'dataset': dataset, 'data_mode': data_mode, 'model_type': model_type,
                        'use_curriculum': False, 'use_gate': True, 'preparation': 'standard',
                    })

                    experiments.append({
                        'dataset': dataset, 'data_mode': data_mode, 'model_type': model_type,
                        'use_curriculum': True, 'use_gate': True, 'preparation': 'standard',
                    })

            # CatBoost baseline: a model baseline (not a data-preparation
            # baseline like autogluon/cp/saga above), so it runs once per
            # dataset/data_mode, independent of model_types.
            if self.run_catboost:
                if 'clean' in self.data_modes:
                    experiments.append({
                        'dataset': dataset, 'data_mode': 'clean', 'model_type': 'catboost',
                        'use_curriculum': False, 'use_gate': False, 'preparation': 'catboost',
                    })
                for data_mode in ar_nar_modes:
                    experiments.append({
                        'dataset': dataset, 'data_mode': data_mode, 'model_type': 'catboost',
                        'use_curriculum': False, 'use_gate': False, 'preparation': 'catboost',
                    })

        return experiments

    def run_all_experiments(self) -> pd.DataFrame:
        """
        Run all experiments across all configurations.

        Experiment design:
        1. Baselines:
           - linear on clean/ar/nar (baseline, no curriculum, no gates)
           - mlp on clean/ar/nar (baseline, no curriculum, no gates)
        2. State-of-art:
           - mlp + curriculum on ar/nar
        3. Proposed approach:
           - mlp + gate on ar/nar
           - mlp + gate + curriculum on ar/nar

        Returns
        -------
        pd.DataFrame
            Combined results from all experiments
        """
        all_results = []

        # Generate experiment list matching the benchmark per model:
        #  1. Clean baseline
        #  2. AR/NAR poisoned baseline
        #  3. AR/NAR + AutoGluon  (if run_autogluon)
        #  4. AR/NAR + CP         (if run_cp)
        #  5. AR/NAR + Saga       (if run_saga)
        #  6. AR/NAR + curriculum
        #  7. AR/NAR + gate
        #  8. AR/NAR + gate + curriculum
        # All configs apply symmetrically to both Linear and MLP.
        experiments = []
        ar_nar_modes = [m for m in self.data_modes if m in ('ar', 'nar')]

        for dataset in self.datasets:
            for model_type in self.model_types:
                # 1. Clean baseline
                if 'clean' in self.data_modes:
                    experiments.append({
                        'dataset': dataset,
                        'data_mode': 'clean',
                        'model_type': model_type,
                        'use_curriculum': False,
                        'use_gate': False,
                        'preparation': 'standard',
                    })

                for data_mode in ar_nar_modes:
                    # 2. Poisoned baseline
                    experiments.append({
                        'dataset': dataset,
                        'data_mode': data_mode,
                        'model_type': model_type,
                        'use_curriculum': False,
                        'use_gate': False,
                        'preparation': 'standard',
                    })

                    # 3. AutoGluon data-preparation baseline
                    if self.run_autogluon:
                        experiments.append({
                            'dataset': dataset,
                            'data_mode': data_mode,
                            'model_type': model_type,
                            'use_curriculum': False,
                            'use_gate': False,
                            'preparation': 'autogluon',
                        })

                    # 4. CP (custom pipeline) data-preparation baseline
                    if self.run_cp:
                        experiments.append({
                            'dataset': dataset,
                            'data_mode': data_mode,
                            'model_type': model_type,
                            'use_curriculum': False,
                            'use_gate': False,
                            'preparation': 'cp',
                        })

                    # 5. Saga data-preparation baseline
                    if self.run_saga:
                        experiments.append({
                            'dataset': dataset,
                            'data_mode': data_mode,
                            'model_type': model_type,
                            'use_curriculum': False,
                            'use_gate': False,
                            'preparation': 'saga',
                        })

                    # 6. Baseline 0 (zero/random imputation)
                    if self.run_baseline_zero:
                        experiments.append({
                            'dataset': dataset,
                            'data_mode': data_mode,
                            'model_type': model_type,
                            'use_curriculum': False,
                            'use_gate': False,
                            'preparation': 'baseline_zero',
                        })

                    # 7. KNN imputation baseline
                    if self.run_knn:
                        experiments.append({
                            'dataset': dataset,
                            'data_mode': data_mode,
                            'model_type': model_type,
                            'use_curriculum': False,
                            'use_gate': False,
                            'preparation': 'knn',
                        })

                    # 9. Curriculum learning
                    experiments.append({
                        'dataset': dataset,
                        'data_mode': data_mode,
                        'model_type': model_type,
                        'use_curriculum': True,
                        'use_gate': False,
                        'preparation': 'standard',
                    })

                    # 10. quAIL gate
                    experiments.append({
                        'dataset': dataset,
                        'data_mode': data_mode,
                        'model_type': model_type,
                        'use_curriculum': False,
                        'use_gate': True,
                        'preparation': 'standard',
                    })

                    # 11. quAIL gate + curriculum
                    if self.config.get('run_gate_curriculum', True):
                        experiments.append({
                            'dataset': dataset,
                            'data_mode': data_mode,
                            'model_type': model_type,
                            'use_curriculum': True,
                            'use_gate': True,
                            'preparation': 'standard',
                        })

            # 12. CatBoost baseline: a model baseline (not a data-preparation
            # baseline like AutoGluon/CP/Saga above), so it runs once per
            # dataset/data_mode, independent of model_types. Trained on raw
            # data (NaN preserved, categorical features native) — no
            # imputation, one-hot encoding, or scaling.
            if self.run_catboost:
                if 'clean' in self.data_modes:
                    experiments.append({
                        'dataset': dataset,
                        'data_mode': 'clean',
                        'model_type': 'catboost',
                        'use_curriculum': False,
                        'use_gate': False,
                        'preparation': 'catboost',
                    })
                for data_mode in ar_nar_modes:
                    experiments.append({
                        'dataset': dataset,
                        'data_mode': data_mode,
                        'model_type': 'catboost',
                        'use_curriculum': False,
                        'use_gate': False,
                        'preparation': 'catboost',
                    })

        total_experiments = len(experiments)
        experiment_count = 0

        for exp in experiments:
            experiment_count += 1

            if self.verbose > 0:
                print(f"\n\nExperiment {experiment_count}/{total_experiments}")

            try:
                study, top_results = self.run_experiment(
                    dataset_name=exp['dataset'],
                    data_mode=exp['data_mode'],
                    model_type=exp['model_type'],
                    use_curriculum=exp['use_curriculum'],
                    use_gate=exp['use_gate'],
                    preparation=exp.get('preparation', 'standard'),
                )

                all_results.append(top_results)

            except Exception as e:
                if self.verbose > 0:
                    print(f"Error in experiment: {e}")
                continue

        # Combine all results
        if all_results:
            combined_results = pd.concat(all_results, ignore_index=True)

            # Save combined results
            combined_path = self.output_dir / "all_experiments_results.csv"
            combined_results.to_csv(combined_path, index=False)

            if self.verbose > 0:
                print(f"\n\n{'='*80}")
                print(f"All experiments completed!")
                print(f"Combined results saved to: {combined_path}")
                print(f"{'='*80}")

            return combined_results
        else:
            print("No experiments completed successfully.")
            return pd.DataFrame()


    def run_final_evaluation(
        self,
        top_k: int = 1,
        n_final_seeds: int = 10,
        seed_offset: int = 1000,
    ) -> pd.DataFrame:
        """
        Re-evaluate the top-k configurations from each completed study using a disjoint seed range.

        Takes the best trial hyperparameters from each study and reruns training with
        seeds [seed_start + seed_offset, ..., seed_start + seed_offset + n_final_seeds - 1].
        These seeds are disjoint from the HPO seeds, giving an unbiased estimate of
        test performance suitable for reporting mean ± std in a paper.

        Parameters
        ----------
        top_k : int, default=1
            Number of top trials per study to re-evaluate.
        n_final_seeds : int, default=10
            Number of seeds for the final evaluation.
        seed_offset : int, default=1000
            Offset from seed_start to generate disjoint seeds.

        Returns
        -------
        pd.DataFrame
            One row per (study, rank, seed). Aggregated mean/std over seeds are
            included as final_test_mean, final_test_std, final_val_mean, final_val_std.
        """
        final_seeds = [self.seed_start + seed_offset + i for i in range(n_final_seeds)]
        experiments = self._build_experiment_list()
        all_rows = []

        total = len(experiments)
        for idx, exp in enumerate(experiments, 1):
            dataset_name = exp['dataset']
            data_mode = exp['data_mode']
            model_type = exp['model_type']
            use_curriculum = exp['use_curriculum']
            use_gate = exp['use_gate']
            preparation = exp.get('preparation', 'standard')

            study_name = self._get_study_name(
                dataset_name, data_mode, model_type, use_curriculum, use_gate, preparation
            )

            try:
                study = optuna.load_study(study_name=study_name, storage=self.storage_url)
            except KeyError:
                if self.verbose > 0:
                    print(f"[{idx}/{total}] Study not found: {study_name}, skipping.")
                continue

            completed_trials = sorted(
                [t for t in study.trials if t.value is not None],
                key=lambda t: t.value,
                reverse=True,
            )[:top_k]

            if not completed_trials:
                if self.verbose > 0:
                    print(f"[{idx}/{total}] No completed trials for {study_name}, skipping.")
                continue

            if self.verbose > 0:
                print(f"\n[{idx}/{total}] Final eval: {study_name} "
                      f"(top-{len(completed_trials)}, {n_final_seeds} seeds)")

            for rank, trial in enumerate(completed_trials, 1):
                with warnings.catch_warnings():
                    warnings.filterwarnings("ignore")
                    results = Parallel(n_jobs=self.n_jobs_seeds, backend='loky')(
                        delayed(self._evaluate_single_seed)(
                            dataset_name, data_mode, model_type,
                            use_curriculum, use_gate, trial.params, seed, preparation,
                        )
                        for seed in final_seeds
                    )

                valid_results = [r for r in results if 'error' not in r]
                primary_metric = 'f1' if any('val_f1' in r for r in valid_results) else 'r2'

                test_metrics = [r[f'test_{primary_metric}'] for r in valid_results
                                if r.get(f'test_{primary_metric}') is not None]
                val_metrics = [r.get('best_val_metric', r.get(f'val_{primary_metric}', float('nan')))
                               for r in valid_results]

                final_test_mean = float(np.mean(test_metrics)) if test_metrics else float('nan')
                final_test_std = float(np.std(test_metrics)) if test_metrics else float('nan')
                final_val_mean = float(np.mean(val_metrics)) if val_metrics else float('nan')
                final_val_std = float(np.std(val_metrics)) if val_metrics else float('nan')

                for seed_result in valid_results:
                    all_rows.append({
                        'dataset': dataset_name,
                        'data_mode': data_mode,
                        'model_type': model_type,
                        'use_curriculum': use_curriculum,
                        'use_gate': use_gate,
                        'preparation': preparation,
                        'study_name': study_name,
                        'hpo_rank': rank,
                        'trial_number': trial.number,
                        'hpo_val_metric': trial.value,
                        'final_test_mean': final_test_mean,
                        'final_test_std': final_test_std,
                        'final_val_mean': final_val_mean,
                        'final_val_std': final_val_std,
                        **{k: v for k, v in seed_result.items() if k != 'history'},
                    })

                if self.verbose > 0:
                    print(f"  rank {rank} | val {final_val_mean:.4f} ± {final_val_std:.4f} | "
                          f"test {final_test_mean:.4f} ± {final_test_std:.4f}")

        if not all_rows:
            return pd.DataFrame()

        results_df = pd.DataFrame(all_rows)
        save_path = self.output_dir / "final_evaluation_results.csv"
        results_df.to_csv(save_path, index=False)

        if self.verbose > 0:
            print(f"\nFinal evaluation saved to: {save_path}")

        return results_df


def load_config(config_path: str) -> Dict[str, Any]:
    """
    Load experiment configuration from YAML file.

    Parameters
    ----------
    config_path : str
        Path to YAML configuration file

    Returns
    -------
    dict
        Configuration dictionary
    """
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def run_experiments_from_config(config_path: str) -> pd.DataFrame:
    """
    Run experiments from a YAML configuration file.

    Parameters
    ----------
    config_path : str
        Path to YAML configuration file

    Returns
    -------
    pd.DataFrame
        Combined results from all experiments
    """
    config = load_config(config_path)
    experiment = OptunaExperiment(config)
    return experiment.run_all_experiments()
