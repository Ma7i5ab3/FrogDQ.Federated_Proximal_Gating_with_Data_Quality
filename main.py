#!/usr/bin/env python3
"""
Main entry point for running FrogDQ hyperparameter optimization experiments.

This script provides a command-line interface for running Optuna-based
hyperparameter optimization across different model configurations, data
quality modes, and datasets.

Usage:
    # Run with default config file
    python main.py

    # Run with custom config file
    python main.py --config my_config.yaml

    # Run quick test (overrides config)
    python main.py --quick-test

    # List available datasets
    python main.py --list-datasets

Examples:
    # Full experiment run
    python main.py --config config.yaml

    # Quick test on iris dataset
    python main.py --quick-test --datasets iris

    # Run on specific datasets with custom settings
    python main.py --config config.yaml --datasets iris wine --n-trials 10
"""

import argparse
import sys
from pathlib import Path

import pandas as pd

from frogdq.data import get_datasets
from frogdq.optimization import OptunaExperiment, load_config


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description='Run FrogDQ hyperparameter optimization experiments',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )

    parser.add_argument(
        '--config',
        type=str,
        default='config.yaml',
        help='Path to YAML configuration file (default: config.yaml)'
    )

    parser.add_argument(
        '--list-datasets',
        action='store_true',
        help='List all available datasets and exit'
    )

    parser.add_argument(
        '--quick-test',
        action='store_true',
        help='Run a quick test with minimal trials (overrides config)'
    )

    # Override options (override config file settings)
    parser.add_argument(
        '--datasets',
        nargs='+',
        help='Override datasets to run on (space-separated list)'
    )

    parser.add_argument(
        '--data-modes',
        nargs='+',
        choices=['clean', 'ar', 'nar'],
        help='Override data quality modes to test'
    )

    parser.add_argument(
        '--model-types',
        nargs='+',
        choices=['linear', 'mlp'],
        help='Override model types to test'
    )

    parser.add_argument(
        '--n-trials',
        type=int,
        help='Override number of Optuna trials per configuration'
    )

    parser.add_argument(
        '--n-seeds',
        type=int,
        help='Override number of random seeds to evaluate per trial'
    )

    parser.add_argument(
        '--n-jobs-seeds',
        type=int,
        help='Override number of parallel jobs for seed evaluation'
    )

    parser.add_argument(
        '--n-jobs-optuna',
        type=int,
        help='Override number of parallel jobs for Optuna trials'
    )

    parser.add_argument(
        '--output-dir',
        type=str,
        help='Override output directory for results'
    )

    parser.add_argument(
        '--verbose',
        type=int,
        choices=[0, 1, 2],
        help='Override verbosity level (0=silent, 1=progress, 2=detailed)'
    )

    parser.add_argument(
        '--warm-start',
        action='store_true',
        help='Resume from previous incomplete runs (uses database checkpoints)'
    )

    parser.add_argument(
        '--no-warm-start',
        action='store_true',
        help='Always start fresh, ignoring previous runs'
    )

    parser.add_argument(
        '--reuse-params',
        action='store_true',
        help='For MLP with curriculum/gates on AR/NAR, reuse baseline (no curr/gate) hyperparameters from same data mode. Reduces trials for curriculum-only (3 params) and combined (8 params).'
    )

    parser.add_argument(
        '--reuse-top-n',
        type=int,
        default=5,
        help='Number of top baseline trials to sample from (default: 5, requires --reuse-params)'
    )

    parser.add_argument(
        '--run-autogluon',
        action='store_true',
        help=(
            'Include AutoGluon as a data-preparation benchmark. '
            'Runs experiments on AR/NAR modes using pre-computed AutoGluon features. '
            'Requires running scripts/autogluon.py first.'
        )
    )

    parser.add_argument(
        '--autogluon-data-dir',
        type=str,
        help='Directory containing pre-computed AutoGluon features (default: data_autogluon)'
    )

    parser.add_argument(
        '--run-cp',
        action='store_true',
        help=(
            'Include the custom pipeline (CP) as a data-preparation benchmark. '
            'Runs experiments on AR/NAR modes using CP-cleaned data (MICE + IQR + '
            'association-rule repair). '
            'Requires running scripts/data_preparation_pipeline.py first.'
        )
    )

    parser.add_argument(
        '--cp-data-dir',
        type=str,
        help='Directory containing CP-cleaned data (default: data_cleaned_cp)'
    )

    parser.add_argument(
        '--run-saga',
        action='store_true',
        help=(
            'Include Saga++ as a data-preparation benchmark. '
            'Runs experiments on AR/NAR modes using Saga++-cleaned data. '
            'Requires running scripts/saga.py first.'
        )
    )

    parser.add_argument(
        '--saga-data-dir',
        type=str,
        help='Directory containing Saga++-cleaned data (default: data_cleaned_saga)'
    )

    return parser.parse_args()


def list_datasets():
    """List all available datasets with their properties."""
    try:
        datasets_df = get_datasets()
        print("\n" + "="*80)
        print("Available Datasets")
        print("="*80 + "\n")

        # Display key columns
        display_cols = [
            'dataset_name', 'uci_id', 'n_samples', 'n_features',
            'n_classes', 'task_type', 'has_ar', 'has_nar'
        ]

        # Filter to existing columns
        display_cols = [col for col in display_cols if col in datasets_df.columns]

        print(datasets_df[display_cols].to_string(index=False))
        print(f"\nTotal: {len(datasets_df)} datasets")
        print("\nTo use a dataset, add its dataset_name to the config file.")
        print("Example: datasets: ['iris', 'wine', 'adult']")

    except Exception as e:
        print(f"Error listing datasets: {e}")
        print("Make sure the data directory exists and contains CSV files.")
        sys.exit(1)


def get_quick_test_config():
    """Get configuration for quick testing."""
    return {
        'datasets': ['iris'],
        'data_modes': ['clean'],
        'model_types': ['linear', 'mlp'],
        'curriculum_settings': [False],
        'gate_settings': [False],
        'n_trials': 5,
        'n_seeds': 2,
        'seed_start': 42,
        'top_m': 3,
        'n_jobs_seeds': 2,
        'n_jobs_optuna': 1,
        'clean_val': True,
        'clean_test': True,
        'output_dir': 'results_quick_test',
        'verbose': 1,
        'optuna_sampler_seed': 42,
    }


def sort_datasets_by_samples(datasets: list, data_dir: str = "data") -> list:
    """Return datasets sorted by ascending number of samples, using the data directory."""
    from pathlib import Path
    counts = {}
    data_path = Path(data_dir)
    for name in datasets:
        matches = list(data_path.glob(f"*_{name}.csv"))
        if matches:
            with open(matches[0]) as f:
                counts[name] = sum(1 for _ in f) - 1  # subtract header
        else:
            counts[name] = float('inf')
    return sorted(datasets, key=lambda d: counts[d])


def main():
    """Main entry point."""
    args = parse_args()

    # Handle --list-datasets
    if args.list_datasets:
        list_datasets()
        return

    # Load configuration
    if args.quick_test:
        print("\n" + "="*80)
        print("Running QUICK TEST mode")
        print("="*80)
        config = get_quick_test_config()
    else:
        config_path = Path(args.config)
        if not config_path.exists():
            print(f"Error: Configuration file not found: {config_path}")
            print(f"Create a config file or use --quick-test for testing.")
            sys.exit(1)

        print(f"\nLoading configuration from: {config_path}")
        config = load_config(config_path)

    # Apply command-line overrides
    if args.datasets:
        config['datasets'] = args.datasets
    if args.data_modes:
        config['data_modes'] = args.data_modes
    if args.model_types:
        config['model_types'] = args.model_types
    if args.n_trials is not None:
        config['n_trials'] = args.n_trials
    if args.n_seeds is not None:
        config['n_seeds'] = args.n_seeds
    if args.n_jobs_seeds is not None:
        config['n_jobs_seeds'] = args.n_jobs_seeds
    if args.n_jobs_optuna is not None:
        config['n_jobs_optuna'] = args.n_jobs_optuna
    if args.output_dir:
        config['output_dir'] = args.output_dir
    if args.verbose is not None:
        config['verbose'] = args.verbose
    if args.warm_start:
        config['warm_start'] = True
    if args.no_warm_start:
        config['warm_start'] = False
    if args.reuse_params:
        config['reuse_params'] = True
        config['reuse_top_n'] = args.reuse_top_n
    if args.run_autogluon:
        config['run_autogluon'] = True
    if args.autogluon_data_dir:
        config['autogluon_data_dir'] = args.autogluon_data_dir
    if args.run_cp:
        config['run_cp'] = True
    if args.cp_data_dir:
        config['cp_data_dir'] = args.cp_data_dir
    if args.run_saga:
        config['run_saga'] = True
    if args.saga_data_dir:
        config['saga_data_dir'] = args.saga_data_dir

    # Validate configuration
    if not config.get('datasets'):
        print("Error: No datasets specified in configuration.")
        print("Use --datasets to specify datasets or --list-datasets to see available options.")
        sys.exit(1)
    
    if config.get('datasets') == 'all':
        datasets_df = get_datasets()
        config['datasets'] = datasets_df['dataset_name'].tolist()

    config['datasets'] = sort_datasets_by_samples(config['datasets'])

    # Print experiment summary
    print("\n" + "="*80)
    print("Experiment Configuration")
    print("="*80)
    print(f"Datasets: {config['datasets']}")
    print(f"Data modes: {config['data_modes']}")
    print(f"Model types: {config['model_types']}")
    print(f"Curriculum settings: {config['curriculum_settings']} (applied to all model types on AR/NAR)")
    print(f"Gate settings: {config['gate_settings']} (applied to all model types on AR/NAR)")
    print(f"Trials per config: {config['n_trials']}")
    print(f"Seeds per trial: {config['n_seeds']}")
    print(f"Parallel jobs (seeds): {config['n_jobs_seeds']}")
    print(f"Parallel jobs (Optuna): {config.get('n_jobs_optuna', 1)}")
    print(f"Output directory: {config['output_dir']}")
    print(f"Warm start (resume): {config.get('warm_start', False)}")
    print(f"Reuse baseline params: {config.get('reuse_params', False)}")
    if config.get('reuse_params'):
        print(f"  - Sample from top-{config.get('reuse_top_n', 5)} baseline trials (AR/NAR baseline)")
        print(f"  - Reduced trials: curriculum-only (3), combined ({config['n_trials']})")
    print(f"AutoGluon benchmark: {config.get('run_autogluon', False)}")
    if config.get('run_autogluon'):
        print(f"  - Pre-computed features dir: {config.get('autogluon_data_dir', 'data_autogluon')}")
        print(f"  - To pre-compute: python scripts/autogluon.py")
    print(f"Custom pipeline (CP) benchmark: {config.get('run_cp', False)}")
    if config.get('run_cp'):
        print(f"  - Cleaned data dir: {config.get('cp_data_dir', 'data_cleaned_cp')}")
        print(f"  - To pre-compute: python scripts/data_preparation_pipeline.py")
    print(f"Saga benchmark: {config.get('run_saga', False)}")
    if config.get('run_saga'):
        print(f"  - Cleaned data dir: {config.get('saga_data_dir', 'data_cleaned_saga')}")
        print(f"  - To pre-compute: python scripts/saga.py")

    # Calculate total configurations per model × n_models × n_datasets,
    # adjusted for which optional benchmarks are enabled.
    n_datasets = len(config['datasets'])
    n_ar_nar = sum(1 for m in config['data_modes'] if m in ['ar', 'nar'])
    has_clean = 'clean' in config['data_modes']
    n_models = len(config['model_types'])

    # Per model per AR/NAR mode: poisoned + curriculum + gate + gate+curriculum = 4 standard
    # + autogluon (opt) + cp (opt) + saga (opt)
    per_model_ar_nar = 4
    if config.get('run_autogluon'):
        per_model_ar_nar += 1
    if config.get('run_cp'):
        per_model_ar_nar += 1
    if config.get('run_saga'):
        per_model_ar_nar += 1

    clean_configs = n_datasets * n_models * int(has_clean)
    ar_nar_configs = n_datasets * n_models * n_ar_nar * per_model_ar_nar
    total_configs = clean_configs + ar_nar_configs
    total_trials = total_configs * config['n_trials']
    total_evaluations = total_trials * config['n_seeds']

    print(f"\nConfiguration breakdown (per model: Linear + MLP):")
    print(f"  Clean baseline: {clean_configs} configs")
    print(f"  Per AR/NAR mode per model:")
    print(f"    - Poisoned baseline, curriculum, gate, gate+curriculum: 4 standard configs")
    if config.get('run_autogluon'):
        print(f"    - AutoGluon baseline: 1 config")
    if config.get('run_cp'):
        print(f"    - Custom pipeline (CP) baseline: 1 config")
    if config.get('run_saga'):
        print(f"    - Saga baseline: 1 config")
    print(f"  AR/NAR configs total: {ar_nar_configs}")
    print(f"  Total configurations: {total_configs}")
    print(f"  Total trials: {total_trials}")
    print(f"  Total evaluations: {total_evaluations}")
    print("="*80)

    # Confirm before starting
    if not args.quick_test and config.get('verbose', 1) > 0:
        response = input("\nProceed with experiments? [y/N]: ")
        if response.lower() not in ['y', 'yes']:
            print("Cancelled.")
            return

    # Run experiments
    try:
        experiment = OptunaExperiment(config)
        results = experiment.run_all_experiments()

        # Print summary
        if len(results) > 0:
            print("\n" + "="*80)
            print("Experiment Summary")
            print("="*80)

            # Group by configuration
            summary = results.groupby(['dataset', 'data_mode', 'model_type',
                                      'use_curriculum', 'use_gate']).agg({
                'trial_value': ['mean', 'std', 'max'],
                'trial_number': 'count'
            }).round(4)

            print(summary)

            print(f"\n\nDetailed results saved to: {config['output_dir']}/")
            print(f"Combined results: {config['output_dir']}/all_experiments_results.csv")

        else:
            print("\nNo results generated. Check for errors above.")

    except KeyboardInterrupt:
        print("\n\nExperiments interrupted by user.")
        sys.exit(1)
    except Exception as e:
        print(f"\nError running experiments: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()
