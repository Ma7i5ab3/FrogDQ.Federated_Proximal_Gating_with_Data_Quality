#!/usr/bin/env python3
"""
Generate LaTeX tables for classification and regression results.

Usage:
    python scripts/generate_latex_tables.py                         # Basic usage
    python scripts/generate_latex_tables.py --clickable-links       # With clickable UCI links
    python scripts/generate_latex_tables.py --use-cache             # Use cached CSV results
    python scripts/generate_latex_tables.py --aggregate-gates       # Aggregate MLP+gates columns
    python scripts/generate_latex_tables.py --confidence-intervals  # Include confidence intervals
    python scripts/generate_latex_tables.py --include-shape         # Include dataset shape column
"""

import argparse
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import optuna
import pandas as pd
from tqdm import tqdm

# =============================================================================
# CAPTIONS - Edit these strings to modify table captions
# =============================================================================

CLASSIFICATION_CAPTION = (
    r"F1 Score ($\uparrow$) mean over multiple seeds for each classification dataset. "
    r"All values are scaled by a factor of 100 for readability. "
    r"For each dataset and summary row, the best model within NAR and NNAR configurations "
    r"is highlighted in \cellcolor{green!25}$\mathbf{bold + green}$."
)

REGRESSION_CAPTION = (
    r"R$^2$ ($\uparrow$) mean over multiple seeds for each regression dataset. "
    r"All values are scaled by a factor of 100 for readability. "
    r"For each dataset and summary row, the best model (highest R$^2$) within NAR and NNAR configurations "
    r"is highlighted in \cellcolor{green!25}$\mathbf{bold + green}$."
)

# =============================================================================
# DATASET METADATA
# =============================================================================

UCI_IDS = {
    'abalone': 1,
    'adult': 2,
    'aids_clinical_trials_group_study_175': 890,
    'appliances_energy_prediction': 374,
    'auto_mpg': 9,
    'automobile': 10,
    'bank_marketing': 222,
    'banknote_authentication': 267,
    'bike_sharing': 275,
    'breast_cancer': 14,
    'breast_cancer_wisconsin_diagnostic': 17,
    'breast_cancer_wisconsin_original': 15,
    'car_evaluation': 19,
    'chronic_kidney_disease': 336,
    'combined_cycle_power_plant': 294,
    'concrete_compressive_strength': 165,
    'credit_approval': 27,
    'default_of_credit_card_clients': 350,
    'dry_bean': 602,
    'early_stage_diabetes_risk_prediction': 529,
    'estimation_of_obesity_levels_based_on_eating_habits_and_physical_condition': 544,
    'forest_fires': 162,
    'glass_identification': 42,
    'heart_disease': 45,
    'heart_failure_clinical_records': 519,
    'hepatitis': 46,
    'higher_education_students_performance_evaluation': 856,
    'iranian_churn': 563,
    'iris': 53,
    'letter_recognition': 59,
    'liver_disorders': 60,
    'magic_gamma_telescope': 159,
    'maternal_health_risk': 863,
    'mushroom': 73,
    'national_poll_on_healthy_aging_npha': 936,
    'online_shoppers_purchasing_intention_dataset': 468,
    'optical_recognition_of_handwritten_digits': 80,
    'parkinsons': 174,
    'phishing_websites': 327,
    'predict_students_dropout_and_academic_success': 697,
    'productivity_prediction_of_garment_employees': 597,
    'real_estate_valuation': 477,
    'rice_cammeo_and_osmancik': 545,
    'seoul_bike_sharing_demand': 560,
    'spambase': 94,
    'statlog_german_credit_data': 144,
    'wholesale_customers': 292,
    'wine': 109,
    'wine_quality': 186,
    'zoo': 111
}

DATASET_ABBREVIATIONS = {
    'abalone': 'Abal.',
    'adult': 'Adult',
    'aids_clinical_trials_group_study_175': 'AIDS',
    'appliances_energy_prediction': 'AppEn.',
    'auto_mpg': 'AutoMPG',
    'automobile': 'Auto.',
    'bank_marketing': 'BankMkt.',
    'banknote_authentication': 'Banknote',
    'bike_sharing': 'BikeShare',
    'breast_cancer': 'BrCan.',
    'breast_cancer_wisconsin_diagnostic': 'BrCanWD',
    'breast_cancer_wisconsin_original': 'BrCanWO',
    'car_evaluation': 'CarEval.',
    'chronic_kidney_disease': 'CKD',
    'combined_cycle_power_plant': 'CCPP',
    'concrete_compressive_strength': 'Concrete',
    'credit_approval': 'Credit',
    'default_of_credit_card_clients': 'DefCredit',
    'dry_bean': 'DryBean',
    'early_stage_diabetes_risk_prediction': 'Diabetes',
    'estimation_of_obesity_levels_based_on_eating_habits_and_physical_condition': 'Obesity',
    'forest_fires': 'ForestFire',
    'glass_identification': 'Glass',
    'heart_disease': 'HeartDis.',
    'heart_failure_clinical_records': 'HeartFail.',
    'hepatitis': 'Hepatitis',
    'higher_education_students_performance_evaluation': 'HigherEd',
    'iranian_churn': 'IranChurn',
    'iris': 'Iris',
    'letter_recognition': 'LetterRec.',
    'liver_disorders': 'Liver',
    'magic_gamma_telescope': 'MagicGamma',
    'maternal_health_risk': 'MatHealth',
    'mushroom': 'Mushroom',
    'national_poll_on_healthy_aging_npha': 'NPHA',
    'online_shoppers_purchasing_intention_dataset': 'OnlineShop',
    'optical_recognition_of_handwritten_digits': 'OptDigits',
    'parkinsons': 'Parkinson',
    'phishing_websites': 'Phishing',
    'predict_students_dropout_and_academic_success': 'StudentDrop',
    'productivity_prediction_of_garment_employees': 'Garment',
    'real_estate_valuation': 'RealEst.',
    'rice_cammeo_and_osmancik': 'Rice',
    'seoul_bike_sharing_demand': 'SeoulBike',
    'spambase': 'Spambase',
    'statlog_german_credit_data': 'GermanCred.',
    'wholesale_customers': 'Wholesale',
    'wine': 'Wine',
    'wine_quality': 'WineQual.',
    'zoo': 'Zoo'
}

# Column configuration: (data_mode, model_type, use_curriculum, use_gate)
# Full mode: all columns including separate MLP+gates and MLP+gates+curr
COLUMNS_ORDER_FULL = [
    ('clean', 'linear', False, False),
    ('clean', 'mlp', False, False),
    ('ar', 'linear', False, False),
    ('ar', 'mlp', False, False),
    ('ar', 'mlp', True, False),
    ('ar', 'mlp', False, True),
    ('ar', 'mlp', True, True),
    ('nar', 'linear', False, False),
    ('nar', 'mlp', False, False),
    ('nar', 'mlp', True, False),
    ('nar', 'mlp', False, True),
    ('nar', 'mlp', True, True),
]

# Aggregated mode: MLP+gates combines (MLP+gates) and (MLP+gates+curr)
# The 'aggregated' marker indicates these columns should be aggregated
COLUMNS_ORDER_AGGREGATED = [
    ('clean', 'linear', False, False),
    ('clean', 'mlp', False, False),
    ('ar', 'linear', False, False),
    ('ar', 'mlp', False, False),
    ('ar', 'mlp', True, False),
    ('ar', 'mlp', 'aggregated', True),  # Aggregates (False, True) and (True, True)
    ('nar', 'linear', False, False),
    ('nar', 'mlp', False, False),
    ('nar', 'mlp', True, False),
    ('nar', 'mlp', 'aggregated', True),  # Aggregates (False, True) and (True, True)
]

CACHE_FILENAME = 'results_cache.csv'


# =============================================================================
# DATA LOADING
# =============================================================================

def parse_study_name(study_name: str) -> Optional[Dict[str, str]]:
    """Parse study name to extract configuration."""
    parts = study_name.rsplit('_', 4)
    if len(parts) != 5:
        return None

    dataset, data_mode, model_type, curr_part, gate_part = parts
    return {
        'dataset': dataset,
        'data_mode': data_mode,
        'model_type': model_type,
        'use_curriculum': curr_part == 'curr1',
        'use_gate': gate_part == 'gate1',
    }


def load_results_from_db(results_dir: Path, dataset_task_map: Dict[str, str]) -> pd.DataFrame:
    """Load all completed studies from the database."""
    db_path = results_dir / 'optuna_studies.db'
    storage_url = f'sqlite:///{db_path}'

    study_summaries = optuna.study.get_all_study_summaries(storage=storage_url)
    all_results = []

    for summary in tqdm(study_summaries, desc="Loading studies from database"):
        study_name = summary.study_name
        config = parse_study_name(study_name)

        if config is None:
            continue

        try:
            study = optuna.load_study(study_name=study_name, storage=storage_url)
            completed_trials = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]

            if not completed_trials:
                continue

            best_trial = max(completed_trials, key=lambda t: t.value if t.value is not None else float('-inf'))
            seed_results = best_trial.user_attrs.get('seed_results', [])

            if not seed_results:
                continue

            task_type = dataset_task_map.get(config['dataset'], 'unknown')
            primary_metric = 'f1' if task_type == 'classification' else 'r2'

            for seed_result in seed_results:
                if 'error' in seed_result:
                    continue

                result_dict = {
                    'study_name': study_name,
                    'dataset': config['dataset'],
                    'task_type': task_type,
                    'data_mode': config['data_mode'],
                    'model_type': config['model_type'],
                    'use_curriculum': config['use_curriculum'],
                    'use_gate': config['use_gate'],
                    'seed': seed_result['seed'],
                    f'test_{primary_metric}': seed_result.get(f'test_{primary_metric}'),
                    'epochs_to_90pct': seed_result.get('epochs_to_90pct'),
                    'epochs_to_95pct': seed_result.get('epochs_to_95pct'),
                }
                all_results.append(result_dict)

        except Exception:
            continue

    return pd.DataFrame(all_results)


def save_results_cache(df: pd.DataFrame, results_dir: Path) -> None:
    """Save results DataFrame to CSV cache."""
    cache_path = results_dir / CACHE_FILENAME
    df.to_csv(cache_path, index=False)
    print(f"Results cached to: {cache_path}")


def load_results_from_cache(results_dir: Path) -> pd.DataFrame:
    """Load results DataFrame from CSV cache."""
    cache_path = results_dir / CACHE_FILENAME
    if not cache_path.exists():
        raise FileNotFoundError(f"Cache file not found: {cache_path}. Run without --use-cache first.")
    df = pd.read_csv(cache_path)
    print(f"Loaded {len(df)} results from cache: {cache_path}")
    return df


def load_results(results_dir: Path, dataset_task_map: Dict[str, str], use_cache: bool) -> pd.DataFrame:
    """Load results either from cache or database."""
    if use_cache:
        return load_results_from_cache(results_dir)

    df = load_results_from_db(results_dir, dataset_task_map)
    save_results_cache(df, results_dir)
    return df


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def format_dataset_name(
    dataset: str,
    use_links: bool,
    include_shape: bool = False,
    dataset_shapes: Optional[Dict[str, Tuple[int, int]]] = None
) -> str:
    """Format dataset name with abbreviated name, UCI ID, and optionally shape."""
    abbrev = DATASET_ABBREVIATIONS.get(dataset, dataset)
    uci_id = UCI_IDS.get(dataset)

    if uci_id is None:
        name_part = abbrev
    elif use_links:
        name_part = f"{abbrev} (\\href{{https://archive.ics.uci.edu/dataset/{uci_id}}}{{{uci_id}}})"
    else:
        name_part = f"{abbrev} ({uci_id})"

    return name_part


def get_performance_stats(
    df: pd.DataFrame,
    dataset: str,
    data_mode: str,
    model_type: str,
    use_curr: bool,
    use_gate: bool,
    metric: str
) -> Tuple[Optional[float], Optional[float]]:
    """Get mean and std performance for a specific configuration."""
    subset = df[
        (df['dataset'] == dataset) &
        (df['data_mode'] == data_mode) &
        (df['model_type'] == model_type) &
        (df['use_curriculum'] == use_curr) &
        (df['use_gate'] == use_gate)
    ][metric].values

    if len(subset) == 0:
        return None, None
    return np.mean(subset), np.std(subset)


def get_aggregated_performance_stats(
    df: pd.DataFrame,
    dataset: str,
    data_mode: str,
    model_type: str,
    metric: str
) -> Tuple[Optional[float], Optional[float]]:
    """Get aggregated stats for MLP+gates (max of gate-only and gate+curr for performance)."""
    # Get stats for MLP + gates (no curriculum)
    mean1, std1 = get_performance_stats(df, dataset, data_mode, model_type, False, True, metric)
    # Get stats for MLP + gates + curriculum
    mean2, std2 = get_performance_stats(df, dataset, data_mode, model_type, True, True, metric)

    if mean1 is None and mean2 is None:
        return None, None
    if mean1 is None:
        return mean2, std2
    if mean2 is None:
        return mean1, std1

    # Return the one with higher mean (for performance, higher is better)
    if mean1 >= mean2:
        return mean1, std1
    return mean2, std2


def get_convergence_stats(
    df: pd.DataFrame,
    data_mode: str,
    model_type: str,
    use_curr: bool,
    use_gate: bool,
    metric: str
) -> Tuple[Optional[float], Optional[float]]:
    """Get mean and std convergence metric across all datasets."""
    subset = df[
        (df['data_mode'] == data_mode) &
        (df['model_type'] == model_type) &
        (df['use_curriculum'] == use_curr) &
        (df['use_gate'] == use_gate)
    ]

    if len(subset) == 0:
        return None, None

    dataset_means = subset.groupby('dataset')[metric].mean()
    return dataset_means.mean(), dataset_means.std()


def get_aggregated_convergence_stats(
    df: pd.DataFrame,
    data_mode: str,
    model_type: str,
    metric: str
) -> Tuple[Optional[float], Optional[float]]:
    """Get aggregated convergence stats (mean of gate-only and gate+curr)."""
    mean1, std1 = get_convergence_stats(df, data_mode, model_type, False, True, metric)
    mean2, std2 = get_convergence_stats(df, data_mode, model_type, True, True, metric)

    if mean1 is None and mean2 is None:
        return None, None
    if mean1 is None:
        return mean2, std2
    if mean2 is None:
        return mean1, std1

    # Average the two for convergence
    avg_mean = (mean1 + mean2) / 2
    # Propagate uncertainty
    avg_std = np.sqrt((std1**2 + std2**2) / 2) if std1 is not None and std2 is not None else None
    return avg_mean, avg_std


def find_best_indices(
    values: List[Optional[float]],
    index_range: Tuple[int, int],
    higher_is_better: bool = True
) -> Optional[int]:
    """Find the index of the best value within a given range."""
    candidates = [(i, values[i]) for i in range(index_range[0], index_range[1]) if values[i] is not None]
    if not candidates:
        return None

    if higher_is_better:
        return max(candidates, key=lambda x: x[1])[0]
    return min(candidates, key=lambda x: x[1])[0]


def format_cell(
    mean: Optional[float],
    std: Optional[float],
    col_idx: int,
    best_ar_idx: Optional[int],
    best_nar_idx: Optional[int],
    scale: float = 100.0,
    decimals: int = 2,
    na_str: str = "",
    show_ci: bool = False
) -> str:
    """Format a table cell with optional highlighting and confidence intervals."""
    if mean is None:
        return f" & {na_str}"

    scaled_mean = mean * scale
    is_best = col_idx == best_ar_idx or col_idx == best_nar_idx

    if show_ci and std is not None:
        scaled_std = std * scale
        if is_best:
            return f" & \\cellcolor{{green!25}}$\\mathbf{{{scaled_mean:.{decimals}f}}} \\pm {scaled_std:.{decimals}f}$"
        return f" & ${scaled_mean:.{decimals}f} \\pm {scaled_std:.{decimals}f}$"
    else:
        if is_best:
            return f" & \\cellcolor{{green!25}}$\\mathbf{{{scaled_mean:.{decimals}f}}}$"
        return f" & ${scaled_mean:.{decimals}f}$"


def format_epoch_cell(
    mean: Optional[float],
    std: Optional[float],
    col_idx: int,
    best_ar_idx: Optional[int],
    best_nar_idx: Optional[int],
    show_ci: bool = False
) -> str:
    """Format an epoch cell with optional highlighting and confidence intervals."""
    if mean is None:
        return " & $-$"

    is_best = col_idx == best_ar_idx or col_idx == best_nar_idx

    if show_ci and std is not None:
        if is_best:
            return f" & \\cellcolor{{green!25}}$\\mathbf{{{mean:.1f}}} \\pm {std:.1f}$"
        return f" & ${mean:.1f} \\pm {std:.1f}$"
    else:
        if is_best:
            return f" & \\cellcolor{{green!25}}$\\mathbf{{{mean:.1f}}}$"
        return f" & ${mean:.1f}$"


# =============================================================================
# LATEX TABLE GENERATION
# =============================================================================

def generate_table_header(columns_order: List, include_shape: bool = False) -> List[str]:
    """Generate the common table header rows."""
    lines = []

    # Determine column counts based on mode
    if len(columns_order) == 12:  # Full mode
        ar_cols = 5
        nar_cols = 5
    else:  # Aggregated mode (10 columns)
        ar_cols = 4
        nar_cols = 4

    # Build column spec
    if include_shape:
        col_spec = f"@{{}} l c | Y Y | {'Y ' * ar_cols}| {'Y ' * nar_cols}@{{}}"
    else:
        col_spec = f"@{{}} l | Y Y | {'Y ' * ar_cols}| {'Y ' * nar_cols}@{{}}"

    lines.append(f"\\begin{{tabularx}}{{\\textwidth}}{{{col_spec.strip()}}}")
    lines.append(r"\toprule")

    # Header row 1: multicolumn for data modes
    if include_shape:
        lines.append(f"Dataset (ID) & Shape & \\multicolumn{{2}}{{c|}}{{Clean}} & \\multicolumn{{{ar_cols}}}{{c|}}{{Noise At Random}} & \\multicolumn{{{nar_cols}}}{{c}}{{Noise Not At Random}} \\\\")
    else:
        lines.append(f"Dataset (ID) & \\multicolumn{{2}}{{c|}}{{Clean}} & \\multicolumn{{{ar_cols}}}{{c|}}{{Noise At Random}} & \\multicolumn{{{nar_cols}}}{{c}}{{Noise Not At Random}} \\\\")

    lines.append(r"\midrule")

    # Model configuration rows
    shape_col = " & " if include_shape else ""

    for label in ["Linear", "MLP", "Curriculum", "Gates"]:
        marks = []
        for col in columns_order:
            data_mode, model_type, use_curr, use_gate = col
            if label == "Linear":
                marks.append(r"$\checkmark$" if model_type == 'linear' else "")
            elif label == "MLP":
                marks.append(r"$\checkmark$" if model_type == 'mlp' else "")
            elif label == "Curriculum":
                # In aggregated mode, curriculum column shows checkmark only for curriculum-only
                if use_curr == 'aggregated':
                    marks.append("")  # Aggregated gates column doesn't show curriculum checkmark
                else:
                    marks.append(r"$\checkmark$" if use_curr else "")
            else:  # Gates
                marks.append(r"$\checkmark$" if use_gate else "")
        lines.append(f"{label}{shape_col} & " + " & ".join(marks) + r" \\")

    lines.append(r"\midrule")
    return lines


def generate_table_footer() -> List[str]:
    """Generate the common table footer."""
    return [
        r"\bottomrule",
        r"\end{tabularx}",
        r"\end{sc}",
        r"\end{small}",
        r"\end{center}",
        r"\end{table*}",
    ]


def generate_latex_table(
    df: pd.DataFrame,
    task_type: str,
    metric: str,
    caption: str,
    label: str,
    use_links: bool,
    aggregate_gates: bool = False,
    show_ci: bool = False,
    include_shape: bool = False,
    dataset_shapes: Optional[Dict[str, Tuple[int, int]]] = None
) -> str:
    """Generate a complete LaTeX table for the given task type."""
    task_df = df[df['task_type'] == task_type].copy()
    datasets = sorted(task_df['dataset'].unique())

    columns_order = COLUMNS_ORDER_AGGREGATED if aggregate_gates else COLUMNS_ORDER_FULL

    # Determine index ranges for AR and NAR based on mode
    if aggregate_gates:
        ar_range = (2, 6)  # indices 2-5 (4 columns)
        nar_range = (6, 10)  # indices 6-9 (4 columns)
    else:
        ar_range = (2, 7)  # indices 2-6 (5 columns)
        nar_range = (7, 12)  # indices 7-11 (5 columns)

    # Collect performance data
    table_data = {}  # {dataset: [(mean, std), ...]}
    for dataset in datasets:
        row_data = []
        for col in columns_order:
            data_mode, model_type, use_curr, use_gate = col
            if use_curr == 'aggregated':
                mean, std = get_aggregated_performance_stats(task_df, dataset, data_mode, model_type, metric)
            else:
                mean, std = get_performance_stats(task_df, dataset, data_mode, model_type, use_curr, use_gate, metric)
            row_data.append((mean, std))
        table_data[dataset] = row_data

    # Calculate averages
    avg_performance = []
    for col_idx in range(len(columns_order)):
        means = [table_data[ds][col_idx][0] for ds in datasets if table_data[ds][col_idx][0] is not None]
        if means:
            avg_performance.append((np.mean(means), np.std(means)))
        else:
            avg_performance.append((None, None))

    # Calculate convergence metrics
    avg_epochs_90 = []
    avg_epochs_95 = []
    for col in columns_order:
        data_mode, model_type, use_curr, use_gate = col
        if model_type == 'linear':
            avg_epochs_90.append((None, None))
            avg_epochs_95.append((None, None))
        elif use_curr == 'aggregated':
            avg_epochs_90.append(get_aggregated_convergence_stats(task_df, data_mode, model_type, 'epochs_to_90pct'))
            avg_epochs_95.append(get_aggregated_convergence_stats(task_df, data_mode, model_type, 'epochs_to_95pct'))
        else:
            avg_epochs_90.append(get_convergence_stats(task_df, data_mode, model_type, use_curr, use_gate, 'epochs_to_90pct'))
            avg_epochs_95.append(get_convergence_stats(task_df, data_mode, model_type, use_curr, use_gate, 'epochs_to_95pct'))

    # Build LaTeX
    lines = []
    lines.append(r"\begin{table*}[t]")
    lines.append(f"\\caption{{{caption}}}")
    lines.append(f"\\label{{{label}}}")
    lines.append(r"\begin{center}")
    lines.append(r"\begin{small}")
    lines.append(r"\begin{sc}")
    lines.extend(generate_table_header(columns_order, include_shape))

    # Data rows
    for dataset in datasets:
        row_data = table_data[dataset]
        means_only = [x[0] for x in row_data]
        best_ar_idx = find_best_indices(means_only, ar_range, higher_is_better=True)
        best_nar_idx = find_best_indices(means_only, nar_range, higher_is_better=True)

        dataset_display = format_dataset_name(dataset, use_links)
        row_str = dataset_display

        # Add shape column if requested
        if include_shape and dataset_shapes and dataset in dataset_shapes:
            n_samples, n_features = dataset_shapes[dataset]
            row_str += f" & ${n_samples} \\times {n_features}$"
        elif include_shape:
            row_str += " & "

        for col_idx, (mean, std) in enumerate(row_data):
            row_str += format_cell(mean, std, col_idx, best_ar_idx, best_nar_idx, show_ci=show_ci)

        lines.append(row_str + r" \\")

    lines.append(r"\midrule")

    # Average performance row
    means_only = [x[0] for x in avg_performance]
    best_ar_avg = find_best_indices(means_only, ar_range, higher_is_better=True)
    best_nar_avg = find_best_indices(means_only, nar_range, higher_is_better=True)

    row_str = "Average"
    if include_shape:
        row_str += " & "
    for col_idx, (mean, std) in enumerate(avg_performance):
        row_str += format_cell(mean, std, col_idx, best_ar_avg, best_nar_avg, show_ci=show_ci)
    lines.append(row_str + r" \\")

    # Epochs to 90%
    means_ep90 = [x[0] for x in avg_epochs_90]
    best_ar_ep90 = find_best_indices(means_ep90, ar_range, higher_is_better=False)
    best_nar_ep90 = find_best_indices(means_ep90, nar_range, higher_is_better=False)

    row_str = "Avg Epochs to 90\\%"
    if include_shape:
        row_str += " & "
    for col_idx, (mean, std) in enumerate(avg_epochs_90):
        row_str += format_epoch_cell(mean, std, col_idx, best_ar_ep90, best_nar_ep90, show_ci=show_ci)
    lines.append(row_str + r" \\")

    # Epochs to 95%
    means_ep95 = [x[0] for x in avg_epochs_95]
    best_ar_ep95 = find_best_indices(means_ep95, ar_range, higher_is_better=False)
    best_nar_ep95 = find_best_indices(means_ep95, nar_range, higher_is_better=False)

    row_str = "Avg Epochs to 95\\%"
    if include_shape:
        row_str += " & "
    for col_idx, (mean, std) in enumerate(avg_epochs_95):
        row_str += format_epoch_cell(mean, std, col_idx, best_ar_ep95, best_nar_ep95, show_ci=show_ci)
    lines.append(row_str + r" \\")

    lines.extend(generate_table_footer())

    return "\n".join(lines)


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Generate LaTeX tables for classification and regression results.")
    parser.add_argument(
        "--clickable-links",
        action="store_true",
        help="Make UCI dataset IDs clickable hyperlinks"
    )
    parser.add_argument(
        "--use-cache",
        action="store_true",
        help="Load results from cached CSV instead of parsing database (faster)"
    )
    parser.add_argument(
        "--aggregate-gates",
        action="store_true",
        help="Aggregate MLP+gates and MLP+gates+curr columns (max for performance, mean for epochs)"
    )
    parser.add_argument(
        "--confidence-intervals",
        action="store_true",
        help="Include confidence intervals (mean ± std) in table cells"
    )
    parser.add_argument(
        "--include-shape",
        action="store_true",
        help="Include a Shape column with original dataset dimensions (n_samples x n_features)"
    )
    args = parser.parse_args()

    # Paths
    script_dir = Path(__file__).parent
    project_root = script_dir.parent
    results_dir = project_root / 'results'

    # Check for alternative path
    if not (results_dir / 'optuna_studies.db').exists() and not args.use_cache:
        alt_path = Path.home() / 'storage_megaverse' / 'frogdq' / 'results'
        if alt_path.exists():
            results_dir = alt_path

    # Load dataset metadata
    from frogdq.data import get_datasets
    datasets_info = get_datasets(
        data_dir=project_root / 'data',
        poisoned_dir=project_root / 'data_poisoned'
    )
    dataset_task_map = dict(zip(datasets_info['dataset_name'], datasets_info['task_type']))

    # Create dataset shapes dictionary if needed
    dataset_shapes = None
    if args.include_shape:
        dataset_shapes = dict(zip(
            datasets_info['dataset_name'],
            zip(datasets_info['n_samples'], datasets_info['n_features'])
        ))

    # Load results
    print("Loading results...")
    results_df = load_results(results_dir, dataset_task_map, args.use_cache)
    print(f"Loaded {len(results_df)} results")

    # Generate classification table
    print("\n" + "=" * 80)
    print("CLASSIFICATION TABLE")
    print("=" * 80)
    classification_table = generate_latex_table(
        df=results_df,
        task_type='classification',
        metric='test_f1',
        caption=CLASSIFICATION_CAPTION,
        label='tab:classification_results',
        use_links=args.clickable_links,
        aggregate_gates=args.aggregate_gates,
        show_ci=args.confidence_intervals,
        include_shape=args.include_shape,
        dataset_shapes=dataset_shapes
    )
    print(classification_table)

    # Generate regression table
    print("\n" + "=" * 80)
    print("REGRESSION TABLE")
    print("=" * 80)
    regression_table = generate_latex_table(
        df=results_df,
        task_type='regression',
        metric='test_r2',
        caption=REGRESSION_CAPTION,
        label='tab:regression_results',
        use_links=args.clickable_links,
        aggregate_gates=args.aggregate_gates,
        show_ci=args.confidence_intervals,
        include_shape=args.include_shape,
        dataset_shapes=dataset_shapes
    )
    print(regression_table)


if __name__ == "__main__":
    main()
