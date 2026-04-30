#!/usr/bin/env python3
"""
Data Poisoning Pipeline for ML Datasets
Implements AR (At Random) and NAR (Not At Random) poisoning modes with state-of-the-art mechanisms

Note: Target columns (cls_*, reg_*) are NEVER poisoned to preserve labels.
"""
import os
import sys
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd
import yaml
from loguru import logger
from scipy import stats

logger.remove()
logger.add(
    sys.stderr,
    level="INFO",
    format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <level>{message}</level>",
)


class DataPoisoner:
    """
    Implements data poisoning mechanisms for tabular datasets.

    References:
    - Missing mechanisms: Rubin (1976), Little & Rubin (2002)
    - Poisoning attacks: Biggio et al. (2012), Steinhardt et al. (2017)
    - Data quality: Redman (2001), Wang & Strong (1996)
    """

    def __init__(self, seed=42, noise_percentages=None):
        self.seed = seed
        np.random.seed(seed)
        self.poison_log = {}

        if noise_percentages is None:
            self.noise_percentages = {
                "mild": (0.5625, 0.05),
                "moderate": (0.25, 0.10),
                "heavy": (0.125, 0.20),
                "severe": (0.0625, 0.40),
            }
        else:
            self.noise_percentages = noise_percentages

    def identify_column_types(self, df: pd.DataFrame) -> Dict[str, list]:
        """Identify column types based on prefix naming convention."""
        col_types = {
            "numerical": [],
            "categorical": [],
            "target_cls": [],
            "target_reg": [],
            "date": [],
            "id": [],
        }

        for col in df.columns:
            if col.startswith("num_"):
                col_types["numerical"].append(col)
            elif col.startswith("cat_"):
                col_types["categorical"].append(col)
            elif col.startswith("cls_"):
                col_types["target_cls"].append(col)
            elif col.startswith("reg_"):
                col_types["target_reg"].append(col)
            elif col.startswith("dat_"):
                col_types["date"].append(col)
            elif col.startswith("id_"):
                col_types["id"].append(col)

        return col_types

    def _create_stratified_noise_distribution(self, columns: list) -> Dict[str, float]:
        """
        Create stratified noise distribution across columns.

        Uses the noise_percentages configuration to distribute noise levels.

        Args:
            columns: List of column names

        Returns:
            Dict mapping column name to poison rate
        """
        n_cols = len(columns)
        if n_cols == 0:
            return {}

        shuffled_cols = columns.copy()
        np.random.shuffle(shuffled_cols)

        mild_frac, mild_rate = self.noise_percentages["mild"]
        moderate_frac, moderate_rate = self.noise_percentages["moderate"]
        heavy_frac, heavy_rate = self.noise_percentages["heavy"]
        severe_frac, severe_rate = self.noise_percentages["severe"]

        n_mild = int(n_cols * mild_frac)
        n_moderate = int(n_cols * moderate_frac)
        n_heavy = int(n_cols * heavy_frac)
        n_severe = int(n_cols * severe_frac) if severe_frac is not None else 0

        # Ensure at least 1 column for each non-zero fraction
        if mild_frac > 0 and n_mild == 0:
            n_mild = 1
        if moderate_frac > 0 and n_moderate == 0:
            n_moderate = 1
        if heavy_frac > 0 and n_heavy == 0:
            n_heavy = 1
        if severe_frac is not None and severe_frac > 0 and n_severe == 0:
            n_severe = 1

        noise_dist = {}
        idx = 0

        for i in range(n_mild):
            if idx >= n_cols:
                break
            noise_dist[shuffled_cols[idx]] = mild_rate
            idx += 1

        for i in range(n_moderate):
            if idx >= n_cols:
                break
            noise_dist[shuffled_cols[idx]] = moderate_rate
            idx += 1

        for i in range(n_heavy):
            if idx >= n_cols:
                break
            noise_dist[shuffled_cols[idx]] = heavy_rate
            idx += 1

        for i in range(n_severe):
            if idx >= n_cols:
                break
            noise_dist[shuffled_cols[idx]] = severe_rate
            idx += 1

        # Apply mild rate to remaining columns
        while idx < n_cols:
            noise_dist[shuffled_cols[idx]] = mild_rate
            idx += 1

        return noise_dist

    def poison_ar(
        self,
        df: pd.DataFrame,
        column_noise_dist: dict = None,
        enable_numerical_noise: bool = True,
        enable_categorical_flips: bool = True,
        enable_missing: bool = True,
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        AR MODE (At Random): Random poisoning mechanisms applied uniformly

        Mechanisms:
        1. Additive Gaussian noise for numerical features (SNR-based)
        2. Random label flips for categorical features
        3. MCAR (Missing Completely At Random)

        Note: Target columns (cls_*, reg_*) are NEVER poisoned to preserve labels.

        Args:
            df: Input dataframe
            column_noise_dist: Dict mapping column indices to noise rates
                             If None, uses default stratified distribution
            enable_numerical_noise: Enable Gaussian noise on numerical features
            enable_categorical_flips: Enable random flips for categorical features
            enable_missing: Enable MCAR (Missing Completely At Random)

        Returns:
            poisoned_df: Poisoned dataframe
            poison_mask: Boolean mask (True = poisoned, False = clean)
        """
        enabled_mechanisms = []
        if enable_numerical_noise:
            enabled_mechanisms.append("numerical_noise")
        if enable_categorical_flips:
            enabled_mechanisms.append("categorical_flips")
        if enable_missing:
            enabled_mechanisms.append("MCAR")

        logger.info(f"Applying AR (At Random) poisoning with stratified column distribution")
        logger.info(
            f"  Enabled mechanisms: {', '.join(enabled_mechanisms) if enabled_mechanisms else 'none'}"
        )

        df_poison = df.copy()
        poison_mask = pd.DataFrame(False, index=df.index, columns=df.columns, dtype=bool)
        col_types = self.identify_column_types(df)

        # Only poison features, never targets
        all_data_cols = col_types["numerical"] + col_types["categorical"]

        if column_noise_dist is None:
            column_noise_dist = self._create_stratified_noise_distribution(all_data_cols)

        mild_rate = self.noise_percentages["mild"][1]
        moderate_rate = self.noise_percentages["moderate"][1]
        heavy_rate = self.noise_percentages["heavy"][1]
        severe_rate = self.noise_percentages["severe"][1]

        logger.info(
            f"  Column noise distribution: "
            f"{len([c for c in column_noise_dist.values() if c == mild_rate])} cols @ {mild_rate*100:.0f}%, "
            f"{len([c for c in column_noise_dist.values() if c == moderate_rate])} cols @ {moderate_rate*100:.0f}%, "
            f"{len([c for c in column_noise_dist.values() if c == heavy_rate])} cols @ {heavy_rate*100:.0f}%, "
            f"{len([c for c in column_noise_dist.values() if c == severe_rate])} cols @ {severe_rate*100:.0f}%"
        )

        # 1. Numerical features: Additive Gaussian noise
        if enable_numerical_noise:
            for col in col_types["numerical"]:
                if df[col].isna().all() or col not in column_noise_dist:
                    continue

                poison_rate = column_noise_dist[col]
                values = df[col].dropna()
                if len(values) == 0:
                    continue

                if df_poison[col].dtype in ["int64", "int32", "int16", "int8"]:
                    df_poison[col] = df_poison[col].astype("float64")

                std_signal = values.std()
                if std_signal == 0:
                    std_signal = abs(values.mean()) * 0.1 if values.mean() != 0 else 1.0

                snr_db = 20
                noise_std = std_signal / (10 ** (snr_db / 20))

                n_poison = int(len(df) * poison_rate)
                if n_poison == 0:
                    continue
                poison_idx = np.random.choice(df.index, size=n_poison, replace=False)

                noise = np.random.normal(0, noise_std, size=n_poison)
                df_poison.loc[poison_idx, col] = df_poison.loc[poison_idx, col] + noise
                poison_mask.loc[poison_idx, col] = True

        # 2. Categorical features: Random label flips
        if enable_categorical_flips:
            for col in col_types["categorical"]:
                if df[col].isna().all() or col not in column_noise_dist:
                    continue

                poison_rate = column_noise_dist[col]
                unique_vals = df[col].dropna().unique()
                if len(unique_vals) <= 1:
                    continue

                n_poison = int(len(df) * poison_rate)
                if n_poison == 0:
                    continue
                poison_idx = np.random.choice(df.index, size=n_poison, replace=False)

                for idx in poison_idx:
                    if pd.notna(df.loc[idx, col]):
                        current_val = df.loc[idx, col]
                        other_vals = [v for v in unique_vals if v != current_val]
                        if other_vals:
                            df_poison.loc[idx, col] = np.random.choice(other_vals)
                            poison_mask.loc[idx, col] = True

        # 3. MCAR: Random missing values
        if enable_missing:
            for col in all_data_cols:
                if col not in column_noise_dist:
                    continue

                poison_rate = column_noise_dist[col]

                # Convert integer columns to float before introducing NaN
                if df_poison[col].dtype in ["int64", "int32", "int16", "int8"]:
                    df_poison[col] = df_poison[col].astype("float64")

                n_missing = int(len(df) * poison_rate * 0.3)
                if n_missing == 0:
                    continue
                missing_idx = np.random.choice(df.index, size=n_missing, replace=False)
                df_poison.loc[missing_idx, col] = np.nan
                poison_mask.loc[missing_idx, col] = True

        logger.success(f"AR poisoning complete: {poison_mask.sum().sum()} values poisoned")
        return df_poison, poison_mask

    def poison_nar(
        self,
        df: pd.DataFrame,
        column_noise_dist: dict = None,
        enable_nnar: bool = True,
        enable_mnar: bool = True,
        enable_systematic_flips: bool = True,
        enable_correlated_noise: bool = True,
        enable_rare_missing: bool = True,
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        NAR MODE (Not At Random): Advanced non-random poisoning mechanisms

        Mechanisms:
        1. NNAR (Noise Not At Random) - noise magnitude depends on values
        2. MNAR (Missing Not At Random) - missingness depends on values
        3. Systematic categorical flips - confusion patterns
        4. Correlated noise across features - chain poisoning
        5. Rare value missingness - rare categories more likely missing

        Note: Target columns (cls_*, reg_*) are NEVER poisoned to preserve labels.

        Args:
            df: Input dataframe
            column_noise_dist: Dict mapping column indices to noise rates
                             If None, uses default stratified distribution
            enable_nnar: Enable heteroscedastic noise (value-dependent)
            enable_mnar: Enable MNAR (extreme values more likely missing)
            enable_systematic_flips: Enable systematic categorical confusion
            enable_correlated_noise: Enable correlated noise propagation
            enable_rare_missing: Enable rare category missingness

        Returns:
            poisoned_df: Poisoned dataframe
            poison_mask: Boolean mask (True = poisoned, False = clean)
        """
        enabled_mechanisms = []
        if enable_nnar:
            enabled_mechanisms.append("NNAR")
        if enable_mnar:
            enabled_mechanisms.append("MNAR")
        if enable_systematic_flips:
            enabled_mechanisms.append("systematic_flips")
        if enable_correlated_noise:
            enabled_mechanisms.append("correlated_noise")
        if enable_rare_missing:
            enabled_mechanisms.append("rare_missing")

        logger.info(f"Applying NAR (Not At Random) poisoning with stratified column distribution")
        logger.info(
            f"  Enabled mechanisms: {', '.join(enabled_mechanisms) if enabled_mechanisms else 'none'}"
        )

        df_poison = df.copy()
        poison_mask = pd.DataFrame(False, index=df.index, columns=df.columns, dtype=bool)
        col_types = self.identify_column_types(df)

        # Only poison features, never targets
        all_data_cols = col_types["numerical"] + col_types["categorical"]

        if column_noise_dist is None:
            column_noise_dist = self._create_stratified_noise_distribution(all_data_cols)

        mild_rate = self.noise_percentages["mild"][1]
        moderate_rate = self.noise_percentages["moderate"][1]
        heavy_rate = self.noise_percentages["heavy"][1]
        severe_rate = self.noise_percentages["severe"][1]

        logger.info(
            f"  Column noise distribution: "
            f"{len([c for c in column_noise_dist.values() if c == mild_rate])} cols @ {mild_rate*100:.0f}%, "
            f"{len([c for c in column_noise_dist.values() if c == moderate_rate])} cols @ {moderate_rate*100:.0f}%, "
            f"{len([c for c in column_noise_dist.values() if c == heavy_rate])} cols @ {heavy_rate*100:.0f}%, "
            f"{len([c for c in column_noise_dist.values() if c == severe_rate])} cols @ {severe_rate*100:.0f}%"
        )

        # 1. Numerical features: NNAR (Noise Not At Random)
        if enable_nnar:
            for col in col_types["numerical"]:
                if df[col].isna().all() or col not in column_noise_dist:
                    continue

                poison_rate = column_noise_dist[col]
                values = df[col].dropna()
                if len(values) == 0:
                    continue

                if df_poison[col].dtype in ["int64", "int32", "int16", "int8"]:
                    df_poison[col] = df_poison[col].astype("float64")

                min_val, max_val = values.min(), values.max()
                if min_val == max_val:
                    continue

                n_poison = int(len(df) * poison_rate)
                if n_poison == 0:
                    continue
                poison_idx = np.random.choice(df.index, size=n_poison, replace=False)

                std_signal = values.std()
                if std_signal == 0:
                    std_signal = abs(values.mean()) * 0.1 if values.mean() != 0 else 1.0
                snr_db = 15
                base_noise_std = std_signal / (10 ** (snr_db / 20))

                candidate_vals = df_poison.loc[poison_idx, col]
                valid_mask = candidate_vals.notna()
                valid_idx = poison_idx[valid_mask.values]
                if len(valid_idx) == 0:
                    continue
                vals = df_poison.loc[valid_idx, col].values
                normalized_vals = (vals - min_val) / (max_val - min_val)
                noise_stds = base_noise_std * (1 + 3 * normalized_vals)
                noise = np.random.normal(0, noise_stds)
                df_poison.loc[valid_idx, col] = vals + noise
                poison_mask.loc[valid_idx, col] = True

        # 2. Numerical features: MNAR (Missing Not At Random)
        if enable_mnar:
            for col in col_types["numerical"]:
                if df[col].isna().all() or col not in column_noise_dist:
                    continue

                poison_rate = column_noise_dist[col]
                values = df[col].dropna()
                if len(values) == 0:
                    continue

                if df_poison[col].dtype in ["int64", "int32", "int16", "int8"]:
                    df_poison[col] = df_poison[col].astype("float64")

                q25, q75 = values.quantile(0.25), values.quantile(0.75)
                iqr = q75 - q25

                if iqr == 0:
                    continue

                median = values.median()

                valid = df_poison[col].notna()
                col_vals = df_poison.loc[valid, col]
                prob = ((col_vals - median).abs() / (2 * iqr) * poison_rate).clip(upper=0.5)
                missing_mask = np.random.random(valid.sum()) < prob.values
                missing_idx = col_vals.index[missing_mask]
                df_poison.loc[missing_idx, col] = np.nan
                poison_mask.loc[missing_idx, col] = True

        # 3. Categorical features: Systematic flips with patterns
        if enable_systematic_flips:
            for col in col_types["categorical"]:
                if df[col].isna().all() or col not in column_noise_dist:
                    continue

                poison_rate = column_noise_dist[col]
                unique_vals = df[col].dropna().unique()
                if len(unique_vals) <= 1:
                    continue

                confusion_pairs = {}
                vals_list = list(unique_vals)
                for i, val in enumerate(vals_list):
                    partner_idx = (i + 1) % len(vals_list)
                    confusion_pairs[val] = vals_list[partner_idx]

                n_poison = int(len(df) * poison_rate)
                if n_poison == 0:
                    continue
                poison_idx = np.random.choice(df.index, size=n_poison, replace=False)

                for idx in poison_idx:
                    if pd.notna(df.loc[idx, col]):
                        current_val = df.loc[idx, col]
                        if np.random.random() < 0.7 and current_val in confusion_pairs:
                            df_poison.loc[idx, col] = confusion_pairs[current_val]
                        else:
                            other_vals = [v for v in unique_vals if v != current_val]
                            if other_vals:
                                df_poison.loc[idx, col] = np.random.choice(other_vals)
                        poison_mask.loc[idx, col] = True

        # 4. Correlated noise across numerical features
        num_cols = col_types["numerical"]
        if len(num_cols) > 1:
            n_chains = max(1, len(num_cols) // 3)
            anchor_cols = np.random.choice(
                num_cols, size=min(n_chains, len(num_cols)), replace=False
            )

            for anchor_col in anchor_cols:
                poisoned_rows = poison_mask[poison_mask[anchor_col]].index

                for other_col in num_cols:
                    if other_col == anchor_col:
                        continue

                    if df_poison[other_col].dtype in ["int64", "int32", "int16", "int8"]:
                        df_poison[other_col] = df_poison[other_col].astype("float64")

                    values = df[other_col].dropna()
                    if len(values) == 0:
                        continue
                    std_signal = values.std()
                    if std_signal <= 0:
                        continue
                    snr_db = 15
                    noise_std = std_signal / (10 ** (snr_db / 20))

                    col_at_rows = df_poison.loc[poisoned_rows, other_col]
                    eligible = col_at_rows.notna().values & (np.random.random(len(poisoned_rows)) < 0.6)
                    selected = poisoned_rows[eligible]
                    if len(selected) == 0:
                        continue
                    noise = np.random.normal(0, noise_std, size=len(selected))
                    df_poison.loc[selected, other_col] += noise
                    poison_mask.loc[selected, other_col] = True

        # 5. Value-dependent missingness for categorical
        for col in col_types["categorical"]:
            if df[col].isna().all() or col not in column_noise_dist:
                continue

            poison_rate = column_noise_dist[col]

            if df_poison[col].dtype in ["int64", "int32", "int16", "int8"]:
                df_poison[col] = df_poison[col].astype("float64")

            value_counts = df[col].value_counts()
            total = len(df[col].dropna())

            if total == 0:
                continue

            valid = df_poison[col].notna()
            col_vals = df_poison.loc[valid, col]
            freq_series = value_counts / total
            frequencies = col_vals.map(freq_series).fillna(0)
            prob = poison_rate * (1 - frequencies)
            missing_mask = np.random.random(valid.sum()) < prob.values
            missing_idx = col_vals.index[missing_mask]
            df_poison.loc[missing_idx, col] = np.nan
            poison_mask.loc[missing_idx, col] = True

        logger.success(f"NAR poisoning complete: {poison_mask.sum().sum()} values poisoned")
        return df_poison, poison_mask

    def calculate_cleanliness_metrics(self, poison_mask: pd.DataFrame) -> Dict:
        """
        Calculate cleanliness metrics at column and row level.

        Returns:
            dict with:
            - column_cleanliness: % clean values per column
            - row_cleanliness: % clean values per row
            - overall_cleanliness: % clean values overall
        """
        col_clean = (1 - poison_mask.sum(axis=0) / len(poison_mask)) * 100

        row_clean = (1 - poison_mask.sum(axis=1) / len(poison_mask.columns)) * 100

        overall_clean = (1 - poison_mask.sum().sum() / poison_mask.size) * 100

        return {
            "column_cleanliness": col_clean.to_dict(),
            "row_cleanliness": row_clean.to_dict(),
            "overall_cleanliness": overall_clean,
        }


def check_dataset_complete(csv_file: Path, output_dir: str) -> bool:
    """
    Check if all output files exist for a dataset.

    Args:
        csv_file: Path to the input CSV file
        output_dir: Output directory

    Returns:
        True if all output files exist, False otherwise
    """
    required_files = [
        os.path.join(output_dir, "ar", csv_file.name),
        os.path.join(output_dir, "ar", csv_file.stem + "_mask.csv"),
        os.path.join(output_dir, "metrics", csv_file.stem + "_ar_metrics.csv"),
        os.path.join(output_dir, "nar", csv_file.name),
        os.path.join(output_dir, "nar", csv_file.stem + "_mask.csv"),
        os.path.join(output_dir, "metrics", csv_file.stem + "_nar_metrics.csv"),
        os.path.join(output_dir, "test", csv_file.name),
    ]

    return all(os.path.exists(f) for f in required_files)


def process_all_datasets(
    input_dir: str,
    output_dir: str,
    dataset_name: str = None,
    noise_percentages: dict = None,
    ar_mechanisms: dict = None,
    nar_mechanisms: dict = None,
    test_size: float = 0.3,
):
    """
    Process all datasets and create poisoned versions.

    Args:
        input_dir: Directory containing clean CSV files
        output_dir: Directory to save poisoned files
        dataset_name: Optional specific dataset name to process
        noise_percentages: Optional custom noise distribution percentages
        ar_mechanisms: Dict of enabled AR mechanisms
        nar_mechanisms: Dict of enabled NAR mechanisms
        test_size: Fraction of data to hold out as clean test set (default 0.4)
    """
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.join(output_dir, "ar"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "nar"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "metrics"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "test"), exist_ok=True)

    poisoner = DataPoisoner(seed=42, noise_percentages=noise_percentages)

    if ar_mechanisms is None:
        ar_mechanisms = {}
    if nar_mechanisms is None:
        nar_mechanisms = {}

    csv_files = sorted(Path(input_dir).glob("*.csv"))

    if dataset_name:
        csv_files = [f for f in csv_files if f.stem[8:] == dataset_name]
        if not csv_files:
            logger.error(f"Dataset '{dataset_name}' not found in {input_dir}")
            return
        logger.info(f"Processing specific dataset: {dataset_name}")
    else:
        logger.info(f"Found {len(csv_files)} datasets to poison")

    skipped = 0
    for csv_file in csv_files:
        if check_dataset_complete(csv_file, output_dir):
            logger.info(f"Skipping {csv_file.name} (already complete)")
            skipped += 1

    if skipped > 0:
        logger.info(f"Skipped {skipped} already-processed dataset(s)")

    for csv_file in csv_files:
        if check_dataset_complete(csv_file, output_dir):
            continue

        logger.info(f"Processing {csv_file.name}")

        try:
            # Handle common missing value indicators like '?'
            df = pd.read_csv(csv_file, na_values=["?", "NA", "N/A", "NaN", "nan", "NAN", "", " "])
            logger.info(f"  Loaded: {df.shape[0]} rows × {df.shape[1]} columns")

            # Split into train (1 - test_size) and test (test_size)
            df_test = df.sample(frac=test_size, random_state=42)
            df_train = df.drop(df_test.index).reset_index(drop=True)
            df_test = df_test.reset_index(drop=True)
            logger.info(
                f"  Split: {len(df_train)} train rows, {len(df_test)} test rows "
                f"({test_size*100:.0f}% test)"
            )

            # Save clean test split
            test_path = os.path.join(output_dir, "test", csv_file.name)
            df_test.to_csv(test_path, index=False)

            # Calculate column noise distribution once for both AR and NAR modes
            # Only poison features, never targets — computed on train only
            col_types = poisoner.identify_column_types(df_train)
            all_data_cols = col_types["numerical"] + col_types["categorical"]
            column_noise_dist = poisoner._create_stratified_noise_distribution(all_data_cols)

            df_ar, mask_ar = poisoner.poison_ar(
                df_train, column_noise_dist=column_noise_dist, **ar_mechanisms
            )
            metrics_ar = poisoner.calculate_cleanliness_metrics(mask_ar)

            ar_path = os.path.join(output_dir, "ar", csv_file.name)
            df_ar.to_csv(ar_path, index=False)

            mask_ar_path = os.path.join(output_dir, "ar", csv_file.stem + "_mask.csv")
            mask_ar.to_csv(mask_ar_path, index=False)

            metrics_ar_path = os.path.join(output_dir, "metrics", csv_file.stem + "_ar_metrics.csv")
            pd.DataFrame(
                {
                    "column": list(metrics_ar["column_cleanliness"].keys()),
                    "cleanliness_pct": list(metrics_ar["column_cleanliness"].values()),
                }
            ).to_csv(metrics_ar_path, index=False)

            logger.info(f"  AR: {metrics_ar['overall_cleanliness']:.2f}% clean overall")

            df_nar, mask_nar = poisoner.poison_nar(
                df_train, column_noise_dist=column_noise_dist, **nar_mechanisms
            )
            metrics_nar = poisoner.calculate_cleanliness_metrics(mask_nar)

            nar_path = os.path.join(output_dir, "nar", csv_file.name)
            df_nar.to_csv(nar_path, index=False)

            mask_nar_path = os.path.join(output_dir, "nar", csv_file.stem + "_mask.csv")
            mask_nar.to_csv(mask_nar_path, index=False)

            metrics_nar_path = os.path.join(
                output_dir, "metrics", csv_file.stem + "_nar_metrics.csv"
            )
            pd.DataFrame(
                {
                    "column": list(metrics_nar["column_cleanliness"].keys()),
                    "cleanliness_pct": list(metrics_nar["column_cleanliness"].values()),
                }
            ).to_csv(metrics_nar_path, index=False)

            logger.info(f"  NAR: {metrics_nar['overall_cleanliness']:.2f}% clean overall")

            logger.success(f"  Completed {csv_file.name}")

        except Exception as e:
            logger.error(f"  Error processing {csv_file.name}: {e}")
            continue

    logger.success(f"All datasets processed! Output in {output_dir}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Apply data poisoning to ML datasets using AR and NAR modes",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
            Examples:
            python poison_data.py data data_poisoned
            python poison_data.py data data_poisoned --dataset my_dataset
            python poison_data.py data data_poisoned --severe-frac 0.1 --severe-rate 0.50
            python poison_data.py data data_poisoned --no-ar-numerical --no-nar-correlated

            Poisoning Modes:
            AR (At Random): Random poisoning with stratified column distribution
            NAR (Not At Random): Advanced non-random poisoning mechanisms

            Note:
            Target columns (cls_*, reg_*) are NEVER poisoned to preserve labels.
                """,
    )
    parser.add_argument(
        "--input_dir", type=str, help="Directory containing clean CSV files", default="data"
    )
    parser.add_argument(
        "--output_dir", type=str, help="Directory to save poisoned files", default="data_poisoned"
    )
    parser.add_argument(
        "--config", type=str, default="config.yaml",
        help="Path to experiment config file (default: config.yaml)",
    )
    parser.add_argument(
        "--dataset", type=str, help="Process only this specific dataset (name without .csv)"
    )
    parser.add_argument(
        "--test-size", type=float, default=None,
        help="Fraction of data held out as clean test set (overrides config.yaml test_size)",
    )

    noise_group = parser.add_argument_group("Noise Distribution")
    noise_group.add_argument(
        "--mild-rate", type=float, default=0.10, help="Mild noise rate for remaining columns"
    )
    noise_group.add_argument(
        "--moderate-frac", type=float, default=0.3, help="Fraction of columns at moderate noise"
    )
    noise_group.add_argument(
        "--moderate-rate", type=float, default=0.20, help="Moderate noise rate"
    )
    noise_group.add_argument(
        "--heavy-frac", type=float, default=0.2, help="Fraction of columns at heavy noise"
    )
    noise_group.add_argument("--heavy-rate", type=float, default=0.30, help="Heavy noise rate")
    noise_group.add_argument(
        "--severe-frac", type=float, default=0.1, help="Fraction of columns at severe noise"
    )
    noise_group.add_argument("--severe-rate", type=float, default=0.40, help="Severe noise rate")

    ar_group = parser.add_argument_group("AR Mode Mechanisms")
    ar_group.add_argument(
        "--no-ar-numerical",
        action="store_true",
        help="Disable Gaussian noise on numerical features",
    )
    ar_group.add_argument(
        "--no-ar-categorical",
        action="store_true",
        help="Disable random flips for categorical features",
    )
    ar_group.add_argument(
        "--no-ar-missing", action="store_true", help="Disable MCAR (Missing Completely At Random)"
    )

    nar_group = parser.add_argument_group("NAR Mode Mechanisms")
    nar_group.add_argument(
        "--no-nar-nnar", action="store_true", help="Disable NNAR (heteroscedastic noise)"
    )
    nar_group.add_argument(
        "--no-nar-mnar",
        action="store_true",
        help="Disable MNAR (extreme values more likely missing)",
    )
    nar_group.add_argument(
        "--no-nar-systematic", action="store_true", help="Disable systematic categorical confusion"
    )
    nar_group.add_argument(
        "--no-nar-correlated", action="store_true", help="Disable correlated noise propagation"
    )
    nar_group.add_argument(
        "--no-nar-rare-missing", action="store_true", help="Disable rare category missingness"
    )

    args = parser.parse_args()

    # Load config and resolve test_size (CLI overrides config)
    _config: dict = {}
    if Path(args.config).exists():
        with open(args.config) as _f:
            _config = yaml.safe_load(_f) or {}
    test_size: float = args.test_size if args.test_size is not None else _config.get("test_size", 0.3)

    # Calculate mild_frac as the remaining fraction
    mild_frac = 1.0 - args.moderate_frac - args.heavy_frac - args.severe_frac

    noise_percentages = {
        "mild": (mild_frac, args.mild_rate),
        "moderate": (args.moderate_frac, args.moderate_rate),
        "heavy": (args.heavy_frac, args.heavy_rate),
        "severe": (args.severe_frac if args.severe_frac > 0 else None, args.severe_rate),
    }

    ar_mechanisms = {
        "enable_numerical_noise": not args.no_ar_numerical,
        "enable_categorical_flips": not args.no_ar_categorical,
        "enable_missing": not args.no_ar_missing,
    }

    nar_mechanisms = {
        "enable_nnar": not args.no_nar_nnar,
        "enable_mnar": not args.no_nar_mnar,
        "enable_systematic_flips": not args.no_nar_systematic,
        "enable_correlated_noise": not args.no_nar_correlated,
        "enable_rare_missing": not args.no_nar_rare_missing,
    }

    process_all_datasets(
        args.input_dir,
        args.output_dir,
        args.dataset,
        noise_percentages,
        ar_mechanisms,
        nar_mechanisms,
        test_size=test_size,
    )
