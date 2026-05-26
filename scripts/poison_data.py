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

    def __init__(
        self,
        seed=42,
        noise_percentages=None,
        nar_min_corruption: float = 0.25,
        nar_max_corruption: float = 0.50,
        clean_feature_frac: float = 0.0,
    ):
        self.seed = seed
        np.random.seed(seed)
        self.poison_log = {}
        self.nar_min_corruption = nar_min_corruption
        self.nar_max_corruption = nar_max_corruption
        self.clean_feature_frac = max(0.0, min(1.0, clean_feature_frac))

        if noise_percentages is None:
            # Tier rates calibrated to produce ~12 % noise + ~9 % MCAR ≈ 21 % AR total.
            # Column fractions: 56.25 % mild / 25 % moderate / 12.5 % heavy / 6.25 % severe.
            self.noise_percentages = {
                "mild": (0.5625, 0.06),
                "moderate": (0.25, 0.12),
                "heavy": (0.125, 0.24),
                "severe": (0.0625, 0.48),
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

        # Reserve the first n_clean columns as completely unpoisoned.
        # They are excluded from noise_dist, so every downstream mechanism
        # that checks `col not in column_noise_dist` will skip them.
        n_clean = int(n_cols * self.clean_feature_frac)
        n_clean = min(n_clean, n_cols)  # never exceed total column count
        if n_clean > 0:
            logger.info(
                f"  Clean feature reservation: {n_clean}/{n_cols} columns kept fully clean "
                f"({self.clean_feature_frac*100:.1f}%)"
            )
        shuffled_cols = shuffled_cols[n_clean:]   # only poisonable columns from here on
        n_cols = len(shuffled_cols)
        if n_cols == 0:
            return {}

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

                k = np.random.uniform(1.2, 1.5)
                noise_std = k * std_signal
                delta = np.random.choice([-1.0, 1.0]) * k * std_signal

                n_poison = int(len(df) * poison_rate)
                if n_poison == 0:
                    continue
                poison_idx = np.random.choice(df.index, size=n_poison, replace=False)

                noise = np.random.normal(delta, noise_std, size=n_poison)
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

        # 3. MCAR: Random missing values — sampled from rows not yet corrupted
        if enable_missing:
            for col in all_data_cols:
                if col not in column_noise_dist:
                    continue

                poison_rate = column_noise_dist[col]

                # Convert integer columns to float before introducing NaN
                if df_poison[col].dtype in ["int64", "int32", "int16", "int8"]:
                    df_poison[col] = df_poison[col].astype("float64")

                n_missing = int(len(df) * poison_rate * 0.75)
                if n_missing == 0:
                    continue

                # Prefer rows not yet corrupted to avoid noise-missingness overlap
                clean_idx = df.index[~poison_mask[col]].to_numpy()
                if len(clean_idx) >= n_missing:
                    missing_idx = np.random.choice(clean_idx, size=n_missing, replace=False)
                else:
                    missing_idx = clean_idx.copy()
                    remaining = n_missing - len(clean_idx)
                    if remaining > 0:
                        overlap_pool = df.index[poison_mask[col]].to_numpy()
                        extra = np.random.choice(overlap_pool, size=min(remaining, len(overlap_pool)), replace=False)
                        missing_idx = np.concatenate([missing_idx, extra])

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
                k = np.random.uniform(1.2, 1.5)
                base_noise_std = k * std_signal
                delta = np.random.choice([-1.0, 1.0]) * k * std_signal

                candidate_vals = df_poison.loc[poison_idx, col]
                valid_mask = candidate_vals.notna()
                valid_idx = poison_idx[valid_mask.values]
                if len(valid_idx) == 0:
                    continue
                vals = df_poison.loc[valid_idx, col].values
                normalized_vals = (vals - min_val) / (max_val - min_val)
                noise_stds = base_noise_std * (1 + 3 * normalized_vals)
                noise = np.random.normal(delta, noise_stds)
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
                already_corrupted = poison_mask.loc[col_vals.index, col]
                prob = prob.where(~already_corrupted, other=0.0)
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
        # Restrict to poisonable columns only — clean-reserved columns must not
        # receive any corruption, including as propagation targets.
        num_cols = [c for c in col_types["numerical"] if c in column_noise_dist]
        if len(num_cols) > 1:
            n_chains = max(1, min(len(num_cols) // 3, 3))
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
            already_corrupted = poison_mask.loc[col_vals.index, col]
            prob = prob.where(~already_corrupted, other=0.0)
            missing_mask = np.random.random(valid.sum()) < prob.values
            missing_idx = col_vals.index[missing_mask]
            df_poison.loc[missing_idx, col] = np.nan
            poison_mask.loc[missing_idx, col] = True

        # Enforce corruption budget: clamp total cell corruption to [min, max].
        # Only poisonable columns (those in column_noise_dist) count toward the
        # budget — clean-reserved columns are never touched by this step.
        poison_mask = self._apply_corruption_budget(
            poison_mask, col_types,
            min_rate=self.nar_min_corruption,
            max_rate=self.nar_max_corruption,
            poisonable_cols=set(column_noise_dist.keys()),
        )

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

    def _apply_corruption_budget(
        self,
        poison_mask: pd.DataFrame,
        col_types: Dict[str, list],
        min_rate: float,
        max_rate: float,
        poisonable_cols: set = None,
    ) -> pd.DataFrame:
        """
        Clamp total cell corruption to [min_rate, max_rate].

        If above max_rate: randomly clear True entries until the rate hits max_rate.
        If below min_rate: randomly set False entries to True until rate hits min_rate.

        Only poisonable feature columns are considered; label columns and
        clean-reserved columns (absent from poisonable_cols) are never touched.

        Args:
            poisonable_cols: Set of column names eligible for corruption.
                             When None, all numerical + categorical columns are used.
        """
        all_feature_cols = col_types["numerical"] + col_types["categorical"]
        if poisonable_cols is not None:
            feature_cols = [c for c in all_feature_cols if c in poisonable_cols]
        else:
            feature_cols = all_feature_cols

        if not feature_cols:
            return poison_mask

        mask_vals = poison_mask[feature_cols].values.copy()  # (N, F) bool array
        total_cells = mask_vals.size
        n_corrupt = int(mask_vals.sum())
        rate = n_corrupt / total_cells

        if rate > max_rate:
            target = int(total_cells * max_rate)
            n_clear = n_corrupt - target
            true_idx = np.flatnonzero(mask_vals)
            clear_idx = np.random.choice(true_idx, size=n_clear, replace=False)
            mask_vals.flat[clear_idx] = False
            logger.info(
                f"  NAR budget cap:   {rate*100:.1f}% → {max_rate*100:.0f}%  "
                f"(cleared {n_clear:,} cells)"
            )

        elif rate < min_rate:
            target = int(total_cells * min_rate)
            n_add = target - n_corrupt
            false_idx = np.flatnonzero(~mask_vals)
            if len(false_idx) >= n_add:
                add_idx = np.random.choice(false_idx, size=n_add, replace=False)
                mask_vals.flat[add_idx] = True
            logger.info(
                f"  NAR budget floor: {rate*100:.1f}% → {min_rate*100:.0f}%  "
                f"(added {n_add:,} cells)"
            )

        poison_mask = poison_mask.copy()
        poison_mask[feature_cols] = mask_vals
        return poison_mask


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
    datasets: list = None,
    noise_percentages: dict = None,
    ar_mechanisms: dict = None,
    nar_mechanisms: dict = None,
    test_size: float = 0.3,
    nar_min_corruption: float = 0.25,
    nar_max_corruption: float = 0.50,
    clean_feature_frac: float = 0.0,
):
    """
    Process all datasets and create poisoned versions.

    Args:
        input_dir: Directory containing clean CSV files
        output_dir: Directory to save poisoned files
        datasets: Optional list of dataset names to process (from config.yaml)
        noise_percentages: Optional custom noise distribution percentages
        ar_mechanisms: Dict of enabled AR mechanisms
        nar_mechanisms: Dict of enabled NAR mechanisms
        test_size: Fraction of data to hold out as clean test set
        nar_min_corruption: Minimum total cell corruption for NAR (floor)
        nar_max_corruption: Maximum total cell corruption for NAR (cap)
        clean_feature_frac: Fraction of feature columns kept completely unpoisoned (0–1)
    """
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.join(output_dir, "ar"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "nar"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "metrics"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "test"), exist_ok=True)

    poisoner = DataPoisoner(
        seed=42,
        noise_percentages=noise_percentages,
        nar_min_corruption=nar_min_corruption,
        nar_max_corruption=nar_max_corruption,
        clean_feature_frac=clean_feature_frac,
    )

    if ar_mechanisms is None:
        ar_mechanisms = {}
    if nar_mechanisms is None:
        nar_mechanisms = {}

    csv_files = sorted(Path(input_dir).glob("*.csv"))

    if datasets:
        dataset_set = set(datasets)
        csv_files = [f for f in csv_files if f.stem[10:] in dataset_set]
        if not csv_files:
            logger.error(f"None of the specified datasets found in {input_dir}")
            return
        logger.info(f"Processing {len(csv_files)} configured dataset(s)")
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
            python poison_data.py
            python poison_data.py --dataset my_dataset
            python poison_data.py --severe-frac 0.1 --severe-rate 0.50
            python poison_data.py --no-ar-numerical --no-nar-correlated

            Parameter priority (highest to lowest):
              1. CLI flags passed explicitly
              2. config.yaml  poisoning:  section
              3. Built-in defaults

            Poisoning Modes:
            AR (At Random): Random poisoning with stratified column distribution
            NAR (Not At Random): Advanced non-random poisoning mechanisms

            Note:
            Target columns (cls_*, reg_*) are NEVER poisoned to preserve labels.
                """,
    )
    parser.add_argument(
        "--input_dir", type=str, default="data",
        help="Directory containing clean CSV files (default: data)",
    )
    parser.add_argument(
        "--output_dir", type=str, default="data_poisoned",
        help="Directory to save poisoned files (default: data_poisoned)",
    )
    parser.add_argument(
        "--config", type=str, default="config.yaml",
        help="Path to experiment config file (default: config.yaml)",
    )
    parser.add_argument(
        "--dataset", type=str,
        help="Process only this specific dataset (name without .csv)",
    )
    parser.add_argument(
        "--test-size", type=float, default=None,
        help="Fraction of data held out as clean test set "
             "(overrides config.yaml test_size; default: 0.3)",
    )

    noise_group = parser.add_argument_group(
        "Noise Distribution",
        "Per-tier corruption rates and column fractions. "
        "Defaults are read from config.yaml poisoning.noise when not set here. "
        "mild_frac is computed as 1 - moderate_frac - heavy_frac - severe_frac.",
    )
    noise_group.add_argument(
        "--mild-rate", type=float, default=None,
        help="Corruption rate for mildly noisy columns (config default: 0.06)",
    )
    noise_group.add_argument(
        "--moderate-frac", type=float, default=None,
        help="Fraction of columns assigned to moderate noise (config default: 0.30)",
    )
    noise_group.add_argument(
        "--moderate-rate", type=float, default=None,
        help="Corruption rate for moderately noisy columns (config default: 0.12)",
    )
    noise_group.add_argument(
        "--heavy-frac", type=float, default=None,
        help="Fraction of columns assigned to heavy noise (config default: 0.20)",
    )
    noise_group.add_argument(
        "--heavy-rate", type=float, default=None,
        help="Corruption rate for heavily noisy columns (config default: 0.24)",
    )
    noise_group.add_argument(
        "--severe-frac", type=float, default=None,
        help="Fraction of columns assigned to severe noise (config default: 0.10)",
    )
    noise_group.add_argument(
        "--severe-rate", type=float, default=None,
        help="Corruption rate for severely noisy columns (config default: 0.48)",
    )

    ar_group = parser.add_argument_group(
        "AR Mode Mechanisms",
        "Flags disable individual AR mechanisms. "
        "Defaults are read from config.yaml poisoning.ar.",
    )
    ar_group.add_argument(
        "--no-ar-numerical", action="store_true",
        help="Disable Gaussian noise on numerical features",
    )
    ar_group.add_argument(
        "--no-ar-categorical", action="store_true",
        help="Disable random flips for categorical features",
    )
    ar_group.add_argument(
        "--no-ar-missing", action="store_true",
        help="Disable MCAR (Missing Completely At Random)",
    )

    nar_group = parser.add_argument_group(
        "NAR Mode Mechanisms",
        "Flags disable individual NAR mechanisms. "
        "Defaults are read from config.yaml poisoning.nar.",
    )
    nar_group.add_argument(
        "--no-nar-nnar", action="store_true",
        help="Disable NNAR (heteroscedastic noise)",
    )
    nar_group.add_argument(
        "--no-nar-mnar", action="store_true",
        help="Disable MNAR (extreme values more likely missing)",
    )
    nar_group.add_argument(
        "--no-nar-systematic", action="store_true",
        help="Disable systematic categorical confusion",
    )
    nar_group.add_argument(
        "--no-nar-correlated", action="store_true",
        help="Disable correlated noise propagation",
    )
    nar_group.add_argument(
        "--no-nar-rare-missing", action="store_true",
        help="Disable rare category missingness",
    )
    nar_group.add_argument(
        "--nar-min-corruption", type=float, default=None,
        help="Minimum total cell corruption for NAR, 0–1 (config default: 0.25)",
    )
    nar_group.add_argument(
        "--nar-max-corruption", type=float, default=None,
        help="Maximum total cell corruption for NAR, 0–1 (config default: 0.50)",
    )

    noise_group.add_argument(
        "--clean-feature-frac", type=float, default=None,
        help=(
            "Fraction of feature columns kept completely clean (0–1, config default: 0.0). "
            "0.0 = all features may be poisoned; 0.3 = 30%% of columns are never touched."
        ),
    )

    args = parser.parse_args()

    # ── Load config ────────────────────────────────────────────────────────────
    _config: dict = {}
    if Path(args.config).exists():
        with open(args.config) as _f:
            _config = yaml.safe_load(_f) or {}

    # Sub-sections of the poisoning block (all optional; fall back to built-ins)
    _poison_cfg  = _config.get("poisoning", {})
    _noise_cfg   = _poison_cfg.get("noise", {})
    _ar_cfg      = _poison_cfg.get("ar", {})
    _nar_cfg     = _poison_cfg.get("nar", {})

    # Helper: CLI value wins if given, else config, else built-in default
    def _resolve(cli_val, cfg_section, cfg_key, builtin):
        if cli_val is not None:
            return cli_val
        return cfg_section.get(cfg_key, builtin)

    # ── Resolve test_size ──────────────────────────────────────────────────────
    test_size: float = _resolve(args.test_size, _config, "test_size", 0.3)

    # ── Datasets ───────────────────────────────────────────────────────────────
    datasets = [args.dataset] if args.dataset else (_config.get("datasets") or None)

    # ── Noise distribution ─────────────────────────────────────────────────────
    mild_rate     = _resolve(args.mild_rate,     _noise_cfg, "mild_rate",     0.06)
    moderate_frac = _resolve(args.moderate_frac, _noise_cfg, "moderate_frac", 0.30)
    moderate_rate = _resolve(args.moderate_rate, _noise_cfg, "moderate_rate", 0.12)
    heavy_frac    = _resolve(args.heavy_frac,    _noise_cfg, "heavy_frac",    0.20)
    heavy_rate    = _resolve(args.heavy_rate,    _noise_cfg, "heavy_rate",    0.24)
    severe_frac   = _resolve(args.severe_frac,   _noise_cfg, "severe_frac",   0.10)
    severe_rate   = _resolve(args.severe_rate,   _noise_cfg, "severe_rate",   0.48)

    mild_frac = 1.0 - moderate_frac - heavy_frac - severe_frac

    noise_percentages = {
        "mild":     (mild_frac,                              mild_rate),
        "moderate": (moderate_frac,                          moderate_rate),
        "heavy":    (heavy_frac,                             heavy_rate),
        "severe":   (severe_frac if severe_frac > 0 else None, severe_rate),
    }

    # ── AR mechanisms ──────────────────────────────────────────────────────────
    # Config provides the base default; --no-* CLI flags override to False.
    ar_mechanisms = {
        "enable_numerical_noise":   _ar_cfg.get("enable_numerical_noise",   True) and not args.no_ar_numerical,
        "enable_categorical_flips": _ar_cfg.get("enable_categorical_flips", True) and not args.no_ar_categorical,
        "enable_missing":           _ar_cfg.get("enable_missing",           True) and not args.no_ar_missing,
    }

    # ── NAR mechanisms ─────────────────────────────────────────────────────────
    nar_min_corruption = _resolve(args.nar_min_corruption, _nar_cfg, "min_corruption", 0.25)
    nar_max_corruption = _resolve(args.nar_max_corruption, _nar_cfg, "max_corruption", 0.50)

    nar_mechanisms = {
        "enable_nnar":             _nar_cfg.get("enable_nnar",             True) and not args.no_nar_nnar,
        "enable_mnar":             _nar_cfg.get("enable_mnar",             True) and not args.no_nar_mnar,
        "enable_systematic_flips": _nar_cfg.get("enable_systematic_flips", True) and not args.no_nar_systematic,
        "enable_correlated_noise": _nar_cfg.get("enable_correlated_noise", True) and not args.no_nar_correlated,
        "enable_rare_missing":     _nar_cfg.get("enable_rare_missing",     True) and not args.no_nar_rare_missing,
    }

    # ── Clean feature fraction ─────────────────────────────────────────────────
    clean_feature_frac = _resolve(args.clean_feature_frac, _noise_cfg, "clean_feature_frac", 0.0)

    process_all_datasets(
        args.input_dir,
        args.output_dir,
        datasets,
        noise_percentages,
        ar_mechanisms,
        nar_mechanisms,
        test_size=test_size,
        nar_min_corruption=nar_min_corruption,
        nar_max_corruption=nar_max_corruption,
        clean_feature_frac=clean_feature_frac,
    )
