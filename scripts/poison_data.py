#!/usr/bin/env python3
"""
Data Poisoning Pipeline for ML Datasets
Implements AR (At Random) and NAR (Not At Random) poisoning modes with state-of-the-art mechanisms

Note: Target columns (cls_*, reg_*) are NEVER poisoned to preserve labels.
"""
import itertools
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import yaml
from loguru import logger

logger.remove()
logger.add(
    sys.stderr,
    level="INFO",
    format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <level>{message}</level>",
)


def compute_ar_expected_rate(noise_percentages: Dict[str, tuple], clean_feature_frac: float = 0.0) -> float:
    """
    Nominal preset total: the fraction of ALL feature cells (clean-reserved
    columns included) a preset is designed to corrupt,

        total = (Σ tier_column_share × tier_rate) × (1 − clean_feature_frac)

    It is exact only in the limit of many columns: a dataset has an integer
    number of columns per tier, so its realized total is the one produced by
    ``allocate_columns`` — which picks, among the admissible roundings, the one
    closest to this value.
    """
    weighted_rate = sum((frac or 0.0) * rate for frac, rate in noise_percentages.values())
    poisonable_share = 1.0 - clean_feature_frac
    return weighted_rate * poisonable_share


TIER_ORDER = ("mild", "moderate", "heavy", "severe")


def allocate_columns(
    n_features: int,
    n_unpoisonable: int,
    noise_percentages: Dict[str, tuple],
    clean_feature_frac: float,
) -> Tuple[int, Dict[str, int], float]:
    """
    How many feature columns stay clean and how many go to each tier.

    Every count may only be its proportional quota rounded down or up — the
    clean share ``n_features × clean_feature_frac`` and each tier's share of the
    poisonable columns — so the column shares of POISONING.md are kept to within
    one column each. Among those admissible roundings, the one whose dataset
    total (``Σ count × tier_rate / n_features``) is closest to the preset target
    wins; ties go to the rounding closest to the quotas.

    ``n_unpoisonable`` columns (entirely missing in the source, so there is
    nothing to corrupt) are forced into the clean set.

    Returns (n_clean, {tier: n_columns}, realized_total).
    """
    tiers = [t for t in TIER_ORDER
             if t in noise_percentages and (noise_percentages[t][0] or 0) > 0]
    target = compute_ar_expected_rate(noise_percentages, clean_feature_frac)
    if n_features == 0:
        return 0, {t: 0 for t in tiers}, 0.0

    quota_clean = n_features * clean_feature_frac
    clean_opts = {int(np.floor(quota_clean)), int(np.ceil(quota_clean))}
    clean_opts = sorted({min(max(c, n_unpoisonable), n_features) for c in clean_opts})

    best = None
    for n_clean in clean_opts:
        n_poison = n_features - n_clean
        quotas = [noise_percentages[t][0] * n_poison for t in tiers]
        options = [sorted({int(np.floor(q)), int(np.ceil(q))}) for q in quotas]
        for combo in itertools.product(*options):
            if sum(combo) != n_poison:
                continue
            realized = sum(c * noise_percentages[t][1] for c, t in zip(combo, tiers)) / n_features
            key = (
                round(abs(realized - target), 12),
                round(abs(n_clean - quota_clean) + sum(abs(c - q) for c, q in zip(combo, quotas)), 12),
            )
            if best is None or key < best[0]:
                best = (key, n_clean, dict(zip(tiers, combo)), realized)

    if best is None:  # no tiers configured: nothing can be poisoned
        return n_features, {t: 0 for t in tiers}, 0.0
    _, n_clean, counts, realized = best
    return n_clean, counts, realized


def column_budget(n_rows: int, rate: float) -> int:
    """Cells a column must have corrupted at ``rate``: rate × rows, rounded half up."""
    return int(np.floor(n_rows * rate + 0.5))


def split_budget(n_total: int, value_on: bool, value_possible: bool, missing_on: bool) -> Tuple[int, int]:
    """
    Split a column budget between its value-level mechanism (noise / flip) and
    its missingness mechanism: 50/50 when both apply, the whole budget to the
    one that does otherwise. A value mechanism that cannot act on the column —
    a flip on a single-category column — counts as not applicable.
    """
    value_active = value_on and value_possible
    if value_active and missing_on:
        n_value = n_total // 2
        return n_value, n_total - n_value
    if value_active:
        return n_total, 0
    if missing_on:
        return 0, n_total
    return 0, 0


def corruption_mask(df_clean: pd.DataFrame, df_poison: pd.DataFrame, columns: List[str]) -> pd.DataFrame:
    """
    The poison mask, derived from the data itself: True where a feature cell of
    ``df_poison`` differs from ``df_clean`` — a value changed, or a value made
    missing — False where the cell is untouched (including cells that were
    already missing in the source). Every other column is False.
    """
    mask = pd.DataFrame(False, index=df_clean.index, columns=df_clean.columns, dtype=bool)
    for col in columns:
        c, p = df_clean[col], df_poison[col]
        c_na, p_na = c.isna(), p.isna()
        both = ~c_na & ~p_na
        cn, pn = pd.to_numeric(c, errors="coerce"), pd.to_numeric(p, errors="coerce")
        numeric = both & cn.notna() & pn.notna()
        differs = numeric & ((pn - cn).abs() > 1e-12 * np.maximum(1.0, cn.abs()))
        other = both & ~numeric
        differs |= other & (c.astype(str) != p.astype(str))
        mask[col] = (c_na != p_na) | differs
    return mask


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
        nar_min_corruption: float = None,
        nar_max_corruption: float = None,
        ar_max_corruption: float = None,
        clean_feature_frac: float = 0.0,
    ):
        self.seed = seed
        np.random.seed(seed)
        self.clean_feature_frac = max(0.0, min(1.0, clean_feature_frac))

        if noise_percentages is None:
            # Tier rates calibrated to produce ~21.7 % AR total over poisonable
            # columns (split 50/50 between noise and MCAR, no overlap).
            # Column fractions: 56.25 % mild / 25 % moderate / 12.5 % heavy / 6.25 % severe.
            self.noise_percentages = {
                "mild": (0.5625, 0.105),
                "moderate": (0.25, 0.21),
                "heavy": (0.125, 0.42),
                "severe": (0.0625, 0.84),
            }
        else:
            self.noise_percentages = noise_percentages

        # Nominal preset total (fraction of ALL feature cells). Each dataset
        # realizes the closest total its integer column counts allow — see
        # allocate_columns — and AR and NAR both land on that realized total by
        # construction, because every poisonable column is corrupted at exactly
        # its tier rate in both modes.
        self.target_rate = compute_ar_expected_rate(self.noise_percentages, self.clean_feature_frac)

        # Explicit overrides only. Left unset (None), no clamp runs: the per-column
        # budgets already fix the total. When set, the clamp acts on the data —
        # it restores or corrupts real cells — never on the mask alone.
        self.nar_min_corruption = nar_min_corruption
        self.nar_max_corruption = nar_max_corruption
        self.ar_max_corruption = ar_max_corruption

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

    def _create_stratified_noise_distribution(
        self, columns: list, df: Optional[pd.DataFrame] = None
    ) -> Dict[str, float]:
        """
        Assign every feature column either to the clean-reserved set or to a
        noise tier, and return {column: tier rate} for the poisonable ones.

        The counts come from ``allocate_columns`` (clean share and tier shares
        kept to within one column of POISONING.md, the rounding closest to the
        preset total); which column gets which role is random. Columns with no
        observed value in ``df`` cannot be corrupted and are always clean.
        """
        n_cols = len(columns)
        if n_cols == 0:
            return {}

        unpoisonable = [c for c in columns if df is not None and df[c].isna().all()]
        n_clean, counts, realized = allocate_columns(
            n_cols, len(unpoisonable), self.noise_percentages, self.clean_feature_frac
        )

        shuffled_cols = [c for c in columns if c not in unpoisonable]
        np.random.shuffle(shuffled_cols)
        n_extra_clean = n_clean - len(unpoisonable)
        clean_cols = unpoisonable + shuffled_cols[:n_extra_clean]
        poison_cols = shuffled_cols[n_extra_clean:]

        noise_dist = {}
        idx = 0
        for tier in TIER_ORDER:
            for _ in range(counts.get(tier, 0)):
                noise_dist[poison_cols[idx]] = self.noise_percentages[tier][1]
                idx += 1

        logger.info(
            f"  Columns: {len(clean_cols)}/{n_cols} clean-reserved "
            f"({100 * len(clean_cols) / n_cols:.1f}%, nominal {100 * self.clean_feature_frac:.1f}%)"
            + (f", {len(unpoisonable)} entirely missing" if unpoisonable else "")
            + " | tiers: " + ", ".join(f"{counts.get(t, 0)} {t}" for t in TIER_ORDER if t in counts)
        )
        logger.info(
            f"  Planned corruption: {100 * realized:.2f}% of feature cells "
            f"(preset target {100 * self.target_rate:.2f}%)"
        )
        return noise_dist

    def _observed(self, df: pd.DataFrame, col: str) -> np.ndarray:
        """Rows holding a value in the source: the only cells that can be corrupted."""
        return df.index[df[col].notna()].to_numpy()

    def _column_budget(self, df: pd.DataFrame, col: str, rate: float, observed: np.ndarray) -> int:
        """rate × rows, capped by the observed cells available (with a warning)."""
        n_total = column_budget(len(df), rate)
        if n_total > len(observed):
            logger.warning(
                f"  {col}: budget {n_total} cells ({100 * rate:.1f}%) but only "
                f"{len(observed)} observed values — corrupting all of them"
            )
            n_total = len(observed)
        return n_total

    def _log_realized(self, df: pd.DataFrame, poison_mask: pd.DataFrame, col_types: Dict, label: str):
        feats = col_types["numerical"] + col_types["categorical"]
        if not feats:
            return
        rate = poison_mask[feats].values.mean()
        logger.info(
            f"  {label} realized: {int(poison_mask[feats].values.sum()):,} cells = "
            f"{100 * rate:.2f}% of feature cells (preset target {100 * self.target_rate:.2f}%)"
        )

    def poison_ar(
        self,
        df: pd.DataFrame,
        column_noise_dist: dict = None,
        enable_numerical_noise: bool = True,
        enable_categorical_flips: bool = True,
        enable_missing: bool = True,
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        AR MODE (At Random): corruption applied uniformly at random.

        Each poisonable column gets exactly ``round(rows × tier rate)`` corrupted
        cells, drawn uniformly among the cells that hold a value in the source
        (a cell that is already missing cannot be poisoned), split 50/50 with no
        overlap between:
        1. a value-level mechanism — additive Gaussian noise (numerical) or a
           flip to a uniformly random other category (categorical);
        2. MCAR missingness.
        If one of the two is disabled or cannot act (a flip on a single-category
        column), the other receives the whole budget.

        The mask is derived from the data: True exactly where a cell was changed
        or made missing, False where it is untouched.

        Note: Target columns (cls_*, reg_*) are NEVER poisoned to preserve labels.

        Returns:
            poisoned_df: Poisoned dataframe
            poison_mask: Boolean mask (True = poisoned, False = untouched)
        """
        enabled_mechanisms = []
        if enable_numerical_noise:
            enabled_mechanisms.append("numerical_noise")
        if enable_categorical_flips:
            enabled_mechanisms.append("categorical_flips")
        if enable_missing:
            enabled_mechanisms.append("MCAR")

        logger.info("Applying AR (At Random) poisoning with stratified column distribution")
        logger.info(
            f"  Enabled mechanisms: {', '.join(enabled_mechanisms) if enabled_mechanisms else 'none'}"
        )

        df_poison = df.copy()
        col_types = self.identify_column_types(df)
        all_data_cols = col_types["numerical"] + col_types["categorical"]

        if column_noise_dist is None:
            column_noise_dist = self._create_stratified_noise_distribution(all_data_cols, df)

        # 1. Numerical features: Gaussian noise + MCAR.
        for col in col_types["numerical"]:
            if col not in column_noise_dist:
                continue
            observed = self._observed(df, col)
            if len(observed) == 0:
                continue
            n_total = self._column_budget(df, col, column_noise_dist[col], observed)
            n_noise, n_missing = split_budget(n_total, enable_numerical_noise, True, enable_missing)
            if n_noise + n_missing == 0:
                continue

            if df_poison[col].dtype in ["int64", "int32", "int16", "int8"]:
                df_poison[col] = df_poison[col].astype("float64")

            poison_idx = np.random.choice(observed, size=n_noise + n_missing, replace=False)
            noise_idx, missing_idx = poison_idx[:n_noise], poison_idx[n_noise:]

            if len(noise_idx) > 0:
                values = df[col].dropna()
                std_signal = values.std()
                if not std_signal or np.isnan(std_signal):
                    std_signal = abs(values.mean()) * 0.1 if values.mean() != 0 else 1.0
                k = np.random.uniform(1.2, 1.5)
                noise_std = k * std_signal
                delta = np.random.choice([-1.0, 1.0]) * k * std_signal
                noise = np.random.normal(delta, noise_std, size=len(noise_idx))
                df_poison.loc[noise_idx, col] = df_poison.loc[noise_idx, col] + noise

            if len(missing_idx) > 0:
                df_poison.loc[missing_idx, col] = np.nan

        # 2. Categorical features: random label flips + MCAR.
        for col in col_types["categorical"]:
            if col not in column_noise_dist:
                continue
            observed = self._observed(df, col)
            if len(observed) == 0:
                continue
            unique_vals = df[col].dropna().unique()
            n_total = self._column_budget(df, col, column_noise_dist[col], observed)
            n_flip, n_missing = split_budget(
                n_total, enable_categorical_flips, len(unique_vals) > 1, enable_missing
            )
            if n_flip + n_missing == 0:
                continue

            poison_idx = np.random.choice(observed, size=n_flip + n_missing, replace=False)
            flip_idx, missing_idx = poison_idx[:n_flip], poison_idx[n_flip:]

            for idx in flip_idx:
                current_val = df.loc[idx, col]
                other_vals = [v for v in unique_vals if v != current_val]
                df_poison.loc[idx, col] = np.random.choice(other_vals)

            if len(missing_idx) > 0:
                if df_poison[col].dtype in ["int64", "int32", "int16", "int8"]:
                    df_poison[col] = df_poison[col].astype("float64")
                df_poison.loc[missing_idx, col] = np.nan

        # Optional explicit cap (poisoning.ar.max_corruption). Unset, nothing runs:
        # the per-column budgets above already fix the total.
        df_poison = self._apply_corruption_budget(
            df, df_poison, col_types,
            max_rate=self.ar_max_corruption,
            poisonable_cols=set(column_noise_dist.keys()),
            label="AR",
        )

        poison_mask = corruption_mask(df, df_poison, all_data_cols)
        self._log_realized(df, poison_mask, col_types, "AR")
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
        NAR MODE (Not At Random): value-dependent corruption.

        As in AR, each poisonable column gets exactly ``round(rows × tier rate)``
        corrupted cells among its observed ones, split 50/50 with no overlap
        between a paired "noise" and "missing" mechanism — but which rows land
        in each half depends on the values:
        1. Numerical: NNAR (heteroscedastic noise, up to 4x larger near the
           column max) + MNAR (missingness weighted by distance from the median).
        2. Categorical: systematic confusion-pattern flips + rare-value
           missingness (frequency-inverse weighting).
        3. Correlated noise: a few anchor numerical columns propagate noise to
           the other numerical columns on the anchors' corrupted rows. It lands
           on top of the budgets above, so each column's *noise* half is then
           brought back to its size by restoring randomly chosen noisy cells to
           their clean value — correlated and own noise compete for the same
           half, the missing half is untouched. Every column therefore ends at
           exactly its tier rate with an exact 50/50 split, and the dataset total
           equals AR's.

        The mask is derived from the data: True exactly where a cell was changed
        or made missing, False where it is untouched.

        Note: Target columns (cls_*, reg_*) are NEVER poisoned to preserve labels.

        Returns:
            poisoned_df: Poisoned dataframe
            poison_mask: Boolean mask (True = poisoned, False = untouched)
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

        logger.info("Applying NAR (Not At Random) poisoning with stratified column distribution")
        logger.info(
            f"  Enabled mechanisms: {', '.join(enabled_mechanisms) if enabled_mechanisms else 'none'}"
        )

        df_poison = df.copy()
        col_types = self.identify_column_types(df)
        all_data_cols = col_types["numerical"] + col_types["categorical"]

        if column_noise_dist is None:
            column_noise_dist = self._create_stratified_noise_distribution(all_data_cols, df)

        # Size of each numerical column's noise half, for the rebalancing in step 3.
        noise_budget: Dict[str, int] = {}

        # 1. Numerical features: NNAR + MNAR. MNAR's candidates are exactly the
        #    observed rows NNAR didn't take, so the two never overlap.
        for col in col_types["numerical"]:
            if col not in column_noise_dist:
                continue
            observed = self._observed(df, col)
            if len(observed) == 0:
                continue
            n_total = self._column_budget(df, col, column_noise_dist[col], observed)
            n_noise, n_missing = split_budget(n_total, enable_nnar, True, enable_mnar)
            noise_budget[col] = n_noise
            if n_noise + n_missing == 0:
                continue

            if df_poison[col].dtype in ["int64", "int32", "int16", "int8"]:
                df_poison[col] = df_poison[col].astype("float64")

            shuffled_idx = np.random.permutation(observed)
            noise_idx = shuffled_idx[:n_noise]
            remaining_pool = shuffled_idx[n_noise:]
            values = df[col].dropna()

            if len(noise_idx) > 0:
                min_val, max_val = values.min(), values.max()
                std_signal = values.std()
                if not std_signal or np.isnan(std_signal):
                    std_signal = abs(values.mean()) * 0.1 if values.mean() != 0 else 1.0
                k = np.random.uniform(1.2, 1.5)
                base_noise_std = k * std_signal
                delta = np.random.choice([-1.0, 1.0]) * k * std_signal
                vals = df.loc[noise_idx, col].to_numpy(dtype=float)
                # A constant column has no range: every value gets the base noise.
                normalized_vals = (
                    (vals - min_val) / (max_val - min_val) if max_val > min_val
                    else np.zeros_like(vals)
                )
                noise_stds = base_noise_std * (1 + 3 * normalized_vals)
                df_poison.loc[noise_idx, col] = vals + np.random.normal(delta, noise_stds)

            if n_missing > 0:
                median = values.median()
                weights = (df.loc[remaining_pool, col] - median).abs().to_numpy() + 1e-9
                probs = weights / weights.sum()
                missing_idx = np.random.choice(remaining_pool, size=n_missing, replace=False, p=probs)
                df_poison.loc[missing_idx, col] = np.nan

        # 2. Categorical features: systematic flips + rare-value missingness.
        for col in col_types["categorical"]:
            if col not in column_noise_dist:
                continue
            observed = self._observed(df, col)
            if len(observed) == 0:
                continue
            unique_vals = df[col].dropna().unique()
            n_total = self._column_budget(df, col, column_noise_dist[col], observed)
            n_flip, n_missing = split_budget(
                n_total, enable_systematic_flips, len(unique_vals) > 1, enable_rare_missing
            )
            if n_flip + n_missing == 0:
                continue

            shuffled_idx = np.random.permutation(observed)
            flip_idx = shuffled_idx[:n_flip]
            remaining_pool = shuffled_idx[n_flip:]

            if len(flip_idx) > 0:
                vals_list = list(unique_vals)
                confusion_pairs = {v: vals_list[(i + 1) % len(vals_list)] for i, v in enumerate(vals_list)}
                for idx in flip_idx:
                    current_val = df.loc[idx, col]
                    if np.random.random() < 0.7:
                        df_poison.loc[idx, col] = confusion_pairs[current_val]
                    else:
                        other_vals = [v for v in unique_vals if v != current_val]
                        df_poison.loc[idx, col] = np.random.choice(other_vals)

            if n_missing > 0:
                if df_poison[col].dtype in ["int64", "int32", "int16", "int8"]:
                    df_poison[col] = df_poison[col].astype("float64")
                freq_series = df[col].value_counts() / len(observed)
                frequencies = df.loc[remaining_pool, col].map(freq_series).fillna(0)
                weights = (1 - frequencies).to_numpy() + 1e-9
                probs = weights / weights.sum()
                missing_idx = np.random.choice(remaining_pool, size=n_missing, replace=False, p=probs)
                df_poison.loc[missing_idx, col] = np.nan

        # 3. Correlated noise across numerical features, then per-column
        #    rebalancing of the noise half. Only poisonable columns take part.
        num_cols = [c for c in col_types["numerical"] if c in column_noise_dist]
        if enable_correlated_noise and len(num_cols) > 1:
            n_chains = max(1, min(len(num_cols) // 3, 3))
            anchor_cols = np.random.choice(num_cols, size=min(n_chains, len(num_cols)), replace=False)
            current = corruption_mask(df, df_poison, num_cols)

            for anchor_col in anchor_cols:
                poisoned_rows = current.index[current[anchor_col]]
                for other_col in num_cols:
                    if other_col == anchor_col:
                        continue
                    values = df[other_col].dropna()
                    if len(values) == 0:
                        continue
                    std_signal = values.std()
                    if not std_signal or std_signal <= 0:
                        continue
                    noise_std = std_signal / (10 ** (15 / 20))  # 15 dB SNR
                    col_at_rows = df_poison.loc[poisoned_rows, other_col]
                    eligible = col_at_rows.notna().values & (np.random.random(len(poisoned_rows)) < 0.6)
                    selected = poisoned_rows[eligible]
                    if len(selected) == 0:
                        continue
                    df_poison.loc[selected, other_col] += np.random.normal(0, noise_std, size=len(selected))

            restored = 0
            after = corruption_mask(df, df_poison, num_cols)
            for col in num_cols:
                noisy = after.index[after[col] & df_poison[col].notna()].to_numpy()
                excess = len(noisy) - noise_budget.get(col, 0)
                if excess > 0:
                    back = np.random.choice(noisy, size=excess, replace=False)
                    df_poison.loc[back, col] = df.loc[back, col]
                    restored += excess
            logger.info(
                f"  Correlated noise: {len(anchor_cols)} anchor(s); {restored:,} noisy cells restored "
                f"so every column keeps exactly its tier budget"
            )

        # Optional explicit clamp (poisoning.nar.min/max_corruption). Unset,
        # nothing runs: NAR already lands on AR's total.
        df_poison = self._apply_corruption_budget(
            df, df_poison, col_types,
            max_rate=self.nar_max_corruption,
            min_rate=self.nar_min_corruption,
            poisonable_cols=set(column_noise_dist.keys()),
            label="NAR",
        )

        poison_mask = corruption_mask(df, df_poison, all_data_cols)
        self._log_realized(df, poison_mask, col_types, "NAR")
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
        df: pd.DataFrame,
        df_poison: pd.DataFrame,
        col_types: Dict[str, list],
        max_rate: Optional[float] = None,
        min_rate: Optional[float] = None,
        poisonable_cols: set = None,
        label: str = "",
    ) -> pd.DataFrame:
        """
        Clamp total cell corruption to [min_rate, max_rate], as a fraction of ALL
        feature cells (clean-reserved columns included) — acting on the *data*.

        Above max_rate: randomly chosen corrupted cells are restored to their
        clean value. Below min_rate: randomly chosen untouched observed cells are
        corrupted with their column's value-level mechanism (Gaussian noise for
        numerical columns, a flip to another category for categorical ones, or
        missingness where a flip is impossible). The mask, derived afterwards
        from the data, follows automatically.

        Both bounds default to None (no clamp): the per-column budgets already
        fix the total. They only matter as explicit overrides, and then they
        trade per-column exactness for the requested dataset total.
        """
        if max_rate is None and min_rate is None:
            return df_poison

        all_feature_cols = col_types["numerical"] + col_types["categorical"]
        feature_cols = [c for c in all_feature_cols
                        if poisonable_cols is None or c in poisonable_cols]
        if not feature_cols:
            return df_poison

        total_cells = len(df) * len(all_feature_cols)
        mask = corruption_mask(df, df_poison, feature_cols)[feature_cols]
        n_corrupt = int(mask.values.sum())
        rate = n_corrupt / total_cells
        tag = f"{label} budget" if label else "Budget"
        df_poison = df_poison.copy()

        if max_rate is not None and rate > max_rate:
            n_clear = n_corrupt - int(total_cells * max_rate)
            rows, cols = np.nonzero(mask.values)
            pick = np.random.choice(len(rows), size=n_clear, replace=False)
            for r, c in zip(rows[pick], cols[pick]):
                col = feature_cols[c]
                df_poison.iat[r, df_poison.columns.get_loc(col)] = df[col].iat[r]
            logger.info(f"  {tag} cap:   {rate*100:.2f}% → {max_rate*100:.2f}%  (restored {n_clear:,} cells)")

        elif min_rate is not None and rate < min_rate:
            n_add = int(total_cells * min_rate) - n_corrupt
            free = ~mask.values & df[feature_cols].notna().values
            rows, cols = np.nonzero(free)
            n_add = min(n_add, len(rows))
            pick = np.random.choice(len(rows), size=n_add, replace=False)
            for r, c in zip(rows[pick], cols[pick]):
                col = feature_cols[c]
                j = df_poison.columns.get_loc(col)
                if col in col_types["numerical"]:
                    if df_poison[col].dtype.kind in "iu":
                        df_poison[col] = df_poison[col].astype("float64")
                    std = df[col].std() or 1.0
                    df_poison.iat[r, j] = df[col].iat[r] + np.random.choice([-1.0, 1.0]) * np.random.uniform(1.2, 1.5) * std
                else:
                    others = [v for v in df[col].dropna().unique() if v != df[col].iat[r]]
                    df_poison.iat[r, j] = np.random.choice(others) if others else np.nan
            logger.info(f"  {tag} floor: {rate*100:.2f}% → {min_rate*100:.2f}%  (corrupted {n_add:,} cells)")

        return df_poison


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
    nar_min_corruption: float = None,
    nar_max_corruption: float = None,
    ar_max_corruption: float = None,
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
        nar_min_corruption: Explicit floor on NAR's total cell corruption, as a
                            fraction of ALL feature cells. None (default): no
                            floor — NAR lands on AR's total by construction.
        nar_max_corruption: Explicit cap on NAR's total, same base. None: no cap.
        ar_max_corruption: Explicit cap on AR's total, same base. None: no cap.
                           Any bound that is set acts on the data (restoring or
                           corrupting real cells), and trades per-column
                           exactness for the requested total.
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
        ar_max_corruption=ar_max_corruption,
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
            column_noise_dist = poisoner._create_stratified_noise_distribution(all_data_cols, df_train)

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
        "Per-tier corruption rates and column fractions. Rates default to the "
        "preset named by config.yaml poisoning.presets.default; column fractions "
        "come from config.yaml poisoning.noise. "
        "mild_frac is computed as 1 - moderate_frac - heavy_frac - severe_frac.",
    )
    noise_group.add_argument(
        "--mild-rate", type=float, default=None,
        help="Corruption rate for mildly noisy columns",
    )
    noise_group.add_argument(
        "--moderate-frac", type=float, default=None,
        help="Fraction of columns assigned to moderate noise (config default: 0.35)",
    )
    noise_group.add_argument(
        "--moderate-rate", type=float, default=None,
        help="Corruption rate for moderately noisy columns",
    )
    noise_group.add_argument(
        "--heavy-frac", type=float, default=None,
        help="Fraction of columns assigned to heavy noise (config default: 0.22)",
    )
    noise_group.add_argument(
        "--heavy-rate", type=float, default=None,
        help="Corruption rate for heavily noisy columns",
    )
    noise_group.add_argument(
        "--severe-frac", type=float, default=None,
        help="Fraction of columns assigned to severe noise (config default: 0.10)",
    )
    noise_group.add_argument(
        "--severe-rate", type=float, default=None,
        help="Corruption rate for severely noisy columns",
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
    ar_group.add_argument(
        "--ar-max-corruption", type=float, default=None,
        help="Explicit cap on AR's total cell corruption, 0-1, as a fraction of ALL "
             "feature cells (default: none — the per-column tier budgets fix the "
             "total). When set, excess corrupted cells are restored to their clean value",
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
        help="Explicit floor on NAR's total cell corruption, 0-1, as a fraction of "
             "ALL feature cells (default: none — NAR lands on AR's total by "
             "construction). When set, untouched cells are corrupted to reach it",
    )
    nar_group.add_argument(
        "--nar-max-corruption", type=float, default=None,
        help="Explicit cap on NAR's total cell corruption, 0-1, as a fraction of "
             "ALL feature cells (default: none). When set, excess corrupted cells "
             "are restored to their clean value",
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
    _presets_cfg = _poison_cfg.get("presets", {})

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
    # Tier rates come from the preset named by poisoning.presets.default, so they
    # live in exactly one place (run_pipeline.sh passes them explicitly per
    # preset). Column shares are preset-independent and read from poisoning.noise.
    _preset_rates = {str(k): v for k, v in (_presets_cfg.get("rates") or {}).items()}
    _default_preset = str(_presets_cfg.get("default", ""))
    _tier = _preset_rates.get(_default_preset) or {}

    def _resolve_rate(cli_val, tier_name):
        if cli_val is not None:
            return cli_val
        if tier_name in _tier:
            return _tier[tier_name]
        raise SystemExit(
            f"No corruption rate for the '{tier_name}' tier: pass --{tier_name}-rate, or set "
            f"poisoning.presets.default in {args.config} to a preset defined under "
            f"poisoning.presets.rates (defined: {', '.join(sorted(_preset_rates, key=float)) or 'none'})"
        )

    mild_rate     = _resolve_rate(args.mild_rate,     "mild")
    moderate_rate = _resolve_rate(args.moderate_rate, "moderate")
    heavy_rate    = _resolve_rate(args.heavy_rate,    "heavy")
    severe_rate   = _resolve_rate(args.severe_rate,   "severe")

    moderate_frac = _resolve(args.moderate_frac, _noise_cfg, "moderate_frac", 0.35)
    heavy_frac    = _resolve(args.heavy_frac,    _noise_cfg, "heavy_frac",    0.22)
    severe_frac   = _resolve(args.severe_frac,   _noise_cfg, "severe_frac",   0.10)

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

    # None (unset here, unset in config.yaml) means "no clamp": the per-column
    # tier budgets already fix the total — see DataPoisoner.__init__.
    ar_max_corruption = _resolve(args.ar_max_corruption, _ar_cfg, "max_corruption", None)

    # ── NAR mechanisms ─────────────────────────────────────────────────────────
    nar_min_corruption = _resolve(args.nar_min_corruption, _nar_cfg, "min_corruption", None)
    nar_max_corruption = _resolve(args.nar_max_corruption, _nar_cfg, "max_corruption", None)

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
        ar_max_corruption=ar_max_corruption,
        clean_feature_frac=clean_feature_frac,
    )
