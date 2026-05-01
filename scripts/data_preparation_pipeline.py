#!/usr/bin/env python3
"""
Data Preparation Pipeline for ML Datasets
Applies cleaning and repair techniques to AR and NAR poisoned datasets, producing
cleaned versions that serve as a data-preparation baseline for comparison against
the quAIL gate-based approach.

Input:  data_poisoned/{ar,nar}/  (poisoned CSV + mask produced by poison_data.py)
Output: data_cleaned/{ar,nar}/   (cleaned CSV + residual mask + metrics)

Note: Target columns (cls_*, reg_*) are never modified.
"""
import os
import inspect
import sys
import time
import tracemalloc
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd
import psutil
from loguru import logger
from tqdm import tqdm

if not hasattr(np, "Inf"):
    np.Inf = np.inf
if not hasattr(np, "NaN"):
    np.NaN = np.nan
if not hasattr(np, "PINF"):
    np.PINF = np.inf
if not hasattr(np, "NINF"):
    np.NINF = -np.inf

try:
    import lightgbm as lgb

    if "categorical_feature" not in inspect.signature(lgb.train).parameters:
        _lgb_train = lgb.train

        def _train_compat(*args, **kwargs):
            kwargs.pop("categorical_feature", None)
            return _lgb_train(*args, **kwargs)

        lgb.train = _train_compat

    if "categorical_feature" not in inspect.signature(lgb.cv).parameters:
        _lgb_cv = lgb.cv

        def _cv_compat(*args, **kwargs):
            kwargs.pop("categorical_feature", None)
            return _lgb_cv(*args, **kwargs)

        lgb.cv = _cv_compat
except Exception:
    pass

try:
    import miceforest as mf
except ImportError as e:
    raise ImportError(
        "miceforest is required for MICE RF imputation. "
        "Install it with: pip install miceforest"
    ) from e
except OSError as e:
    if "libomp" in str(e):
        raise OSError(
            "miceforest depends on LightGBM, which on macOS requires libomp. "
            "Install it with: brew install libomp"
        ) from e
    raise

try:
    from mlxtend.frequent_patterns import apriori, association_rules
except ImportError as e:
    raise ImportError(
        "mlxtend is required for association rule mining. "
        "Install it with: pip install mlxtend"
    ) from e

logger.remove()
logger.add(
    sys.stderr,
    level="INFO",
    format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <level>{message}</level>",
)


class DataPreparation:
    """
    Implements data cleaning and repair techniques for poisoned tabular datasets.

    Cleaning is applied at the raw CSV level, before any TabularPreprocessor
    scaling or encoding takes place.  Two source modes are supported:

    - AR (At Random)   : random Gaussian noise, random label flips, MCAR
    - NAR (Not At Random): heteroscedastic noise, MNAR, systematic flips,
                           correlated noise, rare-category missingness

    Techniques:
      1. Outlier clipping    – IQR-based, clips extreme numerical values to
                               [Q1 - k*IQR, Q3 + k*IQR] (default k=1.5)
      2. Numerical imputation – replaces remaining NaN with column median
      3. Categorical imputation – replaces NaN with column mode
      4. Categorical repair   – replaces values absent from the training
                                vocabulary with the most frequent valid value

    The residual mask records cells that were originally poisoned; after
    cleaning all such cells are marked as "repaired" (False), so downstream
    quality scores reflect a fully-cleaned dataset.
    """

    def __init__(self, seed: int = 42, iqr_factor: float = 1.5):
        self.seed = seed
        self.iqr_factor = iqr_factor
        np.random.seed(seed)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def identify_column_types(self, df: pd.DataFrame) -> Dict[str, list]:
        """Identify column types based on the prefix naming convention."""
        col_types: Dict[str, list] = {
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

    def calculate_residual_metrics(self, residual_mask: pd.DataFrame) -> Dict:
        """
        Calculate cleanliness metrics on the residual mask.

        Returns:
            dict with column_cleanliness, row_cleanliness, overall_cleanliness
        """
        col_clean = (1 - residual_mask.sum(axis=0) / len(residual_mask)) * 100
        row_clean = (1 - residual_mask.sum(axis=1) / len(residual_mask.columns)) * 100
        overall_clean = (1 - residual_mask.sum().sum() / residual_mask.size) * 100
        return {
            "column_cleanliness": col_clean.to_dict(),
            "row_cleanliness": row_clean.to_dict(),
            "overall_cleanliness": overall_clean,
        }

    # ------------------------------------------------------------------
    # SHAP feature ranking
    # ------------------------------------------------------------------

    def _compute_shap_ranking(
        self, df: pd.DataFrame, col_types: Dict[str, list]
    ) -> list:
        """
        Rank features by mean absolute SHAP value using a fast RandomForest.

        A RandomForestClassifier is used when a cls_* target is available,
        otherwise a RandomForestRegressor on the first reg_* target.
        Categoricals are ordinal-encoded and NaN are temporarily filled
        (median / mode) solely for this fit — the original dataframe is
        never modified.

        Returns:
            Feature column names (numerical + categorical) sorted by
            mean |SHAP| descending.  Returns the unsorted list on failure
            (e.g. missing dependencies, no target column).
        """
        try:
            import shap
            from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
            from sklearn.preprocessing import OrdinalEncoder
        except ImportError as exc:
            logger.warning(f"  SHAP ranking skipped (missing dependency: {exc})")
            return col_types["numerical"] + col_types["categorical"]

        feature_cols = col_types["numerical"] + col_types["categorical"]
        if not feature_cols:
            return []

        # Select target (classification preferred)
        target_col = None
        is_classification = False
        if col_types["target_cls"]:
            target_col = col_types["target_cls"][0]
            is_classification = True
        elif col_types["target_reg"]:
            target_col = col_types["target_reg"][0]

        if target_col is None:
            logger.warning("  SHAP ranking skipped (no target column found)")
            return feature_cols

        X = df[feature_cols].copy()
        y = df[target_col].copy()

        # Drop rows where the target is missing
        valid_idx = y.notna()
        X, y = X[valid_idx], y[valid_idx]

        # Temporarily fill NaN for model fitting only
        for col in col_types["numerical"]:
            if col in X.columns:
                X[col] = X[col].fillna(X[col].median())
        for col in col_types["categorical"]:
            if col in X.columns:
                mode = X[col].mode()
                X[col] = X[col].fillna(mode.iloc[0] if not mode.empty else "missing")

        # Ordinal-encode categoricals
        cat_present = [c for c in col_types["categorical"] if c in X.columns]
        if cat_present:
            enc = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
            X[cat_present] = enc.fit_transform(X[cat_present].astype(str))

        # Fit a small, fast RandomForest
        rf_kwargs = dict(n_estimators=50, max_depth=6, random_state=self.seed, n_jobs=-1)
        model = (
            RandomForestClassifier(**rf_kwargs)
            if is_classification
            else RandomForestRegressor(**rf_kwargs)
        )
        model.fit(X, y)

        # Compute SHAP values on a subsample for speed
        explainer = shap.TreeExplainer(model)
        sample_size = min(500, len(X))
        X_sample = (
            X.sample(n=sample_size, random_state=self.seed) if len(X) > sample_size else X
        )
        shap_values = explainer.shap_values(X_sample)

        # SHAP can return:
        # - list[n_classes] of (n_samples, n_features)
        # - ndarray of (n_samples, n_features)
        # - ndarray of (n_samples, n_features, n_outputs)
        # Collapse every non-feature axis so we always obtain one score per feature.
        if isinstance(shap_values, list):
            shap_arr = np.stack([np.abs(np.asarray(sv)) for sv in shap_values], axis=-1)
        else:
            shap_arr = np.abs(np.asarray(shap_values))

        if shap_arr.ndim == 1:
            feature_importance = shap_arr
        else:
            feature_axes = [i for i, size in enumerate(shap_arr.shape) if size == len(feature_cols)]
            if not feature_axes:
                raise ValueError(
                    f"Unable to identify feature axis in SHAP output with shape {shap_arr.shape}"
                )
            feature_axis = feature_axes[0]
            reduce_axes = tuple(i for i in range(shap_arr.ndim) if i != feature_axis)
            feature_importance = shap_arr.mean(axis=reduce_axes)

        mean_shap = pd.Series(feature_importance, index=feature_cols)
        ranked = mean_shap.sort_values(ascending=False)

        logger.info(
            "  SHAP ranking (top 5): "
            + ", ".join(f"{c}={v:.4f}" for c, v in ranked.head(5).items())
        )
        return list(ranked.index)

    # ------------------------------------------------------------------
    # Cleaning steps
    # ------------------------------------------------------------------

    def _clip_outliers(
        self, df: pd.DataFrame, col_types: Dict[str, list]
    ) -> pd.DataFrame:
        """
        Nullify numerical values outside [Q1 - k*IQR, Q3 + k*IQR].

        Extreme values introduced by noise mechanisms (especially NAR) are
        replaced with NaN rather than clipped to the fence.
        """
        df_out = df.copy()
        n_rows = len(df_out)
        for col in col_types["numerical"]:
            series = df_out[col].dropna()
            if len(series) == 0:
                continue
            q1, q3 = series.quantile(0.25), series.quantile(0.75)
            iqr = q3 - q1
            if iqr == 0:
                continue
            lo = q1 - self.iqr_factor * iqr
            hi = q3 + self.iqr_factor * iqr

            if df_out[col].dtype in ["int64", "int32", "int16", "int8"]:
                df_out[col] = df_out[col].astype("float64")

            outlier_mask = df_out[col].notna() & (
                (df_out[col] < lo) | (df_out[col] > hi)
            )
            n_outliers = int(outlier_mask.sum())
            df_out[col] = df_out[col].where(~outlier_mask)
            logger.info(
                f"    {col}: {n_outliers} outliers ({n_outliers / n_rows * 100:.2f}% of rows)"
            )
        return df_out

    def _impute_numerical(
        self, df: pd.DataFrame, col_types: Dict[str, list]
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Fill NaN in numerical columns with the column median.

        Returns the imputed dataframe and a boolean mask marking imputed cells.
        """
        df_out = df.copy()
        imputed_mask = pd.DataFrame(False, index=df.index, columns=df.columns, dtype=bool)
        for col in col_types["numerical"]:
            nan_idx = df_out[col].isna()
            if not nan_idx.any():
                continue
            if df_out[col].dtype in ["int64", "int32", "int16", "int8"]:
                df_out[col] = df_out[col].astype("float64")
            median_val = df_out[col].median()
            df_out.loc[nan_idx, col] = median_val
            imputed_mask.loc[nan_idx, col] = True
        return df_out, imputed_mask

    def _impute_categorical(
        self, df: pd.DataFrame, col_types: Dict[str, list]
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Fill NaN in categorical columns with the column mode.

        Returns the imputed dataframe and a boolean mask marking imputed cells.
        """
        df_out = df.copy()
        imputed_mask = pd.DataFrame(False, index=df.index, columns=df.columns, dtype=bool)
        for col in col_types["categorical"]:
            nan_idx = df_out[col].isna()
            if not nan_idx.any():
                continue
            mode_vals = df_out[col].mode()
            if len(mode_vals) == 0:
                continue
            df_out.loc[nan_idx, col] = mode_vals.iloc[0]
            imputed_mask.loc[nan_idx, col] = True
        return df_out, imputed_mask

    def _repair_categorical(
        self, df: pd.DataFrame, col_types: Dict[str, list]
    ) -> pd.DataFrame:
        """
        Replace out-of-vocabulary categorical values with the most frequent
        valid value observed in the same column.

        This addresses label-flip poisoning where a value was replaced by a
        different but existing category; we cannot distinguish flipped from
        genuine values without the ground truth, so this step is a no-op for
        standard string categories.  It *does* fix numeric-coded categoricals
        that were pushed outside the valid range by noise.
        """
        df_out = df.copy()
        for col in col_types["categorical"]:
            valid_vocab = set(df_out[col].dropna().value_counts().index)
            if not valid_vocab:
                continue
            mode_val = df_out[col].mode().iloc[0]
            invalid_idx = ~df_out[col].isin(valid_vocab) & df_out[col].notna()
            if invalid_idx.any():
                df_out.loc[invalid_idx, col] = mode_val
        return df_out

    def _impute_mice(
        self,
        df: pd.DataFrame,
        selected: set,
    ) -> pd.DataFrame:
        """
        Apply MICE Random Forest imputation using miceforest.

        Columns included in the kernel:
          - ALL feature columns (num_*, cat_*) that have at least one NaN, OR
          - Feature columns that belong to the SHAP-selected set.
        Feature columns with no missing values that are outside ``selected`` are excluded.
        Target columns (cls_*, reg_*) are never touched.
        """
        all_feature_cols = [c for c in df.columns if c.startswith("num_") or c.startswith("cat_")]
        cols_with_nan = {c for c in all_feature_cols if df[c].isna().any()}
        mice_cols = [c for c in all_feature_cols if c in selected or c in cols_with_nan]

        # miceforest requires at least one observed value per column to train a predictor;
        # columns that are entirely NaN would cause numpy.random.choice(0) to raise.
        mice_cols = [c for c in mice_cols if df[c].notna().any()]

        # LightGBM multiclass requires >= 2 unique classes; categorical columns with only
        # one unique non-null value cannot be modelled — fill them directly instead.
        trivial_cat_cols = [
            c for c in mice_cols
            if c.startswith("cat_") and df[c].nunique(dropna=True) < 2 and df[c].isna().any()
        ]
        df_out = df.copy()
        for col in trivial_cat_cols:
            fill_val = df_out[col].dropna().iloc[0]
            df_out[col] = df_out[col].fillna(fill_val)
        if trivial_cat_cols:
            logger.info(
                f"  Trivial-fill (1 unique value) categorical columns: {trivial_cat_cols}"
            )
        mice_cols = [c for c in mice_cols if c not in trivial_cat_cols]

        if not mice_cols or not any(df_out[c].isna().any() for c in mice_cols):
            return df_out

        df_mice = df_out[mice_cols].copy()

        for col in df_mice.select_dtypes(include=["object", "bool"]).columns:
            df_mice[col] = df_mice[col].astype("category")

        try:
            kernel = mf.ImputationKernel(
                data=df_mice,
                random_state=self.seed,
            )

            # Total number of MICE cycles. More cycles improve convergence but cost time.
            n_mice_iterations = 3
            # Number of LightGBM trees trained per imputed column per iteration.
            # This is the single biggest runtime lever: default is 100.
            # Use 10-20 for quick experiments, 50-100 for final runs.
            n_estimators = 10
            # Maximum number of leaves per tree. Controls model complexity:
            # fewer leaves = shallower trees = faster training. Default is 31.
            num_leaves = 20
            logger.info(
                f"Running MICE ({n_mice_iterations} iter, "
                f"n_estimators={n_estimators}, num_leaves={num_leaves})..."
            )
            for _ in tqdm(range(n_mice_iterations), desc="MICE imputation", unit="iter"):
                kernel.mice(1, n_estimators=n_estimators, num_leaves=num_leaves)
            df_imputed = kernel.complete_data()
            logger.info("Imputation Completed!")

            for col in df_imputed.select_dtypes(include=["category"]).columns:
                df_imputed[col] = df_imputed[col].astype("object")

            df_out[mice_cols] = df_imputed.values
        except Exception as exc:
            raise RuntimeError(f"MICE imputation failed: {exc}") from exc

        return df_out

    def _detect_categorical_outliers_ar(
        self,
        df: pd.DataFrame,
        col_types: Dict[str, list],
        selected: set,
        min_support: float = 0.8,
        min_confidence: float = 0.8,
    ) -> pd.DataFrame:
        """
        Detect and nullify categorical cells that violate pairwise (1→1)
        association rules mined from the dataset.

        Rules are mined with the Apriori algorithm (mlxtend) from complete rows
        only.  For each rule X=x → Y=y, any row where X=x but Y≠y has Y set to
        NaN so that the subsequent MICE re-imputation step can fill it correctly.
        Only categorical features present in ``selected`` are considered.
        """
        cat_cols = [c for c in col_types["categorical"] if c in selected]
        if len(cat_cols) < 2:
            return df

        df_out = df.copy()
        df_cat = df_out[cat_cols].copy()

        # Mine rules from rows that are complete across all selected categorical cols
        complete_rows = df_cat.notna().all(axis=1)
        df_complete = df_cat[complete_rows].astype(str)

        if len(df_complete) < 10:
            logger.warning(
                "  Too few complete rows for association rule mining; skipping"
            )
            return df_out

        # One-hot encode as "col=value" items for mlxtend (prefix_sep="=")
        df_ohe = pd.get_dummies(df_complete, prefix_sep="=").astype(bool)

        try:
            frequent_itemsets = apriori(
                df_ohe, min_support=min_support, use_colnames=True
            )
        except Exception as exc:
            logger.warning(f"  Apriori failed ({exc}); skipping AR outlier detection")
            return df_out

        if frequent_itemsets.empty:
            logger.info(
                f"  No frequent itemsets at support≥{min_support}; "
                "skipping AR outlier detection"
            )
            return df_out

        rules_df = association_rules(
            frequent_itemsets, metric="confidence", min_threshold=min_confidence
        )

        logger.info(
            f"  {len(rules_df)} association rules detected "
            f"(support≥{min_support}, confidence≥{min_confidence})"
        )

        # Keep only pairwise 1-antecedent → 1-consequent rules
        rules_df = rules_df[
            (rules_df["antecedents"].apply(len) == 1)
            & (rules_df["consequents"].apply(len) == 1)
        ]

        if rules_df.empty:
            logger.info("  No pairwise association rules found")
            return df_out

        logger.info(f"  {len(rules_df)} pairwise (1→1) rules retained")

        nullified = 0
        violating_rows = pd.Series(False, index=df_out.index)
        for _, rule in rules_df.iterrows():
            ant_item = next(iter(rule["antecedents"]))
            con_item = next(iter(rule["consequents"]))

            ant_col, ant_val = ant_item.split("=", 1)
            con_col, con_val = con_item.split("=", 1)

            if ant_col not in df_out.columns or con_col not in df_out.columns:
                continue

            violating = (
                df_out[ant_col].notna()
                & (df_out[ant_col].astype(str) == ant_val)
                & df_out[con_col].notna()
                & (df_out[con_col].astype(str) != con_val)
            )
            if violating.any():
                violating_rows |= violating
                df_out.loc[violating, con_col] = np.nan
                nullified += int(violating.sum())

        n_rows = len(df_out)
        n_violating_rows = int(violating_rows.sum())
        logger.info(
            f"  Nullified {nullified} cells violating association rules; "
            f"{n_violating_rows} / {n_rows} rows affected "
            f"({n_violating_rows / n_rows * 100:.2f}% of rows)"
        )
        return df_out

    # ------------------------------------------------------------------
    # Main public method
    # ------------------------------------------------------------------

    def prepare(
        self,
        df_poisoned: pd.DataFrame,
        mask_df: pd.DataFrame,
        shap_top_pct: float = 1.0,
        ar_min_support: float = 0.8,
        ar_min_confidence: float = 0.8,
    ) -> Tuple[pd.DataFrame, pd.DataFrame, Dict]:
        """
        Apply the full data preparation pipeline to a poisoned dataset.

        Steps (in order):
          0. SHAP feature ranking   – ranks features by mean |SHAP| and restricts
                                      subsequent cleaning to the top ``shap_top_pct``
                                      fraction (1.0 = all features, default)
          1. MICE RF imputation     – fills existing NaN in the dirty input data
                                      using miceforest; kernel restricted to
                                      selected features ∪ columns with NaN
          2. IQR outlier clipping   – nullifies numerical values outside
                                      [Q1 - k*IQR, Q3 + k*IQR]
          3. AR categorical repair  – mines pairwise (1→1) association rules from
                                      the data and nullifies cells that violate them
          4. MICE RF re-imputation  – fills NaN introduced by steps 2 and 3
          5. Categorical OOV repair – replaces out-of-vocabulary categorical values
                                      with the column mode

        Target columns (cls_*, reg_*) are never touched.

        Args:
            df_poisoned       : Poisoned dataframe loaded from data_poisoned/{mode}/
            mask_df           : Boolean poison mask (True = originally poisoned cell)
            shap_top_pct      : Fraction (0.0–1.0] of top-ranked SHAP features to
                                include in all cleaning steps.  1.0 keeps all
                                features (default, preserves original behaviour).
            ar_min_support    : Minimum support threshold for association rule
                                mining in step 3 (default 0.8).
            ar_min_confidence : Minimum confidence threshold for association rule
                                mining in step 3 (default 0.8).

        Returns:
            df_cleaned    : Cleaned dataframe
            residual_mask : Boolean mask — True for cells that were originally
                            poisoned but could not be fully resolved
            perf_metrics  : Dict with wall_time_s, cpu_time_s, ram_before_mb,
                            ram_after_mb, ram_peak_mb, throughput_rows_per_s, kwh
        """
        logger.info(
            "  Cleaning steps: mice_imputation, outlier_clipping, "
            "categorical_ar_repair, mice_re_imputation, categorical_oov_repair"
        )

        proc = psutil.Process()
        ram_before_mb = proc.memory_info().rss / 1024 / 1024
        tracemalloc.start()
        wall_t0 = time.perf_counter()
        cpu_t0 = time.process_time()

        col_types = self.identify_column_types(df_poisoned)
        df_clean = df_poisoned.copy()

        # Step 0: SHAP feature ranking
        ranked_features = self._compute_shap_ranking(df_poisoned, col_types)
        # Always define selected; when shap_top_pct==1.0, all features are kept
        selected = (
            set(ranked_features)
            if ranked_features
            else set(col_types["numerical"] + col_types["categorical"])
        )
        if ranked_features and shap_top_pct < 1.0:
            n_keep = max(1, int(np.ceil(len(ranked_features) * shap_top_pct)))
            selected = set(ranked_features[:n_keep])
            logger.info(
                f"  SHAP top {shap_top_pct * 100:.0f}%: "
                f"retaining {n_keep}/{len(ranked_features)} features for cleaning"
            )
            col_types = {
                k: [c for c in v if c in selected]
                if k in ("numerical", "categorical")
                else v
                for k, v in col_types.items()
            }

        # Step 1: MICE RF imputation on missing values of the dirty input data
        logger.info("  Step 1: MICE RF imputation")
        df_clean = self._impute_mice(df_clean, selected)

        # Step 2: Outlier detection with IQR for numerical features
        logger.info("  Step 2: IQR outlier clipping")
        df_clean = self._clip_outliers(df_clean, col_types)

        # Step 3: Outlier detection with pairwise association rules for categorical features
        logger.info("  Step 3: Categorical AR outlier detection")
        df_clean = self._detect_categorical_outliers_ar(
            df_clean, col_types, selected, ar_min_support, ar_min_confidence
        )

        # Step 4: Reapply MICE RF to fill NaN introduced by steps 2 and 3
        logger.info("  Step 4: MICE RF re-imputation")
        df_clean = self._impute_mice(df_clean, selected)

        # Step 5: Replace any remaining out-of-vocabulary categorical values with mode
        df_clean = self._repair_categorical(df_clean, col_types)

        # Residual mask: cells that were poisoned AND still contain NaN after cleaning
        remaining_nan = df_clean.isna()
        residual_mask = mask_df & remaining_nan

        repaired = mask_df.sum().sum() - residual_mask.sum().sum()
        logger.info(
            f"  Repaired {repaired} / {mask_df.sum().sum()} poisoned cells "
            f"({residual_mask.sum().sum()} residual)"
        )

        wall_time_s = time.perf_counter() - wall_t0
        cpu_time_s = time.process_time() - cpu_t0
        _, peak_bytes = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        ram_after_mb = proc.memory_info().rss / 1024 / 1024
        ram_peak_mb = peak_bytes / 1024 / 1024

        n_rows = len(df_poisoned)
        perf_metrics = {
            "n_rows": n_rows,
            "n_cols": len(df_poisoned.columns),
            "wall_time_s": round(wall_time_s, 4),
            "cpu_time_s": round(cpu_time_s, 4),
            "ram_before_mb": round(ram_before_mb, 3),
            "ram_after_mb": round(ram_after_mb, 3),
            "ram_peak_mb": round(ram_peak_mb, 3),
            "throughput_rows_per_s": round(n_rows / wall_time_s, 4) if wall_time_s > 0 else float("inf"),
        }

        return df_clean, residual_mask, perf_metrics


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def check_dataset_complete(csv_file: Path, output_dir: str) -> bool:
    """
    Check whether all output files already exist for a given dataset.

    Args:
        csv_file   : Path to the poisoned CSV file (used to derive output names)
        output_dir : Root output directory (e.g. data_cleaned)

    Returns:
        True if all expected output files are present, False otherwise
    """
    required_files = [
        os.path.join(output_dir, "ar", csv_file.name),
        os.path.join(output_dir, "ar", csv_file.stem + "_mask.csv"),
        os.path.join(output_dir, "metrics", csv_file.stem + "_ar_metrics.csv"),
        os.path.join(output_dir, "metrics", csv_file.stem + "_ar_perf_metrics.csv"),
        os.path.join(output_dir, "nar", csv_file.name),
        os.path.join(output_dir, "nar", csv_file.stem + "_mask.csv"),
        os.path.join(output_dir, "metrics", csv_file.stem + "_nar_metrics.csv"),
        os.path.join(output_dir, "metrics", csv_file.stem + "_nar_perf_metrics.csv"),
    ]
    return all(os.path.exists(f) for f in required_files)


def process_all_datasets(
    input_dir: str,
    output_dir: str,
    dataset_name: str = None,
    iqr_factor: float = 1.5,
    shap_top_pct: float = 1.0,
    ar_min_support: float = 0.8,
    ar_min_confidence: float = 0.8,
):
    """
    Apply the data preparation pipeline to all poisoned datasets.

    Reads poisoned CSVs and their masks from ``input_dir/{ar,nar}/`` and
    writes cleaned CSVs, residual masks, and cleanliness metrics to
    ``output_dir/{ar,nar}/``.

    Args:
        input_dir       : Root poisoned-data directory (default: data_poisoned)
        output_dir      : Root output directory for cleaned data (default: data_cleaned)
        dataset_name    : If given, process only this dataset (name without .csv suffix,
                          matched against the part after the 8-char prefix, e.g. "iris")
        iqr_factor      : Whisker multiplier for IQR-based outlier clipping (default 1.5)
        shap_top_pct    : Fraction (0.0–1.0] of top-ranked SHAP features passed to
                          cleaning steps (1.0 = all features, default)
        ar_min_support  : Minimum support threshold for association rule mining (default 0.8)
        ar_min_confidence: Minimum confidence threshold for association rule mining (default 0.8)
    """
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.join(output_dir, "ar"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "nar"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "metrics"), exist_ok=True)

    preparer = DataPreparation(seed=42, iqr_factor=iqr_factor)

    # Collect CSV files from the AR poisoned directory
    # (AR and NAR always share the same filenames; we iterate once and handle both)
    ar_dir = Path(input_dir) / "ar"
    nar_dir = Path(input_dir) / "nar"

    if not ar_dir.exists():
        logger.error(f"AR poisoned directory not found: {ar_dir}")
        return
    if not nar_dir.exists():
        logger.error(f"NAR poisoned directory not found: {nar_dir}")
        return

    csv_files = sorted(ar_dir.glob("*.csv"))
    # Exclude mask files
    csv_files = [f for f in csv_files if not f.stem.endswith("_mask")]

    if not csv_files:
        logger.error(f"No poisoned CSV files found in {ar_dir}")
        return

    if dataset_name:
        # Match against the dataset name part (after 8-char prefix "XXX_YYY_")
        csv_files = [f for f in csv_files if f.stem[8:] == dataset_name]
        if not csv_files:
            logger.error(f"Dataset '{dataset_name}' not found in {ar_dir}")
            return
        logger.info(f"Processing specific dataset: {dataset_name}")
    else:
        logger.info(f"Found {len(csv_files)} poisoned datasets to prepare")

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
            # --- AR mode ---
            ar_csv = ar_dir / csv_file.name
            ar_mask_file = ar_dir / f"{csv_file.stem}_mask.csv"
            ar_out_files = [
                os.path.join(output_dir, "ar", csv_file.name),
                os.path.join(output_dir, "ar", csv_file.stem + "_mask.csv"),
                os.path.join(output_dir, "metrics", csv_file.stem + "_ar_metrics.csv"),
                os.path.join(output_dir, "metrics", csv_file.stem + "_ar_perf_metrics.csv"),
            ]

            if all(os.path.exists(f) for f in ar_out_files):
                logger.info(f"  AR already processed, skipping: {csv_file.name}")
            elif not ar_csv.exists():
                logger.warning(f"  AR CSV not found, skipping: {ar_csv}")
            elif not ar_mask_file.exists():
                logger.warning(f"  AR mask not found, skipping: {ar_mask_file}")
            else:
                df_ar = pd.read_csv(
                    ar_csv, na_values=["?", "NA", "N/A", "NaN", "nan", "NAN", "", " "]
                )
                mask_ar = pd.read_csv(ar_mask_file).astype(bool)
                logger.info(f"  AR loaded: {df_ar.shape[0]} rows × {df_ar.shape[1]} columns")

                df_ar_clean, residual_ar, perf_ar = preparer.prepare(df_ar, mask_ar, shap_top_pct=shap_top_pct, ar_min_support=ar_min_support, ar_min_confidence=ar_min_confidence)
                metrics_ar = preparer.calculate_residual_metrics(residual_ar)

                df_ar_clean.to_csv(os.path.join(output_dir, "ar", csv_file.name), index=False)
                residual_ar.to_csv(
                    os.path.join(output_dir, "ar", csv_file.stem + "_mask.csv"), index=False
                )
                pd.DataFrame(
                    {
                        "column": list(metrics_ar["column_cleanliness"].keys()),
                        "cleanliness_pct": list(metrics_ar["column_cleanliness"].values()),
                    }
                ).to_csv(
                    os.path.join(output_dir, "metrics", csv_file.stem + "_ar_metrics.csv"),
                    index=False,
                )
                pd.DataFrame([{"dataset": csv_file.stem, "corruption": "ar", **perf_ar}]).to_csv(
                    os.path.join(output_dir, "metrics", csv_file.stem + "_ar_perf_metrics.csv"),
                    index=False,
                )
                logger.info(
                    f"  AR cleaned: {metrics_ar['overall_cleanliness']:.2f}% clean overall "
                    f"| {perf_ar['wall_time_s']:.1f}s wall | {perf_ar['cpu_time_s']:.1f}s CPU "
                    f"| {perf_ar['ram_peak_mb']:.1f} MB peak"
                )

            # --- NAR mode ---
            nar_csv = nar_dir / csv_file.name
            nar_mask_file = nar_dir / f"{csv_file.stem}_mask.csv"
            nar_out_files = [
                os.path.join(output_dir, "nar", csv_file.name),
                os.path.join(output_dir, "nar", csv_file.stem + "_mask.csv"),
                os.path.join(output_dir, "metrics", csv_file.stem + "_nar_metrics.csv"),
                os.path.join(output_dir, "metrics", csv_file.stem + "_nar_perf_metrics.csv"),
            ]

            if all(os.path.exists(f) for f in nar_out_files):
                logger.info(f"  NAR already processed, skipping: {csv_file.name}")
            elif not nar_csv.exists():
                logger.warning(f"  NAR CSV not found, skipping: {nar_csv}")
            elif not nar_mask_file.exists():
                logger.warning(f"  NAR mask not found, skipping: {nar_mask_file}")
            else:
                df_nar = pd.read_csv(
                    nar_csv, na_values=["?", "NA", "N/A", "NaN", "nan", "NAN", "", " "]
                )
                mask_nar = pd.read_csv(nar_mask_file).astype(bool)
                logger.info(f"  NAR loaded: {df_nar.shape[0]} rows × {df_nar.shape[1]} columns")

                df_nar_clean, residual_nar, perf_nar = preparer.prepare(df_nar, mask_nar, shap_top_pct=shap_top_pct, ar_min_support=ar_min_support, ar_min_confidence=ar_min_confidence)
                metrics_nar = preparer.calculate_residual_metrics(residual_nar)

                df_nar_clean.to_csv(os.path.join(output_dir, "nar", csv_file.name), index=False)
                residual_nar.to_csv(
                    os.path.join(output_dir, "nar", csv_file.stem + "_mask.csv"), index=False
                )
                pd.DataFrame(
                    {
                        "column": list(metrics_nar["column_cleanliness"].keys()),
                        "cleanliness_pct": list(metrics_nar["column_cleanliness"].values()),
                    }
                ).to_csv(
                    os.path.join(output_dir, "metrics", csv_file.stem + "_nar_metrics.csv"),
                    index=False,
                )
                pd.DataFrame([{"dataset": csv_file.stem, "corruption": "nar", **perf_nar}]).to_csv(
                    os.path.join(output_dir, "metrics", csv_file.stem + "_nar_perf_metrics.csv"),
                    index=False,
                )
                logger.info(
                    f"  NAR cleaned: {metrics_nar['overall_cleanliness']:.2f}% clean overall "
                    f"| {perf_nar['wall_time_s']:.1f}s wall | {perf_nar['cpu_time_s']:.1f}s CPU "
                    f"| {perf_nar['ram_peak_mb']:.1f} MB peak"
                )

            logger.success(f"  Completed {csv_file.name}")

        except Exception as e:
            logger.error(f"  Error processing {csv_file.name}: {e}")
            continue

    logger.success(f"All datasets prepared! Output in {output_dir}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Apply data preparation / cleaning to AR and NAR poisoned datasets",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
                Examples:
                python data_preparation_pipeline.py
                python data_preparation_pipeline.py --input_dir data_poisoned --output_dir data_cleaned
                python data_preparation_pipeline.py --dataset iris
                python data_preparation_pipeline.py --iqr-factor 3.0

                Cleaning Steps:
                mice_imputation         : fills NaN with MICE Random Forest (miceforest) – step 1
                outlier_clipping        : nullifies numerical values outside [Q1 - k*IQR, Q3 + k*IQR] – step 2
                categorical_ar_repair   : nullifies categorical cells violating pairwise association rules – step 3
                mice_re_imputation      : refills NaN introduced by steps 2 & 3 with MICE RF – step 4
                categorical_oov_repair  : replaces out-of-vocabulary categorical values with the column mode – step 5

                Note:
                Target columns (cls_*, reg_*) are NEVER modified.
                Input  : data_poisoned/{ar,nar}/<dataset>.csv  +  <dataset>_mask.csv
                Output : data_cleaned/{ar,nar}/<dataset>.csv   +  <dataset>_mask.csv  +  metrics
            """,
    )

    parser.add_argument(
        "--input_dir",
        type=str,
        default="data_poisoned",
        help="Root directory of poisoned data (default: data_poisoned)",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="data_cleaned_cp",
        help="Root directory for cleaned output (default: data_cleaned)",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default=None,
        help="Process only this dataset (name without .csv, e.g. 'iris')",
    )
    parser.add_argument(
        "--iqr-factor",
        type=float,
        default=2.5,
        help="IQR whisker multiplier for outlier clipping (default: 1.5)",
    )
    parser.add_argument(
        "--shap-top-pct",
        type=float,
        default=1.0,
        help=(
            "Fraction (0.0–1.0] of top SHAP-ranked features to include in "
            "cleaning steps. 1.0 keeps all features (default)."
        ),
    )

    ar_group = parser.add_argument_group("Association Rule Mining")
    ar_group.add_argument(
        "--ar-min-support",
        type=float,
        default=0.8,
        help="Minimum support for association rule mining in step 3 (default: 0.8)",
    )
    ar_group.add_argument(
        "--ar-min-confidence",
        type=float,
        default=0.8,
        help="Minimum confidence for association rule mining in step 3 (default: 0.8)",
    )

    args = parser.parse_args()

    process_all_datasets(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        dataset_name=args.dataset,
        iqr_factor=args.iqr_factor,
        shap_top_pct=args.shap_top_pct,
        ar_min_support=args.ar_min_support,
        ar_min_confidence=args.ar_min_confidence,
    )
