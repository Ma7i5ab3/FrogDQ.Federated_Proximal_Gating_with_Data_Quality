#!/usr/bin/env python3
"""
Saga++ Data Cleaning Pipeline Baseline
Python reimplementation of the Saga++ framework from:

  Siddiqi et al. "Saga++: A Scalable Framework for Optimizing Data Cleaning
  Pipelines for Machine Learning Applications." ACM TODS 51(2), 2026.

The original Saga++ is built on Apache SystemDS (JVM). This implementation
faithfully reproduces the three-function API and core algorithms in pure Python:

  topk_cleaning()  — evolutionary algorithm + Hyperband to find top-K pipelines
  fit_pipeline()   — fit a selected pipeline on training data, return state
  apply_pipeline() — transform new data using a fitted pipeline state

Cleaning primitives (Table 1 from the paper):
  Outliers      : outlierByIQR, outlierBySd, winsorize
  MV Imputation : imputeByMean, imputeByMedian, fillForward, fillDefault
  Data Prep     : normalize
  Class Imbal.  : underSampling
  Labels        : abstain
  String (stage0): correctTypos

Input:  data_poisoned/{ar,nar}/  (poisoned CSV + mask produced by poison_data.py)
Output: data_cleaned_saga/{ar,nar}/  (cleaned CSV + residual mask + metrics)

Note: Target columns (cls_*, reg_*) are never modified.
"""

import os
import sys
import time
import tracemalloc
import copy
import random
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import psutil
from joblib import Parallel, delayed
from loguru import logger

warnings.filterwarnings("ignore")

if not hasattr(np, "Inf"):
    np.Inf = np.inf
if not hasattr(np, "NaN"):
    np.NaN = np.nan

logger.remove()
logger.add(
    sys.stderr,
    level="INFO",
    format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <level>{message}</level>",
)

# ---------------------------------------------------------------------------
# Column-type identification (mirrors data_preparation_pipeline.py)
# ---------------------------------------------------------------------------

def _identify_col_types(df: pd.DataFrame) -> Dict[str, list]:
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


# ---------------------------------------------------------------------------
# Primitive catalogue
# ---------------------------------------------------------------------------
# Every primitive exposes two functions:
#   fit(X, y, col_types, **params) -> state dict
#   apply(X, col_types, state)     -> pd.DataFrame (X_cleaned)
#
# "train_only" primitives (abstain, underSampling) have no-op apply() and
# operate by returning modified (X, y) pairs during fit via the key
# "_train_rows" in state (list of index values to keep).

PRIMITIVE_DEFAULTS: Dict[str, Dict] = {
    "outlierByIQR":  {"k": 1.5},
    "outlierBySd":   {"k": 3.0},
    "winsorize":     {"q_l": 0.05, "q_u": 0.95},
    "imputeByMean":  {},
    "imputeByMedian":{},
    "fillForward":   {},
    "fillDefault":   {"v": 0},
    "normalize":     {},
    "underSampling": {"N": 0.5},
    "abstain":       {"P": 0.9},
    "correctTypos":  {"min_count": 2},
}

# Hyperparameter grids for physical pipeline tuning (Hyperband)
PARAM_GRIDS: Dict[str, List[Dict]] = {
    "outlierByIQR":   [{"k": k} for k in [1.0, 1.5, 2.0, 2.5, 3.0]],
    "outlierBySd":    [{"k": k} for k in [1.0, 1.5, 2.0, 2.5, 3.0]],
    "winsorize":      [
        {"q_l": 0.01, "q_u": 0.99},
        {"q_l": 0.05, "q_u": 0.95},
        {"q_l": 0.10, "q_u": 0.90},
    ],
    "imputeByMean":   [{}],
    "imputeByMedian": [{}],
    "fillForward":    [{}],
    "fillDefault":    [{"v": 0}, {"v": "mean"}, {"v": "median"}],
    "normalize":      [{}],
    "underSampling":  [{"N": 0.3}, {"N": 0.5}, {"N": 0.7}],
    "abstain":        [{"P": 0.80}, {"P": 0.90}, {"P": 0.95}],
    "correctTypos":   [{"min_count": 1}, {"min_count": 2}, {"min_count": 5}],
}

# Which primitives are monotonic (used for pruning, per paper Table 1)
MONOTONIC_PRIMITIVES = {
    "outlierByIQR", "outlierBySd", "winsorize",
    "imputeByFd", "flipLabels", "abstain",
    "fixLength",
}

# Stage-0 primitives (applied first, before numerical cleaning)
STAGE0_PRIMITIVES = {"correctTypos"}

# Train-only primitives (apply() is a no-op)
TRAIN_ONLY_PRIMITIVES = {"underSampling", "abstain"}

# Outlier-detection primitives (skipped when applying pipeline to test data)
OUTLIER_PRIMITIVES = {"outlierByIQR", "outlierBySd", "winsorize"}

ALL_PRIMITIVES = list(PRIMITIVE_DEFAULTS.keys())


# ---------------------------------------------------------------------------
# Primitive implementations
# ---------------------------------------------------------------------------

def _outlierByIQR_fit(X: pd.DataFrame, y, col_types: Dict, k: float = 1.5) -> Dict:
    state: Dict = {}
    for col in col_types["numerical"]:
        if col not in X.columns:
            continue
        s = X[col].dropna()
        if len(s) < 4:
            continue
        q1, q3 = float(s.quantile(0.25)), float(s.quantile(0.75))
        iqr = q3 - q1
        if iqr > 0:
            state[col] = {"lo": q1 - k * iqr, "hi": q3 + k * iqr}
    return state


def _outlierByIQR_apply(X: pd.DataFrame, col_types: Dict, state: Dict) -> pd.DataFrame:
    X_out = X.copy()
    for col, bounds in state.items():
        if col not in X_out.columns:
            continue
        X_out[col] = X_out[col].astype(float, errors="ignore")
        mask = X_out[col].notna() & (
            (X_out[col] < bounds["lo"]) | (X_out[col] > bounds["hi"])
        )
        X_out.loc[mask, col] = np.nan
    return X_out


def _outlierBySd_fit(X: pd.DataFrame, y, col_types: Dict, k: float = 3.0) -> Dict:
    state: Dict = {}
    for col in col_types["numerical"]:
        if col not in X.columns:
            continue
        s = X[col].dropna()
        if len(s) < 4:
            continue
        mean, std = float(s.mean()), float(s.std())
        if std > 0:
            state[col] = {"lo": mean - k * std, "hi": mean + k * std}
    return state


def _outlierBySd_apply(X: pd.DataFrame, col_types: Dict, state: Dict) -> pd.DataFrame:
    X_out = X.copy()
    for col, bounds in state.items():
        if col not in X_out.columns:
            continue
        X_out[col] = X_out[col].astype(float, errors="ignore")
        mask = X_out[col].notna() & (
            (X_out[col] < bounds["lo"]) | (X_out[col] > bounds["hi"])
        )
        X_out.loc[mask, col] = np.nan
    return X_out


def _winsorize_fit(
    X: pd.DataFrame, y, col_types: Dict, q_l: float = 0.05, q_u: float = 0.95
) -> Dict:
    state: Dict = {}
    for col in col_types["numerical"]:
        if col not in X.columns:
            continue
        s = X[col].dropna()
        if len(s) < 4:
            continue
        state[col] = {"lo": float(s.quantile(q_l)), "hi": float(s.quantile(q_u))}
    return state


def _winsorize_apply(X: pd.DataFrame, col_types: Dict, state: Dict) -> pd.DataFrame:
    X_out = X.copy()
    for col, bounds in state.items():
        if col not in X_out.columns:
            continue
        X_out[col] = X_out[col].astype(float, errors="ignore")
        X_out[col] = X_out[col].clip(lower=bounds["lo"], upper=bounds["hi"])
    return X_out


def _imputeByMean_fit(X: pd.DataFrame, y, col_types: Dict) -> Dict:
    return {
        col: float(X[col].mean())
        for col in col_types["numerical"]
        if col in X.columns and X[col].notna().any()
    }


def _imputeByMean_apply(X: pd.DataFrame, col_types: Dict, state: Dict) -> pd.DataFrame:
    X_out = X.copy()
    for col, val in state.items():
        if col not in X_out.columns:
            continue
        X_out[col] = X_out[col].astype(float, errors="ignore")
        X_out[col] = X_out[col].fillna(val)
    return X_out


def _imputeByMedian_fit(X: pd.DataFrame, y, col_types: Dict) -> Dict:
    return {
        col: float(X[col].median())
        for col in col_types["numerical"]
        if col in X.columns and X[col].notna().any()
    }


def _imputeByMedian_apply(X: pd.DataFrame, col_types: Dict, state: Dict) -> pd.DataFrame:
    X_out = X.copy()
    for col, val in state.items():
        if col not in X_out.columns:
            continue
        X_out[col] = X_out[col].astype(float, errors="ignore")
        X_out[col] = X_out[col].fillna(val)
    return X_out


def _fillForward_fit(X: pd.DataFrame, y, col_types: Dict) -> Dict:
    # State stores last valid value per column (for applying to unseen data)
    state: Dict = {}
    for col in col_types["numerical"] + col_types["categorical"]:
        if col not in X.columns:
            continue
        last_valid = X[col].dropna()
        state[col] = last_valid.iloc[-1] if len(last_valid) > 0 else np.nan
    return state


def _fillForward_apply(X: pd.DataFrame, col_types: Dict, state: Dict) -> pd.DataFrame:
    X_out = X.copy()
    for col in col_types["numerical"] + col_types["categorical"]:
        if col not in X_out.columns:
            continue
        X_out[col] = X_out[col].ffill().bfill()
        # Any remaining NaN (e.g., all-NaN column) get the training last-value
        if X_out[col].isna().any() and col in state and not pd.isna(state[col]):
            X_out[col] = X_out[col].fillna(state[col])
    return X_out


def _fillDefault_fit(X: pd.DataFrame, y, col_types: Dict, v: Any = 0) -> Dict:
    state: Dict = {"v": v, "num_fills": {}, "cat_fills": {}}
    for col in col_types["numerical"]:
        if col not in X.columns:
            continue
        if v == "mean":
            state["num_fills"][col] = float(X[col].mean()) if X[col].notna().any() else 0.0
        elif v == "median":
            state["num_fills"][col] = float(X[col].median()) if X[col].notna().any() else 0.0
        else:
            state["num_fills"][col] = float(v)
    for col in col_types["categorical"]:
        if col not in X.columns:
            continue
        mode = X[col].mode()
        state["cat_fills"][col] = mode.iloc[0] if len(mode) > 0 else "unknown"
    return state


def _fillDefault_apply(X: pd.DataFrame, col_types: Dict, state: Dict) -> pd.DataFrame:
    X_out = X.copy()
    for col, val in state.get("num_fills", {}).items():
        if col not in X_out.columns:
            continue
        X_out[col] = X_out[col].astype(float, errors="ignore").fillna(val)
    for col, val in state.get("cat_fills", {}).items():
        if col not in X_out.columns:
            continue
        X_out[col] = X_out[col].fillna(val)
    return X_out


def _normalize_fit(X: pd.DataFrame, y, col_types: Dict) -> Dict:
    state: Dict = {}
    for col in col_types["numerical"]:
        if col not in X.columns:
            continue
        s = X[col].dropna()
        mean, std = float(s.mean()) if len(s) > 0 else 0.0, float(s.std()) if len(s) > 1 else 1.0
        state[col] = {"mean": mean, "std": std if std > 0 else 1.0}
    return state


def _normalize_apply(X: pd.DataFrame, col_types: Dict, state: Dict) -> pd.DataFrame:
    X_out = X.copy()
    for col, stats in state.items():
        if col not in X_out.columns:
            continue
        X_out[col] = X_out[col].astype(float, errors="ignore")
        X_out[col] = (X_out[col] - stats["mean"]) / stats["std"]
    return X_out


def _underSampling_fit(X: pd.DataFrame, y, col_types: Dict, N: float = 0.5) -> Dict:
    """Under-sample majority class to ratio N (minority/majority)."""
    if y is None:
        return {"_train_rows": list(X.index)}
    y_s = pd.Series(y.values, index=X.index) if not isinstance(y, pd.Series) else y
    counts = y_s.value_counts()
    if len(counts) < 2:
        return {"_train_rows": list(X.index)}
    minority_cls = counts.index[-1]
    majority_cls = counts.index[0]
    n_min = int(counts[minority_cls])
    n_maj_target = min(int(counts[majority_cls]), max(n_min, int(n_min / N)))
    rng = np.random.RandomState(42)
    maj_idx = y_s[y_s == majority_cls].index.tolist()
    kept_maj = rng.choice(maj_idx, size=min(n_maj_target, len(maj_idx)), replace=False).tolist()
    other_idx = y_s[~y_s.isin([minority_cls, majority_cls])].index.tolist()
    min_idx = y_s[y_s == minority_cls].index.tolist()
    return {"_train_rows": sorted(min_idx + kept_maj + other_idx)}


def _underSampling_apply(X: pd.DataFrame, col_types: Dict, state: Dict) -> pd.DataFrame:
    return X  # no-op at inference time


def _abstain_fit(X: pd.DataFrame, y, col_types: Dict, P: float = 0.9) -> Dict:
    """Remove training rows where a quick logistic model mispredicts with confidence >= P."""
    if y is None or not col_types.get("target_cls"):
        return {"_train_rows": list(X.index)}
    try:
        from sklearn.linear_model import LogisticRegression
        from sklearn.preprocessing import OrdinalEncoder

        feat_cols = [c for c in col_types["numerical"] + col_types["categorical"] if c in X.columns]
        if not feat_cols:
            return {"_train_rows": list(X.index)}

        X_enc = X[feat_cols].copy()
        cat_present = [c for c in col_types["categorical"] if c in X_enc.columns]
        if cat_present:
            enc = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
            X_enc[cat_present] = enc.fit_transform(X_enc[cat_present].astype(str))
        for col in X_enc.columns:
            X_enc[col] = pd.to_numeric(X_enc[col], errors="coerce")
            X_enc[col] = X_enc[col].fillna(X_enc[col].median() if X_enc[col].notna().any() else 0.0)

        y_s = pd.Series(y.values, index=X.index) if not isinstance(y, pd.Series) else y
        valid = X_enc.notna().all(axis=1) & y_s.notna()
        if valid.sum() < 10:
            return {"_train_rows": list(X.index)}

        model = LogisticRegression(max_iter=200, random_state=42, C=1.0, solver="lbfgs")
        model.fit(X_enc[valid], y_s[valid])
        pred = model.predict(X_enc[valid])
        proba = model.predict_proba(X_enc[valid]).max(axis=1)
        mispredict = (pred != y_s[valid]) & (proba >= P)
        remove_idx = set(X_enc[valid][mispredict].index.tolist())
        keep_idx = [i for i in X.index if i not in remove_idx]
        return {"_train_rows": keep_idx}
    except Exception:
        return {"_train_rows": list(X.index)}


def _abstain_apply(X: pd.DataFrame, col_types: Dict, state: Dict) -> pd.DataFrame:
    return X  # no-op at inference time


def _correctTypos_fit(X: pd.DataFrame, y, col_types: Dict, min_count: int = 2) -> Dict:
    """Build valid-value vocab for each categorical column."""
    state: Dict = {}
    for col in col_types["categorical"]:
        if col not in X.columns:
            continue
        vc = X[col].value_counts()
        valid = list(vc[vc >= min_count].index)
        if valid:
            mode_val = valid[0]  # most frequent valid value
            state[col] = {"valid": set(str(v) for v in valid), "mode": mode_val}
    return state


def _correctTypos_apply(X: pd.DataFrame, col_types: Dict, state: Dict) -> pd.DataFrame:
    X_out = X.copy()
    for col, col_state in state.items():
        if col not in X_out.columns:
            continue
        valid = col_state["valid"]
        mode = col_state["mode"]
        mask = X_out[col].notna() & ~X_out[col].astype(str).isin(valid)
        if mask.any():
            X_out.loc[mask, col] = mode
    return X_out


# Dispatch tables
_FIT_FNS = {
    "outlierByIQR":  _outlierByIQR_fit,
    "outlierBySd":   _outlierBySd_fit,
    "winsorize":     _winsorize_fit,
    "imputeByMean":  _imputeByMean_fit,
    "imputeByMedian": _imputeByMedian_fit,
    "fillForward":   _fillForward_fit,
    "fillDefault":   _fillDefault_fit,
    "normalize":     _normalize_fit,
    "underSampling": _underSampling_fit,
    "abstain":       _abstain_fit,
    "correctTypos":  _correctTypos_fit,
}

_APPLY_FNS = {
    "outlierByIQR":  _outlierByIQR_apply,
    "outlierBySd":   _outlierBySd_apply,
    "winsorize":     _winsorize_apply,
    "imputeByMean":  _imputeByMean_apply,
    "imputeByMedian": _imputeByMedian_apply,
    "fillForward":   _fillForward_apply,
    "fillDefault":   _fillDefault_apply,
    "normalize":     _normalize_apply,
    "underSampling": _underSampling_apply,
    "abstain":       _abstain_apply,
    "correctTypos":  _correctTypos_apply,
}


# ---------------------------------------------------------------------------
# Pipeline representation and operations
# ---------------------------------------------------------------------------
# A pipeline is a list of (primitive_name: str, params: dict) tuples.

Pipeline = List[Tuple[str, Dict]]


def _pipeline_key(pipeline: Pipeline) -> str:
    """Hashable string representation of a pipeline for deduplication."""
    return "|".join(f"{n}:{sorted(p.items())}" for n, p in pipeline)


def _reorder_pipeline(pipeline: Pipeline) -> Pipeline:
    """Move stage-0 primitives to the front (Section 2.3 of the paper)."""
    stage0 = [(n, p) for n, p in pipeline if n in STAGE0_PRIMITIVES]
    rest = [(n, p) for n, p in pipeline if n not in STAGE0_PRIMITIVES]
    return stage0 + rest


def _apply_genetic_transition(
    pipeline: Pipeline,
    all_pipelines: List[Pipeline],
    rng: random.Random,
) -> Pipeline:
    """
    Apply one of four genetic transitions (Section 3.1):
      Addition, Crossover, Mutation, Removal
    """
    r = rng.random()
    if r < 0.30 or len(pipeline) == 0:
        # Addition
        prim = rng.choice(ALL_PRIMITIVES)
        pos = rng.randint(0, len(pipeline))
        new_p = list(pipeline)
        new_p.insert(pos, (prim, copy.deepcopy(PRIMITIVE_DEFAULTS[prim])))
        return _reorder_pipeline(new_p)
    elif r < 0.55 and len(pipeline) >= 2:
        # Mutation (swap two primitives)
        new_p = list(pipeline)
        i, j = rng.sample(range(len(pipeline)), 2)
        new_p[i], new_p[j] = new_p[j], new_p[i]
        return _reorder_pipeline(new_p)
    elif r < 0.75 and len(all_pipelines) >= 2:
        # Crossover: take prefix of current + suffix of another
        other = rng.choice(all_pipelines)
        split_a = rng.randint(0, len(pipeline))
        split_b = rng.randint(0, len(other))
        new_p = list(pipeline[:split_a]) + list(other[split_b:])
        return _reorder_pipeline(new_p) if new_p else [(rng.choice(ALL_PRIMITIVES),
                                                         copy.deepcopy(PRIMITIVE_DEFAULTS[rng.choice(ALL_PRIMITIVES)]))]
    else:
        # Removal
        if len(pipeline) <= 1:
            return list(pipeline)
        pos = rng.randint(0, len(pipeline) - 1)
        new_p = [p for i, p in enumerate(pipeline) if i != pos]
        return _reorder_pipeline(new_p)


# ---------------------------------------------------------------------------
# Model encoding helper (for pipeline scoring)
# ---------------------------------------------------------------------------

def _encode_for_model(X: pd.DataFrame, col_types: Dict) -> pd.DataFrame:
    """
    Encode a feature dataframe for quick sklearn model evaluation.
    Ordinal-encodes categoricals, fills remaining NaN with median/0.
    """
    from sklearn.preprocessing import OrdinalEncoder

    X_enc = X.copy()
    cat_present = [c for c in col_types["categorical"] if c in X_enc.columns]
    if cat_present:
        enc = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
        X_enc[cat_present] = enc.fit_transform(X_enc[cat_present].astype(str))
    for col in X_enc.columns:
        X_enc[col] = pd.to_numeric(X_enc[col], errors="coerce")
        if X_enc[col].isna().any():
            fill = X_enc[col].median() if X_enc[col].notna().any() else 0.0
            X_enc[col] = X_enc[col].fillna(fill)
    return X_enc


# ---------------------------------------------------------------------------
# Core Saga++ class
# ---------------------------------------------------------------------------

class SagaPP:
    """
    Python implementation of the Saga++ framework (Siddiqi et al., TODS 2026).

    Parameters
    ----------
    K          : int   – number of top-K pipelines to return (default 3)
    max_iter   : int   – max evolutionary iterations for logical enumeration (default 10)
    resources  : int   – Hyperband resource budget R per bucket (default 20)
    seed       : int   – random seed
    n_cv_folds : int   – cross-validation folds for pipeline scoring (default 3)
    pop_size   : int   – evolutionary population size (default 16)
    """

    def __init__(
        self,
        K: int = 3,
        max_iter: int = 10,
        resources: int = 20,
        seed: int = 42,
        n_cv_folds: int = 3,
        pop_size: int = 16,
        n_jobs: int = 16,
    ):
        self.K = K
        self.max_iter = max_iter
        self.resources = resources
        self.seed = seed
        self.n_cv_folds = n_cv_folds
        self.pop_size = pop_size
        self.n_jobs = n_jobs
        self._rng = random.Random(seed)
        np.random.seed(seed)

    # ------------------------------------------------------------------
    # Public API (Section 2.3 of the paper)
    # ------------------------------------------------------------------

    def topk_cleaning(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        col_types: Dict,
    ) -> List[Tuple[Pipeline, float]]:
        """
        Find the top-K most effective data cleaning pipelines.

        Implements the two-level optimization from the paper:
          1. Logical pipeline enumeration (evolutionary algorithm, Algorithm 1)
          2. Physical pipeline tuning (Hyperband, Algorithm 2)

        Returns
        -------
        List of (pipeline, score) sorted by loss ascending (best first).
        Only pipelines that improve upon the dirty score are returned.
        """
        logger.info("  [Saga++] Computing dirty baseline score ...")
        dirty_loss = self._dirty_score(X, y, col_types)
        logger.info(f"  [Saga++] Dirty loss = {dirty_loss:.4f}")

        logger.info("  [Saga++] Phase 1: Logical pipeline enumeration ...")
        logical_pipelines = self._enumerate_logical_pipelines(X, y, col_types, dirty_loss)
        logger.info(f"  [Saga++] Found {len(logical_pipelines)} candidate logical pipelines")

        if not logical_pipelines:
            logger.warning("  [Saga++] No improving logical pipelines found; returning empty-pipeline")
            empty: Pipeline = []
            return [(empty, dirty_loss)]

        logger.info("  [Saga++] Phase 2: Physical pipeline tuning (Hyperband) ...")
        top_k = self._tune_physical_pipelines(logical_pipelines, X, y, col_types, dirty_loss)
        logger.info(f"  [Saga++] Top-K result: {len(top_k)} pipelines")
        return top_k

    def fit_pipeline(
        self,
        pipeline: Pipeline,
        X: pd.DataFrame,
        y: Optional[pd.Series],
        col_types: Dict,
    ) -> Tuple[Dict, pd.DataFrame, Optional[pd.Series]]:
        """
        Fit a cleaning pipeline on training data.

        Returns
        -------
        states     : dict primitive_name → fitted state (to pass to apply_pipeline)
        X_cleaned  : cleaned feature dataframe
        y_cleaned  : cleaned labels (may have fewer rows if train-only primitives removed rows)
        """
        states: Dict[str, Any] = {}
        X_cur = X.copy()
        y_cur = y.copy() if y is not None else None

        for name, params in pipeline:
            fit_fn = _FIT_FNS.get(name)
            if fit_fn is None:
                continue
            try:
                state = fit_fn(X_cur, y_cur, col_types, **params)
            except Exception as exc:
                logger.debug(f"    fit {name} failed: {exc}")
                state = {}
            states[name] = state

            # Apply transformation to current data
            apply_fn = _APPLY_FNS.get(name)
            if apply_fn is None:
                continue
            try:
                X_cur = apply_fn(X_cur, col_types, state)
            except Exception as exc:
                logger.debug(f"    apply {name} failed: {exc}")

            # Handle row-removing train-only primitives
            if name in TRAIN_ONLY_PRIMITIVES and "_train_rows" in state:
                keep = [i for i in state["_train_rows"] if i in X_cur.index]
                X_cur = X_cur.loc[keep]
                if y_cur is not None:
                    y_cur = y_cur.loc[keep]

        return states, X_cur, y_cur

    def apply_pipeline(
        self,
        pipeline: Pipeline,
        X: pd.DataFrame,
        states: Dict,
        col_types: Dict,
    ) -> pd.DataFrame:
        """
        Transform data for inference using a previously fitted pipeline.
        Train-only primitives (abstain, underSampling) are skipped.
        """
        X_cur = X.copy()
        for name, _ in pipeline:
            if name in TRAIN_ONLY_PRIMITIVES:
                continue
            apply_fn = _APPLY_FNS.get(name)
            if apply_fn is None:
                continue
            state = states.get(name, {})
            try:
                X_cur = apply_fn(X_cur, col_types, state)
            except Exception as exc:
                logger.debug(f"    apply {name} failed: {exc}")
        return X_cur

    # ------------------------------------------------------------------
    # Main entry point (mirrors DataPreparation.prepare())
    # ------------------------------------------------------------------

    def prepare(
        self,
        df_poisoned: pd.DataFrame,
        mask_df: pd.DataFrame,
    ) -> Tuple[pd.DataFrame, pd.DataFrame, Dict, Dict]:
        """
        Apply the Saga++ top-K cleaning pipeline to a poisoned dataset.

        Steps:
          1. topk_cleaning() — find the best cleaning pipeline
          2. fit_pipeline()  — fit the best pipeline on the full dataset
          3. apply_pipeline()— apply to produce the cleaned dataset
          4. pipeline_pruning— remove redundant primitives (Algorithm 4)

        Returns
        -------
        df_cleaned    : cleaned dataframe
        residual_mask : boolean mask (True = originally poisoned, still NaN)
        perf_metrics  : timing / memory metrics dict
        pipeline_info : dict with keys "pipeline", "states", "col_types" for
                        reapplying the fitted pipeline to held-out data
        """
        proc = psutil.Process()
        ram_before_mb = proc.memory_info().rss / 1024 / 1024
        tracemalloc.start()
        wall_t0 = time.perf_counter()
        cpu_t0 = time.process_time()

        col_types = _identify_col_types(df_poisoned)
        feat_cols = col_types["numerical"] + col_types["categorical"]

        # Build feature matrix and target vector
        X = df_poisoned[feat_cols].copy() if feat_cols else pd.DataFrame(index=df_poisoned.index)
        y: Optional[pd.Series] = None
        is_cls = bool(col_types["target_cls"])
        if col_types["target_cls"]:
            y = df_poisoned[col_types["target_cls"][0]].copy()
        elif col_types["target_reg"]:
            y = df_poisoned[col_types["target_reg"][0]].copy()

        # Phase 1 + 2: top-K cleaning
        top_pipelines = self.topk_cleaning(X, y, col_types)
        best_pipeline, best_score = top_pipelines[0]
        logger.info(
            f"  [Saga++] Best pipeline (score={best_score:.4f}): "
            + " → ".join(n for n, _ in best_pipeline) if best_pipeline else "(empty)"
        )

        # Phase 3: Pipeline pruning (Algorithm 4 from the paper)
        best_pipeline = self._prune_pipeline(best_pipeline, X, y, col_types)
        logger.info(
            "  [Saga++] Pruned pipeline: "
            + (" → ".join(n for n, _ in best_pipeline) if best_pipeline else "(empty)")
        )

        # Fit the best pipeline on the full dataset
        states, X_cleaned, _ = self.fit_pipeline(best_pipeline, X, y, col_types)

        # Pipeline info for later reapplication to held-out (test) data
        pipeline_info: Dict = {
            "pipeline": best_pipeline,
            "states": states,
            "col_types": col_types,
        }

        # Reconstruct the full dataframe
        df_clean = df_poisoned.copy()
        if feat_cols:
            # align index (row-removing primitives may have shrunk X_cleaned)
            df_clean = df_poisoned.copy()
            for col in X_cleaned.columns:
                df_clean.loc[X_cleaned.index, col] = X_cleaned[col].values

        # Residual mask: originally poisoned cells that are still NaN
        remaining_nan = df_clean.isna()
        residual_mask = mask_df & remaining_nan

        repaired = int(mask_df.sum().sum()) - int(residual_mask.sum().sum())
        logger.info(
            f"  [Saga++] Repaired {repaired} / {int(mask_df.sum().sum())} poisoned cells "
            f"({int(residual_mask.sum().sum())} residual)"
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
            "best_pipeline": " → ".join(n for n, _ in best_pipeline) if best_pipeline else "(empty)",
            "best_score": round(best_score, 6),
        }

        return df_clean, residual_mask, perf_metrics, pipeline_info

    # ------------------------------------------------------------------
    # Internal: pipeline scoring
    # ------------------------------------------------------------------

    def _dirty_score(self, X: pd.DataFrame, y: Optional[pd.Series], col_types: Dict) -> float:
        """Score the unclean data (baseline loss D in the paper)."""
        return self._score_pipeline([], X, y, col_types)

    def _score_pipeline(
        self,
        pipeline: Pipeline,
        X: pd.DataFrame,
        y: Optional[pd.Series],
        col_types: Dict,
    ) -> float:
        """
        Score a pipeline via k-fold cross-validation on the target ML application.
        Returns loss (lower is better; loss = 1-accuracy for cls, 1-max(0,R²) for reg).
        """
        if y is None or y.isna().all():
            return 0.5  # cannot score without labels

        from sklearn.linear_model import LogisticRegression, Ridge
        from sklearn.model_selection import StratifiedKFold, KFold

        is_cls = bool(col_types["target_cls"])
        feat_cols = [c for c in col_types["numerical"] + col_types["categorical"] if c in X.columns]
        if not feat_cols:
            return 0.5

        y_clean = y.dropna()
        X_align = X.loc[y_clean.index]

        try:
            if is_cls:
                kf = StratifiedKFold(n_splits=self.n_cv_folds, shuffle=True, random_state=self.seed)
                splits = list(kf.split(X_align, y_clean))
            else:
                kf = KFold(n_splits=self.n_cv_folds, shuffle=True, random_state=self.seed)
                splits = list(kf.split(X_align))
        except Exception:
            return 0.5

        fold_losses = []
        for train_idx, val_idx in splits:
            X_tr = X_align.iloc[train_idx][feat_cols]
            X_vl = X_align.iloc[val_idx][feat_cols]
            y_tr = y_clean.iloc[train_idx]
            y_vl = y_clean.iloc[val_idx]

            # Fit pipeline on training fold, then apply fitted state to validation fold
            try:
                states, X_tr_c, y_tr_c = self.fit_pipeline(pipeline, X_tr, y_tr, col_types)
                X_vl_c = self.apply_pipeline(pipeline, X_vl, states, col_types)
            except Exception:
                fold_losses.append(1.0)
                continue

            if len(X_tr_c) < 3:
                fold_losses.append(1.0)
                continue

            X_tr_enc = _encode_for_model(X_tr_c[feat_cols] if feat_cols else X_tr_c, col_types)
            X_vl_enc = _encode_for_model(X_vl_c[feat_cols] if feat_cols else X_vl_c, col_types)

            # Align validation labels with potentially missing rows
            y_vl_aligned = y_vl.loc[y_vl.index.intersection(X_vl_enc.index)]
            X_vl_enc = X_vl_enc.loc[y_vl_aligned.index]

            try:
                if is_cls:
                    model = LogisticRegression(max_iter=200, random_state=self.seed, C=1.0, solver="lbfgs")
                    model.fit(X_tr_enc, y_tr_c)
                    from sklearn.metrics import accuracy_score
                    loss = 1.0 - accuracy_score(y_vl_aligned, model.predict(X_vl_enc))
                else:
                    model = Ridge(alpha=1.0, random_state=self.seed)
                    model.fit(X_tr_enc, y_tr_c)
                    from sklearn.metrics import r2_score
                    r2 = r2_score(y_vl_aligned, model.predict(X_vl_enc))
                    loss = 1.0 - max(0.0, r2)
            except Exception:
                loss = 1.0

            fold_losses.append(loss)

        return float(np.mean(fold_losses)) if fold_losses else 1.0

    # ------------------------------------------------------------------
    # Internal: Algorithm 1 — Logical Pipeline Enumeration
    # ------------------------------------------------------------------

    def _enumerate_logical_pipelines(
        self,
        X: pd.DataFrame,
        y: Optional[pd.Series],
        col_types: Dict,
        dirty_loss: float,
    ) -> List[Tuple[Pipeline, float]]:
        """
        Evolutionary algorithm for logical pipeline enumeration (Algorithm 1).
        Uses genetic transitions (addition, crossover, mutation, removal) and
        successive halving (η=2) to find promising pipeline structures.
        Returns list of (pipeline, default-param score) sorted ascending.
        """
        # Seed population: one 1-primitive pipeline per primitive
        population: List[Pipeline] = [[(name, copy.deepcopy(PRIMITIVE_DEFAULTS[name]))]
                                       for name in ALL_PRIMITIVES]
        # Pad to pop_size with random 2-prim combinations
        while len(population) < self.pop_size:
            a, b = self._rng.sample(ALL_PRIMITIVES, 2)
            population.append(_reorder_pipeline([
                (a, copy.deepcopy(PRIMITIVE_DEFAULTS[a])),
                (b, copy.deepcopy(PRIMITIVE_DEFAULTS[b])),
            ]))

        seen: set = set()
        all_logical: List[Tuple[Pipeline, float]] = []
        loss_history: List[float] = []
        target_loss = dirty_loss * 0.95  # target: 5% improvement

        for iteration in range(self.max_iter):
            # Collect unseen pipelines, mark seen before dispatching to workers
            unseen: List[Pipeline] = []
            for pip in population:
                key = _pipeline_key(pip)
                if key not in seen:
                    seen.add(key)
                    unseen.append(pip)

            # Score unseen pipelines in parallel across all cores
            losses: List[float] = Parallel(n_jobs=self.n_jobs, backend="loky")(
                delayed(self._score_pipeline)(pip, X, y, col_types)
                for pip in unseen
            )

            scored: List[Tuple[Pipeline, float]] = list(zip(unseen, losses))
            all_logical.extend(scored)

            if not scored:
                break

            # Sort ascending (lower loss = better) and successive halving (η=2)
            scored.sort(key=lambda t: t[1])
            top_half = scored[: max(1, len(scored) // 2)]
            best_loss = top_half[0][1]
            loss_history.append(best_loss)

            # Convergence check: last 3 iterations didn't improve, or reached target
            converged = (
                best_loss <= target_loss
                or (
                    len(loss_history) >= 3
                    and len(set(round(v, 5) for v in loss_history[-3:])) == 1
                )
            )
            if converged:
                logger.debug(
                    f"    Converged at iteration {iteration + 1}, best_loss={best_loss:.4f}"
                )
                break

            # Generate next population via genetic transitions
            top_pipelines = [p for p, _ in top_half]
            new_population: List[Pipeline] = list(top_pipelines)
            while len(new_population) < self.pop_size:
                base = self._rng.choice(top_pipelines)
                child = _apply_genetic_transition(base, top_pipelines, self._rng)
                if child:
                    new_population.append(child)
            population = new_population

        # Return only pipelines that improve upon dirty loss
        improving = [(p, l) for p, l in all_logical if l < dirty_loss]
        improving.sort(key=lambda t: t[1])
        # Deduplicate by pipeline key
        seen_keys: set = set()
        deduped: List[Tuple[Pipeline, float]] = []
        for pip, loss in improving:
            k = _pipeline_key(pip)
            if k not in seen_keys:
                seen_keys.add(k)
                deduped.append((pip, loss))
        return deduped

    # ------------------------------------------------------------------
    # Internal: Algorithm 2 — Physical Pipeline Tuning (Hyperband)
    # ------------------------------------------------------------------

    def _tune_physical_pipelines(
        self,
        logical_pipelines: List[Tuple[Pipeline, float]],
        X: pd.DataFrame,
        y: Optional[pd.Series],
        col_types: Dict,
        dirty_loss: float,
    ) -> List[Tuple[Pipeline, float]]:
        """
        Hyperband-based physical pipeline tuning (Algorithm 2).
        Creates buckets (s_max brackets) with different resource allocations
        and applies successive halving within each bucket.
        Returns top-K physical pipelines that improve upon dirty_loss.
        """
        import math

        R = self.resources
        eta = 2
        s_max = max(1, int(math.floor(math.log(R, eta))))
        n_logical = len(logical_pipelines)
        n_per_bucket = max(1, n_logical // (s_max + 1))

        top_k_all: List[Tuple[Pipeline, float]] = []

        # Sort logical pipelines by default score; assign to buckets
        logical_sorted = sorted(logical_pipelines, key=lambda t: t[1])

        for s in range(s_max, -1, -1):
            # Bucket-specific resource (more resources for better buckets)
            weight = (s_max - s + 1) / (s_max + 1)
            r_base = max(1, int(R * weight * (eta ** (-s))))

            # Assign n_per_bucket logical pipelines to this bucket
            start = (s_max - s) * n_per_bucket
            bucket_logical = logical_sorted[start: start + n_per_bucket]
            if not bucket_logical:
                continue

            # Materialise all physical pipelines for this bucket, then score in parallel
            all_phys_pips: List[Pipeline] = []
            for log_pip, _ in bucket_logical:
                param_configs = self._sample_param_configs(log_pip, r_base)
                for config in param_configs:
                    all_phys_pips.append(
                        [(name, config.get(name, params)) for name, params in log_pip]
                    )

            phys_losses: List[float] = Parallel(n_jobs=self.n_jobs, backend="loky")(
                delayed(self._score_pipeline)(phys_pip, X, y, col_types)
                for phys_pip in all_phys_pips
            )
            physical_candidates: List[Tuple[Pipeline, float]] = list(zip(all_phys_pips, phys_losses))

            # Successive halving within bucket
            current = sorted(physical_candidates, key=lambda t: t[1])
            for i in range(s + 1):
                n_keep = max(1, len(current) // eta)
                current = current[:n_keep]
                if len(current) == 1:
                    break

            top_k_all.extend(current)

        # Collect and rank top-K physical pipelines across all buckets
        top_k_all.sort(key=lambda t: t[1])
        # Keep only those better than dirty; deduplicate by pipeline key
        seen_keys: set = set()
        result: List[Tuple[Pipeline, float]] = []
        for pip, loss in top_k_all:
            if loss >= dirty_loss:
                continue
            k = _pipeline_key(pip)
            if k in seen_keys:
                continue
            seen_keys.add(k)
            result.append((pip, loss))
            if len(result) >= self.K:
                break

        if not result:
            # Fall back to best logical pipeline with default params
            if logical_pipelines:
                result = [logical_pipelines[0]]
            else:
                result = [([], dirty_loss)]

        return result

    def _sample_param_configs(
        self, pipeline: Pipeline, n: int
    ) -> List[Dict[str, Dict]]:
        """Sample n random hyperparameter configurations for a logical pipeline."""
        configs: List[Dict[str, Dict]] = []
        for _ in range(n):
            config: Dict[str, Dict] = {}
            for name, _ in pipeline:
                grid = PARAM_GRIDS.get(name, [{}])
                config[name] = copy.deepcopy(self._rng.choice(grid))
            configs.append(config)
        return configs

    # ------------------------------------------------------------------
    # Internal: Algorithm 4 — Pipeline Pruning
    # ------------------------------------------------------------------

    def _prune_pipeline(
        self,
        pipeline: Pipeline,
        X: pd.DataFrame,
        y: Optional[pd.Series],
        col_types: Dict,
    ) -> Pipeline:
        """
        Post-processing pruning (Algorithm 4): enumerate consecutive subsets of
        primitives and return the smallest subset with the best accuracy.
        Reduces pipeline from 2^(n-1) to n(n+1)/2 candidates by restricting to
        contiguous sub-pipelines. Also removes duplicate consecutive primitives
        and category-repeated primitives.
        """
        if len(pipeline) <= 1:
            return pipeline

        # Step 1: Remove redundant consecutive primitives
        deduped: Pipeline = [pipeline[0]]
        for name, params in pipeline[1:]:
            if name != deduped[-1][0]:
                deduped.append((name, params))
        pipeline = deduped

        # Step 2: Enumerate all contiguous subsets
        n = len(pipeline)
        best_loss = self._score_pipeline(pipeline, X, y, col_types)
        best_subset = pipeline

        # Category-based pruning helper
        _CATEGORIES = {
            "outlierByIQR": "outlier", "outlierBySd": "outlier", "winsorize": "outlier",
            "imputeByMean": "impute", "imputeByMedian": "impute",
            "fillForward": "impute", "fillDefault": "impute",
        }

        for start_idx in range(n):
            for end_idx in range(start_idx, n):
                subset = pipeline[start_idx: end_idx + 1]
                if subset == pipeline:
                    continue
                # Category-based pruning: skip subsets with 2+ same-category prims
                cats = [_CATEGORIES.get(nm, nm) for nm, _ in subset]
                if len(cats) != len(set(cats)):
                    continue
                loss = self._score_pipeline(subset, X, y, col_types)
                if loss < best_loss:
                    best_loss = loss
                    best_subset = subset

        return best_subset


# ---------------------------------------------------------------------------
# Metrics helper (mirrors data_preparation_pipeline.py)
# ---------------------------------------------------------------------------

def calculate_residual_metrics(residual_mask: pd.DataFrame) -> Dict:
    col_clean = (1 - residual_mask.sum(axis=0) / len(residual_mask)) * 100
    row_clean = (1 - residual_mask.sum(axis=1) / residual_mask.shape[1]) * 100
    overall_clean = (1 - residual_mask.sum().sum() / residual_mask.size) * 100
    return {
        "column_cleanliness": col_clean.to_dict(),
        "row_cleanliness": row_clean.to_dict(),
        "overall_cleanliness": overall_clean,
    }


# ---------------------------------------------------------------------------
# Pipeline persistence helpers
# ---------------------------------------------------------------------------

def save_pipeline_info(path: str, pipeline_info: Dict) -> None:
    """Persist pipeline steps, fitted states and column types to a pickle file."""
    import pickle as _pickle
    with open(path, "wb") as f:
        _pickle.dump(pipeline_info, f)


# ---------------------------------------------------------------------------
# Dataset processing (mirrors process_all_datasets in data_preparation_pipeline.py)
# ---------------------------------------------------------------------------

def check_dataset_complete(csv_file: Path, output_dir: str) -> bool:
    required_files = []
    for mode in ("ar", "nar"):
        mode_dir = os.path.join(output_dir, mode)
        required_files += [
            os.path.join(mode_dir, csv_file.name),
            os.path.join(mode_dir, csv_file.stem + "_mask.csv"),
            os.path.join(mode_dir, csv_file.stem + "_pipeline.pkl"),
            os.path.join(output_dir, "metrics", f"{csv_file.stem}_{mode}_metrics.csv"),
            os.path.join(output_dir, "metrics", f"{csv_file.stem}_{mode}_perf_metrics.csv"),
        ]
    return all(os.path.exists(f) for f in required_files)


def process_all_datasets(
    input_dir: str,
    output_dir: str,
    datasets: Optional[List[str]] = None,
    K: int = 3,
    max_iter: int = 20,
    resources: int = 40,
    seed: int = 42,
    n_jobs: int = 16,
):
    """
    Apply the Saga++ cleaning pipeline to all poisoned datasets.

    Reads poisoned CSVs and their masks from ``input_dir/{ar,nar}/`` and
    writes cleaned CSVs, residual masks, fitted pipeline states and metrics to
    ``output_dir/{ar,nar}/``.

    Args:
        input_dir : Root poisoned-data directory (default: data_poisoned)
        output_dir: Root output directory (default: data_cleaned_saga)
        datasets  : List of dataset names to process (from config.yaml); None = all
        K         : Number of top-K pipelines to search (default 3)
        max_iter  : Evolutionary iterations for logical enumeration (default 10)
        resources : Hyperband resource budget R (default 20)
        seed      : Random seed
    """
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.join(output_dir, "ar"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "nar"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "metrics"), exist_ok=True)

    cleaner = SagaPP(K=K, max_iter=max_iter, resources=resources, seed=seed, n_jobs=n_jobs)

    ar_dir = Path(input_dir) / "ar"
    nar_dir = Path(input_dir) / "nar"

    if not ar_dir.exists():
        logger.error(f"AR poisoned directory not found: {ar_dir}")
        return
    if not nar_dir.exists():
        logger.error(f"NAR poisoned directory not found: {nar_dir}")
        return

    csv_files = sorted(ar_dir.glob("*.csv"))
    csv_files = [f for f in csv_files if not f.stem.endswith("_mask")]

    if not csv_files:
        logger.error(f"No poisoned CSV files found in {ar_dir}")
        return

    if datasets:
        dataset_set = set(datasets)
        csv_files = [f for f in csv_files if f.stem[10:] in dataset_set]
        if not csv_files:
            logger.error(f"None of the specified datasets found in {ar_dir}")
            return
        logger.info(f"Processing {len(csv_files)} configured dataset(s)")
    else:
        logger.info(f"Found {len(csv_files)} poisoned datasets to prepare")

    skipped = sum(1 for f in csv_files if check_dataset_complete(f, output_dir))
    if skipped:
        logger.info(f"Skipped {skipped} already-processed dataset(s)")

    for csv_file in csv_files:
        if check_dataset_complete(csv_file, output_dir):
            logger.info(f"Skipping {csv_file.name} (already complete)")
            continue

        logger.info(f"Processing {csv_file.name}")

        try:
            for mode, mode_dir in (("ar", ar_dir), ("nar", nar_dir)):
                src_csv = mode_dir / csv_file.name
                src_mask = mode_dir / f"{csv_file.stem}_mask.csv"
                out_dir = Path(output_dir) / mode

                out_csv = out_dir / csv_file.name
                out_mask = out_dir / f"{csv_file.stem}_mask.csv"
                out_pkl = out_dir / f"{csv_file.stem}_pipeline.pkl"
                out_metrics = Path(output_dir) / "metrics" / f"{csv_file.stem}_{mode}_metrics.csv"
                out_perf = Path(output_dir) / "metrics" / f"{csv_file.stem}_{mode}_perf_metrics.csv"

                if all(p.exists() for p in [out_csv, out_mask, out_pkl, out_metrics, out_perf]):
                    logger.info(f"  {mode.upper()} already processed, skipping")
                    continue

                if not src_csv.exists():
                    logger.warning(f"  {mode.upper()} CSV not found, skipping: {src_csv}")
                    continue
                if not src_mask.exists():
                    logger.warning(f"  {mode.upper()} mask not found, skipping: {src_mask}")
                    continue

                df_mode = pd.read_csv(
                    src_csv, na_values=["?", "NA", "N/A", "NaN", "nan", "NAN", "", " "]
                )
                mask_mode = pd.read_csv(src_mask).astype(bool)
                logger.info(
                    f"  {mode.upper()} loaded: {df_mode.shape[0]} rows × {df_mode.shape[1]} cols"
                )

                df_clean, residual_mask, perf, pipeline_info = cleaner.prepare(df_mode, mask_mode)
                metrics = calculate_residual_metrics(residual_mask)

                df_clean.to_csv(out_csv, index=False)
                residual_mask.to_csv(out_mask, index=False)
                save_pipeline_info(str(out_pkl), pipeline_info)

                pd.DataFrame({
                    "column": list(metrics["column_cleanliness"].keys()),
                    "cleanliness_pct": list(metrics["column_cleanliness"].values()),
                }).to_csv(out_metrics, index=False)

                pd.DataFrame([{
                    "dataset": csv_file.stem,
                    "corruption": mode,
                    **perf,
                }]).to_csv(out_perf, index=False)

                logger.info(
                    f"  {mode.upper()} cleaned: {metrics['overall_cleanliness']:.2f}% clean "
                    f"| {perf['wall_time_s']:.1f}s wall | {perf['cpu_time_s']:.1f}s CPU "
                    f"| {perf['ram_peak_mb']:.1f} MB peak "
                    f"| pipeline: {perf.get('best_pipeline', '?')}"
                )

            logger.success(f"  Completed {csv_file.name}")

        except Exception as e:
            logger.error(f"  Error processing {csv_file.name}: {e}")
            import traceback
            logger.debug(traceback.format_exc())
            continue

    logger.success(f"All datasets prepared! Output in {output_dir}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse
    import yaml

    parser = argparse.ArgumentParser(
        description="Saga++ automated data cleaning pipeline baseline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
                Paper: Siddiqi et al. "Saga++: A Scalable Framework for Optimizing Data
                    Cleaning Pipelines for ML Applications." ACM TODS 51(2), 2026.

                Framework API in Python:
                topk_cleaning()  — finds top-K cleaning pipelines via evolutionary
                                    algorithm (logical) + Hyperband (physical tuning)
                fit_pipeline()   — fits a selected pipeline, returns state + cleaned data
                apply_pipeline() — applies a fitted pipeline to new data

                Cleaning primitives (Table 1 from the paper):
                Outliers      : outlierByIQR, outlierBySd, winsorize
                MV Imputation : imputeByMean, imputeByMedian, fillForward, fillDefault
                Data Prep     : normalize
                Class Imbal.  : underSampling
                Labels        : abstain
                String stage0 : correctTypos

                Examples:
                python saga.py
                python saga.py --input_dir data_poisoned --output_dir data_cleaned_saga
                python saga.py --dataset iris
                python saga.py --max_iter 15 --resources 50 --K 5
        """,
    )
    parser.add_argument(
        "--input_dir", type=str, default="data_poisoned",
        help="Root directory of poisoned data (default: data_poisoned)",
    )
    parser.add_argument(
        "--output_dir", type=str, default="data_cleaned_saga",
        help="Root directory for cleaned output (default: data_cleaned_saga)",
    )
    parser.add_argument(
        "--config", type=str, default="config.yaml",
        help="Path to experiment config file (default: config.yaml)",
    )
    parser.add_argument(
        "--dataset", type=str, default=None,
        help="Process only this dataset (overrides config; name without .csv, e.g. 'iris')",
    )
    parser.add_argument(
        "--K", type=int, default=3,
        help="Number of top-K pipelines to search (default: 3)",
    )
    parser.add_argument(
        "--max_iter", type=int, default=10,
        help="Max evolutionary iterations for logical pipeline enumeration (default: 10)",
    )
    parser.add_argument(
        "--resources", type=int, default=20,
        help="Hyperband resource budget R per bucket (default: 20)",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed (default: 42)",
    )
    parser.add_argument(
        "--n_jobs", type=int, default=16,
        help="Parallel workers for pipeline scoring (default: 16; -1 = all cores)",
    )

    args = parser.parse_args()

    _config: dict = {}
    if Path(args.config).exists():
        with open(args.config) as _f:
            _config = yaml.safe_load(_f) or {}

    if args.dataset:
        datasets = [args.dataset]
    else:
        datasets = _config.get("datasets") or None

    process_all_datasets(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        datasets=datasets,
        K=args.K,
        max_iter=args.max_iter,
        resources=args.resources,
        seed=args.seed,
        n_jobs=args.n_jobs,
    )
