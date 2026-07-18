"""CatBoost baseline training utilities.

CatBoost natively handles missing values and categorical features, so this
module intentionally does NOT run any imputation, one-hot encoding, or
scaling: numerical columns keep their NaN values (CatBoost bins missing
values as their own split candidate) and categorical columns are passed as
native strings (CatBoost treats each category as its own value). This is a
deliberate contrast with the linear/MLP pipeline in ``quail.training``,
which relies on ``quail.preprocessing.TabularPreprocessor``.

Use ``quail.data.load_raw_data`` to obtain the unprocessed splits this
module expects.
"""

from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier, CatBoostRegressor, Pool
from sklearn.metrics import log_loss, mean_squared_error

from quail.training import _compute_metrics

# CatBoost's Pool cannot accept a float NaN for categorical columns (only
# int/string values are allowed there); missing categories are given this
# explicit sentinel category instead. This is not statistical imputation —
# it preserves "missingness" as its own distinct category rather than
# guessing a plausible value (mean/mode) for it.
_CAT_MISSING_TOKEN = "__MISSING__"


def _make_pool(X: pd.DataFrame, cat_features: List[str], y: Optional[np.ndarray] = None) -> Pool:
    X = X.copy()
    for col in cat_features:
        if col in X.columns:
            X[col] = X[col].fillna(_CAT_MISSING_TOKEN)
    return Pool(X, y, cat_features=cat_features)


def fit_catboost(
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    X_val: pd.DataFrame,
    y_val: np.ndarray,
    X_test: Optional[pd.DataFrame],
    y_test: Optional[np.ndarray],
    cat_features: List[str],
    task: str,
    iterations: int = 1000,
    learning_rate: float = 0.05,
    depth: int = 6,
    l2_leaf_reg: float = 3.0,
    random_strength: float = 1.0,
    bagging_temperature: float = 1.0,
    border_count: int = 128,
    min_data_in_leaf: int = 1,
    grow_policy: str = "SymmetricTree",
    early_stopping_rounds: int = 50,
    random_seed: int = 42,
    thread_count: int = -1,
    verbose: int = 0,
) -> Tuple[Any, Dict[str, List[float]]]:
    """
    Train a CatBoost model directly on raw (unprocessed) tabular data.

    Parameters
    ----------
    X_train, X_val, X_test : pd.DataFrame
        Raw feature frames (as returned by ``quail.data.load_raw_data``).
        ``X_test`` may be None.
    y_train, y_val, y_test : np.ndarray
        Label arrays (already label-encoded to 0..n-1 for classification by
        the caller, matching the rest of the Quail pipeline).
    cat_features : list of str
        Names of the categorical columns in ``X_train`` (passed to CatBoost's
        ``Pool`` so it can apply its native categorical handling).
    task : str
        "classification" or "regression".
    iterations, learning_rate, depth, l2_leaf_reg, random_strength,
    bagging_temperature, border_count, min_data_in_leaf, grow_policy,
    early_stopping_rounds : CatBoost hyperparameters
        See CatBoost documentation for details.
    random_seed : int, default=42
    thread_count : int, default=-1
        Number of CPU threads (-1 = use all available cores).
    verbose : int, default=0
        CatBoost's own verbosity (0 = silent).

    Returns
    -------
    model : CatBoostClassifier or CatBoostRegressor
        The fitted model (best iteration restored via ``use_best_model``).
    history : dict
        Single-entry-per-key metrics dict shaped like the histories produced
        by ``quail.training.fit`` (train/val/test loss + task metrics), so
        it flows through the same downstream aggregation/reporting code used
        for the linear/MLP baselines.
    """
    cat_features = [c for c in cat_features if c in X_train.columns]

    is_classification = task == "classification"
    all_labels = [y_train, y_val] + ([y_test] if y_test is not None else [])
    num_classes = len(np.unique(np.concatenate(all_labels))) if is_classification else None

    common_kwargs = dict(
        iterations=iterations,
        learning_rate=learning_rate,
        depth=depth,
        l2_leaf_reg=l2_leaf_reg,
        random_strength=random_strength,
        bagging_temperature=bagging_temperature,
        border_count=border_count,
        min_data_in_leaf=min_data_in_leaf,
        grow_policy=grow_policy,
        random_seed=random_seed,
        thread_count=thread_count,
        verbose=verbose,
        allow_writing_files=False,
    )

    if is_classification:
        loss_function = "Logloss" if num_classes == 2 else "MultiClass"
        model = CatBoostClassifier(loss_function=loss_function, **common_kwargs)
    else:
        model = CatBoostRegressor(loss_function="RMSE", **common_kwargs)

    train_pool = _make_pool(X_train, cat_features, y_train)
    val_pool = _make_pool(X_val, cat_features, y_val)

    model.fit(
        train_pool,
        eval_set=val_pool,
        use_best_model=True,
        early_stopping_rounds=early_stopping_rounds,
        verbose=verbose,
    )

    def _eval_split(X_split: pd.DataFrame, y_split: np.ndarray) -> Tuple[float, Dict[str, float]]:
        pool = _make_pool(X_split, cat_features)
        if is_classification:
            probs = model.predict_proba(pool)
            preds = np.argmax(probs, axis=1)
            all_probs_for_auc = probs[:, 1] if num_classes == 2 else probs
            loss = log_loss(y_split, probs, labels=list(range(num_classes)))
            metrics = _compute_metrics(
                np.asarray(y_split), preds, task, num_classes, all_probabilities=all_probs_for_auc
            )
        else:
            preds = np.asarray(model.predict(pool)).reshape(-1)
            loss = mean_squared_error(y_split, preds)
            metrics = _compute_metrics(np.asarray(y_split), preds, task, num_classes)
        return float(loss), metrics

    train_loss, train_metrics = _eval_split(X_train, y_train)
    val_loss, val_metrics = _eval_split(X_val, y_val)

    history: Dict[str, List[float]] = {
        "train_loss": [train_loss],
        "val_loss": [val_loss],
        "learning_rate": [learning_rate],
    }
    for name, value in train_metrics.items():
        history[f"train_{name}"] = [value]
    for name, value in val_metrics.items():
        history[f"val_{name}"] = [value]

    if X_test is not None and y_test is not None:
        test_loss, test_metrics = _eval_split(X_test, y_test)
        history["test_loss"] = [test_loss]
        for name, value in test_metrics.items():
            history[f"test_{name}"] = [value]

    history["n_iterations_trained"] = [model.tree_count_]

    return model, history
