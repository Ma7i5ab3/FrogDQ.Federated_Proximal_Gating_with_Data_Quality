"""
Data loading utilities for FrogDQ datasets.

This module provides functions to list and load preprocessed datasets with various
poisoning modes (clean, AR, NAR), as well as pre-computed AutoGluon features.
"""

from pathlib import Path
from typing import Dict, Literal, Optional, Tuple

import numpy as np
import pandas as pd

from frogdq.preprocessing import TabularPreprocessor


def get_datasets(data_dir: str = "data", poisoned_dir: str = "data_poisoned") -> pd.DataFrame:
    """
    List all available datasets from the data directories.

    This function scans both the clean data directory and the poisoned data directory
    to create a comprehensive summary of all available datasets with their statistics.

    Parameters
    ----------
    data_dir : str, default="data"
        Path to the directory containing clean CSV files.
    poisoned_dir : str, default="data_poisoned"
        Path to the directory containing poisoned data subdirectories (ar/, nar/, metrics/).

    Returns
    -------
    pd.DataFrame
        A dataframe with the following columns:
        - dataset_name: Name of the dataset (without extension)
        - uci_id: UCI repository ID extracted from filename (XXX from XXX_uciid_datasetname)
        - n_samples: Number of samples/rows in the dataset
        - n_features: Total number of features (excluding label)
        - n_numerical: Number of numerical features
        - n_categorical: Number of categorical features
        - n_classes: Number of unique classes (for classification) or 0 (for regression)
        - task_type: "classification" or "regression"
        - has_ar: Whether AR poisoned version exists
        - has_nar: Whether NAR poisoned version exists
        - file_size_mb: File size in megabytes

    Example
    -------
    >>> datasets = get_datasets()
    >>> print(datasets[['dataset_name', 'uci_id', 'n_samples', 'n_features']])
    """
    data_path = Path(data_dir)
    poisoned_path = Path(poisoned_dir)

    if not data_path.exists():
        raise ValueError(f"Data directory not found: {data_dir}")

    csv_files = sorted(data_path.glob("*.csv"))

    if not csv_files:
        raise ValueError(f"No CSV files found in {data_dir}")

    results = []

    for csv_file in csv_files:
        try:
            # Parse filename: format is XXX_YYY_dataset_name.csv
            # where XXX is our index and YYY is UCI ID
            filename = csv_file.stem
            parts = filename.split("_", 2)  # Split into at most 3 parts

            if len(parts) >= 2:
                uci_id = parts[1]  # e.g., "053"
                dataset_name = parts[2] if len(parts) > 2 else filename  # e.g., "iris"
            else:
                # Fallback if filename doesn't match expected format
                uci_id = "000"
                dataset_name = filename

            # Load dataset to extract statistics
            df = pd.read_csv(csv_file, na_values=["?", "NA", "N/A", "NaN", "nan", "NAN", "", " "])

            n_samples = len(df)
            n_features = len(df.columns)

            # Identify feature types by prefix
            numerical_features = [c for c in df.columns if c.startswith("num_")]
            categorical_features = [c for c in df.columns if c.startswith("cat_")]
            cls_labels = [c for c in df.columns if c.startswith("cls_")]
            reg_labels = [c for c in df.columns if c.startswith("reg_")]

            # Determine task type and number of classes
            if cls_labels:
                task_type = "classification"
                n_classes = df[cls_labels[0]].nunique()
            elif reg_labels:
                task_type = "regression"
                n_classes = 0
            else:
                task_type = "unknown"
                n_classes = 0

            # Exclude label from feature count
            n_features_excluding_label = n_features - len(cls_labels) - len(reg_labels)

            # Check if poisoned versions exist
            has_ar = False
            has_nar = False
            if poisoned_path.exists():
                ar_file = poisoned_path / "ar" / csv_file.name
                nar_file = poisoned_path / "nar" / csv_file.name
                has_ar = ar_file.exists()
                has_nar = nar_file.exists()

            # File size in MB
            file_size_mb = csv_file.stat().st_size / (1024 * 1024)

            results.append(
                {
                    "dataset_name": dataset_name,
                    "uci_id": uci_id,
                    "n_samples": n_samples,
                    "n_features": n_features_excluding_label,
                    "n_numerical": len(numerical_features),
                    "n_categorical": len(categorical_features),
                    "n_classes": n_classes,
                    "task_type": task_type,
                    "has_ar": has_ar,
                    "has_nar": has_nar,
                    "file_size_mb": round(file_size_mb, 3),
                }
            )

        except Exception as e:
            print(f"Warning: Error processing {csv_file.name}: {e}")
            continue

    return pd.DataFrame(results)


def load_data(
    dataset_name: str,
    mode: Literal["clean", "ar", "nar"] = "clean",
    seed: int = 42,
    clean_val: bool = False,
    clean_test: bool = True,
    data_dir: str = "data",
    poisoned_dir: str = "data_poisoned",
    test_dir: str = "data_poisoned",
    val_size: float = 0.2,
    test_sample_size: float = 0.8,
    poison_test_size: float = 0.3,
    **preprocessor_kwargs,
) -> Tuple[
    Tuple[np.ndarray, np.ndarray, np.ndarray],
    Tuple[np.ndarray, np.ndarray, np.ndarray],
    TabularPreprocessor,
    Dict,
]:
    """
    Load and preprocess a dataset with optional data quality issues.

    This function loads a dataset in clean, AR (At Random), or NAR (Not At Random) mode,
    applies preprocessing with TabularPreprocessor, and returns train/val/test splits
    along with metadata about data quality.

    Parameters
    ----------
    dataset_name : str
        Name of the dataset to load (e.g., "iris", "wine", "adult").
        This should match the dataset name from the filename (after the UCI ID).
    mode : {"clean", "ar", "nar"}, default="clean"
        Data quality mode:
        - "clean": Original clean data
        - "ar": At Random poisoning (random noise/missing values)
        - "nar": Not At Random poisoning (systematic/correlated errors)
    seed : int, default=42
        Random seed for reproducible train/val/test splits.
    clean_val : bool, default=False
        If True, use clean data for validation set (even when mode is ar/nar).
    clean_test : bool, default=True
        Kept for API compatibility; the test set is always the clean split
        saved in ``test_dir/test/``, so this flag has no effect.
    data_dir : str, default="data"
        Path to the directory containing clean CSV files.
    poisoned_dir : str, default="data_poisoned"
        Path to the directory containing poisoned data subdirectories
        (``ar/``, ``nar/``).  Train and val splits are read from here for
        ar/nar modes.
    test_dir : str, default="data_poisoned"
        Root directory that contains the ``test/`` subdirectory produced by
        ``poison_data.py``.  Always use ``"data_poisoned"`` unless you have
        a custom setup.
    val_size : float, default=0.2
        Fraction of the train+val data to use for validation.
    test_sample_size : float, default=0.8
        Fraction of ``test_dir/test/`` to use as the final test set
        (sampled with ``random_state=seed`` for reproducibility).
    poison_test_size : float, default=0.3
        Fraction that was held out as test when running ``poison_data.py``
        (default ``--test-size 0.3``).  Used in clean mode to exclude those
        same rows from the train+val pool, preventing leakage.
    **preprocessor_kwargs
        Additional keyword arguments passed to TabularPreprocessor.

    Returns
    -------
    splits : tuple of (X_train, X_val, X_test)
        Preprocessed feature arrays as numpy arrays.
    labels : tuple of (y_train, y_val, y_test)
        Label arrays as numpy arrays.
    preprocessor : TabularPreprocessor
        Fitted preprocessor object for inverse transforms or additional processing.
    metadata : dict
        Dictionary containing:
        - "mask_train": Boolean mask for training data (True = poisoned/dirty cell)
        - "mask_val": Boolean mask for validation data
        - "mask_test": Boolean mask for test data
        - "sample_quality_train": Per-sample percentage of clean cells (0-100)
        - "sample_quality_val": Per-sample percentage of clean cells
        - "sample_quality_test": Per-sample percentage of clean cells
        - "feature_quality": Per-feature percentage of clean cells after preprocessing
                            (dict mapping feature name to percentage)
        - "overall_quality": Overall percentage of clean cells
        - "mode": Data quality mode used
        - "dataset_name": Name of the dataset

    Raises
    ------
    ValueError
        If dataset is not found or mode is invalid.
    FileNotFoundError
        If required data files do not exist.

    Example
    -------
    >>> # Load clean iris dataset
    >>> (X_train, X_val, X_test), (y_train, y_val, y_test), prep, meta = load_data("iris")
    >>> print(f"Train shape: {X_train.shape}")
    >>> print(f"Overall quality: {meta['overall_quality']:.2f}%")

    >>> # Load with AR poisoning, but clean validation/test
    >>> splits, labels, prep, meta = load_data("iris", mode="ar", clean_val=True, clean_test=True)

    >>> # Access per-sample quality scores
    >>> print(f"Sample quality (train): {meta['sample_quality_train'][:5]}")
    """
    if mode not in ["clean", "ar", "nar"]:
        raise ValueError(f"Invalid mode: {mode}. Must be 'clean', 'ar', or 'nar'.")

    data_path = Path(data_dir)
    poisoned_path = Path(poisoned_dir)
    test_path = Path(test_dir) / "test"

    # Find the dataset file by matching the dataset name
    csv_files = list(data_path.glob(f"*_{dataset_name}.csv"))

    if not csv_files:
        raise FileNotFoundError(
            f"Dataset '{dataset_name}' not found in {data_dir}. "
            f"Use get_datasets() to see available datasets."
        )

    if len(csv_files) > 1:
        raise ValueError(
            f"Multiple files found for dataset '{dataset_name}': {[f.name for f in csv_files]}"
        )

    csv_file = csv_files[0]
    dataset_filename = csv_file.name

    # Load train+val source data.
    # For clean mode: use data_dir but exclude the rows held out as test by
    # poison_data.py (same frac/seed), so there is no leakage with the test set.
    # For ar/nar: the file in poisoned_dir already contains only the train portion.
    if mode == "clean":
        full_df = pd.read_csv(csv_file, na_values=["?", "NA", "N/A", "NaN", "nan", "NAN", "", " "])
        test_rows = full_df.sample(frac=poison_test_size, random_state=42)
        df = full_df.drop(test_rows.index).reset_index(drop=True)
        mask_df = pd.DataFrame(False, index=df.index, columns=df.columns, dtype=bool)
        clean_trainval_df = None
    else:
        poisoned_file = poisoned_path / mode / dataset_filename
        mask_file = poisoned_path / mode / f"{csv_file.stem}_mask.csv"

        if not poisoned_file.exists():
            raise FileNotFoundError(
                f"Poisoned data file not found: {poisoned_file}. "
                f"Run the poison_data.py script first."
            )
        if not mask_file.exists():
            raise FileNotFoundError(f"Mask file not found: {mask_file}")

        df = pd.read_csv(poisoned_file, na_values=["?", "NA", "N/A", "NaN", "nan", "NAN", "", " "])
        mask_df = pd.read_csv(mask_file).astype(bool)

        # Build clean train+val counterpart (same rows as poisoned file) for clean_val.
        clean_trainval_df = None
        if clean_val:
            full_clean_df = pd.read_csv(
                csv_file, na_values=["?", "NA", "N/A", "NaN", "nan", "NAN", "", " "]
            )
            test_rows = full_clean_df.sample(frac=poison_test_size, random_state=42)
            clean_trainval_df = full_clean_df.drop(test_rows.index).reset_index(drop=True)

    # Load test set: always from test_dir/test/, sample test_sample_size fraction.
    test_file = test_path / dataset_filename
    if not test_file.exists():
        raise FileNotFoundError(
            f"Test file not found: {test_file}. "
            f"Run scripts/poison_data.py first."
        )
    df_test_full = pd.read_csv(
        test_file, na_values=["?", "NA", "N/A", "NaN", "nan", "NAN", "", " "]
    )
    df_test = df_test_full.sample(frac=test_sample_size, random_state=seed).reset_index(drop=True)

    # Detect label column
    label_cols = [col for col in df.columns if col.startswith(("cls_", "reg_"))]
    if not label_cols:
        raise ValueError(f"No label column found in dataset {dataset_name}")

    label_col = label_cols[0]
    task_type = "classification" if label_col.startswith("cls_") else "regression"

    # Split train+val into train / val
    from sklearn.model_selection import train_test_split

    indices = np.arange(len(df))
    y_full = df[label_col].values
    stratify_split = y_full if task_type == "classification" else None

    train_idx, val_idx = train_test_split(
        indices, test_size=val_size, random_state=seed, stratify=stratify_split
    )

    # Build split dataframes
    df_train = df.iloc[train_idx].reset_index(drop=True)
    df_val = df.iloc[val_idx].reset_index(drop=True)

    mask_train = mask_df.iloc[train_idx].reset_index(drop=True)
    mask_val = mask_df.iloc[val_idx].reset_index(drop=True)
    # Test is always the clean held-out split, so mask is all False
    mask_test = pd.DataFrame(False, index=df_test.index, columns=df_test.columns, dtype=bool)

    # Replace val with clean data if requested (ar/nar modes only)
    if clean_val and clean_trainval_df is not None:
        df_val = clean_trainval_df.iloc[val_idx].reset_index(drop=True)
        mask_val = pd.DataFrame(False, index=df_val.index, columns=df_val.columns, dtype=bool)

    # Initialize and fit preprocessor on training data only
    preprocessor = TabularPreprocessor(
        random_state=seed,
        test_size=0.0,  # We handle splitting ourselves
        val_size=0.0,  # We handle splitting ourselves
        **preprocessor_kwargs,
    )

    # Fit on training data
    X_train_features = df_train.drop(columns=[label_col])
    y_train = df_train[label_col].values

    preprocessor.fit(X_train_features, y_train)

    # Transform all splits
    X_train = preprocessor.transform(X_train_features)
    X_val = preprocessor.transform(df_val.drop(columns=[label_col]))
    X_test = preprocessor.transform(df_test.drop(columns=[label_col]))

    y_val = df_val[label_col].values
    y_test = df_test[label_col].values

    # Compute quality metrics at sample level (before preprocessing)
    # Sample quality = percentage of clean cells per row (excluding label)
    feature_cols = [col for col in df.columns if col != label_col]

    def compute_sample_quality(mask: pd.DataFrame) -> np.ndarray:
        """Compute per-sample quality (% of clean cells)."""
        mask_features = mask[feature_cols]
        dirty_per_sample = mask_features.sum(axis=1).values
        total_cells_per_sample = len(feature_cols)
        clean_per_sample = total_cells_per_sample - dirty_per_sample
        quality_pct = (clean_per_sample / total_cells_per_sample) * 100
        return quality_pct

    sample_quality_train = compute_sample_quality(mask_train)
    sample_quality_val = compute_sample_quality(mask_val)
    sample_quality_test = compute_sample_quality(mask_test)

    # Compute feature-level quality AFTER preprocessing
    # This is more complex because of one-hot encoding
    feature_names_out = preprocessor.get_feature_names_out()

    # Map original features to transformed features
    feature_quality = {}

    # For numerical features, mapping is 1:1
    for num_feat in preprocessor.numerical_features_:
        if num_feat in feature_cols:
            # Calculate quality for this feature across all splits (or just train)
            mask_col_train = mask_train[num_feat].values
            quality_pct = (1 - mask_col_train.sum() / len(mask_col_train)) * 100
            feature_quality[num_feat] = quality_pct

    # For categorical features, the percentage applies to ALL one-hot encoded columns
    for cat_feat in preprocessor.categorical_features_:
        if cat_feat in feature_cols:
            mask_col_train = mask_train[cat_feat].values
            quality_pct = (1 - mask_col_train.sum() / len(mask_col_train)) * 100

            # Find all one-hot encoded columns for this feature
            # They follow the pattern: cat_feat_value
            for out_feat in feature_names_out:
                if out_feat.startswith(f"{cat_feat}_"):
                    feature_quality[out_feat] = quality_pct

    # Overall quality (percentage of clean cells across all data)
    all_masks = pd.concat([mask_train, mask_val, mask_test], ignore_index=True)
    all_masks_features = all_masks[feature_cols]
    total_dirty = all_masks_features.sum().sum()
    total_cells = all_masks_features.size
    overall_quality = (1 - total_dirty / total_cells) * 100

    # Prepare metadata dictionary
    metadata = {
        "mask_train": mask_train[feature_cols].values,
        "mask_val": mask_val[feature_cols].values,
        "mask_test": mask_test[feature_cols].values,
        "sample_quality_train": sample_quality_train,
        "sample_quality_val": sample_quality_val,
        "sample_quality_test": sample_quality_test,
        "feature_quality": feature_quality,
        "overall_quality": overall_quality,
        "mode": mode,
        "dataset_name": dataset_name,
        "task_type": task_type,
        "n_samples": {"train": len(X_train), "val": len(X_val), "test": len(X_test)},
        "n_features_raw": len(feature_cols),
        "n_features_preprocessed": X_train.shape[1],
    }

    return (X_train, X_val, X_test), (y_train, y_val, y_test), preprocessor, metadata


def load_raw_data(
    dataset_name: str,
    mode: Literal["clean", "ar", "nar"] = "clean",
    seed: int = 42,
    clean_val: bool = False,
    clean_test: bool = True,
    data_dir: str = "data",
    poisoned_dir: str = "data_poisoned",
    test_dir: str = "data_poisoned",
    val_size: float = 0.2,
    test_sample_size: float = 0.8,
    poison_test_size: float = 0.3,
) -> Tuple[
    Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame],
    Tuple[np.ndarray, np.ndarray, np.ndarray],
    None,
    Dict,
]:
    """
    Load a dataset split with NO preprocessing applied.

    Unlike ``load_data``, this skips ``TabularPreprocessor`` entirely: numerical
    features keep their original values with NaN untouched (no imputation, no
    scaling), and categorical features keep their original raw values with NaN
    preserved (no imputation, no one-hot encoding). Intended for models that
    natively handle missing values and categorical features (e.g. CatBoost).

    File loading, poisoning-mode handling, and the train/val/test split logic
    mirror ``load_data`` exactly, so the same rows end up in the same splits
    for a given ``seed``.

    Parameters
    ----------
    Same as ``load_data`` (minus ``**preprocessor_kwargs``, which does not
    apply here since no preprocessor is fitted).

    Returns
    -------
    splits : tuple of (X_train, X_val, X_test) pandas DataFrames
        Raw, unprocessed feature frames.
    labels : tuple of (y_train, y_val, y_test) numpy arrays
    preprocessor : None
        No preprocessor is fitted; kept for API symmetry with ``load_data``.
    metadata : dict
        Dictionary containing ``task_type``, ``mode``, ``dataset_name``,
        ``categorical_features`` (raw column names, cast to string with NaN
        preserved), ``numerical_features``, and ``n_samples``.
    """
    if mode not in ["clean", "ar", "nar"]:
        raise ValueError(f"Invalid mode: {mode}. Must be 'clean', 'ar', or 'nar'.")

    data_path = Path(data_dir)
    poisoned_path = Path(poisoned_dir)
    test_path = Path(test_dir) / "test"

    csv_files = list(data_path.glob(f"*_{dataset_name}.csv"))
    if not csv_files:
        raise FileNotFoundError(
            f"Dataset '{dataset_name}' not found in {data_dir}. "
            f"Use get_datasets() to see available datasets."
        )
    if len(csv_files) > 1:
        raise ValueError(
            f"Multiple files found for dataset '{dataset_name}': {[f.name for f in csv_files]}"
        )

    csv_file = csv_files[0]
    dataset_filename = csv_file.name
    na_values = ["?", "NA", "N/A", "NaN", "nan", "NAN", "", " "]

    if mode == "clean":
        full_df = pd.read_csv(csv_file, na_values=na_values)
        test_rows = full_df.sample(frac=poison_test_size, random_state=42)
        df = full_df.drop(test_rows.index).reset_index(drop=True)
        clean_trainval_df = None
    else:
        poisoned_file = poisoned_path / mode / dataset_filename
        if not poisoned_file.exists():
            raise FileNotFoundError(
                f"Poisoned data file not found: {poisoned_file}. "
                f"Run the poison_data.py script first."
            )
        df = pd.read_csv(poisoned_file, na_values=na_values)

        clean_trainval_df = None
        if clean_val:
            full_clean_df = pd.read_csv(csv_file, na_values=na_values)
            test_rows = full_clean_df.sample(frac=poison_test_size, random_state=42)
            clean_trainval_df = full_clean_df.drop(test_rows.index).reset_index(drop=True)

    test_file = test_path / dataset_filename
    if not test_file.exists():
        raise FileNotFoundError(
            f"Test file not found: {test_file}. "
            f"Run scripts/poison_data.py first."
        )
    df_test_full = pd.read_csv(test_file, na_values=na_values)
    df_test = df_test_full.sample(frac=test_sample_size, random_state=seed).reset_index(drop=True)

    label_cols = [col for col in df.columns if col.startswith(("cls_", "reg_"))]
    if not label_cols:
        raise ValueError(f"No label column found in dataset {dataset_name}")
    label_col = label_cols[0]
    task_type = "classification" if label_col.startswith("cls_") else "regression"

    from sklearn.model_selection import train_test_split

    indices = np.arange(len(df))
    y_full = df[label_col].values
    stratify_split = y_full if task_type == "classification" else None
    train_idx, val_idx = train_test_split(
        indices, test_size=val_size, random_state=seed, stratify=stratify_split
    )

    df_train = df.iloc[train_idx].reset_index(drop=True)
    df_val = df.iloc[val_idx].reset_index(drop=True)

    if clean_val and clean_trainval_df is not None:
        df_val = clean_trainval_df.iloc[val_idx].reset_index(drop=True)

    # Feature type detection mirrors TabularPreprocessor's prefix convention,
    # but nothing is imputed/scaled/encoded — values are only cast to a
    # consistent dtype so downstream libraries can recognize the column kind.
    feature_cols = [c for c in df.columns if c != label_col]
    numerical_features = [c for c in feature_cols if c.startswith("num_")]
    categorical_features = [c for c in feature_cols if c.startswith("cat_")]
    id_features = [c for c in feature_cols if c.startswith("id_")]
    # Anything without a recognized prefix (other than id_, which is dropped
    # since it's metadata, not a feature) is treated as categorical, kept raw.
    other_features = [
        c for c in feature_cols
        if c not in numerical_features and c not in categorical_features and c not in id_features
    ]
    categorical_features = categorical_features + other_features

    def _to_raw_features(frame: pd.DataFrame) -> pd.DataFrame:
        X = frame[feature_cols].drop(columns=id_features, errors="ignore").copy()
        for col in categorical_features:
            if col in X.columns:
                # dtype-only cast (object -> string) so the column is a
                # consistent type; NaN is preserved as NaN, not filled in.
                X[col] = X[col].astype(str).replace("nan", np.nan)
        return X

    X_train = _to_raw_features(df_train)
    X_val = _to_raw_features(df_val)
    X_test = _to_raw_features(df_test)

    y_train = df_train[label_col].values
    y_val = df_val[label_col].values
    y_test = df_test[label_col].values

    metadata = {
        "mode": mode,
        "dataset_name": dataset_name,
        "task_type": task_type,
        "categorical_features": [c for c in categorical_features if c in X_train.columns],
        "numerical_features": [c for c in numerical_features if c in X_train.columns],
        "n_samples": {"train": len(X_train), "val": len(X_val), "test": len(X_test)},
        "n_features_raw": len(feature_cols),
    }

    return (X_train, X_val, X_test), (y_train, y_val, y_test), None, metadata


def _apply_saga_no_outliers(df: pd.DataFrame, pipeline_info: dict) -> pd.DataFrame:
    """
    Apply a fitted Saga++ pipeline to a DataFrame, skipping outlier-detection
    and train-only primitives.

    Used to preprocess held-out (test) data using the same normalization and
    imputation steps that were fitted on the training partition, without
    modifying valid test values via outlier removal.
    """
    _OUTLIER = {"outlierByIQR", "outlierBySd", "winsorize"}
    _TRAIN_ONLY = {"underSampling", "abstain"}
    _SKIP = _OUTLIER | _TRAIN_ONLY

    pipeline = pipeline_info.get("pipeline", [])
    states = pipeline_info.get("states", {})
    col_types = pipeline_info.get("col_types", {})

    X = df.copy()
    for name, _ in pipeline:
        if name in _SKIP:
            continue
        state = states.get(name, {})

        if name in ("imputeByMean", "imputeByMedian"):
            for col, val in state.items():
                if col in X.columns:
                    X[col] = X[col].astype(float, errors="ignore").fillna(val)

        elif name == "fillDefault":
            for col, val in state.get("num_fills", {}).items():
                if col in X.columns:
                    X[col] = X[col].astype(float, errors="ignore").fillna(val)
            for col, val in state.get("cat_fills", {}).items():
                if col in X.columns:
                    X[col] = X[col].fillna(val)

        elif name == "fillForward":
            num_cols = col_types.get("numerical", [])
            cat_cols = col_types.get("categorical", [])
            for col in num_cols + cat_cols:
                if col not in X.columns:
                    continue
                X[col] = X[col].ffill().bfill()
                last_val = state.get(col)
                if X[col].isna().any() and last_val is not None and not (
                    isinstance(last_val, float) and np.isnan(last_val)
                ):
                    X[col] = X[col].fillna(last_val)

        elif name == "normalize":
            for col, stats in state.items():
                if col in X.columns:
                    X[col] = X[col].astype(float, errors="ignore")
                    X[col] = (X[col] - stats["mean"]) / stats["std"]

        elif name == "correctTypos":
            for col, col_state in state.items():
                if col not in X.columns:
                    continue
                valid = col_state.get("valid", set())
                mode_val = col_state.get("mode")
                if mode_val is not None:
                    mask = X[col].notna() & ~X[col].astype(str).isin(valid)
                    if mask.any():
                        X.loc[mask, col] = mode_val

    return X


def load_saga_data(
    dataset_name: str,
    mode: str,
    seed: int = 42,
    saga_dir: str = "data_cleaned_saga",
    clean_val: bool = True,
    clean_test: bool = True,
    data_dir: str = "data",
    poisoned_dir: str = "data_poisoned",
    val_size: float = 0.2,
    test_sample_size: float = 0.8,
    poison_test_size: float = 0.3,
    **preprocessor_kwargs,
) -> Tuple[
    Tuple[np.ndarray, np.ndarray, np.ndarray],
    Tuple[np.ndarray, np.ndarray, np.ndarray],
    TabularPreprocessor,
    Dict,
]:
    """
    Load Saga++-cleaned data for a dataset split.

    Train and validation splits are read from the Saga++-cleaned CSV saved by
    ``scripts/saga.py`` at ``saga_dir/{mode}/``.  The test split is an 80 %
    random sample from ``data_poisoned/test/``; the fitted Saga++ pipeline
    (excluding outlier-detection steps) is applied to it before
    TabularPreprocessor transforms the data.

    Parameters
    ----------
    dataset_name : str
        Name of the dataset (e.g. "iris").
    mode : str
        Poisoning mode ("ar" or "nar").
    seed : int, default=42
        Random seed for the train/val/test split.
    saga_dir : str, default="data_cleaned_saga"
        Root directory written by ``scripts/saga.py``.
    clean_val : bool, default=True
        If True, replace the validation split with clean (unpoisoned) rows.
    clean_test : bool, default=True
        Kept for API compatibility; has no effect — the test set always comes
        from ``data_poisoned/test/`` and has the Saga++ pipeline applied.
    data_dir : str, default="data"
        Directory containing the original clean CSV files (used for clean_val).
    poisoned_dir : str, default="data_poisoned"
        Root directory that contains the ``test/`` sub-directory.
    val_size : float, default=0.2
        Fraction of the train+val partition to use as validation.
    test_sample_size : float, default=0.8
        Fraction of ``poisoned_dir/test/`` to sample as the final test set.
    poison_test_size : float, default=0.4
        Fraction held out as test when running ``poison_data.py`` (used to
        exclude those rows from the clean train+val pool when clean_val=True).

    Returns
    -------
    Same structure as ``load_data``.

    Raises
    ------
    FileNotFoundError
        If the Saga-cleaned file does not exist; run ``scripts/saga.py`` first.
    """
    import pickle

    data_path = Path(data_dir)
    saga_model_path = Path(saga_dir) / mode
    test_path = Path(poisoned_dir) / "test"

    # Locate dataset file by matching name suffix
    csv_files = list(data_path.glob(f"*_{dataset_name}.csv"))
    if not csv_files:
        raise FileNotFoundError(
            f"Dataset '{dataset_name}' not found in {data_dir}. "
            f"Use get_datasets() to see available datasets."
        )
    if len(csv_files) > 1:
        raise ValueError(
            f"Multiple files found for dataset '{dataset_name}': {[f.name for f in csv_files]}"
        )
    csv_file = csv_files[0]
    dataset_filename = csv_file.name

    # ---- Load Saga++-cleaned train+val partition ----
    saga_csv = saga_model_path / dataset_filename
    saga_mask_file = saga_model_path / f"{csv_file.stem}_mask.csv"
    pipeline_pkl = saga_model_path / f"{csv_file.stem}_pipeline.pkl"

    if not saga_csv.exists():
        raise FileNotFoundError(
            f"Saga-cleaned data not found: {saga_csv}. "
            "Run 'python scripts/saga.py' first."
        )

    df = pd.read_csv(saga_csv, na_values=["?", "NA", "N/A", "NaN", "nan", "NAN", "", " "])
    if saga_mask_file.exists():
        mask_df = pd.read_csv(saga_mask_file).astype(bool)
    else:
        mask_df = pd.DataFrame(False, index=df.index, columns=df.columns, dtype=bool)

    # Load fitted pipeline info for test-set preprocessing
    pipeline_info: Optional[dict] = None
    if pipeline_pkl.exists():
        with open(pipeline_pkl, "rb") as f:
            pipeline_info = pickle.load(f)

    # ---- Load test set: 80 % sample of data_poisoned/test/ ----
    test_file = test_path / dataset_filename
    if not test_file.exists():
        raise FileNotFoundError(
            f"Test file not found: {test_file}. "
            "Run scripts/poison_data.py first."
        )
    df_test_full = pd.read_csv(
        test_file, na_values=["?", "NA", "N/A", "NaN", "nan", "NAN", "", " "]
    )
    df_test_raw = df_test_full.sample(frac=test_sample_size, random_state=seed).reset_index(drop=True)

    # Apply Saga++ pipeline to test (skip outlier detection)
    if pipeline_info is not None:
        df_test = _apply_saga_no_outliers(df_test_raw, pipeline_info)
    else:
        df_test = df_test_raw.copy()

    # ---- Detect label column ----
    label_cols = [col for col in df.columns if col.startswith(("cls_", "reg_"))]
    if not label_cols:
        raise ValueError(f"No label column found in dataset {dataset_name}")
    label_col = label_cols[0]
    task_type = "classification" if label_col.startswith("cls_") else "regression"

    # ---- Split train+val ----
    from sklearn.model_selection import train_test_split

    indices = np.arange(len(df))
    y_full = df[label_col].values
    stratify_split = y_full if task_type == "classification" else None
    train_idx, val_idx = train_test_split(
        indices, test_size=val_size, random_state=seed, stratify=stratify_split
    )

    df_train = df.iloc[train_idx].reset_index(drop=True)
    df_val = df.iloc[val_idx].reset_index(drop=True)
    mask_train = mask_df.iloc[train_idx].reset_index(drop=True)
    mask_val = mask_df.iloc[val_idx].reset_index(drop=True)
    mask_test = pd.DataFrame(False, index=df_test.index, columns=df_test.columns, dtype=bool)

    # Replace val with clean rows when requested
    if clean_val:
        full_clean_df = pd.read_csv(
            csv_file, na_values=["?", "NA", "N/A", "NaN", "nan", "NAN", "", " "]
        )
        test_rows = full_clean_df.sample(frac=poison_test_size, random_state=42)
        clean_trainval_df = full_clean_df.drop(test_rows.index).reset_index(drop=True)
        if len(clean_trainval_df) == len(df):
            df_val = clean_trainval_df.iloc[val_idx].reset_index(drop=True)
            mask_val = pd.DataFrame(False, index=df_val.index, columns=df_val.columns, dtype=bool)

    # ---- TabularPreprocessor ----
    preprocessor = TabularPreprocessor(
        random_state=seed,
        test_size=0.0,
        val_size=0.0,
        **preprocessor_kwargs,
    )
    X_train_features = df_train.drop(columns=[label_col])
    y_train = df_train[label_col].values
    preprocessor.fit(X_train_features, y_train)

    X_train = preprocessor.transform(X_train_features)
    X_val = preprocessor.transform(df_val.drop(columns=[label_col]))
    X_test = preprocessor.transform(df_test.drop(columns=[label_col]))

    y_val = df_val[label_col].values
    y_test = df_test_raw[label_col].values  # labels are never transformed

    # ---- Quality metrics ----
    feature_cols = [col for col in df.columns if col != label_col]

    def _sample_quality(mask: pd.DataFrame) -> np.ndarray:
        cols = [c for c in feature_cols if c in mask.columns]
        dirty = mask[cols].sum(axis=1).values
        return (len(cols) - dirty) / max(len(cols), 1) * 100

    sample_quality_train = _sample_quality(mask_train)
    sample_quality_val = _sample_quality(mask_val)
    sample_quality_test = np.full(len(df_test), 100.0)

    feature_names_out = preprocessor.get_feature_names_out()
    feature_quality: Dict = {}
    for num_feat in getattr(preprocessor, "numerical_features_", []):
        if num_feat in feature_cols and num_feat in mask_train.columns:
            col_mask = mask_train[num_feat].values
            feature_quality[num_feat] = (1 - col_mask.sum() / max(len(col_mask), 1)) * 100
    for cat_feat in getattr(preprocessor, "categorical_features_", []):
        if cat_feat in feature_cols and cat_feat in mask_train.columns:
            col_mask = mask_train[cat_feat].values
            q = (1 - col_mask.sum() / max(len(col_mask), 1)) * 100
            for out_feat in feature_names_out:
                if out_feat.startswith(f"{cat_feat}_"):
                    feature_quality[out_feat] = q

    feat_cols_present = [c for c in feature_cols if c in mask_train.columns]
    all_masks = pd.concat([mask_train[feat_cols_present], mask_val[feat_cols_present]], ignore_index=True)
    total_dirty = all_masks.sum().sum()
    total_cells = all_masks.size
    overall_quality = (1 - total_dirty / max(total_cells, 1)) * 100

    metadata = {
        "mask_train": mask_train[feat_cols_present].values if feat_cols_present else np.zeros((len(df_train), 0), dtype=bool),
        "mask_val": mask_val[feat_cols_present].values if feat_cols_present else np.zeros((len(df_val), 0), dtype=bool),
        "mask_test": np.zeros((len(df_test), len(feat_cols_present)), dtype=bool),
        "sample_quality_train": sample_quality_train,
        "sample_quality_val": sample_quality_val,
        "sample_quality_test": sample_quality_test,
        "feature_quality": feature_quality,
        "overall_quality": overall_quality,
        "mode": mode,
        "dataset_name": dataset_name,
        "task_type": task_type,
        "n_samples": {"train": len(X_train), "val": len(X_val), "test": len(X_test)},
        "n_features_raw": len(feature_cols),
        "n_features_preprocessed": X_train.shape[1],
    }

    return (X_train, X_val, X_test), (y_train, y_val, y_test), preprocessor, metadata


def load_cp_data(
    dataset_name: str,
    mode: str,
    seed: int = 42,
    cp_dir: str = "data_cleaned_cp",
    clean_val: bool = True,
    clean_test: bool = True,
    data_dir: str = "data",
    poisoned_dir: str = "data_poisoned",
    **kwargs,
) -> Tuple[
    Tuple[np.ndarray, np.ndarray, np.ndarray],
    Tuple[np.ndarray, np.ndarray, np.ndarray],
    "TabularPreprocessor",
    Dict,
]:
    """
    Load data cleaned by the custom pipeline (scripts/data_preparation_pipeline.py).

    The CP pipeline applies MICE imputation, IQR outlier clipping, association-rule
    categorical repair, and OOV repair, writing results to ``data_cleaned_cp/{ar,nar}/``.
    The output format is identical to Saga's, so this is a thin wrapper over
    ``load_data`` that redirects ``poisoned_dir`` to the CP output directory.

    Parameters
    ----------
    dataset_name : str
        Name of the dataset (e.g. "iris").
    mode : str
        Data quality mode ("ar" or "nar").
    seed : int, default=42
        Random seed for the train/val/test split.
    cp_dir : str, default="data_cleaned_cp"
        Root directory containing the CP-cleaned CSVs and masks.
    clean_val : bool, default=True
        If True, return the clean validation split.
    clean_test : bool, default=True
        If True, return the clean test split.
    **kwargs
        Additional keyword arguments forwarded to ``load_data``.

    Returns
    -------
    Same as ``load_data``.

    Raises
    ------
    FileNotFoundError
        If the CP-cleaned file does not exist; run
        ``scripts/data_preparation_pipeline.py`` first.
    """
    return load_data(
        dataset_name=dataset_name,
        mode=mode,
        seed=seed,
        clean_val=clean_val,
        clean_test=clean_test,
        data_dir=data_dir,
        poisoned_dir=cp_dir,
        test_dir=poisoned_dir,
        **kwargs,
    )


def load_baseline_zero_data(
    dataset_name: str,
    mode: str,
    seed: int = 42,
    baseline_zero_dir: str = "data_baseline_zero",
    clean_val: bool = True,
    clean_test: bool = True,
    data_dir: str = "data",
    poisoned_dir: str = "data_poisoned",
    **kwargs,
) -> Tuple[
    Tuple[np.ndarray, np.ndarray, np.ndarray],
    Tuple[np.ndarray, np.ndarray, np.ndarray],
    "TabularPreprocessor",
    Dict,
]:
    """
    Load data imputed by Baseline 0 (scripts/baseline_zero.py).

    Baseline 0 fills numerical NaN with 0 and categorical NaN with a random
    observed value.  The output format is identical to the CP pipeline, so this
    is a thin wrapper over ``load_data`` that redirects ``poisoned_dir`` to the
    Baseline-0 output directory.

    Parameters
    ----------
    dataset_name : str
        Name of the dataset (e.g. "iris").
    mode : str
        Data quality mode ("ar" or "nar").
    seed : int, default=42
        Random seed for the train/val/test split.
    baseline_zero_dir : str, default="data_baseline_zero"
        Root directory containing the Baseline-0 CSVs and masks.
    clean_val : bool, default=True
        If True, return the clean validation split.
    clean_test : bool, default=True
        If True, return the clean test split.
    **kwargs
        Additional keyword arguments forwarded to ``load_data``.

    Returns
    -------
    Same as ``load_data``.

    Raises
    ------
    FileNotFoundError
        If the Baseline-0 file does not exist; run
        ``scripts/baseline_zero.py`` first.
    """
    return load_data(
        dataset_name=dataset_name,
        mode=mode,
        seed=seed,
        clean_val=clean_val,
        clean_test=clean_test,
        data_dir=data_dir,
        poisoned_dir=baseline_zero_dir,
        test_dir=poisoned_dir,
        **kwargs,
    )


def load_knn_data(
    dataset_name: str,
    mode: str,
    seed: int = 42,
    knn_dir: str = "data_knn",
    clean_val: bool = True,
    clean_test: bool = True,
    data_dir: str = "data",
    poisoned_dir: str = "data_poisoned",
    **kwargs,
) -> Tuple[
    Tuple[np.ndarray, np.ndarray, np.ndarray],
    Tuple[np.ndarray, np.ndarray, np.ndarray],
    "TabularPreprocessor",
    Dict,
]:
    """
    Load data imputed by the KNN baseline (scripts/knn.py).

    KNN imputation uses ``sklearn.impute.KNNImputer`` on a joint numerical +
    ordinal-encoded categorical matrix.  The output format is identical to the
    CP pipeline, so this is a thin wrapper over ``load_data`` that redirects
    ``poisoned_dir`` to the KNN output directory.

    Parameters
    ----------
    dataset_name : str
        Name of the dataset (e.g. "iris").
    mode : str
        Data quality mode ("ar" or "nar").
    seed : int, default=42
        Random seed for the train/val/test split.
    knn_dir : str, default="data_knn"
        Root directory containing the KNN-imputed CSVs and masks.
    clean_val : bool, default=True
        If True, return the clean validation split.
    clean_test : bool, default=True
        If True, return the clean test split.
    **kwargs
        Additional keyword arguments forwarded to ``load_data``.

    Returns
    -------
    Same as ``load_data``.

    Raises
    ------
    FileNotFoundError
        If the KNN-imputed file does not exist; run ``scripts/knn.py`` first.
    """
    return load_data(
        dataset_name=dataset_name,
        mode=mode,
        seed=seed,
        clean_val=clean_val,
        clean_test=clean_test,
        data_dir=data_dir,
        poisoned_dir=knn_dir,
        test_dir=poisoned_dir,
        **kwargs,
    )


def load_autogluon_data(
    dataset_name: str,
    mode: str,
    model_type: str,
    seed: int,
    autogluon_dir: str = "data_autogluon",
    clean_val: bool = True,
    clean_test: bool = True,
) -> Tuple[
    Tuple[np.ndarray, np.ndarray, np.ndarray],
    Tuple[np.ndarray, np.ndarray, np.ndarray],
    Optional[TabularPreprocessor],
    Dict,
]:
    """
    Load pre-computed AutoGluon-transformed features for a dataset split.

    Features must have been generated by ``scripts/autogluon.py`` before calling
    this function.  No further preprocessing is applied; the arrays are used
    directly in Optuna experiments.

    Each NPZ file stores both the clean and the corrupted variant of the
    validation and test splits (``X_val_clean`` / ``X_val_corrupted`` and
    ``X_test_clean`` / ``X_test_corrupted``).  Use ``clean_val`` and
    ``clean_test`` to choose which variant is returned.

    Parameters
    ----------
    dataset_name : str
        Name of the dataset (e.g. "iris").
    mode : str
        Data quality mode used during AutoGluon fitting ("ar" or "nar").
    model_type : str
        Model type AutoGluon was tuned for ("linear" or "mlp").
    seed : int
        Random seed that determines the train/val/test split.
    autogluon_dir : str, default="data_autogluon"
        Root directory containing the pre-computed NPZ files.
    clean_val : bool, default=True
        If True, return the clean validation split; if False, return the
        corrupted (poisoned) validation split.
    clean_test : bool, default=True
        If True, return the clean test split; if False, return the corrupted
        (poisoned) test split.

    Returns
    -------
    splits : tuple of (X_train, X_val, X_test)
        AutoGluon-transformed feature arrays as float64 numpy arrays.
        X_val and X_test are the clean or corrupted variant according to
        ``clean_val`` / ``clean_test``.
    labels : tuple of (y_train, y_val, y_test)
        Label arrays (targets are never poisoned, so these are identical
        regardless of the clean/corrupted choice).
    preprocessor : None
        Always None – AutoGluon's pipeline is baked into the features.
    metadata : dict
        Dictionary with keys ``task_type``, ``mode``, ``dataset_name``,
        ``preparation``, ``clean_val``, ``clean_test``,
        ``sample_quality_train``, ``feature_quality``,
        ``n_samples``, and ``n_features_preprocessed``.

    Raises
    ------
    FileNotFoundError
        If the NPZ file does not exist; run ``scripts/autogluon.py`` first.
    """
    out_file = Path(autogluon_dir) / mode / model_type / dataset_name / f"seed_{seed}.npz"

    if not out_file.exists():
        # Folder may include a numeric prefix (e.g. "00003_kr_vs_kp"); match by substring
        parent = Path(autogluon_dir) / mode / model_type
        candidates = [d for d in parent.iterdir() if d.is_dir() and dataset_name in d.name]
        if len(candidates) == 1:
            out_file = candidates[0] / f"seed_{seed}.npz"
        elif len(candidates) > 1:
            raise FileNotFoundError(
                f"Ambiguous AutoGluon folders for '{dataset_name}': {[d.name for d in candidates]}\n"
                "Run 'python scripts/autogluon.py' to regenerate with consistent naming."
            )

    if not out_file.exists():
        raise FileNotFoundError(
            f"AutoGluon features not found: {out_file}\n"
            "Run 'python scripts/autogluon.py' to pre-compute them."
        )

    data = np.load(out_file, allow_pickle=True)

    X_train = data["X_train"].astype(np.float64)
    X_val   = data["X_val_clean" if clean_val else "X_val_corrupted"].astype(np.float64)
    X_test  = data["X_test_clean" if clean_test else "X_test_corrupted"].astype(np.float64)

    y_train = data["y_train"]
    y_val   = data["y_val"]
    y_test  = data["y_test"]
    task_type = str(data["task_type"][0])

    metadata: Dict = {
        "task_type": task_type,
        "mode": mode,
        "dataset_name": dataset_name,
        "preparation": "autogluon",
        "clean_val": clean_val,
        "clean_test": clean_test,
        # Curriculum/gate quality scores are not available for this baseline
        "sample_quality_train": np.ones(len(X_train)) * 100.0,
        "feature_quality": {},
        "n_samples": {"train": len(X_train), "val": len(X_val), "test": len(X_test)},
        "n_features_preprocessed": X_train.shape[1],
    }

    return (X_train, X_val, X_test), (y_train, y_val, y_test), None, metadata
