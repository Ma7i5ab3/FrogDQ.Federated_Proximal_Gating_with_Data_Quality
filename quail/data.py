"""
Data loading utilities for Quail datasets.

This module provides functions to list and load preprocessed datasets with various
poisoning modes (clean, AR, NAR).
"""

from pathlib import Path
from typing import Dict, Literal, Optional, Tuple

import numpy as np
import pandas as pd

from quail.preprocessing import TabularPreprocessor


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

    # Replace val with clean rows when requested. They stand in for rows of the
    # training partition, so they go through the fitted Saga++ pipeline exactly
    # like the test set does: if the pipeline normalizes, raw clean rows would
    # sit on another scale than the (normalized) training rows.
    if clean_val:
        full_clean_df = pd.read_csv(
            csv_file, na_values=["?", "NA", "N/A", "NaN", "nan", "NAN", "", " "]
        )
        test_rows = full_clean_df.sample(frac=poison_test_size, random_state=42)
        clean_trainval_df = full_clean_df.drop(test_rows.index).reset_index(drop=True)
        if len(clean_trainval_df) == len(df):
            df_val = clean_trainval_df.iloc[val_idx].reset_index(drop=True)
            if pipeline_info is not None:
                df_val = _apply_saga_no_outliers(df_val, pipeline_info)
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


def load_learn2clean_data(
    dataset_name: str,
    mode: str,
    seed: int = 42,
    learn2clean_dir: str = "data_cleaned_learn2clean",
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
    Load Learn2Clean-cleaned data for a dataset split.

    Unlike the CP and Saga++ baselines, Learn2Clean is not shape-preserving: its
    outlier-detection, deduplication and consistency-checking actions drop rows,
    and its feature-selection actions drop columns. Three things follow, and this
    loader exists to handle them:

    * The train+val frame read from ``learn2clean_dir/{mode}/`` is a *subset* of
      the poisoned partition, and it has been rescaled. ``clean_val`` therefore
      cannot swap in raw clean rows: it reads ``learn2clean_dir/clean/{mode}/``,
      where ``scripts/learn2clean.py`` wrote the clean counterparts of exactly
      the surviving rows and columns, put through the normalization and
      imputation fitted on the training frame — so they line up row by row with
      the cleaned CSV and sit on the same scale as the rows the model is fitted
      on.
    * The hold-out cannot come from ``data_poisoned/test/`` unchanged — it would
      carry columns the model was never fitted on, on a different scale. It is
      read from ``learn2clean_dir/test/{mode}/`` instead, where the cleaning
      script already projected and rescaled it.
    * Because rows were dropped, the residual mask is aligned to the reduced
      frame; the quality metadata below is computed on that same frame.

    Parameters
    ----------
    dataset_name : str
        Name of the dataset (e.g. "iris").
    mode : str
        Poisoning mode ("ar" or "nar").
    seed : int, default=42
        Random seed for the train/val split and the test subsample.
    learn2clean_dir : str, default="data_cleaned_learn2clean"
        Root directory written by ``scripts/learn2clean.py``.
    clean_val : bool, default=True
        If True, replace the validation split with the corresponding clean rows.
    clean_test : bool, default=True
        Kept for API compatibility; the test set is always the prepared clean
        hold-out, so this flag has no effect.
    data_dir : str, default="data"
        Directory containing the original clean CSV files (used for clean_val).
    poisoned_dir : str, default="data_poisoned"
        Root directory holding the untouched ``test/`` split, used as a fallback
        when the prepared hold-out is missing.
    val_size : float, default=0.2
        Fraction of the train+val partition to use as validation.
    test_sample_size : float, default=0.8
        Fraction of the hold-out to sample as the final test set.
    poison_test_size : float, default=0.3
        Fraction held out as test when running ``poison_data.py`` (used to
        exclude those rows from the clean train+val pool when clean_val=True).

    Returns
    -------
    Same structure as ``load_data``.

    Raises
    ------
    FileNotFoundError
        If the Learn2Clean-cleaned file does not exist; run
        ``scripts/learn2clean.py`` first.
    """
    import pickle

    from sklearn.model_selection import train_test_split

    data_path = Path(data_dir)
    l2c_mode_path = Path(learn2clean_dir) / mode

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

    # ---- Load Learn2Clean-cleaned train+val partition ----
    l2c_csv = l2c_mode_path / dataset_filename
    l2c_mask_file = l2c_mode_path / f"{csv_file.stem}_mask.csv"
    pipeline_pkl = l2c_mode_path / f"{csv_file.stem}_pipeline.pkl"

    if not l2c_csv.exists():
        raise FileNotFoundError(
            f"Learn2Clean-cleaned data not found: {l2c_csv}. "
            "Run 'python scripts/learn2clean.py' first."
        )

    df = pd.read_csv(l2c_csv, na_values=["?", "NA", "N/A", "NaN", "nan", "NAN", "", " "])
    if l2c_mask_file.exists():
        mask_df = pd.read_csv(l2c_mask_file).astype(bool)
    else:
        mask_df = pd.DataFrame(False, index=df.index, columns=df.columns, dtype=bool)

    pipeline_info: Optional[dict] = None
    if pipeline_pkl.exists():
        with open(pipeline_pkl, "rb") as f:
            pipeline_info = pickle.load(f)

    # ---- Detect label column ----
    label_cols = [col for col in df.columns if col.startswith(("cls_", "reg_"))]
    if not label_cols:
        raise ValueError(f"No label column found in dataset {dataset_name}")
    label_col = label_cols[0]
    task_type = "classification" if label_col.startswith("cls_") else "regression"

    # ---- Load the prepared hold-out ----
    # scripts/learn2clean.py writes it under test/{mode}/ with the surviving
    # columns and the same rescaling; fall back to the raw clean hold-out
    # projected onto those columns if the prepared copy is missing.
    prepared_test = Path(learn2clean_dir) / "test" / mode / dataset_filename
    raw_test = Path(poisoned_dir) / "test" / dataset_filename
    if prepared_test.exists():
        test_file = prepared_test
    elif raw_test.exists():
        test_file = raw_test
    else:
        raise FileNotFoundError(
            f"Test file not found: {prepared_test} (nor {raw_test}). "
            "Run scripts/learn2clean.py (and scripts/poison_data.py) first."
        )

    df_test_full = pd.read_csv(
        test_file, na_values=["?", "NA", "N/A", "NaN", "nan", "NAN", "", " "]
    )
    df_test = df_test_full.sample(frac=test_sample_size, random_state=seed).reset_index(drop=True)

    # Keep the hold-out on exactly the feature set the model is fitted on.
    feature_cols = [c for c in df.columns if c != label_col]
    missing_in_test = [c for c in feature_cols if c not in df_test.columns]
    if missing_in_test:
        raise ValueError(
            f"Prepared hold-out {test_file} is missing columns kept in training: "
            f"{missing_in_test}. Re-run scripts/learn2clean.py for '{dataset_name}'."
        )
    df_test = df_test[feature_cols + ([label_col] if label_col in df_test.columns else [])]

    # ---- Split train+val ----
    indices = np.arange(len(df))
    y_full = df[label_col].values
    stratify_split = y_full if task_type == "classification" else None
    if stratify_split is not None:
        counts = pd.Series(y_full).value_counts()
        if counts.min() < 2:
            stratify_split = None  # a singleton class cannot be stratified
    train_idx, val_idx = train_test_split(
        indices, test_size=val_size, random_state=seed, stratify=stratify_split
    )

    df_train = df.iloc[train_idx].reset_index(drop=True)
    df_val = df.iloc[val_idx].reset_index(drop=True)
    mask_train = mask_df.iloc[train_idx].reset_index(drop=True)
    mask_val = mask_df.iloc[val_idx].reset_index(drop=True)

    # Replace val with the corresponding clean rows when requested. They come
    # from learn2clean_dir/clean/{mode}/: the clean counterparts of the rows and
    # columns Learn2Clean kept, already put through the normalization and
    # imputation fitted on the training frame. Raw clean rows would sit on
    # another scale than the (normalized) training rows, and validation would
    # then measure that shift instead of the preparation.
    if clean_val:
        clean_file = Path(learn2clean_dir) / "clean" / mode / dataset_filename
        if not clean_file.exists():
            raise FileNotFoundError(
                f"Prepared clean train+val partition not found: {clean_file}. "
                "Re-run scripts/learn2clean.py (older outputs predate it), "
                "or pass clean_val=False."
            )
        clean_kept = pd.read_csv(
            clean_file, na_values=["?", "NA", "N/A", "NaN", "nan", "NAN", "", " "]
        )
        if len(clean_kept) != len(df):
            raise ValueError(
                f"Prepared clean partition {clean_file} has {len(clean_kept)} rows but "
                f"the Learn2Clean-cleaned partition has {len(df)}; they must line up "
                f"row by row. Re-run scripts/learn2clean.py for '{dataset_name}'."
            )
        clean_cols = [c for c in df.columns if c in clean_kept.columns]
        df_val = clean_kept.iloc[val_idx][clean_cols].reset_index(drop=True)
        mask_val = pd.DataFrame(
            False, index=df_val.index, columns=df_val.columns, dtype=bool
        )

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
    y_test = df_test[label_col].values  # labels are never transformed

    # ---- Quality metrics ----
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
    all_masks = pd.concat(
        [mask_train[feat_cols_present], mask_val[feat_cols_present]], ignore_index=True
    )
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
        # Learn2Clean-specific: how much of the poisoned partition survived, and
        # which strategy produced it. Reported so the reduction is visible in the
        # results rather than hidden behind an equal-looking metrics row.
        "learn2clean_strategy": (pipeline_info or {}).get("strategy"),
        "learn2clean_goal": (pipeline_info or {}).get("goal"),
        "learn2clean_rows_kept": len(df),
        "learn2clean_cols_kept": len(df.columns),
        "learn2clean_dropped_columns": (pipeline_info or {}).get("dropped_columns", []),
    }

    return (X_train, X_val, X_test), (y_train, y_val, y_test), preprocessor, metadata


def load_diffprep_data(
    dataset_name: str,
    mode: str,
    seed: int = 42,
    diffprep_dir: str = "data_cleaned_diffprep",
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
    Load DiffPrep-cleaned data for a dataset split.

    DiffPrep (Li et al., SIGMOD '23) searches a per-feature preprocessing
    pipeline by relaxing the discrete space into a continuous one and solving a
    bi-level optimization problem with gradient descent.  ``scripts/diffprep.py``
    writes the result as a shape-preserving CSV — same rows, same num_* / cat_* /
    cls_* columns as the poisoned input — so this loader is close to
    ``load_saga_data``.  Two things are specific to it:

    * **No second standardization.** The num_* values written by DiffPrep are
      already on the scale its search chose, one normalizer (and possibly a
      discretizer) per feature.  Standardizing again would compose another
      affine map on top and throw that choice away, so the TabularPreprocessor
      is built with ``scale_numerical=False`` unless the caller overrides it.
      Imputation and one-hot encoding still run: one-hot encoding is the single
      step of the DiffPrep pipeline that ``scripts/diffprep.py`` deliberately
      does not materialize, precisely because this preprocessor performs it.

    * **Pre-transformed companion frames.** The clean hold-out and, when
      ``clean_val`` is set, the clean train+val partition are read from
      ``diffprep_dir/test/{mode}/`` and ``diffprep_dir/clean/{mode}/``, where
      ``scripts/diffprep.py`` wrote them after pushing them through the
      transformers fitted on the training frame.  Reading the raw frames instead
      would put the validation and test rows in a different space from the
      training rows.

    Parameters
    ----------
    dataset_name : str
        Name of the dataset (e.g. "iris").
    mode : str
        Poisoning mode ("ar" or "nar").
    seed : int, default=42
        Random seed for the train/val split.
    diffprep_dir : str, default="data_cleaned_diffprep"
        Root directory written by ``scripts/diffprep.py``.
    clean_val : bool, default=True
        If True, replace the validation split with the corresponding clean
        (un-poisoned) rows, transformed by the same pipeline.
    clean_test : bool, default=True
        Kept for API compatibility; has no effect — the test set always comes
        from the prepared clean hold-out.
    data_dir : str, default="data"
        Directory containing the original clean CSV files (used to locate the
        dataset's filename).
    poisoned_dir : str, default="data_poisoned"
        Kept for API compatibility; the prepared hold-out is read from
        ``diffprep_dir`` instead.
    val_size : float, default=0.2
        Fraction of the train+val partition used as validation.
    test_sample_size : float, default=0.8
        Fraction of the prepared hold-out sampled as the final test set, with
        ``seed`` — the same draw every other loader makes, so all methods are
        scored on the same test rows for a given seed.
    poison_test_size : float, default=0.3
        Fraction held out as test when running ``poison_data.py``.

    Returns
    -------
    Same structure as ``load_data``.

    Raises
    ------
    FileNotFoundError
        If the DiffPrep-cleaned files do not exist; run ``scripts/diffprep.py``
        first.
    """
    import pickle

    from sklearn.model_selection import train_test_split

    data_path = Path(data_dir)
    dp_mode_path = Path(diffprep_dir) / mode

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

    # ---- Load DiffPrep-cleaned train+val partition ----
    dp_csv = dp_mode_path / dataset_filename
    dp_mask_file = dp_mode_path / f"{csv_file.stem}_mask.csv"
    pipeline_pkl = dp_mode_path / f"{csv_file.stem}_pipeline.pkl"

    if not dp_csv.exists():
        raise FileNotFoundError(
            f"DiffPrep-cleaned data not found: {dp_csv}. "
            "Run 'python scripts/diffprep.py' first."
        )

    df = pd.read_csv(dp_csv, na_values=["?", "NA", "N/A", "NaN", "nan", "NAN", "", " "])
    if dp_mask_file.exists():
        mask_df = pd.read_csv(dp_mask_file).astype(bool)
    else:
        mask_df = pd.DataFrame(False, index=df.index, columns=df.columns, dtype=bool)

    pipeline_info: Optional[dict] = None
    if pipeline_pkl.exists():
        with open(pipeline_pkl, "rb") as f:
            pipeline_info = pickle.load(f)

    # ---- Detect label column ----
    label_cols = [col for col in df.columns if col.startswith(("cls_", "reg_"))]
    if not label_cols:
        raise ValueError(f"No label column found in dataset {dataset_name}")
    label_col = label_cols[0]
    task_type = "classification" if label_col.startswith("cls_") else "regression"
    feature_cols = [col for col in df.columns if col != label_col]

    # ---- Load the prepared clean hold-out ----
    test_file = Path(diffprep_dir) / "test" / mode / dataset_filename
    if not test_file.exists():
        raise FileNotFoundError(
            f"Prepared DiffPrep hold-out not found: {test_file}. "
            "Run scripts/diffprep.py (and scripts/poison_data.py) first."
        )
    df_test_full = pd.read_csv(
        test_file, na_values=["?", "NA", "N/A", "NaN", "nan", "NAN", "", " "]
    )
    df_test = df_test_full.sample(frac=test_sample_size, random_state=seed).reset_index(drop=True)
    missing_in_test = [c for c in feature_cols if c not in df_test.columns]
    if missing_in_test:
        raise ValueError(
            f"Prepared hold-out {test_file} is missing columns kept in training: "
            f"{missing_in_test}. Re-run scripts/diffprep.py for '{dataset_name}'."
        )
    df_test = df_test[feature_cols + ([label_col] if label_col in df_test.columns else [])]

    # ---- Split train+val ----
    indices = np.arange(len(df))
    y_full = df[label_col].values
    stratify_split = y_full if task_type == "classification" else None
    if stratify_split is not None:
        counts = pd.Series(y_full).value_counts()
        if counts.min() < 2:
            stratify_split = None  # a singleton class cannot be stratified
    train_idx, val_idx = train_test_split(
        indices, test_size=val_size, random_state=seed, stratify=stratify_split
    )

    df_train = df.iloc[train_idx].reset_index(drop=True)
    df_val = df.iloc[val_idx].reset_index(drop=True)
    mask_train = mask_df.iloc[train_idx].reset_index(drop=True)
    mask_val = mask_df.iloc[val_idx].reset_index(drop=True)

    # Replace val with the corresponding clean rows when requested. They come
    # from diffprep_dir/clean/{mode}/, i.e. already pushed through the pipeline
    # fitted on the poisoned training frame — raw clean rows would sit in a
    # different space than the rows the model is fitted on.
    if clean_val:
        clean_file = Path(diffprep_dir) / "clean" / mode / dataset_filename
        if not clean_file.exists():
            raise FileNotFoundError(
                f"Prepared clean train+val partition not found: {clean_file}. "
                "Re-run scripts/diffprep.py, or pass clean_val=False."
            )
        clean_trainval_df = pd.read_csv(
            clean_file, na_values=["?", "NA", "N/A", "NaN", "nan", "NAN", "", " "]
        )
        if len(clean_trainval_df) != len(df):
            raise ValueError(
                f"Prepared clean partition {clean_file} has {len(clean_trainval_df)} rows "
                f"but the DiffPrep-cleaned partition has {len(df)}; they must line up "
                f"row by row (poison_test_size={poison_test_size})."
            )
        clean_cols = [c for c in df.columns if c in clean_trainval_df.columns]
        df_val = clean_trainval_df.iloc[val_idx][clean_cols].reset_index(drop=True)
        mask_val = pd.DataFrame(
            False, index=df_val.index, columns=df_val.columns, dtype=bool
        )

    # ---- TabularPreprocessor ----
    # scale_numerical defaults to False here: DiffPrep already chose the scale.
    preprocessor_kwargs.setdefault("scale_numerical", False)
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
    y_test = df_test[label_col].values  # labels are never transformed

    # ---- Quality metrics ----
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
    all_masks = pd.concat(
        [mask_train[feat_cols_present], mask_val[feat_cols_present]], ignore_index=True
    )
    total_dirty = all_masks.sum().sum()
    total_cells = all_masks.size
    overall_quality = (1 - total_dirty / max(total_cells, 1)) * 100

    dp_params = (pipeline_info or {}).get("params", {})
    dp_result = (pipeline_info or {}).get("result", {})

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
        # DiffPrep-specific: which variant ran, what it found, and how well the
        # end model it was searched against did — so the search that produced
        # this data is visible in the results rather than only in its log.
        "diffprep_method": (pipeline_info or {}).get("method"),
        "diffprep_end_model": (pipeline_info or {}).get("end_model"),
        "diffprep_prenormalized": (pipeline_info or {}).get("prenormalized"),
        "diffprep_passthrough_columns": (pipeline_info or {}).get("passthrough_columns", []),
        "diffprep_model_lr": dp_params.get("model_lr"),
        "diffprep_best_epoch": dp_result.get("best_epoch"),
        "diffprep_search_val_acc": dp_result.get("best_val_acc"),
        "diffprep_search_test_acc": dp_result.get("best_test_acc"),
    }

    return (X_train, X_val, X_test), (y_train, y_val, y_test), preprocessor, metadata


def load_ctxpipe_data(
    dataset_name: str,
    mode: str,
    seed: int = 42,
    ctxpipe_dir: str = "data_cleaned_ctxpipe",
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
    Load CtxPipe-prepared data for a dataset split.

    CtxPipe (Gao et al., SIGMOD '25) builds the preparation pipeline with deep
    Q-network agents guided by a text embedding of the data (the context
    plug-in).  ``scripts/ctxpipe.py`` writes out the whole pipeline the agents
    chose — imputation, encoding, scaling, feature engineering and feature
    selection — so the CSV keeps the rows of the poisoned input but not its
    columns: every feature is numeric and prefixed ``num_*``, either an input
    column transformed in place or a feature a step derived (``num_pca_0``,
    ``num_poly_12``, ``num_rte_431``, ...).  This loader is otherwise close to
    ``load_diffprep_data``:

    * **No second standardization.** The features already are on the scale,
      and in the space, CtxPipe chose, so the TabularPreprocessor is built with
      ``scale_numerical=False`` unless the caller overrides it.  Imputation
      still runs (it is a no-op: CtxPipe leaves no missing value behind).

    * **Pre-transformed companion frames.** The clean hold-out and, when
      ``clean_val`` is set, the clean train+val partition are read from
      ``ctxpipe_dir/test/{mode}/`` and ``ctxpipe_dir/clean/{mode}/``, where
      ``scripts/ctxpipe.py`` wrote them after pushing them through the pipeline
      fitted on the training frame.  A raw row would not even have the same
      columns.

    The ``_mask.csv`` next to the CSV is the residual mask over those output
    columns: a derived feature is flagged in a row when one of the input
    columns it is computed from was poisoned there *and* the value is still
    missing.

    Parameters
    ----------
    dataset_name : str
        Name of the dataset (e.g. "iris").
    mode : str
        Poisoning mode ("ar" or "nar").
    seed : int, default=42
        Random seed for the train/val split.
    ctxpipe_dir : str, default="data_cleaned_ctxpipe"
        Root directory written by ``scripts/ctxpipe.py``.
    clean_val : bool, default=True
        If True, replace the validation split with the corresponding clean
        (un-poisoned) rows, transformed by the same pipeline.
    clean_test : bool, default=True
        Kept for API compatibility; has no effect — the test set always comes
        from the prepared clean hold-out.
    data_dir : str, default="data"
        Directory containing the original clean CSV files (used to locate the
        dataset's filename).
    poisoned_dir : str, default="data_poisoned"
        Kept for API compatibility; the prepared hold-out is read from
        ``ctxpipe_dir`` instead.
    val_size : float, default=0.2
        Fraction of the train+val partition used as validation.
    test_sample_size : float, default=0.8
        Fraction of the prepared hold-out sampled as the final test set, with
        ``seed`` — the same draw every other loader makes, so all methods are
        scored on the same test rows for a given seed.
    poison_test_size : float, default=0.3
        Fraction held out as test when running ``poison_data.py``.

    Returns
    -------
    Same structure as ``load_data``.

    Raises
    ------
    FileNotFoundError
        If the CtxPipe-prepared files do not exist; run ``scripts/ctxpipe.py``
        first.
    """
    import pickle

    from sklearn.model_selection import train_test_split

    data_path = Path(data_dir)
    cp_mode_path = Path(ctxpipe_dir) / mode

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

    # ---- Load CtxPipe-prepared train+val partition ----
    cp_csv = cp_mode_path / dataset_filename
    cp_mask_file = cp_mode_path / f"{csv_file.stem}_mask.csv"
    pipeline_pkl = cp_mode_path / f"{csv_file.stem}_pipeline.pkl"

    if not cp_csv.exists():
        raise FileNotFoundError(
            f"CtxPipe-prepared data not found: {cp_csv}. "
            "Run 'python scripts/ctxpipe.py' first."
        )

    df = pd.read_csv(cp_csv, na_values=["?", "NA", "N/A", "NaN", "nan", "NAN", "", " "])
    if cp_mask_file.exists():
        mask_df = pd.read_csv(cp_mask_file).astype(bool)
    else:
        mask_df = pd.DataFrame(False, index=df.index, columns=df.columns, dtype=bool)

    pipeline_info: Optional[dict] = None
    if pipeline_pkl.exists():
        with open(pipeline_pkl, "rb") as f:
            pipeline_info = pickle.load(f)

    # ---- Detect label column ----
    label_cols = [col for col in df.columns if col.startswith(("cls_", "reg_"))]
    if not label_cols:
        raise ValueError(f"No label column found in dataset {dataset_name}")
    label_col = label_cols[0]
    task_type = "classification" if label_col.startswith("cls_") else "regression"
    feature_cols = [col for col in df.columns if col != label_col]

    # ---- Load the prepared clean hold-out ----
    test_file = Path(ctxpipe_dir) / "test" / mode / dataset_filename
    if not test_file.exists():
        raise FileNotFoundError(
            f"Prepared CtxPipe hold-out not found: {test_file}. "
            "Run scripts/ctxpipe.py (and scripts/poison_data.py) first."
        )
    df_test_full = pd.read_csv(
        test_file, na_values=["?", "NA", "N/A", "NaN", "nan", "NAN", "", " "]
    )
    df_test = df_test_full.sample(frac=test_sample_size, random_state=seed).reset_index(drop=True)
    missing_in_test = [c for c in feature_cols if c not in df_test.columns]
    if missing_in_test:
        raise ValueError(
            f"Prepared hold-out {test_file} is missing columns kept in training: "
            f"{missing_in_test[:10]}. Re-run scripts/ctxpipe.py for '{dataset_name}'."
        )
    df_test = df_test[feature_cols + ([label_col] if label_col in df_test.columns else [])]

    # ---- Split train+val ----
    indices = np.arange(len(df))
    y_full = df[label_col].values
    stratify_split = y_full if task_type == "classification" else None
    if stratify_split is not None:
        counts = pd.Series(y_full).value_counts()
        if counts.min() < 2:
            stratify_split = None  # a singleton class cannot be stratified
    train_idx, val_idx = train_test_split(
        indices, test_size=val_size, random_state=seed, stratify=stratify_split
    )

    df_train = df.iloc[train_idx].reset_index(drop=True)
    df_val = df.iloc[val_idx].reset_index(drop=True)
    mask_train = mask_df.iloc[train_idx].reset_index(drop=True)
    mask_val = mask_df.iloc[val_idx].reset_index(drop=True)

    # Replace val with the corresponding clean rows when requested. They come
    # from ctxpipe_dir/clean/{mode}/, i.e. already pushed through the pipeline
    # fitted on the poisoned training frame.
    if clean_val:
        clean_file = Path(ctxpipe_dir) / "clean" / mode / dataset_filename
        if not clean_file.exists():
            raise FileNotFoundError(
                f"Prepared clean train+val partition not found: {clean_file}. "
                "Re-run scripts/ctxpipe.py, or pass clean_val=False."
            )
        clean_trainval_df = pd.read_csv(
            clean_file, na_values=["?", "NA", "N/A", "NaN", "nan", "NAN", "", " "]
        )
        if len(clean_trainval_df) != len(df):
            raise ValueError(
                f"Prepared clean partition {clean_file} has {len(clean_trainval_df)} rows "
                f"but the CtxPipe-prepared partition has {len(df)}; they must line up "
                f"row by row (poison_test_size={poison_test_size})."
            )
        clean_cols = [c for c in df.columns if c in clean_trainval_df.columns]
        df_val = clean_trainval_df.iloc[val_idx][clean_cols].reset_index(drop=True)
        mask_val = pd.DataFrame(
            False, index=df_val.index, columns=df_val.columns, dtype=bool
        )

    # ---- TabularPreprocessor ----
    # scale_numerical defaults to False here: CtxPipe already chose the scale.
    preprocessor_kwargs.setdefault("scale_numerical", False)
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
    y_test = df_test[label_col].values  # labels are never transformed

    # ---- Quality metrics ----
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
    all_masks = pd.concat(
        [mask_train[feat_cols_present], mask_val[feat_cols_present]], ignore_index=True
    )
    total_dirty = all_masks.sum().sum()
    total_cells = all_masks.size
    overall_quality = (1 - total_dirty / max(total_cells, 1)) * 100

    cp_result = (pipeline_info or {}).get("result", {})

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
        # CtxPipe-specific: the pipeline the agents built and how the reward
        # model scored it — so the search that produced this data is visible in
        # the results rather than only in its log.
        "ctxpipe_logical_pipeline": (pipeline_info or {}).get("logical_pipeline"),
        "ctxpipe_physical_pipeline": [
            step.get("primitive") for step in (pipeline_info or {}).get("physical_pipeline", [])
        ],
        "ctxpipe_input_features": len((pipeline_info or {}).get("input_feature_columns", [])),
        "ctxpipe_dropped_columns": (pipeline_info or {}).get("dropped_columns", []),
        "ctxpipe_search_acc": cp_result.get("search_score"),
        "ctxpipe_holdout_acc": cp_result.get("holdout_test_acc"),
    }

    return (X_train, X_val, X_test), (y_train, y_val, y_test), preprocessor, metadata
