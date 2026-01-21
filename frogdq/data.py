"""
Data loading utilities for FrogDQ datasets.

This module provides functions to list and load preprocessed datasets with various
poisoning modes (clean, AR, NAR).
"""

from pathlib import Path
from typing import Dict, Literal, Tuple

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
    test_size: float = 0.2,
    val_size: float = 0.2,
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
        If True, use clean data for test set (even when mode is ar/nar).
    data_dir : str, default="data"
        Path to the directory containing clean CSV files.
    poisoned_dir : str, default="data_poisoned"
        Path to the directory containing poisoned data subdirectories.
    test_size : float, default=0.2
        Proportion of dataset for test split.
    val_size : float, default=0.2
        Proportion of training data for validation split.
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

    # Load the appropriate dataset based on mode
    if mode == "clean":
        df = pd.read_csv(csv_file, na_values=["?", "NA", "N/A", "NaN", "nan", "NAN", "", " "])
        # For clean mode, mask is all False (no poisoned cells)
        mask_df = pd.DataFrame(False, index=df.index, columns=df.columns, dtype=bool)
    else:
        # Load poisoned data
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
        mask_df = pd.read_csv(mask_file)
        # Convert mask to boolean
        mask_df = mask_df.astype(bool)

    # Load clean data for validation/test if requested
    clean_df = None
    if (clean_val or clean_test) and mode != "clean":
        clean_df = pd.read_csv(
            csv_file, na_values=["?", "NA", "N/A", "NaN", "nan", "NAN", "", " "]
        )

    # Detect label column
    label_cols = [col for col in df.columns if col.startswith(("cls_", "reg_"))]
    if not label_cols:
        raise ValueError(f"No label column found in dataset {dataset_name}")

    label_col = label_cols[0]

    # Split data into train/val/test using the same random seed for reproducibility
    # We need to split indices first, then apply to data/mask separately
    from sklearn.model_selection import train_test_split

    indices = np.arange(len(df))
    y_full = df[label_col].values

    # Determine if stratification should be used
    task_type = "classification" if label_col.startswith("cls_") else "regression"
    stratify_split = y_full if task_type == "classification" else None

    # First split: train+val vs test
    train_val_idx, test_idx = train_test_split(
        indices, test_size=test_size, random_state=seed, stratify=stratify_split
    )

    # Second split: train vs val
    val_size_adjusted = val_size / (1 - test_size)
    stratify_split_val = y_full[train_val_idx] if task_type == "classification" else None
    train_idx, val_idx = train_test_split(
        train_val_idx, test_size=val_size_adjusted, random_state=seed, stratify=stratify_split_val
    )

    # Split the dataframes
    df_train = df.iloc[train_idx].reset_index(drop=True)
    df_val = df.iloc[val_idx].reset_index(drop=True)
    df_test = df.iloc[test_idx].reset_index(drop=True)

    # Split masks
    mask_train = mask_df.iloc[train_idx].reset_index(drop=True)
    mask_val = mask_df.iloc[val_idx].reset_index(drop=True)
    mask_test = mask_df.iloc[test_idx].reset_index(drop=True)

    # Replace val/test with clean data if requested
    if clean_val and clean_df is not None:
        df_val = clean_df.iloc[val_idx].reset_index(drop=True)
        mask_val = pd.DataFrame(False, index=df_val.index, columns=df_val.columns, dtype=bool)

    if clean_test and clean_df is not None:
        df_test = clean_df.iloc[test_idx].reset_index(drop=True)
        mask_test = pd.DataFrame(False, index=df_test.index, columns=df_test.columns, dtype=bool)

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
