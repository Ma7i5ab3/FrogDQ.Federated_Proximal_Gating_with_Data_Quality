import pickle
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

__all__ = ["TabularPreprocessor"]


class TabularPreprocessor:
    """
    A comprehensive preprocessor for tabular data that handles:
    - Automatic feature type detection (with support for prefixed column names)
    - Missing value imputation
    - Outlier clipping for numerical features
    - Rare category encoding for categorical features
    - Train/validation/test splitting
    - Scikit-learn compatibility

    Supports column name prefixes from download_data.py:
    - cls_* : classification target
    - reg_* : regression target
    - cat_* : categorical feature
    - num_* : numerical feature
    - dat_* : date feature
    - id_* : ID/metadata (excluded from training)

    Example:
        >>> preprocessor = TabularPreprocessor()
        >>> X_train, X_val, X_test, y_train, y_val, y_test = preprocessor.fit_transform(df)
    """

    def __init__(
        self,
        label_column: Optional[str] = None,
        test_size: float = 0.2,
        val_size: float = 0.2,
        random_state: int = 42,
        stratify: bool = False,
        numerical_impute_strategy: str = "median",
        categorical_impute_strategy: str = "most_frequent",
        outlier_clip_percentile: Tuple[float, float] = (1, 99),
        rare_category_threshold: float = 0.01,
        max_categories: int = 50,
        numerical_features: Optional[List[str]] = None,
        categorical_features: Optional[List[str]] = None,
        use_column_prefixes: bool = True,
        verbose: bool = False,
    ):
        """
        Initialize the TabularPreprocessor.

        Parameters:
        -----------
        label_column : str, optional
            Name of the label column in the dataframe. If None, auto-detected from prefixes.
        test_size : float, default=0.2
            Proportion of dataset to include in test split.
        val_size : float, default=0.2
            Proportion of training data to include in validation split.
        random_state : int, default=42
            Random state for reproducibility.
        stratify : bool, default=False
            Whether to stratify splits (useful for classification).
        numerical_impute_strategy : str, default='median'
            Strategy for imputing numerical features ('mean', 'median', 'most_frequent').
        categorical_impute_strategy : str, default='most_frequent'
            Strategy for imputing categorical features.
        outlier_clip_percentile : tuple, default=(1, 99)
            Percentiles for clipping outliers in numerical features.
        rare_category_threshold : float, default=0.01
            Frequency threshold below which categories are grouped as 'rare'.
        max_categories : int, default=50
            Maximum number of categories per feature before rare grouping.
        numerical_features : list, optional
            Explicit list of numerical feature names. If None, auto-detected.
        categorical_features : list, optional
            Explicit list of categorical feature names. If None, auto-detected.
        use_column_prefixes : bool, default=True
            Whether to use column name prefixes (cls_, reg_, cat_, num_, etc.) for type detection.
        verbose : bool, default=False
            Whether to print detailed logs during processing.
        """
        self.label_column = label_column
        self.test_size = test_size
        self.val_size = val_size
        self.random_state = random_state
        self.stratify = stratify
        self.numerical_impute_strategy = numerical_impute_strategy
        self.categorical_impute_strategy = categorical_impute_strategy
        self.outlier_clip_percentile = outlier_clip_percentile
        self.rare_category_threshold = rare_category_threshold
        self.max_categories = max_categories
        self.numerical_features = numerical_features
        self.categorical_features = categorical_features
        self.use_column_prefixes = use_column_prefixes
        self.verbose = verbose

        # These will be set during fit
        self.preprocessor_ = None
        self.feature_names_out_ = None
        self.numerical_features_ = None
        self.categorical_features_ = None
        self.date_features_ = None
        self.outlier_bounds_ = {}
        self.rare_category_mapping_ = {}
        self.date_bounds_ = {}
        self.is_classification_ = None
        self.label_type_ = None

    def _detect_label_column(self, df: pd.DataFrame) -> Optional[str]:
        """Auto-detect label column from prefixes (cls_ or reg_)."""
        if not self.use_column_prefixes:
            return None

        label_cols = [col for col in df.columns if col.startswith(("cls_", "reg_"))]

        if len(label_cols) == 0:
            raise ValueError(
                "No label column detected with prefixes 'cls_' or 'reg_'. Please specify label_column."
            )
        elif len(label_cols) == 1:
            return label_cols[0]
        else:
            raise ValueError(
                f"Multiple label columns detected: {label_cols}. Please specify label_column."
            )

    def _detect_label_type(self, label_col: str, y: np.ndarray) -> str:
        """
        Detect if label is classification or regression.

        Uses prefix if available, otherwise infers from data.
        """
        if self.use_column_prefixes and label_col:
            if label_col.startswith("cls_"):
                return "classification"
            elif label_col.startswith("reg_"):
                return "regression"

        # Fallback: infer from data
        # If numeric and many unique values relative to dataset size, likely regression
        if pd.api.types.is_numeric_dtype(y):
            n_unique = pd.Series(y).nunique()
            if n_unique / len(y) > 0.05:  # More than 5% unique values
                return "regression"

        return "classification"

    def _detect_feature_types(self, X: pd.DataFrame) -> Tuple[List[str], List[str], List[str]]:
        """Automatically detect numerical, categorical, and date features."""
        if self.numerical_features is not None and self.categorical_features is not None:
            return self.numerical_features, self.categorical_features, []

        numerical = []
        categorical = []
        date_features = []
        id_columns = []

        for col in X.columns:
            if self.numerical_features and col in self.numerical_features:
                numerical.append(col)
            elif self.categorical_features and col in self.categorical_features:
                categorical.append(col)
            elif self.use_column_prefixes:
                # Use prefix-based detection
                if col.startswith("num_"):
                    numerical.append(col)
                elif col.startswith("cat_"):
                    categorical.append(col)
                elif col.startswith("dat_"):
                    # Date features - will be converted to normalized numerical values
                    date_features.append(col)
                elif col.startswith("id_"):
                    # Skip ID columns
                    id_columns.append(col)
                else:
                    # Fallback to heuristic detection
                    if pd.api.types.is_numeric_dtype(X[col]):
                        n_unique = X[col].nunique()
                        if n_unique <= 10 and n_unique / len(X) < 0.05:
                            categorical.append(col)
                        else:
                            numerical.append(col)
                    else:
                        categorical.append(col)
            else:
                # Original auto-detect logic when prefixes are disabled
                if pd.api.types.is_numeric_dtype(X[col]):
                    # Check if it's actually categorical (few unique values)
                    n_unique = X[col].nunique()
                    if n_unique <= 10 and n_unique / len(X) < 0.05:
                        categorical.append(col)
                    else:
                        numerical.append(col)
                else:
                    categorical.append(col)

        if id_columns and self.verbose:
            print(
                f"Skipping {len(id_columns)} ID columns: {id_columns[:3]}{'...' if len(id_columns) > 3 else ''}"
            )

        return numerical, categorical, date_features

    def _compute_date_bounds(self, X: pd.DataFrame, date_features: List[str]) -> Dict:
        """Compute min/max dates for normalization."""
        bounds = {}
        for col in date_features:
            # Convert to datetime, suppressing warnings
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", category=UserWarning)
                date_col = pd.to_datetime(X[col], errors="coerce", dayfirst=True)
            min_date = date_col.min()
            max_date = date_col.max()
            bounds[col] = (min_date, max_date)
        return bounds

    def _transform_dates(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Transform date columns to normalized numerical values [0, 1].

        Converts dates to timestamps and normalizes based on min/max in training data.
        """
        X = X.copy()
        for col in self.date_features_:
            if col in X.columns and col in self.date_bounds_:
                min_date, max_date = self.date_bounds_[col]

                # Convert to datetime, suppressing warnings
                with warnings.catch_warnings():
                    warnings.filterwarnings("ignore", category=UserWarning)
                    date_col = pd.to_datetime(X[col], errors="coerce", dayfirst=True)

                # Convert to timestamps (seconds since epoch)
                timestamps = date_col.astype(np.int64) / 10**9  # nanoseconds to seconds

                # Normalize to [0, 1] based on training min/max
                min_ts = min_date.value / 10**9 if pd.notna(min_date) else 0
                max_ts = max_date.value / 10**9 if pd.notna(max_date) else 1

                if max_ts > min_ts:
                    X[col] = (timestamps - min_ts) / (max_ts - min_ts)
                else:
                    X[col] = 0  # All dates are the same

        return X

    def _compute_outlier_bounds(self, X: pd.DataFrame, numerical_features: List[str]) -> Dict:
        """Compute outlier clipping bounds for numerical features."""
        bounds = {}
        for col in numerical_features:
            # Convert to numeric first to handle any string representations
            numeric_col = pd.to_numeric(X[col], errors="coerce")
            lower = np.nanpercentile(numeric_col, self.outlier_clip_percentile[0])
            upper = np.nanpercentile(numeric_col, self.outlier_clip_percentile[1])
            bounds[col] = (lower, upper)
        return bounds

    def _compute_rare_categories(self, X: pd.DataFrame, categorical_features: List[str]) -> Dict:
        """Identify rare categories to be grouped."""
        rare_mapping = {}
        for col in categorical_features:
            value_counts = X[col].value_counts(normalize=True, dropna=True)
            n_unique = len(value_counts)

            # Apply rare category threshold or max categories limit
            if n_unique > self.max_categories:
                # Keep top max_categories-1 and group rest as rare
                top_categories = value_counts.nlargest(self.max_categories - 1).index.tolist()
                rare_mapping[col] = top_categories
            else:
                # Group by frequency threshold
                rare_categories = value_counts[
                    value_counts < self.rare_category_threshold
                ].index.tolist()
                if rare_categories:
                    keep_categories = value_counts[
                        value_counts >= self.rare_category_threshold
                    ].index.tolist()
                    rare_mapping[col] = keep_categories

        return rare_mapping

    def _apply_rare_category_mapping(self, X: pd.DataFrame) -> pd.DataFrame:
        """Apply rare category mapping to categorical features."""
        X = X.copy()
        for col, keep_categories in self.rare_category_mapping_.items():
            if col in X.columns:
                # Ensure column is object dtype first to avoid dtype warnings
                if X[col].dtype != "object":
                    X[col] = X[col].astype("object")
                # Use a more efficient approach with isin
                mask = X[col].isin(keep_categories) | X[col].isna()
                X.loc[~mask, col] = "RARE_CATEGORY"
        return X

    def _convert_categorical_to_string(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Convert all categorical features to string type.

        This prevents the OneHotEncoder error about mixed string/numeric types
        by ensuring all categorical columns are uniformly strings.
        """
        X = X.copy()
        for col in self.categorical_features_:
            if col in X.columns:
                # Convert to string, preserving NaN as NaN (not "nan")
                X[col] = X[col].astype(str).replace("nan", np.nan)
        return X

    def _clip_outliers(self, X: pd.DataFrame) -> pd.DataFrame:
        """Clip outliers in numerical features."""
        X = X.copy()
        for col, (lower, upper) in self.outlier_bounds_.items():
            if col in X.columns:
                # Convert to numeric first to handle any string representations
                X[col] = pd.to_numeric(X[col], errors="coerce")
                # Use infer_objects to avoid downcasting warning
                with warnings.catch_warnings():
                    warnings.filterwarnings("ignore", category=FutureWarning)
                    X[col] = X[col].clip(lower, upper).infer_objects(copy=False)
        return X

    def fit(self, X: Union[pd.DataFrame, np.ndarray], y: Optional[np.ndarray] = None):
        """
        Fit the preprocessor on the training data.

        Parameters:
        -----------
        X : pd.DataFrame or np.ndarray
            Feature matrix or full dataframe (including label if label_column is set).
        y : np.ndarray, optional
            Target vector (only needed if label_column is None).

        Returns:
        --------
        self
        """
        # Handle input format
        if isinstance(X, np.ndarray):
            raise ValueError("X must be a pandas DataFrame for fitting.")

        # Auto-detect label column if not provided and y is not given
        if self.label_column is None and y is None:
            detected_label = self._detect_label_column(X)
            if detected_label is not None:
                self.label_column = detected_label
                if self.verbose:
                    print(f"Auto-detected label column: {self.label_column}")

        if self.label_column is not None:
            if self.label_column not in X.columns:
                raise ValueError(f"Label column '{self.label_column}' not found in DataFrame.")
            X_features = X.drop(columns=[self.label_column])
            y = X[self.label_column].values
        else:
            X_features = X

        # Detect label type (classification vs regression)
        if y is not None:
            self.label_type_ = self._detect_label_type(self.label_column, y)
            self.is_classification_ = self.label_type_ == "classification"
            if self.verbose:
                print(f"Detected task type: {self.label_type_}")

        # Detect feature types
        self.numerical_features_, self.categorical_features_, self.date_features_ = (
            self._detect_feature_types(X_features)
        )

        num_display = self.numerical_features_[:5] if self.numerical_features_ else []
        cat_display = self.categorical_features_[:5] if self.categorical_features_ else []
        date_display = self.date_features_[:5] if self.date_features_ else []
        if self.verbose:
            print(
                f"Detected {len(self.numerical_features_)} numerical features: {num_display}{'...' if len(self.numerical_features_) > 5 else ''}"
            )
            print(
                f"Detected {len(self.categorical_features_)} categorical features: {cat_display}{'...' if len(self.categorical_features_) > 5 else ''}"
            )
            if self.date_features_:
                print(
                    f"Detected {len(self.date_features_)} date features: {date_display}{'...' if len(self.date_features_) > 5 else ''}"
                )

        # Transform dates to numerical values first
        if self.date_features_:
            self.date_bounds_ = self._compute_date_bounds(X_features, self.date_features_)
            X_features = self._transform_dates(X_features)
            # Add transformed date columns to numerical features for processing
            self.numerical_features_ = self.numerical_features_ + self.date_features_

        # Compute outlier bounds on original data (before imputation)
        if self.numerical_features_:
            self.outlier_bounds_ = self._compute_outlier_bounds(
                X_features, self.numerical_features_
            )

        # Compute rare category mappings
        if self.categorical_features_:
            self.rare_category_mapping_ = self._compute_rare_categories(
                X_features, self.categorical_features_
            )

        # Build preprocessing pipeline
        numerical_pipeline = Pipeline(
            [
                ("imputer", SimpleImputer(strategy=self.numerical_impute_strategy)),
                ("scaler", StandardScaler()),
            ]
        )

        categorical_pipeline = Pipeline(
            [
                (
                    "imputer",
                    SimpleImputer(strategy=self.categorical_impute_strategy, fill_value="MISSING"),
                ),
                (
                    "onehot",
                    OneHotEncoder(handle_unknown="ignore", sparse_output=False, dtype=np.float64),
                ),
            ]
        )

        transformers = []
        if self.numerical_features_:
            transformers.append(("num", numerical_pipeline, self.numerical_features_))
        if self.categorical_features_:
            transformers.append(("cat", categorical_pipeline, self.categorical_features_))

        self.preprocessor_ = ColumnTransformer(transformers=transformers)

        # Apply preprocessing steps
        X_processed = X_features.copy()
        X_processed = self._clip_outliers(X_processed)
        X_processed = self._apply_rare_category_mapping(X_processed)
        X_processed = self._convert_categorical_to_string(X_processed)

        # Fit the preprocessor
        self.preprocessor_.fit(X_processed)

        # Store feature names
        self._compute_feature_names()

        return self

    def transform(self, X: pd.DataFrame) -> np.ndarray:
        """
        Transform the data using the fitted preprocessor.

        Parameters:
        -----------
        X : pd.DataFrame
            Feature matrix to transform.

        Returns:
        --------
        X_transformed : np.ndarray
            Transformed feature matrix.
        """
        if self.preprocessor_ is None:
            raise ValueError("Preprocessor must be fitted before transform.")

        if self.label_column is not None and self.label_column in X.columns:
            X = X.drop(columns=[self.label_column])

        X_processed = X.copy()

        # Transform dates first if any exist
        if self.date_features_:
            X_processed = self._transform_dates(X_processed)

        X_processed = self._clip_outliers(X_processed)
        X_processed = self._apply_rare_category_mapping(X_processed)
        X_processed = self._convert_categorical_to_string(X_processed)

        return self.preprocessor_.transform(X_processed)

    def fit_transform(
        self, X: Union[pd.DataFrame, np.ndarray], y: Optional[np.ndarray] = None
    ) -> Union[np.ndarray, Tuple[np.ndarray, ...]]:
        """
        Fit the preprocessor and transform data, with train/val/test splitting.

        Parameters:
        -----------
        X : pd.DataFrame or np.ndarray
            Feature matrix or full dataframe (including label if label_column is set).
        y : np.ndarray, optional
            Target vector (only needed if label_column is None).

        Returns:
        --------
        X_train, X_val, X_test, y_train, y_val, y_test : tuple of np.ndarray
            Preprocessed and split data.
        """
        # Handle input format
        if isinstance(X, np.ndarray):
            raise ValueError("X must be a pandas DataFrame.")

        # Auto-detect label column if not provided
        if self.label_column is None:
            detected_label = self._detect_label_column(X)
            if detected_label is not None:
                self.label_column = detected_label
                if self.verbose:
                    print(f"Auto-detected label column: {self.label_column}")

        # Extract features and labels
        if self.label_column is not None:
            if self.label_column not in X.columns:
                raise ValueError(f"Label column '{self.label_column}' not found in DataFrame.")
            X_features = X.drop(columns=[self.label_column])
            y = X[self.label_column].values
        else:
            if y is None:
                raise ValueError("Either label_column must be set or y must be provided.")
            X_features = X

        # Determine stratification: use self.stratify if explicitly set,
        # otherwise auto-detect based on label type
        should_stratify = self.stratify
        if not should_stratify and self.label_column:
            # Check if this looks like a classification task for auto-stratification
            label_type = self._detect_label_type(self.label_column, y)
            should_stratify = label_type == "classification"

        # First split: train+val vs test
        stratify_split = y if should_stratify else None
        X_train_val, X_test, y_train_val, y_test = train_test_split(
            X_features,
            y,
            test_size=self.test_size,
            random_state=self.random_state,
            stratify=stratify_split,
        )

        # Second split: train vs val
        val_size_adjusted = self.val_size / (1 - self.test_size)
        stratify_split_val = y_train_val if should_stratify else None
        X_train, X_val, y_train, y_val = train_test_split(
            X_train_val,
            y_train_val,
            test_size=val_size_adjusted,
            random_state=self.random_state,
            stratify=stratify_split_val,
        )

        if self.verbose:
            print(f"\nDataset split:")
            print(f"  Train: {len(X_train)} samples ({len(X_train)/len(X):.1%})")
            print(f"  Val:   {len(X_val)} samples ({len(X_val)/len(X):.1%})")
            print(f"  Test:  {len(X_test)} samples ({len(X_test)/len(X):.1%})")

        # Fit on training data only (prevent data leakage!)
        # X_train already has label removed (it's in y_train), so pass y separately
        # Store the detected label type info before fitting
        original_label_column = self.label_column
        detected_label_type = (
            self._detect_label_type(self.label_column, y_train) if self.label_column else None
        )
        detected_is_classification = (
            (detected_label_type == "classification") if detected_label_type else None
        )

        # Temporarily set label_column to None for fitting (since X_train doesn't have it)
        self.label_column = None
        self.fit(X_train, y_train)

        # Restore label column name and set task type
        self.label_column = original_label_column
        if detected_label_type:
            self.label_type_ = detected_label_type
            self.is_classification_ = detected_is_classification

        # Transform all splits
        X_train_transformed = self.transform(X_train)
        X_val_transformed = self.transform(X_val)
        X_test_transformed = self.transform(X_test)

        if self.verbose:
            print(f"\nTransformed feature shape: {X_train_transformed.shape}")

        return (X_train_transformed, X_val_transformed, X_test_transformed, y_train, y_val, y_test)

    def _compute_feature_names(self):
        """Compute output feature names after transformation."""
        feature_names = []

        if self.numerical_features_:
            feature_names.extend(self.numerical_features_)

        if self.categorical_features_:
            cat_encoder = self.preprocessor_.named_transformers_["cat"]["onehot"]
            cat_feature_names = cat_encoder.get_feature_names_out(self.categorical_features_)
            feature_names.extend(cat_feature_names)

        self.feature_names_out_ = feature_names

    def get_feature_names_out(self) -> List[str]:
        """Get output feature names after transformation."""
        if self.feature_names_out_ is None:
            raise ValueError("Preprocessor must be fitted first.")
        return self.feature_names_out_

    def is_classification(self) -> bool:
        """
        Return whether the dataset is a classification task.

        Returns:
        --------
        bool
            True if classification, False if regression.

        Raises:
        -------
        ValueError
            If preprocessor has not been fitted yet.
        """
        if self.is_classification_ is None:
            raise ValueError("Preprocessor must be fitted first to determine task type.")
        return self.is_classification_
