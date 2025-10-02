from sklearn.preprocessing import StandardScaler, MinMaxScaler
import torch
import numpy as np
import pandas as pd
from ucimlrepo import fetch_ucirepo
from imblearn.over_sampling import SMOTENC
from sklearn.datasets import fetch_openml, load_breast_cancer, load_iris
from sklearn.model_selection import train_test_split
from data_poisoning import *

pd.set_option("display.max_columns", None)  # Show all columns
pd.set_option("display.max_rows", None)


class DataPreparation:
    def __init__(self, dataset_name: str) -> None:
        """Initialize the data preparation utility.

        Parameters
        ----------
        dataset_name : str
            Name of the dataset to be loaded. Supported values:
            "adult", "mushroom", "breast_cancer", "iris", "mnist",
            "diabates" (UCI CDC Diabetes), and "heart" (local CSV under
            `data/real/heart.csv`).
        """
        self.dataset_name = dataset_name

    def load(self):
        """Load the selected dataset into memory.

        Notes
        -----
        - For OpenML-based datasets, the data is fetched on demand.
        - Categorical/string columns are preserved for later one-hot encoding.
        - The `label_col` parameter is currently only used for the "heart"
          dataset where the target column is named "target". It is kept for
          compatibility and possible future use.

        Parameters
        ----------
        label_col : str
            Name of the label/target column (kept for compatibility).

        Raises
        ------
        ValueError
            If `dataset_name` is not one of the supported values.
        """
        lname = self.dataset_name.lower()
        if lname == "adult":
            ds = fetch_openml(name="adult", version=2, as_frame=True, parser="auto")
            X = ds.data.copy()
            y = (ds.target.astype(str).str.contains(">")).astype(int).to_numpy()
        elif lname == "mushroom":
            ds = fetch_openml(name="mushroom", version=1, as_frame=True, parser="auto")
            X = ds.data.copy()
            y = (ds.target.astype(str) == "p").astype(int).to_numpy()
        elif lname == "breast_cancer":
            ds = load_breast_cancer(as_frame=True)
            X = ds.frame.drop(columns=["target"]).copy()
            y = ds.target.to_numpy()
        elif lname == "iris":
            ds = load_iris(as_frame=True)
            X = ds.frame.drop(columns=["target"]).copy()
            y = ds.target.to_numpy()
        elif lname == "mnist":
            ds = fetch_openml(name="mnist_784", version=1, as_frame=True, parser="auto")
            X = ds.data.copy()
            y = (
                ds.target.astype(int).to_numpy()
                if ds.target.dtype.kind in "iu"
                else ds.target.astype(str).astype(int).to_numpy()
            )
        elif lname == "diabates":
            cdc_diabetes_health_indicators = fetch_ucirepo(id=891)
            X = cdc_diabetes_health_indicators.data.features
            y = cdc_diabetes_health_indicators.data.targets
        elif lname == "heart":
            X = pd.read_csv("data/real/heart.csv")
            y = X["target"]
            X = X.drop(columns=["target"])
        else:
            raise ValueError(f"Unknown dataset '{self.dataset_name}'")

        self.__X = X
        self.__y = y

    def run_preprocessing(
        self,
        splitting_perc_train_test: float = 0.8,
        splitting_perc_test_val: float = 0.5,
        features_percentage: float = 0.2,
        poisoning_percentage: float = 0.8,
        random_state: int = 42
    ):
        """Prepare data for modeling: encode, split, rebalance, poison, normalize.

        This method performs the complete preprocessing pipeline:
        1) One-hot encodes categorical/string features and converts booleans to 0/1
        2) Splits the dataset into train/test/val using the provided ratios
        3) Rebalances the training set via undersampling to the minority class
        4) Generates multiple poisoned variants of the training set
        5) Fits a scaler on non-binary numeric features and normalizes splits
        6) Converts inputs and labels to PyTorch tensors

        Parameters
        ----------
        label_col : str
            Name of the target column (kept for compatibility; not directly used
            after dataset-specific loading).
        splitting_perc_train_test : float, default=0.8
            Proportion of data reserved for training before creating validation.
        splitting_perc_test_val : float, default=0.5
            Proportion of the remaining non-train split assigned to the test set
            (the rest goes to validation).
        features_percentage : float, default=0.2
            Fraction of features to apply poisoning to.
        poisoning_percentage : float, default=0.8
            Fraction of samples/features affected by poisoning within the selected
            subset, depending on the poisoning function.

        Returns
        -------
        dict
            A dictionary with the following structure:
            {
              'clean': { 'X_train': Tensor, 'X_val': Tensor, 'X_test': Tensor, 'q': Tensor },
              'flipping': { 'X_train': Tensor, 'X_val': Tensor, 'X_test': Tensor, 'q': Tensor },
              'noise': { 'X_train': Tensor, 'X_val': Tensor, 'X_test': Tensor, 'q': Tensor },
              'nan': { 'X_train': Tensor, 'X_val': Tensor, 'X_test': Tensor, 'q': Tensor },
              'all': { 'X_train': Tensor, 'X_val': Tensor, 'X_test': Tensor, 'q': Tensor },
              'y_train': LongTensor, 'y_val': LongTensor, 'y_test': LongTensor
            }

        Notes
        -----
        - The scaler is fit separately for each split type (clean/flipping/noise/nan/all)
          on their respective training data to avoid leakage across types. Validation
          and test sets are transformed with the same scaler as their corresponding type.
        - Non-binary numeric columns are normalized; binary numeric columns are left
          unchanged.
        """
        # Apply one hot encoding to categorical and string columns
        categorical_cols = self.__X.select_dtypes(
            include=["object", "string", "category"]
        ).columns.tolist()
        self.__X = pd.get_dummies(
            self.__X,
            columns=categorical_cols,
            drop_first=False,
            dtype=int,
        )
        # Convert boolean columns in 0,1
        bool_cols = self.__X.select_dtypes(include="bool").columns
        self.__X[bool_cols] = self.__X[bool_cols].astype(int)

        # Shuffle Data and Split in Train and Test
        X_train, X_test, y_train, y_test = train_test_split(
            self.__X,
            self.__y,
            test_size=1 - splitting_perc_train_test,
            shuffle=True,
            stratify=self.__y,
            random_state=random_state
        )

        # Split in Test and Val
        X_test, X_val, y_test, y_val = train_test_split(
            X_test,
            y_test,
            test_size=1 - splitting_perc_test_val,
            shuffle=True,
            stratify=y_test,
            random_state=random_state
        )

        # Compute distribution on y values and rebalance X_train with undersampling techniques
        X_train, y_train = self.__undersampling(X_train, y_train)

        # Create Poisoned Splits
        data_dct = {
            'clean': {},
            'flipping': {},
            'noise': {},
            'nan': {},
            'all': {},
        }
        data_dct['clean']['X_train'] = X_train
        data_dct['clean']['q'] = torch.ones(X_train.shape[1])
        data_dct['flipping']['X_train'], data_dct['flipping']['q'] = flipping_poisoning(
            X=X_train,
            features_percentage=features_percentage,
            poisoning_percentage=poisoning_percentage,
            random_state=random_state
        )
        data_dct['noise']['X_train'], data_dct['noise']['q'] = noise_poisoning(
            X=X_train,
            features_percentage=features_percentage,
            poisoning_percentage=poisoning_percentage,
            random_state=random_state
        )
        data_dct['nan']['X_train'], data_dct['nan']['q'] = incompleteness_poisoning(
            X=X_train,
            features_percentage=features_percentage,
            poisoning_percentage=poisoning_percentage,
            random_state=random_state
        )
        data_dct['all']['X_train'], data_dct['all']['q'] = combined_poisoning(
            X=X_train,
            features_percentage=features_percentage,
            flipping_percentage=poisoning_percentage,
            noise_percentage=poisoning_percentage,
            incompleteness_percentage=poisoning_percentage,
            random_state=random_state
        )

        # Select continuous features to be normalized
        numeric_cols = X_train.select_dtypes(include=["number"]).columns
        non_binary_cols = [
            col
            for col in numeric_cols
            if not set(X_train[col].dropna().unique()).issubset({0, 1})
        ]

        #Normalize features and convert to tensors
        for type in data_dct.keys():
            self.__set_scaler(feature_to_norm=non_binary_cols, df=data_dct[type]['X_train'])
            data_dct[type]['X_train'] = torch.tensor(data=self.__normalize(feature_to_norm=non_binary_cols, df=data_dct[type]['X_train']).to_numpy(), dtype=torch.float32)
            data_dct[type]['X_val'] = torch.tensor(data=self.__normalize(feature_to_norm=non_binary_cols, df=X_val.copy()).to_numpy(), dtype=torch.float32)
            data_dct[type]['X_test'] = torch.tensor(data=self.__normalize(feature_to_norm=non_binary_cols, df=X_test.copy()).to_numpy(), dtype=torch.float32)
            
            data_dct[type]['q'] = torch.tensor(data_dct[type]['q'], dtype=torch.float32)

        data_dct['y_train'] = torch.tensor(y_train.to_numpy() if not isinstance(y_train, np.ndarray) else y_train, dtype=torch.long)
        data_dct['y_val'] = torch.tensor(y_val.squeeze().to_numpy() if not isinstance(y_val.squeeze(), np.ndarray) else y_val.squeeze(), dtype=torch.long)
        data_dct['y_test'] = torch.tensor(y_test.squeeze().to_numpy() if not isinstance(y_test.squeeze(), np.ndarray) else y_test.squeeze(), dtype=torch.long)

        return data_dct


    def __undersampling(self, X: pd.DataFrame, y: pd.Series):
        """Undersample the majority classes to match the minority class size.

        The function operates class-wise and samples without replacement so that
        each class ends up with the same number of samples as the minority class.

        Parameters
        ----------
        X : pandas.DataFrame
            Feature matrix aligned with `y`.
        y : pandas.Series
            Target vector. If not a Series, it will be converted and aligned to `X`.

        Returns
        -------
        tuple[pandas.DataFrame, pandas.Series]
            The balanced feature matrix and target vector.

        Notes
        -----
        - If inputs are invalid or class counts are empty/non-positive, the inputs
          are returned unchanged.
        - If `self.random_state` is set, it is used to seed the sampler/shuffler.
        """
        # Validate inputs
        if X is None or y is None:
            return X, y
        if len(X) == 0 or len(y) == 0 or len(X) != len(y):
            return X, y

        # Ensure y is a Series aligned with X
        if not isinstance(y, pd.Series):
            if isinstance(y, pd.DataFrame):
                y = pd.Series(y.squeeze())
            else:
                y = pd.Series(y)
        y = y.reset_index(drop=True)
        X = X.reset_index(drop=True)

        # Compute class distribution and target size (minimum class count)
        class_counts = y.value_counts(dropna=False)
        if class_counts.empty:
            return X, y
        min_size = int(class_counts.min())
        if min_size <= 0:
            return X, y

        random_state = getattr(self, "random_state", None)

        # Sample indices per class down to the minority size
        sampled_indices_parts: list[np.ndarray] = []
        for class_value in class_counts.index:
            class_indices = y[y == class_value].index.to_numpy()
            if len(class_indices) > min_size:
                rng = np.random.default_rng(random_state)
                sampled_idx = rng.choice(class_indices, size=min_size, replace=False)
            else:
                sampled_idx = class_indices
            sampled_indices_parts.append(sampled_idx)

        # Concatenate sampled indices and shuffle
        all_indices = np.concatenate(sampled_indices_parts)
        if random_state is not None:
            rng = np.random.default_rng(random_state)
            rng.shuffle(all_indices)
        else:
            np.random.shuffle(all_indices)

        X_bal = X.iloc[all_indices].reset_index(drop=True)
        y_bal = y.iloc[all_indices].reset_index(drop=True)

        return X_bal, y_bal

    def __set_scaler(self, feature_to_norm: list[str], df: pd.DataFrame):
        """Fit a `StandardScaler` on the provided DataFrame columns.

        Parameters
        ----------
        feature_to_norm : list[str]
            List of column names to be normalized.
        df : pandas.DataFrame
            Training DataFrame that will be used to fit the scaler.
        """
        # Initialize scaler
        self.scaler = StandardScaler().fit(df[feature_to_norm])

    def __normalize(self, feature_to_norm: list[str], df: pd.DataFrame):
        """Normalize the given DataFrame columns using the fitted scaler.

        The method fits and transforms the provided columns using `self.scaler`.
        It mutates and returns the input DataFrame for convenience.

        Parameters
        ----------
        feature_to_norm : list[str]
            List of column names to be normalized.
        df : pandas.DataFrame
            DataFrame whose specified columns will be transformed.

        Returns
        -------
        pandas.DataFrame
            The same DataFrame with normalized columns applied in-place.
        """
        # Fit on training data and transform
        df[feature_to_norm] = self.scaler.transform(df[feature_to_norm])

        return df