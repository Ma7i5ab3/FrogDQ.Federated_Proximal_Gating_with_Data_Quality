from sklearn.preprocessing import StandardScaler
import torch
import numpy as np
import pandas as pd
from ucimlrepo import fetch_ucirepo
from imblearn.over_sampling import SMOTENC
from sklearn.datasets import fetch_openml, load_breast_cancer, load_iris
from sklearn.model_selection import train_test_split


pd.set_option("display.max_columns", None)  # Show all columns
pd.set_option("display.max_rows", None) 

class DataPreparation:
    def __init__(self, dataset_name: str) -> None:
        self.dataset_name = dataset_name
    
    def load(self, label_col: str):
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
        else:
            raise ValueError(f"Unknown dataset '{self.dataset_name}'")
        
        self.__X = X
        self.__y = y
 
    def run_preprocessing(self, label_col: str, splitting_perc_train_test: float = 0.8, splitting_perc_test_val: float = 0.5):
        #Apply one hot encoding to categorical and string columns
        categorical_cols = self.__X.select_dtypes(include=["object", "string", "category"]).columns.tolist()
        self.__X = pd.get_dummies(
                self.__X,
                columns=categorical_cols,
                drop_first=False,
                dtype=int,
            )
        #Convert boolean columns in 0,1
        bool_cols = self.__X.select_dtypes(include="bool").columns
        self.__X[bool_cols] = self.__X[bool_cols].astype(int)

        #Shuffle Data and Split in Train and Test
        X_train, X_test, y_train, y_test = train_test_split(
            self.__X, self.__y, test_size=1-splitting_perc_train_test, random_state=42, shuffle=True
        )

        #Split in Test and Val
        X_test, X_val, y_test, y_val = train_test_split(
            X_test, y_test, test_size=1-splitting_perc_test_val, random_state=42, shuffle=True
        )

        #Compute distribution on y values and rebalance X_train with undersampling techniques
        X_train, y_train = self.__undersampling(X_train, y_train)

        #Select continuous features to be normalized
        numeric_cols = X_train.select_dtypes(include=["number"]).columns
        non_binary_cols = [
            col for col in numeric_cols
            if not set(X_train[col].dropna().unique()).issubset({0, 1})
        ]

        # TODO: Data poisoning before normalization

        self.__set_scaler(feature_to_norm=non_binary_cols, df=X_train)
        self.__normalize(feature_to_norm=non_binary_cols, df=X_train)
        self.__normalize(feature_to_norm=non_binary_cols, df=X_val)
        self.__normalize(feature_to_norm=non_binary_cols, df=X_test)

        # TODO: Data poisoning after normalization

        #Conversion to tensors
        '''X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
        y_train_tensor = torch.tensor(X_train, dtype=torch.long)
        X_test_tensor = torch.tensor(X_train, dtype=torch.float32)
        y_test_tensor = torch.tensor(X_train, dtype=torch.float32)
        X_val_tensor = torch.tensor(X_train, dtype=torch.float32)
        y_val_tensor = torch.tensor(X_train, dtype=torch.float32)'''

        return X_train, y_train, X_val, y_val, X_test, y_test

    
    def __undersampling(self, X: pd.DataFrame, y: pd.Series):
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
        # Initialize scaler
        self.scaler = StandardScaler().fit(df[feature_to_norm])
    
    def __normalize(self, feature_to_norm: list[str], df: pd.DataFrame):
        # Fit on training data and transform
        df[feature_to_norm] = self.scaler.fit_transform(df[feature_to_norm])
   

    def __to_tensor(self, obj: pd.DataFrame | list[pd.DataFrame]):
        if isinstance(obj, list):
            for index, df in enumerate(obj):
                obj[index] = torch.tensor(df.values.astype(np.float32), dtype=torch.float32)
            return obj
        else:
            return torch.tensor(obj.values.astype(np.float32), dtype=torch.float32)



