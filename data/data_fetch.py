import numpy as np
import inspect
from sklearn import datasets
from sklearn.utils.multiclass import type_of_target
from sklearn.datasets import fetch_openml
import openml
import pandas as pd


# -------------------------------
# Utility: multiclass → binary
# -------------------------------
def binarize_majority_vs_rest(y):
    y = np.asarray(y)
    classes, counts = np.unique(y, return_counts=True)
    majority = classes[np.argmax(counts)]
    return (y == majority).astype(int), majority


# -------------------------------
# 1. Iterate sklearn datasets
# -------------------------------
def iter_sklearn_tabular_datasets(
    verbose=False,
    min_samples=0,
    max_samples=None,
    min_features=0,
    max_features=None,
    allow_multiclass_binarization=True,
    allow_missing=False,  # 👈 NEW PARAM
):
    skip = {
        "load_diabetes",            # regression
        "fetch_california_housing", # regression
        "fetch_20newsgroups",
        "fetch_rcv1",
        "fetch_kddcup99",
        "fetch_lfw_people",
        "fetch_lfw_pairs",
        "fetch_olivetti_faces",
        "load_sample_images",
        "load_sample_image",
    }

    for name, func in inspect.getmembers(datasets, inspect.isfunction):
        if not (name.startswith("load_") or name.startswith("fetch_")):
            continue
        if name in skip:
            continue

        if verbose:
            print(f"[sklearn] trying {name}")

        try:
            bunch = func()
        except Exception as e:
            if verbose:
                print(f"  -> error calling {name}: {repr(e)}")
            continue

        X = getattr(bunch, "data", None)
        y = getattr(bunch, "target", None)
        if X is None or y is None:
            if verbose:
                print("  -> missing data/target, skipping")
            continue

        X = np.asarray(X)
        y = np.asarray(y)

        # Must be tabular
        if X.ndim != 2:
            if verbose:
                print(f"  -> X.ndim={X.ndim} (not 2), skipping")
            continue

        # Handle missing values (for numeric dtypes)
        if not allow_missing:
            X_has_nan = np.issubdtype(X.dtype, np.number) and np.isnan(X).any()
            y_has_nan = np.issubdtype(y.dtype, np.number) and np.isnan(y).any()
            if X_has_nan or y_has_nan:
                if verbose:
                    print(f"  -> contains missing values (X_nan={X_has_nan}, y_nan={y_has_nan}), skipping")
                continue

        n_samples, n_features = X.shape

        # Apply constraints
        if n_samples < min_samples:
            if verbose:
                print(f"  -> n_samples={n_samples} < min_samples={min_samples}, skipping")
            continue
        if max_samples is not None and n_samples > max_samples:
            if verbose:
                print(f"  -> n_samples={n_samples} > max_samples={max_samples}, skipping")
            continue
        if n_features < min_features:
            if verbose:
                print(f"  -> n_features={n_features} < min_features={min_features}, skipping")
            continue
        if max_features is not None and n_features > max_features:
            if verbose:
                print(f"  -> n_features={n_features} > max_features={max_features}, skipping")
            continue

        tgt = type_of_target(y)
        u = np.unique(y)

        if tgt == "binary":
            # Skip if any of the classes is NaN
            if any(pd.isna(u)):
                if verbose:
                    print("  -> one of the classes is NaN, skipping")
                continue

            # Handle string targets (numpy strings, python str, or object dtype)
            if (
                np.issubdtype(u.dtype, np.str_)
                or u.dtype == object
                or any(isinstance(c, str) for c in u)
            ):
                if u.size == 2:
                    label_to_int = {u[0]: 0, u[1]: 1}
                else:
                    label_to_int = {cls: i for i, cls in enumerate(u)}
                y = np.vectorize(label_to_int.get)(y)

            if verbose:
                print(f"  -> ACCEPTED (binary), shape={X.shape}")
            yield f"sk_{name}", pd.DataFrame(X), pd.DataFrame(y)

        elif tgt == "multiclass" and allow_multiclass_binarization:
            y_bin, maj = binarize_majority_vs_rest(y)
            if verbose:
                print(
                    f"  -> multiclass -> binarized majority-vs-rest "
                    f"(majority class={maj}), shape={X.shape}"
                )
            yield f"sk_{name}_majvsrest", pd.DataFrame(X), pd.DataFrame(y_bin)

        else:
            if verbose:
                print(f"  -> target_type={tgt}, not accepted")
            continue


# -------------------------------
# 2. Iterate OpenML datasets
# -------------------------------
def iter_openml_tabular_datasets(
    verbose=False,
    min_samples=0,
    max_samples=None,
    min_features=0,
    max_features=None,
    allow_multiclass_binarization=True,
    max_datasets=None,
    allow_missing=False,   
):

    # Fetch metadata table from OpenML
    meta = openml.datasets.list_datasets(output_format="dataframe")

    # Base filters
    df = meta[
        (meta["status"] == "active") &
        (meta["NumberOfClasses"] >= 2) &
        (meta["NumberOfInstances"] >= min_samples) &
        (meta["NumberOfFeatures"] >= min_features)
    ]

    if max_samples is not None:
        df = df[df["NumberOfInstances"] <= max_samples]
    if max_features is not None:
        df = df[df["NumberOfFeatures"] <= max_features]

    # Filter out datasets with missing values at metadata level (if requested)
    if not allow_missing and "NumberOfMissingValues" in df.columns:
        df = df[df["NumberOfMissingValues"] == 0]

    if verbose:
        print(f"OpenML: {len(df)} datasets match metadata constraints ")

    count = 0

    for did, row in df.iterrows():
        if max_datasets is not None and count >= max_datasets:
            return

        try:
            dataset = openml.datasets.get_dataset(did)

            # Use dataframe format to easily inspect NaNs
            X, y, categorical, attribute_names = dataset.get_data(
                target=dataset.default_target_attribute,
                dataset_format="dataframe",
                include_ignore_attribute=True
            )
        except Exception as e:
            if verbose:
                print(f"Skipping dataset {did} ({row['name']}):", repr(e))
            continue

        # Check for missing values at data level (robust)
        if not allow_missing:
            has_missing = X.isna().values.any() or pd.isna(y).values.any()
            if has_missing:
                if verbose:
                    print(f"Skipping {did} ({row['name']}): contains missing values")
                continue

        # Convert to numpy for downstream pipelines
        if hasattr(X, "to_numpy"):
            X = X.to_numpy()
        else:
            X = np.asarray(X)

        y = np.asarray(y)

        if X.ndim != 2:
            continue

        tgt = type_of_target(y)
        u = np.unique(y)

        if tgt == "binary" or u.size <= 2:
            print("Binary Target")

            # Skip if any of the classes is NaN
            if any(pd.isna(u)):
                if verbose:
                    print(f"Skipping {did} ({row['name']}): one of the classes is NaN")
                continue

            print(type(u[0]))

            # Handle string targets (numpy strings, python str, or object dtype)
            if (
                np.issubdtype(u.dtype, np.str_)
                or u.dtype == object
                or any(isinstance(c, str) for c in u)
            ):
                if u.size == 2:
                    label_to_int = {u[0]: 0, u[1]: 1}
                else:
                    label_to_int = {cls: i for i, cls in enumerate(u)}
                y = np.vectorize(label_to_int.get)(y)

            yield f"oml_{did}_{row['name']}", pd.DataFrame(X), pd.DataFrame(y)
            count += 1

        elif (tgt == "multiclass" or u.size > 2) and allow_multiclass_binarization:
            print("Multiclass Target")
            classes, counts = np.unique(y, return_counts=True)
            majority = classes[np.argmax(counts)]
            y_bin = (y == majority).astype(int)
            yield f"oml_{did}_{row['name']}_majvsrest", pd.DataFrame(X), pd.DataFrame(y_bin)


# -------------------------------
# 3. Unified iterator: sklearn + OpenML
# -------------------------------
def iter_all_binary_tabular_datasets(
    *,
    verbose=False,
    min_samples=0,
    max_samples=None,
    min_features=0,
    max_features=None,
    allow_multiclass_binarization=True,
    allow_missing=False,
    max_openml=None,
):
    # sklearn
    for name, X, y in iter_sklearn_tabular_datasets(
        verbose=verbose,
        min_samples=min_samples,
        max_samples=max_samples,
        min_features=min_features,
        max_features=max_features,
        allow_multiclass_binarization=allow_multiclass_binarization,
        allow_missing=allow_missing
    ):
        yield name, X, y

    # OpenML
    for name, X, y in iter_openml_tabular_datasets(
        verbose=verbose,
        min_samples=min_samples,
        max_samples=max_samples,
        min_features=min_features,
        max_features=max_features,
        allow_multiclass_binarization=allow_multiclass_binarization,
        max_datasets=max_openml,
        allow_missing=allow_missing
    ):
        yield name, X, y