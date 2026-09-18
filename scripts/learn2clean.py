#!/usr/bin/env python3
"""
Learn2Clean Data Preparation Baseline
Python port of the official reference implementation from:

  Laure Berti-Equille. "Learn2Clean: Optimizing the Sequence of Tasks for Web
  Data Preparation." WWW '19, San Francisco, pp. 2580-2586.
  https://doi.org/10.1145/3308558.3313602

Upstream source: https://github.com/LaureBerti/Learn2Clean
                 (Learn2Clean_V1/python-package/learn2clean, v0.2.1)

The upstream package targets Python 3.6 / numpy 1.14 / pandas 0.23 and pulls in
several dependencies that no longer build on this stack (impyute, fancyimpute,
py_stringsimjoin, py_stringmatching, tdda, sklearn-contrib-py-earth). This file
ports the same algorithms to numpy 2 / pandas 2 / scikit-learn 1.7 with no new
dependencies. Every deviation from upstream is flagged with an "L2C-PORT" note.

Architecture (Figure 1 of the paper) — 18 preparation/cleaning methods + 1 goal:
  Normalization        : MM, ZS, DS
  Feature selection    : MR, WR, LC, Tree
  Imputation           : MICE, EM, KNN, MF
  Outlier detection    : IQR, LOF, ZSB
  Inconsistency check  : CC, PC
  Deduplication        : ED, AD
  Goal state (ML model): CART / LDA / NB    (classification, accuracy)
                         LASSO / OLS / MARS (regression, MSE)
                         HCA / KMEANS       (clustering, silhouette)

Q-learning (Section 3) explores the state-action graph encoded in the reward
matrix R; `show_traverse` then executes the greedy traversal from every possible
starting state and keeps the pipeline whose quality metric is optimal.

Input:  data_poisoned/{ar,nar}/          (poisoned CSV + mask from poison_data.py)
Output: data_cleaned_learn2clean/{ar,nar}/ (cleaned CSV + residual mask + metrics)

Unlike the Saga++ and CP baselines, Learn2Clean is *not* shape-preserving: outlier
detection, deduplication and consistency checking remove rows, and feature
selection removes columns. The cleaned CSV, its residual mask and the surviving
row/column labels recorded in <dataset>_pipeline.pkl are therefore all aligned to
the reduced frame.

Note: the target column (cls_*, reg_*) is passed as `target_prepare` so that the
upstream `exclude` mechanism never drops or transforms it.
"""

import copy
import os
import random
import re
import sys
import time
import tracemalloc
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
# Upstream console output
# ---------------------------------------------------------------------------
# L2C-PORT: the reference implementation narrates every step with print().
# Those messages are preserved verbatim but routed to loguru's DEBUG level, so
# they stay available (--verbose) without flooding the pipeline log.

def _p(*args) -> None:
    logger.debug(" ".join(str(a) for a in args))


# ---------------------------------------------------------------------------
# Module-level knobs set from the CLI / config.yaml
# ---------------------------------------------------------------------------
# L2C-PORT: upstream feeds the goal model `dataset['train'].select_dtypes(['number'])`,
# so a fully categorical dataset yields zero features and no strategy can ever be
# scored (accuracy is None everywhere). ENCODE_CATEGORICALS ordinal-encodes the
# categorical columns *for scoring only* — the written CSV keeps the original
# labels. This mirrors `_encode_for_model` in scripts/saga.py. Set it to False
# for strict upstream behaviour.
ENCODE_CATEGORICALS = True


# ---------------------------------------------------------------------------
# Column-type identification (mirrors saga.py / data_preparation_pipeline.py)
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


def _model_matrix(df: pd.DataFrame, target: Optional[str]) -> pd.DataFrame:
    """
    Feature matrix seen by a goal-state model.

    Upstream: ``dataset['train'].select_dtypes(['number']).dropna()`` followed by
    dropping the target column. With ENCODE_CATEGORICALS the categorical columns
    are ordinal-encoded first; NaN stays NaN so that ``dropna()`` keeps punishing
    strategies that leave missing values behind (which is what drives the reward).
    """
    X = df
    if target is not None and target in X.columns:
        X = X.drop(columns=[target])

    if ENCODE_CATEGORICALS:
        X = X.copy()
        for col in X.columns:
            if X[col].dtype == object or isinstance(X[col].dtype, pd.CategoricalDtype):
                codes = pd.Categorical(X[col]).codes.astype(float)
                codes[codes < 0] = np.nan  # -1 is pandas' NaN sentinel
                X[col] = codes
        X = X.select_dtypes(["number"])
    else:
        X = X.select_dtypes(["number"])

    return X.dropna()


def _safe_k(k: int, y: pd.Series, n_rows: int) -> int:
    """Largest usable number of CV folds (>= 2) for this label distribution."""
    if len(y) == 0:
        return 2
    counts = y.value_counts()
    k = int(min(k, int(counts.min()), max(2, n_rows // 2)))
    return max(2, k)


def _has_test(dataset: Dict) -> bool:
    """True when a genuine held-out frame was supplied in dataset['test']."""
    test = dataset.get("test")
    return isinstance(test, pd.DataFrame) and not test.empty


# ===========================================================================
# learn2clean.normalization.normalizer
# ===========================================================================

class Normalizer:
    """
    Normalize the numerical variables of the dataset.

    * strategy: 'ZS' z-score, 'MM' MinMax scaling, 'DS' decimal scaling,
      'Log10' log10 scaling
    * exclude : str, name of the variable excluded from normalization
    """

    def __init__(self, dataset, strategy="ZS", exclude=None, verbose=False):
        self.dataset = dataset
        self.strategy = strategy
        self.exclude = exclude
        self.verbose = verbose

    def get_params(self, deep=True):
        return {"strategy": self.strategy, "exclude": self.exclude, "verbose": self.verbose}

    def ZS_normalization(self, dataset):
        # Normalize numeric columns with Z-score normalisation
        d = dataset
        if self.verbose:
            _p("ZS normalizing... ")

        X = dataset.select_dtypes(["number"]).copy()
        Y = dataset.select_dtypes(["object"])
        Z = dataset.select_dtypes(["datetime64"])

        for column in X.columns:
            X[column] = X[column] - X[column].mean()
            std = X[column].std()
            # L2C-PORT: guard the constant-column division upstream leaves to numpy
            if std and not np.isnan(std):
                X[column] = X[column] / std

        df = X.join(Y).join(Z)
        if self.exclude in list(df.columns.values):
            df[str(self.exclude)] = d[str(self.exclude)].values
        return df.sort_index()

    def MM_normalization(self, dataset):
        # Normalize numeric columns with MinMax normalization
        from sklearn.preprocessing import MinMaxScaler

        d = dataset
        if self.verbose:
            _p("MM normalizing...")

        Xf = dataset.select_dtypes(["number"])
        Y = dataset.select_dtypes(["object"])
        Z = dataset.select_dtypes(["datetime64"])

        # L2C-PORT: upstream scales only `Xf.dropna()` and concatenates the rows
        # carrying any missing numeric value back untouched, so a single column
        # ends up holding two scales at once (sick AR: 2249 rows mapped into
        # [0, 1] sitting next to 312 raw ones reaching 1.5). It also silently
        # normalizes nothing at all when no row is complete, which on
        # data_poisoned is the majority of the AR datasets. MinMaxScaler
        # disregards NaN when fitting and preserves it when transforming, so the
        # statistics are taken per column over the observed values and every row
        # is scaled — exactly the semantics ZS_normalization already has.
        if Xf.shape[1] == 0 or Xf.dropna(how="all").empty:
            scaled_Xf = Xf
        else:
            scaled_values = MinMaxScaler().fit_transform(Xf)
            scaled_Xf = pd.DataFrame(scaled_values, index=Xf.index, columns=Xf.columns)

        df = scaled_Xf.join(Y).join(Z)
        if self.exclude in list(df.columns.values):
            df[str(self.exclude)] = d[str(self.exclude)].values
        return df.sort_index()

    def DS_normalization(self, dataset):
        # Decimal scaling, implemented upstream with a 10-quantile transform
        from sklearn.preprocessing import QuantileTransformer

        d = dataset
        if self.verbose:
            _p("DS normalizing...")

        Xf = dataset.select_dtypes(["number"])
        Y = dataset.select_dtypes(["object"])
        Z = dataset.select_dtypes(["datetime64"])

        # L2C-PORT: same fix as MM_normalization — upstream transforms only the
        # complete rows and re-attaches the others raw, leaving two scales inside
        # one column. QuantileTransformer disregards NaN when fitting and
        # maintains it when transforming, so the quantiles are estimated per
        # column over the observed values and every row is transformed.
        if Xf.shape[1] == 0 or Xf.dropna(how="all").empty:
            scaled_Xf = Xf
        else:
            # L2C-PORT: n_quantiles must not exceed the sample count in sklearn >= 1.0
            n_quantiles = max(int(min(10, len(Xf))), 2)
            qt = QuantileTransformer(n_quantiles=n_quantiles, random_state=0)
            scaled_values = qt.fit_transform(Xf)
            scaled_Xf = pd.DataFrame(scaled_values, index=Xf.index, columns=Xf.columns)

        df = scaled_Xf.join(Y).join(Z)
        if self.exclude in list(df.columns.values):
            df[str(self.exclude)] = d[str(self.exclude)].values
        return df.sort_index()

    def Log10_normalization(self, dataset):
        # Normalize numeric columns with log10 scaling
        d = dataset
        if self.verbose:
            _p("Log10 normalizing...")

        X = dataset.select_dtypes(["number"]).copy()
        Y = dataset.select_dtypes(["object"])
        Z = dataset.select_dtypes(["datetime64"])

        for column in X.columns:
            X[column] = np.around(np.log10(X[column].max())) + 1

        df = X.join(Y).join(Z)
        if self.exclude in list(df.columns.values):
            df[str(self.exclude)] = d[str(self.exclude)].values
        return df

    def transform(self):
        normd = self.dataset
        start_time = time.time()
        _p(">>Normalization ")

        for key in ["train", "test"]:
            if not isinstance(self.dataset[key], dict):
                d = self.dataset[key]
                _p("* For", key, "dataset")

                if self.strategy == "DS":
                    dn = self.DS_normalization(d)
                elif self.strategy == "ZS":
                    dn = self.ZS_normalization(d)
                elif self.strategy == "MM":
                    dn = self.MM_normalization(d)
                elif self.strategy == "Log10":
                    dn = self.Log10_normalization(d)
                else:
                    raise ValueError("The normalization function should be MM, ZS, DS or Log10")

                if self.exclude in list(pd.DataFrame(d).columns.values):
                    dn[self.exclude] = d[self.exclude]

                normd[key] = dn
                _p("...", key, "dataset")
            else:
                normd[key] = self.dataset[key]
                _p("No", key, "dataset, no normalization")

        _p("Normalization done -- CPU time: %s seconds" % (time.time() - start_time))
        return normd


# ===========================================================================
# learn2clean.imputation.imputer
# ===========================================================================

class Imputer:
    """
    Replace or remove the missing values using a particular strategy.

    * strategy:
        - 'EM'    : expectation-maximization (numerical only)
        - 'MICE'  : multivariate imputation by chained equations (numerical only)
        - 'KNN'   : k-nearest-neighbour imputation, k=4 (numerical only)
        - 'MF'    : most frequent value (numerical and categorical)
        - 'RAND'  : random value from the variable domain
        - 'MEAN' / 'MEDIAN' : numerical only
        - 'DROP'  : remove every row holding at least one missing value
    """

    def __init__(self, dataset, strategy="DROP", verbose=False, exclude=None, threshold=None):
        self.dataset = dataset
        self.strategy = strategy
        self.verbose = verbose
        self.threshold = threshold
        self.exclude = exclude

    def get_params(self, deep=True):
        return {
            "strategy": self.strategy,
            "verbose": self.verbose,
            "exclude": self.exclude,
            "threshold": self.threshold,
        }

    def mean_imputation(self, dataset):
        # replace missing numerical values by the mean of the variable
        df = dataset
        if dataset.select_dtypes(["number"]).isnull().sum().sum() > 0:
            X = dataset.select_dtypes(["number"]).copy()
            for i in X.columns:
                X[i] = X[i].fillna(X[i].mean())
            Z = dataset.select_dtypes(exclude=["number"])
            df = X.join(Z)
        return df

    def median_imputation(self, dataset):
        # replace missing numerical values by the median of the variable
        df = dataset
        if dataset.select_dtypes(["number"]).isnull().sum().sum() > 0:
            X = dataset.select_dtypes(["number"]).copy()
            for i in X.columns:
                X[i] = X[i].fillna(X[i].median())
            Z = dataset.select_dtypes(include=["object"])
            df = X.join(Z)
        return df

    def NaN_drop(self, dataset):
        # drop observations with missing values
        _p("Dataset size reduced from", len(dataset), "to", len(dataset.dropna()))
        return dataset.dropna()

    def MF_most_frequent_imputation(self, dataset):
        # replace missing values by the most frequent value of the variable
        dataset = dataset.copy()  # L2C-PORT: upstream mutates the caller's frame
        for i in dataset.columns:
            counts = dataset[i].value_counts()
            if counts.empty:
                continue
            mfv = counts.idxmax()
            dataset[i] = dataset[i].fillna(mfv)
            if self.verbose:
                _p("Most frequent value for ", i, "is:", mfv)
        return dataset

    def NaN_random_replace(self, dataset):
        # replace missing data with a random observation with data
        dataset = dataset.copy()
        for col in dataset.columns:
            observed = dataset[col].dropna()
            if observed.empty:
                continue
            na_idx = dataset.index[dataset[col].isna()]
            if len(na_idx) == 0:
                continue
            draws = observed.sample(n=len(na_idx), replace=True, random_state=0).to_numpy()
            dataset.loc[na_idx, col] = draws
        return dataset

    def KNN_imputation(self, dataset, k=4):
        # Nearest-neighbour imputation weighting samples by the mean squared
        # difference on the features both rows observe.
        # L2C-PORT: fancyimpute.KNN(k=4) -> sklearn KNNImputer(n_neighbors=4,
        # weights="distance"), the maintained equivalent of the same estimator.
        from sklearn.impute import KNNImputer

        df = dataset
        if dataset.select_dtypes(["number"]).isnull().sum().sum() > 0:
            X = dataset.select_dtypes(["number"])
            if X.shape[1] == 0 or X.dropna(how="all").empty:
                return df
            imputed = KNNImputer(n_neighbors=k, weights="distance").fit_transform(X)
            X = pd.DataFrame(imputed, index=X.index, columns=X.columns)
            Z = dataset.select_dtypes(include=["object"])
            df = X.join(Z)
        return df

    def MICE_imputation(self, dataset):
        # Multivariate Imputation by Chained Equations, only suitable for data
        # Missing At Random (MAR).
        # L2C-PORT: impyute.mice -> sklearn IterativeImputer, the same chained
        # -equations scheme (van Buuren & Groothuis-Oudshoorn, 2011).
        from sklearn.experimental import enable_iterative_imputer  # noqa: F401
        from sklearn.impute import IterativeImputer

        df = dataset
        if dataset.select_dtypes(["number"]).isnull().sum().sum() > 0:
            Xdf = dataset.select_dtypes(["number"])
            if Xdf.shape[1] == 0 or Xdf.dropna(how="all").empty:
                return df
            X = IterativeImputer(random_state=0, max_iter=10).fit_transform(Xdf)
            Z = dataset.select_dtypes(include=["object"])
            X = pd.DataFrame(X, index=Xdf.index, columns=Xdf.columns)
            df = X.join(Z)
        return df

    def EM_imputation(self, dataset, loops=50):
        # Imputes the given data using expectation maximization.
        # E-step: calculates the expected complete data log likelihood ratio.
        # L2C-PORT: direct port of impyute.imputation.cs.em (univariate Gaussian
        # EM per column), which is the function upstream calls.
        df = dataset
        if dataset.select_dtypes(["number"]).isnull().sum().sum() == 0:
            return df

        Xdf = dataset.select_dtypes(["number"])
        if Xdf.shape[1] == 0:
            return df

        rng = random.Random(0)
        X = Xdf.to_numpy(dtype=float, copy=True)
        nan_xy = np.argwhere(np.isnan(X))

        for x_i, y_i in nan_xy:
            col = X[:, int(y_i)]
            observed = col[~np.isnan(col)]
            if observed.size == 0:
                continue
            mu, std = observed.mean(), observed.std()
            col[x_i] = rng.gauss(mu, std)
            previous = 1.0
            for _ in range(loops):
                observed = col[~np.isnan(col)]
                mu, std = observed.mean(), observed.std()
                col[x_i] = rng.gauss(mu, std)
                delta = (col[x_i] - previous) / previous if previous else np.inf
                if abs(delta) < 0.1:
                    break
                previous = col[x_i]

        X = pd.DataFrame(X, index=Xdf.index, columns=Xdf.columns)
        Z = dataset.select_dtypes(include=["object"])
        return X.join(Z)

    def transform(self):
        start_time = time.time()
        _p(">>Imputation ")
        impd = self.dataset

        for key in ["train", "test"]:
            if not isinstance(self.dataset[key], dict):
                d = self.dataset[key].copy()
                _p("* For", key, "dataset")

                total_missing_before = d.isnull().sum().sum()
                _p("Before imputation:")

                if total_missing_before == 0:
                    _p("No missing values in the given data")
                    continue

                _p("Total", total_missing_before, "missing values in",
                   d.columns[d.isnull().any()].tolist())

                if self.strategy == "EM":
                    dn = self.EM_imputation(d)
                elif self.strategy == "MICE":
                    dn = self.MICE_imputation(d)
                elif self.strategy == "KNN":
                    dn = self.KNN_imputation(d)
                elif self.strategy == "RAND":
                    dn = self.NaN_random_replace(d)
                elif self.strategy == "MF":
                    dn = self.MF_most_frequent_imputation(d)
                elif self.strategy == "MEAN":
                    dn = self.mean_imputation(d)
                elif self.strategy == "MEDIAN":
                    dn = self.median_imputation(d)
                elif self.strategy == "DROP":
                    dn = self.NaN_drop(d)
                else:
                    raise ValueError(
                        "Strategy invalid. Please choose between 'EM', 'MICE', "
                        "'KNN', 'RAND', 'MF', 'MEAN', 'MEDIAN', or 'DROP'"
                    )

                impd[key] = dn
                _p("After imputation:", impd[key].isnull().sum().sum(), "missing values")
            else:
                _p("No", key, "dataset, no imputation")

        _p("Imputation done -- CPU time: %s seconds" % (time.time() - start_time))
        return impd


# ===========================================================================
# learn2clean.feature_selection.feature_selector
# ===========================================================================

class Feature_selector:
    """
    Select the most relevant subset of variables.

    * strategy: 'MR' missing ratio, 'LC' linear correlation, 'WR' wrapper subset
      evaluator, 'Tree' tree-based selection ('VAR', 'L1', 'IMP', 'SVC' are also
      available upstream but are outside the Learn2Clean action space)
    * threshold: float, default = 0.3
    * exclude  : str, variable kept whatever the selection decides
    """

    def __init__(self, dataset, strategy="LC", exclude=None, threshold=0.3, verbose=False):
        self.dataset = dataset
        self.strategy = strategy
        self.exclude = exclude
        self.threshold = threshold
        self.verbose = verbose

    def get_params(self, deep=True):
        return {
            "strategy": self.strategy,
            "exclude": self.exclude,
            "threshold": self.threshold,
            "verbose": self.verbose,
        }

    @staticmethod
    def _keep_in_order(dataset: pd.DataFrame, to_drop) -> pd.DataFrame:
        # L2C-PORT: upstream computes `set(columns) - set(to_drop)` and indexes
        # with the resulting set, which scrambles the column order from run to
        # run. Keeping the original order makes the output reproducible.
        to_drop = set(to_drop)
        return dataset[[c for c in dataset.columns if c not in to_drop]]

    def FS_MR_missing_ratio(self, dataset, missing_threshold=0.2):
        _p("Apply MR feature selection with missing threshold=", missing_threshold)

        missing_series = dataset.isnull().sum() / dataset.shape[0]
        record_missing = (
            pd.DataFrame(missing_series[missing_series > missing_threshold])
            .reset_index()
            .rename(columns={"index": "feature", 0: "missing_fraction"})
        )
        to_drop = list(record_missing["feature"])

        _p("%d features with greater than %0.2f missing values." % (len(to_drop), missing_threshold))
        _p("List of variables to be removed :", to_drop)
        return self._keep_in_order(dataset, to_drop)

    def FS_LC_identify_collinear(self, dataset, correlation_threshold=0.8):
        # For each pair of features whose correlation coefficient exceeds
        # `correlation_threshold`, one of the pair is identified for removal.
        _p("Apply LC feature selection with threshold=", correlation_threshold)

        if dataset.shape[1] < 2:
            return dataset

        corr_matrix = dataset.corr()
        # L2C-PORT: np.bool was removed in numpy 1.24
        upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))

        to_drop = [c for c in upper.columns if any(upper[c].abs() > correlation_threshold)]

        _p("%d features with linear correlation greater than %0.2f."
           % (len(to_drop), correlation_threshold))
        _p("List of correlated variables to be removed :", to_drop)
        return self._keep_in_order(dataset, to_drop)

    def FS_WR_identify_best_subset(self, df_train, df_target, k=10):
        # Wrapper subset evaluator (Kohavi & John, 1997); chi2 requires
        # non-negative features, so only those are eligible.
        from sklearn.feature_selection import SelectKBest, chi2

        _p("Apply WR feature selection")

        if df_train.isnull().sum().sum() > 0:
            df_train = df_train.dropna()
            _p("WR requires no missing values, so missing values have been "
               "removed applying DROP on the train dataset.")

        X = df_train.select_dtypes(["number"])
        Y = df_target.loc[X.index] if len(X) else df_target

        if len(df_train.columns) < 1 or len(df_train) < 1:
            _p("Error: Need at least one continous variable for identifying "
               "the best subset of features")
            return df_train

        # L2C-PORT: upstream deletes from `lis` while iterating over its own
        # indices, which shifts positions and drops arbitrary columns. The
        # intent — keep the non-negative variables chi2 can consume — is
        # implemented directly here.
        negative_counts = X.lt(0).sum()
        lis = [col for col in X.columns if negative_counts[col] == 0]

        if len(lis) == 0:
            _p("Input dataset has no positive variables. WR feature selection "
               "is not applicable. Dataset unchanged.")
            return df_train

        X = X[lis]
        _p("Input variables must be non-negative. WR feature selection is only "
           "applied to positive variables.")

        if X.shape[1] < 1 or len(X) < 2 or Y.nunique() < 2:
            return X

        selector = SelectKBest(score_func=chi2, k="all")
        selector.fit(X, Y)
        Best_Flist = X.columns[selector.get_support(indices=True)].tolist()
        if self.verbose:
            _p("Best features to keep", Best_Flist)
        return X[Best_Flist]

    def FS_SVC_based(self, df_train, df_target):
        from sklearn.feature_selection import SelectFromModel
        from sklearn.svm import LinearSVC

        _p("Apply SVC feature selection")
        if df_train.isnull().sum().sum() > 0:
            df_train = df_train.dropna()

        if len(df_train.columns) < 1 or len(df_train) < 1:
            _p("Error: Need at least one continous variable for feature selection")
            return df_train

        X = df_train.select_dtypes(["number"])
        Y = df_target.loc[X.index]
        lsvc = LinearSVC(C=0.01, penalty="l1", dual=False).fit(X, Y)
        model = SelectFromModel(lsvc, prefit=True)
        Best_Flist = X.columns[model.get_support(indices=True)].tolist()
        return X[Best_Flist] if Best_Flist else X

    def FS_Tree_based(self, df_train, df_target):
        # Feature extraction using an extremely-randomized tree ensemble
        from sklearn.ensemble import ExtraTreesClassifier

        _p("Apply Tree-based feature selection ")
        if df_train.isnull().sum().sum() > 0:
            df_train = df_train.dropna()
            _p("Tree requires no missing values, so missing values have been "
               "removed applying DROP on the train dataset.")

        if len(df_train.columns) < 1 or len(df_train) < 1:
            _p("Error: Need at least one continous variable for feature selection")
            return df_train

        X = df_train.select_dtypes(["number"])
        Y = df_target.loc[X.index]
        if X.shape[1] < 1 or len(X) < 2 or Y.nunique() < 2:
            return X

        clf = ExtraTreesClassifier(n_estimators=50, random_state=0).fit(X, Y)
        # L2C-PORT: SelectFromModel(prefit=True).get_support() is deprecated for
        # prefit estimators; the default "mean importance" rule is applied here.
        importances = clf.feature_importances_
        keep = importances >= importances.mean()
        Best_Flist = X.columns[keep].tolist()
        if self.verbose:
            _p("Best features to keep", Best_Flist)
        return X[Best_Flist] if Best_Flist else X

    def transform(self):
        df = self.dataset["train"].copy()
        fsd = self.dataset
        start_time = time.time()

        _p(">>Feature selection ")
        _p("Before feature selection:", self.dataset["train"].shape[1], "features ")

        if self.strategy == "MR":
            dn = self.FS_MR_missing_ratio(df, missing_threshold=self.threshold)

        elif self.strategy == "LC":
            d = df.select_dtypes(["number"])
            do = df.select_dtypes(exclude=["number"])
            dn = self.FS_LC_identify_collinear(d, correlation_threshold=self.threshold)
            dn = dn.join(do)

        elif self.strategy == "VAR":
            dn = df.select_dtypes(["number"])
            coef = dn.std()
            _p("Apply VAR feature selection with threshold=", self.threshold)
            abstract_threshold = np.percentile(coef, 100.0 * self.threshold)
            to_discard = coef[coef < abstract_threshold].index
            dn = dn.drop(columns=list(to_discard))

        elif not isinstance(self.dataset["target"], dict):
            dn = df.select_dtypes(["number"])
            if dn.isnull().sum().sum() > 0:
                dn = dn.dropna()
                _p("Warning: This strategy requires no missing values, so missing "
                   "values have been removed applying DROP on the dataset.")
            dt = self.dataset["target"].loc[dn.index]

            if self.strategy == "L1":
                from sklearn.linear_model import Lasso

                model = Lasso(alpha=100.0, tol=0.01, random_state=0).fit(dn, dt)
                coef = np.abs(model.coef_)
                abstract_threshold = np.percentile(coef, 100.0 * self.threshold)
                dn = dn.drop(columns=list(dn.columns[coef < abstract_threshold]))

            elif self.strategy == "IMP":
                from sklearn.ensemble import RandomForestRegressor

                model = RandomForestRegressor(n_estimators=50, n_jobs=1, random_state=0)
                model.fit(dn, dt)
                coef = model.feature_importances_
                abstract_threshold = np.percentile(coef, 100.0 * self.threshold)
                dn = dn.drop(columns=list(dn.columns[coef < abstract_threshold]))

            elif self.strategy == "Tree":
                dn = self.FS_Tree_based(dn, dt)

            elif self.strategy == "WR":
                dn = self.FS_WR_identify_best_subset(dn, dt)

            elif self.strategy == "SVC":
                dn = self.FS_SVC_based(dn, dt)

            else:
                _p("Strategy invalid -- No feature selection done on the train dataset")
                dn = self.dataset["train"].copy()
        else:
            _p("Strategy invalid -- No feature selection done on the train dataset")
            dn = self.dataset["train"].copy()

        to_keep = [column for column in dn.columns]

        if self.exclude is None:
            fsd["train"] = dn[to_keep]
        elif self.exclude not in self.dataset["train"].columns.values:
            _p("Exclude variable invalid. Please choose a variable from the "
               "input training dataset.")
            fsd["train"] = dn[to_keep]
        elif self.exclude in dn.columns.values:
            fsd["train"] = dn[to_keep]
        else:
            _p("and keep variable", self.exclude)
            to_keep.append(self.exclude)
            dn = self.dataset["train"]
            fsd["train"] = dn[to_keep]

        if not isinstance(self.dataset["test"], dict):
            df_test = pd.DataFrame(self.dataset["test"])
            kept_test = [c for c in to_keep if c in df_test.columns]
            fsd["test"] = df_test[kept_test]

        _p("After feature selection:", len(to_keep), "features remain", to_keep)
        _p("Feature selection done -- CPU time: %s seconds" % (time.time() - start_time))
        return fsd


# ===========================================================================
# learn2clean.outlier_detection.outlier_detector
# ===========================================================================

class Outlier_detector:
    """
    Detect and remove the outlying rows of the dataset.

    * strategy : 'ZSB' robust z-score (MAD), 'IQR' inter-quartile range,
      'LOF' local outlier factor
    * threshold: float, default = 0.3. A row is an outlier when more than
      `threshold` of its variables are outlying; -1 means "any outlying value".
    """

    def __init__(self, dataset, strategy="ZSB", threshold=0.3, verbose=False, exclude=None):
        self.dataset = dataset
        self.strategy = strategy
        self.threshold = threshold
        self.verbose = verbose
        self.exclude = exclude

    def get_params(self, deep=True):
        return {
            "strategy": self.strategy,
            "threshold": self.threshold,
            "verbose": self.verbose,
            "exclude": self.exclude,
        }

    @staticmethod
    def _keep_rows(X: pd.DataFrame, to_drop) -> pd.DataFrame:
        # L2C-PORT: upstream uses `X.loc[list(set(X.index) - set(to_drop))]`,
        # whose row order depends on set hashing. Preserving the original order
        # keeps the output rows aligned with the poisoning mask.
        to_drop = set(to_drop)
        return X.loc[[i for i in X.index if i not in to_drop]]

    def IQR_outlier_detection(self, dataset, threshold):
        X = dataset.select_dtypes(["number"])
        Y = dataset.select_dtypes(["object"])

        if len(X.columns) < 1:
            _p("Error: Need at least one numeric variable for IQR outlier "
               "detection\n Dataset inchanged")
            return dataset

        Q1 = X.quantile(0.25)
        Q3 = X.quantile(0.75)
        IQR = Q3 - Q1

        outliers = X[((X < (Q1 - 1.5 * IQR)) | (X > (Q3 + 1.5 * IQR)))]
        to_drop = X[outliers.sum(axis=1) / outliers.shape[1] > threshold].index

        if threshold == -1:
            X = X[~((X < (Q1 - 1.5 * IQR)) | (X > (Q3 + 1.5 * IQR))).any(axis=1)]
        else:
            X = self._keep_rows(X, to_drop)

        df = X.join(Y)
        _p(len(to_drop), "outlying rows have been removed")
        if len(to_drop) > 0 and self.verbose:
            _p("with indexes:", list(to_drop))
        return df

    def ZSB_outlier_detection(self, dataset, threshold):
        # Robust z-score defined from the median and the median absolute
        # deviation (MAD):  z-score = |x - median(x)| / mad(x)
        X = dataset.select_dtypes(["number"])
        Y = dataset.select_dtypes(["object"])

        if len(X.columns) < 1:
            _p("Error: Need at least one numeric variable for ZSB outlier "
               "detection\n Dataset inchanged")
            return dataset

        median = X.median(axis=0)
        median_absolute_deviation = 1.4296 * np.abs(X - median).median(axis=0)
        # L2C-PORT: guard the 0-MAD division upstream leaves to numpy
        median_absolute_deviation = median_absolute_deviation.replace(0, np.nan)
        modified_z_scores = (X - median) / median_absolute_deviation

        outliers = X[np.abs(modified_z_scores) > 1.6]
        to_drop = outliers[(outliers.count(axis=1) / outliers.shape[1]) > threshold].index

        if threshold == -1:
            X = X[~(np.abs(modified_z_scores) > 1.6).any(axis=1)]
        else:
            # e.g., remove rows where 40% of variables have a z-score above
            # a threshold = 0.4
            X = self._keep_rows(X, to_drop)

        df = X.join(Y)
        _p(len(to_drop), "outlying rows have been removed")
        if len(to_drop) > 0 and self.verbose:
            _p("with indexes:", list(to_drop))
        return df

    def LOF_outlier_detection(self, dataset, threshold):
        # requires no missing value; selects the top-k outliers
        from sklearn.neighbors import LocalOutlierFactor

        # L2C-PORT (1/2): upstream replaces the frame with `dataset.dropna()`
        # because the estimator cannot consume NaN, and never restores the rows
        # it discarded — so on poisoned data the frame collapses to its complete
        # cases (cylinder_bands NAR: 27 rows left out of 378, and that strategy
        # then wins because the metric does not penalise losing rows). The
        # estimator is still fitted on the complete cases, but only the rows LOF
        # actually flags are removed, exactly as IQR and ZSB do.
        complete = dataset.dropna()
        if len(complete) < len(dataset):
            _p("LOF requires no missing values, so it is fitted on the",
               len(complete), "complete rows")

        X = complete.select_dtypes(["number"])
        k = int(threshold * 100)

        if len(X.columns) < 1 or len(X) < 1 or k < 1 or len(X) <= k:
            _p("Error: Need at least one continous variable for LOF outlier "
               "detection\n Dataset inchanged")
            return dataset

        clf = LocalOutlierFactor(n_neighbors=4, contamination=0.1)
        clf.fit_predict(X)
        # sklearn: the higher negative_outlier_factor_, the more normal the row
        # (inliers sit close to -1, outliers are more negative).
        LOF_scores = clf.negative_outlier_factor_

        # L2C-PORT (2/2): upstream keeps `X[LOF_scores < top_k_values[0]]`, where
        # top_k_values comes from `np.argsort(LOF_scores)[-k:]` — the k *highest*
        # scores, i.e. the k most normal rows. It therefore drops the k most
        # normal rows and keeps the outliers. The comparison is inverted here so
        # the k most abnormal rows are the ones removed.
        to_drop = X.index[np.argsort(LOF_scores)[:k]]

        df = self._keep_rows(dataset, to_drop)
        _p(len(to_drop), "outlying rows have been removed")
        if len(to_drop) > 0 and self.verbose:
            _p("with indexes:", list(to_drop))
        return df

    def transform(self):
        start_time = time.time()
        osd = self.dataset
        _p(">>Outlier detection and removal:")

        for key in ["train", "test"]:
            if not isinstance(self.dataset[key], dict) and not self.dataset[key].empty:
                _p("* For", key, "dataset")
                d = self.dataset[key]

                if self.strategy == "ZSB":
                    dn = self.ZSB_outlier_detection(d, self.threshold)
                elif self.strategy == "IQR":
                    dn = self.IQR_outlier_detection(d, self.threshold)
                elif self.strategy == "LOF":
                    dn = self.LOF_outlier_detection(d, self.threshold)
                else:
                    raise ValueError(
                        "Strategy invalid. Please choose between 'ZSB', 'IQR' or 'LOF'"
                    )
                osd[key] = dn
            else:
                _p("No outlier detection for", key, "dataset")

        _p("Outlier detection and removal done -- CPU time: %s seconds"
           % (time.time() - start_time))
        return osd


# ===========================================================================
# learn2clean.consistency_checking.consistency_checker
# ===========================================================================
# L2C-PORT: upstream delegates constraint discovery/verification to `tdda`
# (discover_df / detect_df / verify_df) and pattern induction to `tdda.rexpy`.
# Both are reimplemented below on top of pandas/re so the baseline carries no
# extra dependency. The constraint and pattern files keep upstream's role: they
# are discovered once per dataset and cached under `save_dir`, and a
# hand-written file dropped in that directory is picked up instead.
#
# Caveat, stated explicitly because it changes what CC/PC can do here: upstream's
# examples verify against *reference* constraints authored from clean data. The
# quAIL pipeline has no clean reference available to a cleaning baseline (using
# data_poisoned/test/ would leak the clean hold-out), so constraints and patterns
# are induced from the poisoned frame itself. Self-discovered constraints pass by
# construction, so CC only removes rows when an external file is supplied. PC
# still bites: pattern induction keeps the `max_patterns` most frequent value
# shapes per column, so rare shapes — typically the injected typos and
# out-of-vocabulary categories — fall outside the induced patterns.

_CHAR_CLASSES = (
    ("[A-Z]", lambda c: c.isupper() and c.isalpha()),
    ("[a-z]", lambda c: c.islower() and c.isalpha()),
    ("[0-9]", lambda c: c.isdigit()),
)


def _char_class(c: str) -> str:
    for name, test in _CHAR_CLASSES:
        if test(c):
            return name
    return c


def _runs(value: str) -> List[Tuple[str, int]]:
    """Collapse a string into consecutive runs of the same character class."""
    runs: List[Tuple[str, int]] = []
    for ch in value:
        cls = _char_class(ch)
        if runs and runs[-1][0] == cls:
            runs[-1] = (cls, runs[-1][1] + 1)
        else:
            runs.append((cls, 1))
    return runs


def rexpy_extract(corpus: List[str], max_patterns: int = 4) -> List[str]:
    """
    Induce regular expressions covering a corpus of strings.

    Reimplementation of ``tdda.rexpy.extract``: values are reduced to runs of
    character classes, grouped by shape, and each group becomes one anchored
    regex with ``{min,max}`` repetition bounds. Only the `max_patterns` most
    frequent shapes are emitted, so rare shapes stay uncovered.
    """
    shapes: Dict[Tuple[str, ...], List[List[Tuple[str, int]]]] = {}
    for value in corpus:
        if value is None or value == "" or value == "nan":
            continue
        rs = _runs(str(value))
        key = tuple(cls for cls, _ in rs)
        shapes.setdefault(key, []).append(rs)

    ranked = sorted(shapes.items(), key=lambda kv: -len(kv[1]))[:max_patterns]

    patterns = []
    for key, members in ranked:
        parts = []
        for pos, cls in enumerate(key):
            lengths = [m[pos][1] for m in members]
            lo, hi = min(lengths), max(lengths)
            atom = cls if cls.startswith("[") else re.escape(cls)
            parts.append(atom if lo == hi == 1 else "%s{%d,%d}" % (atom, lo, hi))
        patterns.append("^" + "".join(parts) + "$")
    return patterns


def constraint_discovery(dataset: pd.DataFrame, file_name: str, save_dir: str = "save") -> Dict:
    """
    Discover per-field constraints and write them next to the dataset.

    Reimplementation of ``tdda.constraints.discover_df``: type, min/max,
    sign, min_length/max_length, max_nulls, allowed_values and no_duplicates.
    """
    import json

    fields: Dict[str, Dict] = {}
    for col in dataset.columns:
        s = dataset[col]
        observed = s.dropna()
        spec: Dict[str, Any] = {}

        if pd.api.types.is_numeric_dtype(s):
            is_int = bool(observed.size) and bool((observed == observed.round()).all())
            spec["type"] = "int" if is_int else "real"
            if observed.size:
                spec["min"] = float(observed.min())
                spec["max"] = float(observed.max())
                if spec["min"] > 0:
                    spec["sign"] = "positive"
                elif spec["min"] >= 0:
                    spec["sign"] = "non-negative"
                elif spec["max"] < 0:
                    spec["sign"] = "negative"
                elif spec["max"] <= 0:
                    spec["sign"] = "non-positive"
        else:
            spec["type"] = "string"
            as_str = observed.astype(str)
            if as_str.size:
                spec["min_length"] = int(as_str.str.len().min())
                spec["max_length"] = int(as_str.str.len().max())
                uniques = as_str.unique().tolist()
                if len(uniques) <= 20:
                    spec["allowed_values"] = sorted(uniques)

        if s.isnull().sum() == 0:
            spec["max_nulls"] = 0
        if observed.size and observed.is_unique:
            spec["no_duplicates"] = True

        fields[col] = spec

    constraints = {
        "creation_metadata": {"n_records": int(len(dataset)), "creator": "learn2clean port"},
        "fields": fields,
    }

    os.makedirs(save_dir, exist_ok=True)
    with open(os.path.join(save_dir, file_name + "_constraints.json"), "w") as f:
        json.dump(constraints, f, indent=4)
    return constraints


def pattern_discovery(
    dataset: pd.DataFrame, file_name: str, save_dir: str = "save", max_patterns: int = 4
) -> pd.DataFrame:
    """
    Discover per-column patterns and write them next to the dataset.

    Reimplementation of upstream's ``pattern_discovery``: one ``col;num;pattern``
    row per induced regex, written to ``<file_name>_patterns.txt``.
    """
    listp = []
    for c in dataset.columns.values:
        corpus = dataset[c].dropna().unique().astype("str").tolist()
        if not corpus:
            continue
        for i, pattern in enumerate(rexpy_extract(corpus, max_patterns=max_patterns)):
            listp.append((c, i, "'" + str(pattern) + "'"))

    p = pd.DataFrame(listp, columns=["col", "num", "pattern"])
    os.makedirs(save_dir, exist_ok=True)
    p.to_csv(
        os.path.join(save_dir, file_name + "_patterns.txt"),
        header=("col", "num", "pattern"),
        index=False,
        sep=";",
    )
    return p


class Consistency_checker:
    """
    Identify and remove rows violating the constraints or the patterns
    specified in `<file_name>_constraints.json` / `<file_name>_patterns.txt`
    for the strategy 'CC' or 'PC' respectively.
    """

    def __init__(self, dataset, file_name, strategy="CC", verbose=False,
                 save_dir="save", max_patterns=4):
        self.dataset = dataset
        self.file_name = file_name
        self.strategy = strategy
        self.verbose = verbose
        self.save_dir = save_dir
        self.max_patterns = max_patterns

    def get_params(self, deep=True):
        return {
            "strategy": self.strategy,
            "file_name": self.file_name,
            "verbose": self.verbose,
            "save_dir": self.save_dir,
        }

    def _constraints(self, dataset):
        import json

        path = os.path.join(self.save_dir, self.file_name + "_constraints.json")
        if os.path.exists(path):
            with open(path) as f:
                return json.load(f)
        return constraint_discovery(dataset, self.file_name, self.save_dir)

    def _patterns(self, dataset):
        path = os.path.join(self.save_dir, self.file_name + "_patterns.txt")
        if os.path.exists(path):
            return pd.read_csv(path, sep=";")
        return pattern_discovery(
            dataset.select_dtypes(["object"]), self.file_name, self.save_dir, self.max_patterns
        )

    def CC_constraint_checking(self, dataset, file_name, verbose):
        constraints = self._constraints(dataset)
        _p("Constraints from the file:", file_name + "_constraints.json")

        failing = pd.Series(False, index=dataset.index)
        passes = failures = 0

        for col, spec in (constraints.get("fields") or {}).items():
            if col not in dataset.columns:
                continue
            s = dataset[col]
            for rule, value in spec.items():
                violated = pd.Series(False, index=dataset.index)
                if rule == "min":
                    violated = pd.to_numeric(s, errors="coerce") < value
                elif rule == "max":
                    violated = pd.to_numeric(s, errors="coerce") > value
                elif rule == "sign":
                    num = pd.to_numeric(s, errors="coerce")
                    violated = {
                        "positive": num <= 0,
                        "non-negative": num < 0,
                        "negative": num >= 0,
                        "non-positive": num > 0,
                    }.get(value, violated)
                elif rule == "min_length":
                    violated = s.astype(str).str.len() < value
                elif rule == "max_length":
                    violated = s.astype(str).str.len() > value
                elif rule == "allowed_values":
                    violated = s.notna() & ~s.astype(str).isin(set(value))
                elif rule == "max_nulls" and value == 0:
                    violated = s.isna()
                else:
                    continue

                violated = violated.fillna(False)
                if violated.any():
                    failures += 1
                    failing |= violated
                else:
                    passes += 1

        _p("Constraints passing: %d" % passes)
        _p("Constraints failing: %d" % failures)

        detection_index = dataset.index[failing]
        if len(detection_index) and verbose:
            _p("Row index with constraint failure:", list(detection_index))

        # return the dataset with inconsistent tuples removed
        to_keep = [i for i in dataset.index if i not in set(detection_index)]
        _p(len(detection_index), "inconsistent rows have been removed")
        return dataset.loc[to_keep] if to_keep else dataset

    def PC_pattern_checking(self, dataset, file_name, verbose):
        df = dataset.copy()
        p = self._patterns(dataset)
        obj = dataset.select_dtypes(["object"])

        if verbose:
            _p("Patterns:", p)

        # Several patterns may exist for one variable: a row is a violation only
        # when the value matches none of the patterns declared for its column.
        to_drop: set = set()
        for c, group in p.groupby("col"):
            if c not in obj.columns:
                continue
            regexes = []
            for pt in group["pattern"]:
                try:
                    regexes.append(re.compile(eval(pt) if isinstance(pt, str) else pt))
                except Exception:
                    continue
            if not regexes:
                continue

            values = obj[c].astype(str)
            matched = pd.Series(False, index=obj.index)
            for rx in regexes:
                matched |= values.str.contains(rx, regex=True, na=False)

            check = obj.index[~matched & obj[c].notna()]
            if len(check):
                _p("Number of pattern violations on variable '", c, "':", len(check))
                to_drop |= set(check)

        if not to_drop:
            return df

        to_keep = [i for i in df.index if i not in to_drop]
        _p(len(to_drop), "rows violating the discovered patterns have been removed")
        if not to_keep:
            _p("No record from the dataset satisfied the patterns!")
            return df
        return df.loc[to_keep]

    def transform(self):
        ccd = self.dataset
        start_time = time.time()
        _p(">>Consistency checking")

        for key in ["train", "test"]:
            _p("* For", key, "dataset")
            if not isinstance(self.dataset[key], dict) and not self.dataset[key].empty:
                dn = self.dataset[key]
                if self.strategy == "CC":
                    dn = self.CC_constraint_checking(dn, self.file_name, self.verbose)
                elif self.strategy == "PC":
                    dn = self.PC_pattern_checking(dn, self.file_name, self.verbose)
                else:
                    raise ValueError("Strategy invalid. Please choose between 'CC' or 'PC'")
                ccd[key] = dn

        _p("Consistency checking done -- CPU time: %s seconds" % (time.time() - start_time))
        return ccd


# ===========================================================================
# learn2clean.duplicate_detection.duplicate_detector
# ===========================================================================
# L2C-PORT: `jellyfish`, `py_stringmatching` and `py_stringsimjoin` are replaced
# by the equivalent algorithms implemented below. `py_stringsimjoin` no longer
# builds on this stack at all.

def add_key_reindex(dataset, rand=False):
    dataset = dataset.copy()
    if rand:
        dataset = dataset.reindex(np.random.permutation(dataset.index))
    dataset["New_ID"] = range(1, 1 + len(dataset))
    return dataset


def _jaro_winkler(s1: str, s2: str, p: float = 0.1) -> float:
    """Jaro-Winkler similarity (the measure jellyfish.jaro_winkler provides)."""
    if s1 == s2:
        return 1.0
    len1, len2 = len(s1), len(s2)
    if len1 == 0 or len2 == 0:
        return 0.0

    window = max(len1, len2) // 2 - 1
    window = max(window, 0)
    s1_flags = [False] * len1
    s2_flags = [False] * len2

    matches = 0
    for i in range(len1):
        lo = max(0, i - window)
        hi = min(i + window + 1, len2)
        for j in range(lo, hi):
            if not s2_flags[j] and s1[i] == s2[j]:
                s1_flags[i] = s2_flags[j] = True
                matches += 1
                break
    if matches == 0:
        return 0.0

    k = transpositions = 0
    for i in range(len1):
        if s1_flags[i]:
            for j in range(k, len2):
                if s2_flags[j]:
                    k = j + 1
                    break
            if s1[i] != s2[j]:
                transpositions += 1
    transpositions //= 2

    jaro = (matches / len1 + matches / len2 + (matches - transpositions) / matches) / 3.0

    prefix = 0
    for a, b in zip(s1[:4], s2[:4]):
        if a != b:
            break
        prefix += 1
    return jaro + prefix * p * (1 - jaro)


def _levenshtein(s1: str, s2: str) -> int:
    """Levenshtein edit distance (jellyfish.levenshtein_distance)."""
    if s1 == s2:
        return 0
    if not s1:
        return len(s2)
    if not s2:
        return len(s1)
    previous = list(range(len(s2) + 1))
    for i, c1 in enumerate(s1):
        current = [i + 1]
        for j, c2 in enumerate(s2):
            current.append(min(previous[j + 1] + 1, current[j] + 1, previous[j] + (c1 != c2)))
        previous = current
    return previous[-1]


def _damerau_levenshtein(s1: str, s2: str) -> int:
    """Damerau-Levenshtein distance (jellyfish.damerau_levenshtein_distance)."""
    if s1 == s2:
        return 0
    len1, len2 = len(s1), len(s2)
    d = [[0] * (len2 + 1) for _ in range(len1 + 1)]
    for i in range(len1 + 1):
        d[i][0] = i
    for j in range(len2 + 1):
        d[0][j] = j
    for i in range(1, len1 + 1):
        for j in range(1, len2 + 1):
            cost = 0 if s1[i - 1] == s2[j - 1] else 1
            d[i][j] = min(d[i - 1][j] + 1, d[i][j - 1] + 1, d[i - 1][j - 1] + cost)
            if i > 1 and j > 1 and s1[i - 1] == s2[j - 2] and s1[i - 2] == s2[j - 1]:
                d[i][j] = min(d[i][j], d[i - 2][j - 2] + cost)
    return d[len1][len2]


def _jaccard_self_join(tokens: Dict[Any, frozenset], threshold: float) -> List[Tuple[Any, Any]]:
    """
    Self-join returning every ordered pair whose Jaccard similarity is at least
    `threshold`, the way ``py_stringsimjoin.jaccard_join`` does.

    Candidates are generated from an inverted index and pruned with the standard
    size filter |A| * t <= |B| <= |A| / t.
    """
    inverted: Dict[Any, List[Any]] = {}
    for key, toks in tokens.items():
        for tok in toks:
            inverted.setdefault(tok, []).append(key)

    pairs: List[Tuple[Any, Any]] = []
    for key, toks in tokens.items():
        size = len(toks)
        if size == 0:
            continue
        overlap: Dict[Any, int] = {}
        for tok in toks:
            for other in inverted.get(tok, ()):
                if other == key:
                    continue
                overlap[other] = overlap.get(other, 0) + 1
        for other, ov in overlap.items():
            other_size = len(tokens[other])
            if other_size < size * threshold or other_size > size / threshold:
                continue
            if ov / (size + other_size - ov) >= threshold:
                pairs.append((key, other))
    return pairs


class Duplicate_detector:
    """
    Remove the duplicate records from the dataset.

    * strategy: 'ED' exact duplicate removal, 'AD' approximate duplicate removal
      based on Jaccard similarity, 'METRIC' using the distance named in `metric`
    * metric  : 'DL' Damerau-Levenshtein, 'LM' Levenshtein, 'JW' Jaro-Winkler
    * threshold: float, default = 0.6
    """

    def __init__(self, dataset, strategy="ED", threshold=0.6, metric="DL",
                 verbose=False, exclude=None, max_rows_metric=5000):
        self.dataset = dataset
        self.strategy = strategy
        self.threshold = threshold
        self.metric = metric
        self.verbose = verbose
        self.exclude = exclude
        self.max_rows_metric = max_rows_metric

    def get_params(self, deep=True):
        return {
            "strategy": self.strategy,
            "threshold": self.threshold,
            "metric": self.metric,
            "verbose": self.verbose,
            "exclude": self.exclude,
        }

    @staticmethod
    def _row_strings(dataset: pd.DataFrame) -> pd.Series:
        # concatenate all columns into one string per row, '*'-separated
        data = dataset.astype(str).apply(lambda x: "*".join(x.values.tolist()), axis=1)
        return data.astype(str).str.replace(" ", "", regex=False).str.lower()

    def ED_Exact_duplicate_removal(self, dataset):
        if dataset.empty:
            _p("No duplicate detection, empty dataframe")
            return dataset
        df = dataset.drop_duplicates()
        _p("Initial number of rows:", len(dataset))
        _p("After deduplication: Number of rows:", len(df))
        return df

    def jaccard_similarity(self, dataset, threshold):
        # 'AD' strategy. Upstream tokenizes the '*'-joined row with
        # py_stringmatching's WhitespaceTokenizer *after* stripping every space,
        # so each row collapses to a single token and the Jaccard join matches
        # exactly-equal rows. That behaviour is reproduced as-is; the generic
        # token join below degrades to it for single-token sets.
        if dataset.empty:
            return dataset

        df = add_key_reindex(dataset)
        rows = self._row_strings(dataset)
        tokens = {nid: frozenset(text.split()) for nid, text in zip(df["New_ID"], rows)}

        pairs = _jaccard_self_join(tokens, threshold)
        # L2C-PORT-FAITHFUL: the self-join yields both (a, b) and (b, a), and
        # upstream drops every row appearing on the right-hand side, so *all*
        # members of a duplicate group are removed — the original included.
        dup_ids = {r for _, r in pairs}

        out = df[~df["New_ID"].isin(dup_ids)].drop(columns=["New_ID"])
        _p("Number of duplicate rows removed:", len(dup_ids))
        return out

    def AD_Approx_string_duplicate_removal(self, dataset, threshold, metric="DL"):
        # 'METRIC' strategy — pairwise string distance, O(n^2), so it is capped.
        if dataset.empty:
            return dataset
        if len(dataset) > self.max_rows_metric:
            _p("METRIC deduplication skipped:", len(dataset), "rows exceeds the",
               self.max_rows_metric, "cap for the pairwise comparison")
            return dataset

        df = add_key_reindex(dataset, rand=True)
        data = self._row_strings(dataset)
        ids = list(df["New_ID"])
        texts = list(data)

        drop_ids = set()
        for i in range(len(texts)):
            for j in range(len(texts)):
                if i == j or texts[i] == texts[j]:
                    continue
                bound = (len(texts[i]) + len(texts[j]) / 2) * threshold
                if metric == "DL":
                    hit = _damerau_levenshtein(texts[i], texts[j]) < bound
                elif metric == "LM":
                    hit = _levenshtein(texts[i], texts[j]) < bound
                elif metric == "JW":
                    hit = _jaro_winkler(texts[i], texts[j]) > bound
                else:
                    raise ValueError("Metric invalid. Please choose between 'LM', 'JW' or 'DL'.")
                if hit:
                    drop_ids.add(ids[j])

        out = df[~df["New_ID"].isin(drop_ids)].drop(columns=["New_ID"]).sort_index()
        _p("Number of duplicate rows removed:", len(dataset) - len(out))
        return out

    def transform(self):
        dedup = self.dataset
        start_time = time.time()
        _p(">>Duplicate detection and removal:")

        for key in ["train", "test"]:
            if not isinstance(self.dataset[key], dict) and not self.dataset[key].empty:
                _p("* For", key, "dataset")
                if self.strategy == "ED":
                    dn = self.ED_Exact_duplicate_removal(self.dataset[key])
                elif self.strategy == "AD":
                    dn = self.jaccard_similarity(self.dataset[key], self.threshold)
                elif self.strategy == "METRIC":
                    dn = self.AD_Approx_string_duplicate_removal(
                        self.dataset[key], metric=self.metric, threshold=self.threshold
                    )
                else:
                    raise ValueError("Strategy invalid. Please choose between 'ED', 'METRIC' or 'AD'")
                dedup[key] = dn
            else:
                _p("No", key, "dataset, no duplicate detection")

        _p("Deduplication done -- CPU time: %s seconds" % (time.time() - start_time))
        return dedup


# ===========================================================================
# Goal states — learn2clean.classification / regression / clustering
# ===========================================================================
# L2C-PORT: every goal model is fitted on the numeric feature matrix produced by
# `_model_matrix`. When `dataset['test']` is empty — which is how this script
# drives Learn2Clean, since the quAIL hold-out lives outside the frame being
# cleaned — upstream's own "no target in the test set" branch applies and the
# quality metric is the k-fold cross-validated score on the cleaned data.

CART_NUM_TRIALS = 10


class Classifier:
    """
    Classification task. * strategy: 'LDA', 'CART', 'NB' or 'MNB'.
    Quality metric: accuracy.
    """

    def __init__(self, dataset, target, strategy="NB", k_folds=10, verbose=False):
        self.dataset = dataset
        self.target = target
        self.strategy = strategy
        self.k_folds = k_folds
        self.verbose = verbose

    def get_params(self, deep=True):
        return {
            "strategy": self.strategy,
            "target": self.target,
            "k_folds": self.k_folds,
            "verbose": self.verbose,
        }

    def _prepare(self, dataset):
        """Common upstream preamble: build X_train/y_train and the fold count."""
        X_train = _model_matrix(dataset["train"], self.target)
        k = self.k_folds

        if (len(X_train.columns) <= 1) or (len(X_train) < k):
            _p("Error: Need at least one continous variable and", k,
               "observations for classification")
            return None, None, None, None, None

        y_train = dataset["target"].loc[X_train.index]
        if y_train.nunique() < 2:
            _p("Error: the target has a single class after preparation")
            return None, None, None, None, None
        if int(y_train.value_counts().min()) < 2:
            # a class left with a single member cannot be cross-validated;
            # report "not scorable" rather than letting StratifiedKFold raise
            _p("Error: a target class has a single member after preparation")
            return None, None, None, None, None

        k = _safe_k(k, y_train, len(X_train))

        X_test = y_test = None
        if _has_test(dataset):
            X_test = _model_matrix(dataset["test"], self.target)
            if isinstance(dataset.get("target_test"), dict):
                y_test = dataset["target"].loc[X_test.index]
            else:
                y_test = dataset["target_test"].loc[X_test.index]

        return X_train, y_train, X_test, y_test, k

    @staticmethod
    def _score(gs, X_test, y_test):
        results = gs.cv_results_
        best_index = int(np.nonzero(results["rank_test_score"] == 1)[0][0])
        accuracy = float(results["mean_test_score"][best_index])
        if X_test is not None and len(X_test) > 0:
            accuracy = float(gs.best_estimator_.score(X_test, y_test))
        return accuracy

    def LDA_classification(self, dataset, target):
        from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
        from sklearn.model_selection import GridSearchCV, StratifiedKFold

        X_train, y_train, X_test, y_test, k = self._prepare(dataset)
        if X_train is None:
            return None

        skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=1)
        # L2C-PORT: sklearn >= 0.24 rejects n_components > min(n_classes-1, n_features)
        n_components = int(min(2, y_train.nunique() - 1, X_train.shape[1]))
        model = LinearDiscriminantAnalysis(n_components=max(n_components, 1))
        gs = GridSearchCV(model, cv=skf, param_grid={}, scoring="accuracy")
        gs.fit(X_train, y_train)

        accuracy = self._score(gs, X_test, y_test)
        _p("Accuracy of LDA result for", k, "cross-validation :", accuracy)
        return accuracy

    def CART_classification(self, dataset, target):
        from sklearn.model_selection import GridSearchCV, KFold, cross_val_score
        from sklearn.tree import DecisionTreeClassifier

        X_train, y_train, X_test, y_test, k = self._prepare(dataset)
        if X_train is None:
            return None

        params = {"max_depth": [3, 5, 7, 9, 10]}
        non_nested_scores = np.zeros(CART_NUM_TRIALS)
        nested_scores = np.zeros(CART_NUM_TRIALS)
        gs = None

        for i in range(1, CART_NUM_TRIALS):
            inner_cv = KFold(n_splits=k, shuffle=True, random_state=i)
            outer_cv = KFold(n_splits=k, shuffle=True, random_state=i)
            model = DecisionTreeClassifier(random_state=i)
            gs = GridSearchCV(model, cv=inner_cv, param_grid=params, scoring="accuracy")
            gs.fit(X_train, y_train)
            non_nested_scores[i] = gs.best_score_
            nested_scores[i] = cross_val_score(gs, X=X_train, y=y_train, cv=outer_cv).mean()

        accuracy = self._score(gs, X_test, y_test)
        _p("Avg accuracy of CART classification for", k, "cross-validation :", accuracy)
        return accuracy

    def NB_classification(self, dataset, target):
        from sklearn.model_selection import GridSearchCV, StratifiedKFold
        from sklearn.naive_bayes import GaussianNB

        X_train, y_train, X_test, y_test, k = self._prepare(dataset)
        if X_train is None:
            return None

        skf = StratifiedKFold(n_splits=k)
        gs = GridSearchCV(GaussianNB(), cv=skf, param_grid={}, scoring="accuracy")
        gs.fit(X_train, y_train)

        accuracy = self._score(gs, X_test, y_test)
        _p("Accuracy of Naive Bayes classification for", k, "cross-validation :", accuracy)
        return accuracy

    def MNB_classification(self, dataset, target):
        from sklearn.model_selection import GridSearchCV, StratifiedKFold
        from sklearn.naive_bayes import MultinomialNB

        X_train, y_train, X_test, y_test, k = self._prepare(dataset)
        if X_train is None:
            return None
        if (X_train < 0).any().any():
            _p("MultinomialNB requires non-negative features -- skipped")
            return None

        skf = StratifiedKFold(n_splits=k)
        gs = GridSearchCV(MultinomialNB(), cv=skf, param_grid={}, scoring="accuracy")
        gs.fit(X_train, y_train)

        accuracy = self._score(gs, X_test, y_test)
        _p("Accuracy of Multinomial NB classification for", k, "cross-validation :", accuracy)
        return accuracy

    def transform(self):
        start_time = time.time()
        d = self.dataset

        if self.target != d["target"].name:
            raise ValueError("Target variable invalid.")

        _p(">>Classification task")
        if self.strategy == "LDA":
            dn = self.LDA_classification(dataset=d, target=self.target)
        elif self.strategy == "CART":
            dn = self.CART_classification(dataset=d, target=self.target)
        elif self.strategy == "NB":
            dn = self.NB_classification(dataset=d, target=self.target)
        elif self.strategy == "MNB":
            dn = self.MNB_classification(dataset=d, target=self.target)
        else:
            raise ValueError("The classification function should be LDA, CART, NB or MNB.")

        _p("Classification done -- CPU time: %s seconds" % (time.time() - start_time))
        return {"quality_metric": dn}


class Regressor:
    """
    Regression task. * strategy: 'LASSO', 'OLS' or 'MARS'.
    Quality metric: MSE.
    """

    def __init__(self, dataset, target, strategy="LASSO", k_folds=10, verbose=False):
        self.dataset = dataset
        self.target = target
        self.strategy = strategy
        self.k_folds = k_folds
        self.verbose = verbose

    def get_params(self, deep=True):
        return {"strategy": self.strategy, "target": self.target,
                "k_folds": self.k_folds, "verbose": self.verbose}

    def LASSO_regression(self, dataset, target):
        from sklearn.linear_model import LassoCV

        k = self.k_folds
        X_train = _model_matrix(dataset["train"], target)
        if (len(X_train.columns) <= 1) or (len(X_train) < k):
            _p("Error: Need at least one continous variable and", k, "observations for regression")
            return None

        y_train = dataset["target"].loc[X_train.index]
        my_alphas = np.array([0.001, 0.01, 0.02, 0.025, 0.05, 0.1, 0.25, 0.5, 0.8, 1.0, 1.2])
        # L2C-PORT: LassoCV.normalize was removed in sklearn 1.2 (it defaulted to False)
        lcv = LassoCV(alphas=my_alphas, fit_intercept=False, random_state=0, cv=k, tol=0.0001)
        lcv.fit(X_train, y_train)

        avg_mse = np.mean(lcv.mse_path_, axis=1)
        _p("Best alpha = ", lcv.alpha_)

        if not _has_test(dataset):
            mse = float(min(avg_mse))
            _p("MSE of LASSO with", k, "folds for cross-validation:", mse)
            return mse

        from sklearn.metrics import mean_squared_error

        X_test = _model_matrix(dataset["test"], target)
        y_test = dataset["target_test"].loc[X_test.index]
        return float(mean_squared_error(y_test, lcv.predict(X_test)))

    def OLS_regression(self, dataset, target):
        import statsmodels.api as sm
        from sklearn.metrics import mean_squared_error

        k = self.k_folds
        X_train = _model_matrix(dataset["train"], target)
        if (len(X_train.columns) <= 1) or (len(X_train) < k):
            _p("Error: Need at least one continous variable and", k, "observations for regression")
            return None

        y_train = dataset["target"].loc[X_train.index]
        resReg = sm.OLS(y_train, sm.add_constant(X_train)).fit()
        if self.verbose:
            _p(resReg.summary())

        if not _has_test(dataset):
            return float(resReg.mse_total)

        X_test = _model_matrix(dataset["test"], target)
        y_test = dataset["target_test"].loc[X_test.index]
        ypReg = resReg.predict(sm.add_constant(X_test))
        return float(mean_squared_error(y_test, ypReg))

    def MARS_regression(self, dataset, target):
        # L2C-PORT: upstream uses pyearth.Earth (sklearn-contrib-py-earth), which
        # has no release compatible with this Python/numpy and does not build.
        raise NotImplementedError(
            "MARS requires sklearn-contrib-py-earth, which no longer builds on "
            "Python 3.12 / numpy 2. Use 'LASSO' or 'OLS' as the regression goal."
        )

    def transform(self):
        start_time = time.time()
        d = self.dataset
        _p(">>Regression task")

        if self.strategy == "LASSO":
            dn = self.LASSO_regression(d, self.target)
        elif self.strategy == "OLS":
            dn = self.OLS_regression(d, self.target)
        elif self.strategy == "MARS":
            dn = self.MARS_regression(d, self.target)
        else:
            raise ValueError("The regression function should be LASSO, OLS or MARS.")

        _p("Regression done -- CPU time: %s seconds" % (time.time() - start_time))
        return {"quality_metric": dn}


def compare_k_means(k_list, X):
    from sklearn import metrics
    from sklearn.cluster import KMeans

    X = _model_matrix(X, None)
    silhouette_list = []
    for p in k_list:
        clusterer = KMeans(n_clusters=p, n_init=10, random_state=0).fit(X)
        silhouette_list.append(round(metrics.silhouette_score(X, clusterer.labels_), 4))
    key = silhouette_list.index(max(silhouette_list))
    _p("Best silhouette =", max(silhouette_list), " for k=", k_list[key])
    return k_list[key]


def compare_k_AggClustering(k_list, X):
    from sklearn import metrics
    from sklearn.cluster import AgglomerativeClustering

    X = _model_matrix(X, None)
    silhouette_list = []
    for p in k_list:
        clusterer = AgglomerativeClustering(n_clusters=p, linkage="average").fit(X)
        silhouette_list.append(round(metrics.silhouette_score(X, clusterer.labels_), 4))
    key = silhouette_list.index(max(silhouette_list))
    _p("Best silhouette =", max(silhouette_list), " for k=", k_list[key])
    return k_list[key]


class Clusterer:
    """
    Clustering task. * strategy: 'KMEANS', 'HCA' or 'DBSCAN'.
    Quality metric: silhouette.
    """

    def __init__(self, dataset, strategy="HCA", metric="euclidean", verbose=False):
        self.dataset = dataset
        self.strategy = strategy
        self.metric = metric
        self.verbose = verbose

    def get_params(self, deep=True):
        return {"strategy": self.strategy, "metric": self.metric, "verbose": self.verbose}

    def KMEANS_clustering(self, dataset):
        from sklearn import metrics
        from sklearn.cluster import KMeans

        X = _model_matrix(dataset, None)
        if X.shape[1] < 1 or len(X) < 6:
            _p("Error: There are too few observations")
            return None, dataset

        k = compare_k_means([2, 3, 4, 5], dataset)
        final = KMeans(n_clusters=k, n_init=10, random_state=0).fit(X)
        silhouette = round(metrics.silhouette_score(X, final.labels_), 4)
        _p("Quality of clustering", silhouette)
        return silhouette, X.assign(cluster_ID=final.labels_)

    def HCA_clustering(self, dataset, metric):
        from sklearn import metrics
        from sklearn.cluster import AgglomerativeClustering

        X = _model_matrix(dataset, None)
        if X.shape[1] < 1 or len(X) < 6:
            _p("Error: There are too few observations")
            return None, dataset

        k = compare_k_AggClustering([2, 3, 4, 5], dataset)
        # L2C-PORT: AgglomerativeClustering.affinity was renamed to metric in sklearn 1.2
        final = AgglomerativeClustering(n_clusters=k, linkage="average", metric=metric).fit(X)
        silhouette = round(metrics.silhouette_score(X, final.labels_), 4)
        _p("Quality of clustering", silhouette)
        return silhouette, X.assign(cluster_ID=final.labels_)

    def DBSCAN_clustering(self, dataset):
        from sklearn import metrics
        from sklearn.cluster import DBSCAN

        X = _model_matrix(dataset, None)
        if X.shape[1] < 1 or len(X) < 6:
            _p("Error: There are too few observations")
            return None, dataset

        final = DBSCAN(eps=0.1).fit(X)
        if len(set(final.labels_)) < 2:
            return None, dataset
        silhouette = round(metrics.silhouette_score(X, final.labels_), 4)
        return silhouette, X.assign(cluster_ID=final.labels_)

    def transform(self):
        start_time = time.time()
        clustd = self.dataset
        _p(">>Clustering task (applied on the training dataset only)")
        d = self.dataset["train"]

        if self.strategy == "KMEANS":
            dn = self.KMEANS_clustering(d)
        elif self.strategy == "DBSCAN":
            dn = self.DBSCAN_clustering(d)
        elif self.strategy == "HCA":
            if self.metric not in ("cosine", "euclidean", "cityblock"):
                raise ValueError("The clustering metric should be cosine, euclidean or cityblock")
            dn = self.HCA_clustering(d, self.metric)
        else:
            raise ValueError("The Clustering function should be KMEANS, DBSCAN or HCA")

        _p("Clustering done -- CPU time: %s seconds" % (time.time() - start_time))
        # L2C-PORT: upstream overwrites dataset['train'] with the cluster labels,
        # which would corrupt the frame this script has to write out.
        return {"quality_metric": dn[0], "result": dn[1]}


# ===========================================================================
# learn2clean.qlearning.qlearner
# ===========================================================================

def update_q(q, r, state, next_state, action, beta, gamma):
    """Q-learning update, Eq. (2)-(3) of the paper."""
    rsa = r[state, action]
    qsa = q[state, action]
    new_q = qsa + beta * (rsa + gamma * max(q[next_state, :]) - qsa)
    q[state, action] = new_q
    # renormalize row to be between 0 and 1
    rn = q[state][q[state] > 0] / np.sum(q[state][q[state] > 0])
    q[state][q[state] > 0] = rn
    return r[state, action]


def remove_adjacent(nums):
    previous = ""
    for i in nums[:]:  # using the copy of nums
        if i == previous:
            nums.remove(i)
        else:
            previous = i
    return nums


def _parallel_backend() -> str:
    """
    Pick the joblib backend for the greedy traversals.

    Run as a script — the way run_pipeline.sh invokes it — this module is
    ``__main__``, cloudpickle serializes the traversal callable by value and the
    loky processes escape the GIL.

    Loaded any other way (by path from the test-suite, or imported under its own
    name) loky would pickle it by reference and the worker could not resolve the
    module, so the traversals run on threads instead. They stay independent
    either way: each one works on its own deep copy of the dataset.
    """
    return "loky" if __name__ == "__main__" else "threading"


def _deepcopy_dataset(dataset: Dict) -> Dict:
    """Independent copy of a Learn2Clean dataset dict."""
    out = {}
    for key, value in dataset.items():
        if isinstance(value, (pd.DataFrame, pd.Series)):
            out[key] = value.copy(deep=True)
        else:
            out[key] = copy.deepcopy(value)
    return out


class Qlearner:
    """
    Learn2Clean class with Q-learning for data preparation, plus the random
    cleaning and no-preparation baselines.

    Parameters
    ----------
    * dataset: input dataset dict with dataset['train'], dataset['test'] and
        dataset['target'] (and dataset['target_test'])
    * goal: str, the ML method defining the goal state:
        - 'NB', 'LDA', 'CART' and 'MNB' for classification
        - 'HCA' or 'KMEANS' for clustering
        - 'MARS', 'LASSO' or 'OLS' for regression
    * target_goal: str, name of the target variable encoded as int64
    * target_prepare: str, name of the variable excluded from data preparation
    * threshold: float, threshold shared by the preparation methods
    """

    def __init__(self, dataset, goal, target_goal, target_prepare, verbose=False,
                 file_name=None, threshold=None, save_dir="save",
                 fs_threshold=0.3, od_threshold=0.3, dd_threshold=0.6,
                 k_folds=10, n_episodes=1000, gamma=0.8, beta=1.0, epsilon=0.05,
                 seed=1999, n_jobs=1, max_patterns=4, encode_categoricals=True):
        self.dataset = dataset
        self.goal = goal
        self.target_goal = target_goal
        self.target_prepare = target_prepare
        self.verbose = verbose
        self.file_name = file_name
        self.threshold = threshold
        self.save_dir = save_dir
        # L2C-PORT: upstream stores a single `threshold` on the Qlearner but never
        # forwards it to the methods (its own TODO: "handle an array of
        # thresholds"), so each one silently falls back to its default. The three
        # defaults below are exactly those upstream defaults, and they are now
        # forwarded, which makes them tunable from config.yaml.
        self.fs_threshold = fs_threshold
        self.od_threshold = od_threshold
        self.dd_threshold = dd_threshold
        self.k_folds = k_folds
        self.n_episodes = n_episodes
        self.gamma = gamma
        self.beta = beta
        self.epsilon = epsilon
        self.seed = seed
        self.n_jobs = n_jobs
        self.max_patterns = max_patterns
        self.encode_categoricals = encode_categoricals

    def get_params(self, deep=True):
        return {
            "goal": self.goal,
            "target_goal": self.target_goal,
            "target_prepare": self.target_prepare,
            "verbose": self.verbose,
            "file_name": self.file_name,
            "threshold": self.threshold,
        }

    def Initialization_Reward_Matrix(self, dataset):
        """
        Defines the reward/connection graph between the 18 preprocessing methods
        and 1 ML model: a 19x19 matrix when the data has missing values
          4 (MICE EM KNN MF) for imputation
          3 (DS MM ZS) for normalization
          4 (MR WR LC TB) for feature selection
          3 (ZSB LOF IQR) for outlier detection
          2 (CC PC) for inconsistency checking
          2 (AD ED) for duplication detection
          1 (LASSO or OLS or MARS) regression, (HCA or KMEANS) clustering
            or (CART or LDA or NB) classification
        """
        if dataset["train"].copy().isnull().sum().sum() > 0:
            r = np.array([
                [-1, -1, -1, -1, 0, 0, 0, -1, 0, 0, 0, 0, 0, 0, -1, -1, 0, 0, 100],
                [-1, -1, -1, -1, 0, 0, 0, -1, 0, 0, 0, 0, 0, 0, -1, -1, 0, 0, 100],
                [-1, -1, -1, -1, 0, 0, 0, -1, 0, 0, 0, 0, 0, 0, -1, -1, 0, 0, 100],
                [-1, -1, -1, -1, 0, 0, 0, -1, 0, 0, 0, 0, 0, 0, -1, -1, 0, 0, 100],

                [-1, -1, -1, -1, -1, -1, -1, 0, 0, 0, 0, 0, 0, 0, -1, -1, 0, 0, -1],
                [-1, -1, -1, -1, -1, -1, -1, 0, 0, 0, 0, 0, 0, 0, -1, -1, 0, 0, -1],
                [0, 0, 0, 0, -1, -1, -1, 0, 0, 0, 0, 0, 0, 0, -1, -1, 0, 0, -1],

                [0, 0, 0, 0, 0, 0, 0, -1, -1, -1, -1, 0, 0, 0, -1, -1, 0, 0, -1],
                [0, 0, 0, 0, 0, 0, 0, -1, -1, -1, -1, 0, 0, 0, -1, -1, 0, 0, -1],
                [0, 0, 0, 0, 0, 0, 0, -1, -1, -1, -1, 0, 0, 0, -1, -1, 0, 0, -1],
                [0, 0, 0, 0, 0, 0, 0, -1, -1, -1, -1, 0, 0, 0, -1, -1, 0, 0, -1],

                [0, 0, 0, 0, -1, -1, -1, 0, 0, 0, 0, -1, -1, -1, -1, -1, 0, 0, 100],
                [-1, -1, -1, -1, -1, -1, -1, 0, 0, 0, 0, -1, -1, -1, -1, -1, 0, 0, 100],
                [0, 0, 0, 0, -1, -1, -1, 0, 0, 0, 0, -1, -1, -1, -1, -1, 0, 0, 100],

                [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 0, 0, 0, -1, -1, 0, 0, 100],
                [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 0, 0, 0, -1, -1, 0, 0, 100],

                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1, -1, 100],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1, -1, 100],
                [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
            ]).astype("float32")

            n_actions = 19
            n_states = 19
            check_missing = True

        else:  # no imputation needed
            """
            14x14 graph when there are no missing values:
            3 (DS MM ZS) normalization, 3 (WR LC TB) feature selection,
            3 (ZSB LOF IQR) outlier detection, 2 (CC PC) inconsistency checking,
            2 (AD ED) duplication detection, 1 ML model.
            """
            r = np.array([
                [-1, -1, -1, 0, 0, 0, 0, 0, 0, -1, -1, 0, 0, -1],
                [-1, -1, -1, 0, 0, 0, 0, 0, 0, -1, -1, 0, 0, -1],
                [-1, -1, -1, 0, 0, 0, 0, 0, 0, -1, -1, 0, 0, -1],

                [0, 0, 0, -1, -1, -1, 0, 0, 0, -1, -1, 0, 0, -1],
                [0, 0, 0, -1, -1, -1, 0, 0, 0, -1, -1, 0, 0, -1],
                [0, 0, 0, -1, -1, -1, 0, 0, 0, -1, -1, 0, 0, -1],

                [-1, -1, -1, 0, 0, 0, -1, -1, -1, -1, -1, 0, 0, 100],
                [-1, -1, -1, 0, 0, 0, -1, -1, -1, -1, -1, 0, 0, 100],
                [-1, -1, -1, 0, 0, 0, -1, -1, -1, -1, -1, 0, 0, 100],

                [-1, -1, -1, -1, -1, -1, 0, 0, 0, -1, -1, 0, 0, 100],
                [-1, -1, -1, -1, -1, -1, 0, 0, 0, -1, -1, 0, 0, 100],

                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1, -1, 100],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1, -1, 100],

                [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
            ]).astype("float32")

            n_actions = 14
            n_states = 14
            check_missing = False

        q = np.zeros_like(r)

        # we prevent the transition from any ML model (LASSO OLS MARS HCA
        # KMEANS CART LDA NB, the last rows) back to preprocessing
        r = r[~np.all(r == -1, axis=1)]

        if self.verbose:
            _p("Reward matrix", r)

        return q, r, n_actions, n_states, check_missing

    # ------------------------------------------------------------------
    # Pipeline execution
    # ------------------------------------------------------------------

    def pipeline(self, dataset, actions_list, target_goal, target_prepare, check_missing):
        dataset = _deepcopy_dataset(dataset)

        goals_name = ["LASSO", "OLS", "MARS", "HCA", "KMEANS", "CART", "LDA", "NB"]
        res = None

        if check_missing:
            actions_name = ["MICE", "EM", "KNN", "MF",
                            "DS", "MM", "ZS",
                            "MR", "WR", "LC", "Tree",
                            "ZSB", "LOF", "IQR",
                            "CC", "PC",
                            "ED", "AD"]

            L2C_class = [Imputer, Imputer, Imputer, Imputer,
                         Normalizer, Normalizer, Normalizer,
                         Feature_selector, Feature_selector, Feature_selector, Feature_selector,
                         Outlier_detector, Outlier_detector, Outlier_detector,
                         Consistency_checker, Consistency_checker,
                         Duplicate_detector, Duplicate_detector,
                         Regressor, Regressor, Regressor,
                         Clusterer, Clusterer,
                         Classifier, Classifier, Classifier]

            impute_idx = (0, 1, 2, 3)
            # L2C-PORT: upstream tests `a in range(4, 10)` here and `range(0, 5)`
            # in the no-missing branch, which leaves index 10 ("Tree") / 5
            # ("Tree") matching no branch at all — the tree-based feature
            # selector (TB in the paper) then silently never runs. The ranges are
            # corrected so the action space matches Figure 1 of the paper.
            prep_idx = range(4, 11)
            outlier_dedup_idx = (11, 12, 13, 16, 17)
            consistency_idx = (14, 15)
            supervised_goal_idx = (18, 19, 20, 23, 24, 25)
            clustering_goal_idx = (21, 22)
        else:
            actions_name = ["DS", "MM", "ZS",
                            "WR", "LC", "Tree",
                            "ZSB", "LOF", "IQR", "CC",
                            "PC", "ED", "AD"]

            L2C_class = [Normalizer, Normalizer, Normalizer,
                         Feature_selector, Feature_selector, Feature_selector,
                         Outlier_detector, Outlier_detector, Outlier_detector,
                         Consistency_checker, Consistency_checker,
                         Duplicate_detector, Duplicate_detector,
                         Regressor, Regressor, Regressor,
                         Clusterer, Clusterer,
                         Classifier, Classifier, Classifier]

            impute_idx = ()
            prep_idx = range(0, 6)
            outlier_dedup_idx = (6, 7, 8, 11, 12)
            consistency_idx = (9, 10)
            supervised_goal_idx = (13, 14, 15, 18, 19, 20)
            clustering_goal_idx = (16, 17)

        _p("Start pipeline", actions_list)
        start_time = time.time()

        # L2C-PORT: upstream wraps this whole dispatch in
        #     if len(dataset['train'].dropna()) == 0: pass
        # so a frame without a single complete row gets no preparation at all —
        # not even imputation, the action that would fix it. On data_poisoned
        # that silences Learn2Clean entirely on 9 of 15 AR datasets and 2 of 15
        # NAR ones. The check belongs to the goal models, which genuinely need
        # complete cases, and they already enforce it via `len(X_train) < k`.
        for a in actions_list:
            if a >= len(actions_name) and dataset["train"].empty:
                break

            try:
                if a in impute_idx:
                    dataset = L2C_class[a](
                        dataset=dataset, strategy=actions_name[a], verbose=self.verbose
                    ).transform()

                elif a in prep_idx:
                    cls = L2C_class[a]
                    if cls is Feature_selector:
                        dataset = cls(
                            dataset=dataset, strategy=actions_name[a],
                            exclude=target_prepare, threshold=self.fs_threshold,
                            verbose=self.verbose,
                        ).transform()
                    else:
                        dataset = cls(
                            dataset=dataset, strategy=actions_name[a],
                            exclude=target_prepare, verbose=self.verbose,
                        ).transform()

                elif a in outlier_dedup_idx:
                    cls = L2C_class[a]
                    threshold = (
                        self.od_threshold if cls is Outlier_detector else self.dd_threshold
                    )
                    dataset = cls(
                        dataset=dataset, strategy=actions_name[a],
                        threshold=threshold, verbose=self.verbose,
                    ).transform()

                elif a in consistency_idx:
                    dataset = L2C_class[a](
                        dataset=dataset, strategy=actions_name[a],
                        file_name=self.file_name, save_dir=self.save_dir,
                        max_patterns=self.max_patterns, verbose=self.verbose,
                    ).transform()

                elif a in supervised_goal_idx:
                    strategy = goals_name[a - len(actions_name)]
                    res = L2C_class[a](
                        dataset=dataset, strategy=strategy,
                        target=target_goal, k_folds=self.k_folds, verbose=self.verbose,
                    ).transform()

                elif a in clustering_goal_idx:
                    strategy = goals_name[a - len(actions_name)]
                    res = L2C_class[a](
                        dataset=dataset, strategy=strategy, verbose=self.verbose
                    ).transform()

            except Exception as exc:  # a broken step must not kill the traversal
                logger.debug("    action %s failed: %s" % (a, exc))
                if a in supervised_goal_idx or a in clustering_goal_idx:
                    res = {"quality_metric": None}

        t = time.time() - start_time
        _p("End Pipeline CPU time: %s seconds" % t)
        return dataset, res, t

    # ------------------------------------------------------------------
    # Greedy traversals
    # ------------------------------------------------------------------

    def _greedy_traversals(self, q, g, check_missing):
        """Enumerate the greedy traversal starting from every possible state."""
        if check_missing:
            methods = ["MICE", "EM", "KNN", "MF",
                       "DS", "MM", "ZS", "MR", "WR", "LC", "Tree",
                       "ZSB", "LOF", "IQR", "CC", "PC",
                       "ED", "AD"]
        else:
            methods = ["DS", "MM", "ZS",
                       "WR", "LC", "Tree",
                       "ZSB", "LOF", "IQR",
                       "CC", "PC",
                       "ED", "AD"]
        goals = ["LASSO", "OLS", "MARS", "HCA", "KMEANS", "CART", "LDA", "NB"]

        n_states = len(methods) + 1
        methods.append(str(goals[g]))

        traversals = []
        for i in range(len(q) - 1):
            actions_list: List[int] = []
            current_state = i
            traverse_name = "%s -> " % methods[i]
            n_steps = 0

            while current_state != n_states - 1 and n_steps < 20:
                actions_list.append(current_state)
                next_state = int(np.argmax(q[current_state]))
                current_state = next_state
                traverse_name += "%s -> " % methods[next_state]
                actions_list.append(next_state)
                n_steps += 1
                actions_list = remove_adjacent(actions_list)

            del actions_list[-1]
            actions_list.append(g + len(methods) - 1)
            traverse_name = traverse_name[:-4]
            traversals.append((traverse_name, actions_list))

        return traversals, methods, goals

    def show_traverse(self, dataset, q, g, target1, target2, check_missing):
        """
        Execute the greedy traversal from every starting state and collect the
        resulting quality metric *and* the prepared dataset for each of them.
        """
        traversals, methods, goals = self._greedy_traversals(q, g, check_missing)

        # The last candidate is the "no preparation" baseline: the goal model
        # applied straight to the input data.
        traversals.append((goals[g], [g + len(methods) - 1]))

        def _run(traverse_name, actions_list, encode_categoricals=self.encode_categoricals):
            # joblib's loky backend re-imports this module in the worker, which
            # resets module-level globals to their defaults; re-apply the setting.
            global ENCODE_CATEGORICALS
            ENCODE_CATEGORICALS = encode_categoricals
            _p("Greedy traversal:", traverse_name)
            prepared, res, _ = self.pipeline(
                dataset, actions_list, target1, target2, check_missing
            )
            metric = res.get("quality_metric") if isinstance(res, dict) else None
            return traverse_name, actions_list, metric, prepared

        if self.n_jobs and self.n_jobs != 1:
            results = Parallel(n_jobs=self.n_jobs, backend=_parallel_backend())(
                delayed(_run)(name, actions) for name, actions in traversals
            )
        else:
            results = [_run(name, actions) for name, actions in traversals]

        actions_strategy = [r[0] for r in results[:-1]]
        strategy = [{"quality_metric": r[2]} for r in results]
        prepared_datasets = [r[3] for r in results]
        action_lists = [r[1] for r in results]

        _p("==== Recap ====")
        _p("List of strategies tried by Learn2Clean:", actions_strategy)
        _p("List of corresponding quality metrics:", [r[2] for r in results])

        return actions_strategy, strategy, prepared_datasets, action_lists

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------

    def learn2clean(self) -> Dict:
        """
        Run Learn2Clean end to end and return the best strategy together with
        the dataset it produced.

        L2C-PORT: upstream prints the winning strategy and appends it to
        ``./save/<file_name>_results.txt`` without ever handing back the prepared
        data. The tuple it logs is still produced (as ``result_record``) but the
        prepared dataset is returned as well, which is what this baseline needs.
        """
        goals = ["LASSO", "OLS", "MARS", "HCA", "KMEANS", "CART", "LDA", "NB"]
        metrics_name = ["MSE", "MSE", "MSE", "silhouette", "silhouette",
                        "accuracy", "accuracy", "accuracy"]

        if self.goal not in goals:
            raise ValueError(
                "Goal invalid. Please choose between 'LASSO', 'OLS', 'MARS' for "
                "regression, 'HCA' or 'KMEANS' for clustering, 'CART', 'LDA', or "
                "'NB' for classification."
            )
        g = goals.index(self.goal)

        if self.target_goal != self.dataset["target"].name:
            raise ValueError("Target variable invalid.")

        start_l2c = time.time()
        _p("Start Learn2Clean")

        gamma = self.gamma
        beta = self.beta
        n_episodes = self.n_episodes
        epsilon = self.epsilon
        random_state = np.random.RandomState(self.seed)

        q, r, n_actions, n_states, check_missing = self.Initialization_Reward_Matrix(self.dataset)

        for _e in range(int(n_episodes)):
            states = list(range(n_states))
            random_state.shuffle(states)
            current_state = states[0]
            goal_reached = False

            while (not goal_reached) and (current_state != n_states - 1):
                # epsilon greedy
                valid_moves = r[current_state] >= 0

                if random_state.rand() < epsilon:
                    actions = np.array(list(range(n_actions)))[valid_moves]
                    random_state.shuffle(actions)
                    action = actions[0]
                    next_state = action
                else:
                    if np.sum(q[current_state]) > 0:
                        action = int(np.argmax(q[current_state]))
                    else:
                        actions = np.array(list(range(n_actions)))[valid_moves]
                        random_state.shuffle(actions)
                        action = actions[0]
                    next_state = action

                reward = update_q(q, r, current_state, next_state, action, beta, gamma)
                if reward > 1:
                    goal_reached = True
                current_state = next_state

        if self.verbose:
            _p("Q-value matrix", q)

        qlearning_time = time.time() - start_l2c
        logger.debug(
            "Learn2Clean - Pipeline construction -- CPU time: %s seconds" % qlearning_time
        )

        _p("=== Start Pipeline Execution ===")
        start_pipexec = time.time()

        actions_strategy, strategy, prepared, action_lists = self.show_traverse(
            self.dataset, q, g, self.target_goal, self.target_prepare, check_missing
        )

        quality_metric_list = [s["quality_metric"] for s in strategy]
        actions_strategy = list(actions_strategy) + [goals[g]]

        result = None
        result_l = None
        scored = [x for x in quality_metric_list if x is not None]

        if scored:
            # L2C-PORT: upstream tests `g in range(0, 2)`, which leaves MARS (g=2,
            # an MSE goal) on the maximization branch. The metric family decides.
            if metrics_name[g] == "MSE":
                result = min(scored)
            else:
                result = max(scored)
            result_l = quality_metric_list.index(result)

        pipeline_time = time.time() - start_pipexec
        _p("=== End of Learn2Clean - Pipeline execution -- CPU time: %s seconds" % pipeline_time)

        if result_l is not None:
            result_record = (self.file_name, "learn2clean", goals[g], self.target_goal,
                             self.target_prepare, actions_strategy[result_l],
                             metrics_name[g], result, pipeline_time)
        else:
            result_record = (self.file_name, "learn2clean", goals[g], self.target_goal,
                             self.target_prepare, None, metrics_name[g], result, pipeline_time)

        _p("**** Best strategy ****", result_record)

        return {
            "goal": goals[g],
            "metric_name": metrics_name[g],
            "quality_metric": result,
            "strategy": actions_strategy[result_l] if result_l is not None else None,
            "actions": action_lists[result_l] if result_l is not None else None,
            "dataset": prepared[result_l] if result_l is not None else None,
            "check_missing": check_missing,
            "all_strategies": actions_strategy,
            "all_metrics": quality_metric_list,
            "qlearning_time_s": qlearning_time,
            "pipeline_time_s": pipeline_time,
            "result_record": result_record,
        }

    # ------------------------------------------------------------------
    # Baselines shipped with the reference implementation
    # ------------------------------------------------------------------

    def random_cleaning(self, dataset_name: str) -> Dict:
        """Random cleaning baseline (RAND in the paper's Table 2)."""
        rng = random.Random(self.seed)
        check_missing = self.dataset["train"].isnull().sum().sum() > 0
        goals = ["LASSO", "OLS", "MARS", "HCA", "KMEANS", "CART", "LDA", "NB"]

        if self.goal not in goals:
            raise ValueError("Goal invalid.")
        g = goals.index(self.goal)

        if check_missing:
            blocks = [(0, 3), (4, 6), (7, 10), (11, 13), (14, 15), (16, 17)]
            n_methods = 18
        else:
            blocks = [(0, 2), (3, 5), (6, 8), (9, 10), (11, 12)]
            n_methods = 13

        # one (or no) curation task per block, in no particular order
        actions_list = []
        for lo, hi in blocks:
            if rng.random() < 0.5:
                continue
            actions_list.append(rng.randint(lo, hi))
        rng.shuffle(actions_list)
        actions_list.append(g + n_methods)

        _p("Random cleaning strategy:", actions_list)
        prepared, res, t = self.pipeline(
            self.dataset, actions_list, self.target_goal, self.target_prepare, check_missing
        )
        metric = res.get("quality_metric") if isinstance(res, dict) else None
        return {"dataset": prepared, "quality_metric": metric, "actions": actions_list, "time_s": t}

    def no_prep(self, dataset_name: str) -> Dict:
        """No-preparation baseline (NO-PRE in the paper's Table 2)."""
        goals = ["LASSO", "OLS", "MARS", "HCA", "KMEANS", "CART", "LDA", "NB"]
        if self.goal not in goals:
            raise ValueError("Goal invalid.")
        g = goals.index(self.goal)

        check_missing = self.dataset["train"].isnull().sum().sum() > 0
        len_m = 18 if check_missing else 13

        prepared, res, t = self.pipeline(
            self.dataset, [g + len_m], self.target_goal, self.target_prepare, check_missing
        )
        metric = res.get("quality_metric") if isinstance(res, dict) else None
        return {"dataset": prepared, "quality_metric": metric, "time_s": t}


# ---------------------------------------------------------------------------
# Driver: Learn2Clean applied to one poisoned dataset
# ---------------------------------------------------------------------------

ACTION_NAMES_MISSING = ["MICE", "EM", "KNN", "MF",
                        "DS", "MM", "ZS",
                        "MR", "WR", "LC", "Tree",
                        "ZSB", "LOF", "IQR",
                        "CC", "PC",
                        "ED", "AD"]

ACTION_NAMES_NO_MISSING = ["DS", "MM", "ZS",
                           "WR", "LC", "Tree",
                           "ZSB", "LOF", "IQR",
                           "CC", "PC",
                           "ED", "AD"]

GOAL_NAMES = ["LASSO", "OLS", "MARS", "HCA", "KMEANS", "CART", "LDA", "NB"]


def _action_names(actions: Optional[List[int]], check_missing: bool) -> List[str]:
    if not actions:
        return []
    names = ACTION_NAMES_MISSING if check_missing else ACTION_NAMES_NO_MISSING
    out = []
    for a in actions:
        out.append(names[a] if a < len(names) else GOAL_NAMES[a - len(names)])
    return out


# Actions replayed on a held-out frame: the ones that transform values in place.
# Feature selection is replayed as a plain column projection (kept_columns), and
# the row-removing actions (outlier detection, deduplication, consistency
# checking) are skipped so the hold-out keeps every row — the same split of
# responsibilities as `_apply_saga_no_outliers` in quail/data.py.
TEST_REPLAY_IMPUTERS = {"MICE", "EM", "KNN", "MF", "MEAN", "MEDIAN", "RAND"}
TEST_REPLAY_NORMALIZERS = {"ZS", "MM", "DS", "Log10"}
TEST_REPLAY_ACTIONS = TEST_REPLAY_IMPUTERS | TEST_REPLAY_NORMALIZERS


class Learn2Clean:
    """
    Learn2Clean baseline driver.

    Wraps the Q-learner in the same interface the other data-preparation
    baselines of this repository expose (``prepare`` -> cleaned frame, residual
    mask, performance metrics, reusable pipeline description).
    """

    def __init__(self, goal="LDA", k_folds=10, n_episodes=1000, gamma=0.8, beta=1.0,
                 epsilon=0.05, seed=1999, n_jobs=1, fs_threshold=0.3, od_threshold=0.3,
                 dd_threshold=0.6, max_patterns=4, save_dir="save", verbose=False,
                 encode_categoricals=None):
        self.goal = goal
        self.k_folds = k_folds
        self.n_episodes = n_episodes
        self.gamma = gamma
        self.beta = beta
        self.epsilon = epsilon
        self.seed = seed
        self.n_jobs = n_jobs
        self.fs_threshold = fs_threshold
        self.od_threshold = od_threshold
        self.dd_threshold = dd_threshold
        self.max_patterns = max_patterns
        self.save_dir = save_dir
        self.verbose = verbose
        self.encode_categoricals = (
            ENCODE_CATEGORICALS if encode_categoricals is None else encode_categoricals
        )

    def apply_to_test(self, df_test: pd.DataFrame, pipeline_info: Dict) -> pd.DataFrame:
        """
        Project a held-out frame onto the preparation chosen for the training data.

        Learn2Clean drops columns (feature selection) and rescales values
        (normalization), so a hold-out read straight from ``data_poisoned/test/``
        would no longer match the frame the model was fitted on. This replays the
        value-transforming actions of the winning strategy, projects the columns
        onto the ones that survived, and skips every row-removing action so the
        hold-out keeps all of its rows.

        No fitted state has to be carried over: upstream normalizes and imputes
        ``dataset['train']`` and ``dataset['test']`` independently inside the same
        ``transform()`` call, each from its own values, so replaying the actions
        here is exactly what ``Qlearner.pipeline`` would have done had the frame
        been sitting in ``dataset['test']``.
        """
        kept = [c for c in pipeline_info["kept_columns"] if c in df_test.columns]
        out = df_test[kept].copy()

        replay = [
            name
            for name in (pipeline_info.get("action_names") or [])
            if name in TEST_REPLAY_ACTIONS
        ]
        if not replay:
            return out

        target_col = pipeline_info["target_col"]
        if target_col in out.columns:
            y = pd.Series(pd.factorize(out[target_col])[0], index=out.index, name=target_col)
        else:
            y = pd.Series(0, index=out.index, name=target_col)

        dataset = {"train": out, "test": {}, "target": y, "target_test": {}}
        for name in replay:
            try:
                if name in TEST_REPLAY_IMPUTERS:
                    dataset = Imputer(
                        dataset=dataset, strategy=name, verbose=self.verbose
                    ).transform()
                else:
                    dataset = Normalizer(
                        dataset=dataset, strategy=name, exclude=target_col,
                        verbose=self.verbose,
                    ).transform()
            except Exception as exc:
                logger.debug(f"    test replay of {name} failed: {exc}")

        out = dataset["train"]
        return out[[c for c in kept if c in out.columns]].sort_index()

    def prepare(
        self,
        df_poisoned: pd.DataFrame,
        mask_df: pd.DataFrame,
        file_name: str = "dataset",
    ) -> Tuple[pd.DataFrame, pd.DataFrame, Dict, Dict]:
        """
        Run Learn2Clean on one poisoned dataset.

        Returns
        -------
        df_clean     : cleaned dataframe (rows/columns may have been removed)
        residual_mask: boolean mask over df_clean (True = poisoned and still NaN)
        perf_metrics : timing / memory metrics dict
        pipeline_info: the winning strategy plus the surviving row and column
                       labels, for reapplying the same preparation downstream
        """
        proc = psutil.Process()
        ram_before_mb = proc.memory_info().rss / 1024 / 1024
        tracemalloc.start()
        wall_t0 = time.perf_counter()
        cpu_t0 = time.process_time()

        col_types = _identify_col_types(df_poisoned)
        if col_types["target_cls"]:
            target_col = col_types["target_cls"][0]
            task = "classification"
        elif col_types["target_reg"]:
            target_col = col_types["target_reg"][0]
            task = "regression"
        else:
            raise ValueError("No cls_* or reg_* target column found")

        # The goal models need an int64-encoded target (upstream's `target_goal`),
        # while the frame keeps the original labels so they are written back as-is.
        y_raw = df_poisoned[target_col]
        if task == "classification":
            y = pd.Series(pd.factorize(y_raw)[0], index=df_poisoned.index, name=target_col)
        else:
            y = pd.to_numeric(y_raw, errors="coerce").rename(target_col)

        dataset = {
            "train": df_poisoned.copy(),
            # No hold-out frame: the quAIL test split lives outside the data being
            # cleaned, so the goal model scores by k-fold CV (upstream's own
            # "target absent from the test set" branch).
            "test": {},
            "target": y,
            "target_test": {},
        }

        learner = Qlearner(
            dataset=dataset,
            goal=self.goal,
            target_goal=target_col,
            target_prepare=target_col,
            verbose=self.verbose,
            file_name=file_name,
            threshold=self.od_threshold,
            save_dir=self.save_dir,
            fs_threshold=self.fs_threshold,
            od_threshold=self.od_threshold,
            dd_threshold=self.dd_threshold,
            k_folds=self.k_folds,
            n_episodes=self.n_episodes,
            gamma=self.gamma,
            beta=self.beta,
            epsilon=self.epsilon,
            seed=self.seed,
            n_jobs=self.n_jobs,
            max_patterns=self.max_patterns,
            encode_categoricals=self.encode_categoricals,
        )

        result = learner.learn2clean()

        # ---- rebuild the cleaned frame ------------------------------------
        if result["dataset"] is None:
            logger.warning(
                "  [Learn2Clean] No strategy produced a quality metric; "
                "returning the input data unchanged"
            )
            df_clean = df_poisoned.copy()
        else:
            df_clean = result["dataset"]["train"]

        # The upstream transformers rebuild frames with joins, which reorders
        # columns; restore the input order over the surviving columns.
        kept_cols = [c for c in df_poisoned.columns if c in df_clean.columns]
        df_clean = df_clean[kept_cols]

        # The target is passed as `target_prepare`, so upstream's `exclude`
        # mechanism keeps it; re-attach it defensively if a step still lost it.
        if target_col not in df_clean.columns:
            df_clean.insert(len(df_clean.columns), target_col, df_poisoned.loc[df_clean.index, target_col])
            kept_cols = [c for c in df_poisoned.columns if c in df_clean.columns]
            df_clean = df_clean[kept_cols]

        df_clean = df_clean.sort_index()

        # ---- residual mask, aligned on the surviving rows and columns ------
        mask_sub = mask_df.reindex(index=df_clean.index, columns=df_clean.columns, fill_value=False)
        residual_mask = mask_sub & df_clean.isna()

        poisoned_kept = int(mask_sub.sum().sum())
        poisoned_total = int(mask_df.sum().sum())
        residual = int(residual_mask.sum().sum())
        logger.info(
            f"  [Learn2Clean] {poisoned_total - poisoned_kept} poisoned cells dropped with "
            f"rows/columns, {poisoned_kept - residual} repaired, {residual} residual"
        )

        wall_time_s = time.perf_counter() - wall_t0
        cpu_time_s = time.process_time() - cpu_t0
        _, peak_bytes = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        ram_after_mb = proc.memory_info().rss / 1024 / 1024
        ram_peak_mb = peak_bytes / 1024 / 1024

        n_rows = len(df_poisoned)
        action_names = _action_names(result["actions"], result["check_missing"])
        perf_metrics = {
            "n_rows": n_rows,
            "n_cols": len(df_poisoned.columns),
            "n_rows_out": len(df_clean),
            "n_cols_out": len(df_clean.columns),
            "wall_time_s": round(wall_time_s, 4),
            "cpu_time_s": round(cpu_time_s, 4),
            "ram_before_mb": round(ram_before_mb, 3),
            "ram_after_mb": round(ram_after_mb, 3),
            "ram_peak_mb": round(ram_peak_mb, 3),
            "throughput_rows_per_s": round(n_rows / wall_time_s, 4) if wall_time_s > 0 else float("inf"),
            "goal": result["goal"],
            "metric_name": result["metric_name"],
            "best_pipeline": " → ".join(action_names) if action_names else "(none)",
            "best_score": result["quality_metric"],
            "qlearning_time_s": round(result["qlearning_time_s"], 4),
            "pipeline_time_s": round(result["pipeline_time_s"], 4),
            "n_strategies_tried": len(result["all_strategies"]),
        }

        pipeline_info = {
            "method": "learn2clean",
            "goal": result["goal"],
            "metric_name": result["metric_name"],
            "quality_metric": result["quality_metric"],
            "strategy": result["strategy"],
            "actions": result["actions"],
            "action_names": action_names,
            "test_replay_actions": [n for n in action_names if n in TEST_REPLAY_ACTIONS],
            "check_missing": result["check_missing"],
            "target_col": target_col,
            "col_types": col_types,
            "kept_rows": list(df_clean.index),
            "kept_columns": list(df_clean.columns),
            "dropped_columns": [c for c in df_poisoned.columns if c not in df_clean.columns],
            "all_strategies": result["all_strategies"],
            "all_metrics": result["all_metrics"],
            "params": {
                "k_folds": self.k_folds,
                "n_episodes": self.n_episodes,
                "gamma": self.gamma,
                "beta": self.beta,
                "epsilon": self.epsilon,
                "seed": self.seed,
                "fs_threshold": self.fs_threshold,
                "od_threshold": self.od_threshold,
                "dd_threshold": self.dd_threshold,
                "max_patterns": self.max_patterns,
                "encode_categoricals": self.encode_categoricals,
            },
        }

        return df_clean, residual_mask, perf_metrics, pipeline_info


# ---------------------------------------------------------------------------
# Metrics helper (mirrors saga.py / data_preparation_pipeline.py)
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
    """Persist the winning strategy and the surviving row/column labels."""
    import pickle as _pickle

    with open(path, "wb") as f:
        _pickle.dump(pipeline_info, f)


# ---------------------------------------------------------------------------
# Dataset processing (mirrors process_all_datasets in saga.py)
# ---------------------------------------------------------------------------

def check_dataset_complete(csv_file: Path, output_dir: str) -> bool:
    required_files = []
    for mode in ("ar", "nar"):
        mode_dir = os.path.join(output_dir, mode)
        required_files += [
            os.path.join(mode_dir, csv_file.name),
            os.path.join(mode_dir, csv_file.stem + "_mask.csv"),
            os.path.join(mode_dir, csv_file.stem + "_pipeline.pkl"),
            os.path.join(output_dir, "test", mode, csv_file.name),
            os.path.join(output_dir, "metrics", f"{csv_file.stem}_{mode}_metrics.csv"),
            os.path.join(output_dir, "metrics", f"{csv_file.stem}_{mode}_perf_metrics.csv"),
        ]
    return all(os.path.exists(f) for f in required_files)


def process_all_datasets(
    input_dir: str,
    output_dir: str,
    datasets: Optional[List[str]] = None,
    goal: str = "LDA",
    k_folds: int = 10,
    n_episodes: int = 1000,
    gamma: float = 0.8,
    beta: float = 1.0,
    epsilon: float = 0.05,
    seed: int = 1999,
    n_jobs: int = 1,
    fs_threshold: float = 0.3,
    od_threshold: float = 0.3,
    dd_threshold: float = 0.6,
    max_patterns: int = 4,
    verbose: bool = False,
):
    """
    Apply Learn2Clean to all poisoned datasets.

    Reads poisoned CSVs and their masks from ``input_dir/{ar,nar}/`` and writes
    cleaned CSVs, residual masks, the selected strategy and metrics to
    ``output_dir/{ar,nar}/``.
    """
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.join(output_dir, "ar"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "nar"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "metrics"), exist_ok=True)
    # The clean hold-out has to follow the columns and the scale the training
    # frame ended up with, so one prepared copy is written per corruption mode.
    os.makedirs(os.path.join(output_dir, "test", "ar"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "test", "nar"), exist_ok=True)
    # Upstream keeps the discovered constraints/patterns in a `save/` directory;
    # here they live beside the cleaned data so a run is self-contained.
    save_dir = os.path.join(output_dir, "constraints")
    os.makedirs(save_dir, exist_ok=True)

    cleaner = Learn2Clean(
        goal=goal, k_folds=k_folds, n_episodes=n_episodes, gamma=gamma, beta=beta,
        epsilon=epsilon, seed=seed, n_jobs=n_jobs, fs_threshold=fs_threshold,
        od_threshold=od_threshold, dd_threshold=dd_threshold,
        max_patterns=max_patterns, save_dir=save_dir, verbose=verbose,
    )

    ar_dir = Path(input_dir) / "ar"
    nar_dir = Path(input_dir) / "nar"

    if not ar_dir.exists():
        logger.error(f"AR poisoned directory not found: {ar_dir}")
        return
    if not nar_dir.exists():
        logger.error(f"NAR poisoned directory not found: {nar_dir}")
        return

    test_dir = Path(input_dir) / "test"
    if not test_dir.exists():
        logger.warning(
            f"Clean hold-out directory not found: {test_dir}; "
            "no prepared test set will be written"
        )

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

    logger.info(f"Learn2Clean goal state: {goal} | k_folds={k_folds} | n_jobs={n_jobs}")

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
                out_test = Path(output_dir) / "test" / mode / csv_file.name
                out_metrics = Path(output_dir) / "metrics" / f"{csv_file.stem}_{mode}_metrics.csv"
                out_perf = Path(output_dir) / "metrics" / f"{csv_file.stem}_{mode}_perf_metrics.csv"

                if all(p.exists() for p in [out_csv, out_mask, out_pkl, out_test, out_metrics, out_perf]):
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

                df_clean, residual_mask, perf, pipeline_info = cleaner.prepare(
                    df_mode, mask_mode, file_name=f"{csv_file.stem}_{mode}"
                )
                metrics = calculate_residual_metrics(residual_mask)

                df_clean.to_csv(out_csv, index=False)
                residual_mask.to_csv(out_mask, index=False)
                save_pipeline_info(str(out_pkl), pipeline_info)

                # Prepare the clean hold-out the same way, so the columns and the
                # scale match what the model will be fitted on.
                src_test = test_dir / csv_file.name
                if src_test.exists():
                    df_test = pd.read_csv(
                        src_test, na_values=["?", "NA", "N/A", "NaN", "nan", "NAN", "", " "]
                    )
                    df_test_clean = cleaner.apply_to_test(df_test, pipeline_info)
                    df_test_clean.to_csv(out_test, index=False)
                    logger.info(
                        f"  {mode.upper()} hold-out prepared: "
                        f"{df_test_clean.shape[0]}×{df_test_clean.shape[1]} "
                        f"(from {df_test.shape[0]}×{df_test.shape[1]}) "
                        f"| replayed: {' → '.join(pipeline_info['test_replay_actions']) or '(none)'}"
                    )
                else:
                    logger.warning(f"  {mode.upper()} clean hold-out not found: {src_test}")

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
                    f"| {perf['n_rows_out']}×{perf['n_cols_out']} (from {perf['n_rows']}×{perf['n_cols']}) "
                    f"| {perf['wall_time_s']:.1f}s wall | {perf['ram_peak_mb']:.1f} MB peak "
                    f"| {perf['metric_name']}={perf['best_score']} "
                    f"| pipeline: {perf['best_pipeline']}"
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
        description="Learn2Clean reinforcement-learning data preparation baseline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
                Paper: Laure Berti-Equille. "Learn2Clean: Optimizing the Sequence of
                    Tasks for Web Data Preparation." WWW '19, pp. 2580-2586.
                Code : https://github.com/LaureBerti/Learn2Clean

                Q-learning explores the state-action graph of Figure 1; the greedy
                traversal from every starting state is then executed and the
                pipeline optimizing the goal metric is kept.

                Preparation / cleaning methods:
                Normalization       : MM, ZS, DS
                Feature selection   : MR, WR, LC, Tree
                Imputation          : MICE, EM, KNN, MF
                Outlier detection   : IQR, LOF, ZSB
                Inconsistency check : CC, PC
                Deduplication       : ED, AD

                Goal states: CART / LDA / NB (accuracy), LASSO / OLS (MSE),
                             HCA / KMEANS (silhouette)

                Examples:
                python scripts/learn2clean.py
                python scripts/learn2clean.py --input_dir data_poisoned --output_dir data_cleaned_learn2clean
                python scripts/learn2clean.py --dataset tic_tac_toe --goal CART --verbose
        """,
    )
    parser.add_argument(
        "--input_dir", type=str, default="data_poisoned",
        help="Root directory of poisoned data (default: data_poisoned)",
    )
    parser.add_argument(
        "--output_dir", type=str, default=None,
        help="Root directory for cleaned output "
             "(default: config.yaml learn2clean_data_dir, else data_cleaned_learn2clean)",
    )
    parser.add_argument(
        "--config", type=str, default="config.yaml",
        help="Path to experiment config file (default: config.yaml)",
    )
    parser.add_argument(
        "--dataset", type=str, default=None,
        help="Process only this dataset (overrides config; name without .csv)",
    )
    parser.add_argument("--goal", type=str, default=None,
                        help="Goal-state ML model: LDA, CART, NB, MNB, LASSO, OLS, HCA, KMEANS")
    parser.add_argument("--k_folds", type=int, default=None,
                        help="Cross-validation folds used to score a pipeline")
    parser.add_argument("--n_episodes", type=int, default=None,
                        help="Q-learning exploration episodes")
    parser.add_argument("--gamma", type=float, default=None, help="Q-learning discount factor")
    parser.add_argument("--beta", type=float, default=None, help="Q-learning learning rate")
    parser.add_argument("--epsilon", type=float, default=None,
                        help="Epsilon-greedy exploration probability")
    parser.add_argument("--seed", type=int, default=None, help="Random seed")
    parser.add_argument("--n_jobs", type=int, default=None,
                        help="Parallel workers for the greedy traversals (-1 = all cores)")
    parser.add_argument("--fs_threshold", type=float, default=None,
                        help="Feature-selection threshold (MR missing ratio / LC correlation)")
    parser.add_argument("--od_threshold", type=float, default=None,
                        help="Outlier-detection threshold (-1 = any outlying value in a row)")
    parser.add_argument("--dd_threshold", type=float, default=None,
                        help="Deduplication similarity threshold")
    parser.add_argument("--max_patterns", type=int, default=None,
                        help="Patterns induced per column for PC consistency checking")
    parser.add_argument("--no_encode_categoricals", action="store_true",
                        help="Strict upstream behaviour: the goal model only sees num_* columns")
    parser.add_argument("--verbose", action="store_true",
                        help="Show the reference implementation's step-by-step output")

    args = parser.parse_args()

    _config: dict = {}
    if Path(args.config).exists():
        with open(args.config) as _f:
            _config = yaml.safe_load(_f) or {}

    l2c_cfg = _config.get("learn2clean") or {}

    def _opt(name, default):
        """CLI flag > config.yaml learn2clean.<name> > built-in default."""
        cli = getattr(args, name, None)
        if cli is not None:
            return cli
        return l2c_cfg.get(name, default)

    if args.verbose:
        logger.remove()
        logger.add(
            sys.stderr,
            level="DEBUG",
            format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <level>{message}</level>",
        )

    ENCODE_CATEGORICALS = not args.no_encode_categoricals and bool(
        l2c_cfg.get("encode_categoricals", True)
    )

    output_dir = args.output_dir or _config.get(
        "learn2clean_data_dir", "data_cleaned_learn2clean"
    )

    if args.dataset:
        datasets = [args.dataset]
    else:
        datasets = _config.get("datasets") or None

    process_all_datasets(
        input_dir=args.input_dir,
        output_dir=output_dir,
        datasets=datasets,
        goal=_opt("goal", "LDA"),
        k_folds=_opt("k_folds", 10),
        n_episodes=_opt("n_episodes", 1000),
        gamma=_opt("gamma", 0.8),
        beta=_opt("beta", 1.0),
        epsilon=_opt("epsilon", 0.05),
        seed=_opt("seed", 1999),
        n_jobs=_opt("n_jobs", 1),
        fs_threshold=_opt("fs_threshold", 0.3),
        od_threshold=_opt("od_threshold", 0.3),
        dd_threshold=_opt("dd_threshold", 0.6),
        max_patterns=_opt("max_patterns", 4),
        verbose=args.verbose,
    )
