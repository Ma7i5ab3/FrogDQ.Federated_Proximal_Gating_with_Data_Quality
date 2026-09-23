#!/usr/bin/env python3
"""
DiffPrep Data Preparation Baseline
Python port of the official reference implementation from:

  Peng Li, Zhiyi Chen, Xu Chu, Kexin Rong. "DiffPrep: Differentiable Data
  Preprocessing Pipeline Search for Learning over Tabular Data." SIGMOD '23,
  Seattle, Article 183. https://doi.org/10.1145/3589328

Upstream source: https://github.com/chu-data-lab/DiffPrep

DiffPrep formalizes preprocessing-pipeline selection as a bi-level optimization
problem (Section 2.3): the inner problem fits the ML model on the transformed
training data, the outer problem moves the pipeline parameters to minimize the
validation loss. The discrete space of pipelines is relaxed into a continuous
one — softmax over the operator choice of every transformation type (the beta
matrices, Section 3.2) and Sinkhorn normalization over the order of the
transformation types (the alpha doubly-stochastic matrix, Section 4.2) — so the
whole search runs by gradient descent while the ML model is trained only once.

Two variants, exactly as in the paper:
  DiffPrep-Fix  (Section 3) — pre-defined transformation order, learns which
                 operator each transformation type applies to each feature.
  DiffPrep-Flex (Section 4) — also learns the order, one permutation per
                 feature, through Sinkhorn normalization.

Search space (Table 1 / prep_space.py upstream), per feature:
  Missing value imputation : mean, median, DT, MICE, mode        (numerical)
                             mode, dummy variable                (categorical)
  Normalization            : ZS, MM, MA, RS
  Outlier removal          : ZS_4, ZS_3, ZS_2, MAD_3, MAD_2.5, MAD_2,
                             IQR_2, IQR_1.5, IQR_1, identity
  Discretization           : uniform/quantile x {5, 10, 20} bins, identity

Input:  data_poisoned/{ar,nar}/       (poisoned CSV + mask from poison_data.py)
        data_poisoned/test/           (clean hold-out)
Output: data_cleaned_diffprep/{ar,nar}/      (cleaned CSV + residual mask + pipeline)
        data_cleaned_diffprep/test/{ar,nar}/  (hold-out, transformed with the
                                               transformers fitted on training)
        data_cleaned_diffprep/clean/{ar,nar}/ (the un-poisoned train+val partition,
                                               transformed the same way, so a clean
                                               validation split stays on the scale
                                               the model was fitted on)

Every deviation from upstream is flagged with a "DP-PORT" note. The three that
matter for reading the results:

  1. Schema-preserving output. Upstream feeds its ML model a fully numeric
     matrix in which categorical columns have been one-hot encoded. The other
     data-preparation baselines of this repository write a CSV that keeps the
     num_* / cat_* / cls_* schema, so the poison mask and the per-column
     cleanliness metrics stay aligned. This port therefore materializes every
     learned step *except* the one-hot encoding, which quail's
     TabularPreprocessor performs downstream anyway. Numerical columns come out
     imputed, normalized, outlier-repaired and discretized; categorical columns
     come out imputed and keep their labels.

  2. No second standardization downstream. Because the written num_* values are
     already on the scale DiffPrep chose, quail.data.load_diffprep_data builds
     its TabularPreprocessor with scale_numerical=False.

  3. Discrete materialization. Upstream never writes a dataset out: it reads the
     accuracy off the relaxed pipeline. Here, once the search has converged, the
     argmax operator of every feature (and, for Flex, the argmax order, resolved
     into a valid permutation with the Hungarian algorithm) is re-fitted on the
     full poisoned partition and applied. That is the pipeline DiffPrep selected,
     executed discretely — which is what a data-preparation baseline has to hand
     to the next stage.

Note: the target column (cls_*, reg_*) is never transformed.
"""

import copy
import gc
import os
import sys
import time
import tracemalloc
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import psutil
import torch
import torch.nn as nn
from loguru import logger
from scipy.optimize import linear_sum_assignment
from sklearn.experimental import enable_iterative_imputer  # noqa: F401
from sklearn.impute import IterativeImputer, KNNImputer, SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import (
    MaxAbsScaler,
    MinMaxScaler,
    OneHotEncoder,
    RobustScaler,
    StandardScaler,
)
from sklearn.tree import DecisionTreeRegressor
from torch.autograd import Variable
from torch.distributions.utils import logits_to_probs, probs_to_logits

warnings.filterwarnings("ignore")

logger.remove()
logger.add(
    sys.stderr,
    level="INFO",
    format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <level>{message}</level>",
)


# ===========================================================================
# Transformation operators — upstream TFs/
# ===========================================================================

class Identity(object):
    """Identity Transformer (TFs/identity.py)."""

    def __init__(self):
        self.method = "identity"

    def fit(self, X):
        pass

    def transform(self, X):
        return X

    def fit_transform(self, X):
        return X


class Normalizer(object):
    """Normalize the dataset (TFs/normalizer.py).

    Available methods:
        - 'ZS' z-score normalization
        - 'MM' MinMax scaler
        - 'RS' Robust scaling
        - 'MA' MaxAbsolute scaling
    """

    def __init__(self, method="ZS"):
        self.method = method
        if self.method == "ZS":
            self.tf = StandardScaler()
        elif self.method == "MM":
            self.tf = MinMaxScaler(clip=True)
        elif self.method == "MA":
            self.tf = MaxAbsScaler()
        elif self.method == "RS":
            self.tf = RobustScaler()
        else:
            raise Exception("Invalid normalization method: {}".format(method))

    def fit(self, X):
        self.tf.fit(X)

    def transform(self, X):
        X = self.tf.transform(X)
        X = X.clip(-1e10, 1e10)
        return X

    def fit_transform(self, X):
        self.fit(X)
        return self.transform(X)


class UniformDiscretizer(object):
    def __init__(self, n_bins):
        self.n_bins = n_bins

    def fit(self, X):
        self.x_max = X.max(axis=0, keepdims=True)
        self.x_min = X.min(axis=0, keepdims=True)
        self.step = (self.x_max - self.x_min) / self.n_bins

    def transform(self, X):
        X_trans = (X - self.x_min) / (self.step + 1e-12)
        X_trans = X_trans.astype(int)
        X_trans = np.clip(X_trans, 0, self.n_bins - 1)
        return X_trans

    def fit_transform(self, X):
        self.fit(X)
        return self.transform(X)


class QuantileDiscretizer(object):
    def __init__(self, n_bins):
        self.n_bins = n_bins

    def fit(self, X):
        sort_X = np.sort(X, axis=0)
        step = round(len(X) / self.n_bins)
        # DP-PORT: upstream assumes len(X) >= n_bins, which holds for its own
        # datasets but not for a small mini-batch of a small dataset; a step of
        # 0 makes np.arange hang. Fall back to a single split in that case.
        step = max(int(step), 1)
        indices = np.arange(step - 1, len(X) - 1, step)
        if len(indices) == 0:
            indices = np.array([max(len(X) - 1, 0)])
        self.split = sort_X[indices, :]

    def transform(self, X):
        X_trans = np.zeros_like(X)
        for i in range(self.split.shape[0]):
            X_trans += X > self.split[i : i + 1]
        X_trans = np.clip(X_trans, 0, self.n_bins - 1)
        return X_trans

    def fit_transform(self, X):
        self.fit(X)
        return self.transform(X)


class Discretizer(object):
    """Discretize data (TFs/discretizer.py).

    Args:
        n_bins: number of bins
        strategy: {'uniform', 'quantile'}
    """

    def __init__(self, n_bins=5, strategy="uniform"):
        self.method = "{}_{}".format(strategy, n_bins)
        self.n_bins = n_bins
        self.strategy = strategy
        if strategy == "uniform":
            self.tf = UniformDiscretizer(n_bins)
        else:
            self.tf = QuantileDiscretizer(n_bins)

    def fit(self, X):
        # DP-PORT: upstream's Discretizer.fit ends in a bare `raise` (leftover
        # debugging). It is dead code there because the search only ever calls
        # fit_transform/transform, but materializing the chosen pipeline on the
        # clean hold-out needs a real fit, so it delegates to the sub-estimator.
        self.tf.fit(X)

    def transform(self, X):
        return self.tf.transform(X)

    def fit_transform(self, X):
        return self.tf.fit_transform(X)


class ZSOutlierDetector(object):
    """Values out of nstd x std are considered as outliers."""

    def __init__(self, nstd=3):
        self.nstd = nstd

    def fit(self, X):
        mean = X.mean(axis=0)
        std = X.std(axis=0)
        cut_off = std * self.nstd
        self.lower = (mean - cut_off).reshape(1, -1)
        self.upper = (mean + cut_off).reshape(1, -1)

    def detect(self, X):
        great = X > self.upper
        low = X < self.lower
        return np.logical_or(great, low)


class IQROutlierDetector(object):
    """Interquartile Range method."""

    def __init__(self, k=1.5):
        self.k = k

    def fit(self, X):
        q25 = np.nanpercentile(X, 25, axis=0)
        q75 = np.nanpercentile(X, 75, axis=0)
        iqr = q75 - q25
        cut_off = iqr * self.k
        self.lower = (q25 - cut_off).reshape(1, -1)
        self.upper = (q75 + cut_off).reshape(1, -1)

    def detect(self, X):
        great = X > self.upper
        low = X < self.lower
        return np.logical_or(great, low)


class MADOutlierDetector(object):
    """Median absolute deviation."""

    def __init__(self, nmad=2.5):
        self.nmad = nmad

    def fit(self, X):
        median = np.median(X, axis=0, keepdims=True)
        mad = np.median(np.abs(X - median), axis=0, keepdims=True)
        self.lower = (median - self.nmad * mad).reshape(1, -1)
        self.upper = (median + self.nmad * mad).reshape(1, -1)

    def detect(self, X):
        great = X > self.upper
        low = X < self.lower
        return np.logical_or(great, low)


class OutlierCleaner(object):
    """Detect outliers and repair them with mean imputation (TFs/outlier_cleaner.py).

    Available methods: 'ZS[_nstd]', 'IQR[_k]', 'MAD[_nmad]'.
    """

    def __init__(self, method):
        self.method = method
        if self.method == "ZS":
            self.detector = ZSOutlierDetector()
        elif self.method == "IQR":
            self.detector = IQROutlierDetector()
        elif self.method == "MAD":
            self.detector = MADOutlierDetector()
        elif "ZS" in self.method:
            self.detector = ZSOutlierDetector(nstd=float(self.method.split("_")[1]))
        elif "IQR" in self.method:
            self.detector = IQROutlierDetector(k=float(self.method.split("_")[1]))
        elif "MAD" in self.method:
            self.detector = MADOutlierDetector(nmad=float(self.method.split("_")[1]))
        else:
            raise Exception("Invalid outlier method: {}".format(method))

        self.repairer = SimpleImputer(keep_empty_features=True)

    def fit(self, X):
        self.detector.fit(X)
        indicator = self.detector.detect(X)
        X_clean = copy.deepcopy(X)
        X_clean[indicator] = np.nan
        self.repairer.fit(X_clean)

    def transform(self, X):
        indicator = self.detector.detect(X)
        X_trans = copy.deepcopy(X)
        X_trans[indicator] = np.nan
        return self.repairer.transform(X_trans)

    def fit_transform(self, X):
        self.fit(X)
        return self.transform(X)


class EMImputer(object):
    """Expectation-maximization imputation.

    DP-PORT: upstream imports ``impyute.imputation.cs.em``; impyute pins
    numpy<1.20 and no longer installs on this stack, so the same algorithm is
    reimplemented here with no new dependency. It is not part of the default
    search space (upstream's prep_space.py does not list it either), it is kept
    so a user can add NumMVImputer("EM") to the space.
    """

    def __init__(self, loops=50, eps=1e-6):
        self.loops = loops
        self.eps = eps

    def fit(self, X):
        self.mean_ = np.nanmean(np.asarray(X, dtype=float), axis=0)
        self.mean_ = np.nan_to_num(self.mean_)

    def transform(self, X):
        X = np.array(X, dtype=float, copy=True)
        nan_xy = np.argwhere(np.isnan(X))
        for x_i, y_i in nan_xy:
            col = X[:, y_i]
            observed = col[~np.isnan(col)]
            if observed.size == 0:
                X[x_i, y_i] = self.mean_[y_i]
                continue
            mu, std = observed.mean(), observed.std()
            previous, i = 1.0, 1
            while i < self.loops:
                # Expectation
                mu, std = observed.mean(), observed.std()
                # Maximization
                X[x_i, y_i] = np.random.normal(loc=mu, scale=std if std > 0 else 1e-9)
                delta = np.abs(X[x_i, y_i] - previous) / (previous if previous else 1.0)
                if delta < self.eps and i > 5:
                    break
                previous = X[x_i, y_i]
                i += 1
        return X

    def fit_transform(self, X):
        self.fit(X)
        return self.transform(X)


class ModeImputer(object):
    """Most-frequent imputation over numerical and categorical columns jointly."""

    def __init__(self):
        self.num_imputer = SimpleImputer(strategy="most_frequent", keep_empty_features=True)
        self.cat_imputer = SimpleImputer(strategy="most_frequent", keep_empty_features=True)

    def fit(self, X_num, X_cat):
        if X_num.shape[1] > 0:
            self.num_imputer.fit(X_num)
        else:
            self.num_imputer = None
        if X_cat.shape[1] > 0:
            self.cat_imputer.fit(X_cat)
        else:
            self.cat_imputer = None

    def transform(self, X_num, X_cat):
        X_num_trans = None
        X_cat_trans = None
        if self.num_imputer is not None:
            X_num_trans = self.num_imputer.transform(X_num)
        if self.cat_imputer is not None:
            X_cat_trans = self.cat_imputer.transform(X_cat)
        return X_num_trans, X_cat_trans

    def fit_transform(self, X_num, X_cat):
        self.fit(X_num, X_cat)
        return self.transform(X_num, X_cat)


class NumMVImputer(object):
    """Impute missing values on numerical columns (TFs/mv_imputer.py).

    Available methods: 'mean', 'median', 'EM', 'KNN', 'MICE', 'DT'.
    """

    def __init__(self, method="mean"):
        self.input_type = "numerical"
        self.method = method
        if self.method == "mean":
            self.tf = SimpleImputer(strategy="mean", keep_empty_features=True)
        elif self.method == "median":
            self.tf = SimpleImputer(strategy="median", keep_empty_features=True)
        elif self.method == "EM":
            self.tf = EMImputer()
        elif self.method == "KNN":
            self.tf = KNNImputer(n_neighbors=5, keep_empty_features=True)
        elif self.method == "MICE":
            self.tf = IterativeImputer(random_state=0, skip_complete=True, keep_empty_features=True)
        elif self.method == "DT":
            self.tf = IterativeImputer(
                DecisionTreeRegressor(max_features="sqrt", random_state=0),
                random_state=0,
                skip_complete=True,
                keep_empty_features=True,
            )
        else:
            raise Exception("Invalid imputation method: {}".format(method))

    def fit(self, X):
        self.tf.fit(X)

    def transform(self, X):
        return self.tf.transform(X)

    def fit_transform(self, X):
        self.fit(X)
        return self.transform(X)


class CatMVImputer(object):
    """Impute missing values on categorical columns.

    Available methods: 'dummy' (adds a new 'dummy_category' level).
    """

    def __init__(self, method="dummy"):
        self.input_type = "categorical"
        self.method = method
        if self.method == "dummy":
            self.tf = SimpleImputer(
                strategy="constant", fill_value="dummy_category", keep_empty_features=True
            )
        else:
            raise Exception("Invalid imputation method: {}".format(method))

    def fit(self, X):
        self.tf.fit(X)

    def transform(self, X):
        return self.tf.transform(X)

    def fit_transform(self, X):
        self.fit(X)
        return self.transform(X)


class NumCatMVImputer(object):
    """Impute missing values on numerical and categorical columns.

    Available methods: 'mode' (most frequent value).
    """

    def __init__(self, method="mode"):
        self.input_type = "mixed"
        self.method = method
        if self.method == "mode":
            self.tf = ModeImputer()
        else:
            raise Exception("Invalid imputation method: {}".format(method))

    def fit(self, X_num, X_cat):
        self.tf.fit(X_num, X_cat)

    def transform(self, X_num, X_cat):
        return self.tf.transform(X_num, X_cat)

    def fit_transform(self, X_num, X_cat):
        self.fit(X_num, X_cat)
        return self.transform(X_num, X_cat)


class NumMVIdentity(object):
    """Identity imputer for numerical columns (used when there is no NaN at all)."""

    def __init__(self):
        self.method = "num_mv_identity"
        self.input_type = "numerical"

    def fit(self, X):
        pass

    def transform(self, X):
        return X

    def fit_transform(self, X):
        return X


class CatMVIdentity(object):
    """Identity imputer for categorical columns (used when there is no NaN at all)."""

    def __init__(self):
        self.method = "cat_mv_identity"
        self.input_type = "categorical"

    def fit(self, X):
        pass

    def transform(self, X):
        return X

    def fit_transform(self, X):
        return X


# ===========================================================================
# Preprocessing search space — upstream prep_space.py
# ===========================================================================

def build_prep_space() -> List[Dict]:
    """The four transformation types of Table 1, with the operators of prep_space.py.

    Rebuilt on every call: the operators are stateful (they are fitted), so two
    concurrent pipelines must not share instances.
    """
    return [
        {
            "name": "missing_value_imputation",
            "num_tf_options": [
                NumMVImputer("mean"),
                NumMVImputer("median"),
                NumMVImputer("DT"),
                NumMVImputer("MICE"),
                NumCatMVImputer("mode"),
            ],
            "cat_tf_options": [
                NumCatMVImputer("mode"),
                CatMVImputer("dummy"),
            ],
            "default": [NumMVImputer("mean"), NumCatMVImputer("mode")],
            "init": [(NumMVImputer("mean"), 0.5), (NumCatMVImputer("mode"), 0.5)],
        },
        {
            "name": "normalization",
            "tf_options": [
                Normalizer("ZS"),
                Normalizer("MM"),
                Normalizer("MA"),
                Normalizer("RS"),
            ],
            "default": Normalizer("ZS"),
            "init": (Normalizer("ZS"), 0.5),
        },
        {
            "name": "cleaning_outliers",
            "tf_options": [
                OutlierCleaner("ZS_4"),
                OutlierCleaner("ZS_3"),
                OutlierCleaner("ZS_2"),
                OutlierCleaner("MAD_3"),
                OutlierCleaner("MAD_2.5"),
                OutlierCleaner("MAD_2"),
                OutlierCleaner("IQR_2"),
                OutlierCleaner("IQR_1.5"),
                OutlierCleaner("IQR_1"),
                Identity(),
            ],
            "default": Identity(),
            "init": (Identity(), 0.5),
        },
        {
            "name": "discretization",
            "tf_options": [
                Discretizer(n_bins=5, strategy="uniform"),
                Discretizer(n_bins=10, strategy="uniform"),
                Discretizer(n_bins=20, strategy="uniform"),
                Discretizer(n_bins=5, strategy="quantile"),
                Discretizer(n_bins=10, strategy="quantile"),
                Discretizer(n_bins=20, strategy="quantile"),
                Identity(),
            ],
            "default": Identity(),
            "init": (Identity(), 0.5),
        },
    ]


# ===========================================================================
# End model — upstream model.py
# ===========================================================================

class LogisticRegression(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(LogisticRegression, self).__init__()
        self.linear = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        return self.linear(x)


class TwoLayerNet(nn.Module):
    """Two-layer neural network with 100 hidden ReLU neurons (Section 5.3).

    DP-PORT: main.py upstream exposes --model two but model.py only ships
    LogisticRegression, so the network the paper describes in its sensitivity
    analysis is defined here.
    """

    def __init__(self, input_dim, output_dim, hidden_dim=100):
        super(TwoLayerNet, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x):
        return self.net(x)


# ===========================================================================
# Differentiable pipeline — upstream pipeline/diffprep_fix_pipeline.py
# ===========================================================================

def is_contain_mv(df: pd.DataFrame) -> bool:
    return df.isnull().values.sum() > 0


class Transformer(nn.Module):
    """One transformer in the data preparation pipeline.

    Holds the beta parameters of Equation 1/7: one row of logits per feature,
    one column per operator of this transformation type.
    """

    def __init__(self, name, tf_options, in_features, init_tf=(None, None),
                 diff_method="num_diff", beta=None):
        super(Transformer, self).__init__()
        self.name = name
        # DP-PORT: upstream's Fix pipeline shares the operator instances between
        # transformers while Flex deep-copies them. The operators are stateful
        # (fit stores means, quantiles, ...), so both variants deep-copy here.
        self.tf_options = copy.deepcopy(tf_options)
        self.tf_methods = [tf.method for tf in self.tf_options]
        self.num_tf_options = len(self.tf_options)
        self.diff_method = diff_method
        self.in_features = in_features
        self.out_features = in_features  # the TF operators do not change the dim
        self.init_tf_option, self.init_p = init_tf
        self.init_parameters(beta)

    def init_parameters(self, beta=None):
        """Initialize the beta logits.

        `beta` is only passed by the Flex pipeline, which shares one beta matrix
        per transformation type across all the positions of the permutation.
        """
        if beta is not None:
            self.tf_prob_logits = beta
        elif self.init_tf_option is None:
            self.tf_prob_logits = nn.Parameter(
                torch.randn(self.out_features, self.num_tf_options), requires_grad=True
            )
        else:
            # The default operator gets init_p, the rest share the remainder.
            init_tf_probs = torch.ones(self.out_features, self.num_tf_options) * (
                (1 - self.init_p) / (self.num_tf_options - 1)
            )
            init_idx = self.tf_methods.index(self.init_tf_option.method)
            init_tf_probs[:, init_idx] = self.init_p
            self.tf_prob_logits = nn.Parameter(probs_to_logits(init_tf_probs), requires_grad=True)

        self.tf_prob_sample = None  # shape (num_features, num_tfs)
        self.is_sampled = False

    def numerical_diff(self, X, eps=1e-6, alpha=None):
        """Equation 13: the gradient of a black-box TF operator by central differences."""
        X = X.detach().numpy()
        X_pos = X + eps
        X_neg = X - eps

        X_grads = []
        for tf in self.tf_options:
            f1 = tf.transform(X_pos)
            f2 = tf.transform(X_neg)
            grad = (f1 - f2) / (2 * eps)
            X_grads.append(np.expand_dims(grad, axis=-1))

        X_grads = np.concatenate(X_grads, axis=2)
        X_sample_grad = (X_grads * self.tf_prob_sample.detach().numpy()).sum(axis=2)
        if alpha is not None:
            X_sample_grad = X_sample_grad * alpha.detach().numpy()
            X_sample_grad = np.clip(X_sample_grad, -10, 10)
        return torch.Tensor(X_sample_grad)

    def forward(self, X, is_fit, X_type, max_only=False, require_grad=True, alpha=None):
        X_trans = []
        for tf in self.tf_options:
            if is_fit:
                X_t = tf.fit_transform(X.detach().numpy())
            else:
                X_t = tf.transform(X.detach().numpy())
            X_trans.append(torch.Tensor(X_t).unsqueeze(-1))

        # shape (num_examples, num_features, num_tfs)
        X_trans = torch.cat(X_trans, dim=2)
        return self.select_X_sample(X, X_trans, max_only, require_grad=require_grad, alpha=alpha)

    def select_X_sample(self, X, X_trans, max_only, require_grad=True, alpha=None):
        """Equation 2/17 plus the straight-through gradient of Equation 14."""
        if max_only:
            tf_prob_sample = self.sample_with_max_probs()
        else:
            tf_prob_sample = self.tf_prob_sample

        X_trans_sample = (X_trans * tf_prob_sample.unsqueeze(0)).sum(axis=2)
        if alpha is not None:
            X_trans_sample = X_trans_sample * alpha.unsqueeze(0)

        if not require_grad:
            return X_trans_sample

        if self.diff_method == "num_diff":
            X_grad = self.numerical_diff(X, alpha=alpha)
        else:
            raise Exception("invalid diff method {}".format(self.diff_method))

        # Equation 14: same forward value, but the black-box operators now carry
        # the numerical gradient through the autodiff engine.
        return X_trans_sample + (X_grad * X - (X_grad * X).detach())

    def categorical_max(self, logits):
        max_idx = torch.argmax(logits, dim=1)
        max_sample = torch.zeros_like(logits)
        max_sample[np.arange(max_sample.shape[0]), max_idx] = 1
        return max_sample

    def categorical_sample(self, logits, temperature, use_sample=True):
        if not use_sample:
            # Equation 7: plain softmax relaxation.
            return logits_to_probs(logits, is_binary=False)
        # Gumbel-softmax straight-through sample.
        samples = torch.distributions.RelaxedOneHotCategorical(temperature, logits=logits).rsample()
        indicator = torch.max(samples, dim=-1, keepdim=True)[1]
        one_h = torch.zeros_like(samples).scatter_(-1, indicator, 1.0)
        return samples + (one_h - samples.detach())

    def sample(self, temperature, use_sample=True):
        self.tf_prob_sample = self.categorical_sample(self.tf_prob_logits, temperature, use_sample)
        self.is_sampled = True

    def sample_with_max_probs(self):
        return self.categorical_max(self.tf_prob_logits)


class FirstTransformer(Transformer):
    """The first transformer of the pipeline: missing value imputation + one-hot encoding.

    Its output is cached per split, because imputation does not depend on the
    earlier steps and is by far the most expensive operator set (MICE, DT).
    """

    def __init__(self, num_tf_options, cat_tf_options, init_num_tf=(None, None),
                 init_cat_tf=(None, None)):
        super(Transformer, self).__init__()
        self.name = "missing_value_imputation"
        self.num_tf_options = copy.deepcopy(num_tf_options)
        self.cat_tf_options = copy.deepcopy(cat_tf_options)
        self.num_tf_methods = [tf.method for tf in self.num_tf_options]
        self.cat_tf_methods = [tf.method for tf in self.cat_tf_options]
        self.num_num_tf_options = len(self.num_tf_options)
        self.num_cat_tf_options = len(self.cat_tf_options)
        self.init_num_tf_option, self.init_num_p = init_num_tf
        self.init_cat_tf_option, self.init_cat_p = init_cat_tf
        self.cache = {}

    def fit_transform(self, X: pd.DataFrame):
        X_num = X.select_dtypes(include="number")
        X_cat = X.select_dtypes(exclude="number")
        self.num_columns = X_num.columns
        self.cat_columns = X_cat.columns

        X_num_trans = []
        X_cat_trans = []
        self.contain_num = X_num.shape[1] > 0
        self.contain_cat = X_cat.shape[1] > 0

        self.out_num_features = 0
        self.out_cat_features = 0
        self.cache["train"] = {"X_num_trans": None, "X_cat_trans": None}

        if self.contain_num:
            for tf in self.num_tf_options:
                assert tf.input_type in ["numerical", "mixed"]
                if tf.input_type == "numerical":
                    X_num_trans.append(tf.fit_transform(X_num.values))
                else:
                    X_num_t, _ = tf.fit_transform(X_num.values, X_cat.values)
                    X_num_trans.append(X_num_t)

            X_num_trans = torch.Tensor(np.array(X_num_trans, dtype=float)).permute(1, 2, 0)
            self.cache["train"]["X_num_trans"] = X_num_trans
            self.out_num_features = X_num_trans.shape[1]

        if self.contain_cat:
            for tf in self.cat_tf_options:
                assert tf.input_type in ["categorical", "mixed"]
                if tf.input_type == "categorical":
                    X_cat_trans.append(tf.fit_transform(X_cat.values))
                else:
                    _, X_cat_t = tf.fit_transform(X_num.values, X_cat.values)
                    X_cat_trans.append(X_cat_t)

            # DP-PORT: sparse= was renamed sparse_output= in scikit-learn 1.2.
            self.one_hot_encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
            self.one_hot_encoder.fit(np.vstack(X_cat_trans))
            X_cat_trans_enc = [self.one_hot_encoder.transform(x) for x in X_cat_trans]

            X_cat_trans = torch.Tensor(np.array(X_cat_trans_enc, dtype=float)).permute(1, 2, 0)
            self.cache["train"]["X_cat_trans"] = X_cat_trans
            self.out_cat_features = X_cat_trans.shape[1]

        self.out_features = self.out_num_features + self.out_cat_features
        self.feature_names = self.get_feature_names()
        self.init_parameters()

    def get_feature_names(self):
        num_feature_names = [c for c in self.num_columns]
        if self.contain_cat:
            cat_feature_names = list(self.one_hot_encoder.get_feature_names_out(self.cat_columns))
        else:
            cat_feature_names = []
        return num_feature_names + cat_feature_names

    def init_parameters(self, beta=None):
        self.num_tf_prob_logits = None
        self.cat_tf_prob_logits = None
        self.num_tf_prob_sample = None
        self.cat_tf_prob_sample = None

        if self.contain_num:
            if self.init_num_tf_option is None:
                num_tf_prob_logits = torch.randn(self.out_num_features, self.num_num_tf_options)
            else:
                idx = self.num_tf_methods.index(self.init_num_tf_option.method)
                probs = torch.ones(self.out_num_features, self.num_num_tf_options) * (
                    (1 - self.init_num_p) / (self.num_num_tf_options - 1)
                )
                probs[:, idx] = self.init_num_p
                num_tf_prob_logits = probs_to_logits(probs)
            self.num_tf_prob_logits = nn.Parameter(num_tf_prob_logits, requires_grad=True)

        if self.contain_cat:
            if self.init_cat_tf_option is None:
                cat_tf_prob_logits = torch.randn(self.out_cat_features, self.num_cat_tf_options)
            else:
                idx = self.cat_tf_methods.index(self.init_cat_tf_option.method)
                probs = torch.ones(self.out_cat_features, self.num_cat_tf_options) * (
                    (1 - self.init_cat_p) / (self.num_cat_tf_options - 1)
                )
                probs[:, idx] = self.init_cat_p
                cat_tf_prob_logits = probs_to_logits(probs)
            self.cat_tf_prob_logits = nn.Parameter(cat_tf_prob_logits, requires_grad=True)

        self.is_sampled = False

    def transform(self, X: pd.DataFrame):
        X_num = X[self.num_columns]
        X_cat = X[self.cat_columns]
        X_num_trans = []
        X_cat_trans = []

        if self.contain_num:
            for tf in self.num_tf_options:
                if tf.input_type == "numerical":
                    X_num_trans.append(tf.transform(X_num.values))
                else:
                    X_num_t, _ = tf.transform(X_num.values, X_cat.values)
                    X_num_trans.append(X_num_t)
            X_num_trans = torch.Tensor(np.array(X_num_trans, dtype=float)).permute(1, 2, 0)

        if self.contain_cat:
            for tf in self.cat_tf_options:
                if tf.input_type == "categorical":
                    X_cat_t = tf.transform(X_cat.values)
                else:
                    _, X_cat_t = tf.transform(X_num.values, X_cat.values)
                X_cat_trans.append(self.one_hot_encoder.transform(X_cat_t))
            X_cat_trans = torch.Tensor(np.array(X_cat_trans, dtype=float)).permute(1, 2, 0)

        return X_num_trans, X_cat_trans

    def pre_cache(self, X: pd.DataFrame, X_type: str):
        X_num_trans, X_cat_trans = self.transform(X)
        self.cache[X_type] = {"X_num_trans": X_num_trans, "X_cat_trans": X_cat_trans}

    def forward(self, X, is_fit, X_type, max_only=False, require_grad=True, alpha=None):
        indices = X.index
        X_num_trans = self.cache[X_type]["X_num_trans"][indices] if self.contain_num else None
        X_cat_trans = self.cache[X_type]["X_cat_trans"][indices] if self.contain_cat else None
        return self.select_X_sample(X_num_trans, X_cat_trans, max_only)

    def select_X_sample(self, X_num_trans, X_cat_trans, max_only, require_grad=True, alpha=None):
        if max_only:
            num_tf_prob_sample, cat_tf_prob_sample = self.sample_with_max_probs()
        else:
            num_tf_prob_sample = self.num_tf_prob_sample
            cat_tf_prob_sample = self.cat_tf_prob_sample

        X_num_trans_sample = None
        X_cat_trans_sample = None
        if num_tf_prob_sample is not None:
            X_num_trans_sample = (X_num_trans * num_tf_prob_sample.unsqueeze(0)).sum(axis=2)
        if cat_tf_prob_sample is not None:
            X_cat_trans_sample = (X_cat_trans * cat_tf_prob_sample.unsqueeze(0)).sum(axis=2)

        return self.concat_num_cat(X_num_trans_sample, X_cat_trans_sample)

    def sample_with_max_probs(self):
        num_tf_prob_sample = None
        cat_tf_prob_sample = None
        if self.num_tf_prob_logits is not None:
            num_tf_prob_sample = self.categorical_max(self.num_tf_prob_logits)
        if self.cat_tf_prob_logits is not None:
            cat_tf_prob_sample = self.categorical_max(self.cat_tf_prob_logits)
        return num_tf_prob_sample, cat_tf_prob_sample

    def sample(self, temperature=0.1, use_sample=True):
        if self.num_tf_prob_logits is not None:
            self.num_tf_prob_sample = self.categorical_sample(
                self.num_tf_prob_logits, temperature, use_sample
            )
        if self.cat_tf_prob_logits is not None:
            self.cat_tf_prob_sample = self.categorical_sample(
                self.cat_tf_prob_logits, temperature, use_sample
            )
        self.is_sampled = True

    @staticmethod
    def concat_num_cat(X_num, X_cat):
        if X_num is None:
            return X_cat
        if X_cat is None:
            return X_num
        return torch.cat((X_num, X_cat), dim=1)


class DiffPrepFixPipeline(nn.Module):
    """DiffPrep-Fix (Section 3): fixed transformation order, learned operators."""

    def __init__(self, prep_space, temperature=0.1, use_sample=False,
                 diff_method="num_diff", init_method="default"):
        super(DiffPrepFixPipeline, self).__init__()
        self.prep_space = prep_space
        self.temperature = temperature
        self.use_sample = use_sample
        self.diff_method = diff_method
        self.is_fitted = False
        self.init_method = init_method

    def init_parameters(self, X_train, X_val, X_test):
        pipeline = []
        self.contain_mv = (
            is_contain_mv(X_train) or is_contain_mv(X_val) or is_contain_mv(X_test)
        )

        if self.contain_mv:
            first_tf_dict = self.prep_space[0]
            if self.init_method == "default":
                init_num_tf, init_cat_tf = first_tf_dict["init"][0], first_tf_dict["init"][1]
            elif self.init_method == "random":
                init_num_tf, init_cat_tf = (None, None), (None, None)
            else:
                raise Exception("Wrong init method")
            first_transformer = FirstTransformer(
                first_tf_dict["num_tf_options"], first_tf_dict["cat_tf_options"],
                init_num_tf=init_num_tf, init_cat_tf=init_cat_tf,
            )
        else:
            first_transformer = FirstTransformer([NumMVIdentity()], [CatMVIdentity()])

        first_transformer.fit_transform(X_train)
        first_transformer.pre_cache(X_val, "val")
        first_transformer.pre_cache(X_test, "test")
        pipeline.append(first_transformer)

        in_features = first_transformer.out_features
        for tf_dict in self.prep_space[1:]:
            if self.init_method == "default":
                init_tf = tf_dict["init"]
            elif self.init_method == "random":
                init_tf = (None, None)
            else:
                raise Exception("Wrong init method")
            pipeline.append(
                Transformer(tf_dict["name"], tf_dict["tf_options"], in_features,
                            init_tf=init_tf, diff_method=self.diff_method)
            )

        self.pipeline = nn.ModuleList(pipeline)
        self.out_features = in_features

    def forward(self, X, is_fit, X_type, resample=False, max_only=False, require_grad=True):
        X_output = copy.deepcopy(X)
        for transformer in self.pipeline:
            if resample or not transformer.is_sampled:
                transformer.sample(temperature=self.temperature, use_sample=self.use_sample)
            X_output = transformer(X_output, is_fit, X_type, max_only=max_only,
                                   require_grad=require_grad)
        return X_output

    def fit(self, X):
        self.is_fitted = True
        return self.forward(X, is_fit=True, X_type="train", resample=True)

    def transform(self, X, X_type, max_only=False, resample=False, require_grad=True):
        if not self.is_fitted:
            raise Exception("transformer is not fitted")
        return self.forward(X, is_fit=False, X_type=X_type, resample=resample,
                            max_only=max_only, require_grad=require_grad)

    def get_final_dataset(self, X, X_type):
        return self.forward(X, is_fit=False, X_type=X_type, resample=False, max_only=True)


# ===========================================================================
# Differentiable pipeline — upstream pipeline/diffprep_flex_pipeline.py
# ===========================================================================

def sinkhorn(X, eps=1e-6, max_iter=500):
    """Sinkhorn normalization (Section 4.2): any non-negative square matrix
    becomes doubly stochastic by alternating row and column normalization."""
    X = torch.exp(X)
    for _ in range(max_iter):
        col_sum = X.sum(axis=1, keepdim=True)
        X = X / col_sum
        row_sum = X.sum(axis=2, keepdim=True)
        X = X / row_sum
        if ((col_sum - 1).abs() < eps).all() and ((row_sum - 1).abs() < eps).all():
            return X
    return X


class DiffPrepFlexPipeline(nn.Module):
    """DiffPrep-Flex (Section 4): the order of the transformation types is learned too.

    One alpha doubly-stochastic matrix per feature (Equation 16), shared beta
    matrices per transformation type: position i of the prototype applies the
    alpha-weighted mixture of every transformation type, Equation 17.
    """

    def __init__(self, prep_space, temperature=0.1, use_sample=False,
                 diff_method="num_diff", init_method="default"):
        super(DiffPrepFlexPipeline, self).__init__()
        self.prep_space = prep_space
        self.temperature = temperature
        self.use_sample = use_sample
        self.diff_method = diff_method
        self.is_fitted = False
        self.init_method = init_method
        self.n_tf_types = len(prep_space)

    def init_parameters(self, X_train, X_val, X_test):
        pipeline = []
        self.contain_mv = (
            is_contain_mv(X_train) or is_contain_mv(X_val) or is_contain_mv(X_test)
        )

        if self.contain_mv:
            first_tf_dict = self.prep_space[0]
            if self.init_method == "default":
                init_num_tf, init_cat_tf = first_tf_dict["init"][0], first_tf_dict["init"][1]
            elif self.init_method == "random":
                init_num_tf, init_cat_tf = (None, None), (None, None)
            else:
                raise Exception("Wrong init method")
            first_transformer = FirstTransformer(
                first_tf_dict["num_tf_options"], first_tf_dict["cat_tf_options"],
                init_num_tf=init_num_tf, init_cat_tf=init_cat_tf,
            )
        else:
            first_transformer = FirstTransformer([NumMVIdentity()], [CatMVIdentity()])

        first_transformer.fit_transform(X_train)
        first_transformer.pre_cache(X_val, "val")
        first_transformer.pre_cache(X_test, "test")
        pipeline.append(first_transformer)

        in_features = first_transformer.out_features

        # One beta matrix per transformation type, reused at every position of
        # the prototype — that is what makes the order, not the operators, the
        # thing alpha decides.
        beta_list = []
        for tf_dict in self.prep_space[1:]:
            if self.init_method == "default":
                init_tf_option, init_p = tf_dict["init"]
            elif self.init_method == "random":
                init_tf_option, init_p = (None, None)
            else:
                raise Exception("Wrong init method")

            tf_methods = [tf.method for tf in tf_dict["tf_options"]]
            num_tf_options = len(tf_dict["tf_options"])
            if init_tf_option is None:
                tf_prob_logits = torch.randn(in_features, num_tf_options)
            else:
                init_tf_probs = torch.ones(in_features, num_tf_options) * (
                    (1 - init_p) / (num_tf_options - 1)
                )
                init_tf_probs[:, tf_methods.index(init_tf_option.method)] = init_p
                tf_prob_logits = probs_to_logits(init_tf_probs)
            beta_list.append(nn.Parameter(tf_prob_logits, requires_grad=True))

        self.betas = nn.ParameterList(beta_list)

        for _ in range(self.n_tf_types - 1):
            for idx, tf_dict in enumerate(self.prep_space[1:]):
                if self.init_method == "default":
                    init_tf = tf_dict["init"]
                else:
                    init_tf = (None, None)
                pipeline.append(
                    Transformer(tf_dict["name"], tf_dict["tf_options"], in_features,
                                init_tf=init_tf, diff_method=self.diff_method,
                                beta=beta_list[idx])
                )

        self.pipeline = nn.ModuleList(pipeline)
        self.out_features = in_features

        # Equation 16, relaxed: one (n_types-1) x (n_types-1) matrix per feature.
        self.alpha = nn.Parameter(
            torch.randn(in_features, self.n_tf_types - 1, self.n_tf_types - 1),
            requires_grad=True,
        )
        self.alpha_probs = sinkhorn(self.alpha)

    def forward(self, X, is_fit, X_type, resample=False, max_only=False, require_grad=True):
        X_output = copy.deepcopy(X)

        transformer = self.pipeline[0]
        if resample or not transformer.is_sampled:
            transformer.sample(temperature=self.temperature, use_sample=self.use_sample)
        X_output = transformer(X_output, is_fit, X_type, max_only=max_only)

        self.alpha_probs = sinkhorn(self.alpha)

        cur_idx = 1
        for i in range(self.n_tf_types - 1):
            X_output_i = torch.zeros_like(X_output)
            for j in range(self.n_tf_types - 1):
                transformer = self.pipeline[cur_idx]
                if resample or not transformer.is_sampled:
                    transformer.sample(temperature=self.temperature, use_sample=self.use_sample)
                X_output_i = X_output_i + transformer(
                    X_output, is_fit, X_type, max_only=max_only,
                    require_grad=require_grad, alpha=self.alpha_probs[:, i, j],
                )
                cur_idx += 1
            X_output = X_output_i
        return X_output

    def fit(self, X):
        self.is_fitted = True
        return self.forward(X, is_fit=True, X_type="train", resample=True)

    def transform(self, X, X_type, max_only=False, resample=False, require_grad=True):
        if not self.is_fitted:
            raise Exception("transformer is not fitted")
        return self.forward(X, is_fit=False, X_type=X_type, resample=resample,
                            max_only=max_only, require_grad=require_grad)

    def get_final_dataset(self, X, X_type):
        return self.forward(X, is_fit=False, X_type=X_type, resample=False, max_only=True)


# ===========================================================================
# Bi-level optimization — upstream trainer/diffprep_trainer.py
# ===========================================================================

def _concat(xs):
    return torch.cat([x.view(-1) for x in xs])


def make_batch(X, y, batch_size, shuffle=False):
    indices = np.arange(X.shape[0])
    if shuffle:
        np.random.shuffle(indices)
    start_idx = 0
    while True:
        batch_idx = indices[start_idx : start_idx + batch_size]
        yield X.iloc[batch_idx], y[batch_idx]
        start_idx += batch_size
        if start_idx >= X.shape[0]:
            break


def take_random_batch(X, y, batch_size):
    indices = np.random.permutation(X.shape[0])
    batch_idx = indices[:batch_size]
    return X.iloc[batch_idx], y[batch_idx]


class DiffPrepSGD(object):
    """Algorithm 2 / Algorithm 3: alternate one gradient step on the pipeline
    parameters (validation loss) with one on the model parameters (training loss).
    """

    def __init__(self, prep_pipeline, model, loss_fn, model_optimizer,
                 prep_pipeline_optimizer, model_scheduler, prep_pipeline_scheduler,
                 params):
        self.prep_pipeline = prep_pipeline
        self.model = model
        self.loss_fn = loss_fn
        self.model_optimizer = model_optimizer
        self.prep_pipeline_optimizer = prep_pipeline_optimizer
        self.model_scheduler = model_scheduler
        self.prep_pipeline_scheduler = prep_pipeline_scheduler
        self.params = params
        self.device = self.params["device"]

    def forward_propagate(self, X, y, X_type, require_transform_grad=False,
                          require_model_grad=False, max_only=False):
        with torch.set_grad_enabled(require_transform_grad):
            X_trans = self.prep_pipeline.transform(
                X, X_type=X_type, max_only=max_only, resample=False,
                require_grad=require_transform_grad,
            )

        if X_type == "train":
            self.model.train()
        else:
            self.model.eval()

        with torch.set_grad_enabled(require_model_grad or require_transform_grad):
            X_trans = X_trans.to(self.device)
            output = self.model(X_trans)
        y = y.to(self.device)
        loss = self.loss_fn(output, y)
        return output, loss

    def fit(self, X_train, y_train, X_val, y_val, X_test=None, y_test=None, verbose=False):
        best_val_loss = float("inf")
        best_model = None
        best_result = None

        last_best_val_acc = float("-inf")
        patience = self.params["patience"]
        e = 0

        while e < self.params["num_epochs"]:
            self.global_step = e
            tr_loss, tr_acc = self.train(X_train, y_train, X_val, y_val)

            val_loss, val_acc = self.evaluate(X_val, y_val, X_type="val", max_only=False)

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                # The test frame is only ever read to report best_test_acc, never
                # to select: upstream evaluates it every epoch and throws the
                # value away unless the validation loss improved. Evaluating it
                # here instead keeps best_test_acc identical while skipping one
                # full-test forward pass on every non-improving epoch.
                _, test_acc = self.evaluate(X_test, y_test, X_type="test", max_only=False)
                best_result = {
                    "best_epoch": e,
                    "best_val_loss": val_loss,
                    "best_tr_acc": tr_acc,
                    "best_val_acc": val_acc,
                    "best_test_acc": test_acc,
                }
                best_model = {
                    "prep_pipeline": copy.deepcopy(self.prep_pipeline.state_dict()),
                    "end_model": copy.deepcopy(self.model.state_dict()),
                }

            if self.model_scheduler is not None:
                self.model_scheduler.step(val_loss)
            if self.prep_pipeline_scheduler is not None:
                self.prep_pipeline_scheduler.step(val_loss)

            # Early stopping: upstream only re-checks every 100 epochs.
            if e % 100 == 0:
                if best_result["best_val_acc"] - last_best_val_acc < 0.001:
                    patience = patience - 1
                else:
                    last_best_val_acc = best_result["best_val_acc"]
                    patience = self.params["patience"]

            if patience <= 0:
                break

            e += 1

            if verbose and e % 50 == 0:
                logger.debug(
                    f"    epoch {e}: tr_loss={tr_loss:.4f} val_loss={val_loss:.4f} "
                    f"val_acc={val_acc:.4f}"
                )

        best_result["n_epochs_run"] = e
        return best_result, best_model

    def train(self, X_train, y_train, X_val, y_val):
        # Line 3-7 of Algorithm 2: refit the TF operators on a mini-batch, then
        # move the pipeline parameters with the validation-loss gradient.
        X_val_batch, y_val_batch = take_random_batch(
            X_val, y_val, self.params["pipeline_update_sample_size"]
        )
        X_train_batch, y_train_batch = take_random_batch(
            X_train, y_train, self.params["pipeline_update_sample_size"]
        )

        if not self.prep_pipeline.is_fitted:
            self.prep_pipeline.fit(X_train_batch)
        self.update_prep_pipeline(X_train_batch, y_train_batch, X_val_batch, y_val_batch)
        self.prep_pipeline.fit(X_train_batch)

        # Line 8: one epoch of model updates.
        tr_correct = 0
        tr_loss = 0
        n_batches = 0
        X_train_iter = make_batch(X_train, y_train, self.params["batch_size"], shuffle=True)
        for X_train_batch, y_train_batch in X_train_iter:
            loss, correct = self.update_model(X_train_batch, y_train_batch)
            tr_correct += correct
            tr_loss += loss
            n_batches += 1

        return tr_loss / max(n_batches, 1), tr_correct / len(y_train)

    def evaluate(self, X, y, X_type, max_only=True):
        output, loss = self.forward_propagate(X, y, X_type=X_type, max_only=max_only)
        _, preds = torch.max(output, 1)
        correct = torch.sum(preds.cpu() == y)
        return loss.item(), correct.item() / len(y)

    def update_model(self, X_train, y_train):
        self.model_optimizer.zero_grad()
        output_train, loss_train = self.forward_propagate(
            X_train, y_train, X_type="train", require_model_grad=True
        )
        loss_train.backward()
        self.model_optimizer.step()
        _, preds = torch.max(output_train, 1)
        correct = torch.sum(preds.cpu() == y_train)
        return loss_train.item(), correct.item()

    def update_prep_pipeline(self, X_train, y_train, X_val, y_val):
        """Equation 10: dval/dtau minus eta2 * the second-order term."""
        self.prep_pipeline_optimizer.zero_grad()

        dval_dalpha, dval_dw = self.compute_dval(X_train, y_train, X_val, y_val)
        hessian_product = self.compute_hessian_product(X_train, y_train, dval_dw)

        for i, alpha in enumerate(self.prep_pipeline.parameters()):
            dval = dval_dalpha[i]
            dtrain = hessian_product[i]
            if dval is None:
                continue
            if dtrain is None:
                dalpha = dval
            else:
                dalpha = dval - self.model_optimizer.param_groups[0]["lr"] * dtrain

            if alpha.grad is None:
                alpha.grad = Variable(dalpha.data.clone())
            else:
                alpha.grad.data.copy_(dalpha.data.clone())

        self.prep_pipeline_optimizer.step()

    def compute_dval(self, X_train, y_train, X_val, y_val):
        """Equation 9: approximate w* by a single training step, then read
        dLval/dbeta and dLval/dw at that virtual point."""
        model_backup = copy.deepcopy(self.model.state_dict())
        self.update_model(X_train, y_train)
        self.model_optimizer.zero_grad()
        _, loss_val = self.forward_propagate(
            X_val, y_val, X_type="val", require_transform_grad=True, require_model_grad=True
        )
        loss_val.backward(retain_graph=True)
        dval_dalpha = [
            param.grad.data.clone() if param.grad is not None else None
            for param in self.prep_pipeline.parameters()
        ]
        dval_dw = [param.grad.data.clone() for param in self.model.parameters()]
        self.model.load_state_dict(model_backup)
        return dval_dalpha, dval_dw

    def compute_hessian_product(self, X_train, y_train, dval_dw):
        """Equation 11: the second-order term by central differences on w."""
        model_backup = copy.deepcopy(self.model.state_dict())
        denom = _concat(dval_dw).data.detach().norm()
        if denom == 0:
            self.model.load_state_dict(model_backup)
            return [None] * len(list(self.prep_pipeline.parameters()))
        eps = 0.001 * _concat(self.model.parameters()).data.detach().norm() / denom

        for w, dw in zip(self.model.parameters(), dval_dw):
            w.data += eps * dw
        _, loss_train = self.forward_propagate(
            X_train, y_train, X_type="train", require_transform_grad=True
        )
        grads_p = torch.autograd.grad(
            loss_train, self.prep_pipeline.parameters(), retain_graph=True, allow_unused=True
        )

        for w, dw in zip(self.model.parameters(), dval_dw):
            w.data -= 2 * eps * dw
        _, loss_train = self.forward_propagate(
            X_train, y_train, X_type="train", require_transform_grad=True
        )
        grads_n = torch.autograd.grad(
            loss_train, self.prep_pipeline.parameters(), retain_graph=True, allow_unused=True
        )

        hessian_product = [
            None if (x is None or y is None) else (x - y).div_(2 * eps.cpu())
            for x, y in zip(grads_p, grads_n)
        ]
        self.model.load_state_dict(model_backup)
        return hessian_product


# ===========================================================================
# quAIL glue: column types and data preparation for the search
# ===========================================================================

def _identify_col_types(df: pd.DataFrame) -> Dict[str, list]:
    """Split columns by the download_data.py naming convention."""
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


def _coerce_types(df: pd.DataFrame, col_types: Dict[str, list]) -> pd.DataFrame:
    """Make the dtypes match the prefixes.

    DP-PORT: upstream reads which columns are categorical from each dataset's
    info.json and casts them with ``astype(str).replace('nan', np.nan)``
    (load_df in experiment/experiment_utils.py). Here the prefixes carry that
    information, so cat_* is cast to str and num_* to numeric — without this a
    cat_* column holding integer labels would be picked up by
    ``select_dtypes(include='number')`` and normalized as if it were numeric.
    """
    out = df.copy()
    for col in col_types["numerical"]:
        out[col] = pd.to_numeric(out[col], errors="coerce")
    for col in col_types["categorical"]:
        out[col] = out[col].astype(str).replace({"nan": np.nan, "None": np.nan, "NaN": np.nan})
        out.loc[df[col].isna(), col] = np.nan
    return out


def _searchable_features(
    df: pd.DataFrame, col_types: Dict[str, list], max_categories: int = 1000
) -> Tuple[List[str], List[str], List[str]]:
    """Which feature columns take part in the search, and which pass through.

    Mirrors upstream's ``remove_all_na`` and ``remove_large_cat``: a column that
    is entirely missing cannot be imputed and a column with more than 1000
    categories would blow up the one-hot matrix. Upstream drops them; this port
    keeps them in the written CSV untouched, because the output has to stay
    schema-compatible with the poison mask.
    """
    num_cols, cat_cols, passthrough = [], [], []

    for col in col_types["numerical"]:
        if df[col].isna().all():
            passthrough.append(col)
        else:
            num_cols.append(col)

    for col in col_types["categorical"]:
        if df[col].isna().all():
            passthrough.append(col)
        elif df[col].dropna().nunique() > max_categories:
            passthrough.append(col)
        else:
            cat_cols.append(col)

    return num_cols, cat_cols, passthrough


def build_search_data(
    df_trainval: pd.DataFrame,
    df_test: pd.DataFrame,
    num_cols: List[str],
    cat_cols: List[str],
    target_col: str,
    val_size: float,
    seed: int,
) -> Tuple:
    """Build the (train, val, test) frames the bi-level search runs on.

    DP-PORT: upstream splits one file 60/20/20 (split() in experiment_utils.py).
    Here the clean hold-out has already been carved out by scripts/poison_data.py
    and lives in data_poisoned/test/, so only the train/val split is made, with
    the same ratio and stratification as the downstream quail loaders.

    It is *not* the split the end model is later validated on: this one is drawn
    once with ``split_seed``, while the quail loaders redraw theirs for every
    experiment seed. Both halves are poisoned rows either way — the outer
    objective of the search never sees clean data, whatever ``clean_val`` says
    downstream. The hold-out only feeds the reported test accuracy; it never
    steers the selection.
    """
    feature_cols = num_cols + cat_cols

    y_all = pd.concat([df_trainval[target_col], df_test[target_col]], ignore_index=True)
    classes = pd.factorize(y_all.astype(str))[1]
    class_to_idx = {c: i for i, c in enumerate(classes)}

    y_trainval = df_trainval[target_col].astype(str).map(class_to_idx).values
    y_test = df_test[target_col].astype(str).map(class_to_idx).values

    indices = np.arange(len(df_trainval))
    stratify = y_trainval
    counts = pd.Series(y_trainval).value_counts()
    if counts.min() < 2:
        stratify = None  # a singleton class cannot be stratified
    train_idx, val_idx = train_test_split(
        indices, test_size=val_size, random_state=seed, stratify=stratify
    )

    X_train = df_trainval.iloc[train_idx][feature_cols].reset_index(drop=True)
    X_val = df_trainval.iloc[val_idx][feature_cols].reset_index(drop=True)
    X_test = df_test[feature_cols].reset_index(drop=True)

    y_train = torch.tensor(y_trainval[train_idx]).long()
    y_val = torch.tensor(y_trainval[val_idx]).long()
    y_test = torch.tensor(y_test).long()

    return (X_train, y_train, X_val, y_val, X_test, y_test), len(classes)


def min_max_normalize(X_train, X_val, X_test, num_columns):
    """Pre-normalization DiffPrep-Flex applies before the pipeline
    (min_max_normalize in experiment/experiment_utils.py).

    Returns the fitted scaler too, so the same mapping can be replayed when the
    chosen pipeline is materialized.
    """
    if not num_columns:
        return X_train, X_val, X_test, None

    scaler = MinMaxScaler()
    out = []
    scaler.fit(X_train[num_columns].values)
    for X in (X_train, X_val, X_test):
        X = X.copy()
        X[num_columns] = scaler.transform(X[num_columns].values)
        out.append(X)
    return out[0], out[1], out[2], scaler


# ===========================================================================
# Discrete materialization of the selected pipeline
# ===========================================================================

STEP_NAMES = ["normalization", "cleaning_outliers", "discretization"]


def _onehot_groups(one_hot_encoder, cat_columns) -> Dict[str, List[int]]:
    """Map each original categorical column to the indices of its one-hot features."""
    groups: Dict[str, List[int]] = {}
    offset = 0
    for col, cats in zip(cat_columns, one_hot_encoder.categories_):
        groups[col] = list(range(offset, offset + len(cats)))
        offset += len(cats)
    return groups


def extract_selected_pipeline(prep_pipeline, method: str, num_cols, cat_cols) -> Dict:
    """Read the discrete pipeline out of the converged relaxed parameters.

    For every feature this is the argmax operator of each transformation type
    (Equation 1 read back off the relaxed beta of Equation 7); for Flex it is
    also the order, taken from the alpha doubly-stochastic matrix.
    """
    first = prep_pipeline.pipeline[0]

    selection: Dict[str, Any] = {
        "num_imputer": {},
        "cat_imputer": {},
        "steps": {},   # column -> {step_name: operator method}
        "order": {},   # column -> list of step names, in the order applied
    }

    # ---- missing value imputation ----
    if first.num_tf_prob_logits is not None:
        num_probs = logits_to_probs(first.num_tf_prob_logits.detach().data).numpy()
        for i, col in enumerate(num_cols):
            selection["num_imputer"][col] = first.num_tf_methods[int(np.argmax(num_probs[i]))]
    else:
        for col in num_cols:
            selection["num_imputer"][col] = first.num_tf_methods[0]

    if first.cat_tf_prob_logits is not None and cat_cols:
        cat_probs = logits_to_probs(first.cat_tf_prob_logits.detach().data).numpy()
        # DP-PORT: upstream parameterizes the categorical imputer per *one-hot*
        # feature, so one original column can end up with several disagreeing
        # choices. Writing labels back needs one operator per original column,
        # so the probabilities of a column's one-hot features are summed and the
        # argmax of that sum wins — the column's own majority vote.
        groups = _onehot_groups(first.one_hot_encoder, first.cat_columns)
        for col in cat_cols:
            idx = groups.get(col, [])
            if not idx:
                selection["cat_imputer"][col] = first.cat_tf_methods[0]
                continue
            summed = cat_probs[idx].sum(axis=0)
            selection["cat_imputer"][col] = first.cat_tf_methods[int(np.argmax(summed))]
    else:
        for col in cat_cols:
            selection["cat_imputer"][col] = first.cat_tf_methods[0] if cat_cols else None

    if not num_cols:
        return selection

    # ---- the three later transformation types, per numerical column ----
    if method == "diffprep_fix":
        for step_idx, transformer in enumerate(prep_pipeline.pipeline[1:]):
            probs = logits_to_probs(transformer.tf_prob_logits.detach().data).numpy()
            for i, col in enumerate(num_cols):
                selection["steps"].setdefault(col, {})[transformer.name] = (
                    transformer.tf_methods[int(np.argmax(probs[i]))]
                )
        for col in num_cols:
            selection["order"][col] = list(STEP_NAMES)
        return selection

    # diffprep_flex: betas are shared per transformation type, alpha decides
    # the order — one permutation matrix per feature.
    beta_methods = []
    beta_probs = []
    for k, beta in enumerate(prep_pipeline.betas):
        tf_dict = prep_pipeline.prep_space[k + 1]
        beta_methods.append([tf.method for tf in tf_dict["tf_options"]])
        beta_probs.append(logits_to_probs(beta.detach().data).numpy())

    alpha_probs = sinkhorn(prep_pipeline.alpha).detach().numpy()

    for i, col in enumerate(num_cols):
        for k, name in enumerate(STEP_NAMES):
            selection["steps"].setdefault(col, {})[name] = (
                beta_methods[k][int(np.argmax(beta_probs[k][i]))]
            )
        # DP-PORT: the relaxed alpha is doubly stochastic, not a permutation, so
        # a plain argmax per position can assign the same transformation type
        # twice. The Hungarian algorithm returns the permutation of maximum
        # total probability, which is the discrete prototype alpha is relaxing.
        row_ind, col_ind = linear_sum_assignment(-alpha_probs[i])
        order = [None] * len(STEP_NAMES)
        for pos, tf_type in zip(row_ind, col_ind):
            order[pos] = STEP_NAMES[tf_type]
        selection["order"][col] = order

    return selection


class SelectedPipeline:
    """The discrete pipeline DiffPrep selected, fitted and applied to real frames.

    The operators are the same objects the search used; they are simply re-fitted
    on the full poisoned partition and executed in the chosen order, per column.
    Everything is column-wise (scalers, outlier bounds, bin edges), except the
    imputers, which see the whole numerical or categorical block exactly as they
    do in FirstTransformer.
    """

    NUM_IMPUTER_FACTORY = {
        "mean": lambda: NumMVImputer("mean"),
        "median": lambda: NumMVImputer("median"),
        "DT": lambda: NumMVImputer("DT"),
        "MICE": lambda: NumMVImputer("MICE"),
        "EM": lambda: NumMVImputer("EM"),
        "KNN": lambda: NumMVImputer("KNN"),
        "mode": lambda: NumCatMVImputer("mode"),
        "num_mv_identity": lambda: NumMVIdentity(),
    }

    CAT_IMPUTER_FACTORY = {
        "mode": lambda: NumCatMVImputer("mode"),
        "dummy": lambda: CatMVImputer("dummy"),
        "cat_mv_identity": lambda: CatMVIdentity(),
    }

    def __init__(self, selection: Dict, num_cols: List[str], cat_cols: List[str],
                 method: str, prenorm_scaler=None):
        self.selection = selection
        self.num_cols = list(num_cols)
        self.cat_cols = list(cat_cols)
        self.method = method
        self.prenorm_scaler = prenorm_scaler
        self.num_imputers_: Dict[str, Any] = {}
        self.cat_imputers_: Dict[str, Any] = {}
        self.step_ops_: Dict[Tuple[str, str], Any] = {}

    # -- operator construction ------------------------------------------------
    @staticmethod
    def _make_step_op(step_name: str, method: str):
        if method == "identity":
            return Identity()
        if step_name == "normalization":
            return Normalizer(method)
        if step_name == "cleaning_outliers":
            return OutlierCleaner(method)
        if step_name == "discretization":
            strategy, n_bins = method.split("_")
            return Discretizer(n_bins=int(n_bins), strategy=strategy)
        raise ValueError(f"Unknown step {step_name}")

    # -- fit / transform ------------------------------------------------------
    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        return self._run(df, is_fit=True)

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        return self._run(df, is_fit=False)

    def _run(self, df: pd.DataFrame, is_fit: bool) -> pd.DataFrame:
        out = df.copy()

        X_num = out[self.num_cols].astype(float).values if self.num_cols else np.zeros((len(out), 0))
        X_cat = out[self.cat_cols].astype(object).values if self.cat_cols else np.zeros((len(out), 0), dtype=object)

        # DiffPrep-Flex pre-normalizes the numerical block before the pipeline.
        if self.prenorm_scaler is not None and self.num_cols:
            X_num = self.prenorm_scaler.transform(X_num)

        # ---- step 1: missing value imputation (block-wise, as upstream) ----
        num_by_method = self._impute_numerical(X_num, X_cat, is_fit)
        cat_by_method = self._impute_categorical(X_num, X_cat, is_fit)

        for j, col in enumerate(self.num_cols):
            chosen = self.selection["num_imputer"][col]
            out[col] = np.asarray(num_by_method[chosen][:, j], dtype=float)

        for j, col in enumerate(self.cat_cols):
            chosen = self.selection["cat_imputer"][col]
            out[col] = cat_by_method[chosen][:, j]

        # ---- steps 2-4: normalization / outlier removal / discretization ----
        for col in self.num_cols:
            values = out[col].values.astype(float).reshape(-1, 1)
            for step_name in self.selection["order"][col]:
                op_method = self.selection["steps"][col][step_name]
                key = (col, step_name)
                if is_fit:
                    op = self._make_step_op(step_name, op_method)
                    self.step_ops_[key] = op
                    values = op.fit_transform(values)
                else:
                    op = self.step_ops_.get(key)
                    if op is None:
                        continue
                    values = op.transform(values)
                values = np.asarray(values, dtype=float)
            out[col] = values.reshape(-1)

        return out

    def _impute_numerical(self, X_num, X_cat, is_fit) -> Dict[str, np.ndarray]:
        results: Dict[str, np.ndarray] = {}
        if not self.num_cols:
            return results
        for method in sorted(set(self.selection["num_imputer"].values())):
            if is_fit:
                imputer = self.NUM_IMPUTER_FACTORY[method]()
                self.num_imputers_[method] = imputer
            else:
                imputer = self.num_imputers_[method]

            if getattr(imputer, "input_type", "numerical") == "mixed":
                if is_fit:
                    X_num_t, _ = imputer.fit_transform(X_num, X_cat)
                else:
                    X_num_t, _ = imputer.transform(X_num, X_cat)
            else:
                X_num_t = imputer.fit_transform(X_num) if is_fit else imputer.transform(X_num)
            results[method] = np.asarray(X_num_t, dtype=float)
        return results

    def _impute_categorical(self, X_num, X_cat, is_fit) -> Dict[str, np.ndarray]:
        results: Dict[str, np.ndarray] = {}
        if not self.cat_cols:
            return results
        for method in sorted(set(self.selection["cat_imputer"].values())):
            if is_fit:
                imputer = self.CAT_IMPUTER_FACTORY[method]()
                self.cat_imputers_[method] = imputer
            else:
                imputer = self.cat_imputers_[method]

            if getattr(imputer, "input_type", "categorical") == "mixed":
                if is_fit:
                    _, X_cat_t = imputer.fit_transform(X_num, X_cat)
                else:
                    _, X_cat_t = imputer.transform(X_num, X_cat)
            else:
                X_cat_t = imputer.fit_transform(X_cat) if is_fit else imputer.transform(X_cat)
            results[method] = np.asarray(X_cat_t, dtype=object)
        return results

    # -- reporting ------------------------------------------------------------
    def describe(self) -> str:
        """A compact one-line summary: the most frequent operator per step."""
        parts = []
        imputers = list(self.selection["num_imputer"].values()) + list(
            self.selection["cat_imputer"].values()
        )
        if imputers:
            parts.append(f"impute:{pd.Series(imputers).mode().iloc[0]}")
        for step_name in STEP_NAMES:
            ops = [
                self.selection["steps"][c][step_name]
                for c in self.num_cols
                if c in self.selection["steps"]
            ]
            if ops:
                parts.append(f"{step_name.split('_')[0]}:{pd.Series(ops).mode().iloc[0]}")
        return " → ".join(parts) if parts else "(none)"

    def order_summary(self) -> str:
        """The most frequent transformation order across the numerical columns."""
        orders = [" → ".join(self.selection["order"][c]) for c in self.num_cols]
        if not orders:
            return "(none)"
        return pd.Series(orders).mode().iloc[0]


# ===========================================================================
# Driver: DiffPrep applied to one poisoned dataset
# ===========================================================================

DEFAULT_PARAMS = {
    "num_epochs": 2000,
    "batch_size": 512,
    "device": "cpu",
    "model_lr": [0.1, 0.01, 0.001],
    "weight_decay": 0,
    "momentum": 0.9,
    "patience": 3,
    "prep_lr": None,
    "temperature": 0.1,
    "pipeline_update_sample_size": 512,
    "init_method": "default",
    "diff_method": "num_diff",
    "sample": False,
    # Upstream bumps these two for DiffPrep-Flex, whose optimization problem is
    # harder (DiffPrepExperiment.run in experiment/diffprep_experiment.py).
    "flex_num_epochs": 3000,
    "flex_patience": 10,
}


def set_random_seed(seed: int, device: str = "cpu") -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)
    if "cuda" in device:
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def get_param_candidates(param_grid: Dict) -> List[Dict]:
    """Expand the list-valued entries of the parameter grid (experiment_utils.py)."""
    from itertools import product

    fixed_params = {}
    tuned_params = {}
    for name, parameter in param_grid.items():
        if isinstance(parameter, list):
            tuned_params[name] = parameter
        else:
            fixed_params[name] = parameter

    candidates = []
    for tuned in product(*tuned_params.values()):
        cand = copy.deepcopy(fixed_params)
        for n, p in zip(tuned_params.keys(), tuned):
            cand[n] = p
        candidates.append(cand)
    return candidates


class DiffPrep:
    """DiffPrep baseline driver.

    Exposes the same interface as the other data-preparation baselines of this
    repository (``prepare`` -> cleaned frame, residual mask, performance
    metrics, reusable pipeline description).
    """

    def __init__(self, method="diffprep_flex", model="log", train_seed=1, split_seed=1,
                 val_size=0.2, params=None, n_threads=None, verbose=False):
        if method not in ("diffprep_fix", "diffprep_flex"):
            raise ValueError(f"Unknown DiffPrep method: {method}")
        self.method = method
        self.model_name = model
        self.train_seed = train_seed
        self.split_seed = split_seed
        self.val_size = val_size
        self.verbose = verbose
        self.params = copy.deepcopy(DEFAULT_PARAMS)
        if params:
            self.params.update(params)
        if n_threads:
            torch.set_num_threads(int(n_threads))

    # -- one run of the bi-level search --------------------------------------
    def _run_once(self, data, n_classes, params) -> Tuple[Dict, Any]:
        X_train, y_train, X_val, y_val, X_test, y_test = data

        set_random_seed(self.train_seed, params["device"])

        prep_space = build_prep_space()
        if self.method == "diffprep_fix":
            prep_pipeline = DiffPrepFixPipeline(
                prep_space, temperature=params["temperature"], use_sample=params["sample"],
                diff_method=params["diff_method"], init_method=params["init_method"],
            )
        else:
            prep_pipeline = DiffPrepFlexPipeline(
                prep_space, temperature=params["temperature"], use_sample=params["sample"],
                diff_method=params["diff_method"], init_method=params["init_method"],
            )

        prep_pipeline.init_parameters(X_train, X_val, X_test)

        set_random_seed(self.train_seed, params["device"])
        input_dim = prep_pipeline.out_features
        if self.model_name == "log":
            model = LogisticRegression(input_dim, n_classes)
        elif self.model_name == "two":
            model = TwoLayerNet(input_dim, n_classes)
        else:
            raise Exception(f"Wrong model: {self.model_name}")
        model = model.to(params["device"])

        loss_fn = nn.CrossEntropyLoss()
        model_optimizer = torch.optim.SGD(
            model.parameters(), lr=params["model_lr"],
            weight_decay=params["weight_decay"], momentum=params["momentum"],
        )
        prep_lr = params["model_lr"] if params["prep_lr"] is None else params["prep_lr"]
        prep_pipeline_optimizer = torch.optim.Adam(
            prep_pipeline.parameters(), lr=prep_lr, betas=(0.5, 0.999),
            weight_decay=params["weight_decay"],
        )

        diff_prep = DiffPrepSGD(
            prep_pipeline, model, loss_fn, model_optimizer, prep_pipeline_optimizer,
            None, None, params,
        )
        result, best_model = diff_prep.fit(
            X_train, y_train, X_val, y_val, X_test, y_test, verbose=self.verbose
        )

        # Restore the parameters of the epoch the validation loss selected, so
        # the pipeline that gets materialized is the one being reported.
        if best_model is not None:
            prep_pipeline.load_state_dict(best_model["prep_pipeline"])

        return result, prep_pipeline

    def _grid_search(self, data, n_classes, param_grid: Dict) -> Tuple[Dict, Any, Dict]:
        """Tune the learning rate on the validation set (grid_search upstream)."""
        best_result = None
        best_pipeline = None
        best_params = None
        best_val_loss = float("inf")

        for params in get_param_candidates(param_grid):
            logger.debug(f"    model_lr={params['model_lr']}")
            try:
                result, prep_pipeline = self._run_once(data, n_classes, params)
            except Exception as exc:
                logger.warning(f"    model_lr={params['model_lr']} failed: {exc}")
                continue
            if result is None:
                continue
            if result["best_val_loss"] < best_val_loss:
                best_val_loss = result["best_val_loss"]
                best_result = result
                best_pipeline = prep_pipeline
                best_params = params

        if best_result is None:
            raise RuntimeError("every learning-rate candidate failed")
        return best_result, best_pipeline, best_params

    # -- public API -----------------------------------------------------------
    def apply_to_test(self, df_test: pd.DataFrame, pipeline_info: Dict) -> pd.DataFrame:
        """Transform a held-out frame with the transformers fitted on training.

        DiffPrep rescales, bins and imputes, so a hold-out read straight from
        ``data_poisoned/test/`` would no longer be on the scale the model was
        fitted on. Unlike Learn2Clean, which refits on each frame, the
        transformers here keep their training statistics: that is what
        ``FirstTransformer.transform`` and ``Transformer.forward(is_fit=False)``
        do upstream for ``dataset['test']``.
        """
        selected: SelectedPipeline = pipeline_info["selected_pipeline"]
        col_types = pipeline_info["col_types"]
        df = _coerce_types(df_test, col_types)
        out = selected.transform(df)
        return out[[c for c in df_test.columns if c in out.columns]]

    def prepare(
        self,
        df_poisoned: pd.DataFrame,
        mask_df: pd.DataFrame,
        df_test: Optional[pd.DataFrame] = None,
        file_name: str = "dataset",
    ) -> Tuple[pd.DataFrame, pd.DataFrame, Dict, Dict]:
        """Run DiffPrep on one poisoned dataset.

        Returns
        -------
        df_clean     : cleaned dataframe (same rows and columns as the input)
        residual_mask: boolean mask over df_clean (True = poisoned and still NaN)
        perf_metrics : timing / memory / search-quality metrics
        pipeline_info: the selected pipeline, fitted, plus the search metadata
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

        if task == "regression":
            # DP-PORT: upstream is classification-only — cross-entropy loss and
            # a softmax end model (model.py / diffprep_experiment.py). The paper
            # reports test accuracy on 18 classification datasets.
            raise ValueError(
                "DiffPrep only supports classification targets (cls_*); "
                f"'{target_col}' is a regression target"
            )

        df = _coerce_types(df_poisoned, col_types)
        num_cols, cat_cols, passthrough = _searchable_features(df, col_types)
        if passthrough:
            logger.info(
                f"  [DiffPrep] {len(passthrough)} column(s) left untouched "
                f"(all-missing or >1000 categories): {passthrough[:5]}"
                f"{'...' if len(passthrough) > 5 else ''}"
            )
        if not num_cols and not cat_cols:
            raise ValueError("No usable num_*/cat_* feature columns")

        # The clean hold-out is the test frame of the search; without one the
        # validation split doubles as the test frame, which only affects the
        # reported test accuracy, never the pipeline that is selected.
        if df_test is None:
            df_test_coerced = df.copy()
        else:
            df_test_coerced = _coerce_types(df_test, _identify_col_types(df_test))

        data, n_classes = build_search_data(
            df, df_test_coerced, num_cols, cat_cols, target_col,
            val_size=self.val_size, seed=self.split_seed,
        )
        X_train, y_train, X_val, y_val, X_test, y_test = data

        prenorm_scaler = None
        if self.method == "diffprep_flex":
            X_train, X_val, X_test, prenorm_scaler = min_max_normalize(
                X_train, X_val, X_test, num_cols
            )
            data = (X_train, y_train, X_val, y_val, X_test, y_test)

        # Flex has a harder optimization problem: upstream gives it more epochs
        # and more patience.
        run_params = copy.deepcopy(self.params)
        if self.method == "diffprep_flex":
            run_params["num_epochs"] = run_params.pop("flex_num_epochs")
            run_params["patience"] = run_params.pop("flex_patience")
        else:
            run_params.pop("flex_num_epochs", None)
            run_params.pop("flex_patience", None)

        logger.info(
            f"  [DiffPrep] {self.method} | {len(num_cols)} num + {len(cat_cols)} cat features "
            f"| train {len(X_train)} / val {len(X_val)} / test {len(X_test)} | {n_classes} classes"
        )

        search_t0 = time.perf_counter()
        best_result, best_pipeline, best_params = self._grid_search(data, n_classes, run_params)
        search_time_s = time.perf_counter() - search_t0

        out_features = best_pipeline.out_features

        # ---- materialize the selected pipeline ----------------------------
        mat_t0 = time.perf_counter()
        selection = extract_selected_pipeline(best_pipeline, self.method, num_cols, cat_cols)
        selected = SelectedPipeline(
            selection, num_cols, cat_cols, self.method, prenorm_scaler=prenorm_scaler
        )
        # DP-PORT: the search runs on the train split, but what gets written out
        # is the whole poisoned partition, so the chosen operators are re-fitted
        # on it. The hold-out is then transformed with those same statistics.
        df_clean = selected.fit_transform(df)
        materialize_time_s = time.perf_counter() - mat_t0

        # Column order and every passthrough / target / id column come back
        # exactly as they went in.
        df_clean = df_clean[list(df_poisoned.columns)]

        # ---- residual mask -------------------------------------------------
        residual_mask = mask_df & df_clean.isna()
        repaired = int(mask_df.sum().sum() - residual_mask.sum().sum())
        logger.info(
            f"  [DiffPrep] Repaired {repaired} / {int(mask_df.sum().sum())} poisoned cells "
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
            "n_rows_out": len(df_clean),
            "n_cols_out": len(df_clean.columns),
            "wall_time_s": round(wall_time_s, 4),
            "cpu_time_s": round(cpu_time_s, 4),
            "ram_before_mb": round(ram_before_mb, 3),
            "ram_after_mb": round(ram_after_mb, 3),
            "ram_peak_mb": round(ram_peak_mb, 3),
            "throughput_rows_per_s": round(n_rows / wall_time_s, 4) if wall_time_s > 0 else float("inf"),
            "method": self.method,
            "end_model": self.model_name,
            "metric_name": "accuracy",
            "best_pipeline": selected.describe(),
            "best_order": selected.order_summary(),
            "best_score": round(best_result["best_val_acc"], 6),
            "best_epoch": best_result["best_epoch"],
            "best_val_loss": round(best_result["best_val_loss"], 6),
            "best_tr_acc": round(best_result["best_tr_acc"], 6),
            "best_val_acc": round(best_result["best_val_acc"], 6),
            "best_test_acc": round(best_result["best_test_acc"], 6),
            "n_epochs_run": best_result["n_epochs_run"],
            "model_lr": best_params["model_lr"],
            "prep_lr": best_params["prep_lr"] if best_params["prep_lr"] is not None else best_params["model_lr"],
            "n_lr_candidates": len(get_param_candidates(run_params)),
            "n_search_features": out_features,
            "n_num_features": len(num_cols),
            "n_cat_features": len(cat_cols),
            "n_passthrough_features": len(passthrough),
            "search_time_s": round(search_time_s, 4),
            "materialize_time_s": round(materialize_time_s, 4),
        }

        pipeline_info = {
            "method": self.method,
            "end_model": self.model_name,
            "target_col": target_col,
            "col_types": col_types,
            "num_columns": num_cols,
            "cat_columns": cat_cols,
            "passthrough_columns": passthrough,
            "kept_columns": list(df_clean.columns),
            "dropped_columns": [],
            "selection": selection,
            "selected_pipeline": selected,
            "prenormalized": prenorm_scaler is not None,
            "result": best_result,
            "params": {
                "train_seed": self.train_seed,
                "split_seed": self.split_seed,
                "val_size": self.val_size,
                "model_lr": best_params["model_lr"],
                "prep_lr": best_params["prep_lr"],
                "temperature": best_params["temperature"],
                "batch_size": best_params["batch_size"],
                "num_epochs": best_params["num_epochs"],
                "patience": best_params["patience"],
                "init_method": best_params["init_method"],
                "diff_method": best_params["diff_method"],
                "sample": best_params["sample"],
                "pipeline_update_sample_size": best_params["pipeline_update_sample_size"],
                "weight_decay": best_params["weight_decay"],
                "momentum": best_params["momentum"],
            },
        }

        return df_clean, residual_mask, perf_metrics, pipeline_info


# ---------------------------------------------------------------------------
# Metrics helper (mirrors saga.py / learn2clean.py / data_preparation_pipeline.py)
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
    """Persist the selected pipeline description plus the search metadata.

    The live ``SelectedPipeline`` is dropped first: it carries fitted scikit-learn
    estimators whose classes are defined in this file, so a pickle written while
    the file runs as ``__main__`` would not reload anywhere else. What stays is
    the full description under ``selection`` — the operator every column was
    given and the order they run in — which is enough to rebuild and refit the
    pipeline with ``SelectedPipeline(**...)``, and is what the downstream
    loaders actually read.
    """
    import pickle as _pickle

    payload = {k: v for k, v in pipeline_info.items() if k != "selected_pipeline"}
    with open(path, "wb") as f:
        _pickle.dump(payload, f)


# ---------------------------------------------------------------------------
# Dataset processing (mirrors process_all_datasets in learn2clean.py)
# ---------------------------------------------------------------------------

def load_clean_trainval(clean_dir: str, dataset_filename: str, poison_test_size: float
                        ) -> Optional[pd.DataFrame]:
    """The un-poisoned train+val partition, rebuilt the way poison_data.py split it.

    ``scripts/poison_data.py`` holds out ``test_size`` of each clean CSV with
    ``df.sample(frac=test_size, random_state=42)`` and poisons the rest; the
    quail loaders rebuild that same partition with the same call when they swap
    in clean validation rows. Reproducing it here is what lets the clean rows be
    pushed through the pipeline fitted on their poisoned counterparts.
    """
    src = Path(clean_dir) / dataset_filename
    if not src.exists():
        return None
    full_clean = pd.read_csv(
        src, na_values=["?", "NA", "N/A", "NaN", "nan", "NAN", "", " "]
    )
    test_rows = full_clean.sample(frac=poison_test_size, random_state=42)
    return full_clean.drop(test_rows.index).reset_index(drop=True)


def check_dataset_complete(csv_file: Path, output_dir: str) -> bool:
    required_files = []
    for mode in ("ar", "nar"):
        mode_dir = os.path.join(output_dir, mode)
        required_files += [
            os.path.join(mode_dir, csv_file.name),
            os.path.join(mode_dir, csv_file.stem + "_mask.csv"),
            os.path.join(mode_dir, csv_file.stem + "_pipeline.pkl"),
            os.path.join(output_dir, "test", mode, csv_file.name),
            os.path.join(output_dir, "clean", mode, csv_file.name),
            os.path.join(output_dir, "metrics", f"{csv_file.stem}_{mode}_metrics.csv"),
            os.path.join(output_dir, "metrics", f"{csv_file.stem}_{mode}_perf_metrics.csv"),
        ]
    return all(os.path.exists(f) for f in required_files)


def process_all_datasets(
    input_dir: str,
    output_dir: str,
    datasets: Optional[List[str]] = None,
    method: str = "diffprep_flex",
    model: str = "log",
    train_seed: int = 1,
    split_seed: int = 1,
    val_size: float = 0.2,
    params: Optional[Dict] = None,
    n_threads: Optional[int] = None,
    clean_dir: str = "data",
    poison_test_size: float = 0.3,
    verbose: bool = False,
):
    """
    Apply DiffPrep to all poisoned datasets.

    Reads poisoned CSVs and their masks from ``input_dir/{ar,nar}/`` and writes
    cleaned CSVs, residual masks, the selected pipeline and metrics to
    ``output_dir/{ar,nar}/``. The clean hold-out of ``input_dir/test/`` is
    transformed with the transformers fitted on the training frame and written
    to ``output_dir/test/{mode}/``, one copy per corruption mode — AR and NAR
    select different pipelines, so they put the hold-out on different scales.

    The un-poisoned train+val partition is transformed the same way and written
    to ``output_dir/clean/{mode}/``. Downstream, quail can swap the validation
    split for its clean rows; those rows have to come off the same pipeline as
    the training rows, or they would be compared on a different scale — DiffPrep
    picks a normalizer and possibly a discretizer per feature, so a raw clean row
    is simply not in the same space as a transformed one.
    """
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.join(output_dir, "ar"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "nar"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "metrics"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "test", "ar"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "test", "nar"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "clean", "ar"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "clean", "nar"), exist_ok=True)

    cleaner = DiffPrep(
        method=method, model=model, train_seed=train_seed, split_seed=split_seed,
        val_size=val_size, params=params, n_threads=n_threads, verbose=verbose,
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
            "the validation split will stand in for it"
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

    logger.info(
        f"DiffPrep variant: {method} | end model: {model} | "
        f"lr grid: {cleaner.params['model_lr']}"
    )

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
                out_clean = Path(output_dir) / "clean" / mode / csv_file.name
                out_metrics = Path(output_dir) / "metrics" / f"{csv_file.stem}_{mode}_metrics.csv"
                out_perf = Path(output_dir) / "metrics" / f"{csv_file.stem}_{mode}_perf_metrics.csv"

                if all(p.exists() for p in [out_csv, out_mask, out_pkl, out_test, out_clean,
                                            out_metrics, out_perf]):
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

                src_test = test_dir / csv_file.name
                df_test = None
                if src_test.exists():
                    df_test = pd.read_csv(
                        src_test, na_values=["?", "NA", "N/A", "NaN", "nan", "NAN", "", " "]
                    )

                df_clean, residual_mask, perf, pipeline_info = cleaner.prepare(
                    df_mode, mask_mode, df_test=df_test, file_name=f"{csv_file.stem}_{mode}"
                )
                metrics = calculate_residual_metrics(residual_mask)

                df_clean.to_csv(out_csv, index=False)
                residual_mask.to_csv(out_mask, index=False)
                save_pipeline_info(str(out_pkl), pipeline_info)

                if df_test is not None:
                    df_test_clean = cleaner.apply_to_test(df_test, pipeline_info)
                    df_test_clean.to_csv(out_test, index=False)
                    logger.info(
                        f"  {mode.upper()} hold-out prepared: "
                        f"{df_test_clean.shape[0]}×{df_test_clean.shape[1]}"
                    )
                else:
                    logger.warning(f"  {mode.upper()} clean hold-out not found: {src_test}")

                df_clean_trainval = load_clean_trainval(
                    clean_dir, csv_file.name, poison_test_size
                )
                if df_clean_trainval is None:
                    logger.warning(
                        f"  {mode.upper()} original clean CSV not found in {clean_dir}; "
                        "no clean train+val partition will be written"
                    )
                elif len(df_clean_trainval) != len(df_mode):
                    logger.warning(
                        f"  {mode.upper()} clean partition has {len(df_clean_trainval)} rows but "
                        f"the poisoned one has {len(df_mode)}; poison_test_size={poison_test_size} "
                        "probably does not match the one used by poison_data.py — skipping it"
                    )
                else:
                    cleaner.apply_to_test(df_clean_trainval, pipeline_info).to_csv(
                        out_clean, index=False
                    )

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
                    f"| {perf['n_rows_out']}×{perf['n_cols_out']} "
                    f"| {perf['wall_time_s']:.1f}s wall | {perf['ram_peak_mb']:.1f} MB peak "
                    f"| val_acc={perf['best_val_acc']} test_acc={perf['best_test_acc']} "
                    f"(lr={perf['model_lr']}, {perf['n_epochs_run']} epochs) "
                    f"| pipeline: {perf['best_pipeline']} | order: {perf['best_order']}"
                )

                gc.collect()

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
        description="DiffPrep differentiable data-preparation pipeline search baseline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
                Paper: Peng Li, Zhiyi Chen, Xu Chu, Kexin Rong. "DiffPrep:
                    Differentiable Data Preprocessing Pipeline Search for Learning
                    over Tabular Data." SIGMOD '23, Article 183.
                Code : https://github.com/chu-data-lab/DiffPrep

                The pipeline search is relaxed into a continuous problem (softmax
                over operators, Sinkhorn over the order) and solved as a bi-level
                optimization by gradient descent, training the end model once.

                Search space (per feature):
                Missing value imputation : mean, median, DT, MICE, mode / dummy
                Normalization            : ZS, MM, MA, RS
                Outlier removal          : ZS_{2,3,4}, MAD_{2,2.5,3}, IQR_{1,1.5,2}, identity
                Discretization           : {uniform,quantile}_{5,10,20}, identity

                Variants:
                diffprep_fix  - pre-defined transformation order (Section 3)
                diffprep_flex - order learned per feature via Sinkhorn (Section 4)

                Examples:
                python scripts/diffprep.py
                python scripts/diffprep.py --input_dir data_poisoned --output_dir data_cleaned_diffprep
                python scripts/diffprep.py --dataset tic_tac_toe --method diffprep_fix --verbose
        """,
    )
    parser.add_argument(
        "--input_dir", type=str, default="data_poisoned",
        help="Root directory of poisoned data (default: data_poisoned)",
    )
    parser.add_argument(
        "--output_dir", type=str, default=None,
        help="Root directory for cleaned output "
             "(default: config.yaml diffprep_data_dir, else data_cleaned_diffprep)",
    )
    parser.add_argument(
        "--config", type=str, default="config.yaml",
        help="Path to experiment config file (default: config.yaml)",
    )
    parser.add_argument(
        "--dataset", type=str, default=None,
        help="Process only this dataset (overrides config; name without .csv)",
    )
    parser.add_argument(
        "--clean_dir", type=str, default="data",
        help="Directory with the original clean CSVs, used to write the "
             "transformed clean train+val partition (default: data)",
    )
    parser.add_argument(
        "--poison_test_size", type=float, default=None,
        help="Hold-out fraction poison_data.py used (default: config.yaml test_size)",
    )
    parser.add_argument("--method", type=str, default=None,
                        choices=["diffprep_fix", "diffprep_flex"],
                        help="DiffPrep variant (default: diffprep_flex)")
    parser.add_argument("--model", type=str, default=None, choices=["log", "two"],
                        help="End model driving the search: logistic regression or 2-layer NN")
    parser.add_argument("--train_seed", type=int, default=None, help="Seed for model/pipeline init")
    parser.add_argument("--split_seed", type=int, default=None, help="Seed for the train/val split")
    parser.add_argument("--val_size", type=float, default=None,
                        help="Validation share of the poisoned partition (outer objective)")
    parser.add_argument("--num_epochs", type=int, default=None,
                        help="Max epochs for DiffPrep-Fix")
    parser.add_argument("--flex_num_epochs", type=int, default=None,
                        help="Max epochs for DiffPrep-Flex")
    parser.add_argument("--batch_size", type=int, default=None, help="Model mini-batch size")
    parser.add_argument("--patience", type=int, default=None,
                        help="Early-stopping patience for DiffPrep-Fix")
    parser.add_argument("--flex_patience", type=int, default=None,
                        help="Early-stopping patience for DiffPrep-Flex")
    parser.add_argument("--model_lr", type=float, nargs="+", default=None,
                        help="Learning-rate grid tuned on the validation set")
    parser.add_argument("--prep_lr", type=float, default=None,
                        help="Learning rate of the pipeline parameters (default: same as model_lr)")
    parser.add_argument("--temperature", type=float, default=None,
                        help="Temperature of the relaxed categorical sampling")
    parser.add_argument("--pipeline_update_sample_size", type=int, default=None,
                        help="Mini-batch size used to update the pipeline parameters")
    parser.add_argument("--init_method", type=str, default=None, choices=["default", "random"],
                        help="Initialization of the pipeline parameters")
    parser.add_argument("--sample", action="store_true",
                        help="Use Gumbel-softmax sampling instead of the plain softmax relaxation")
    parser.add_argument("--device", type=str, default=None, help="torch device (default: cpu)")
    parser.add_argument("--n_threads", type=int, default=None,
                        help="torch intra-op threads (default: torch's own default)")
    parser.add_argument("--verbose", action="store_true",
                        help="Show per-epoch progress of the bi-level search")

    args = parser.parse_args()

    _config: dict = {}
    if Path(args.config).exists():
        with open(args.config) as _f:
            _config = yaml.safe_load(_f) or {}

    dp_cfg = _config.get("diffprep") or {}

    def _opt(name, default):
        """CLI flag > config.yaml diffprep.<name> > built-in default."""
        cli = getattr(args, name, None)
        if cli is not None:
            return cli
        return dp_cfg.get(name, default)

    if args.verbose:
        logger.remove()
        logger.add(
            sys.stderr,
            level="DEBUG",
            format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <level>{message}</level>",
        )

    output_dir = args.output_dir or _config.get("diffprep_data_dir", "data_cleaned_diffprep")

    if args.dataset:
        datasets = [args.dataset]
    else:
        datasets = _config.get("datasets") or None

    _params = {
        "num_epochs": _opt("num_epochs", DEFAULT_PARAMS["num_epochs"]),
        "flex_num_epochs": _opt("flex_num_epochs", DEFAULT_PARAMS["flex_num_epochs"]),
        "batch_size": _opt("batch_size", DEFAULT_PARAMS["batch_size"]),
        "patience": _opt("patience", DEFAULT_PARAMS["patience"]),
        "flex_patience": _opt("flex_patience", DEFAULT_PARAMS["flex_patience"]),
        "model_lr": list(_opt("model_lr", DEFAULT_PARAMS["model_lr"])),
        "prep_lr": _opt("prep_lr", DEFAULT_PARAMS["prep_lr"]),
        "temperature": _opt("temperature", DEFAULT_PARAMS["temperature"]),
        "pipeline_update_sample_size": _opt(
            "pipeline_update_sample_size", DEFAULT_PARAMS["pipeline_update_sample_size"]
        ),
        "init_method": _opt("init_method", DEFAULT_PARAMS["init_method"]),
        "diff_method": dp_cfg.get("diff_method", DEFAULT_PARAMS["diff_method"]),
        "sample": args.sample or bool(dp_cfg.get("sample", DEFAULT_PARAMS["sample"])),
        "device": _opt("device", DEFAULT_PARAMS["device"]),
        "weight_decay": dp_cfg.get("weight_decay", DEFAULT_PARAMS["weight_decay"]),
        "momentum": dp_cfg.get("momentum", DEFAULT_PARAMS["momentum"]),
    }

    process_all_datasets(
        input_dir=args.input_dir,
        output_dir=output_dir,
        datasets=datasets,
        method=_opt("method", "diffprep_flex"),
        model=_opt("model", "log"),
        train_seed=_opt("train_seed", 1),
        split_seed=_opt("split_seed", 1),
        val_size=_opt("val_size", 0.2),
        params=_params,
        n_threads=_opt("n_threads", None),
        clean_dir=args.clean_dir,
        poison_test_size=(
            args.poison_test_size
            if args.poison_test_size is not None
            else _config.get("test_size", 0.3)
        ),
        verbose=args.verbose,
    )
