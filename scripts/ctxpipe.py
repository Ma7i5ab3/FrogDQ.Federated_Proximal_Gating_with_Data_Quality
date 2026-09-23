#!/usr/bin/env python3
"""
CtxPipe Data Preparation Baseline
Python port of the official reference implementation from:

  Haotian Gao, Shaofeng Cai, Tien Tuan Anh Dinh, Zhiyong Huang, Beng Chin Ooi.
  "CtxPipe: Context-aware Data Preparation Pipeline Construction for Machine
  Learning." Proc. ACM Manag. Data 2(6), SIGMOD '25, Article 231 (2024).
  https://doi.org/10.1145/3698831

Upstream source: https://github.com/ctxpipe/ctxpipe (commit 79caaa1)

CtxPipe constructs a data-preparation pipeline with deep Q-network (DQN) agents,
in two phases (Section 5.3, Algorithm 2). A first agent picks the *logical*
pipeline — the order of the component types — then one agent per type picks the
*physical* component that fills each slot, and that component is executed before
the next one is chosen. Every agent sees column-wise statistical features of the
current intermediate dataset plus the pipeline built so far (Section 4.2,
Figure 5). The context plug-in (Section 5, Figure 6) adds what those statistics
miss, the semantics of the data: sampled rows are serialized to CSV, embedded
with the GTE-large text model, projected by the context integrator to one value
per action, gated, and used to rescale the Q-values.

Search space (comp.py upstream, Table 3 of the paper):
  Logical pipeline      : ImputerNum → ImputerCat → Encoder → one of the 6 orders
                          of {FeaturePreprocessing, FeatureEngine, FeatureSelection}
  ImputerNum            : mean, median, most frequent
  ImputerCat            : most frequent (fixed, not chosen by an agent)
  Encoder               : NumericData, LabelEncoder, OneHotEncoder
  FeaturePreprocessing  : MinMax, MaxAbs, Robust, Standard, Quantile, Power,
                          Normalizer, KBinsDiscretizer, none
  FeatureEngine         : PolynomialFeatures, InteractionFeatures, PCA,
                          IncrementalPCA, KernelPCA, TruncatedSVD,
                          RandomTreesEmbedding, none
  FeatureSelection      : VarianceThreshold, none
  Reward                : accuracy of a logistic regression (liblinear) on an
                          80/20 split of the dataset

As in the paper's evaluation (Section 7), CtxPipe runs here with the agents
released upstream — trained for 32,000 steps on the HAIPipe corpus
(models/ctxpipe-3linear/ctx_32000_*.pkl, copied to scripts/ctxpipe_weights/) —
and builds the pipeline of every unseen dataset greedily, with no further
training: the Tester.inference path of upstream's test.py.

Input:  data_poisoned/{ar,nar}/       (poisoned CSV + mask from poison_data.py)
        data_poisoned/test/           (clean hold-out)
Output: data_cleaned_ctxpipe/{ar,nar}/       (prepared CSV + residual mask + pipeline)
        data_cleaned_ctxpipe/test/{ar,nar}/  (hold-out, put through the pipeline
                                              fitted on training)
        data_cleaned_ctxpipe/clean/{ar,nar}/ (the un-poisoned train+val partition,
                                              put through the same pipeline, so a
                                              clean validation split lives in the
                                              same feature space)

Every deviation from upstream is flagged with a "CTX-PORT" note. The ones that
matter for reading the results:

  1. The whole selected pipeline is materialized. Upstream never writes a
     dataset out: it reads the reward off the pipeline it has just executed.
     Here the six components the agents chose are re-fitted on the full
     poisoned partition, applied to it, and then applied with those same
     statistics to the hold-out and to the clean partition. Encoder, feature
     engineering and feature selection change the columns, so — unlike CP,
     Saga++ and DiffPrep — the output is not column-preserving: every feature
     comes out numeric and prefixed num_* (its original name when a step
     transforms the column in place, num_<op>_<i> when a step derives new
     features). Rows are never dropped, so the clean partition stays aligned
     with the poisoned one row by row. The residual mask is written over the
     output columns: a derived feature inherits the poison mask of the input
     columns it is computed from.

  2. No second standardization downstream. The values already are on the scale,
     and in the feature space, CtxPipe chose, so quail.data.load_ctxpipe_data
     builds its TabularPreprocessor with scale_numerical=False, as for DiffPrep.

  3. Where the paper and the released code disagree, the code is followed,
     because the released agents were trained with it:
       - the context embedding reads 100 sampled rows, not 50 (Section 6);
       - the context gate is applied twice, beta = alpha∘(alpha∘C(e) − 1) + 1,
         instead of beta = alpha∘(C(e) − 1) + 1 (Algorithm 1, lines 6-7);
       - the gate weights W_c and W_b are plain tensors, not nn.Parameters, so
         they are never trained nor saved: every agent draws them afresh from a
         normal distribution. This port re-seeds torch with upstream's
         RANDOM_SEED before building the agents of every run, so the draws, and
         therefore the pipeline, do not depend on which datasets ran before.

  4. A component that raises while executing counts as a rejected action and
     the agent falls back to its next-best one. That is what upstream does
     during training (has_timeout=True); on the inference path
     (has_timeout=False) the same exception escapes the subprocess wrapper and
     aborts the whole dataset.

Note: the target column (cls_*) is never transformed. CtxPipe only supports
classification, as upstream.
"""

import copy
import gc
import inspect
import os
import random
import re
import sys
import time
import tracemalloc
import warnings
from enum import Enum, auto
from itertools import compress
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import psutil
import torch
from loguru import logger
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA, IncrementalPCA, KernelPCA, TruncatedSVD
from sklearn.ensemble import RandomTreesEmbedding
from sklearn.feature_selection import VarianceThreshold
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import (
    KBinsDiscretizer,
    LabelEncoder,
    MaxAbsScaler,
    MinMaxScaler,
    Normalizer,
    PolynomialFeatures,
    PowerTransformer,
    QuantileTransformer,
    RobustScaler,
    StandardScaler,
)
from torch import nn

warnings.filterwarnings("ignore")

# env.init() upstream: the tokenizer must not fork its own thread pool.
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

logger.remove()
logger.add(
    sys.stderr,
    level="INFO",
    format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <level>{message}</level>",
)


# ===========================================================================
# Upstream configuration — config.py, env.py, deterministic.py, agent/model.py
# ===========================================================================

RANDOM_SEED = 1145                          # deterministic.RANDOM_SEED

COLUMN_NUM = 100                            # GlobalConfig.column_num (N, Section 6)
COLUMN_FEATURE_DIM = 19 + 14                # GlobalConfig.column_feature_dim
DATA_DIM = COLUMN_NUM * COLUMN_FEATURE_DIM  # GlobalConfig.data_dim
BLANK_REWARD = 0.0                          # GlobalConfig.blank_reward

SEQ_EMBEDDING_DIM = 96                      # DQNConfig
SEQ_HIDDEN_SIZE = 96
SEQ_NUM_LAYERS = 1
PREDICTOR_EMBEDDING_DIM = 16
LPIPELINE_EMBEDDING_DIM = 8

N_DIM_EMBED = 1024                          # agent/model.py: GTE-large output size
N_DIM_FIRST = 128
INFO_EXTRACTION_POS = 10                    # the gate reads the output of nn[0] ...
CTX_INTEGRATION_POS = 1                     # ... and rescales the output of nn[-1]

CTX_MODEL_PATH = "thenlper/gte-large"       # ctx.GTEEmbedder.CTX_MODEL_PATH
CTX_MAX_LENGTH = 512                        # tokenizer max_length (L in the paper)
N_CTX_ROWS = 100                            # agent/dqn.py Agent.act: sample(n=100)

WEIGHTS_TAG = "ctx_32000"                   # 32,000 training steps (Section 6)
DEFAULT_WEIGHTS_DIR = Path(__file__).resolve().parent / "ctxpipe_weights"

# Train/test split of the search environment (env/pipeline.py Pipeline.load_data).
SEARCH_TRAIN_SIZE = 0.8
SEARCH_SPLIT_SEED = 0


# ---------------------------------------------------------------------------
# Module-level knobs set from the CLI / config.yaml
# ---------------------------------------------------------------------------
# CTX-PORT: KernelPCA builds the full n×n kernel matrix, which runs out of memory
# on large datasets. Upstream only bounds it by the 60 s step timeout it applies
# during training; at inference nothing stops it. When set, KernelPCA refuses
# frames with more rows than this (can_accept -> False), exactly as it refuses a
# frame with too few columns, and the agent falls back to its next-best
# component. None keeps upstream's unbounded behaviour.
KERNEL_PCA_MAX_ROWS: Optional[int] = None


def seed_everything(seed: int = RANDOM_SEED) -> None:
    """deterministic.seed_everything."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ===========================================================================
# Components — upstream env/primitives/
# ===========================================================================
#
# Each component keeps upstream's search-time API verbatim:
#
#   transform(train_x, test_x, train_y) -> (train_x, test_x)
#
# including its quirks (column renaming, statistics re-fitted on the test split),
# because that is what the agents were trained to read.
#
# CTX-PORT: upstream only ever executes a component on the (train, test) pair of
# the search. Writing the prepared data out needs the component fitted on one
# frame and applied, with those statistics, to others (the hold-out, the clean
# partition), with the output columns named after the quail conventions. That is
# the materialization API added to every component:
#
#   fit_frame(X, y)  -> learns the statistics and sets out_sources_, the input
#                       columns every output column is computed from
#   apply_frame(X)   -> transforms any frame with the same columns as X

def _numeric_columns(data: pd.DataFrame) -> pd.Index:
    """``DataFrame._get_numeric_data().columns``, as upstream (bools included)."""
    return data._get_numeric_data().columns


def _has_na_or_inf(data: pd.DataFrame) -> bool:
    """``data.isna().any().any()`` under ``pd.option_context("mode.use_inf_as_na", True)``.

    CTX-PORT: pandas 2.1 deprecated that option (3.0 removes it), so the inf
    check is spelled out.
    """
    if data.shape[1] == 0:
        return False
    if data.isna().to_numpy().any():
        return True
    num = data._get_numeric_data()
    if num.shape[1] == 0:
        return False
    return bool(np.isinf(num.to_numpy(dtype=float, na_value=np.nan)).any())


def catch_num(data: pd.DataFrame, sort: bool = False) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Split a frame into its object (categorical) and other (numerical) columns.

    Upstream defines one copy of this per primitives module; the imputer and
    encoder copies sort the numerical columns by name, the others do not.
    """
    num_cols = [col for col in data.columns if str(data[col].dtypes) != "object"]
    if sort:
        num_cols.sort()
    cat_cols = [col for col in data.columns if col not in num_cols]
    return data[cat_cols], data[num_cols]


def _base_name(col: str) -> str:
    """A column name without its quail type prefix."""
    for prefix in ("num_", "cat_"):
        if col.startswith(prefix):
            return col[len(prefix):]
    return col


def _unique_name(name: str, taken: set) -> str:
    out, k = name, 1
    while out in taken:
        out = f"{name}_{k}"
        k += 1
    taken.add(out)
    return out


class Primitive:
    """Blank component (env/primitives/primitive.py): leaves the data as it is."""

    def __init__(self, name="blank"):
        self.id = 0
        self.gid = 25
        self.name = name
        self.description = str(name)
        self.hyperparams = []
        self.type = "blank"

    def transform(self, train_x: pd.DataFrame, test_x: pd.DataFrame, train_y) -> Tuple[pd.DataFrame, pd.DataFrame]:
        return train_x, test_x

    def can_accept(self, data):
        return True

    def can_accept_a(self, data):
        if data.empty:
            return False
        elif data.shape[1] == 0:
            return False
        num_cols = _numeric_columns(data)
        if not len(num_cols) == 0:
            return True
        return False

    def can_accept_c(self, data):
        if data.empty:
            return False
        elif data.shape[1] == 0:
            return False
        cols = data
        num_cols = _numeric_columns(data)
        cat_cols = list(set(cols) - set(num_cols))

        if _has_na_or_inf(data):
            return False
        if not len(cat_cols) == 0:
            return False
        return True

    def is_needed(self, data):
        return True

    # -- materialization API (CTX-PORT) --------------------------------------
    def fit_frame(self, X: pd.DataFrame, y) -> None:
        self.out_sources_ = {c: [c] for c in X.columns}

    def apply_frame(self, X: pd.DataFrame) -> pd.DataFrame:
        return X.reset_index(drop=True)

    def __repr__(self) -> str:
        return f"<{self.name}>"


# ---------------------------------------------------------------------------
# ImputerNum — env/primitives/imputernum.py
# ---------------------------------------------------------------------------

class _NumImputerPrim(Primitive):
    """Shared body of ImputerMean, ImputerMedian and ImputerNumPrim.

    The three upstream classes are the same code with a different SimpleImputer
    strategy.
    """

    strategy = "mean"

    def __init__(self, name, id_, gid, description):
        super().__init__(name=name)
        self.id = id_
        self.gid = gid
        self.hyperparams = []
        self.type = "ImputerNum"
        self.description = description
        self.imp = SimpleImputer(strategy=self.strategy)
        self.accept_type = "c"
        self.need_y = False

    def can_accept(self, data):
        return True

    def is_needed(self, data):
        if data.isnull().any().any():
            return True
        return False

    def transform(self, train_x, test_x, train_y):
        cat_trainX, num_trainX = catch_num(train_x, sort=True)
        cat_testX, num_testX = catch_num(test_x, sort=True)
        self.imp.fit(num_trainX)
        num_trainX = self.imp.fit_transform(num_trainX)
        num_trainX = pd.DataFrame(num_trainX).reset_index(drop=True).infer_objects()
        cols = ["num_" + str(i) for i in num_trainX.columns]
        num_trainX.columns = cols
        train_data_x = pd.concat(
            [cat_trainX.reset_index(drop=True), num_trainX.reset_index(drop=True)],
            axis=1,
        )

        # Upstream re-fits the imputer on the test split. Kept: it only affects
        # the reward the search reports, never what the agents see.
        num_testX = self.imp.fit_transform(num_testX)
        num_testX = pd.DataFrame(num_testX).reset_index(drop=True).infer_objects()
        cols = ["num_" + str(i) for i in num_testX.columns]
        num_testX.columns = cols
        test_data_x = pd.concat(
            [cat_testX.reset_index(drop=True), num_testX.reset_index(drop=True)], axis=1
        )
        return train_data_x, test_data_x

    def fit_frame(self, X, y):
        cat, num = catch_num(X, sort=True)
        self.frame_cat_ = list(cat.columns)
        self.frame_num_in_ = list(num.columns)
        self.frame_imp_ = None
        self.frame_num_out_ = list(self.frame_num_in_)
        if self.frame_num_in_:
            self.frame_imp_ = SimpleImputer(strategy=self.strategy).fit(num)
            # Entirely-missing columns are dropped, as upstream's SimpleImputer does.
            self.frame_num_out_ = [str(c) for c in self.frame_imp_.get_feature_names_out()]
        self.out_sources_ = {c: [c] for c in self.frame_cat_ + self.frame_num_out_}

    def apply_frame(self, X):
        cat = X[self.frame_cat_].reset_index(drop=True)
        if self.frame_imp_ is None:
            return cat
        num = pd.DataFrame(
            self.frame_imp_.transform(X[self.frame_num_in_]), columns=self.frame_num_out_
        )
        return pd.concat([cat, num], axis=1)


class ImputerMean(_NumImputerPrim):
    strategy = "mean"

    def __init__(self, random_state=0):
        super().__init__(
            "ImputerMean", 1, 1,
            "Imputation transformer for completing missing values by mean.",
        )


class ImputerMedian(_NumImputerPrim):
    strategy = "median"

    def __init__(self, random_state=0):
        super().__init__(
            "ImputerMedian", 2, 2,
            "Imputation transformer for completing missing values by median.",
        )


class ImputerNumPrim(_NumImputerPrim):
    strategy = "most_frequent"

    def __init__(self, random_state=0):
        super().__init__(
            "ImputerNumMode", 4, 4,
            "Imputation transformer for completing missing values by mode.",
        )


# ---------------------------------------------------------------------------
# ImputerCat — env/primitives/imputercat.py
# ---------------------------------------------------------------------------

class ImputerCatPrim(Primitive):
    def __init__(self, random_state=0):
        super().__init__(name="ImputerCatMode")
        self.id = 1
        self.gid = 5
        self.hyperparams = []
        self.type = "ImputerNum"  # sic upstream
        self.description = "Imputation transformer for completing missing values by mode."
        self.imp = SimpleImputer(strategy="most_frequent")
        self.accept_type = "c"
        self.need_y = False

    def can_accept(self, data):
        return True

    def is_needed(self, data):
        if data.isnull().any().any():
            return True
        return False

    def transform(self, train_x, test_x, train_y):
        cat_trainX, num_trainX = catch_num(train_x, sort=True)
        cat_testX, num_testX = catch_num(test_x, sort=True)
        self.imp.fit(cat_trainX)
        cat_trainX = self.imp.fit_transform(cat_trainX.reset_index(drop=True))
        cat_trainX = pd.DataFrame(cat_trainX).reset_index(drop=True).infer_objects()
        cols = ["col_" + str(i) for i in cat_trainX.columns]
        cat_trainX.columns = cols
        cat_trainX = cat_trainX.reset_index(drop=True)
        num_trainX = num_trainX.reset_index(drop=True)

        train_data_x = pd.concat([cat_trainX, num_trainX], axis=1)
        # Re-fitted on the test split, as upstream.
        cat_testX = self.imp.fit_transform(cat_testX.reset_index(drop=True))
        cat_testX = pd.DataFrame(cat_testX).reset_index(drop=True).infer_objects()
        cols = ["col_" + str(i) for i in cat_testX.columns]
        cat_testX.columns = cols
        test_data_x = pd.concat(
            [cat_testX.reset_index(drop=True), num_testX.reset_index(drop=True)], axis=1
        )
        return train_data_x, test_data_x

    def fit_frame(self, X, y):
        cat, num = catch_num(X, sort=True)
        self.frame_cat_in_ = list(cat.columns)
        self.frame_num_ = list(num.columns)
        self.frame_imp_ = None
        self.frame_cat_out_ = list(self.frame_cat_in_)
        if self.frame_cat_in_:
            self.frame_imp_ = SimpleImputer(strategy="most_frequent").fit(cat)
            self.frame_cat_out_ = [str(c) for c in self.frame_imp_.get_feature_names_out()]
        self.out_sources_ = {c: [c] for c in self.frame_cat_out_ + self.frame_num_}

    def apply_frame(self, X):
        num = X[self.frame_num_].reset_index(drop=True)
        if self.frame_imp_ is None:
            return num
        cat = pd.DataFrame(
            self.frame_imp_.transform(X[self.frame_cat_in_]), columns=self.frame_cat_out_
        ).astype(object)
        return pd.concat([cat, num], axis=1)


# ---------------------------------------------------------------------------
# Encoder — env/primitives/encoder.py
# ---------------------------------------------------------------------------

class NumericDataPrim(Primitive):
    def __init__(self, random_state=0):
        super().__init__(name="NumericData")
        self.id = 1
        self.gid = 6
        self.hyperparams = []
        self.type = "Encoder"
        self.description = "Extracts only numeric data columns from input."
        self.accept_type = "a"
        self.need_y = False

    def can_accept(self, data):
        return self.can_accept_a(data)

    def is_needed(self, data):
        cols = data.columns
        num_cols = _numeric_columns(data)
        if not len(cols) == len(num_cols):
            return True
        return False

    def transform(self, train_x, test_x, train_y):
        num_cols = _numeric_columns(train_x)
        train_x = train_x[num_cols]
        num_cols = _numeric_columns(test_x)
        test_x = test_x[num_cols]
        return train_x, test_x

    def fit_frame(self, X, y):
        self.frame_keep_ = list(_numeric_columns(X))
        self.out_sources_ = {c: [c] for c in self.frame_keep_}

    def apply_frame(self, X):
        return X[self.frame_keep_].reset_index(drop=True)


class OneHotEncoderPrim(Primitive):
    # can handle missing values. turns nans to extra category
    def __init__(self, random_state=0):
        super().__init__(name="OneHotEncoder")
        self.id = 2
        self.gid = 7
        self.hyperparams = []
        self.type = "data preprocess"
        self.description = "Encode categorical features as a one-hot numeric array."
        self.accept_type = "c2"
        self.need_y = False

    def can_accept(self, data):
        cols = data
        num_cols = _numeric_columns(data)
        cat_cols = list(set(cols) - set(num_cols))
        if len(cat_cols) > 15:
            return False
        return True

    def is_needed(self, data):
        cols = data
        num_cols = _numeric_columns(data)
        cat_cols = list(set(cols) - set(num_cols))
        if len(cat_cols) == 0:
            return False
        return True

    def transform(self, train_x, test_x, train_y):
        # Upstream also builds a ColumnTransformer here that it never uses.
        cat_trainX, num_trainX = catch_num(train_x, sort=True)
        cat_testX, num_testX = catch_num(test_x, sort=True)
        cat_cols = cat_trainX.columns

        len_trainx = num_trainX.shape[0]
        for col in cat_cols:
            temp = pd.get_dummies(
                pd.concat([cat_trainX[col], cat_testX[col]], axis=0).reset_index(drop=True),
                prefix=col,
            )
            train_d = temp.iloc[0:len_trainx, :].reset_index(drop=True)
            test_d = temp.iloc[len_trainx:, :].reset_index(drop=True)
            train_x = pd.concat([train_x.reset_index(drop=True), train_d], axis=1).reset_index(drop=True)
            test_x = pd.concat([test_x.reset_index(drop=True), test_d], axis=1).reset_index(drop=True)

        train_x = train_x.drop(columns=cat_cols).infer_objects()
        test_x = test_x.drop(columns=cat_cols).infer_objects()
        return train_x, test_x

    def fit_frame(self, X, y):
        # CTX-PORT: upstream derives the dummy columns from train and test
        # together (get_dummies over their concatenation). Here the categories
        # are the ones seen in the fitted frame, in get_dummies' sorted order; a
        # category seen only later gets all-zero dummies.
        cat, _ = catch_num(X, sort=True)
        self.frame_cat_ = list(cat.columns)
        self.frame_keep_ = [c for c in X.columns if c not in self.frame_cat_]
        taken = set(self.frame_keep_)
        self.frame_levels_: Dict[str, List[Tuple[Any, str]]] = {}
        self.out_sources_ = {c: [c] for c in self.frame_keep_}
        for col in self.frame_cat_:
            values = sorted(cat[col].dropna().unique(), key=str)
            levels = [(v, _unique_name(f"num_{_base_name(col)}_{v}", taken)) for v in values]
            self.frame_levels_[col] = levels
            for _, name in levels:
                self.out_sources_[name] = [col]

    def apply_frame(self, X):
        blocks = [X[self.frame_keep_].reset_index(drop=True)]
        for col, levels in self.frame_levels_.items():
            values = X[col].reset_index(drop=True)
            blocks.append(pd.DataFrame({name: (values == v).astype(int) for v, name in levels}))
        return pd.concat(blocks, axis=1)


class LabelEncoderPrim(Primitive):
    def __init__(self, random_state=0):
        super().__init__(name="LabelEncoder")
        self.id = 3
        self.gid = 8
        self.hyperparams = []
        self.type = "data preprocess"
        self.description = "Encode labels with value between 0 and n_classes-1."
        self.preprocess = {}
        self.accept_type = "b"
        self.need_y = False

    def can_accept(self, data):
        return True

    def is_needed(self, data):
        cols = data
        num_cols = _numeric_columns(data)
        cat_cols = list(set(cols) - set(num_cols))
        if len(cat_cols) == 0:
            return False
        return True

    def transform(self, train_x, test_x, train_y):
        cat_trainX, num_trainX = catch_num(train_x, sort=True)
        cat_testX, num_testX = catch_num(test_x, sort=True)
        cat_trainX, cat_testX = cat_trainX.copy(), cat_testX.copy()
        cols = cat_trainX.columns

        for col in cols:
            self.preprocess[col] = LabelEncoder()
            train_arr = self.preprocess[col].fit_transform(cat_trainX[col].astype(str))
            # Re-fitted on the test split, as upstream: the same label can get a
            # different code there. Only the reward of the search sees it.
            test_arr = self.preprocess[col].fit_transform(cat_testX[col].astype(str))
            cat_trainX[col] = train_arr
            cat_testX[col] = test_arr

        cat_trainX = cat_trainX.infer_objects()
        cat_trainX = cat_trainX.iloc[:, ~cat_trainX.columns.duplicated()]
        train_data_x = pd.concat(
            [cat_trainX.reset_index(drop=True), num_trainX.reset_index(drop=True)],
            axis=1,
        ).infer_objects()

        cat_testX = cat_testX.infer_objects()
        cat_testX = cat_testX.iloc[:, ~cat_testX.columns.duplicated()]
        test_data_x = pd.concat(
            [cat_testX.reset_index(drop=True), num_testX.reset_index(drop=True)], axis=1
        ).infer_objects()
        return train_data_x, test_data_x

    def fit_frame(self, X, y):
        # CTX-PORT: the codes are learned once, on the fitted frame, and reused
        # on every other frame; a label never seen there gets the next free code
        # (upstream would re-fit, and silently renumber, instead).
        cat, num = catch_num(X, sort=True)
        self.frame_num_ = list(num.columns)
        taken = set(self.frame_num_)
        self.frame_codes_: Dict[str, Dict[str, int]] = {}
        self.frame_names_: Dict[str, str] = {}
        self.out_sources_ = {}
        for col in cat.columns:
            classes = np.unique(cat[col].astype(str))
            self.frame_codes_[col] = {v: i for i, v in enumerate(classes)}
            self.frame_names_[col] = _unique_name(f"num_{_base_name(col)}", taken)
            self.out_sources_[self.frame_names_[col]] = [col]
        for c in self.frame_num_:
            self.out_sources_[c] = [c]

    def apply_frame(self, X):
        encoded = {}
        for col, codes in self.frame_codes_.items():
            encoded[self.frame_names_[col]] = (
                X[col].astype(str).map(codes).fillna(len(codes)).astype(int).reset_index(drop=True)
            )
        return pd.concat(
            [pd.DataFrame(encoded, index=range(len(X))), X[self.frame_num_].reset_index(drop=True)],
            axis=1,
        )


# ---------------------------------------------------------------------------
# FeaturePreprocessing — env/primitives/fpreprocessing.py
# ---------------------------------------------------------------------------

class _ScalerPrim(Primitive):
    """Shared body of the FeaturePreprocessing components.

    Upstream repeats the same transform() in each of the eight classes; only the
    wrapped scikit-learn estimator changes.
    """

    def __init__(self, name, id_, gid, description):
        super().__init__(name=name)
        self.id = id_
        self.gid = gid
        self.hyperparams = []
        self.type = "FeaturePreprocessing"
        self.description = description
        self.scaler = self.make_scaler()
        self.accept_type = "c_t"
        self.need_y = False

    def make_scaler(self):
        raise NotImplementedError

    def can_accept(self, data):
        return self.can_accept_c(data)

    def is_needed(self, data):
        return True

    def transform(self, train_x, test_x, train_y):
        cat_train_x, num_train_x = catch_num(train_x)
        cat_test_x, num_test_x = catch_num(test_x)

        self.scaler.fit(num_train_x)

        num_train_x = (
            pd.DataFrame(self.scaler.transform(num_train_x), columns=list(num_train_x.columns))
            .reset_index(drop=True)
            .infer_objects()
        )
        train_data_x = pd.concat(
            [cat_train_x.reset_index(drop=True), num_train_x.reset_index(drop=True)],
            axis=1,
        )

        num_test_x = (
            pd.DataFrame(self.scaler.transform(num_test_x), columns=list(num_test_x.columns))
            .reset_index(drop=True)
            .infer_objects()
        )
        test_data_x = pd.concat(
            [cat_test_x.reset_index(drop=True), num_test_x.reset_index(drop=True)],
            axis=1,
        )
        return train_data_x, test_data_x

    def fit_frame(self, X, y):
        cat, num = catch_num(X)
        self.frame_cat_ = list(cat.columns)
        self.frame_num_ = list(num.columns)
        self.frame_scaler_ = self.make_scaler().fit(num)
        self.out_sources_ = {c: [c] for c in self.frame_cat_ + self.frame_num_}

    def apply_frame(self, X):
        num = pd.DataFrame(
            self.frame_scaler_.transform(X[self.frame_num_]), columns=self.frame_num_
        ).infer_objects()
        return pd.concat([X[self.frame_cat_].reset_index(drop=True), num], axis=1)


class MinMaxScalerPrim(_ScalerPrim):
    def __init__(self, random_state=0):
        super().__init__("MinMaxScaler", 1, 9, "Scale each feature to a given range.")

    def make_scaler(self):
        return MinMaxScaler()


class MaxAbsScalerPrim(_ScalerPrim):
    def __init__(self, random_state=0):
        super().__init__("MaxAbsScaler", 2, 10, "Scale each feature by its maximum absolute value.")

    def make_scaler(self):
        return MaxAbsScaler()


class RobustScalerPrim(_ScalerPrim):
    def __init__(self, random_state=0):
        super().__init__("RobustScaler", 3, 11, "Scale features using statistics robust to outliers.")

    def make_scaler(self):
        return RobustScaler()


class StandardScalerPrim(_ScalerPrim):
    def __init__(self, random_state=0):
        super().__init__(
            "StandardScaler", 4, 12,
            "Standardize features by removing the mean and scaling to unit variance",
        )

    def make_scaler(self):
        return StandardScaler()


class QuantileTransformerPrim(_ScalerPrim):
    def __init__(self, random_state=0):
        super().__init__("QuantileTransformer", 5, 13, "Transform features using quantiles information.")

    def make_scaler(self):
        # CTX-PORT: scikit-learn 0.23 (upstream's pin) estimated the quantiles on
        # up to 100,000 rows; 1.5 lowered the default subsample to 10,000.
        return QuantileTransformer(subsample=100_000)


class PowerTransformerPrim(_ScalerPrim):
    def __init__(self, random_state=0):
        super().__init__(
            "PowerTransformer", 6, 14,
            "Apply a power transform featurewise to make data more Gaussian-like.",
        )

    def make_scaler(self):
        return PowerTransformer()


class NormalizerPrim(_ScalerPrim):
    def __init__(self, random_state=0):
        super().__init__("Normalizer", 7, 15, "Normalize samples individually to unit norm.")

    def make_scaler(self):
        return Normalizer()


def _kbins_discretizer() -> KBinsDiscretizer:
    """``KBinsDiscretizer(encode="ordinal")`` as scikit-learn 0.23 ran it.

    CTX-PORT: 0.23 computed the quantile edges on every row with np.percentile's
    linear interpolation. Recent releases subsample to 200,000 rows by default
    and are moving to the averaged-inverted-CDF quantiles; both are pinned back.
    """
    kwargs: Dict[str, Any] = {"encode": "ordinal", "subsample": None}
    if "quantile_method" in inspect.signature(KBinsDiscretizer).parameters:
        kwargs["quantile_method"] = "linear"
    return KBinsDiscretizer(**kwargs)


class KBinsDiscretizerOrdinalPrim(_ScalerPrim):
    def __init__(self, random_state=0):
        super().__init__("KBinsDiscretizerOrdinal", 8, 16, "Bin continuous data into intervals. Ordinal.")
        self.hyperparams_run = {"default": True}
        self.preprocess = None
        self.accept_type = "c_t_kbins"

    def make_scaler(self):
        return None  # built per call, on the columns it is given

    def transform(self, train_x, test_x, train_y):
        cat_train_x, num_train_x = catch_num(train_x)
        cat_test_x, num_test_x = catch_num(test_x)
        self.scaler = ColumnTransformer(
            [("discrit", _kbins_discretizer(), list(num_train_x.columns))]
        )
        self.scaler.fit(num_train_x)

        num_train_x = (
            pd.DataFrame(self.scaler.transform(num_train_x), columns=list(num_train_x.columns))
            .reset_index(drop=True)
            .infer_objects()
        )
        train_data_x = pd.concat(
            [cat_train_x.reset_index(drop=True), num_train_x.reset_index(drop=True)],
            axis=1,
        )

        num_test_x = (
            pd.DataFrame(self.scaler.transform(num_test_x), columns=list(num_test_x.columns))
            .reset_index(drop=True)
            .infer_objects()
        )
        test_data_x = pd.concat(
            [cat_test_x.reset_index(drop=True), num_test_x.reset_index(drop=True)],
            axis=1,
        )
        return train_data_x, test_data_x

    def fit_frame(self, X, y):
        cat, num = catch_num(X)
        self.frame_cat_ = list(cat.columns)
        self.frame_num_ = list(num.columns)
        self.frame_scaler_ = ColumnTransformer(
            [("discrit", _kbins_discretizer(), self.frame_num_)]
        ).fit(num)
        self.out_sources_ = {c: [c] for c in self.frame_cat_ + self.frame_num_}


# ---------------------------------------------------------------------------
# FeatureEngine — env/primitives/fengine.py
# ---------------------------------------------------------------------------

class _FeatureEnginePrim(Primitive):
    """What the FeatureEngine components share in the materialization API.

    Each derives a new set of features from *all* the columns, so every output
    column is computed from every input column (PolynomialFeatures narrows that
    down to the columns each monomial uses).
    """

    frame_prefix = "fe"

    def _frame_estimator(self, X):
        raise NotImplementedError

    def _frame_output(self, est, X) -> np.ndarray:
        return est.transform(X)

    def fit_frame(self, X, y):
        self.frame_in_ = list(X.columns)
        self.frame_est_ = self._frame_estimator(X).fit(X)
        n_out = np.asarray(self._frame_output(self.frame_est_, X.iloc[:1])).shape[1]
        self.frame_out_ = [f"num_{self.frame_prefix}_{i}" for i in range(n_out)]
        self.out_sources_ = {c: list(self.frame_in_) for c in self.frame_out_}

    def apply_frame(self, X):
        values = np.asarray(self._frame_output(self.frame_est_, X[self.frame_in_]))
        return pd.DataFrame(values, columns=self.frame_out_)


class PolynomialFeaturesPrim(_FeatureEnginePrim):
    frame_prefix = "poly"

    def __init__(self, random_state=0):
        super().__init__(name="PolynomialFeatures")
        self.id = 1
        self.gid = 17
        self.hyperparams = []
        self.type = "FeatureEngine"
        self.description = "Generate polynomial and interaction features."
        self.scaler = PolynomialFeatures(include_bias=False)
        self.accept_type = "c_t"
        self.need_y = False

    def can_accept(self, data):
        if data.shape[1] > 100:
            return False
        else:
            return self.can_accept_c(data)

    def is_needed(self, data):
        return True

    def transform(self, train_x, test_x, train_y):
        self.scaler.fit(train_x)

        train_data_x = self.scaler.transform(train_x)
        train_data_x = pd.DataFrame(train_data_x)
        train_data_x = train_data_x.loc[:, ~train_data_x.columns.duplicated()]

        test_data_x = self.scaler.transform(test_x)
        test_data_x = pd.DataFrame(test_data_x)
        test_data_x = test_data_x.loc[:, ~test_data_x.columns.duplicated()]
        return train_data_x, test_data_x

    def _frame_estimator(self, X):
        return copy.deepcopy(self.scaler)

    def fit_frame(self, X, y):
        super().fit_frame(X, y)
        # Each monomial only depends on the columns it raises to a non-zero power.
        powers = self.frame_est_.powers_
        self.out_sources_ = {
            name: [self.frame_in_[j] for j in np.flatnonzero(powers[i])]
            for i, name in enumerate(self.frame_out_)
        }


class InteractionFeaturesPrim(PolynomialFeaturesPrim):
    frame_prefix = "inter"

    def __init__(self, random_state=0):
        super().__init__(random_state)
        self.name = "InteractionFeatures"
        self.description = "Generate interaction features."
        self.id = 2
        self.gid = 18
        self.scaler = PolynomialFeatures(interaction_only=True, include_bias=False)


class _ProjectionPrim(_FeatureEnginePrim):
    """The decomposition components: fitted on the whole frame, and upstream
    names their outputs after the first input columns (cols[:k])."""

    def transform(self, train_x, test_x, train_y):
        cols = list(train_x.columns)
        self.pca.fit(train_x)

        train_data_x = self.pca.transform(train_x)
        train_data_x = pd.DataFrame(train_data_x, columns=cols[: train_data_x.shape[1]])

        test_data_x = self.pca.transform(test_x)
        test_data_x = pd.DataFrame(test_data_x, columns=cols[: test_data_x.shape[1]])
        return train_data_x, test_data_x

    def is_needed(self, data):
        return True

    def _frame_estimator(self, X):
        return copy.deepcopy(self.pca)


class PCA_AUTO_Prim(_ProjectionPrim):
    frame_prefix = "pca"

    def __init__(self, random_state=0):
        super().__init__(name="PCA_AUTO")
        self.id = 3
        self.gid = 19
        self.PCA_AUTO_Prim = []
        self.type = "FeatureEngine"
        self.description = "LAPACK principal component analysis (PCA)."
        self.pca = PCA(svd_solver="auto")  # n_components=0.9
        self.accept_type = "c_t"
        self.need_y = False

    def can_accept(self, data):
        can_num = len(data.columns) > 4
        return self.can_accept_c(data) and can_num


class IncrementalPCA_Prim(_ProjectionPrim):
    frame_prefix = "ipca"

    def __init__(self, random_state=0):
        super().__init__(name="IncrementalPCA")
        self.id = 5
        self.gid = 20
        self.PCA_LAPACK_Prim = []
        self.type = "FeatureEngine"
        self.description = "Incremental principal components analysis (IPCA)."
        self.hyperparams_run = {"default": True}
        self.pca = IncrementalPCA()
        self.accept_type = "c_t"
        self.need_y = False

    def can_accept(self, data):
        return self.can_accept_c(data)


class KernelPCA_Prim(_ProjectionPrim):
    frame_prefix = "kpca"

    def __init__(self, random_state=0):
        super().__init__(name="KernelPCA")
        self.id = 6
        self.gid = 21
        self.PCA_LAPACK_Prim = []
        self.type = "FeatureEngine"
        self.description = "Kernel Principal component analysis (KPCA)."
        self.pca = KernelPCA(n_components=2)  # n_components=5
        self.accept_type = "c_t_krnl"
        self.random_state = random_state
        self.need_y = False

    @staticmethod
    def _eigen_solver(n_samples: int) -> str:
        # CTX-PORT: scikit-learn 0.23 resolved eigen_solver="auto" to ARPACK for
        # n_components < 10 on more than 200 samples; 1.0 moved that case to the
        # randomized solver. The 0.23 rule is pinned back.
        return "arpack" if n_samples > 200 else "dense"

    def can_accept(self, data):
        if data.shape[1] <= 2:
            return False
        if KERNEL_PCA_MAX_ROWS is not None and data.shape[0] > KERNEL_PCA_MAX_ROWS:
            return False
        return self.can_accept_c(data)

    def transform(self, train_x, test_x, train_y):
        self.pca.set_params(eigen_solver=self._eigen_solver(len(train_x)))
        return super().transform(train_x, test_x, train_y)

    def _frame_estimator(self, X):
        est = copy.deepcopy(self.pca)
        return est.set_params(eigen_solver=self._eigen_solver(len(X)))


class TruncatedSVD_Prim(_ProjectionPrim):
    frame_prefix = "tsvd"

    def __init__(self, random_state=0):
        super().__init__(name="TruncatedSVD")
        self.id = 7
        self.gid = 22
        self.PCA_LAPACK_Prim = []
        self.type = "FeatureEngine"
        self.description = "Dimensionality reduction using truncated SVD (aka LSA)."
        self.hyperparams_run = {"default": True}
        self.pca = TruncatedSVD(n_components=2)
        self.accept_type = "c_t_krnl"
        self.need_y = False

    def can_accept(self, data):
        if data.shape[1] <= 2:
            return False
        else:
            return self.can_accept_c(data)


class RandomTreesEmbeddingPrim(_FeatureEnginePrim):
    frame_prefix = "rte"

    def __init__(self, random_state=0):
        super().__init__(name="RandomTreesEmbedding")
        self.id = 8
        self.gid = 23
        self.PCA_LAPACK_Prim = []
        self.type = "FeatureEngine"
        self.description = "FastICA: a fast algorithm for Independent Component Analysis."  # sic
        self.hyperparams_run = {"default": True}
        self.pca = RandomTreesEmbedding(random_state=random_state)
        self.accept_type = "c_t"
        self.need_y = False

    def can_accept(self, data):
        return self.can_accept_c(data)

    def is_needed(self, data):
        return True

    def transform(self, train_x, test_x, train_y):
        self.pca.fit(train_x)

        train_data_x = self.pca.transform(train_x).toarray()
        new_cols = list(map(str, list(range(train_data_x.shape[1]))))
        train_data_x = pd.DataFrame(train_data_x, columns=new_cols)

        test_data_x = self.pca.transform(test_x).toarray()
        new_cols = list(map(str, list(range(test_data_x.shape[1]))))
        test_data_x = pd.DataFrame(test_data_x, columns=new_cols)
        return train_data_x, test_data_x

    def _frame_estimator(self, X):
        return copy.deepcopy(self.pca)

    def _frame_output(self, est, X):
        return est.transform(X).toarray()


# ---------------------------------------------------------------------------
# FeatureSelection — env/primitives/fselection.py
# ---------------------------------------------------------------------------

class VarianceThresholdPrim(Primitive):
    def __init__(self, random_state=0):
        super().__init__(name="VarianceThreshold")
        self.id = 1
        self.gid = 24
        self.PCA_LAPACK_Prim = []
        self.type = "feature selection"
        self.description = "Feature selector that removes all low-variance features."
        self.selector = VarianceThreshold()
        self.accept_type = "c_t"
        self.need_y = True

    def can_accept(self, data):
        return self.can_accept_c(data)

    def is_needed(self, data):
        return True

    def transform(self, train_x, test_x, train_y):
        self.selector.fit(train_x)

        cols = list(train_x.columns)
        mask = self.selector.get_support(indices=False)
        final_cols = list(compress(cols, mask))
        train_data_x = pd.DataFrame(self.selector.transform(train_x), columns=final_cols)

        cols = list(test_x.columns)
        mask = self.selector.get_support(indices=False)
        final_cols = list(compress(cols, mask))
        test_data_x = pd.DataFrame(self.selector.transform(test_x), columns=final_cols)
        return train_data_x, test_data_x

    def fit_frame(self, X, y):
        selector = VarianceThreshold().fit(X)
        self.frame_keep_ = list(compress(list(X.columns), selector.get_support(indices=False)))
        self.out_sources_ = {c: [c] for c in self.frame_keep_}

    def apply_frame(self, X):
        return X[self.frame_keep_].reset_index(drop=True)


# ---------------------------------------------------------------------------
# Predictor and metric — env/primitives/predictor.py, env/metric.py
# ---------------------------------------------------------------------------

def _logistic_regression(n_classes: int):
    """``LogisticRegression(solver="liblinear", random_state=0, n_jobs=5)``.

    CTX-PORT: under scikit-learn 0.23 liblinear handled more than two classes
    one-vs-rest on its own; recent releases deprecate that, so the one-vs-rest
    reduction is made explicit (same binary problems, same argmax of the
    decision functions). n_jobs never had an effect with liblinear.
    """
    model = LogisticRegression(solver="liblinear", random_state=0)
    return OneVsRestClassifier(model) if n_classes > 2 else model


class LogisticRegressionPrim(Primitive):
    """The end model the reward is computed with (comp.selected_prim,
    env.eval_predictor_name). The other 17 predictors of comp.predictors only
    matter through their count, which sizes the predictor embedding."""

    def __init__(self):
        super().__init__(name="LogisticRegression")
        self.hyperparams = []
        self.id = 4
        self.type = "Classifier"
        self.accept_type = "c"

    def can_accept(self, data):
        return self.can_accept_c(data)

    def is_needed(self, data):
        return True

    def transform(self, train_x, train_y, test_x):
        model = _logistic_regression(len(np.unique(train_y)))
        model.fit(train_x, train_y)
        pred_y = model.predict(test_x)
        return pred_y


class AccuracyMetric:
    def __init__(self):
        self.id = 1
        self.type = "Classifier"

    def evaluate(self, pred_y, test_y):
        if pred_y is None or test_y is None:
            return
        if isinstance(pred_y, pd.Series):
            pred_y = pred_y.values
        if isinstance(test_y, pd.Series):
            test_y = test_y.values

        self.score = accuracy_score(test_y, pred_y)
        return self.score


# ---------------------------------------------------------------------------
# Search space — comp.py
# ---------------------------------------------------------------------------

IMPUTERNUMS = [ImputerMean, ImputerMedian, ImputerNumPrim]
ENCODERS = [NumericDataPrim, LabelEncoderPrim, OneHotEncoderPrim]
FPREPROCESSINGS = [
    MinMaxScalerPrim,
    MaxAbsScalerPrim,
    RobustScalerPrim,
    StandardScalerPrim,
    QuantileTransformerPrim,
    PowerTransformerPrim,
    NormalizerPrim,
    KBinsDiscretizerOrdinalPrim,
    Primitive,
]
FENGINES = [
    PolynomialFeaturesPrim,
    InteractionFeaturesPrim,
    PCA_AUTO_Prim,
    IncrementalPCA_Prim,
    KernelPCA_Prim,
    TruncatedSVD_Prim,
    RandomTreesEmbeddingPrim,
    Primitive,
]
FSELECTIONS = [VarianceThresholdPrim, Primitive]

COMPONENTS: Dict[str, list] = {
    "ImputerNum": IMPUTERNUMS,
    "Encoder": ENCODERS,
    "FeaturePreprocessing": FPREPROCESSINGS,
    "FeatureEngine": FENGINES,
    "FeatureSelection": FSELECTIONS,
}

LPIPELINES = [
    ["ImputerNum", "ImputerCat", "Encoder", "FeaturePreprocessing", "FeatureEngine", "FeatureSelection"],
    ["ImputerNum", "ImputerCat", "Encoder", "FeaturePreprocessing", "FeatureSelection", "FeatureEngine"],
    ["ImputerNum", "ImputerCat", "Encoder", "FeatureEngine", "FeatureSelection", "FeaturePreprocessing"],
    ["ImputerNum", "ImputerCat", "Encoder", "FeatureEngine", "FeaturePreprocessing", "FeatureSelection"],
    ["ImputerNum", "ImputerCat", "Encoder", "FeatureSelection", "FeatureEngine", "FeaturePreprocessing"],
    ["ImputerNum", "ImputerCat", "Encoder", "FeatureSelection", "FeaturePreprocessing", "FeatureEngine"],
]
PIPELINE_LENGTH = len(LPIPELINES[0])

NUM_PREDICTORS = 18  # len(comp.predictors)
# One embedding per component instance, plus the blank (25) and "not chosen yet" (26) ids.
PRIM_NUMS = sum(len(v) for v in COMPONENTS.values()) + 1 + 1

DTYPE_ID_MAP = {
    "interval[float64]": 4,
    "uint8": 1,
    "uint16": 1,
    "int64": 1,
    "int": 1,
    "int32": 1,
    "int16": 1,
    "np.int32": 1,
    "np.int64": 1,
    "np.int": 1,
    "np.int16": 1,
    "float64": 2,
    "float": 2,
    "float32": 2,
    "float16": 2,
    "np.float32": 2,
    "np.float64": 2,
    "np.float": 2,
    "np.float16": 2,
    "str": 3,
    "Category": 4,
    "object": 4,
    "bool": 5,
}


def _dtype_id(dtype_name: str) -> int:
    """comp.dtype_id_map lookup.

    CTX-PORT: upstream raises KeyError on a dtype it does not list (int8,
    uint32, ...); those fall back to the id of their family instead.
    """
    if dtype_name in DTYPE_ID_MAP:
        return DTYPE_ID_MAP[dtype_name]
    if "int" in dtype_name:
        return 1
    if "float" in dtype_name:
        return 2
    return 4


def make_primitive(name: str) -> Primitive:
    """A fresh component from its upstream name (as it appears in best_pipeline)."""
    for cls in IMPUTERNUMS + ENCODERS + FPREPROCESSINGS + FENGINES + FSELECTIONS + [ImputerCatPrim]:
        prim = cls()
        if prim.name == name:
            return prim
    raise ValueError(f"Unknown CtxPipe component: {name}")


# ===========================================================================
# Search environment — upstream env/pipeline.py and env/enviroment.py
# ===========================================================================

def _test_value(value) -> float:
    if np.isnan(value) or abs(value) == np.inf:
        return 0.0
    else:
        return value


def _test_frexp(value) -> Tuple[float, int]:
    if np.isnan(value) or abs(value) == np.inf:
        return 0.0, 0
    else:
        return np.frexp(value)


class SearchPipeline:
    """The pipeline under construction (env/pipeline.py Pipeline).

    CTX-PORT: upstream reads the dataset from a CSV and executes every component
    and the final evaluation in a subprocess (to enforce a timeout during
    training, and to kill leftover joblib workers). Here the frame is passed in
    and everything runs in-process; see note 4 of the module docstring for how
    an exception is handled.
    """

    def __init__(self, data_x: pd.DataFrame, data_y: np.ndarray, predictor, metric,
                 ratio: float = SEARCH_TRAIN_SIZE, split_random_state: int = SEARCH_SPLIT_SEED):
        self.metric = metric
        self.predictor = predictor

        self.result = 0
        self.sequence: List[Primitive] = []
        self.index = 0

        self.num_cols: list = []
        self.cat_cols: list = []

        self.load_data(data_x, data_y, ratio, split_random_state)
        self._logic_pipeline_id: Optional[int] = None
        self.gsequence = [26, 26, 26, 26, 26, 26]

    @property
    def logic_pipeline_id(self) -> int:
        if self._logic_pipeline_id is None:
            raise ValueError("self.logic_pipeline_id not initialized")
        return self._logic_pipeline_id

    @logic_pipeline_id.setter
    def logic_pipeline_id(self, value) -> None:
        self._logic_pipeline_id = value

    def load_data(self, data_x, data_y, ratio, split_random_state):
        data_x = data_x.replace([np.inf, -np.inf], np.nan)
        self.data_x = data_x
        self.data_y = np.asarray(data_y)

        # Upstream truncates to 1,500 rows only while training.
        self.train_x, self.test_x, self.train_y, self.test_y = train_test_split(
            self.data_x,
            self.data_y,
            train_size=ratio,
            test_size=1 - ratio,
            random_state=split_random_state,
        )

        self.num_cols = list(_numeric_columns(self.train_x))
        self.cat_cols = list(set(self.train_x) - set(self.num_cols))

    def get_index(self):
        return self.index

    def add_step(self, step: Primitive) -> int:
        if self.index >= len(LPIPELINES[self.logic_pipeline_id]):
            return -1

        # Upstream also rejects `step.type in pre_pipeline`, a list of ints that
        # a type string never belongs to.
        if (
            not step.can_accept(self.train_x)
            or not step.can_accept(self.test_x)
            or (not step.is_needed(self.train_x) and not step.is_needed(self.test_x))
        ):
            return 0

        try:
            train_x, test_x = step.transform(self.train_x, self.test_x, self.train_y)
            num_cols = list(_numeric_columns(train_x))
            cat_cols = list(set(train_x.columns) - set(num_cols))
        except Exception as exc:
            logger.debug(f"    component {step.name} raised {type(exc).__name__}: {exc}")
            return -1

        self.train_x, self.test_x, self.num_cols, self.cat_cols = train_x, test_x, num_cols, cat_cols
        self.sequence.append(step)
        self.gsequence[self.index] = step.gid

        self.index += 1
        return 1

    def evaluate(self):
        if len(self.sequence) < PIPELINE_LENGTH:
            return

        try:
            pred_y = self.predictor.transform(self.train_x, self.train_y, self.test_x)
            self.result = self.metric.evaluate(pred_y, self.test_y)
        except Exception as exc:
            logger.debug(f"    evaluating {self.sequence} failed: {exc}")
            self.result = -1

        return self.result


class Environment:
    """The state the agents read (env/enviroment.py Environment)."""

    def __init__(self, data_x: pd.DataFrame, data_y: np.ndarray,
                 train_size: float = SEARCH_TRAIN_SIZE, split_seed: int = SEARCH_SPLIT_SEED):
        self.column_num = COLUMN_NUM
        self.pipeline = SearchPipeline(
            data_x, data_y, LogisticRegressionPrim(), AccuracyMetric(),
            ratio=train_size, split_random_state=split_seed,
        )
        self.reward = None
        self.done = False
        self.lpip_state = self.get_lpip_state()

    def step(self, step: Primitive) -> Optional[Tuple[np.ndarray, float, bool]]:
        """Execute a component; None when it was rejected or failed.

        Upstream also computes the state *before* the step, which only feeds the
        replay buffer during training.
        """
        step_result = self.pipeline.add_step(step)
        if step_result <= 0:
            return None

        next_prim_state = self.get_state()
        self.get_reward()
        self.set_done()
        return next_prim_state, self.reward, self.done

    def get_data_feature(self) -> np.ndarray:
        inp_data = pd.DataFrame(self.pipeline.train_x)

        column_info = {}

        for i in range(len(inp_data.columns)):
            col = inp_data.iloc[:, i]
            if i >= self.column_num:
                break
            s_s = col

            column_info[i] = {}
            column_info[i]["col_name"] = "unknown_" + str(i)
            column_info[i]["dtype"] = str(s_s.dtypes)  # 1
            column_info[i]["length"], column_info[i]["length_exp"] = _test_frexp(len(s_s.values))  # 2
            column_info[i]["null_ratio"] = s_s.isnull().sum() / len(s_s.values)  # 3
            column_info[i]["ctype"] = 1 if inp_data.columns[i] in self.pipeline.num_cols else 2
            column_info[i]["nunique"], column_info[i]["nunique_exp"] = _test_frexp(s_s.nunique())  # 5
            column_info[i]["nunique_ratio"] = s_s.nunique() / len(s_s.values)  # 6

            d = s_s.describe()

            if "mean" not in d:
                column_info[i]["ctype"] = 2

            if column_info[i]["ctype"] == 1:  # numeric
                column_info[i]["mean"], column_info[i]["mean_exp"] = _test_frexp(d["mean"])  # 7
                column_info[i]["std"], column_info[i]["std_exp"] = _test_frexp(d["std"])  # 8
                column_info[i]["min"], column_info[i]["min_exp"] = _test_frexp(d["min"])  # 9
                column_info[i]["25%"], column_info[i]["25%_exp"] = _test_frexp(d["25%"])
                column_info[i]["50%"], column_info[i]["50%_exp"] = _test_frexp(d["50%"])
                column_info[i]["75%"], column_info[i]["75%_exp"] = _test_frexp(d["75%"])
                column_info[i]["max"], column_info[i]["max_exp"] = _test_frexp(d["max"])
                column_info[i]["median"], column_info[i]["median_exp"] = _test_frexp(s_s.median())

                if len(s_s.mode()) != 0:
                    column_info[i]["mode"], column_info[i]["mode_exp"] = _test_frexp(s_s.mode().iloc[0])
                else:
                    column_info[i]["mode"], column_info[i]["mode_exp"] = 0.0, 0

                mr = s_s.astype("category").describe().iloc[3] / len(s_s.values)
                column_info[i]["mode_ratio"] = _test_value(mr)

                column_info[i]["sum"], column_info[i]["sum_exp"] = _test_frexp(s_s.sum())
                column_info[i]["skew"], column_info[i]["skew_exp"] = _test_frexp(s_s.skew())
                column_info[i]["kurt"], column_info[i]["kurt_exp"] = _test_frexp(s_s.kurt())

            elif column_info[i]["ctype"] == 2:  # category
                column_info[i]["mean"], column_info[i]["mean_exp"] = 0.0, 0
                column_info[i]["std"], column_info[i]["std_exp"] = 0.0, 0
                column_info[i]["min"], column_info[i]["min_exp"] = 0.0, 0
                column_info[i]["25%"], column_info[i]["25%_exp"] = 0.0, 0
                column_info[i]["50%"], column_info[i]["50%_exp"] = 0.0, 0
                column_info[i]["75%"], column_info[i]["75%_exp"] = 0.0, 0
                column_info[i]["max"], column_info[i]["max_exp"] = 0.0, 0
                column_info[i]["median"], column_info[i]["median_exp"] = 0.0, 0

                column_info[i]["mode"], column_info[i]["mode_exp"] = 0.0, 0
                column_info[i]["mode_ratio"] = 0.0
                column_info[i]["sum"], column_info[i]["sum_exp"] = 0.0, 0
                column_info[i]["skew"], column_info[i]["skew_exp"] = 0.0, 0
                column_info[i]["kurt"], column_info[i]["kurt_exp"] = 0.0, 0

        data_feature = []
        for index in column_info.keys():
            one_column_feature = []
            column_dic = column_info[index]
            for kw in column_dic.keys():
                if kw == "col_name" or kw == "content":
                    continue
                elif kw == "dtype":
                    content = _dtype_id(column_dic[kw])
                else:
                    content = column_dic[kw]
                one_column_feature.append(content)
            data_feature.append(one_column_feature)

        if len(column_info) < self.column_num:
            for index in range(len(column_info), self.column_num):
                one_column_feature = np.zeros(COLUMN_FEATURE_DIM)
                data_feature.append(one_column_feature)

        return np.ravel(np.array(data_feature, dtype=float))

    def get_lpip_state(self) -> np.ndarray:
        data_feature = self.get_data_feature()
        predictor = np.array([self.pipeline.predictor.id])
        return np.concatenate((data_feature, predictor))

    def get_state(self) -> np.ndarray:
        data_feature = self.get_data_feature()
        sequence = np.array(self.pipeline.gsequence)
        predictor = np.array([self.pipeline.predictor.id - 1])
        logic_pipeline_id = np.array([self.pipeline.logic_pipeline_id])
        return np.concatenate((data_feature, sequence, predictor, logic_pipeline_id))

    def get_reward(self):
        if len(self.pipeline.sequence) < PIPELINE_LENGTH:
            self.reward = 0.0 if self.pipeline.sequence[-1].id == 0 else BLANK_REWARD
        else:
            self.reward = self.pipeline.evaluate()

    def set_done(self):
        if len(self.pipeline.sequence) < PIPELINE_LENGTH:
            self.done = False
        elif len(self.pipeline.sequence) == PIPELINE_LENGTH:
            self.done = True

    def has_nan(self) -> Tuple[bool, bool]:
        has_num_nan = False
        has_cat_nan = False

        def catch(data):
            num_cols = [col for col in data.columns if str(data[col].dtypes) != "object"]
            cat_cols = [col for col in data.columns if col not in num_cols]
            return data[cat_cols], data[num_cols]

        cat_train_x, num_train_x = catch(self.pipeline.train_x)
        cat_test_x, num_test_x = catch(self.pipeline.test_x)
        if len(self.pipeline.cat_cols) != 0:
            if _has_na_or_inf(cat_train_x):
                has_cat_nan = True
            if _has_na_or_inf(cat_test_x):
                has_cat_nan = True
        if len(self.pipeline.num_cols) != 0:
            if _has_na_or_inf(num_train_x):
                has_num_nan = True
            if _has_na_or_inf(num_test_x):
                has_num_nan = True

        return has_num_nan, has_cat_nan

    def has_cat_cols(self) -> bool:
        return len(self.pipeline.cat_cols) != 0


# ===========================================================================
# Context embedding — upstream ctxpipe/ctx.py
# ===========================================================================

class GTEEmbedder:
    """Text embedder of the context plug-in (ctx.GTEEmbedder): average-pooled
    last hidden states of GTE-large over the first 512 tokens of the CSV text."""

    def __init__(self, model_path: str = CTX_MODEL_PATH, device: str = "cpu",
                 max_length: int = CTX_MAX_LENGTH) -> None:
        try:
            from transformers import AutoModel, AutoTokenizer
        except ImportError as exc:  # pragma: no cover - dependency hint
            raise ImportError(
                "CtxPipe needs the `transformers` package for its GTE-large context "
                "embedder: pip install transformers"
            ) from exc

        self.model_path = model_path
        self.device = torch.device(device)
        self.max_length = max_length
        self._ctx_tokenizer = AutoTokenizer.from_pretrained(model_path)
        self._ctx_model = AutoModel.from_pretrained(model_path).to(self.device)
        self._ctx_model.eval()
        self.n_calls = 0
        self.total_time_s = 0.0

    def embed(self, x: str) -> torch.Tensor:
        # CTX-PORT: upstream runs the forward pass with autograd on and detaches
        # the result; no_grad gives the same embedding without the graph.
        t0 = time.perf_counter()
        with torch.no_grad():
            ctx_dict = self._ctx_tokenizer(
                [x],
                max_length=self.max_length,
                padding=True,
                truncation=True,
                return_tensors="pt",
            ).to(self.device)
            output = self._ctx_model(**ctx_dict)
            attn_mask: torch.Tensor = ctx_dict["attention_mask"]
            embeddings = self._average_pool(output.last_hidden_state, attn_mask).detach().cpu()
        self.n_calls += 1
        self.total_time_s += time.perf_counter() - t0
        return embeddings.squeeze(dim=0)  # 1024

    @staticmethod
    def _average_pool(last_hidden_states: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        last_hidden = last_hidden_states.masked_fill(~attention_mask[..., None].bool(), 0.0)
        return last_hidden.sum(dim=1) / attention_mask.sum(dim=1)[..., None]


_EMBEDDERS: Dict[Tuple[str, str], GTEEmbedder] = {}


def get_embedder(model_path: str, device: str) -> GTEEmbedder:
    """One embedder per process: upstream builds it once, at import time."""
    key = (model_path, str(device))
    if key not in _EMBEDDERS:
        logger.info(f"Loading context embedding model {model_path} on {device}")
        _EMBEDDERS[key] = GTEEmbedder(model_path, device)
    return _EMBEDDERS[key]


# ===========================================================================
# Agents — upstream ctxpipe/agent/model.py and ctxpipe/agent/dqn.py
# ===========================================================================

class ForwardMode(Enum):
    CLOSED = auto()
    GATED = auto()
    OPEN = auto()


def make_ctx_plugin_model(n_input: int, n_output: int) -> nn.Module:
    """The context integrator C (Section 5.2): a two-layer MLP ending in tanh."""
    return nn.Sequential(
        nn.Linear(n_input, N_DIM_FIRST),
        nn.LeakyReLU(),
        nn.Linear(N_DIM_FIRST, n_output),
        nn.Tanh(),
    )


def make_mm_layer(shape: tuple, device: torch.device) -> torch.Tensor:
    """A gate matrix (W_c) or bias (W_b): a plain tensor drawn from
    N(3/d, (1/d)^2), d = shape[0] — never registered as a parameter upstream."""
    mean = 3 / shape[0]
    result = torch.normal(mean=mean, std=mean / 3, size=shape).to(device)
    result.requires_grad_(True)
    return result


def forward_context_gate(model, x: torch.Tensor, ctx: torch.Tensor) -> torch.Tensor:
    """Equation 1: alpha = sigmoid([C(e); Q^(k)] W_c + W_b)."""
    info = torch.concat([x, ctx], dim=1)
    info = torch.matmul(info, model.context_gate)
    info = info + model.context_gate_bias
    info = torch.nan_to_num(info)
    return torch.sigmoid(info)


def _mlp_with_context(model, input_feature: torch.Tensor, ctx: torch.Tensor,
                      mode: ForwardMode, device: torch.device) -> torch.Tensor:
    """The MLP of the main DQN with the context plug-in spliced in.

    Shared by DQN.forward and RnnDQN.forward, which repeat this loop upstream.
    The gate reads the output of the first linear layer (k = 1, Section 6) and
    the gated context rescales the tanh output of the last one.
    """
    for i in range(len(model.nn)):
        input_feature = model.nn[i](input_feature)

        if i == len(model.nn) - INFO_EXTRACTION_POS:
            ctx_integration = model.ctx_linear(ctx)

            if mode == ForwardMode.CLOSED:
                ctx_gate = torch.zeros(ctx_integration.shape).to(device)
            elif mode == ForwardMode.GATED:
                ctx_gate = forward_context_gate(model, input_feature, ctx_integration)
            elif mode == ForwardMode.OPEN:
                ctx_gate = torch.ones(ctx_integration.shape).to(device)
            else:
                raise ValueError(f"No such mode: {mode}")

            # CTX-PORT (note 3): the gate is applied here and once more below.
            ctx_integration = torch.mul(ctx_integration, ctx_gate)

        if i == len(model.nn) - CTX_INTEGRATION_POS:
            ctx_integration = torch.mul(ctx_integration - 1.0, ctx_gate) + 1.0
            model.last_ctx_scale_ = ctx_integration.detach()
            input_feature = input_feature * ctx_integration

    return input_feature


def _main_mlp(num_inputs: int, actions_dim: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(num_inputs, N_DIM_FIRST),
        nn.LeakyReLU(),
        nn.Linear(N_DIM_FIRST, N_DIM_FIRST),
        nn.LeakyReLU(),
        nn.Linear(N_DIM_FIRST, 64),
        nn.LeakyReLU(),
        nn.Linear(64, 32),
        nn.LeakyReLU(),
        nn.Linear(32, actions_dim),
        nn.Tanh(),
    )


class DQN(nn.Module):
    """The logical-pipeline agent: statistical features + predictor id."""

    def __init__(self, num_inputs: int, actions_dim: int, device: torch.device):
        super().__init__()
        self.device = device
        self.nn = _main_mlp(num_inputs, actions_dim)
        self.ctx_linear = make_ctx_plugin_model(n_input=N_DIM_EMBED, n_output=actions_dim)
        self.context_gate = make_mm_layer((actions_dim + N_DIM_FIRST, actions_dim), device=device)
        self.context_gate_bias = make_mm_layer((actions_dim,), device=device)

    def forward(self, x: torch.Tensor, ctx: torch.Tensor, mode: ForwardMode = ForwardMode.GATED):
        data_feature = x[:, :DATA_DIM].to(self.device)
        x = torch.concat([data_feature, x[:, DATA_DIM:]], dim=-1)
        return _mlp_with_context(self, x, ctx, mode, self.device)


class RnnDQN(nn.Module):
    """A component agent (Figure 5): statistical features, the component
    history through an LSTM, and the embeddings of the model and of the
    logical pipeline."""

    def __init__(self, actions_dim: int, device: torch.device):
        super().__init__()
        self.device = device
        self.data_feature_dim = DATA_DIM
        self.seq_feature_dim = PIPELINE_LENGTH

        # Upstream constructs these in this order; it fixes the random draws the
        # gate matrices get (they are the only weights not loaded from disk).
        self.seq_embedding = nn.Embedding(PRIM_NUMS, SEQ_EMBEDDING_DIM)
        self.seq_lstm = nn.LSTM(
            input_size=SEQ_EMBEDDING_DIM,
            hidden_size=SEQ_HIDDEN_SIZE,
            num_layers=SEQ_NUM_LAYERS,
            bias=True,
            batch_first=True,
            bidirectional=False,
        )
        self.predictor_embedding = nn.Embedding(NUM_PREDICTORS, PREDICTOR_EMBEDDING_DIM)
        self.lpipeline_embedding = nn.Embedding(len(LPIPELINES), LPIPELINE_EMBEDDING_DIM)

        self.nn = _main_mlp(
            self.data_feature_dim
            + SEQ_HIDDEN_SIZE * self.seq_feature_dim
            + PREDICTOR_EMBEDDING_DIM
            + LPIPELINE_EMBEDDING_DIM,
            actions_dim,
        )
        self.ctx_linear = make_ctx_plugin_model(n_input=N_DIM_EMBED, n_output=actions_dim)
        self.context_gate = make_mm_layer((actions_dim + N_DIM_FIRST, actions_dim), device=device)
        self.context_gate_bias = make_mm_layer((actions_dim,), device=device)

    def forward(self, x, ctx: torch.Tensor, mode: ForwardMode = ForwardMode.GATED):
        device = self.device
        d, s = self.data_feature_dim, self.seq_feature_dim

        data_feature = x[:, :d].to(device)                                 # (batch, data_dim)
        seq_feature = x[:, d : d + s].type(torch.LongTensor).to(device)     # (batch, 6)
        predictor_feature = x[:, d + s : d + s + 1].type(torch.LongTensor).to(device)
        lpipeline_feature = x[:, d + s + 1 : d + s + 2].type(torch.LongTensor).to(device)

        seq_embed_feature = self.seq_embedding(seq_feature)                # (batch, 6, emb)
        seq_hidden_feature, _ = self.seq_lstm(seq_embed_feature)          # (batch, 6, hidden)
        seq_hidden_feature = torch.flatten(seq_hidden_feature, start_dim=1)

        predictor_embed_feature = torch.flatten(self.predictor_embedding(predictor_feature), start_dim=1)
        lpipeline_embed_feature = torch.flatten(self.lpipeline_embedding(lpipeline_feature), start_dim=1)

        input_feature = torch.cat(
            (data_feature, seq_hidden_feature, predictor_embed_feature, lpipeline_embed_feature),
            1,
        )
        return _mlp_with_context(self, input_feature, ctx, mode, device)


class Agent:
    """The six agents of CtxPipe (agent/dqn.py Agent), inference only.

    At inference upstream sets ``no_random = True``, so every action is the
    greedy one and the epsilon schedule, the replay buffer and the optimizers
    never come into play.
    """

    MODEL_FILES = {
        "ImputerNum": "imputernum_model.pkl",
        "Encoder": "encoder_model.pkl",
        "FeaturePreprocessing": "fpreprocessing_model.pkl",
        "FeatureEngine": "fengine_model.pkl",
        "FeatureSelection": "fselection_model.pkl",
        "LogicPipeline": "logical_pipeline.pkl",
    }

    def __init__(self, embedder: GTEEmbedder, device: torch.device, n_ctx_rows: int = N_CTX_ROWS):
        self.embedder = embedder
        self.device = device
        self.n_ctx_rows = n_ctx_rows
        # Same construction order as upstream (see RnnDQN.__init__).
        self.models: Dict[str, nn.Module] = {
            "ImputerNum": RnnDQN(len(IMPUTERNUMS), device),
            "Encoder": RnnDQN(len(ENCODERS), device),
            "FeaturePreprocessing": RnnDQN(len(FPREPROCESSINGS), device),
            "FeatureEngine": RnnDQN(len(FENGINES), device),
            "FeatureSelection": RnnDQN(len(FSELECTIONS), device),
            "LogicPipeline": DQN(DATA_DIM + 1, len(LPIPELINES), device),
        }
        for model in self.models.values():
            model.to(device)
            model.eval()

    @staticmethod
    def action_dim(index: str) -> int:
        return len(LPIPELINES) if index == "LogicPipeline" else len(COMPONENTS[index])

    def load_weights(self, weights_dir: Path, tag: str = WEIGHTS_TAG) -> None:
        # CTX-PORT: upstream logs a warning and carries on with random weights
        # when a file is missing; a baseline run on untrained agents would be
        # meaningless, so it is an error here.
        for index, fname in self.MODEL_FILES.items():
            path = Path(weights_dir) / (f"{tag}_{fname}" if tag else fname)
            if not path.exists():
                raise FileNotFoundError(f"CtxPipe weight file not found: {path}")
            state_dict = torch.load(path, map_location=self.device, weights_only=True)
            self.models[index].load_state_dict(state_dict)

    def context_text(self, train_x: pd.DataFrame) -> str:
        """The rows the context plug-in reads, serialized to CSV with the header."""
        if len(train_x.index) > self.n_ctx_rows:
            return train_x.sample(n=self.n_ctx_rows).to_csv(index=False)
        return train_x.to_csv(index=False)

    def act(self, train_x: pd.DataFrame, state: np.ndarray, index: str,
            tryed_list=()) -> Tuple[int, Dict[str, Any]]:
        """Greedy action of agent ``index`` among the ones not tried yet."""
        model = self.models[index]
        action_dim = self.action_dim(index)

        state_t = torch.tensor(np.asarray(state, dtype=float), dtype=torch.float).unsqueeze(0).to(self.device)
        ctx_embeddings = self.embedder.embed(self.context_text(train_x)).unsqueeze(dim=0).to(self.device)

        with torch.no_grad():
            q_value = model(state_t, ctx_embeddings).cpu()

        action_index_list = [i for i in range(action_dim) if i not in tryed_list]
        if not action_index_list:
            # Upstream fails the same way (argmax of an empty array).
            raise RuntimeError(f"agent {index} has no untried action left")
        q_value_temp = np.array([float(q_value[0][i]) for i in action_index_list])
        action = int(action_index_list[int(q_value_temp.argmax())])

        diag = {
            "q_values": [round(float(v), 6) for v in q_value[0]],
            "ctx_scale": [round(float(v), 6) for v in model.last_ctx_scale_[0].cpu()],
        }
        return action, diag


# ===========================================================================
# Inference — upstream ctxpipe/tester.py (Tester.inference)
# ===========================================================================

class CtxPipeSearch:
    """Build the pipeline of one dataset, one component at a time."""

    def __init__(self, agent: Agent, env: Environment):
        self.agent = agent
        self.env = env

    def _select(self, component: str, state, tryed_list, has_num_nan, has_cat_nan
                ) -> Tuple[int, Primitive, Optional[Dict]]:
        env = self.env
        train_x = env.pipeline.train_x
        diag = None

        if component == "ImputerNum":
            if has_num_nan:
                action, diag = self.agent.act(train_x, state, component, tryed_list)
                step = IMPUTERNUMS[action]()
            else:
                action = len(IMPUTERNUMS)
                step = Primitive()

        elif component == "ImputerCat":
            action = -1
            step = ImputerCatPrim() if has_cat_nan else Primitive()

        elif component == "Encoder":
            if env.has_cat_cols():
                action, diag = self.agent.act(train_x, state, component, tryed_list)
                step = ENCODERS[action]()
            else:
                action = len(ENCODERS)
                step = Primitive()

        else:  # FeaturePreprocessing / FeatureEngine / FeatureSelection
            action, diag = self.agent.act(train_x, state, component, tryed_list)
            step = COMPONENTS[component][action]()

        return action, step, diag

    def _one_step(self, state) -> Tuple[np.ndarray, float, bool, Dict]:
        """get_five_items_from_pipeline: choose, execute, retry until accepted."""
        env = self.env
        tryed_list: List[int] = []
        has_num_nan, has_cat_nan = env.has_nan()

        component = LPIPELINES[env.pipeline.logic_pipeline_id][env.pipeline.get_index()]
        action, step, diag = self._select(component, state, tryed_list, has_num_nan, has_cat_nan)
        step_result = env.step(step)
        tryed_list.append(action)
        rejected = [] if step_result is not None else [step.name]

        while step_result is None:
            action, step, diag = self._select(component, state, tryed_list, has_num_nan, has_cat_nan)
            if action in tryed_list:
                # Only the components no agent chooses can repeat an action;
                # upstream would spin here forever.
                raise RuntimeError(f"{component} component {step.name} cannot be executed")
            tryed_list.append(action)
            step_result = env.step(step)
            if step_result is None:
                rejected.append(step.name)

        next_state, reward, done = step_result
        log = {
            "component": component,
            "primitive": step.name,
            "action": int(action),
            "rejected": rejected,
            "shape_after": [int(s) for s in env.pipeline.train_x.shape],
        }
        if diag is not None:
            log.update(diag)
        logger.debug(
            f"    {component}: {step.name}"
            + (f" (rejected: {', '.join(rejected)})" if rejected else "")
            + (f" | Q={diag['q_values']} ctx_scale={diag['ctx_scale']}" if diag else "")
        )
        return next_state, reward, done, log

    def run(self) -> Dict[str, Any]:
        env = self.env
        lp_id, lp_diag = self.agent.act(env.pipeline.train_x, env.lpip_state, "LogicPipeline")
        env.pipeline.logic_pipeline_id = lp_id
        logger.debug(
            f"    logical pipeline {lp_id}: {' → '.join(LPIPELINES[lp_id])} "
            f"| Q={lp_diag['q_values']} ctx_scale={lp_diag['ctx_scale']}"
        )

        state = env.get_state()
        steps_log = []
        reward, done = None, False
        # CTX-PORT: upstream loops six times and, on the last step, resets the
        # environment and picks a logical pipeline for a next episode that never
        # runs; that dead work is skipped.
        while not done:
            state, reward, done, log = self._one_step(state)
            steps_log.append(log)

        if reward is None:
            raise ValueError("Invalid reward")

        return {
            "logical_pipeline_id": int(lp_id),
            "logical_pipeline": list(LPIPELINES[lp_id]),
            "logical_q_values": lp_diag["q_values"],
            "sequence": [s.name for s in env.pipeline.sequence],
            "score": float(reward),
            "steps": steps_log,
            "search_train_rows": int(len(env.pipeline.train_x)),
            "search_test_rows": int(len(env.pipeline.test_x)),
            "n_features_search_out": int(env.pipeline.train_x.shape[1]),
        }


# ===========================================================================
# Discrete materialization of the selected pipeline
# ===========================================================================

class MaterializedPipeline:
    """The physical pipeline CtxPipe selected, fitted on one frame and replayed
    on others (note 1 of the module docstring).

    ``lineage_`` maps every output column to the input columns it is computed
    from; ``skipped_`` lists the components that could not be applied to the
    full frame (they are left out for every frame alike).
    """

    def __init__(self, primitive_names: List[str]):
        self.primitive_names = list(primitive_names)
        self.ops_: List[Optional[Primitive]] = []
        self.lineage_: Dict[str, List[str]] = {}
        self.skipped_: List[Dict[str, str]] = []

    def fit_transform(self, X: pd.DataFrame, y) -> pd.DataFrame:
        cur = X.reset_index(drop=True)
        lineage = {c: [c] for c in cur.columns}
        self.ops_, self.skipped_ = [], []

        for name in self.primitive_names:
            op = make_primitive(name)
            if not op.can_accept(cur):
                logger.warning(f"  [CtxPipe] {name} does not accept the full frame; left out")
                self.skipped_.append({"primitive": name, "reason": "can_accept"})
                self.ops_.append(None)
                continue
            try:
                op.fit_frame(cur, y)
                nxt = op.apply_frame(cur)
            except Exception as exc:
                logger.warning(f"  [CtxPipe] {name} failed on the full frame ({exc}); left out")
                self.skipped_.append({"primitive": name, "reason": f"{type(exc).__name__}: {exc}"})
                self.ops_.append(None)
                continue

            lineage = {
                out: list(dict.fromkeys(src for c in srcs for src in lineage.get(c, [c])))
                for out, srcs in op.out_sources_.items()
            }
            cur = nxt
            self.ops_.append(op)

        self.lineage_ = lineage
        return cur

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        cur = X.reset_index(drop=True)
        for op in self.ops_:
            if op is not None:
                cur = op.apply_frame(cur)
        return cur

    def describe(self) -> str:
        return " → ".join(self.primitive_names) if self.primitive_names else "(empty)"


def build_residual_mask(df_clean: pd.DataFrame, mask_df: pd.DataFrame,
                        lineage: Dict[str, List[str]]) -> pd.DataFrame:
    """True where a cell is still missing and comes from a poisoned input cell.

    A derived feature counts as poisoned in a row when any input column it is
    computed from was poisoned there. Only columns that still hold a missing
    value can have a True cell, so only those are looked at.
    """
    residual = pd.DataFrame(
        np.zeros(df_clean.shape, dtype=bool), index=df_clean.index, columns=df_clean.columns
    )
    na = df_clean.isna()
    for col in df_clean.columns[na.any(axis=0).to_numpy()]:
        sources = [s for s in lineage.get(col, [col]) if s in mask_df.columns]
        if not sources:
            continue
        poisoned = mask_df[sources].to_numpy().any(axis=1)
        residual[col] = poisoned & na[col].to_numpy()
    return residual


# ===========================================================================
# quAIL glue: column types
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


_INT_LIKE = re.compile(r"^-?\d+\.0+$")


def _canonical_category(value):
    """One spelling per category label, whatever dtype pandas parsed it with.

    An integer-coded categorical column is read as float once poisoning has put
    a missing value in it (1 -> "1.0"), and as int in the clean hold-out
    (1 -> "1"); without this the two frames would not share a single category.
    """
    if pd.isna(value):
        return np.nan
    if isinstance(value, (float, np.floating)) and float(value).is_integer():
        return str(int(value))
    text = str(value)
    if _INT_LIKE.match(text):
        return text.split(".")[0]
    return text


def _coerce_types(df: pd.DataFrame, col_types: Dict[str, list]) -> pd.DataFrame:
    """Make the dtypes match the prefixes.

    CTX-PORT: upstream infers which columns are categorical from the dtype
    pandas reads the CSV with, so an integer-coded categorical column would be
    treated as numerical. Here the cat_* prefix decides, as in the other ports:
    cat_* is cast to str (canonical spelling, see _canonical_category) and num_*
    to numeric.
    """
    out = df.copy()
    for col in col_types["numerical"]:
        out[col] = pd.to_numeric(out[col], errors="coerce")
    for col in col_types["categorical"]:
        out[col] = out[col].map(_canonical_category).astype(object)
    return out


def _search_frame(df: pd.DataFrame, feature_cols: List[str]) -> Tuple[pd.DataFrame, Dict[str, str]]:
    """The features as upstream would read them from the original CSV.

    CTX-PORT: the column names are part of the context the embedding model
    reads (Figure 2), and the num_/cat_ prefixes are this repository's
    convention, not the dataset's. They are stripped for the search — unless
    that would make two names collide — and never leak into the output.
    """
    bases = [_base_name(c) for c in feature_cols]
    counts = pd.Series(bases).value_counts()
    rename = {c: (b if counts[b] == 1 else c) for c, b in zip(feature_cols, bases)}
    return df[feature_cols].rename(columns=rename).reset_index(drop=True), rename


# ===========================================================================
# Driver: CtxPipe applied to one poisoned dataset
# ===========================================================================

def _resolve_device(device: str) -> torch.device:
    if device == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    if device.startswith("cuda") and not torch.cuda.is_available():
        logger.warning(f"Device {device} requested but CUDA is not available; using cpu")
        return torch.device("cpu")
    if device == "mps" and not (getattr(torch.backends, "mps", None) is not None
                                and torch.backends.mps.is_available()):
        logger.warning("Device mps requested but not available; using cpu")
        return torch.device("cpu")
    return torch.device(device)


class CtxPipe:
    """CtxPipe baseline driver.

    Exposes the same interface as the other data-preparation baselines of this
    repository (``prepare`` -> cleaned frame, residual mask, performance
    metrics, reusable pipeline description).
    """

    def __init__(
        self,
        weights_dir: str = str(DEFAULT_WEIGHTS_DIR),
        weights_tag: str = WEIGHTS_TAG,
        embedding_model: str = CTX_MODEL_PATH,
        device: str = "cpu",
        n_ctx_rows: int = N_CTX_ROWS,
        seed: int = RANDOM_SEED,
        split_seed: int = SEARCH_SPLIT_SEED,
        search_train_size: float = SEARCH_TRAIN_SIZE,
        n_threads: Optional[int] = None,
        verbose: bool = False,
    ):
        self.weights_dir = Path(weights_dir)
        self.weights_tag = weights_tag
        self.embedding_model = embedding_model
        self.device = _resolve_device(device)
        self.n_ctx_rows = int(n_ctx_rows)
        self.seed = int(seed)
        self.split_seed = int(split_seed)
        self.search_train_size = float(search_train_size)
        self.verbose = verbose
        if n_threads:
            torch.set_num_threads(int(n_threads))

        for fname in Agent.MODEL_FILES.values():
            path = self.weights_dir / (f"{weights_tag}_{fname}" if weights_tag else fname)
            if not path.exists():
                raise FileNotFoundError(f"CtxPipe weight file not found: {path}")

        # Loaded once, outside the per-dataset timings (upstream builds it at
        # import time): only the embeddings themselves count as search work.
        t0 = time.perf_counter()
        self.embedder = get_embedder(embedding_model, str(self.device))
        self.embedder_load_time_s = time.perf_counter() - t0

    # -- one greedy construction ---------------------------------------------
    def _search(self, data_x: pd.DataFrame, data_y: np.ndarray) -> Dict[str, Any]:
        # CTX-PORT (note 3): upstream builds a fresh AgentManager — and so draws
        # fresh gate matrices — for every dataset, from wherever the global RNG
        # happens to be. Re-seeding first makes each run self-contained.
        seed_everything(self.seed)
        agent = Agent(self.embedder, self.device, n_ctx_rows=self.n_ctx_rows)
        agent.load_weights(self.weights_dir, self.weights_tag)
        env = Environment(data_x, data_y, train_size=self.search_train_size, split_seed=self.split_seed)
        return CtxPipeSearch(agent, env).run()

    # -- public API -----------------------------------------------------------
    def apply_to_test(self, df_test: pd.DataFrame, pipeline_info: Dict) -> pd.DataFrame:
        """Put a held-out frame through the pipeline fitted on training.

        The imputers, encoders, scalers and projections keep their training
        statistics — what upstream does for its test split, bar the imputers
        and label encoder it re-fits (CTX-PORT notes above). Rows are kept.
        """
        selected: MaterializedPipeline = pipeline_info["materialized_pipeline"]
        col_types = pipeline_info["col_types"]
        feature_cols = pipeline_info["input_feature_columns"]
        passthrough = pipeline_info["passthrough_columns"]
        target_col = pipeline_info["target_col"]

        df = _coerce_types(df_test, col_types)
        features = selected.transform(df[feature_cols])
        blocks = [features, df[[c for c in passthrough if c in df.columns]].reset_index(drop=True)]
        if target_col in df_test.columns:
            blocks.append(df_test[[target_col]].reset_index(drop=True))
        out = pd.concat(blocks, axis=1)
        return out[[c for c in pipeline_info["kept_columns"] if c in out.columns]]

    def _holdout_accuracy(self, df_clean: pd.DataFrame, df_test: pd.DataFrame,
                          pipeline_info: Dict) -> Optional[float]:
        """Accuracy of the reward model — logistic regression — fitted on the
        prepared poisoned partition and scored on the prepared clean hold-out."""
        target_col = pipeline_info["target_col"]
        if target_col not in df_test.columns:
            return None
        feature_cols = pipeline_info["output_feature_columns"]
        test_prepared = self.apply_to_test(df_test, pipeline_info)
        y_train = df_clean[target_col].values
        pred = LogisticRegressionPrim().transform(
            df_clean[feature_cols], y_train, test_prepared[feature_cols]
        )
        return float(accuracy_score(df_test[target_col].values, pred))

    def prepare(
        self,
        df_poisoned: pd.DataFrame,
        mask_df: pd.DataFrame,
        df_test: Optional[pd.DataFrame] = None,
        file_name: str = "dataset",
    ) -> Tuple[pd.DataFrame, pd.DataFrame, Dict, Dict]:
        """Run CtxPipe on one poisoned dataset.

        Returns
        -------
        df_clean     : prepared dataframe (same rows; the columns CtxPipe produced)
        residual_mask: boolean mask over df_clean (True = poisoned and still NaN)
        perf_metrics : timing / memory / search-quality metrics
        pipeline_info: the selected pipeline, fitted, plus the search record
        """
        proc = psutil.Process()
        ram_before_mb = proc.memory_info().rss / 1024 / 1024
        tracemalloc.start()
        wall_t0 = time.perf_counter()
        cpu_t0 = time.process_time()

        col_types = _identify_col_types(df_poisoned)
        if col_types["target_cls"]:
            target_col = col_types["target_cls"][0]
        elif col_types["target_reg"]:
            # CTX-PORT: upstream is classification-only — the reward is the
            # accuracy of a classifier (env/metric.py, comp.selected_prim).
            raise ValueError(
                "CtxPipe only supports classification targets (cls_*); "
                f"'{col_types['target_reg'][0]}' is a regression target"
            )
        else:
            raise ValueError("No cls_* or reg_* target column found")

        df = _coerce_types(df_poisoned, col_types).reset_index(drop=True)
        feature_cols = col_types["numerical"] + col_types["categorical"]
        # CTX-PORT: date and id columns stay out of the search and are written
        # back untouched, as in the other ports.
        passthrough = [c for c in df.columns if c not in feature_cols and c != target_col]
        if not feature_cols:
            raise ValueError("No usable num_*/cat_* feature columns")

        data_x, search_names = _search_frame(df, feature_cols)
        data_y = df[target_col].values

        logger.info(
            f"  [CtxPipe] {len(col_types['numerical'])} num + {len(col_types['categorical'])} cat "
            f"features | {len(df)} rows | {df[target_col].nunique()} classes"
        )

        # ---- search: greedy construction by the agents --------------------
        embed_calls0, embed_time0 = self.embedder.n_calls, self.embedder.total_time_s
        search_t0 = time.perf_counter()
        result = self._search(data_x, data_y)
        search_time_s = time.perf_counter() - search_t0
        n_embeddings = self.embedder.n_calls - embed_calls0
        embedding_time_s = self.embedder.total_time_s - embed_time0

        # ---- materialize the selected pipeline ----------------------------
        mat_t0 = time.perf_counter()
        selected = MaterializedPipeline(result["sequence"])
        features = selected.fit_transform(df[feature_cols], data_y)
        materialize_time_s = time.perf_counter() - mat_t0

        df_clean = pd.concat(
            [
                features,
                df[passthrough].reset_index(drop=True),
                df_poisoned[[target_col]].reset_index(drop=True),
            ],
            axis=1,
        )
        output_feature_cols = list(features.columns)
        lineage = dict(selected.lineage_)
        for c in passthrough + [target_col]:
            lineage[c] = [c]
        reached = {src for srcs in selected.lineage_.values() for src in srcs}
        dropped = [c for c in feature_cols if c not in reached]

        # ---- residual mask -------------------------------------------------
        residual_mask = build_residual_mask(df_clean, mask_df.reset_index(drop=True), lineage)
        n_poisoned = int(mask_df.to_numpy().sum())
        n_residual = int(residual_mask.to_numpy().sum())
        logger.info(
            f"  [CtxPipe] {n_poisoned} poisoned input cells, "
            f"{n_residual} still missing in the output"
        )

        wall_time_s = time.perf_counter() - wall_t0
        cpu_time_s = time.process_time() - cpu_t0
        _, peak_bytes = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        ram_after_mb = proc.memory_info().rss / 1024 / 1024
        ram_peak_mb = peak_bytes / 1024 / 1024

        n_rejected = sum(len(s["rejected"]) for s in result["steps"])

        pipeline_info = {
            "method": "ctxpipe",
            "end_model": "LogisticRegression",
            "target_col": target_col,
            "col_types": col_types,
            "num_columns": col_types["numerical"],
            "cat_columns": col_types["categorical"],
            "passthrough_columns": passthrough,
            "input_feature_columns": feature_cols,
            "output_feature_columns": output_feature_cols,
            "search_column_names": search_names,
            "kept_columns": list(df_clean.columns),
            "dropped_columns": dropped,
            "lineage": selected.lineage_,
            "logical_pipeline_id": result["logical_pipeline_id"],
            "logical_pipeline": result["logical_pipeline"],
            "physical_pipeline": [
                {"component": c, "primitive": p}
                for c, p in zip(result["logical_pipeline"], result["sequence"])
            ],
            "steps": result["steps"],
            "materialization_skipped": selected.skipped_,
            "materialized_pipeline": selected,
            "result": {
                "search_score": result["score"],
                "logical_q_values": result["logical_q_values"],
                "search_train_rows": result["search_train_rows"],
                "search_test_rows": result["search_test_rows"],
                "n_rejected_actions": n_rejected,
            },
            "params": {
                "weights_dir": str(self.weights_dir),
                "weights_tag": self.weights_tag,
                "embedding_model": self.embedding_model,
                "device": str(self.device),
                "n_ctx_rows": self.n_ctx_rows,
                "seed": self.seed,
                "split_seed": self.split_seed,
                "search_train_size": self.search_train_size,
                "kernel_pca_max_rows": KERNEL_PCA_MAX_ROWS,
            },
        }

        # ---- accuracy on the clean hold-out (outside the timed region) -----
        holdout_acc, holdout_time_s = None, None
        if df_test is not None:
            h_t0 = time.perf_counter()
            try:
                holdout_acc = self._holdout_accuracy(df_clean, df_test, pipeline_info)
            except Exception as exc:
                logger.warning(f"  [CtxPipe] hold-out accuracy could not be computed: {exc}")
            holdout_time_s = time.perf_counter() - h_t0
        pipeline_info["result"]["holdout_test_acc"] = holdout_acc

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
            "method": "ctxpipe",
            "end_model": "LogisticRegression",
            "metric_name": "accuracy",
            "best_pipeline": selected.describe(),
            "best_order": " → ".join(result["logical_pipeline"]),
            "best_score": round(result["score"], 6),
            "logical_pipeline_id": result["logical_pipeline_id"],
            "holdout_test_acc": round(holdout_acc, 6) if holdout_acc is not None else None,
            "n_rejected_actions": n_rejected,
            "n_ctx_embeddings": n_embeddings,
            "embedding_time_s": round(embedding_time_s, 4),
            "search_train_rows": result["search_train_rows"],
            "search_test_rows": result["search_test_rows"],
            "n_num_features": len(col_types["numerical"]),
            "n_cat_features": len(col_types["categorical"]),
            "n_passthrough_features": len(passthrough),
            "n_features_out": len(output_feature_cols),
            "n_dropped_features": len(dropped),
            "n_materialization_skipped": len(selected.skipped_),
            "n_poisoned_cells": n_poisoned,
            "n_residual_cells": n_residual,
            "search_time_s": round(search_time_s, 4),
            "materialize_time_s": round(materialize_time_s, 4),
            "holdout_eval_time_s": round(holdout_time_s, 4) if holdout_time_s is not None else None,
            "embedding_model": self.embedding_model,
            "weights_tag": self.weights_tag,
        }

        return df_clean, residual_mask, perf_metrics, pipeline_info


# ---------------------------------------------------------------------------
# Metrics helper (mirrors saga.py / learn2clean.py / diffprep.py)
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
    """Persist the selected pipeline description plus the search record.

    The live ``MaterializedPipeline`` is dropped first: it carries fitted
    components whose classes are defined in this file, so a pickle written
    while the file runs as ``__main__`` would not reload anywhere else. What
    stays — ``physical_pipeline`` and ``input_feature_columns`` — is enough to
    rebuild and refit it with ``MaterializedPipeline([...])``.
    """
    import pickle as _pickle

    payload = {k: v for k, v in pipeline_info.items() if k != "materialized_pipeline"}
    with open(path, "wb") as f:
        _pickle.dump(payload, f)


# ---------------------------------------------------------------------------
# Dataset processing (mirrors process_all_datasets in diffprep.py)
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
    weights_dir: str = str(DEFAULT_WEIGHTS_DIR),
    weights_tag: str = WEIGHTS_TAG,
    embedding_model: str = CTX_MODEL_PATH,
    device: str = "cpu",
    n_ctx_rows: int = N_CTX_ROWS,
    seed: int = RANDOM_SEED,
    split_seed: int = SEARCH_SPLIT_SEED,
    search_train_size: float = SEARCH_TRAIN_SIZE,
    n_threads: Optional[int] = None,
    clean_dir: str = "data",
    poison_test_size: float = 0.3,
    verbose: bool = False,
):
    """
    Apply CtxPipe to all poisoned datasets.

    Reads poisoned CSVs and their masks from ``input_dir/{ar,nar}/`` and writes
    prepared CSVs, residual masks, the selected pipeline and metrics to
    ``output_dir/{ar,nar}/``. The clean hold-out of ``input_dir/test/`` is put
    through the pipeline fitted on the training frame and written to
    ``output_dir/test/{mode}/``, one copy per corruption mode — AR and NAR
    select different pipelines, so they put the hold-out in different feature
    spaces.

    The un-poisoned train+val partition is transformed the same way and written
    to ``output_dir/clean/{mode}/``. Downstream, quail can swap the validation
    split for its clean rows; those rows have to come off the same pipeline as
    the training rows — after an encoder, a projection or a feature selection a
    raw clean row does not even have the same columns.
    """
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.join(output_dir, "ar"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "nar"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "metrics"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "test", "ar"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "test", "nar"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "clean", "ar"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "clean", "nar"), exist_ok=True)

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
            "no prepared hold-out will be written"
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

    skipped = sum(1 for f in csv_files if check_dataset_complete(f, output_dir))
    if skipped:
        logger.info(f"Skipped {skipped} already-processed dataset(s)")
    if skipped == len(csv_files):
        logger.success(f"All datasets prepared! Output in {output_dir}")
        return

    cleaner = CtxPipe(
        weights_dir=weights_dir, weights_tag=weights_tag, embedding_model=embedding_model,
        device=device, n_ctx_rows=n_ctx_rows, seed=seed, split_seed=split_seed,
        search_train_size=search_train_size, n_threads=n_threads, verbose=verbose,
    )
    logger.info(
        f"CtxPipe agents: {weights_tag} from {weights_dir} | context: {embedding_model} "
        f"({n_ctx_rows} rows) on {cleaner.device} | embedder loaded in "
        f"{cleaner.embedder_load_time_s:.1f}s"
    )

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

                holdout = perf["holdout_test_acc"]
                logger.info(
                    f"  {mode.upper()} prepared: {metrics['overall_cleanliness']:.2f}% clean "
                    f"| {perf['n_rows_out']}×{perf['n_cols_out']} "
                    f"| {perf['wall_time_s']:.1f}s wall | {perf['ram_peak_mb']:.1f} MB peak "
                    f"| search_acc={perf['best_score']} "
                    f"holdout_acc={holdout if holdout is not None else 'n/a'} "
                    f"| order: {perf['best_order']} | pipeline: {perf['best_pipeline']}"
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
        description="CtxPipe context-aware data-preparation pipeline construction baseline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
                Paper: Haotian Gao, Shaofeng Cai, Tien Tuan Anh Dinh, Zhiyong Huang,
                    Beng Chin Ooi. "CtxPipe: Context-aware Data Preparation Pipeline
                    Construction for Machine Learning." SIGMOD '25, Article 231.
                Code : https://github.com/ctxpipe/ctxpipe

                DQN agents pick a logical pipeline (order of the component types)
                and then one component per slot, reading statistical features of
                the intermediate data and a GTE-large embedding of sampled rows
                (the gated context plug-in). The released 32,000-step agents are
                used as they are; nothing is trained here.

                Search space:
                ImputerNum            : mean, median, most frequent
                ImputerCat            : most frequent
                Encoder               : NumericData, LabelEncoder, OneHotEncoder
                FeaturePreprocessing  : MinMax, MaxAbs, Robust, Standard, Quantile,
                                        Power, Normalizer, KBinsDiscretizer, none
                FeatureEngine         : Polynomial, Interaction, PCA, IncrementalPCA,
                                        KernelPCA, TruncatedSVD, RandomTreesEmbedding, none
                FeatureSelection      : VarianceThreshold, none

                Examples:
                python scripts/ctxpipe.py
                python scripts/ctxpipe.py --input_dir data_poisoned --output_dir data_cleaned_ctxpipe
                python scripts/ctxpipe.py --dataset tic_tac_toe --verbose
        """,
    )
    parser.add_argument(
        "--input_dir", type=str, default="data_poisoned",
        help="Root directory of poisoned data (default: data_poisoned)",
    )
    parser.add_argument(
        "--output_dir", type=str, default=None,
        help="Root directory for prepared output "
             "(default: config.yaml ctxpipe_data_dir, else data_cleaned_ctxpipe)",
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
    parser.add_argument("--weights_dir", type=str, default=None,
                        help="Directory of the released agent weights (default: scripts/ctxpipe_weights)")
    parser.add_argument("--weights_tag", type=str, default=None,
                        help="Prefix of the weight files (default: ctx_32000)")
    parser.add_argument("--embedding_model", type=str, default=None,
                        help="Hugging Face id or local path of the context embedding model "
                             "(default: thenlper/gte-large)")
    parser.add_argument("--device", type=str, default=None,
                        help="torch device for the embedder and the agents: cpu, cuda, mps or auto "
                             "(default: cpu)")
    parser.add_argument("--n_ctx_rows", type=int, default=None,
                        help="Rows sampled for the context embedding (default: 100, as the code)")
    parser.add_argument("--seed", type=int, default=None,
                        help="Seed re-applied before every run (default: 1145, upstream's RANDOM_SEED)")
    parser.add_argument("--split_seed", type=int, default=None,
                        help="Seed of the search's train/test split (default: 0, as upstream)")
    parser.add_argument("--search_train_size", type=float, default=None,
                        help="Train share of the search's split (default: 0.8, as upstream)")
    parser.add_argument("--kernel_pca_max_rows", type=int, default=None,
                        help="Largest frame KernelPCA accepts (default: unbounded, as upstream)")
    parser.add_argument("--n_threads", type=int, default=None,
                        help="torch intra-op threads (default: torch's own default)")
    parser.add_argument("--verbose", action="store_true",
                        help="Log every agent decision (Q-values, context scale, rejected actions)")

    args = parser.parse_args()

    _config: dict = {}
    if Path(args.config).exists():
        with open(args.config) as _f:
            _config = yaml.safe_load(_f) or {}

    cp_cfg = _config.get("ctxpipe") or {}

    def _opt(name, default):
        """CLI flag > config.yaml ctxpipe.<name> > built-in default."""
        cli = getattr(args, name, None)
        if cli is not None:
            return cli
        value = cp_cfg.get(name, default)
        return default if value is None else value

    if args.verbose:
        logger.remove()
        logger.add(
            sys.stderr,
            level="DEBUG",
            format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <level>{message}</level>",
        )

    output_dir = args.output_dir or _config.get("ctxpipe_data_dir", "data_cleaned_ctxpipe")

    if args.dataset:
        datasets = [args.dataset]
    else:
        datasets = _config.get("datasets") or None

    KERNEL_PCA_MAX_ROWS = (
        args.kernel_pca_max_rows
        if args.kernel_pca_max_rows is not None
        else cp_cfg.get("kernel_pca_max_rows")
    )

    # config.yaml gives the weights directory relative to the repository root.
    weights_dir = Path(_opt("weights_dir", str(DEFAULT_WEIGHTS_DIR)))
    if not weights_dir.is_absolute() and not weights_dir.exists():
        weights_dir = Path(__file__).resolve().parent.parent / weights_dir

    process_all_datasets(
        input_dir=args.input_dir,
        output_dir=output_dir,
        datasets=datasets,
        weights_dir=str(weights_dir),
        weights_tag=_opt("weights_tag", WEIGHTS_TAG),
        embedding_model=_opt("embedding_model", CTX_MODEL_PATH),
        device=_opt("device", "cpu"),
        n_ctx_rows=_opt("n_ctx_rows", N_CTX_ROWS),
        seed=_opt("seed", RANDOM_SEED),
        split_seed=_opt("split_seed", SEARCH_SPLIT_SEED),
        search_train_size=_opt("search_train_size", SEARCH_TRAIN_SIZE),
        n_threads=_opt("n_threads", None),
        clean_dir=args.clean_dir,
        poison_test_size=(
            args.poison_test_size
            if args.poison_test_size is not None
            else _config.get("test_size", 0.3)
        ),
        verbose=args.verbose,
    )
