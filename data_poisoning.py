import numpy as np
import pandas as pd
import random
from typing import Tuple, List, Sequence, Optional, Dict
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from collections import defaultdict


def top_logistic_features(
    X: pd.DataFrame,
    y: Optional[pd.Series],
    n_features: int,
    *,
    random_state: Optional[int] = None,
    max_iter: int = 1000,
) -> List[str]:
    """
    Return up to n_features columns ranked by logistic regression coefficient magnitudes.
    Only numeric features are considered; rows with missing values are ignored.
    """
    if y is None or n_features <= 0:
        return []

    if not isinstance(y, pd.Series):
        try:
            y_series = pd.Series(y, index=X.index)
        except ValueError:
            return []
    else:
        y_series = y.reindex(X.index)

    numeric_X = X.select_dtypes(include=[np.number])
    if numeric_X.empty:
        return []

    combined = pd.concat([numeric_X, y_series], axis=1)
    combined = combined.replace([np.inf, -np.inf], np.nan).dropna()
    if combined.empty:
        return []

    y_valid = combined.iloc[:, -1]
    if y_valid.nunique() < 2:
        return []

    X_valid = combined.iloc[:, :-1]
    if X_valid.shape[1] == 0:
        return []

    X_valid = X_valid.astype(float)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_valid)

    model = LogisticRegression(
        max_iter=max_iter,
        class_weight='balanced',
        solver='lbfgs',
        multi_class='auto',
        random_state=random_state,
    )

    try:
        model.fit(X_scaled, y_valid)
    except Exception:
        return []

    coef = np.abs(model.coef_)
    if coef.ndim == 1:
        coef_magnitude = coef
    else:
        coef_magnitude = coef.max(axis=0)

    importance = pd.Series(coef_magnitude, index=X_valid.columns)
    importance = importance.replace([np.inf, -np.inf], np.nan).dropna()
    if importance.empty:
        return []

    return importance.nlargest(min(n_features, len(importance))).index.tolist()


def flipping_poisoning(
    X: pd.DataFrame,
    features_percentage: float,
    poisoning_percentage: float,
    random_state: int = None,
    *,
    columns_ohe: Optional[list[str]] = None,
    original_dataframe: Optional[bool] = False,
    features_to_poison: Optional[Sequence[str]] = None,
    instances_by_feature: Optional[Dict[str, Sequence[int]]] = None,
) -> Tuple[pd.DataFrame, List[float]]:
    """
    Apply flipping poisoning to the dataset.

    Args:
        X: Original dataframe
        features_percentage: Percentage of features to poison (0-1)
        poisoning_percentage: Percentage of instances to poison per feature (0-1)
        random_state: Random seed for reproducibility
        columns_ohe: If original_dataframe=True, the expected final OHE columns order
        original_dataframe: If True, X is the original df (will be OHE-encoded at the end)
        features_to_poison: Optional explicit list of features to poison
        instances_by_feature: Optional dict {feature: iterable of row indices} to poison

    Returns:
        Tuple of:
            - poisoned dataframe (possibly OHE if original_dataframe=True),
            - q: feature-wise quality vector ordered by (X.columns if not original_dataframe else columns_ohe),
            - r: row-wise quality vector ordered by X.index
              (r[i] = 1 - (# of poisoned features in row i) / n_features)
    """
    if random_state is not None:
        np.random.seed(random_state)
        random.seed(random_state)
    
    X_poisoned = X.copy()
    n_features = X.shape[1]
    n_instances = X.shape[0]
    
    # Select features to poison (external selection if provided)
    if features_to_poison is None:
        n_features_to_poison = max(1, int(n_features * features_percentage))
        features_to_poison = random.sample(list(X.columns), n_features_to_poison)
    else:
        features_to_poison = list(features_to_poison)

    # Initialize quality vector (1 = clean). For poisoned features set to 1 - poisoning_percentage
    feature_quality = {col: 1.0 for col in (X.columns if not original_dataframe else columns_ohe)}
    if original_dataframe:
        categorical_cols = X.select_dtypes(
                include=["object", "string", "category"]
        ).columns.tolist()
        for col in features_to_poison:
            if col not in categorical_cols:
                feature_quality[col] = max(0.0, 1.0 - float(poisoning_percentage))
            #Define later the poisoning percentage per categorical columns
    else:
        categorical_cols = []
        for col in features_to_poison:
            feature_quality[col] = max(0.0, 1.0 - float(poisoning_percentage))

    # Count how many features were flipped in each row
    row_poison_count = defaultdict(int)
    
    for feature in features_to_poison:
        # Get unique values for this feature
        unique_values = X[feature].unique()
        
        # Skip if only one unique value
        if len(unique_values) <= 1:
            continue
            
        # Calculate/select instances to poison for this feature
        if instances_by_feature is not None and feature in instances_by_feature:
            instances_to_poison = list(instances_by_feature[feature])
        else:
            n_instances_to_poison = max(1, int(n_instances * poisoning_percentage))
            instances_to_poison = random.sample(list(X.index), n_instances_to_poison)
        
        for instance_idx in instances_to_poison:
            current_value = X.loc[instance_idx, feature]
            
            # Find other possible values (different from current)
            other_values = [val for val in unique_values if val != current_value]
            
            if other_values:
                # Randomly select a different value
                new_value = random.choice(other_values)
                old_value = X_poisoned.loc[instance_idx, feature]
                X_poisoned.loc[instance_idx, feature] = new_value

                # track for row quality
                row_poison_count[instance_idx] += 1

        #Decrease feature quality of categorical columns for one/n_instances on this iteration    
        if original_dataframe and feature in categorical_cols:
            feature_quality[f"{feature}_{old_value}"] -= 1/n_instances
            feature_quality[f"{feature}_{new_value}"] -= 1/n_instances

    if original_dataframe:
        #Original Dataframe must be converted to the one hot encoded version
        X_poisoned = pd.get_dummies(
            X_poisoned,
            columns=categorical_cols,
            drop_first=False,
            dtype=int,
        )

        X_poisoned = X_poisoned.reindex(columns=columns_ohe, fill_value=0)

    # Build row-wise quality vector r (aligned to X.index)
    r_index_order = list(X.index)
    r = []
    for idx in r_index_order:
        k = row_poison_count.get(idx, 0)
        r.append(max(0.0, 1.0 - (k / n_features)))
    
    # Build ordered q aligned with X.columns
    q = [feature_quality[col] for col in (X.columns if not original_dataframe else columns_ohe)]

    return X_poisoned, q, r


def noise_poisoning(
    X: pd.DataFrame,
    features_percentage: float,
    poisoning_percentage: float,
    noise_type: str = 'gaussian',
    noise_scale: float = 0.8,
    random_state: int = None, 
    *,
    continuous_features: Optional[list[str]] = None,
    features_to_poison: Optional[Sequence[str]] = None,
    instances_by_feature: Optional[Dict[str, Sequence[int]]] = None,
    y: Optional[pd.Series] = None,
) -> Tuple[pd.DataFrame, List[float]]:
    """
    Apply noise poisoning to the dataset.
    
    Args:
        X: Original dataframe
        features_percentage: Percentage of features to poison (0-1)
        poisoning_percentage: Percentage of instances to poison per feature (0-1)
        noise_type: Type of noise ('gaussian', 'uniform', 'laplace')
        noise_scale: Scale parameter for noise (standard deviation for gaussian)
        random_state: Random seed for reproducibility
        
    Returns:
        Tuple of (poisoned dataframe, q vector per feature ordered by X.columns)
    """
    if random_state is not None:
        np.random.seed(random_state)
        random.seed(random_state)
    
    X_poisoned = X.copy()
    n_features = X.shape[1]
    n_instances = X.shape[0]
    
    # Select features to poison (external selection if provided)
    if features_to_poison is None:
        n_features_to_poison = max(1, int(n_features * features_percentage))
        features_to_poison = random.sample(list(X.columns), n_features_to_poison)
    else:
        features_to_poison = list(features_to_poison)

    # Initialize quality vector (1 = clean). For poisoned features set to 1 - poisoning_percentage
    feature_quality = {col: 1.0 for col in X.columns}
    for col in features_to_poison:
        feature_quality[col] = max(0.0, 1.0 - float(poisoning_percentage))

    # Count how many features were flipped in each row
    row_poison_count = defaultdict(int)
    
    for feature in features_to_poison:
        # Skip non-numeric features for noise poisoning
        if not pd.api.types.is_numeric_dtype(X[feature]):
            continue
            
        # Calculate/select instances to poison for this feature
        if instances_by_feature is not None and feature in instances_by_feature:
            instances_to_poison = list(instances_by_feature[feature])
        else:
            n_instances_to_poison = max(1, int(n_instances * poisoning_percentage))
            instances_to_poison = random.sample(list(X.index), n_instances_to_poison)
        
        # Calculate noise based on feature statistics
        feature_std = X[feature].std()
        feature_mean = X[feature].mean()
        
        for instance_idx in instances_to_poison:
            current_value = X.loc[instance_idx, feature]
            
            # Generate noise based on type
            if noise_type == 'gaussian':
                noise = np.random.normal(0, noise_scale * feature_std)
            elif noise_type == 'uniform':
                noise = np.random.uniform(-noise_scale * feature_std, noise_scale * feature_std)
            elif noise_type == 'laplace':
                noise = np.random.laplace(0, noise_scale * feature_std)
            else:
                raise ValueError(f"Unknown noise type: {noise_type}")
            
            # Apply noise
            new_value = current_value + noise
            X_poisoned.loc[instance_idx, feature] = new_value
            # track for row quality
            row_poison_count[instance_idx] += 1
    
    # Build row-wise quality vector r (aligned to X.index)
    r_index_order = list(X.index)
    r = []
    for idx in r_index_order:
        k = row_poison_count.get(idx, 0)
        r.append(max(0.0, 1.0 - (k / n_features)))
    
    # Build ordered q aligned with X.columns
    q = [feature_quality[col] for col in X.columns]

    return X_poisoned, q, r


def incompleteness_poisoning(
    X: pd.DataFrame,
    features_percentage: float,
    poisoning_percentage: float,
    imputation_method: str = 'mean',
    random_state: int = None,
    *,
    columns_ohe: Optional[list[str]] = None,
    original_dataframe: Optional[bool] = False,
    features_to_poison: Optional[Sequence[str]] = None,
    instances_by_feature: Optional[Dict[str, Sequence[int]]] = None,
    y: Optional[pd.Series] = None,
) -> Tuple[pd.DataFrame, List[float]]:
    """
    Apply incompleteness poisoning to the dataset by setting values to NaN and then imputing.
    
    Args:
        X: Original dataframe
        features_percentage: Percentage of features to poison (0-1)
        poisoning_percentage: Percentage of instances to poison per feature (0-1)
        imputation_method: Method for imputation ('mean', 'median', 'mode')
        random_state: Random seed for reproducibility
        
    Returns:
        Tuple of (poisoned dataframe, q vector per feature ordered by X.columns, imputation statistics)
    """
    if random_state is not None:
        np.random.seed(random_state)
        random.seed(random_state)
    
    X_poisoned = X.copy()
    n_features = X.shape[1]
    n_instances = X.shape[0]
    imputation_stats = {}
    
    # Select features to poison (external selection if provided)
    if features_to_poison is None:
        n_features_to_poison = max(1, int(n_features * features_percentage))
        features_to_poison = random.sample(list(X.columns), n_features_to_poison) 
    else:
        features_to_poison = list(features_to_poison)

    # Initialize quality vector (1 = clean). For poisoned features set to 1 - poisoning_percentage
    feature_quality = {col: 1.0 for col in (X.columns if not original_dataframe else columns_ohe)}
    if original_dataframe:
        categorical_cols = X.select_dtypes(
                include=["object", "string", "category"]
        ).columns.tolist()
        for col in features_to_poison:
            if col not in categorical_cols:
                feature_quality[col] = max(0.0, 1.0 - float(poisoning_percentage))
    else:
        categorical_cols = []
        for col in features_to_poison:
            feature_quality[col] = max(0.0, 1.0 - float(poisoning_percentage))
    
    # Count how many features were flipped in each row
    row_poison_count = defaultdict(int)

    for feature in features_to_poison:
        # Calculate/select instances to poison for this feature
        if instances_by_feature is not None and feature in instances_by_feature:
            instances_to_poison = list(instances_by_feature[feature])
        else:
            n_instances_to_poison = max(1, int(n_instances * poisoning_percentage))
            instances_to_poison = random.sample(list(X.index), n_instances_to_poison)
        
        # Store original values for imputation
        original_values = X.loc[instances_to_poison, feature].copy()
        
        # Set selected instances to NaN
        X_poisoned.loc[instances_to_poison, feature] = np.nan

        # Mark that these rows had this feature poisoned
        for idx in instances_to_poison:
            row_poison_count[idx] += 1
        
        # Calculate imputation value based on remaining non-NaN values
        remaining_values = X_poisoned[feature].dropna()
        
        if len(remaining_values) > 0:
            if feature not in categorical_cols:
                if imputation_method == 'mean':
                    imputation_value = remaining_values.mean()
                elif imputation_method == 'median':
                    imputation_value = remaining_values.median()
                elif imputation_method == 'mode':
                    imputation_value = remaining_values.mode().iloc[0] if not remaining_values.mode().empty else remaining_values.mean()
                else:
                    raise ValueError(f"Unknown imputation method: {imputation_method}")
            else:
                #Handling categorical features
                value_counts = original_values.value_counts(normalize=True, dropna=True)
                if value_counts.empty:
                    raise ValueError("Cannot compute categorical imputation probabilities from empty original data.")
                choices = value_counts.index.to_list()
                probabilities = value_counts.to_numpy(dtype=float)
                imputation_value = pd.Series(
                    np.random.choice(choices, size=len(instances_to_poison), p=probabilities),
                    index=instances_to_poison,
                )

                #Update feature quality
                for new_value, old_value in zip(original_values, imputation_value):
                    feature_quality[f"{feature}_{new_value}"] -= 1/n_instances
                    feature_quality[f"{feature}_{old_value}"] -= 1/n_instances
            
            # Apply imputation
            X_poisoned.loc[instances_to_poison, feature] = imputation_value
            
            '''# Store imputation statistics
            imputation_stats[feature] = {
                'imputation_value': imputation_value,
                'imputation_method': imputation_method,
                'n_poisoned_instances': len(instances_to_poison),
                'original_values': original_values.tolist()
            }'''
        else:
            # If all values are NaN, use the mean of original values
            imputation_value = original_values.mean()
            X_poisoned.loc[instances_to_poison, feature] = imputation_value
            
            '''imputation_stats[feature] = {
                'imputation_value': imputation_value,
                'imputation_method': 'fallback_mean',
                'n_poisoned_instances': len(instances_to_poison),
                'original_values': original_values.tolist()
            }'''
    
    if original_dataframe:
        #Original Dataframe must be converted to the one hot encoded version
        X_poisoned = pd.get_dummies(
            X_poisoned,
            columns=categorical_cols,
            drop_first=False,
            dtype=int,
        )

        X_poisoned = X_poisoned.reindex(columns=columns_ohe, fill_value=0)
    
    # Build row-wise quality vector r (aligned to X.index)
    r_index_order = list(X.index)
    r = []
    for idx in r_index_order:
        k = row_poison_count.get(idx, 0)
        r.append(max(0.0, 1.0 - (k / n_features)))
    
    # Build ordered q aligned with X.columns
    q = [feature_quality[col] for col in (X.columns if not original_dataframe else columns_ohe)]

    return X_poisoned, q, r


def _select_disjoint_feature_sets(
    X: pd.DataFrame,
    features_percentage: float,
    random_state: int = None,
) -> Tuple[List[str], List[str], List[str]]:
    """
    Select three disjoint feature sets for flipping, noise, and incompleteness poisoning.

    Args:
        X: Dataframe to select features from
        features_percentage: Percentage of features to allocate to each poisoning method (0-1)
        random_state: Random seed for reproducibility

    Returns:
        Tuple of three disjoint feature name lists: (flip_features, noise_features, incomplete_features)
    """
    if random_state is not None:
        np.random.seed(random_state)
        random.seed(random_state)

    all_features = list(X.columns)
    n_features_total = len(all_features)
    n_per_set = max(1, int(n_features_total * features_percentage))
    # Cap to total features to keep sets internally unique
    n_per_set = min(n_per_set, n_features_total)

    # If we can allocate disjointly, do so
    if 3 * n_per_set <= n_features_total:
        selected = random.sample(all_features, 3 * n_per_set)
        flip_features = selected[0:n_per_set]
        noise_features = selected[n_per_set:2 * n_per_set]
        incomplete_features = selected[2 * n_per_set:3 * n_per_set]
        return flip_features, noise_features, incomplete_features

    # Otherwise, allow overlaps across sets while keeping each set unique internally
    def sample_unique_features(k: int) -> List[str]:
        if k >= n_features_total:
            return all_features.copy()
        chosen = []
        chosen_set = set()
        while len(chosen) < k:
            cand = random.choice(all_features)
            if cand not in chosen_set:
                chosen.append(cand)
                chosen_set.add(cand)
        return chosen

    flip_features = sample_unique_features(n_per_set)
    noise_features = sample_unique_features(n_per_set)
    incomplete_features = sample_unique_features(n_per_set)

    return flip_features, noise_features, incomplete_features


def combined_poisoning(
    X: pd.DataFrame,
    features_percentage: float,
    flipping_percentage: float,
    noise_percentage: float,
    incompleteness_percentage: float,
    *,
    noise_type: str = 'gaussian',
    noise_scale: float = 0.1,
    imputation_method: str = 'mean',
    random_state: int = None,
) -> Tuple[pd.DataFrame, List[float]]:
    """
    Apply flipping, noise, and incompleteness poisoning together on disjoint feature sets.

    Feature selection is performed once and passed to the individual poisoning functions.
    """
    # Select disjoint feature sets
    flip_feats, noise_feats, inc_feats = _select_disjoint_feature_sets(
        X, features_percentage, random_state
    )

    # Coordinate row selection across overlapping feature sets per-feature using proportional, disjoint allocation
    n_instances = X.shape[0]
    flip_instances_map: Dict[str, List[int]] = {}
    noise_instances_map: Dict[str, List[int]] = {}
    inc_instances_map: Dict[str, List[int]] = {}

    feat_in_flip = set(flip_feats)
    feat_in_noise = set(noise_feats)
    feat_in_inc = set(inc_feats)
    all_feats = list(set(flip_feats) | set(noise_feats) | set(inc_feats))

    all_indices = list(X.index)

    for feat in all_feats:
        # Determine which types apply to this feature and their requested percentages
        types_here: List[str] = []
        perc_by_type: Dict[str, float] = {}
        if feat in feat_in_flip:
            types_here.append('flip')
            perc_by_type['flip'] = float(flipping_percentage)
        if feat in feat_in_noise:
            types_here.append('noise')
            perc_by_type['noise'] = float(noise_percentage)
        if feat in feat_in_inc:
            types_here.append('inc')
            perc_by_type['inc'] = float(incompleteness_percentage)

        if not types_here:
            continue

        sum_p = sum(perc_by_type[t] for t in types_here)

        # Compute target counts per type for this feature
        raw_targets: Dict[str, float]
        if sum_p <= 1.0:
            # Use disjoint percentages directly
            raw_targets = {t: perc_by_type[t] * n_instances for t in types_here}
        else:
            # Scale proportionally to fill at most N rows disjointly
            raw_targets = {t: (perc_by_type[t] / sum_p) * n_instances for t in types_here}

        # Convert to integers using floor then distribute remainder by largest fractional parts
        floored = {t: int(np.floor(raw_targets[t])) for t in types_here}
        used = sum(floored.values())
        remainder = max(0, n_instances - used)
        # Sort by fractional part descending
        frac_order = sorted(types_here, key=lambda t: (raw_targets[t] - floored[t]), reverse=True)
        i = 0
        while remainder > 0 and i < len(frac_order):
            floored[frac_order[i]] += 1
            remainder -= 1
            i += 1

        # Now sample disjoint indices according to floored counts
        pool = all_indices.copy()
        random.shuffle(pool)
        offset = 0
        for t in types_here:
            k = max(0, min(floored[t], n_instances - offset))
            chosen = pool[offset:offset + k]
            offset += k
            if t == 'flip':
                flip_instances_map[feat] = chosen
            elif t == 'noise':
                noise_instances_map[feat] = chosen
            else:
                inc_instances_map[feat] = chosen

    # Apply each poisoning with coordinated instance selections
    X_poisoned, _, _ = flipping_poisoning(
        X,
        features_percentage,
        flipping_percentage,
        random_state,
        features_to_poison=flip_feats,
        instances_by_feature=flip_instances_map,
    )
    X_poisoned, _, _ = noise_poisoning(
        X_poisoned,
        features_percentage,
        noise_percentage,
        noise_type,
        noise_scale,
        random_state,
        features_to_poison=noise_feats,
        instances_by_feature=noise_instances_map,
    )
    X_poisoned, _, _ = incompleteness_poisoning(
        X_poisoned,
        features_percentage,
        incompleteness_percentage,
        imputation_method,
        random_state,
        features_to_poison=inc_feats,
        instances_by_feature=inc_instances_map,
    )

    # Build combined quality vector aligned with X.columns considering cumulative poisoning per feature
    quality_by_feature = {}
    for col in X.columns:
        total_poisoning = 0.0
        if col in flip_feats:
            total_poisoning += float(flipping_percentage)
        if col in noise_feats:
            total_poisoning += float(noise_percentage)
        if col in inc_feats:
            total_poisoning += float(incompleteness_percentage)
        total_poisoning = min(1.0, total_poisoning)
        quality_by_feature[col] = max(0.0, 1.0 - total_poisoning)

    # Build row-wise quality vector r (aligned to X.index) based on actual changed cells
    # Treat NaNs as equal to avoid counting untouched missing values as poisoned
    same_mask = X_poisoned.eq(X) | (X_poisoned.isna() & X.isna())
    changed_mask = ~same_mask
    row_changes = changed_mask.sum(axis=1).reindex(X.index).fillna(0)
    n_features = X.shape[1]
    r = [max(0.0, 1.0 - (float(cnt) / n_features)) for cnt in row_changes]

    q = [quality_by_feature[col] for col in X.columns]
    
    return X_poisoned, q, r
