import numpy as np
import pandas as pd
import random
from typing import Tuple, List, Sequence, Optional


def flipping_poisoning(
    X: pd.DataFrame,
    features_percentage: float,
    poisoning_percentage: float,
    random_state: int = None,
    *,
    features_to_poison: Optional[Sequence[str]] = None,
) -> Tuple[pd.DataFrame, List[float]]:
    """
    Apply flipping poisoning to the dataset.
    
    Args:
        X: Original dataframe
        features_percentage: Percentage of features to poison (0-1)
        poisoning_percentage: Percentage of instances to poison per feature (0-1)
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
    
    for feature in features_to_poison:
        # Get unique values for this feature
        unique_values = X[feature].unique()
        
        # Skip if only one unique value
        if len(unique_values) <= 1:
            continue
            
        # Calculate number of instances to poison for this feature
        n_instances_to_poison = max(1, int(n_instances * poisoning_percentage))
        
        # Randomly select instances to poison
        instances_to_poison = random.sample(list(X.index), n_instances_to_poison) 
        
        for instance_idx in instances_to_poison:
            current_value = X.loc[instance_idx, feature]
            
            # Find other possible values (different from current)
            other_values = [val for val in unique_values if val != current_value]
            
            if other_values:
                # Randomly select a different value
                new_value = random.choice(other_values)
                X_poisoned.loc[instance_idx, feature] = new_value
    
    # Build ordered q aligned with X.columns
    q = [feature_quality[col] for col in X.columns]

    return X_poisoned, q


def noise_poisoning(
    X: pd.DataFrame,
    features_percentage: float,
    poisoning_percentage: float,
    noise_type: str = 'gaussian',
    noise_scale: float = 0.1,
    random_state: int = None,
    *,
    features_to_poison: Optional[Sequence[str]] = None,
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
    
    for feature in features_to_poison:
        # Skip non-numeric features for noise poisoning
        if not pd.api.types.is_numeric_dtype(X[feature]):
            continue
            
        # Calculate number of instances to poison for this feature
        n_instances_to_poison = max(1, int(n_instances * poisoning_percentage))
        
        # Randomly select instances to poison
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
    
    # Build ordered q aligned with X.columns
    q = [feature_quality[col] for col in X.columns]

    return X_poisoned, q


def incompleteness_poisoning(
    X: pd.DataFrame,
    features_percentage: float,
    poisoning_percentage: float,
    imputation_method: str = 'mean',
    random_state: int = None,
    *,
    features_to_poison: Optional[Sequence[str]] = None,
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
    feature_quality = {col: 1.0 for col in X.columns}
    for col in features_to_poison:
        feature_quality[col] = max(0.0, 1.0 - float(poisoning_percentage))
    
    for feature in features_to_poison:
        # Calculate number of instances to poison for this feature
        n_instances_to_poison = max(1, int(n_instances * poisoning_percentage))
        
        # Randomly select instances to poison
        instances_to_poison = random.sample(list(X.index), n_instances_to_poison)
        
        # Store original values for imputation
        original_values = X.loc[instances_to_poison, feature].copy()
        
        # Set selected instances to NaN
        X_poisoned.loc[instances_to_poison, feature] = np.nan
        
        # Calculate imputation value based on remaining non-NaN values
        remaining_values = X_poisoned[feature].dropna()
        
        if len(remaining_values) > 0:
            if imputation_method == 'mean':
                imputation_value = remaining_values.mean()
            elif imputation_method == 'median':
                imputation_value = remaining_values.median()
            elif imputation_method == 'mode':
                imputation_value = remaining_values.mode().iloc[0] if not remaining_values.mode().empty else remaining_values.mean()
            else:
                raise ValueError(f"Unknown imputation method: {imputation_method}")
            
            # Apply imputation
            X_poisoned.loc[instances_to_poison, feature] = imputation_value
            
            # Store imputation statistics
            imputation_stats[feature] = {
                'imputation_value': imputation_value,
                'imputation_method': imputation_method,
                'n_poisoned_instances': len(instances_to_poison),
                'original_values': original_values.tolist()
            }
        else:
            # If all values are NaN, use the mean of original values
            imputation_value = original_values.mean()
            X_poisoned.loc[instances_to_poison, feature] = imputation_value
            
            imputation_stats[feature] = {
                'imputation_value': imputation_value,
                'imputation_method': 'fallback_mean',
                'n_poisoned_instances': len(instances_to_poison),
                'original_values': original_values.tolist()
            }
    
    # Build ordered q aligned with X.columns
    q = [feature_quality[col] for col in X.columns]

    return X_poisoned, q


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

    # Sample without replacement for disjoint sets
    selected = random.sample(all_features, min(3 * n_per_set, n_features_total))
    flip_features = selected[0:n_per_set]
    noise_features = selected[n_per_set:2 * n_per_set]
    incomplete_features = selected[2 * n_per_set:3 * n_per_set]

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

    # Apply each poisoning on its feature set
    X_poisoned, _ = flipping_poisoning(
        X, features_percentage, flipping_percentage, random_state, features_to_poison=flip_feats
    )
    X_poisoned, _ = noise_poisoning(
        X_poisoned,
        features_percentage,
        noise_percentage,
        noise_type,
        noise_scale,
        random_state,
        features_to_poison=noise_feats,
    )
    X_poisoned, _ = incompleteness_poisoning(
        X_poisoned,
        features_percentage,
        incompleteness_percentage,
        imputation_method,
        random_state,
        features_to_poison=inc_feats,
    )

    # Build combined quality vector aligned with X.columns
    quality_by_feature = {col: 1.0 for col in X.columns}
    for col in flip_feats:
        quality_by_feature[col] = max(0.0, 1.0 - float(flipping_percentage))
    for col in noise_feats:
        quality_by_feature[col] = max(0.0, 1.0 - float(noise_percentage))
    for col in inc_feats:
        quality_by_feature[col] = max(0.0, 1.0 - float(incompleteness_percentage))

    q = [quality_by_feature[col] for col in X.columns]
    return X_poisoned, q
