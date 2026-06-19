"""Training utilities for neural networks on tabular data."""

import copy
import random
import warnings
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
)
from torch.optim import SGD, Adam, AdamW, RMSprop
from torch.optim.lr_scheduler import (
    CosineAnnealingLR,
    CosineAnnealingWarmRestarts,
    ExponentialLR,
    LinearLR,
    ReduceLROnPlateau,
    SequentialLR,
    StepLR,
)
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler


def fit(
    model: nn.Module,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    X_test: Optional[np.ndarray] = None,
    y_test: Optional[np.ndarray] = None,
    task: str = "classification",
    epochs: int = 100,
    batch_size: Union[int, str] = 32,
    learning_rate: Union[float, str] = 0.001,
    optimizer: str = "adam",
    weight_decay: Union[float, str] = 0.0,
    l1_lambda: Union[float, str] = 0.0,
    early_stopping_patience: Union[int, str] = 10,
    early_stopping_min_delta: Union[float, str] = 1e-4,
    lr_scheduler: str = "none",
    lr_scheduler_patience: Union[int, str] = 5,
    lr_scheduler_factor: Union[float, str] = 0.5,
    warmup_epochs: Union[int, str] = 0,
    gradient_clip_value: Union[float, str] = 0.0,
    use_class_weights: str = "false",
    label_smoothing: Union[float, str] = 0.0,
    mixup_alpha: Union[float, str] = 0.0,
    # Curriculum learning
    use_curriculum: str = "false",
    sample_quality: Optional[np.ndarray] = None,
    curriculum_strategy: str = "exponential",
    # Gate layer
    use_gate: str = "false",
    gate_init: str = "random",
    feature_quality: Optional[np.ndarray] = None,
    gate_loss_weight: Union[float, str] = 0.01,
    gate_anchor_interval: Union[int, str] = 5,
    gate_quality_weighting: str = "linear",
    gate_loss_scheduler: str = "none",
    # Other
    device: str = "cpu",
    random_seed: Union[int, str] = 42,
    verbose: Union[int, str] = 1,
) -> Tuple[nn.Module, Dict[str, List[float]]]:
    """
    Train a PyTorch model on tabular data with comprehensive training features.

    Parameters
    ----------
    model : nn.Module
        PyTorch model to train.
    X_train : np.ndarray
        Training features.
    y_train : np.ndarray
        Training labels.
    X_val : np.ndarray
        Validation features.
    y_val : np.ndarray
        Validation labels.
    X_test : np.ndarray, optional
        Test features.
    y_test : np.ndarray, optional
        Test labels.
    task : str, default="classification"
        Task type: "classification" or "regression".
    epochs : int, default=100
        Maximum number of training epochs.
    batch_size : int or str, default=32
        Batch size for training.
    learning_rate : float or str, default=0.001
        Initial learning rate.
    optimizer : str, default="adam"
        Optimizer: "adam", "adamw", "sgd", "rmsprop".
    weight_decay : float or str, default=0.0
        L2 regularization strength (weight decay).
    l1_lambda : float or str, default=0.0
        L1 regularization strength.
    early_stopping_patience : int or str, default=10
        Number of epochs without improvement before stopping.
    early_stopping_min_delta : float or str, default=1e-4
        Minimum change to qualify as improvement.
    lr_scheduler : str, default="none"
        Learning rate scheduler: "none", "step", "plateau", "cosine",
        "cosine_restarts", "exponential", "linear".
    lr_scheduler_patience : int or str, default=5
        Patience for plateau scheduler.
    lr_scheduler_factor : float or str, default=0.5
        Factor for reducing learning rate.
    warmup_epochs : int or str, default=0
        Number of warmup epochs with linear LR increase.
    gradient_clip_value : float or str, default=0.0
        Maximum gradient norm (0 = no clipping).
    use_class_weights : str, default="false"
        Whether to use class weights for imbalanced classification.
    label_smoothing : float or str, default=0.0
        Label smoothing factor (0 = no smoothing).
    mixup_alpha : float or str, default=0.0
        Mixup augmentation alpha (0 = no mixup).
    use_curriculum : str, default="false"
        Enable curriculum learning (prioritize high-quality samples).
    sample_quality : np.ndarray, optional
        Sample quality scores (same length as training data). Higher = better quality.
    curriculum_strategy : str, default="exponential"
        Sampling strategy: "linear", "exponential", "step".
    use_gate : str, default="false"
        Enable input gate layer (learnable feature weighting).
    gate_init : str, default="random"
        Gate initialization: "random", "ones", "quality" (uses feature_quality).
    feature_quality : np.ndarray, optional
        Feature quality scores (length = input_dim). Used for gate init and loss weighting.
    gate_loss_weight : float or str, default=0.01
        Weight for gate proximal regularization loss.
    gate_anchor_interval : int or str, default=5
        Update gate anchor every N epochs.
    gate_quality_weighting : str, default="linear"
        How feature_quality weights anchor loss: "linear", "quadratic", "exp", "inv_exp".
    gate_loss_scheduler : str, default="none"
        Schedule gate loss weight: "none", "decay", "increase", "cosine".
    device : str, default="auto"
        Device: "auto", "cpu", "cuda", "mps".
    random_seed : int or str, default=42
        Random seed for reproducibility.
    verbose : int or str, default=1
        Verbosity level: 0=silent, 1=progress bar, 2=one line per epoch.

    Returns
    -------
    model : nn.Module
        Trained model.
    history : dict
        Dictionary containing training history with keys:
        - "train_loss", "val_loss", "test_loss"
        - "train_<metric>", "val_<metric>", "test_<metric>" (task-specific)
        - "learning_rate"
        - "gate_weights" (if use_gate=True)

    Examples
    --------
    >>> from frogdq.nn import build_model
    >>> model = build_model(input_dim=10, output_dim=2, hidden_neurons=128)
    >>> model, history = fit(
    ...     model, X_train, y_train, X_val, y_val,
    ...     task="classification", epochs=100, early_stopping_patience=10
    ... )

    >>> # With curriculum learning
    >>> sample_quality = np.array([0.8, 0.9, 0.5, ...])  # Quality per sample
    >>> model, history = fit(
    ...     model, X_train, y_train, X_val, y_val,
    ...     use_curriculum="true", sample_quality=sample_quality
    ... )

    >>> # With gate layer
    >>> feature_quality = np.array([0.9, 0.7, 0.8, ...])  # Quality per feature
    >>> model, history = fit(
    ...     model, X_train, y_train, X_val, y_val,
    ...     use_gate="true", gate_init="quality", feature_quality=feature_quality,
    ...     gate_loss_weight=0.01, gate_anchor_interval=5
    ... )
    """
    # Parse string parameters
    task = str(task).lower()
    optimizer = str(optimizer).lower()
    lr_scheduler = str(lr_scheduler).lower()
    device_str = str(device).lower()
    use_class_weights = str(use_class_weights).lower() in ("true", "1", "yes")
    use_curriculum = str(use_curriculum).lower() in ("true", "1", "yes")
    use_gate = str(use_gate).lower() in ("true", "1", "yes")
    curriculum_strategy = str(curriculum_strategy).lower()
    gate_init = str(gate_init).lower()
    gate_quality_weighting = str(gate_quality_weighting).lower()
    gate_loss_scheduler_str = str(gate_loss_scheduler).lower()

    batch_size = int(batch_size)
    learning_rate = float(learning_rate)
    weight_decay = float(weight_decay)
    l1_lambda = float(l1_lambda)
    early_stopping_patience = int(early_stopping_patience)
    early_stopping_min_delta = float(early_stopping_min_delta)
    lr_scheduler_patience = int(lr_scheduler_patience)
    lr_scheduler_factor = float(lr_scheduler_factor)
    warmup_epochs = int(warmup_epochs)
    gradient_clip_value = float(gradient_clip_value)
    label_smoothing = float(label_smoothing)
    mixup_alpha = float(mixup_alpha)
    gate_loss_weight_init = float(gate_loss_weight)
    gate_anchor_interval = int(gate_anchor_interval)
    random_seed = int(random_seed)
    verbose = int(verbose)

    # Set random seeds for reproducibility
    set_seed(random_seed)

    # Determine device
    device = _get_device(device_str)
    if verbose > 0:
        print(f"Using device: {device}")

    # Wrap model with gate layer if requested
    original_model = model
    if use_gate:
        input_dim = X_train.shape[1]
        model = GatedModel(
            base_model=model,
            input_dim=input_dim,
            gate_init=gate_init,
            feature_quality=feature_quality,
        )
        if verbose > 0:
            print(f"Added gate layer with '{gate_init}' initialization")

    # Move model to device
    model = model.to(device)

    # Prepare data
    train_loader, val_loader, test_loader = _prepare_dataloaders(
        X_train, y_train, X_val, y_val, X_test, y_test,
        batch_size=batch_size,
        task=task,
        use_curriculum=use_curriculum,
        sample_quality=sample_quality,
        curriculum_strategy=curriculum_strategy,
    )

    # Determine number of classes for classification
    num_classes = None
    if task == "classification":
        # Use all unique labels across train/val/test to avoid "target out of bounds" errors
        all_labels = [y_train, y_val]
        if y_test is not None:
            all_labels.append(y_test)
        num_classes = len(np.unique(np.concatenate(all_labels)))

    # Setup loss function
    criterion = _get_loss_function(
        task=task,
        num_classes=num_classes,
        y_train=y_train,
        use_class_weights=use_class_weights,
        label_smoothing=label_smoothing,
        device=device,
    )

    # Setup optimizer
    opt = _get_optimizer(
        optimizer_name=optimizer,
        model=model,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        use_gate=use_gate,  # FIX 1.4: gate gets its own param group with weight_decay=0
    )

    # Setup learning rate scheduler
    scheduler, warmup_scheduler = _get_lr_scheduler(
        optimizer=opt,
        scheduler_name=lr_scheduler,
        epochs=epochs,
        warmup_epochs=warmup_epochs,
        patience=lr_scheduler_patience,
        factor=lr_scheduler_factor,
    )

    # Setup gate loss scheduler
    gate_loss_scheduler_fn = _get_gate_loss_scheduler(
        scheduler_name=gate_loss_scheduler_str,
        initial_weight=gate_loss_weight_init,
        epochs=epochs,
    )

    # Initialize early stopping
    early_stopping = EarlyStopping(
        patience=early_stopping_patience,
        min_delta=early_stopping_min_delta,
        verbose=(verbose > 0),
    )

    # Initialize history with comprehensive metrics
    history = {
        "train_loss": [],
        "val_loss": [],
        "learning_rate": [],
    }
    if X_test is not None:
        history["test_loss"] = []

    # Add task-specific metrics to history
    if task == "classification":
        for split in ["train", "val"] + (["test"] if X_test is not None else []):
            history[f"{split}_accuracy"] = []
            history[f"{split}_f1"] = []
            history[f"{split}_precision"] = []
            history[f"{split}_recall"] = []
            history[f"{split}_auc"] = []
    else:  # regression
        for split in ["train", "val"] + (["test"] if X_test is not None else []):
            history[f"{split}_rmse"] = []
            history[f"{split}_mae"] = []
            history[f"{split}_r2"] = []

    # Add gate history if using gates
    if use_gate:
        history["gate_weights"] = []
        history["gate_loss_weight"] = []

    # Training loop
    best_model_state = None
    best_val_metric = float("-inf")  # Higher is better for both F1 and R2
    gate_quality_weights = None

    # Prepare gate quality weights and fixed anchor if using gate
    # FIX 1.2: gate_anchor is set once before training and never updated.
    # When quality is available it equals the quality vector so gates are
    # pulled toward their reliability values throughout the entire run.
    # When quality is absent we anchor to the initial gate values so the
    # gate at least stays near its starting point.
    gate_anchor = None
    if use_gate:
        if feature_quality is not None:
            gate_quality_weights = _compute_gate_quality_weights(
                feature_quality=feature_quality,
                weighting_strategy=gate_quality_weighting,
            )
            gate_quality_weights = torch.FloatTensor(gate_quality_weights).to(device)
            gate_anchor = torch.FloatTensor(feature_quality).to(device)
        else:
            gate_anchor = model.gate.data.clone().detach()

    # Determine primary metric
    primary_metric = "f1" if task == "classification" else "r2"

    for epoch in range(epochs):
        # Update gate loss weight
        current_gate_loss_weight = gate_loss_scheduler_fn(epoch) if use_gate else 0.0
        # FIX 1.2: anchor is fixed — no per-epoch reset here

        # Training phase
        train_loss, train_metrics = _train_epoch(
            model=model,
            loader=train_loader,
            criterion=criterion,
            optimizer=opt,
            device=device,
            task=task,
            num_classes=num_classes,
            l1_lambda=l1_lambda,
            gradient_clip_value=gradient_clip_value,
            mixup_alpha=mixup_alpha,
            use_gate=use_gate,
            gate_anchor=gate_anchor,
            gate_loss_weight=current_gate_loss_weight,
            gate_quality_weights=gate_quality_weights,
        )

        # Validation phase
        val_loss, val_metrics = _evaluate(
            model=model,
            loader=val_loader,
            criterion=criterion,
            device=device,
            task=task,
            num_classes=num_classes,
        )

        # Test phase (if provided)
        test_loss, test_metrics = None, None
        if test_loader is not None:
            test_loss, test_metrics = _evaluate(
                model=model,
                loader=test_loader,
                criterion=criterion,
                device=device,
                task=task,
                num_classes=num_classes,
            )

        # Update learning rate
        current_lr = opt.param_groups[0]["lr"]
        if warmup_scheduler is not None and epoch < warmup_epochs:
            warmup_scheduler.step()
        elif scheduler is not None:
            if isinstance(scheduler, ReduceLROnPlateau):
                scheduler.step(val_loss)
            else:
                scheduler.step()

        # Record history
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["learning_rate"].append(current_lr)
        if test_loss is not None:
            history["test_loss"].append(test_loss)

        # Record metrics
        for metric_name, metric_value in train_metrics.items():
            history[f"train_{metric_name}"].append(metric_value)
        for metric_name, metric_value in val_metrics.items():
            history[f"val_{metric_name}"].append(metric_value)
        if test_metrics is not None:
            for metric_name, metric_value in test_metrics.items():
                history[f"test_{metric_name}"].append(metric_value)

        # Record gate weights
        if use_gate:
            gate_weights = model.gate.data.cpu().numpy()  # FIX 1.1: Parameter vector, not diagonal of matrix
            history["gate_weights"].append(gate_weights.copy())
            history["gate_loss_weight"].append(current_gate_loss_weight)

        # Print progress
        if verbose > 0:
            msg = (
                f"Epoch {epoch+1}/{epochs} - "
                f"train_loss: {train_loss:.4f} - "
                f"train_{primary_metric}: {train_metrics[primary_metric]:.4f} - "
                f"val_loss: {val_loss:.4f} - "
                f"val_{primary_metric}: {val_metrics[primary_metric]:.4f}"
            )
            if test_loss is not None:
                msg += f" - test_loss: {test_loss:.4f} - test_{primary_metric}: {test_metrics[primary_metric]:.4f}"
            msg += f" - lr: {current_lr:.6f}"
            if use_gate:
                msg += f" - gate_loss_wt: {current_gate_loss_weight:.6f}"
            print(msg)

        # Save best model (higher is better for both F1 and R2)
        is_better = val_metrics[primary_metric] > best_val_metric
        if is_better:
            best_val_metric = val_metrics[primary_metric]
            best_model_state = copy.deepcopy(model.state_dict())

        # Early stopping check
        if early_stopping(val_loss, val_metrics[primary_metric], task):
            if verbose > 0:
                print(f"\nEarly stopping triggered at epoch {epoch+1}")
            break

    # Load best model
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
        if verbose > 0:
            print(f"\nRestored best model from epoch with val_{primary_metric}: {best_val_metric:.4f}")

    return model, history


class GatedModel(nn.Module):
    """Wrapper that adds a learnable gate layer to the input."""

    def __init__(
        self,
        base_model: nn.Module,
        input_dim: int,
        gate_init: str = "random",
        feature_quality: Optional[np.ndarray] = None,
    ):
        super().__init__()
        self.base_model = base_model
        # FIX 1.1: genuine D-dim vector, not a D×D matrix whose off-diagonals are dead weight
        self.gate = nn.Parameter(torch.empty(input_dim))

        if gate_init == "ones":
            self.gate.data.fill_(1.0)
        elif gate_init == "quality":
            if feature_quality is None:
                raise ValueError("feature_quality must be provided when gate_init='quality'")
            if len(feature_quality) != input_dim:
                raise ValueError(f"feature_quality length ({len(feature_quality)}) must match input_dim ({input_dim})")
            self.gate.data = torch.tensor(feature_quality, dtype=torch.float32)
        else:  # random: small noise around 1.0 (not xavier on a D×D matrix)
            self.gate.data = torch.ones(input_dim) + 0.01 * torch.randn(input_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gated_x = x * self.gate
        return self.base_model(gated_x)


class EarlyStopping:
    """Early stopping to stop training when validation metric doesn't improve."""

    def __init__(
        self,
        patience: int = 10,
        min_delta: float = 1e-4,
        verbose: bool = True,
    ):
        """Initialize early stopping."""
        self.patience = patience
        self.min_delta = min_delta
        self.verbose = verbose
        self.counter = 0
        self.best_loss = None
        self.best_metric = None

    def __call__(
        self,
        val_loss: float,
        val_metric: float,
        task: str,
    ) -> bool:
        """Check if training should stop."""
        # Higher is better for both F1 (classification) and R2 (regression)
        improved = (
            self.best_metric is None or
            val_metric > self.best_metric + self.min_delta
        )
        if improved:
            self.best_loss = val_loss
            self.best_metric = val_metric
            self.counter = 0
            return False
        else:
            self.counter += 1
            if self.verbose:
                print(f"  EarlyStopping counter: {self.counter}/{self.patience}")
            return self.counter >= self.patience


def set_seed(seed: int):
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    # Make PyTorch deterministic (may reduce performance)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def _get_device(device: str) -> torch.device:
    """Get PyTorch device."""
    if device == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        elif torch.backends.mps.is_available():
            return torch.device("mps")
        else:
            return torch.device("cpu")
    return torch.device(device)


def _prepare_dataloaders(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    X_test: Optional[np.ndarray],
    y_test: Optional[np.ndarray],
    batch_size: int,
    task: str,
    use_curriculum: bool,
    sample_quality: Optional[np.ndarray],
    curriculum_strategy: str,
) -> Tuple[DataLoader, DataLoader, Optional[DataLoader]]:
    """Prepare PyTorch DataLoaders."""
    # Convert to tensors
    X_train_t = torch.FloatTensor(X_train)
    X_val_t = torch.FloatTensor(X_val)

    if task == "classification":
        y_train_t = torch.LongTensor(y_train)
        y_val_t = torch.LongTensor(y_val)
    else:
        y_train_t = torch.FloatTensor(y_train).reshape(-1, 1)
        y_val_t = torch.FloatTensor(y_val).reshape(-1, 1)

    # Create datasets
    train_dataset = TensorDataset(X_train_t, y_train_t)
    val_dataset = TensorDataset(X_val_t, y_val_t)

    # Create train dataloader with curriculum learning if enabled
    if use_curriculum and sample_quality is not None:
        if len(sample_quality) != len(X_train):
            raise ValueError(f"sample_quality length ({len(sample_quality)}) must match X_train length ({len(X_train)})")

        # Compute sampling weights based on quality
        sampling_weights = _compute_curriculum_weights(
            sample_quality=sample_quality,
            strategy=curriculum_strategy,
        )
        sampler = WeightedRandomSampler(
            weights=sampling_weights,
            num_samples=len(sampling_weights),
            replacement=True,
        )
        train_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=sampler)
    else:
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    # Create val/test dataloaders (no shuffling)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    test_loader = None
    if X_test is not None and y_test is not None:
        X_test_t = torch.FloatTensor(X_test)
        if task == "classification":
            y_test_t = torch.LongTensor(y_test)
        else:
            y_test_t = torch.FloatTensor(y_test).reshape(-1, 1)
        test_dataset = TensorDataset(X_test_t, y_test_t)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader


def _compute_curriculum_weights(
    sample_quality: np.ndarray,
    strategy: str,
) -> np.ndarray:
    """Compute sampling weights for curriculum learning."""
    # Normalize quality to [0, 1]
    quality_min = sample_quality.min()
    quality_max = sample_quality.max()
    if quality_max > quality_min:
        quality_norm = (sample_quality - quality_min) / (quality_max - quality_min)
    else:
        quality_norm = np.ones_like(sample_quality)

    if strategy == "linear":
        weights = quality_norm + 0.1  # Add small offset to avoid zero weights
    elif strategy == "exponential":
        weights = np.exp(2 * quality_norm)  # Exponential emphasis on high quality
    elif strategy == "step":
        # Step function: high quality gets 2x, low quality gets 0.5x
        median_quality = np.median(quality_norm)
        weights = np.where(quality_norm >= median_quality, 2.0, 0.5)
    else:
        raise ValueError(f"Unknown curriculum strategy: {strategy}")

    # Normalize weights to sum to 1
    weights = weights / weights.sum()
    return weights


def _compute_gate_quality_weights(
    feature_quality: np.ndarray,
    weighting_strategy: str,
) -> np.ndarray:
    """
    Compute weights for gate anchor loss based on feature quality.

    High quality features get less anchor loss (more freedom to adapt).
    Formula: weight = f(1 - feature_quality)
    """
    # FIX 1.3: use absolute quality on its true [0,1] scale — no min-max normalization.
    # Min-max would destroy the absolute reliability information: a dataset where all
    # features are 0.91-0.95 reliable would treat the 0.91 feature as maximally bad,
    # indistinguishable from a genuinely 0.50-reliable feature in another dataset.
    inv_quality = 1.0 - np.clip(feature_quality, 0.0, 1.0)

    if weighting_strategy == "linear":
        weights = inv_quality
    elif weighting_strategy == "quadratic":
        weights = inv_quality ** 2
    elif weighting_strategy == "exp":
        # Exponential: exp(2 * inv_quality) - 1
        weights = np.exp(2 * inv_quality) - 1
    elif weighting_strategy == "inv_exp":
        # Inverse exponential: 1 - exp(-2 * inv_quality)
        weights = 1 - np.exp(-2 * inv_quality)
    else:
        raise ValueError(f"Unknown gate quality weighting strategy: {weighting_strategy}")

    return weights


def _get_gate_loss_scheduler(
    scheduler_name: str,
    initial_weight: float,
    epochs: int,
) -> callable:
    """Get gate loss weight scheduler function."""
    if scheduler_name == "none":
        return lambda epoch: initial_weight
    elif scheduler_name == "decay":
        # Linear decay to 0
        return lambda epoch: initial_weight * (1 - epoch / epochs)
    elif scheduler_name == "increase":
        # Linear increase
        return lambda epoch: initial_weight * (1 + epoch / epochs)
    elif scheduler_name == "cosine":
        # Cosine decay
        return lambda epoch: initial_weight * (1 + np.cos(np.pi * epoch / epochs)) / 2
    else:
        raise ValueError(f"Unknown gate loss scheduler: {scheduler_name}")


def _get_loss_function(
    task: str,
    num_classes: Optional[int],
    y_train: np.ndarray,
    use_class_weights: bool,
    label_smoothing: float,
    device: torch.device,
) -> nn.Module:
    """Get loss function based on task."""
    if task == "classification":
        # Compute class weights if requested
        class_weights = None
        if use_class_weights:
            unique_classes, class_counts = np.unique(y_train, return_counts=True)
            # Inverse frequency weighting
            class_weights = len(y_train) / (len(unique_classes) * class_counts)
            class_weights = torch.FloatTensor(class_weights).to(device)

        # Use CrossEntropyLoss for all classification (binary and multi-class)
        criterion = nn.CrossEntropyLoss(
            weight=class_weights,
            label_smoothing=label_smoothing,
        )
    else:
        # Regression
        criterion = nn.MSELoss()

    return criterion


def _get_optimizer(
    optimizer_name: str,
    model: nn.Module,
    learning_rate: float,
    weight_decay: float,
    use_gate: bool = False,
) -> torch.optim.Optimizer:
    """Get optimizer."""
    # FIX 1.4: gate has its own dedicated regularizer (L_gate) so it must NOT also
    # be penalized by weight decay or L1, which would fight the quality prior.
    if use_gate and isinstance(model, GatedModel):
        gate_params = [model.gate]
        base_params = [p for n, p in model.named_parameters() if n != "gate"]
        params = [
            {"params": base_params, "weight_decay": weight_decay},
            {"params": gate_params, "weight_decay": 0.0},
        ]
        wd = 0.0  # per-group weight_decay overrides the optimizer-level default
    else:
        params = list(model.parameters())
        wd = weight_decay

    if optimizer_name == "adam":
        return Adam(params, lr=learning_rate, weight_decay=wd)
    elif optimizer_name == "adamw":
        return AdamW(params, lr=learning_rate, weight_decay=wd)
    elif optimizer_name == "sgd":
        return SGD(params, lr=learning_rate, weight_decay=wd, momentum=0.9)
    elif optimizer_name == "rmsprop":
        return RMSprop(params, lr=learning_rate, weight_decay=wd)
    else:
        raise ValueError(f"Unknown optimizer: {optimizer_name}")


def _get_lr_scheduler(
    optimizer: torch.optim.Optimizer,
    scheduler_name: str,
    epochs: int,
    warmup_epochs: int,
    patience: int,
    factor: float,
) -> Tuple[Optional[Any], Optional[Any]]:
    """Get learning rate scheduler."""
    scheduler = None
    warmup_scheduler = None

    if scheduler_name == "none":
        pass
    elif scheduler_name == "step":
        scheduler = StepLR(optimizer, step_size=max(1, epochs // 3), gamma=factor)
    elif scheduler_name == "plateau":
        scheduler = ReduceLROnPlateau(
            optimizer, mode="min", patience=patience, factor=factor
        )
    elif scheduler_name == "cosine":
        scheduler = CosineAnnealingLR(optimizer, T_max=epochs - warmup_epochs)
    elif scheduler_name == "cosine_restarts":
        scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=max(1, epochs // 4))
    elif scheduler_name == "exponential":
        scheduler = ExponentialLR(optimizer, gamma=0.95)
    elif scheduler_name == "linear":
        scheduler = LinearLR(optimizer, start_factor=1.0, end_factor=0.1, total_iters=epochs)
    else:
        raise ValueError(f"Unknown scheduler: {scheduler_name}")

    # Add warmup if requested
    if warmup_epochs > 0:
        warmup_scheduler = LinearLR(
            optimizer, start_factor=0.1, end_factor=1.0, total_iters=warmup_epochs
        )

    return scheduler, warmup_scheduler


def _train_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    task: str,
    num_classes: Optional[int],
    l1_lambda: float,
    gradient_clip_value: float,
    mixup_alpha: float,
    use_gate: bool,
    gate_anchor: Optional[torch.Tensor],
    gate_loss_weight: float,
    gate_quality_weights: Optional[torch.Tensor],
) -> Tuple[float, Dict[str, float]]:
    """Train for one epoch."""
    model.train()
    total_loss = 0.0
    all_predictions = []
    all_targets = []
    all_probabilities = [] if task == "classification" else None

    for batch_X, batch_y in loader:
        batch_X = batch_X.to(device)
        batch_y = batch_y.to(device)

        # Apply mixup augmentation if enabled
        if mixup_alpha > 0 and task == "classification":
            batch_X, batch_y_a, batch_y_b, lam = _mixup_data(
                batch_X, batch_y, mixup_alpha, device
            )

        # Forward pass
        optimizer.zero_grad()
        outputs = model(batch_X)

        # Compute loss
        if mixup_alpha > 0 and task == "classification":
            loss = lam * criterion(outputs, batch_y_a) + (1 - lam) * criterion(outputs, batch_y_b)
        else:
            loss = criterion(outputs, batch_y)

        # Add L1 regularization if enabled
        # FIX 1.4: exclude gate — it has its own dedicated regularizer (L_gate)
        if l1_lambda > 0:
            if use_gate and isinstance(model, GatedModel):
                l1_norm = sum(p.abs().sum() for n, p in model.named_parameters() if n != "gate")
            else:
                l1_norm = sum(p.abs().sum() for p in model.parameters())
            loss = loss + l1_lambda * l1_norm

        # Add gate proximal loss if enabled
        if use_gate and gate_anchor is not None and gate_loss_weight > 0:
            # FIX 1.1: model.gate is now a D-dim Parameter vector, not a D×D Linear
            # FIX 1.2: gate_anchor is a fixed quality target set before training begins
            gate_diff = (model.gate - gate_anchor) ** 2

            # Weight by feature quality if provided
            if gate_quality_weights is not None:
                gate_diff = gate_diff * gate_quality_weights

            gate_loss = gate_diff.mean()
            loss = loss + gate_loss_weight * gate_loss

        # Backward pass
        loss.backward()

        # Gradient clipping
        if gradient_clip_value > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip_value)

        optimizer.step()

        # Record loss and predictions
        total_loss += loss.item() * batch_X.size(0)

        with torch.no_grad():
            if task == "classification":
                probs = torch.softmax(outputs, dim=1)
                preds = outputs.argmax(dim=1)
                if num_classes == 2:
                    # For binary, store probability of positive class
                    all_probabilities.extend(probs[:, 1].cpu().numpy())
                else:
                    # For multi-class, store all probabilities
                    all_probabilities.extend(probs.cpu().numpy())
                all_predictions.extend(preds.cpu().numpy())
                all_targets.extend(batch_y.cpu().numpy())
            else:
                all_predictions.extend(outputs.cpu().numpy().flatten())
                all_targets.extend(batch_y.cpu().numpy().flatten())

    # Compute average loss
    avg_loss = total_loss / len(loader.dataset)

    # Compute metrics
    metrics = _compute_metrics(
        np.array(all_targets),
        np.array(all_predictions),
        task,
        num_classes,
        all_probabilities=np.array(all_probabilities) if all_probabilities else None,
    )

    return avg_loss, metrics


def _evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    task: str,
    num_classes: Optional[int],
) -> Tuple[float, Dict[str, float]]:
    """Evaluate model."""
    model.eval()
    total_loss = 0.0
    all_predictions = []
    all_targets = []
    all_probabilities = [] if task == "classification" else None

    with torch.no_grad():
        for batch_X, batch_y in loader:
            batch_X = batch_X.to(device)
            batch_y = batch_y.to(device)

            # Forward pass
            outputs = model(batch_X)

            # Compute loss
            loss = criterion(outputs, batch_y)

            # Record loss and predictions
            total_loss += loss.item() * batch_X.size(0)

            if task == "classification":
                probs = torch.softmax(outputs, dim=1)
                preds = outputs.argmax(dim=1)
                if num_classes == 2:
                    # For binary, store probability of positive class
                    all_probabilities.extend(probs[:, 1].cpu().numpy())
                else:
                    # For multi-class, store all probabilities
                    all_probabilities.extend(probs.cpu().numpy())
                all_predictions.extend(preds.cpu().numpy())
                all_targets.extend(batch_y.cpu().numpy())
            else:
                all_predictions.extend(outputs.cpu().numpy().flatten())
                all_targets.extend(batch_y.cpu().numpy().flatten())

    # Compute average loss
    avg_loss = total_loss / len(loader.dataset)

    # Compute metrics
    metrics = _compute_metrics(
        np.array(all_targets),
        np.array(all_predictions),
        task,
        num_classes,
        all_probabilities=np.array(all_probabilities) if all_probabilities else None,
    )

    return avg_loss, metrics


def _compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    task: str,
    num_classes: Optional[int],
    all_probabilities: Optional[np.ndarray] = None,
) -> Dict[str, float]:
    """Compute comprehensive task-specific metrics."""
    metrics = {}

    if task == "classification":
        # F1 score (primary metric)
        if num_classes == 2:
            metrics["f1"] = f1_score(y_true, y_pred, average="binary", zero_division=0)
            metrics["precision"] = precision_score(y_true, y_pred, average="binary", zero_division=0)
            metrics["recall"] = recall_score(y_true, y_pred, average="binary", zero_division=0)
        else:
            metrics["f1"] = f1_score(y_true, y_pred, average="macro", zero_division=0)
            metrics["precision"] = precision_score(y_true, y_pred, average="macro", zero_division=0)
            metrics["recall"] = recall_score(y_true, y_pred, average="macro", zero_division=0)

        # Accuracy
        metrics["accuracy"] = accuracy_score(y_true, y_pred)

        # AUC-ROC (if probabilities available)
        if all_probabilities is not None:
            try:
                if num_classes == 2:
                    # Binary: use probabilities for positive class
                    metrics["auc"] = roc_auc_score(y_true, all_probabilities)
                else:
                    # Multi-class: use one-vs-rest
                    metrics["auc"] = roc_auc_score(
                        y_true, all_probabilities,
                        multi_class="ovr", average="macro"
                    )
            except ValueError:
                # AUC computation can fail if only one class present
                metrics["auc"] = 0.0
        else:
            metrics["auc"] = 0.0

    else:  # regression
        # RMSE (primary metric)
        metrics["rmse"] = np.sqrt(mean_squared_error(y_true, y_pred))

        # MAE
        metrics["mae"] = mean_absolute_error(y_true, y_pred)

        # R2 score
        metrics["r2"] = r2_score(y_true, y_pred)

    return metrics


def _mixup_data(
    x: torch.Tensor,
    y: torch.Tensor,
    alpha: float,
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, float]:
    """Apply mixup augmentation."""
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1
    batch_size = x.size(0)
    index = torch.randperm(batch_size).to(device)
    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam


def predict(
    model: nn.Module,
    X: np.ndarray,
    task: str = "classification",
    batch_size: int = 32,
    device: str = "auto",
) -> np.ndarray:
    """
    Make predictions with a trained model.

    Parameters
    ----------
    model : nn.Module
        Trained PyTorch model.
    X : np.ndarray
        Input features.
    task : str, default="classification"
        Task type: "classification" or "regression".
    batch_size : int, default=32
        Batch size for prediction.
    device : str, default="auto"
        Device: "auto", "cpu", "cuda", "mps".

    Returns
    -------
    predictions : np.ndarray
        Model predictions.
    """
    device = _get_device(device)
    model = model.to(device)
    model.eval()
    X_tensor = torch.FloatTensor(X)
    dataset = TensorDataset(X_tensor)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    predictions = []
    with torch.no_grad():
        for (batch_X,) in loader:
            batch_X = batch_X.to(device)
            outputs = model(batch_X)
            if task == "classification":
                if outputs.shape[1] == 1:
                    preds = (torch.sigmoid(outputs.squeeze()) > 0.5).long()
                else:
                    preds = outputs.argmax(dim=1)
                predictions.extend(preds.cpu().numpy())
            else:
                predictions.extend(outputs.cpu().numpy().flatten())
    return np.array(predictions)


def predict_proba(
    model: nn.Module,
    X: np.ndarray,
    batch_size: int = 32,
    device: str = "auto",
) -> np.ndarray:
    """
    Predict class probabilities for classification tasks.

    Parameters
    ----------
    model : nn.Module
        Trained PyTorch model.
    X : np.ndarray
        Input features.
    batch_size : int, default=32
        Batch size for prediction.
    device : str, default="auto"
        Device: "auto", "cpu", "cuda", "mps".

    Returns
    -------
    probabilities : np.ndarray
        Class probabilities with shape (n_samples, n_classes).
    """
    device = _get_device(device)
    model = model.to(device)
    model.eval()
    X_tensor = torch.FloatTensor(X)
    dataset = TensorDataset(X_tensor)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    probabilities = []
    with torch.no_grad():
        for (batch_X,) in loader:
            batch_X = batch_X.to(device)
            outputs = model(batch_X)
            if outputs.shape[1] == 1:
                probs = torch.sigmoid(outputs.squeeze())
                probs = torch.stack([1 - probs, probs], dim=1)
            else:
                probs = torch.softmax(outputs, dim=1)
            probabilities.extend(probs.cpu().numpy())
    return np.array(probabilities)
