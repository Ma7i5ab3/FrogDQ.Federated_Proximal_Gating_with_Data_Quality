# pylint: disable=too-many-arguments,too-many-locals,too-many-statements,invalid-name

import random
from typing import Any, Literal, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from loguru import logger
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, roc_auc_score
from torch.utils.data import DataLoader, TensorDataset

__all__ = ["build_model", "train", "evaluate"]

# -------------------------
# Reproducibility
# -------------------------


def _set_seed(random_state: int = 42) -> None:
    """Set Python, NumPy and Torch seeds for full determinism."""
    random.seed(random_state)
    np.random.seed(random_state)
    torch.manual_seed(random_state)
    torch.cuda.manual_seed_all(random_state)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# -------------------------
# Model building blocks
# -------------------------


class FeedForward(nn.Module):
    """Generic feed-forward classifier: Linear or MLP."""

    def __init__(
        self,
        input_dim: int,
        hidden: Optional[tuple[int, ...]],
        output_dim: int,
        activation: str = "relu",
        dropout: float = 0.0,
    ):
        super().__init__()
        layers = []
        dims = (input_dim,) + (hidden or ())
        act = nn.ReLU if activation.lower() == "relu" else nn.GELU
        for din, dout in zip(dims[:-1], dims[1:]):
            layers += [nn.Linear(din, dout), act()]
            if dropout > 0:
                layers += [nn.Dropout(dropout)]
        layers += [nn.Linear(dims[-1], output_dim)]
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class FrogGate(nn.Module):
    """Per-feature gate vector g (learnable scale)."""

    def __init__(self, input_dim: int, init: float = 1.0):
        super().__init__()
        self.gates = nn.Parameter(torch.full((input_dim,), float(init)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.gates


class FrogModel(nn.Module):
    """Gate ∘ BaseModel wrapper."""

    def __init__(self, gate: FrogGate, base: nn.Module):
        super().__init__()
        self.gate = gate
        self.base = base

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.base(self.gate(x))


def build_model(
    input_dim: int,
    output_dim: int,
    *,
    arch: Literal["linear", "mlp"] = "mlp",
    hidden: Optional[tuple[int, ...]] = (128,),
    activation: Literal["relu", "gelu"] = "relu",
    dropout: float = 0.0,
    use_frogdq: bool = False,
    gate_init: float = 1.0,
    random_state: int = 42,
) -> nn.Module:
    """
    Build a classifier.

    Parameters
    ----------
    input_dim : int
        Number of input features.
    output_dim : int
        Number of output classes (logits).
    arch : {"linear","mlp"}, default="mlp"
        Network architecture; "linear" uses no hidden layers.
    hidden : tuple[int,...] or None, default=(128,)
        Hidden layer sizes for MLP; ignored if arch="linear".
    activation : {"relu","gelu"}, default="relu"
        Activation function for hidden layers.
    dropout : float, default=0.0
        Dropout probability after each hidden layer.
    use_frogdq : bool, default=False
        If True, prepend a learnable FrogDQ gate over inputs.
    gate_init : float, default=1.0
        Initial value for each gate if use_frogdq=True.
    random_state : int, default=42
        Seed for deterministic initialization.

    Returns
    -------
    nn.Module
        The constructed torch model.
    """
    _set_seed(random_state)
    base = FeedForward(
        input_dim, None if arch == "linear" else hidden, output_dim, activation, dropout
    )
    return FrogModel(FrogGate(input_dim, gate_init), base) if use_frogdq else base


# -------------------------
# FrogDQ prior helpers
# -------------------------


def _dirichlet_kl_prior_loss(
    gates: torch.Tensor,
    q_vec: torch.Tensor,
    tau: float = 1.0,
    eps: float = 1e-8,
) -> torch.Tensor:
    """KL(softmax(g/tau) || softmax(q/tau))."""
    p = torch.softmax(gates / tau, dim=0)
    pi = torch.softmax(q_vec / tau, dim=0)
    return torch.sum(p * (torch.log(p + eps) - torch.log(pi + eps)))


# -------------------------
# Metrics helpers
# -------------------------


@torch.no_grad()
def _eval_on_loader(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> dict[str, float]:
    """Compute loss/acc/bal_acc/f1/auc on a DataLoader."""
    model.eval()
    total_loss, nobs = 0.0, 0
    all_logits, all_y = [], []
    for xb, yb in loader:
        xb, yb = xb.to(device), yb.to(device)
        logits = model(xb)
        loss = criterion(logits, yb)
        total_loss += loss.item() * xb.size(0)
        nobs += xb.size(0)
        all_logits.append(logits.detach().cpu())
        all_y.append(yb.detach().cpu())
    y_true = torch.cat(all_y).numpy()
    logits = torch.cat(all_logits)
    y_pred = logits.argmax(dim=1).numpy()
    probs = torch.softmax(logits, dim=1).numpy()

    acc = accuracy_score(y_true, y_pred)
    bal_acc = balanced_accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average="macro")

    try:
        n_classes = probs.shape[1]
        if n_classes == 2:
            auc = roc_auc_score(y_true, probs[:, 1])
        else:
            auc = roc_auc_score(y_true, probs, multi_class="ovr", average="macro")
    except Exception:
        auc = np.nan

    return {
        "loss": total_loss / max(1, nobs),
        "accuracy": acc,
        "balanced_accuracy": bal_acc,
        "f1": f1,
        "auc_roc": auc,
    }


# -------------------------
# Public API: evaluate
# -------------------------


@torch.no_grad()
def evaluate(
    model: nn.Module,
    X: torch.Tensor,
    y: torch.Tensor,
    *,
    batch_size: int = 512,
    device: Optional[torch.device] = None,
    random_state: int = 42,
) -> dict[str, float]:
    """
    Evaluate a trained model on a dataset.

    Parameters
    ----------
    model : nn.Module
        Trained torch model.
    X : Tensor (N, D)
        Features.
    y : Tensor (N,)
        Integer class labels.
    batch_size : int, default=512
        Evaluation batch size.
    device : torch.device or None, default=None
        Device to run on; defaults to model's device.
    random_state : int, default=42
        Seed to ensure deterministic scoring.

    Returns
    -------
    dict
        Metrics: loss, accuracy, balanced_accuracy, f1, auc_roc.
    """
    _set_seed(random_state)
    device = device or next(model.parameters()).device
    loader = DataLoader(TensorDataset(X, y), batch_size=batch_size, shuffle=False)
    criterion = nn.CrossEntropyLoss()
    return _eval_on_loader(model, loader, criterion, device)


# -------------------------
# Public API: train
# -------------------------


def train(
    model: nn.Module,
    X_train: torch.Tensor,
    y_train: torch.Tensor,
    X_val: torch.Tensor,
    y_val: torch.Tensor,
    *,
    epochs: int = 100,
    batch_size: int = 128,
    lr: float = 1e-3,
    weight_decay: float = 0.0,
    optimizer: Literal["adam", "sgd"] = "adam",
    early_stop: bool = True,
    es_patience: int = 50,
    es_min_delta: float = 1e-4,
    device: Optional[torch.device] = None,
    verbose: bool = False,
    log_every: int = 5,
    random_state: int = 42,
    # FrogDQ options
    q_vec: Optional[torch.Tensor] = None,
    frogdq_mode: Literal["none", "inertia", "gaussian", "dirichlet"] = "none",
    lambda_prox: float = 0.1,
    lambda_gaussian_prior: float = 0.1,
    lambda_dirichlet_kl: float = 0.1,
    kl_temperature: float = 1.0,
    eps: float = 1e-8,
    normalize_inertia_by_mean: bool = True,
    # NEW: temperature scheduling for Frog regularization
    frog_temperature: float = 1.0,
    frog_temp_invert: bool = False,
    frog_temp_tau: float = 1.0,
) -> dict[str, list]:
    """
    Train a classifier with optional FrogDQ losses. Early stopping uses validation F1 (macro).

    Temperature schedule (regularization impact)
    -------------------------------------------
    Let phase = (epoch_idx / (epochs-1)) ** frog_temp_tau.
    scale α_t = 0.5 * (1 + cos(2π * phase)) ∈ [0,1].
    - Default: α_t starts high (strong reg), decreases to low mid-training, then rises again.
    - If frog_temp_invert=True: use (1 - α_t).
    Final multiplier: reg_scale = frog_temperature * α_t.

    Parameters
    ----------
    model : nn.Module
        Model built via `build_model`.
    X_train, y_train : Tensor
        Training features/labels.
    X_val, y_val : Tensor
        Validation features/labels.
    epochs : int, default=100
        Maximum training epochs.
    batch_size : int, default=128
        Minibatch size.
    lr : float, default=1e-3
        Learning rate.
    weight_decay : float, default=0.0
        Optimizer L2 weight decay.
    optimizer : {"adam","sgd"}, default="adam"
        Optimizer choice.
    early_stop : bool, default=True
        Enable early stopping on val F1.
    es_patience : int, default=20
        Patience (epochs) for early stopping.
    es_min_delta : float, default=1e-4
        Minimum val F1 improvement to reset patience.
    device : torch.device or None, default=None
        Device to run on; autodetects GPU if available.
    verbose : bool, default=False
        If True, log epoch metrics with loguru.
    log_every : int, default=1
        Log every k epochs when verbose.
    random_state : int, default=42
        Seed for deterministic training.
    q_vec : Tensor or None, default=None
        Feature-quality vector for FrogDQ (length = D).
    frogdq_mode : {"none","inertia","gaussian","dirichlet"}, default="none"
        Which FrogDQ regularization to apply.
    lambda_prox : float, default=0.1
        Strength of inertia (trust region) term.
    lambda_gaussian_prior : float, default=0.1
        Strength of Gaussian prior (L2 to q_vec).
    lambda_dirichlet_kl : float, default=0.1
        Strength of Dirichlet/KL prior.
    kl_temperature : float, default=1.0
        Temperature for softmax in KL prior.
    eps : float, default=1e-8
        Numerical stability constant.
    normalize_inertia_by_mean : bool, default=True
        If True, scale inertia weights by their mean.
    frog_temperature : float, default=1.0
        Global scale applied to ALL FrogDQ regularization terms.
    frog_temp_invert : bool, default=False
        Invert the default schedule (low→high→low instead of high→low→high).
    frog_temp_tau : float, default=1.0
        Speed/shape control for the schedule (τ>1 compresses early change).

    Returns
    -------
    dict
        History with per-epoch metrics and frog gates (if enabled):
        keys = train_loss, val_loss, train_acc, val_acc, train_bal_acc,
               val_bal_acc, train_f1, val_f1, train_auc, val_auc, frog_gates.
    """
    _set_seed(random_state)

    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(TensorDataset(X_val, y_val), batch_size=batch_size, shuffle=False)

    criterion = nn.CrossEntropyLoss()
    if optimizer == "sgd":
        opt: torch.optim.Optimizer = optim.SGD(
            model.parameters(), lr=lr, momentum=0.9, weight_decay=weight_decay
        )
    else:
        opt: torch.optim.Optimizer = optim.Adam(
            model.parameters(), lr=lr, weight_decay=weight_decay
        )

    history: dict[str, Any] = {
        "train_loss": [],
        "val_loss": [],
        "train_acc": [],
        "val_acc": [],
        "train_bal_acc": [],
        "val_bal_acc": [],
        "train_f1": [],
        "val_f1": [],
        "train_auc": [],
        "val_auc": [],
        # Per-epoch snapshot of FrogDQ gates (np.ndarray); None when FrogDQ is off.
        "frog_gates": (
            [] if (hasattr(model, "gate") and q_vec is not None and frogdq_mode != "none") else None
        ),
    }

    best_metric = -float("inf")
    best_state = None
    patience_left = es_patience

    use_frog = hasattr(model, "gate") and q_vec is not None and frogdq_mode != "none"
    if use_frog:
        q_vec = q_vec.to(device)
        g_prev = model.gate.gates.detach().clone()
        w_inertia = 1.0 - q_vec
        if normalize_inertia_by_mean:
            w_inertia = w_inertia / (w_inertia.mean() + eps)

    # Helper for temperature schedule
    def _reg_scale_for_epoch(e: int) -> float:
        if epochs <= 1:
            phase = 1.0
        else:
            phase = ((e - 1) / (epochs - 1)) ** max(1e-8, frog_temp_tau)
        alpha = 0.5 * (1.0 + np.cos(2.0 * np.pi * phase))  # in [0,1], high→low→high
        if frog_temp_invert:
            alpha = 1.0 - alpha
        return float(frog_temperature * alpha)

    for epoch in range(1, epochs + 1):
        model.train()
        running_loss, nobs = 0.0, 0
        reg_scale = _reg_scale_for_epoch(epoch)

        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            logits = model(xb)
            loss = criterion(logits, yb)

            if use_frog and reg_scale > 0.0:
                g_curr = model.gate.gates
                if frogdq_mode in {"temp", "inertia", "gaussian", "dirichlet"}:
                    if frogdq_mode == 'temp':
                        loss = (
                            loss
                            + reg_scale * lambda_prox * (((g_curr - g_prev) ** 2) * w_inertia).sum()
                        )
                    else:
                        loss = (
                            loss
                            + lambda_prox * (((g_curr - g_prev) ** 2) * w_inertia).sum()
                        )
                if frogdq_mode == "gaussian":
                    loss = loss + lambda_gaussian_prior * ((g_curr - q_vec) ** 2).sum()
                if frogdq_mode == "dirichlet":
                    loss = loss + lambda_dirichlet_kl * _dirichlet_kl_prior_loss(
                        g_curr, q_vec, tau=kl_temperature, eps=eps
                    )

            loss.backward()
            opt.step()
            if use_frog:
                g_prev = model.gate.gates.detach().clone()
            running_loss += loss.item() * xb.size(0)
            nobs += xb.size(0)

        train_loss = running_loss / max(1, nobs)
        train_metrics = _eval_on_loader(model, train_loader, criterion, device)
        val_metrics = _eval_on_loader(model, val_loader, criterion, device)

        # --- History (metrics)
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_metrics["loss"])
        history["train_acc"].append(train_metrics["accuracy"])
        history["val_acc"].append(val_metrics["accuracy"])
        history["train_bal_acc"].append(train_metrics["balanced_accuracy"])
        history["val_bal_acc"].append(val_metrics["balanced_accuracy"])
        history["train_f1"].append(train_metrics["f1"])
        history["val_f1"].append(val_metrics["f1"])
        history["train_auc"].append(train_metrics["auc_roc"])
        history["val_auc"].append(val_metrics["auc_roc"])

        # --- History (FrogDQ gates per epoch, as NumPy)
        if use_frog and history["frog_gates"] is not None:
            g_now = model.gate.gates.detach().cpu().numpy().copy()
            history["frog_gates"].append(g_now)

        if verbose and (epoch % log_every == 0):
            logger.info(
                f"Epoch {epoch:3d} | reg_scale={reg_scale:.3f} | "
                f"train: loss={train_loss:.4f}, acc={train_metrics['accuracy']:.4f}, "
                f"bal_acc={train_metrics['balanced_accuracy']:.4f}, "
                f"f1={train_metrics['f1']:.4f}, auc={train_metrics['auc_roc']:.4f} | "
                f"val: loss={val_metrics['loss']:.4f}, acc={val_metrics['accuracy']:.4f}, "
                f"bal_acc={val_metrics['balanced_accuracy']:.4f}, "
                f"f1={val_metrics['f1']:.4f}, auc={val_metrics['auc_roc']:.4f}"
            )

        # Early stopping on validation F1 (macro)
        val_score = val_metrics["f1"]
        if val_score > best_metric + es_min_delta:
            best_metric = val_score
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            patience_left = es_patience
        else:
            patience_left -= 1
            if early_stop and patience_left <= 0:
                if verbose:
                    logger.info(f"Early stopping at epoch {epoch} (best val F1={best_metric:.4f}).")
                break

    if best_state is not None:
        model.load_state_dict({k: v.to(device) for k, v in best_state.items()})
    return history
