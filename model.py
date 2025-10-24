# pylint: disable=too-many-arguments,too-many-locals,too-many-statements,invalid-name

import random
from dataclasses import dataclass
from typing import Any, Literal, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from loguru import logger
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, roc_auc_score
from torch.utils.data import DataLoader, TensorDataset

__all__ = ["build_model", "train", "evaluate", "parse_frogdq_mode"]

# -------------------------
# FrogDQ module parsing
# -------------------------

FROGDQ_ALLOWED_MODULES: tuple[str, ...] = (
    "inertia",
    "temp",
    "cosine",
    "gaussian",
    "dirichlet",
    "samplewise",
    "q",
)
_MODULE_ORDER = {name: idx for idx, name in enumerate(FROGDQ_ALLOWED_MODULES)}
_GATE_RELEVANT_MODULES = frozenset({"inertia", "gaussian", "dirichlet", "temp", "cosine", "q"})
_ALIASES = {
    "sample-wise": "samplewise",
    "sample_wise": "samplewise",
    "sample": "samplewise",
    "samplew": "samplewise",
    "sampleweights": "samplewise",
    "tempcos": "cosine",
    "cos": "cosine",
    "cosine": "cosine",
    "qinit": "q",
    "quality": "q",
}
_DISPLAY_NAMES = {
    "cosine": "temp_cos",
}


@dataclass(frozen=True)
class FrogDQModules:
    """Structured representation of requested/active FrogDQ modules."""

    requested: tuple[str, ...]
    active: frozenset[str]
    canonical: str


def parse_frogdq_mode(mode: str) -> FrogDQModules:
    """
    Parse a FrogDQ mode string into modular components.

    Parameters
    ----------
    mode : str
        Mode string composed by joining module names with underscores.

    Returns
    -------
    FrogDQModules
        requested : tuple[str, ...]
            Normalized module names explicitly requested by the user.
        active : frozenset[str]
            Module set closed under dependencies (e.g., 'temp' enables 'inertia').
        canonical : str
            Canonical underscore-joined representation of the requested modules,
            or "none" when no module is active.
    """

    if mode is None:
        mode = "none"
    normalized = mode.strip().lower()
    if not normalized or normalized == "none":
        return FrogDQModules(requested=tuple(), active=frozenset(), canonical="none")

    # Normalise separators/aliases before splitting.
    normalized = normalized.replace("-", "_")
    # Legacy combined names.
    normalized = normalized.replace("temp_cos", "cosine")

    raw_tokens = [tok for tok in normalized.split("_") if tok]
    resolved: list[str] = []
    for token in raw_tokens:
        token = _ALIASES.get(token, token)
        if token not in FROGDQ_ALLOWED_MODULES:
            if token == "none":
                continue
            raise ValueError(
                f"Unknown FrogDQ module '{token}' derived from mode '{mode}'. "
                f"Allowed modules: {', '.join(FROGDQ_ALLOWED_MODULES)}"
            )
        if token not in resolved:
            resolved.append(token)

    requested = tuple(sorted(resolved, key=lambda name: _MODULE_ORDER[name]))
    if not requested:
        return FrogDQModules(requested=tuple(), active=frozenset(), canonical="none")

    if "temp" in requested and "cosine" in requested:
        raise ValueError("Modules 'temp' (exponential schedule) and 'cosine' are mutually exclusive.")

    active = set(requested)
    if active & {"temp", "cosine", "gaussian", "dirichlet", "samplewise", "q"}:
        active.add("inertia")

    canonical_parts = [_DISPLAY_NAMES.get(name, name) for name in requested]
    canonical = "_".join(canonical_parts)
    return FrogDQModules(requested=requested, active=frozenset(active), canonical=canonical)

# -------------------------
# Reproducibility
# -------------------------


def _set_seed(random_state: int = 42) -> None:
    """Set Python, NumPy and Torch seeds for full determinism."""
    random.seed(random_state)
    np.random.seed(random_state)
    torch.manual_seed(random_state)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(random_state)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    if hasattr(torch, "mps") and torch.mps.is_available():
        torch.mps.manual_seed(random_state)


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

    def __init__(self, input_dim: int, init: float = 1.0, init_vector: list[float] = None):
        super().__init__()
        self.gates = nn.Parameter(torch.full((input_dim,), float(init)) if init_vector is None else torch.tensor(init_vector))

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
    gate_init_vector: list[float] = None,
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
    if gate_init_vector is None:
        return FrogModel(FrogGate(input_dim, gate_init), base) if use_frogdq else base
    else:
        if len(gate_init_vector) == input_dim:
            return FrogModel(FrogGate(input_dim, init_vector=gate_init_vector), base) if use_frogdq else base
        else:
            raise ValueError("The length of gate_init_vector does not correspond to input_dim")
   


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
    for batch in loader:
        if isinstance(batch, (tuple, list)):
            xb, yb = batch[0], batch[1]
            rb = batch[2] if len(batch) > 2 else None
        else:
            xb, yb = batch
            rb = None
        xb, yb = xb.to(device), yb.to(device)
        logits = model(xb)
        if rb is not None:
            rb = rb.to(device).view(-1)
            per_sample = F.cross_entropy(logits, yb, weight=getattr(criterion, "weight", None), reduction="none")
            loss = torch.mean(rb * per_sample)
        else:
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
    # --- Class-weighted loss to mitigate imbalance (computed from y_train)
    with torch.no_grad():
        # Ensure y_train is 1D LongTensor on CPU for bincount
        y_for_count = y.detach().view(-1).cpu().to(torch.long)
        if y_for_count.numel() > 0:
            num_classes = int(torch.max(y_for_count).item() + 1)
            counts = torch.bincount(y_for_count, minlength=num_classes).to(torch.float32)
            # Prevent division by zero for any missing classes
            counts = torch.where(counts > 0, counts, torch.ones_like(counts))
            class_weights = (counts.sum() / counts)
            class_weights = class_weights / class_weights.mean()
            class_weights = class_weights.to(device)
        else:
            class_weights = None
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    #criterion = nn.CrossEntropyLoss()
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
    batch_size: int = 256,
    lr: float = 1e-3,
    weight_decay: float = 1e-3,
    optimizer: Literal["adam", "sgd"] = "sgd",
    lr_scheduler: Literal["none", "plateau", "cosine"] = "none",
    scheduler_patience: int = 10,
    scheduler_factor: float = 0.5,
    scheduler_min_lr: float = 1e-6,
    scheduler_T_max: Optional[int] = None,
    early_stop: bool = True,
    es_patience: int = 50,
    es_min_delta: float = 1e-4,
    device: Optional[torch.device] = None,
    verbose: bool = False,
    log_every: int = 5,
    random_state: int = 42,
    # FrogDQ options
    q_vec: Optional[torch.Tensor] = None,
    r_vec: Optional[torch.Tensor] = None,
    fetch_g_every: int = 1,
    frogdq_mode: str = "none",
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
    # For frogdq_mode="temp": μ_n = μ * η^n (n = epoch-1)
    frog_temp_eta: float = 1.0,
    sample_weight_norm: Literal["none", "mean", "mean_rms"] = "mean_rms",
    r_clip_min: float = 0.0,          
    use_kappa_ema: bool = True,     
    kappa_ema_beta: float = 0.9,     
    detach_weight_norm: bool = True,  
) -> dict[str, list]:
    """
    Train a classifier with optional FrogDQ losses. Early stopping uses validation loss.

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
    lr_scheduler : {"none","plateau","cosine"}, default="none"
        Learning-rate scheduler; "plateau" reacts to validation loss, "cosine" anneals deterministically.
    scheduler_patience : int, default=10
        Patience (epochs) before reducing lr when using "plateau".
    scheduler_factor : float, default=0.5
        Multiplicative drop applied to lr for "plateau".
    scheduler_min_lr : float, default=1e-6
        Minimum lr allowed by schedulers.
    scheduler_T_max : int or None, default=None
        Period for "cosine" scheduler (defaults to `epochs`).
    early_stop : bool, default=True
        Enable early stopping on validation loss.
    es_patience : int, default=20
        Patience (epochs) for early stopping.
    es_min_delta : float, default=1e-4
        Minimum validation loss decrease to reset patience.
    device : torch.device or None, default=None
        Device to run on; autodetects GPU if available.
    verbose : bool, default=False
        If True, log epoch metrics with loguru.
    log_every : int, default=1
        Log every k epochs when verbose.
    random_state : int, default=42
        Seed for deterministic training.
    q_vec : Tensor or None, default=None
        Feature-quality vector for FrogDQ (length = D). Required when any gate-dependent
        module is active (e.g., inertia/gaussian/dirichlet/q).
    r_vec : Tensor or None, default=None
        Sample-quality vector for FrogDQ (length = N); required when module "samplewise"
        is requested.
    fetch_g_every : int, default = 1
        How frequently g_prev must be fetched.
    frogdq_mode : str, default="none"
        Underscore-joined list of FrogDQ modules (e.g., "dirichlet_q", "gaussian_temp").
        Supported atomic modules: {"inertia","temp","cosine","gaussian","dirichlet",
        "samplewise","q"}. Use "none" to disable.
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
        Global scale applied to FrogDQ regularization terms.
    frog_temp_invert : bool, default=False
        When module "cosine" is active, invert schedule (low→high→low instead of high→low→high).
    frog_temp_tau : float, default=1.0
        For module "cosine": speed/shape control (τ>1 compresses early change).
    frog_temp_eta : float, default=1.0
        For module "temp": exponential factor η in μ_n = μ * η^n (n = epoch-1).
    sample_weight_norm : {"none","mean","mean_rms"}, default="mean_rms"
        Normalization strategy for per-sample weights r when using module "samplewise".
        "none"     -> use r as provided
        "mean"     -> r_hat = B * r / sum(r)  (preserve mean scale)
        "mean_rms" -> r_hat, then divide by κ = sqrt(mean(r_hat^2)) (preserve gradient energy)
    r_clip_min : float, default=0.0
        Minimum value to clip r before normalization (e.g., 0.05).
    use_kappa_ema : bool, default=True
        If True, smooth κ with EMA across batches to avoid jitter.
    kappa_ema_beta : float, default=0.9
        EMA coefficient for κ smoothing.
    detach_weight_norm : bool, default=True
        If True, detach normalized r* from autograd graph.  

    Returns
    -------
    dict
        History with per-epoch metrics and frog gates (if enabled):
        keys = train_loss, val_loss, train_acc, val_acc, train_bal_acc,
               val_bal_acc, train_f1, val_f1, train_auc, val_auc, lr, frog_gates.
    """
    _set_seed(random_state)
    modules = parse_frogdq_mode(frogdq_mode)
    frogdq_mode = modules.canonical
    requested_modules = set(modules.requested)
    active_modules = set(modules.active)

    if device is None:
        if torch.cuda.is_available():
            device = torch.device("cuda")
        elif hasattr(torch, "mps") and torch.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")
    model.to(device)

    needs_gating = bool(active_modules & _GATE_RELEVANT_MODULES)
    use_sample_weights = "samplewise" in active_modules

    if use_sample_weights:
        if r_vec is None:
            raise ValueError("Module 'samplewise' requires r_vec with per-sample qualities.")
        r_vec = r_vec.view(-1).detach().to(dtype=torch.float32)
        if r_vec.numel() != X_train.size(0):
            raise ValueError("Length of r_vec must match the number of training samples.")
        train_dataset = TensorDataset(X_train, y_train, r_vec)
    else:
        train_dataset = TensorDataset(X_train, y_train)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(TensorDataset(X_val, y_val), batch_size=batch_size, shuffle=False)

    # Build class-weighted cross-entropy to address class imbalance
    with torch.no_grad():
        y_flat = y_train.view(-1).to(torch.long).detach().cpu()
        if y_flat.numel() == 0:
            class_weights = None
        else:
            n_classes = int(y_flat.max().item()) + 1
            counts = torch.bincount(y_flat, minlength=n_classes).to(torch.float32)
            counts = torch.where(counts > 0, counts, torch.ones_like(counts))
            class_weights = (counts.sum() / counts)
            class_weights = class_weights / class_weights.mean()
            class_weights = class_weights.to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    #criterion = nn.CrossEntropyLoss()
    if optimizer == "sgd":
        opt: torch.optim.Optimizer = optim.SGD(
            model.parameters(), lr=lr, momentum=0.9, weight_decay=weight_decay
        )
    else:
        opt: torch.optim.Optimizer = optim.Adam(
            model.parameters(), lr=lr, weight_decay=weight_decay
        )

    scheduler = None
    if lr_scheduler == "plateau":
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            opt,
            mode="min",
            factor=scheduler_factor,
            patience=max(1, scheduler_patience),
            min_lr=scheduler_min_lr,
        )
    elif lr_scheduler == "cosine":
        t_max = scheduler_T_max if scheduler_T_max and scheduler_T_max > 0 else epochs
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            opt,
            T_max=max(1, t_max),
            eta_min=scheduler_min_lr,
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
        "frog_gates": [] if (needs_gating and hasattr(model, "gate")) else None,
    }

    best_metric = -float("inf")
    best_state = None
    patience_left = es_patience

    logger.info(f"FrogDQ requested modules: {modules.requested} | active: {sorted(active_modules)}")
    logger.info(f"qvec: {q_vec}")
    use_frog = needs_gating
    logger.info(f"Use_frog: {use_frog}")

    if use_frog:
        if not hasattr(model, "gate"):
            raise ValueError(
                "FrogDQ modules requiring gating were requested, but the model has no gate. "
                "Instantiate the model with use_frogdq=True."
            )
        if q_vec is None:
            raise ValueError("FrogDQ modules require q_vec (feature quality vector).")
        q_vec = q_vec.to(device)
        g_prev = model.gate.gates.detach().clone()
        w_inertia = 1.0 - q_vec
        if normalize_inertia_by_mean:
            w_inertia = w_inertia / (w_inertia.mean() + eps)
    else:
        if q_vec is not None:
            q_vec = q_vec.to(device)
        g_prev = None
        w_inertia = None

    kappa_ema: Optional[torch.Tensor] = None

    def _inertia_scale_for_epoch(e: int) -> float:
        if "inertia" not in active_modules:
            return 0.0
        base = frog_temperature
        if "temp" in requested_modules:
            n = max(0, e - 1)
            return float(base * (float(frog_temp_eta) ** n))
        if "cosine" in requested_modules:
            if epochs <= 1:
                phase = 1.0
            else:
                phase = ((e - 1) / (epochs - 1)) ** max(1e-8, frog_temp_tau)
            alpha = 0.5 * (1.0 + np.cos(2.0 * np.pi * phase))
            if frog_temp_invert:
                alpha = 1.0 - alpha
            return float(base * alpha)
        return float(base)

    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0
        running_loss_prox = 0.0
        running_loss_prior = 0.0
        nobs = 0
        reg_scale = _inertia_scale_for_epoch(epoch)

        if use_frog and "inertia" in active_modules and (epoch == 1 or epoch % fetch_g_every == 0):
            g_prev = model.gate.gates.detach().clone()

        for batch in train_loader:
            if use_sample_weights:
                xb, yb, rb = batch
                rb = rb.to(device).view(-1)
                if r_clip_min > 0.0:
                    rb = torch.clamp(rb, min=r_clip_min, max=1.0)

                if sample_weight_norm != "none":
                    B = rb.numel()
                    denom = rb.sum().clamp_min(eps)
                    r_hat = (B * rb) / denom

                    if sample_weight_norm == "mean":
                        r_eff = r_hat
                    elif sample_weight_norm == "mean_rms":
                        kappa = (r_hat.pow(2).mean().clamp_min(eps)).sqrt()
                        if use_kappa_ema:
                            if kappa_ema is None:
                                kappa_ema = kappa.detach()
                            else:
                                kappa_ema = kappa_ema * kappa_ema_beta + kappa.detach() * (1.0 - kappa_ema_beta)
                            kappa_use = kappa_ema
                        else:
                            kappa_use = kappa
                        r_eff = r_hat / kappa_use
                    else:
                        r_eff = rb
                else:
                    r_eff = rb

                if detach_weight_norm:
                    r_eff = r_eff.detach()
            else:
                xb, yb = batch
                rb = None
                r_eff = None

            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            logits = model(xb)
            if rb is not None:
                per_sample_loss = F.cross_entropy(logits, yb, weight=class_weights, reduction="none")
                loss = torch.mean(r_eff * per_sample_loss)
            else:
                loss = criterion(logits, yb)

            loss_prox = torch.tensor(0.0, device=device)
            loss_prior = torch.tensor(0.0, device=device)

            if use_frog:
                g_curr = model.gate.gates
                if "inertia" in active_modules and reg_scale > 0.0:
                    inertia_term = (((g_curr - g_prev) ** 2) * w_inertia).sum()
                    loss_prox = lambda_prox * reg_scale * inertia_term
                    loss = loss + loss_prox
                if "gaussian" in active_modules:
                    gaussian_term = ((g_curr - q_vec) ** 2).sum()
                    gaussian_loss = frog_temperature * lambda_gaussian_prior * gaussian_term
                    loss_prior = loss_prior + gaussian_loss
                    loss = loss + gaussian_loss
                if "dirichlet" in active_modules:
                    dirichlet_term = _dirichlet_kl_prior_loss(
                        g_curr, q_vec, tau=kl_temperature, eps=eps
                    )
                    dirichlet_loss = frog_temperature * lambda_dirichlet_kl * dirichlet_term
                    loss_prior = loss_prior + dirichlet_loss
                    loss = loss + dirichlet_loss

            loss.backward()
            opt.step()

            running_loss += loss.item() * xb.size(0)
            running_loss_prox += loss_prox.item() * xb.size(0)
            running_loss_prior += loss_prior.item() * xb.size(0)
            nobs += xb.size(0)

        train_loss = running_loss / max(1, nobs)
        train_loss_prox = running_loss_prox / max(1, nobs)
        train_loss_prior = running_loss_prior / max(1, nobs)
        train_metrics = _eval_on_loader(model, train_loader, criterion, device)
        val_metrics = _eval_on_loader(model, val_loader, criterion, device)

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

        if scheduler is not None:
            if lr_scheduler == "plateau":
                scheduler.step(val_metrics["loss"])
            else:
                scheduler.step()

        if use_frog and history["frog_gates"] is not None:
            g_now = model.gate.gates.detach().cpu().numpy().copy()
            history["frog_gates"].append(g_now)

        if verbose and (epoch % log_every == 0):
            prox_ratio = (train_loss_prox / train_loss) if train_loss != 0 else 0.0
            prior_ratio = (train_loss_prior / train_loss) if train_loss != 0 else 0.0
            logger.info(
                f"Epoch {epoch:3d} | mode={frogdq_mode} | reg_scale={reg_scale:.3f} | lr={opt.param_groups[0]['lr']:.2e} | "
                f"train: loss={train_loss:.4f}, loss_prox={train_loss_prox:.4f}, ratio_loss_prox={prox_ratio:.4f}, "
                f"loss_prior={train_loss_prior:.4f}, ratio_loss_prior={prior_ratio:.4f} "
                f"acc={train_metrics['accuracy']:.4f}, "
                f"bal_acc={train_metrics['balanced_accuracy']:.4f}, "
                f"f1={train_metrics['f1']:.4f}, auc={train_metrics['auc_roc']:.4f} | "
                f"val: loss={val_metrics['loss']:.4f}, acc={val_metrics['accuracy']:.4f}, "
                f"bal_acc={val_metrics['balanced_accuracy']:.4f}, "
                f"f1={val_metrics['f1']:.4f}, auc={val_metrics['auc_roc']:.4f}"
            )

        val_score = val_metrics["balanced_accuracy"]
        if val_score > best_metric + es_min_delta:
            best_metric = val_score
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            patience_left = es_patience
        else:
            patience_left -= 1
            if early_stop and patience_left <= 0:
                if verbose:
                    logger.info(f"Early stopping at epoch {epoch} (best val balanced_accuracy={best_metric:.4f}).")
                break

    if best_state is not None:
        model.load_state_dict({k: v.to(device) for k, v in best_state.items()})
    return history
