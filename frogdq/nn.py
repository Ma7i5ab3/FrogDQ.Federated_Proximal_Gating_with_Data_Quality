"""Neural network model builder for tabular data."""

import torch
import torch.nn as nn
from typing import Union, List, Tuple


def build_model(
    input_dim: int,
    output_dim: int,
    task: str = "classification",
    hidden_neurons: Union[int, str] = 128,
    layer_type: str = "dense",
    num_layers: int = 2,
    dropout: float = 0.0,
    activation: str = "relu",
    use_batch_norm: str = "false",
    output_activation: str = "auto",
) -> nn.Module:
    """
    Build a flexible neural network model for tabular data.

    Parameters
    ----------
    input_dim : int
        Number of input features.
    output_dim : int
        Number of output dimensions (classes for classification, 1 for regression).
    task : str, default="classification"
        Task type: "classification" or "regression".
    hidden_neurons : int or str, default=128
        Number of hidden neurons per layer. Can be:
        - Single int: Same size for all layers (e.g., 128)
        - Comma-separated string: Different sizes per layer (e.g., "256,128,64")
        - 0 or "0": Linear model (no hidden layers)
    layer_type : str, default="dense"
        Type of layers to use:
        - "dense": Standard fully connected layers (MLP)
        - "residual": Dense layers with residual connections
        - "transformer": Self-attention transformer layers (experimental)
    num_layers : int, default=2
        Number of hidden layers (ignored if hidden_neurons specifies layer sizes).
    dropout : float, default=0.0
        Dropout probability (0.0 = no dropout).
    activation : str, default="relu"
        Activation function: "relu", "gelu", "tanh", "sigmoid", "leaky_relu", "elu", "selu".
    use_batch_norm : str, default="false"
        Whether to use batch normalization: "true" or "false".
    output_activation : str, default="auto"
        Output activation: "auto" (based on task), "none", "sigmoid", "softmax", "log_softmax".

    Returns
    -------
    nn.Module
        PyTorch model.

    Examples
    --------
    >>> # Linear model
    >>> model = build_model(input_dim=10, output_dim=2, hidden_neurons=0)

    >>> # Simple MLP with 2 hidden layers of 128 neurons
    >>> model = build_model(input_dim=10, output_dim=2, hidden_neurons=128, num_layers=2)

    >>> # Custom architecture with different layer sizes
    >>> model = build_model(input_dim=10, output_dim=2, hidden_neurons="256,128,64")

    >>> # Deep network with dropout and batch norm
    >>> model = build_model(
    ...     input_dim=10, output_dim=1, task="regression",
    ...     hidden_neurons="512,256,128", dropout=0.3, use_batch_norm="true"
    ... )

    >>> # Residual network with GELU activation
    >>> model = build_model(
    ...     input_dim=50, output_dim=5, hidden_neurons=256,
    ...     layer_type="residual", activation="gelu", num_layers=4
    ... )
    """
    # Parse parameters
    task = str(task).lower()
    layer_type = str(layer_type).lower()
    activation = str(activation).lower()
    use_batch_norm = str(use_batch_norm).lower() in ("true", "1", "yes")
    output_activation = str(output_activation).lower()
    dropout = float(dropout)
    num_layers = int(num_layers)

    # Parse hidden layer sizes
    if isinstance(hidden_neurons, str):
        if "," in hidden_neurons:
            # Comma-separated list: "256,128,64"
            hidden_sizes = [int(x.strip()) for x in hidden_neurons.split(",")]
        else:
            # Single value as string: "128"
            hidden_size = int(hidden_neurons)
            hidden_sizes = [hidden_size] * num_layers if hidden_size > 0 else []
    else:
        # Single int value
        hidden_size = int(hidden_neurons)
        hidden_sizes = [hidden_size] * num_layers if hidden_size > 0 else []

    # Build model based on layer type
    if layer_type == "dense":
        model = DenseNetwork(
            input_dim=input_dim,
            output_dim=output_dim,
            hidden_sizes=hidden_sizes,
            dropout=dropout,
            activation=activation,
            use_batch_norm=use_batch_norm,
            output_activation=output_activation,
            task=task,
        )
    elif layer_type == "residual":
        model = ResidualNetwork(
            input_dim=input_dim,
            output_dim=output_dim,
            hidden_sizes=hidden_sizes,
            dropout=dropout,
            activation=activation,
            use_batch_norm=use_batch_norm,
            output_activation=output_activation,
            task=task,
        )
    elif layer_type == "transformer":
        model = TransformerNetwork(
            input_dim=input_dim,
            output_dim=output_dim,
            hidden_size=hidden_sizes[0] if hidden_sizes else 128,
            num_layers=len(hidden_sizes) if hidden_sizes else num_layers,
            dropout=dropout,
            output_activation=output_activation,
            task=task,
        )
    else:
        raise ValueError(
            f"Unknown layer_type '{layer_type}'. "
            f"Choose from: 'dense', 'residual', 'transformer'"
        )

    return model


class DenseNetwork(nn.Module):
    """Standard fully connected neural network (MLP)."""

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_sizes: List[int],
        dropout: float = 0.0,
        activation: str = "relu",
        use_batch_norm: bool = False,
        output_activation: str = "auto",
        task: str = "classification",
    ):
        super().__init__()
        self.output_activation = output_activation
        self.task = task

        layers = []
        in_dim = input_dim

        # Build hidden layers
        for hidden_dim in hidden_sizes:
            layers.append(nn.Linear(in_dim, hidden_dim))

            if use_batch_norm:
                layers.append(nn.BatchNorm1d(hidden_dim))

            layers.append(_get_activation(activation))

            if dropout > 0:
                layers.append(nn.Dropout(dropout))

            in_dim = hidden_dim

        # Output layer
        layers.append(nn.Linear(in_dim, output_dim))

        # Output activation
        if output_activation == "auto":
            if task == "classification" and output_dim == 1:
                layers.append(nn.Sigmoid())
            elif task == "classification" and output_dim > 1:
                layers.append(nn.LogSoftmax(dim=1))
            # For regression, no activation
        elif output_activation == "sigmoid":
            layers.append(nn.Sigmoid())
        elif output_activation == "softmax":
            layers.append(nn.Softmax(dim=1))
        elif output_activation == "log_softmax":
            layers.append(nn.LogSoftmax(dim=1))
        # else: no activation

        self.network = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)


class ResidualNetwork(nn.Module):
    """Neural network with residual connections."""

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_sizes: List[int],
        dropout: float = 0.0,
        activation: str = "relu",
        use_batch_norm: bool = False,
        output_activation: str = "auto",
        task: str = "classification",
    ):
        super().__init__()
        self.output_activation = output_activation
        self.task = task

        # Input projection if needed
        self.input_projection = None
        if hidden_sizes and input_dim != hidden_sizes[0]:
            self.input_projection = nn.Linear(input_dim, hidden_sizes[0])

        # Build residual blocks
        self.blocks = nn.ModuleList()
        in_dim = hidden_sizes[0] if hidden_sizes else input_dim

        for i, hidden_dim in enumerate(hidden_sizes):
            block = ResidualBlock(
                in_dim if i == 0 and self.input_projection is None else hidden_dim,
                hidden_dim,
                dropout=dropout,
                activation=activation,
                use_batch_norm=use_batch_norm,
            )
            self.blocks.append(block)
            in_dim = hidden_dim

        # Output layer
        final_dim = hidden_sizes[-1] if hidden_sizes else input_dim
        self.output_layer = nn.Linear(final_dim, output_dim)

        # Output activation
        self.output_act = _get_output_activation(output_activation, task, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Project input if needed
        if self.input_projection is not None:
            x = self.input_projection(x)

        # Pass through residual blocks
        for block in self.blocks:
            x = block(x)

        # Output
        x = self.output_layer(x)
        if self.output_act is not None:
            x = self.output_act(x)

        return x


class ResidualBlock(nn.Module):
    """A residual block with optional batch norm and dropout."""

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        dropout: float = 0.0,
        activation: str = "relu",
        use_batch_norm: bool = False,
    ):
        super().__init__()

        layers = [nn.Linear(in_dim, out_dim)]

        if use_batch_norm:
            layers.append(nn.BatchNorm1d(out_dim))

        layers.append(_get_activation(activation))

        if dropout > 0:
            layers.append(nn.Dropout(dropout))

        self.block = nn.Sequential(*layers)

        # Skip connection
        self.skip = nn.Identity() if in_dim == out_dim else nn.Linear(in_dim, out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x) + self.skip(x)


class TransformerNetwork(nn.Module):
    """Transformer-based network for tabular data."""

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_size: int = 128,
        num_layers: int = 2,
        num_heads: int = 8,
        dropout: float = 0.0,
        output_activation: str = "auto",
        task: str = "classification",
    ):
        super().__init__()
        self.output_activation = output_activation
        self.task = task
        self.hidden_size = hidden_size
        self.input_dim = input_dim

        # Input projection (project each feature independently)
        self.input_projection = nn.Linear(1, hidden_size)

        # Positional encoding (learnable)
        self.pos_encoding = nn.Parameter(torch.randn(1, input_dim, hidden_size))

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_size,
            nhead=num_heads,
            dim_feedforward=hidden_size * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Output layer (pool and project)
        self.output_layer = nn.Linear(hidden_size, output_dim)

        # Output activation
        self.output_act = _get_output_activation(output_activation, task, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: (batch_size, input_dim)
        batch_size = x.shape[0]

        # Reshape to (batch_size, input_dim, 1) for feature-wise processing
        x = x.unsqueeze(-1)  # (batch_size, input_dim, 1)

        # Project to hidden size
        x = self.input_projection(x)  # (batch_size, input_dim, hidden_size)

        # Add positional encoding
        x = x + self.pos_encoding

        # Transformer encoding
        x = self.transformer(x)  # (batch_size, input_dim, hidden_size)

        # Global average pooling
        x = x.mean(dim=1)  # (batch_size, hidden_size)

        # Output projection
        x = self.output_layer(x)

        if self.output_act is not None:
            x = self.output_act(x)

        return x


def _get_activation(activation: str) -> nn.Module:
    """Get activation function by name."""
    activations = {
        "relu": nn.ReLU(),
        "gelu": nn.GELU(),
        "tanh": nn.Tanh(),
        "sigmoid": nn.Sigmoid(),
        "leaky_relu": nn.LeakyReLU(),
        "elu": nn.ELU(),
        "selu": nn.SELU(),
    }

    if activation not in activations:
        raise ValueError(
            f"Unknown activation '{activation}'. "
            f"Choose from: {', '.join(activations.keys())}"
        )

    return activations[activation]


def _get_output_activation(
    output_activation: str,
    task: str,
    output_dim: int,
) -> Union[nn.Module, None]:
    """Get output activation function."""
    if output_activation == "auto":
        if task == "classification" and output_dim == 1:
            return nn.Sigmoid()
        elif task == "classification" and output_dim > 1:
            return nn.LogSoftmax(dim=1)
        else:
            return None
    elif output_activation == "sigmoid":
        return nn.Sigmoid()
    elif output_activation == "softmax":
        return nn.Softmax(dim=1)
    elif output_activation == "log_softmax":
        return nn.LogSoftmax(dim=1)
    elif output_activation == "none":
        return None
    else:
        return None
