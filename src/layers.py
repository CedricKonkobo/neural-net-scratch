"""
Dense (fully-connected) layer implementation, including forward and backward
passes for use in a hand-rolled backpropagation pipeline.

Design notes:
    - Each Dense layer optionally applies an activation function (currently
      "relu" or None) as part of its forward pass. The network's final
      output layer is expected to be created with activation=None: softmax
      and cross-entropy loss are combined and applied at the network level
      (see network.py and losses.py), because their gradients simplify
      together into `y_pred - y_true`.
    - Weights are initialized with He initialization (recommended for ReLU
      layers) to keep activations well-scaled at the start of training and
      avoid vanishing/exploding gradients in deeper networks.
    - Gradients (dW, db) are stored as instance attributes after backward()
      is called, so an optimizer can read `layer.dW` / `layer.db` and update
      `layer.W` / `layer.b` without passing them around explicitly.
"""

from typing import Optional

import numpy as np

from src.activations import relu, relu_backward


class Dense:
    """
    A fully-connected (linear) layer: Z = X @ W + b, optionally followed
    by an activation function A = activation(Z).

    Attributes:
        W: Weight matrix of shape (n_in, n_out).
        b: Bias vector of shape (1, n_out).
        activation: Name of the activation applied after the linear
            transform ("relu" or None).
        dW: Gradient of the loss w.r.t. W, populated after backward().
        db: Gradient of the loss w.r.t. b, populated after backward().
    """

    def __init__(
        self,
        n_in: int,
        n_out: int,
        activation: Optional[str] = "relu",
        seed: Optional[int] = None,
    ):
        """
        Initialize a Dense layer with He-initialized weights.

        Args:
            n_in: Number of input features.
            n_out: Number of output units (neurons) in this layer.
            activation: "relu" to apply ReLU after the linear transform,
                or None to leave raw logits (used for the network's final
                layer, where softmax is applied externally).
            seed: Optional random seed, for reproducible initialization.
        """
        rng = np.random.default_rng(seed) if seed is not None else np.random.default_rng()

        # He initialization: variance = 2 / n_in. Well-suited for ReLU
        # activations — keeps activation variance roughly stable across
        # layers at initialization, which helps gradients flow properly
        # during the first training steps.
        self.W = rng.standard_normal((n_in, n_out)) * np.sqrt(2.0 / n_in)
        self.b = np.zeros((1, n_out))

        self.activation = activation

        # Caches populated during forward(), consumed during backward().
        self.X: Optional[np.ndarray] = None
        self.Z: Optional[np.ndarray] = None

        # Gradients populated during backward(), consumed by the optimizer.
        self.dW: Optional[np.ndarray] = None
        self.db: Optional[np.ndarray] = None

    def forward(self, X: np.ndarray) -> np.ndarray:
        """
        Compute the forward pass of this layer.

        Args:
            X: Input batch of shape (n_samples, n_in).

        Returns:
            A: Output of shape (n_samples, n_out), after the linear
                transform and (optionally) the activation function.
        """
        self.X = X
        self.Z = X @ self.W + self.b

        if self.activation == "relu":
            A = relu(self.Z)
        elif self.activation is None:
            A = self.Z
        else:
            raise ValueError(f"Unsupported activation: '{self.activation}'")

        return A

    def backward(self, dA: np.ndarray) -> np.ndarray:
        """
        Compute the backward pass of this layer: propagate the gradient
        from this layer's output back to its input, and compute the
        gradients w.r.t. this layer's own parameters (W, b).

        Args:
            dA: Gradient of the loss w.r.t. this layer's output A, shape
                (n_samples, n_out). If activation=None (typical for the
                output layer), dA is expected to already be dL/dZ (e.g.
                from cross_entropy_softmax_backward), since there is no
                activation left to differentiate through.

        Returns:
            dX: Gradient of the loss w.r.t. this layer's input X, shape
                (n_samples, n_in). Pass this as `dA` to the preceding
                layer's backward() call.
        """
        if self.activation == "relu":
            dZ = relu_backward(dA, self.Z)
        elif self.activation is None:
            dZ = dA
        else:
            raise ValueError(f"Unsupported activation: '{self.activation}'")

        self.dW = self.X.T @ dZ
        self.db = np.sum(dZ, axis=0, keepdims=True)
        dX = dZ @ self.W.T

        return dX