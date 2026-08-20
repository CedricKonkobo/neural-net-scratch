"""
Optimizer implementations used to update network parameters (W, b) from
gradients (dW, db) computed during backpropagation.

Design notes:
    - Optimizers are decoupled from the NeuralNetwork/Dense classes: they
      only require objects exposing .W, .b, .dW, .db attributes, which is
      exactly what Dense (layers.py) provides. This keeps
      NeuralNetwork.train() agnostic to which optimizer is used — swapping
      SGD for Adam requires no change to the training loop logic itself.
    - Per-parameter state (momentum velocity for SGD, first/second moment
      estimates for Adam) is stored inside the optimizer instance, keyed by
      the layer's index in the network's layer list. This assumes the same
      list of layers (same order, same objects) is passed to every
      update() call during a single training run — guaranteed by
      NeuralNetwork.train(), which always passes self.layers.
"""

from abc import ABC, abstractmethod
from typing import Dict, List

import numpy as np


class Optimizer(ABC):
    """Abstract base class for all parameter-update strategies."""

    @abstractmethod
    def update(self, layers: List) -> None:
        """
        Update each layer's W and b in place, using the gradients (dW, db)
        populated by the most recent backward() call.

        Args:
            layers: The network's layers, in forward-pass order. Each
                object must expose W, b, dW, db as attributes.
        """
        raise NotImplementedError


class SGD(Optimizer):
    """
    Stochastic Gradient Descent, with optional momentum.

    Without momentum (momentum=0.0):
        W = W - learning_rate * dW

    With momentum (0.0 < momentum < 1.0), a velocity term accumulates
    gradients across steps, accelerating convergence along consistent
    directions and dampening oscillations in narrow ravines of the loss
    surface:
        v = momentum * v - learning_rate * dW
        W = W + v
    """

    def __init__(self, learning_rate: float = 0.01, momentum: float = 0.0):
        """
        Args:
            learning_rate: Step size applied to gradients.
            momentum: Momentum coefficient in [0, 1). 0.0 disables momentum
                and reduces this to vanilla gradient descent. Common values
                when enabled are 0.9 or 0.99.
        """
        self.learning_rate = learning_rate
        self.momentum = momentum

        # Velocity state, keyed by layer index. Lazily initialized to
        # zeros the first time each layer is seen.
        self._velocity_W: Dict[int, np.ndarray] = {}
        self._velocity_b: Dict[int, np.ndarray] = {}

    def update(self, layers: List) -> None:
        for i, layer in enumerate(layers):
            if self.momentum == 0.0:
                layer.W -= self.learning_rate * layer.dW
                layer.b -= self.learning_rate * layer.db
                continue

            v_W = self._velocity_W.setdefault(i, np.zeros_like(layer.W))
            v_b = self._velocity_b.setdefault(i, np.zeros_like(layer.b))

            v_W[:] = self.momentum * v_W - self.learning_rate * layer.dW
            v_b[:] = self.momentum * v_b - self.learning_rate * layer.db

            layer.W += v_W
            layer.b += v_b


class Adam(Optimizer):
    """
    Adam optimizer (Kingma & Ba, 2014): combines momentum (first moment
    estimate) with per-parameter adaptive learning rates (second moment
    estimate), plus bias correction for both moments during early steps.

    Update rule, per parameter:
        m = beta1 * m + (1 - beta1) * grad
        v = beta2 * v + (1 - beta2) * grad**2
        m_hat = m / (1 - beta1**t)
        v_hat = v / (1 - beta2**t)
        param -= learning_rate * m_hat / (sqrt(v_hat) + epsilon)

    Adam typically converges faster than plain SGD and is less sensitive
    to the exact choice of learning_rate, making it a strong default once
    the forward/backward pipeline has already been validated with simple
    SGD (as we did in the previous step).
    """

    def __init__(
        self,
        learning_rate: float = 0.001,
        beta1: float = 0.9,
        beta2: float = 0.999,
        epsilon: float = 1e-8,
    ):
        """
        Args:
            learning_rate: Step size applied to the bias-corrected update.
            beta1: Decay rate for the first moment estimate (running mean
                of gradients). Standard default: 0.9.
            beta2: Decay rate for the second moment estimate (running mean
                of squared gradients). Standard default: 0.999.
            epsilon: Small constant added to the denominator for numerical
                stability (avoids division by zero when v_hat is near
                zero, e.g. for parameters that rarely change).
        """
        self.learning_rate = learning_rate
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon

        self._m_W: Dict[int, np.ndarray] = {}
        self._v_W: Dict[int, np.ndarray] = {}
        self._m_b: Dict[int, np.ndarray] = {}
        self._v_b: Dict[int, np.ndarray] = {}

        # Global timestep, incremented once per update() call (not once
        # per layer), since bias correction depends on the total number
        # of update steps taken so far.
        self._t = 0

    def update(self, layers: List) -> None:
        self._t += 1

        for i, layer in enumerate(layers):
            m_W = self._m_W.setdefault(i, np.zeros_like(layer.W))
            v_W = self._v_W.setdefault(i, np.zeros_like(layer.W))
            m_b = self._m_b.setdefault(i, np.zeros_like(layer.b))
            v_b = self._v_b.setdefault(i, np.zeros_like(layer.b))

            m_W[:] = self.beta1 * m_W + (1 - self.beta1) * layer.dW
            v_W[:] = self.beta2 * v_W + (1 - self.beta2) * (layer.dW ** 2)
            m_b[:] = self.beta1 * m_b + (1 - self.beta1) * layer.db
            v_b[:] = self.beta2 * v_b + (1 - self.beta2) * (layer.db ** 2)

            # Bias correction: without it, m and v are biased toward zero
            # during early steps (initialized at zero, beta1/beta2 near 1).
            m_W_hat = m_W / (1 - self.beta1 ** self._t)
            v_W_hat = v_W / (1 - self.beta2 ** self._t)
            m_b_hat = m_b / (1 - self.beta1 ** self._t)
            v_b_hat = v_b / (1 - self.beta2 ** self._t)

            layer.W -= self.learning_rate * m_W_hat / (np.sqrt(v_W_hat) + self.epsilon)
            layer.b -= self.learning_rate * m_b_hat / (np.sqrt(v_b_hat) + self.epsilon)