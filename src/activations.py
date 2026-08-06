"""
Activation functions and their derivatives, used during the forward and
backward passes of the neural network.

Design notes:
    - Every activation `foo(Z)` has a matching `foo_backward(dA, Z)` that
      computes the local gradient needed by backpropagation (chain rule).
    - Softmax is implemented with the standard numerical stability trick
      (subtracting the row-wise max before exponentiating) to avoid
      overflow when logits are large.
    - Softmax does NOT have a `softmax_backward` function here on purpose:
      when combined with cross-entropy loss (the standard setup for
      multiclass classification), the gradient of the loss w.r.t. the
      pre-softmax logits simplifies to `y_pred - y_true`. That combined
      gradient is implemented in `losses.py` to avoid an unnecessarily
      complex (and error-prone) full Jacobian computation here.
"""

import numpy as np


# ----------------------------------------------------------------------------
# ReLU
# ----------------------------------------------------------------------------
def relu(Z: np.ndarray) -> np.ndarray:
    """
    Rectified Linear Unit: max(0, Z), applied element-wise.

    Args:
        Z: Pre-activation values of any shape.

    Returns:
        Array of the same shape as Z, with negative values clipped to 0.
    """
    return np.maximum(0, Z)


def relu_backward(dA: np.ndarray, Z: np.ndarray) -> np.ndarray:
    """
    Gradient of the ReLU function.

    The derivative of ReLU is 1 where Z > 0, and 0 elsewhere (using 0 at
    Z == 0 by convention, which is a common and safe choice in practice).

    Args:
        dA: Upstream gradient (dL/dA), same shape as Z.
        Z: Pre-activation values that were passed through relu() in the
            forward pass (must be cached by the layer for this purpose).

    Returns:
        dZ: Gradient of the loss w.r.t. Z, same shape as Z.
    """
    dZ = dA * (Z > 0)
    return dZ


# ----------------------------------------------------------------------------
# Sigmoid (kept for completeness / potential binary sub-tasks)
# ----------------------------------------------------------------------------
def sigmoid(Z: np.ndarray) -> np.ndarray:
    """
    Sigmoid activation: 1 / (1 + exp(-Z)), applied element-wise.

    Args:
        Z: Pre-activation values of any shape.

    Returns:
        Array of the same shape as Z, with values in (0, 1).
    """
    return 1.0 / (1.0 + np.exp(-Z))


def sigmoid_backward(dA: np.ndarray, Z: np.ndarray) -> np.ndarray:
    """
    Gradient of the sigmoid function.

    Args:
        dA: Upstream gradient (dL/dA), same shape as Z.
        Z: Pre-activation values that were passed through sigmoid().

    Returns:
        dZ: Gradient of the loss w.r.t. Z, same shape as Z.
    """
    s = sigmoid(Z)
    dZ = dA * s * (1 - s)
    return dZ


# ----------------------------------------------------------------------------
# Softmax
# ----------------------------------------------------------------------------
def softmax(Z: np.ndarray) -> np.ndarray:
    """
    Softmax activation, applied row-wise, for multiclass output layers.

    Converts raw logits into a probability distribution over classes
    (each row sums to 1).

    Numerical stability:
        exp(Z) can overflow for large Z. Subtracting the row-wise max
        before exponentiating does not change the mathematical result
        (softmax is shift-invariant) but keeps values in a safe range.

    Args:
        Z: Logits of shape (n_samples, n_classes).

    Returns:
        Probabilities of shape (n_samples, n_classes), each row summing to 1.
    """
    shifted_Z = Z - np.max(Z, axis=1, keepdims=True)
    exp_Z = np.exp(shifted_Z)
    return exp_Z / np.sum(exp_Z, axis=1, keepdims=True)