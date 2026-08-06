"""
Loss functions for the multiclass neural network, along with their
gradients w.r.t. the network's output layer.

Design notes:
    - `cross_entropy_loss` expects `y_pred` to already be softmax
      probabilities and `y_true` to be one-hot encoded.
    - `cross_entropy_softmax_backward` computes the gradient of the loss
      w.r.t. the PRE-softmax logits (not the post-softmax probabilities).
      This is intentional: when softmax is used as the final activation
      together with cross-entropy loss, the two gradients combine into
      the remarkably simple form `dZ = y_pred - y_true`. Deriving this
      requires chain-ruling through the softmax Jacobian, which is why
      it's treated as a single combined step rather than two separate
      backward functions (see activations.py for the note on this).
"""

import numpy as np


def cross_entropy_loss(
    y_pred: np.ndarray, y_true: np.ndarray, eps: float = 1e-12
) -> float:
    """
    Categorical cross-entropy loss, averaged over the batch.

    Formula: L = -(1/N) * sum_i sum_c [ y_true[i,c] * log(y_pred[i,c]) ]

    Args:
        y_pred: Predicted probabilities (post-softmax), shape (N, C).
        y_true: One-hot encoded ground-truth labels, shape (N, C).
        eps: Small constant to avoid log(0), which would produce -inf
            (and NaN gradients) if the model predicts a probability of
            exactly 0 for the true class.

    Returns:
        Scalar loss value, averaged over the N samples in the batch.
    """
    n_samples = y_pred.shape[0]
    y_pred_clipped = np.clip(y_pred, eps, 1.0 - eps)
    loss = -np.sum(y_true * np.log(y_pred_clipped)) / n_samples
    return loss


def cross_entropy_softmax_backward(y_pred: np.ndarray, y_true: np.ndarray) -> np.ndarray:
    """
    Gradient of the cross-entropy loss w.r.t. the PRE-softmax logits Z,
    given that softmax was used as the output activation.

    This is the combined gradient dL/dZ, not dL/dA. Mathematically:
        dL/dZ = y_pred - y_true

    This elegant simplification is why softmax and cross-entropy are
    almost always implemented together as a single backward step in
    practice, rather than chaining a generic softmax_backward with a
    generic cross_entropy_backward.

    Args:
        y_pred: Predicted probabilities (post-softmax), shape (N, C).
        y_true: One-hot encoded ground-truth labels, shape (N, C).

    Returns:
        dZ: Gradient of the loss w.r.t. the output layer's logits,
            shape (N, C). Already averaged over the batch (divided by N),
            so it can be plugged directly into backpropagation.
    """
    n_samples = y_pred.shape[0]
    dZ = (y_pred - y_true) / n_samples
    return dZ