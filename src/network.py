"""
NeuralNetwork class: assembles Dense layers into a full multiclass
classifier, with a mini-batch gradient descent training loop.

Design notes:
    - Hidden layers use ReLU activation; the output layer uses NO
      activation inside Dense (activation=None) because softmax is
      applied here, at the network level, so its gradient can be combined
      with cross-entropy loss into the simplified `y_pred - y_true` form
      (see losses.py for details).
    - Parameter updates currently use plain vanilla gradient descent
      (`W -= lr * dW`), implemented directly in `_update_params`. This is
      intentional: it lets us validate the full forward/backward pipeline
      with the simplest possible optimizer before introducing momentum-
      based methods (SGD with momentum, Adam) in optimizers.py. Once that
      module exists, `_update_params` will be replaced by a pluggable
      Optimizer object.
"""

import logging
from typing import Dict, List, Optional

import numpy as np

from src.activations import softmax
from src.layers import Dense
from src.losses import cross_entropy_loss, cross_entropy_softmax_backward

logger = logging.getLogger(__name__)


class NeuralNetwork:
    """
    A feedforward multiclass neural network built from stacked Dense layers.

    Attributes:
        layers: List of Dense layer instances, in forward-pass order.
        y_pred: Cached softmax output from the most recent forward() call,
            reused by backward() so it doesn't need to be passed explicitly.
    """

    def __init__(self, layer_dims: List[int], seed: Optional[int] = None):
        """
        Build the network architecture from a list of layer sizes.

        Args:
            layer_dims: Sizes of each layer, including input and output.
                E.g. [784, 128, 64, 10] builds:
                    Dense(784 -> 128, relu)
                    Dense(128 -> 64,  relu)
                    Dense(64  -> 10,  None)   <- raw logits, softmax applied
                                                  separately in forward()
            seed: Optional random seed for reproducible weight
                initialization. Each layer gets seed + layer_index so
                layers don't share identical initial weights.
        """
        if len(layer_dims) < 2:
            raise ValueError("layer_dims must contain at least [n_input, n_output].")

        self.layers: List[Dense] = []
        n_layers = len(layer_dims) - 1

        for i in range(n_layers):
            n_in, n_out = layer_dims[i], layer_dims[i + 1]
            is_output_layer = i == n_layers - 1
            layer_seed = None if seed is None else seed + i

            self.layers.append(
                Dense(
                    n_in=n_in,
                    n_out=n_out,
                    activation=None if is_output_layer else "relu",
                    seed=layer_seed,
                )
            )

        self.y_pred: Optional[np.ndarray] = None

        logger.info(f"Network built with architecture: {layer_dims}")

    # ------------------------------------------------------------------
    # Forward / backward
    # ------------------------------------------------------------------
    def forward(self, X: np.ndarray) -> np.ndarray:
        """
        Run the full forward pass: through every Dense layer, then softmax.

        Args:
            X: Input batch of shape (n_samples, n_features).

        Returns:
            y_pred: Class probabilities of shape (n_samples, n_classes).
        """
        A = X
        for layer in self.layers:
            A = layer.forward(A)

        # `A` here is the raw output of the last layer (activation=None),
        # i.e. the logits. Softmax converts them into probabilities.
        self.y_pred = softmax(A)
        return self.y_pred

    def backward(self, y_true: np.ndarray) -> None:
        """
        Run the full backward pass: compute gradients for every layer's
        parameters (dW, db), starting from the combined softmax +
        cross-entropy gradient and propagating it back through each layer.

        Must be called AFTER forward(), since it relies on self.y_pred.

        Args:
            y_true: One-hot encoded ground-truth labels, shape
                (n_samples, n_classes).
        """
        if self.y_pred is None:
            raise RuntimeError("Call forward() before backward().")

        dZ = cross_entropy_softmax_backward(self.y_pred, y_true)

        for layer in reversed(self.layers):
            dZ = layer.backward(dZ)

    # ------------------------------------------------------------------
    # Parameter updates (plain gradient descent — replaced by an
    # Optimizer class in optimizers.py)
    # ------------------------------------------------------------------
    def _update_params(self, learning_rate: float) -> None:
        """
        Apply one step of vanilla gradient descent to every layer's
        parameters, using the gradients computed by the last backward()
        call.

        Args:
            learning_rate: Step size for the parameter update.
        """
        for layer in self.layers:
            layer.W -= learning_rate * layer.dW
            layer.b -= learning_rate * layer.db

    # ------------------------------------------------------------------
    # Inference / evaluation
    # ------------------------------------------------------------------
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict class indices for a batch of inputs.

        Args:
            X: Input batch of shape (n_samples, n_features).

        Returns:
            Predicted class indices, shape (n_samples,).
        """
        probabilities = self.forward(X)
        return np.argmax(probabilities, axis=1)

    def accuracy(self, X: np.ndarray, y: np.ndarray) -> float:
        """
        Compute classification accuracy on a dataset.

        Args:
            X: Input batch of shape (n_samples, n_features).
            y: True integer class labels, shape (n_samples,).

        Returns:
            Fraction of correctly classified samples, in [0, 1].
        """
        predictions = self.predict(X)
        return float(np.mean(predictions == y))

    # ------------------------------------------------------------------
    # Training loop
    # ------------------------------------------------------------------
    @staticmethod
    def _iterate_minibatches(
        X: np.ndarray, y_onehot: np.ndarray, batch_size: int, rng: np.random.Generator
    ):
        """
        Yield shuffled mini-batches of (X, y_onehot) for one epoch.

        Args:
            X: Full feature matrix for the epoch, shape (n_samples, n_features).
            y_onehot: Full one-hot label matrix, shape (n_samples, n_classes).
            batch_size: Number of samples per mini-batch.
            rng: NumPy random generator, used to shuffle sample order.

        Yields:
            (X_batch, y_batch) pairs covering the full dataset exactly once.
        """
        n_samples = X.shape[0]
        indices = rng.permutation(n_samples)

        for start in range(0, n_samples, batch_size):
            batch_idx = indices[start : start + batch_size]
            yield X[batch_idx], y_onehot[batch_idx]

    def train(
        self,
        X_train: np.ndarray,
        y_train_onehot: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val_onehot: np.ndarray,
        y_val: np.ndarray,
        epochs: int = 20,
        batch_size: int = 64,
        learning_rate: float = 0.1,
        seed: int = 42,
        verbose: bool = True,
    ) -> Dict[str, List[float]]:
        """
        Train the network using mini-batch gradient descent.

        Args:
            X_train, y_train_onehot, y_train: Training features, one-hot
                labels, and integer labels (the latter used for accuracy).
            X_val, y_val_onehot, y_val: Same, for the validation set.
            epochs: Number of full passes over the training data.
            batch_size: Number of samples per mini-batch.
            learning_rate: Step size for gradient descent updates.
            seed: Random seed controlling mini-batch shuffling order.
            verbose: If True, log loss/accuracy after every epoch.

        Returns:
            history: Dict with keys "train_loss", "train_acc", "val_loss",
                "val_acc", each a list with one entry per epoch — ready to
                be plotted with matplotlib in a later step.
        """
        rng = np.random.default_rng(seed)
        history: Dict[str, List[float]] = {
            "train_loss": [],
            "train_acc": [],
            "val_loss": [],
            "val_acc": [],
        }

        for epoch in range(1, epochs + 1):
            for X_batch, y_batch in self._iterate_minibatches(
                X_train, y_train_onehot, batch_size, rng
            ):
                self.forward(X_batch)
                self.backward(y_batch)
                self._update_params(learning_rate)

            # End-of-epoch metrics, computed on the FULL train/val sets
            # (not just the last mini-batch) for accurate monitoring.
            train_pred = self.forward(X_train)
            train_loss = cross_entropy_loss(train_pred, y_train_onehot)
            train_acc = float(np.mean(np.argmax(train_pred, axis=1) == y_train))

            val_pred = self.forward(X_val)
            val_loss = cross_entropy_loss(val_pred, y_val_onehot)
            val_acc = float(np.mean(np.argmax(val_pred, axis=1) == y_val))

            history["train_loss"].append(train_loss)
            history["train_acc"].append(train_acc)
            history["val_loss"].append(val_loss)
            history["val_acc"].append(val_acc)

            if verbose:
                logger.info(
                    f"Epoch {epoch:3d}/{epochs} | "
                    f"train_loss: {train_loss:.4f} train_acc: {train_acc:.4f} | "
                    f"val_loss: {val_loss:.4f} val_acc: {val_acc:.4f}"
                )

        return history 