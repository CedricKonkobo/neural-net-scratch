"""End-to-end tests for src/network.py."""

import numpy as np

from src.network import NeuralNetwork
from src.preprocessing import one_hot_encode


class TestNetworkForward:
    def test_output_is_a_valid_probability_distribution(self):
        X = np.random.default_rng(0).standard_normal((5, 10))
        net = NeuralNetwork(layer_dims=[10, 8, 4], seed=1)
        y_pred = net.forward(X)

        assert y_pred.shape == (5, 4)
        assert np.all(y_pred >= 0.0)
        np.testing.assert_allclose(np.sum(y_pred, axis=1), np.ones(5), rtol=1e-10)


class TestNetworkTraining:
    def test_overfits_a_tiny_batch(self):
        """
        A network unable to drive the loss near zero on a handful of samples
        signals a bug in forward/backward/update. Fast, deterministic
        end-to-end sanity check.
        """
        rng = np.random.default_rng(0)
        X = rng.standard_normal((8, 10))
        y = rng.integers(0, 3, size=8)
        y_onehot = one_hot_encode(y, num_classes=3)

        net = NeuralNetwork(layer_dims=[10, 16, 3], seed=1)
        history = net.train(
            X_train=X, y_train_onehot=y_onehot, y_train=y,
            X_val=X, y_val_onehot=y_onehot, y_val=y,
            epochs=200, batch_size=8, learning_rate=0.5, verbose=False,
        )

        assert history["train_loss"][-1] < 0.05
        assert history["train_acc"][-1] == 1.0