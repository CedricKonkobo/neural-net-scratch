"""Unit tests for src/layers.py, including numerical gradient checking."""

import numpy as np
import pytest

from src.layers import Dense


@pytest.fixture
def sample_batch():
    rng = np.random.default_rng(0)
    return rng.standard_normal((4, 6))  # batch of 4 samples, 6 input features


class TestDenseForward:
    def test_output_shape(self, sample_batch):
        layer = Dense(n_in=6, n_out=3, activation="relu", seed=1)
        assert layer.forward(sample_batch).shape == (4, 3)

    def test_relu_output_is_non_negative(self, sample_batch):
        layer = Dense(n_in=6, n_out=3, activation="relu", seed=1)
        assert np.all(layer.forward(sample_batch) >= 0.0)

    def test_he_initialization_scale(self):
        # He init variance should be ~2/n_in; check empirically over many units.
        n_in = 1000
        layer = Dense(n_in=n_in, n_out=500, activation="relu", seed=42)
        assert np.var(layer.W) == pytest.approx(2.0 / n_in, rel=0.2)


class TestDenseBackward:
    def test_gradient_shapes_match_parameters(self, sample_batch):
        layer = Dense(n_in=6, n_out=3, activation="relu", seed=1)
        layer.forward(sample_batch)
        dX = layer.backward(np.ones((4, 3)))

        assert dX.shape == sample_batch.shape
        assert layer.dW.shape == layer.W.shape
        assert layer.db.shape == layer.b.shape

    def test_relu_backward_stops_gradient_through_inactive_units(self):
        layer = Dense(n_in=3, n_out=2, activation="relu", seed=5)
        layer.forward(np.array([[1.0, 1.0, 1.0]]))
        layer.Z = np.array([[-1.0, 2.0]])  # unit 0 inactive, unit 1 active
        layer.backward(np.array([[10.0, 10.0]]))

        # Column feeding the inactive unit must receive zero gradient.
        np.testing.assert_array_equal(layer.dW[:, 0], np.zeros(3))
        assert not np.all(layer.dW[:, 1] == 0)


class TestDenseGradientChecking:
    def test_weight_gradients_match_numerical_approximation(self):
        rng = np.random.default_rng(7)
        X = rng.standard_normal((3, 4))
        layer = Dense(n_in=4, n_out=2, activation="relu", seed=3)

        A = layer.forward(X)
        upstream_grad = rng.standard_normal(A.shape)
        layer.backward(upstream_grad)
        analytical_dW = layer.dW.copy()
        original_W = layer.W.copy()

        def scalar_loss(W: np.ndarray) -> float:
            # L = sum(A * upstream_grad) is chosen so dL/dA == upstream_grad,
            # making dL/dW directly comparable to layer.dW above.
            layer.W = W
            return float(np.sum(layer.forward(X) * upstream_grad))

        epsilon = 1e-5
        numerical_dW = np.zeros_like(original_W)

        it = np.nditer(original_W, flags=["multi_index"])
        for _ in it:
            idx = it.multi_index
            W_plus, W_minus = original_W.copy(), original_W.copy()
            W_plus[idx] += epsilon
            W_minus[idx] -= epsilon
            numerical_dW[idx] = (scalar_loss(W_plus) - scalar_loss(W_minus)) / (2 * epsilon)

        layer.W = original_W  # restore

        relative_error = np.linalg.norm(analytical_dW - numerical_dW) / (
            np.linalg.norm(analytical_dW) + np.linalg.norm(numerical_dW) + 1e-12
        )
        assert relative_error < 1e-5