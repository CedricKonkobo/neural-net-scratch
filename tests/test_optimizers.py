"""
Unit tests for src/optimizers.py.

Uses a minimal FakeLayer (exposing only W, b, dW, db) instead of the real
Dense class, to prove and preserve the decoupling described in
optimizers.py: an Optimizer only needs those four attributes, nothing
Dense-specific.
"""

import numpy as np
import pytest

from src.optimizers import SGD, Adam


class FakeLayer:
    """Minimal stand-in for Dense, exposing only what Optimizer needs."""

    def __init__(self, W: np.ndarray, b: np.ndarray, dW: np.ndarray, db: np.ndarray):
        self.W = W
        self.b = b
        self.dW = dW
        self.db = db


class TestSGDWithoutMomentum:
    def test_matches_plain_gradient_descent_formula(self):
        layer = FakeLayer(
            W=np.array([[1.0, 2.0]]),
            b=np.array([[0.5]]),
            dW=np.array([[0.1, 0.2]]),
            db=np.array([[0.05]]),
        )
        optimizer = SGD(learning_rate=0.1, momentum=0.0)
        optimizer.update([layer])

        np.testing.assert_allclose(layer.W, np.array([[0.99, 1.98]]))
        np.testing.assert_allclose(layer.b, np.array([[0.495]]))


class TestSGDWithMomentum:
    def test_velocity_accumulates_across_steps(self):
        layer = FakeLayer(
            W=np.array([[1.0]]), b=np.array([[0.0]]),
            dW=np.array([[1.0]]), db=np.array([[0.0]]),
        )
        optimizer = SGD(learning_rate=0.1, momentum=0.9)

        # Step 1: v = 0.9*0 - 0.1*1 = -0.1  ->  W = 1.0 + (-0.1) = 0.9
        optimizer.update([layer])
        assert layer.W[0, 0] == pytest.approx(0.9)

        # Step 2 (same gradient again): v = 0.9*(-0.1) - 0.1*1 = -0.19
        # -> W = 0.9 + (-0.19) = 0.71
        layer.dW = np.array([[1.0]])
        optimizer.update([layer])
        assert layer.W[0, 0] == pytest.approx(0.71)

    def test_maintains_separate_state_per_layer(self):
        layer0 = FakeLayer(
            W=np.zeros((1, 1)), b=np.zeros((1, 1)),
            dW=np.array([[1.0]]), db=np.zeros((1, 1)),
        )
        layer1 = FakeLayer(
            W=np.zeros((1, 1)), b=np.zeros((1, 1)),
            dW=np.array([[5.0]]), db=np.zeros((1, 1)),
        )
        optimizer = SGD(learning_rate=0.1, momentum=0.9)

        optimizer.update([layer0, layer1])
        optimizer.update([layer0, layer1])

        # Different gradients per layer must produce different velocity
        # accumulation — this would fail if state were shared/global
        # instead of keyed per layer index.
        assert layer0.W[0, 0] != layer1.W[0, 0]


class TestAdam:
    def test_first_step_matches_analytical_formula(self):
        layer = FakeLayer(
            W=np.array([[5.0, -3.0]]),
            b=np.array([[0.0]]),
            dW=np.array([[2.0, -4.0]]),
            db=np.array([[0.0]]),
        )
        original_dW = layer.dW.copy()
        optimizer = Adam(learning_rate=0.1, beta1=0.9, beta2=0.999, epsilon=1e-8)
        optimizer.update([layer])

        # At t=1, bias correction makes m_hat == dW and v_hat == dW**2
        # exactly (the (1 - beta) factor from initialization cancels out).
        expected_update = 0.1 * original_dW / (np.sqrt(original_dW ** 2) + 1e-8)
        expected_W = np.array([[5.0, -3.0]]) - expected_update
        np.testing.assert_allclose(layer.W, expected_W, rtol=1e-6)

    def test_moves_parameters_opposite_to_gradient_sign(self):
        # A positive gradient should decrease the parameter; a negative
        # gradient should increase it — the basic sanity check for any
        # gradient-descent-based optimizer.
        layer = FakeLayer(
            W=np.array([[10.0, -10.0]]),
            b=np.array([[0.0]]),
            dW=np.array([[3.0, -3.0]]),
            db=np.array([[0.0]]),
        )
        optimizer = Adam(learning_rate=0.5)
        optimizer.update([layer])

        assert layer.W[0, 0] < 10.0
        assert layer.W[0, 1] > -10.0

    def test_timestep_increments_once_per_update_call_not_per_layer(self):
        layer0 = FakeLayer(
            W=np.zeros((1, 1)), b=np.zeros((1, 1)),
            dW=np.array([[1.0]]), db=np.zeros((1, 1)),
        )
        layer1 = FakeLayer(
            W=np.zeros((1, 1)), b=np.zeros((1, 1)),
            dW=np.array([[1.0]]), db=np.zeros((1, 1)),
        )
        optimizer = Adam(learning_rate=0.1)
        optimizer.update([layer0, layer1])

        # Both layers see the same gradient at the same timestep (t=1),
        # so their bias-corrected update must be identical.
        assert layer0.W[0, 0] == pytest.approx(layer1.W[0, 0])