"""Unit tests for src/losses.py."""

import numpy as np
import pytest

from src.losses import cross_entropy_loss, cross_entropy_softmax_backward


class TestCrossEntropyLoss:
    def test_known_value_two_samples(self):
        # Sample 0: true class 0, predicted prob 0.5  -> -log(0.5)
        # Sample 1: true class 1, predicted prob 0.25 -> -log(0.25)
        y_pred = np.array([[0.5, 0.5], [0.75, 0.25]])
        y_true = np.array([[1, 0], [0, 1]])
        expected = (-np.log(0.5) - np.log(0.25)) / 2
        assert cross_entropy_loss(y_pred, y_true) == pytest.approx(expected)

    def test_near_zero_for_confident_correct_predictions(self):
        y_pred = np.array([[0.999, 0.001], [0.001, 0.999]])
        y_true = np.array([[1, 0], [0, 1]])
        assert cross_entropy_loss(y_pred, y_true) < 0.01

    def test_does_not_explode_with_zero_probability(self):
        # Without clipping, log(0) would produce -inf / NaN.
        y_pred = np.array([[0.0, 1.0]])
        y_true = np.array([[1, 0]])
        assert np.isfinite(cross_entropy_loss(y_pred, y_true))


class TestCrossEntropySoftmaxBackward:
    def test_output_shape_matches_input(self):
        y_pred = np.random.default_rng(0).dirichlet(np.ones(5), size=8)
        y_true = np.eye(5)[np.random.default_rng(1).integers(0, 5, size=8)]
        assert cross_entropy_softmax_backward(y_pred, y_true).shape == y_pred.shape

    def test_gradient_is_zero_for_perfect_prediction(self):
        y_pred = np.array([[1.0, 0.0, 0.0]])
        y_true = np.array([[1, 0, 0]])
        dZ = cross_entropy_softmax_backward(y_pred, y_true)
        np.testing.assert_allclose(dZ, np.zeros_like(dZ), atol=1e-12)

    def test_matches_manual_formula(self):
        y_pred = np.array([[0.7, 0.2, 0.1]])
        y_true = np.array([[0, 1, 0]])
        expected = (y_pred - y_true) / 1  # n_samples = 1
        np.testing.assert_allclose(cross_entropy_softmax_backward(y_pred, y_true), expected)