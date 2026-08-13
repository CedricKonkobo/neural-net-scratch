"""Unit tests for src/activations.py."""

import numpy as np
import pytest

from src.activations import relu, relu_backward, sigmoid, sigmoid_backward, softmax


class TestRelu:
    def test_zeroes_out_negative_values(self):
        Z = np.array([[-2.0, -0.1, 0.0, 0.5, 3.0]])
        expected = np.array([[0.0, 0.0, 0.0, 0.5, 3.0]])
        np.testing.assert_array_equal(relu(Z), expected)

    def test_backward_masks_gradient_where_input_was_negative(self):
        Z = np.array([[-1.0, 2.0, 0.0, -3.0]])
        dA = np.array([[5.0, 5.0, 5.0, 5.0]])
        expected = np.array([[0.0, 5.0, 0.0, 0.0]])
        np.testing.assert_array_equal(relu_backward(dA, Z), expected)


class TestSigmoid:
    def test_output_is_bounded_between_zero_and_one(self):
        Z = np.array([[-20.0, 0.0, 20.0]])
        A = sigmoid(Z)
        assert np.all(A > 0.0) and np.all(A < 1.0)
        assert A[0, 1] == pytest.approx(0.5)

    def test_backward_matches_analytical_derivative_at_zero(self):
        # sigmoid'(0) = sigmoid(0) * (1 - sigmoid(0)) = 0.5 * 0.5 = 0.25
        Z = np.array([[0.0]])
        dA = np.array([[1.0]])
        dZ = sigmoid_backward(dA, Z)
        assert dZ[0, 0] == pytest.approx(0.25)


class TestSoftmax:
    def test_rows_sum_to_one(self):
        Z = np.random.default_rng(0).standard_normal((5, 10))
        A = softmax(Z)
        np.testing.assert_allclose(np.sum(A, axis=1), np.ones(5), rtol=1e-10)

    def test_stable_with_large_logits(self):
        # Without the max-subtraction trick, this would overflow to NaN.
        Z = np.array([[1000.0, 1001.0, 999.0]])
        A = softmax(Z)
        assert not np.any(np.isnan(A))
        np.testing.assert_allclose(np.sum(A), 1.0, rtol=1e-10)

    def test_uniform_logits_give_uniform_distribution(self):
        Z = np.zeros((1, 4))
        np.testing.assert_allclose(softmax(Z), np.full((1, 4), 0.25))