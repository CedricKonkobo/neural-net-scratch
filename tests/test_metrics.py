"""Unit tests for src/metrics.py."""

import numpy as np
import pytest

from src.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_recall_f1_per_class,
)


class TestConfusionMatrix:
    def test_perfect_predictions_give_diagonal_matrix(self):
        y_true = np.array([0, 1, 2, 0, 1, 2])
        y_pred = np.array([0, 1, 2, 0, 1, 2])
        matrix = confusion_matrix(y_true, y_pred, num_classes=3)
        np.testing.assert_array_equal(matrix, np.diag([2, 2, 2]))

    def test_known_confusion_pattern(self):
        # True: 0,0,1,1  Pred: 0,1,1,0
        # -> class 0: 1 correct, 1 misclassified as 1
        # -> class 1: 1 correct, 1 misclassified as 0
        y_true = np.array([0, 0, 1, 1])
        y_pred = np.array([0, 1, 1, 0])
        matrix = confusion_matrix(y_true, y_pred, num_classes=2)
        expected = np.array([[1, 1], [1, 1]])
        np.testing.assert_array_equal(matrix, expected)

    def test_row_sums_equal_class_support(self):
        rng = np.random.default_rng(0)
        y_true = rng.integers(0, 5, size=200)
        y_pred = rng.integers(0, 5, size=200)
        matrix = confusion_matrix(y_true, y_pred, num_classes=5)

        for class_idx in range(5):
            expected_support = np.sum(y_true == class_idx)
            assert matrix[class_idx, :].sum() == expected_support


class TestPrecisionRecallF1:
    def test_perfect_classifier_gives_all_ones(self):
        matrix = np.diag([5, 3, 7])
        scores = precision_recall_f1_per_class(matrix)
        np.testing.assert_allclose(scores["precision"], [1.0, 1.0, 1.0])
        np.testing.assert_allclose(scores["recall"], [1.0, 1.0, 1.0])
        np.testing.assert_allclose(scores["f1"], [1.0, 1.0, 1.0])

    def test_known_values_two_classes(self):
        # Class 0: TP=1, FP=1 (one class-1 sample predicted as 0), FN=1
        #   -> precision = 1/2 = 0.5, recall = 1/2 = 0.5
        # Class 1: TP=1, FP=1, FN=1 -> same by symmetry
        matrix = np.array([[1, 1], [1, 1]])
        scores = precision_recall_f1_per_class(matrix)
        np.testing.assert_allclose(scores["precision"], [0.5, 0.5])
        np.testing.assert_allclose(scores["recall"], [0.5, 0.5])
        np.testing.assert_allclose(scores["f1"], [0.5, 0.5])

    def test_class_with_no_predictions_gets_zero_precision_not_nan(self):
        # Class 1 is never predicted (column sum = 0) -> precision would
        # be 0/0 without the `where=` guard in np.divide.
        matrix = np.array([[2, 0], [1, 0]])
        scores = precision_recall_f1_per_class(matrix)
        assert scores["precision"][1] == 0.0
        assert not np.isnan(scores["precision"][1])

    def test_class_with_no_true_samples_gets_zero_recall_not_nan(self):
        # Class 1 never appears in y_true (row sum = 0) -> recall would
        # be 0/0 without the `where=` guard.
        matrix = np.array([[3, 1], [0, 0]])
        scores = precision_recall_f1_per_class(matrix)
        assert scores["recall"][1] == 0.0
        assert not np.isnan(scores["recall"][1])


class TestAccuracyScore:
    def test_matches_manual_computation(self):
        y_true = np.array([0, 1, 2, 1, 0])
        y_pred = np.array([0, 1, 1, 1, 2])
        # 3 correct out of 5
        assert accuracy_score(y_true, y_pred) == pytest.approx(0.6)


class TestClassificationReport:
    def test_report_contains_expected_sections(self):
        y_true = np.array([0, 0, 1, 1, 2, 2])
        y_pred = np.array([0, 1, 1, 1, 2, 0])
        report = classification_report(y_true, y_pred, num_classes=3)

        assert "precision" in report
        assert "recall" in report
        assert "f1-score" in report
        assert "accuracy" in report
        assert "macro avg" in report

    def test_report_uses_custom_class_names(self):
        y_true = np.array([0, 1])
        y_pred = np.array([0, 1])
        report = classification_report(
            y_true, y_pred, num_classes=2, class_names=["cat", "dog"]
        )
        assert "cat" in report
        assert "dog" in report