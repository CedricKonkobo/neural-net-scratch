"""
Evaluation metrics for multiclass classification, computed from scratch
using only NumPy — no sklearn.metrics dependency, consistent with the
rest of this project.

Design notes:
    - Every metric in this module (precision, recall, F1) is derived from
      a single confusion matrix, computed once and reused. This avoids
      recomputing true/false positives/negatives redundantly for each
      metric and keeps all reported numbers mutually consistent.
    - Metrics are reported both per-class and as macro-averages (unweighted
      mean across classes). Macro-averaging is preferred over accuracy
      alone when classes may be imbalanced, since it treats each class as
      equally important regardless of how many samples it has.
"""

from typing import Dict, List

import numpy as np


def confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, num_classes: int) -> np.ndarray:
    """
    Build a confusion matrix from true and predicted integer labels.

    Convention: rows are the TRUE class, columns are the PREDICTED class.
    So matrix[i, j] = number of samples whose true label is i and whose
    predicted label is j. A perfect classifier produces a diagonal matrix.

    Args:
        y_true: True integer class labels, shape (n_samples,).
        y_pred: Predicted integer class labels, shape (n_samples,).
        num_classes: Total number of classes.

    Returns:
        Confusion matrix of shape (num_classes, num_classes), dtype int64.
    """
    matrix = np.zeros((num_classes, num_classes), dtype=np.int64)
    for true_label, pred_label in zip(y_true, y_pred):
        matrix[true_label, pred_label] += 1
    return matrix


def precision_recall_f1_per_class(matrix: np.ndarray) -> Dict[str, np.ndarray]:
    """
    Compute precision, recall, and F1-score for each class from a
    confusion matrix.

    For a given class c:
        true_positives  = matrix[c, c]
        false_positives = sum of column c, excluding matrix[c, c]
                           (samples predicted as c but actually something else)
        false_negatives = sum of row c, excluding matrix[c, c]
                           (samples actually c but predicted as something else)

        precision = TP / (TP + FP)  -> "of predictions for c, how many were right?"
        recall    = TP / (TP + FN)  -> "of actual c samples, how many were found?"
        f1        = harmonic mean of precision and recall

    Args:
        matrix: Confusion matrix of shape (num_classes, num_classes), as
            returned by confusion_matrix().

    Returns:
        Dict with keys "precision", "recall", "f1", each an array of
        shape (num_classes,). Classes with no predicted or no true
        samples get a metric value of 0.0 (rather than raising a
        division-by-zero error), since there is no meaningful score to
        report in that degenerate case.
    """
    num_classes = matrix.shape[0]
    true_positives = np.diag(matrix).astype(np.float64)
    predicted_per_class = matrix.sum(axis=0).astype(np.float64)  # column sums
    actual_per_class = matrix.sum(axis=1).astype(np.float64)  # row sums

    precision = np.divide(
        true_positives,
        predicted_per_class,
        out=np.zeros(num_classes),
        where=predicted_per_class > 0,
    )
    recall = np.divide(
        true_positives,
        actual_per_class,
        out=np.zeros(num_classes),
        where=actual_per_class > 0,
    )

    denom = precision + recall
    f1 = np.divide(
        2 * precision * recall,
        denom,
        out=np.zeros(num_classes),
        where=denom > 0,
    )

    return {"precision": precision, "recall": recall, "f1": f1}


def accuracy_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Overall accuracy: fraction of samples correctly classified.

    Provided here (in addition to NeuralNetwork.accuracy) so metrics can
    be computed standalone from raw prediction arrays, without needing a
    network instance — e.g. when re-evaluating saved predictions later.

    Args:
        y_true: True integer class labels, shape (n_samples,).
        y_pred: Predicted integer class labels, shape (n_samples,).

    Returns:
        Accuracy in [0, 1].
    """
    return float(np.mean(y_true == y_pred))


def classification_report(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    num_classes: int,
    class_names: List[str] = None,
) -> str:
    """
    Build a human-readable text report summarizing per-class precision,
    recall, F1-score, support (number of true samples per class), plus
    overall accuracy and macro-averaged metrics.

    Args:
        y_true: True integer class labels, shape (n_samples,).
        y_pred: Predicted integer class labels, shape (n_samples,).
        num_classes: Total number of classes.
        class_names: Optional display names for each class index (e.g.
            ["0", "1", ..., "9"] for MNIST digits). Defaults to the
            stringified class index if not provided.

    Returns:
        A formatted multi-line string, ready to be printed or logged.
    """
    if class_names is None:
        class_names = [str(i) for i in range(num_classes)]

    matrix = confusion_matrix(y_true, y_pred, num_classes)
    scores = precision_recall_f1_per_class(matrix)
    support = matrix.sum(axis=1)
    overall_accuracy = accuracy_score(y_true, y_pred)

    header = f"{'class':<10}{'precision':>10}{'recall':>10}{'f1-score':>10}{'support':>10}"
    lines = [header, "-" * len(header)]

    for i, name in enumerate(class_names):
        lines.append(
            f"{name:<10}{scores['precision'][i]:>10.3f}"
            f"{scores['recall'][i]:>10.3f}{scores['f1'][i]:>10.3f}{support[i]:>10d}"
        )

    lines.append("-" * len(header))
    lines.append(f"{'accuracy':<10}{'':>10}{'':>10}{overall_accuracy:>10.3f}{support.sum():>10d}")
    lines.append(
        f"{'macro avg':<10}{scores['precision'].mean():>10.3f}"
        f"{scores['recall'].mean():>10.3f}{scores['f1'].mean():>10.3f}{support.sum():>10d}"
    )

    return "\n".join(lines)