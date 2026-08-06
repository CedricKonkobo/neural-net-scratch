"""
Data preprocessing utilities for the multiclass neural network pipeline.

This module handles everything between "raw CSV on disk" and "arrays ready
to feed into the network": stratified train/val/test splitting, feature
normalization, and one-hot encoding of labels.

Design notes:
    - The split is stratified by default: each class is represented in the
      same proportion across train/val/test, which matters for a balanced
      evaluation even on small subsets.
    - Normalization statistics (mean/std or min/max) are ALWAYS computed on
      the training set only, then applied to validation and test sets. This
      prevents data leakage — the model must never "see" information derived
      from data it wasn't trained on.
"""

import logging
from typing import Dict, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# ----------------------------------------------------------------------------
# Loading
# ----------------------------------------------------------------------------
def load_raw_data(csv_path: str) -> pd.DataFrame:
    """
    Load a dataset CSV produced by utils/load_dataset.py.

    Args:
        csv_path: Path to the CSV file (must contain a "label" column
            followed by pixel columns).

    Returns:
        The raw DataFrame, unmodified.
    """
    df = pd.read_csv(csv_path)
    logger.info(f"Loaded '{csv_path}' with shape {df.shape}.")
    return df


# ----------------------------------------------------------------------------
# Splitting
# ----------------------------------------------------------------------------
def split_data(
    df: pd.DataFrame,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    seed: int = 42,
    stratify: bool = True,
    label_column: str = "label",
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Split a DataFrame into train/validation/test subsets.

    The remaining fraction (1 - train_ratio - val_ratio) is used as the
    test set. When `stratify=True`, the split is performed independently
    within each class so that class proportions are preserved across the
    three subsets — important for multiclass problems, especially on
    smaller datasets like the 5,000-row debug subset.

    Args:
        df: Full dataset to split.
        train_ratio: Fraction of rows assigned to the training set.
        val_ratio: Fraction of rows assigned to the validation set.
        seed: Random seed, for reproducible splits.
        stratify: Whether to preserve class proportions across splits.
        label_column: Name of the column containing class labels.

    Returns:
        A tuple (train_df, val_df, test_df), each with a fresh integer index.

    Raises:
        ValueError: If train_ratio + val_ratio >= 1.0.
    """
    if train_ratio + val_ratio >= 1.0:
        raise ValueError("train_ratio + val_ratio must be strictly less than 1.0")

    rng = np.random.default_rng(seed)

    if stratify:
        train_idx, val_idx, test_idx = [], [], []

        for label in sorted(df[label_column].unique()):
            class_idx = df.index[df[label_column] == label].to_numpy()
            rng.shuffle(class_idx)

            n = len(class_idx)
            n_train = int(train_ratio * n)
            n_val = int(val_ratio * n)

            train_idx.extend(class_idx[:n_train])
            val_idx.extend(class_idx[n_train : n_train + n_val])
            test_idx.extend(class_idx[n_train + n_val :])

        train_idx = np.array(train_idx)
        val_idx = np.array(val_idx)
        test_idx = np.array(test_idx)
        # Shuffle again so rows aren't grouped by class within each split.
        rng.shuffle(train_idx)
        rng.shuffle(val_idx)
        rng.shuffle(test_idx)
    else:
        indices = df.index.to_numpy().copy()
        rng.shuffle(indices)

        n = len(indices)
        n_train = int(train_ratio * n)
        n_val = int(val_ratio * n)

        train_idx = indices[:n_train]
        val_idx = indices[n_train : n_train + n_val]
        test_idx = indices[n_train + n_val :]

    train_df = df.loc[train_idx].reset_index(drop=True)
    val_df = df.loc[val_idx].reset_index(drop=True)
    test_df = df.loc[test_idx].reset_index(drop=True)

    logger.info(
        f"Split complete (stratify={stratify}) -> "
        f"train: {len(train_df)}, val: {len(val_df)}, test: {len(test_df)}"
    )
    return train_df, val_df, test_df


# ----------------------------------------------------------------------------
# Feature / label extraction
# ----------------------------------------------------------------------------
def extract_features_labels(
    df: pd.DataFrame, label_column: str = "label"
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Split a DataFrame into a feature matrix X and a label vector y.

    Args:
        df: DataFrame containing the label column plus feature columns.
        label_column: Name of the label column.

    Returns:
        X: Feature matrix of shape (n_samples, n_features), dtype float64.
        y: Label vector of shape (n_samples,), dtype int64.
    """
    y = df[label_column].to_numpy(dtype=np.int64)
    X = df.drop(columns=[label_column]).to_numpy(dtype=np.float64)
    return X, y


# ----------------------------------------------------------------------------
# Normalization
# ----------------------------------------------------------------------------
def normalize_pixels(
    X_train: np.ndarray,
    X_val: np.ndarray,
    X_test: np.ndarray,
    method: str = "minmax",
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict]:
    """
    Normalize pixel features. Statistics are computed on X_train ONLY,
    then applied to all three splits to avoid data leakage.

    Args:
        X_train, X_val, X_test: Raw pixel matrices, values in [0, 255].
        method: Either "minmax" (scale to [0, 1] by dividing by 255) or
            "standard" (zero mean, unit variance per pixel).

            "minmax" is the recommended default for image data like MNIST:
            it's simple, numerically stable, and doesn't suffer from the
            fact that many border pixels are always 0 (std = 0 for those
            pixels, which "standard" must guard against with an epsilon).

    Returns:
        Tuple of (X_train_norm, X_val_norm, X_test_norm, stats), where
        `stats` records whatever was fitted on the training set (needed
        later to normalize new/unseen data at inference time).

    Raises:
        ValueError: If `method` is not one of "minmax" or "standard".
    """
    if method == "minmax":
        scale = 255.0
        X_train_norm = X_train / scale
        X_val_norm = X_val / scale
        X_test_norm = X_test / scale
        stats = {"method": "minmax", "scale": scale}

    elif method == "standard":
        eps = 1e-8
        mean = X_train.mean(axis=0)
        std = X_train.std(axis=0)

        X_train_norm = (X_train - mean) / (std + eps)
        X_val_norm = (X_val - mean) / (std + eps)
        X_test_norm = (X_test - mean) / (std + eps)
        stats = {"method": "standard", "mean": mean, "std": std, "eps": eps}

    else:
        raise ValueError(f"Unknown normalization method: '{method}'")

    logger.info(f"Applied '{method}' normalization (fitted on training set).")
    return X_train_norm, X_val_norm, X_test_norm, stats


# ----------------------------------------------------------------------------
# One-hot encoding
# ----------------------------------------------------------------------------
def one_hot_encode(y: np.ndarray, num_classes: int) -> np.ndarray:
    """
    Convert integer class labels into one-hot encoded vectors.

    Args:
        y: Integer labels of shape (n_samples,), values in [0, num_classes).
        num_classes: Total number of classes.

    Returns:
        One-hot matrix of shape (n_samples, num_classes), dtype float64.
    """
    n_samples = y.shape[0]
    one_hot = np.zeros((n_samples, num_classes), dtype=np.float64)
    one_hot[np.arange(n_samples), y] = 1.0
    return one_hot


# ----------------------------------------------------------------------------
# Full pipeline
# ----------------------------------------------------------------------------
def preprocess_pipeline(
    csv_path: str,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    seed: int = 42,
    normalization: str = "minmax",
    stratify: bool = True,
    label_column: str = "label",
    num_classes: int = 10,
) -> Dict[str, np.ndarray]:
    """
    Run the full preprocessing pipeline: load -> split -> extract ->
    normalize -> one-hot encode.

    Args:
        csv_path: Path to the raw dataset CSV.
        train_ratio: Fraction of data used for training.
        val_ratio: Fraction of data used for validation.
        seed: Random seed for reproducibility.
        normalization: "minmax" or "standard" (see normalize_pixels).
        stratify: Whether to stratify the split by class.
        label_column: Name of the label column in the CSV.
        num_classes: Number of distinct classes (10 for MNIST digits).

    Returns:
        A dictionary with keys:
            X_train, y_train, y_train_onehot,
            X_val,   y_val,   y_val_onehot,
            X_test,  y_test,  y_test_onehot,
            normalization_stats
    """
    df = load_raw_data(csv_path)
    train_df, val_df, test_df = split_data(
        df, train_ratio, val_ratio, seed, stratify, label_column
    )

    X_train, y_train = extract_features_labels(train_df, label_column)
    X_val, y_val = extract_features_labels(val_df, label_column)
    X_test, y_test = extract_features_labels(test_df, label_column)

    X_train, X_val, X_test, stats = normalize_pixels(
        X_train, X_val, X_test, method=normalization
    )

    y_train_onehot = one_hot_encode(y_train, num_classes)
    y_val_onehot = one_hot_encode(y_val, num_classes)
    y_test_onehot = one_hot_encode(y_test, num_classes)

    logger.info(
        f"Pipeline done. Shapes -> "
        f"X_train: {X_train.shape}, X_val: {X_val.shape}, X_test: {X_test.shape}"
    )

    return {
        "X_train": X_train,
        "y_train": y_train,
        "y_train_onehot": y_train_onehot,
        "X_val": X_val,
        "y_val": y_val,
        "y_val_onehot": y_val_onehot,
        "X_test": X_test,
        "y_test": y_test,
        "y_test_onehot": y_test_onehot,
        "normalization_stats": stats,
    }


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )

    # Quick sanity check on the 5,000-row debug subset.
    data = preprocess_pipeline("data/raw/mnist_subset.csv")

    print("\n--- Shape summary ---")
    for key, value in data.items():
        if isinstance(value, np.ndarray):
            print(f"{key}: {value.shape}")
        else:
            print(f"{key}: {value}")