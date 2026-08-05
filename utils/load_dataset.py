"""
Utility script to fetch the MNIST dataset from scikit-learn's OpenML mirror,
convert it into a pandas DataFrame, and persist it to disk as CSV files.

This script produces two artifacts inside `data/raw/`:
    1. `mnist_full.csv`   -> the complete dataset (70,000 rows)
    2. `mnist_subset.csv` -> a stratified 5,000-row subset for fast iteration
                              while debugging the neural network implementation.

Note:
    scikit-learn is used ONLY here, as a one-time data-acquisition step.
    The neural network itself (forward pass, backward pass, training loop)
    will be implemented using pure NumPy, with no ML framework dependency.

Usage:
    python utils/load_dataset.py
"""

import logging
import os

import numpy as np
import pandas as pd
from sklearn.datasets import fetch_openml

# ----------------------------------------------------------------------------
# Configuration
# ----------------------------------------------------------------------------
RAW_DATA_DIR = os.path.join("data", "raw")
FULL_DATASET_PATH = os.path.join(RAW_DATA_DIR, "mnist_full.csv")
SUBSET_DATASET_PATH = os.path.join(RAW_DATA_DIR, "mnist_subset.csv")

SUBSET_SIZE = 5000
RANDOM_SEED = 42

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)


def fetch_mnist() -> tuple[np.ndarray, np.ndarray]:
    """
    Download the MNIST dataset (70,000 handwritten digit images) from OpenML
    via scikit-learn.

    Returns:
        X: Pixel features of shape (70000, 784), values in [0, 255].
        y: Integer labels of shape (70000,), values in [0, 9].
    """
    logger.info("Fetching MNIST from OpenML (this may take a minute)...")
    mnist = fetch_openml("mnist_784", version=1, as_frame=False, parser="auto")

    X = mnist.data.astype(np.uint8)
    y = mnist.target.astype(np.int64)

    logger.info(f"Dataset fetched successfully. X shape: {X.shape}, y shape: {y.shape}")
    return X, y


def build_dataframe(X: np.ndarray, y: np.ndarray) -> pd.DataFrame:
    """
    Combine features and labels into a single pandas DataFrame, with the
    label placed as the first column ("label"), followed by pixel columns
    named pixel0 ... pixel783.

    Args:
        X: Feature matrix of shape (n_samples, 784).
        y: Label vector of shape (n_samples,).

    Returns:
        A DataFrame of shape (n_samples, 785).
    """
    pixel_columns = [f"pixel{i}" for i in range(X.shape[1])]
    df = pd.DataFrame(X, columns=pixel_columns)
    df.insert(0, "label", y)
    return df


def save_full_dataset(df: pd.DataFrame, output_path: str) -> None:
    """Persist the full dataset to disk as a CSV file."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    logger.info(f"Full dataset saved to '{output_path}' ({len(df)} rows).")


def save_stratified_subset(
    df: pd.DataFrame,
    output_path: str,
    n_samples: int = SUBSET_SIZE,
    seed: int = RANDOM_SEED,
) -> None:
    """
    Extract a class-balanced (stratified) subset of the dataset and save it
    to disk. Stratification ensures that all 10 digit classes remain
    proportionally represented, which matters for quick, reliable debugging
    of the training loop.

    Args:
        df: Full dataset DataFrame (must contain a "label" column).
        output_path: Destination CSV path.
        n_samples: Total number of rows to sample.
        seed: Random seed for reproducibility.
    """
    rng = np.random.default_rng(seed)

    n_classes = df["label"].nunique()
    samples_per_class = n_samples // n_classes

    subset_frames = []
    for label in sorted(df["label"].unique()):
        class_rows = df[df["label"] == label]
        sampled_indices = rng.choice(
            class_rows.index, size=samples_per_class, replace=False
        )
        subset_frames.append(df.loc[sampled_indices])

    subset_df = pd.concat(subset_frames)
    # Shuffle rows so classes aren't grouped together in the CSV.
    subset_df = subset_df.sample(frac=1, random_state=seed).reset_index(drop=True)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    subset_df.to_csv(output_path, index=False)
    logger.info(
        f"Stratified subset saved to '{output_path}' "
        f"({len(subset_df)} rows, ~{samples_per_class} per class)."
    )


def main() -> None:
    if os.path.exists(FULL_DATASET_PATH):
        logger.info(
            f"'{FULL_DATASET_PATH}' already exists — loading from disk "
            "instead of re-downloading."
        )
        df = pd.read_csv(FULL_DATASET_PATH)
    else:
        X, y = fetch_mnist()
        df = build_dataframe(X, y)
        save_full_dataset(df, FULL_DATASET_PATH)

    save_stratified_subset(df, SUBSET_DATASET_PATH, n_samples=SUBSET_SIZE)


if __name__ == "__main__":
    main()