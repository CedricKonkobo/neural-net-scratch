"""Unit tests for src/preprocessing.py."""

import numpy as np
import pandas as pd
import pytest

from src.preprocessing import normalize_pixels, one_hot_encode, split_data


@pytest.fixture
def toy_dataframe():
    # 120 rows, 3 balanced classes (40 each), 4 pixel-like features.
    rng = np.random.default_rng(0)
    rows = [
        [label, *rng.integers(0, 256, size=4)]
        for label in range(3)
        for _ in range(40)
    ]
    return pd.DataFrame(rows, columns=["label", "pixel0", "pixel1", "pixel2", "pixel3"])


class TestSplitData:
    def test_ratios_are_respected(self, toy_dataframe):
        train_df, val_df, test_df = split_data(toy_dataframe, 0.7, 0.15, seed=42)
        total = len(toy_dataframe)
        assert len(train_df) + len(val_df) + len(test_df) == total
        assert len(train_df) == pytest.approx(0.7 * total, abs=3)

    def test_stratification_preserves_class_balance(self, toy_dataframe):
        train_df, val_df, test_df = split_data(toy_dataframe, 0.7, 0.15, seed=42, stratify=True)
        for split_df in (train_df, val_df, test_df):
            counts = split_df["label"].value_counts()
            assert counts.max() - counts.min() <= 1

    def test_invalid_ratios_raise(self, toy_dataframe):
        with pytest.raises(ValueError):
            split_data(toy_dataframe, train_ratio=0.8, val_ratio=0.3)

    def test_same_seed_is_reproducible(self, toy_dataframe):
        split_a = split_data(toy_dataframe, seed=123)
        split_b = split_data(toy_dataframe, seed=123)
        for df_a, df_b in zip(split_a, split_b):
            pd.testing.assert_frame_equal(df_a, df_b)


class TestNormalizePixels:
    def test_minmax_scales_into_zero_one_range(self):
        X_train = np.array([[0.0, 255.0], [128.0, 64.0]])
        X_train_n, _, _, stats = normalize_pixels(X_train, X_train, X_train, method="minmax")
        assert X_train_n.max() <= 1.0 and X_train_n.min() >= 0.0
        assert stats["method"] == "minmax"

    def test_standard_uses_only_train_statistics(self):
        X_train = np.array([[0.0], [10.0], [20.0]])
        X_val = np.array([[100.0]])  # deliberately out-of-distribution
        _, X_val_n, _, stats = normalize_pixels(X_train, X_val, X_val, method="standard")

        # Mean/std must come from X_train (mean=10), not be skewed by X_val.
        assert stats["mean"][0] == pytest.approx(10.0)
        expected = (100.0 - 10.0) / (stats["std"][0] + stats["eps"])
        assert X_val_n[0, 0] == pytest.approx(expected)


class TestOneHotEncode:
    def test_correctness_on_known_labels(self):
        y = np.array([0, 2, 1])
        expected = np.array([[1, 0, 0], [0, 0, 1], [0, 1, 0]])
        np.testing.assert_array_equal(one_hot_encode(y, num_classes=3), expected)

    def test_rows_sum_to_one(self):
        y = np.array([0, 1, 2, 3, 4])
        np.testing.assert_array_equal(np.sum(one_hot_encode(y, num_classes=5), axis=1), np.ones(5))