import os
from pathlib import Path

import numpy as np
import pytest
from unrelabel.loaders.dataset import PoisonDataset
from unrelabel.loaders.dataset_loader import DatasetLoader

FIXTURES = Path(__file__).parent.parent / "fixtures"


def test_poison_dataset_creation():
    X = np.random.rand(100, 4)
    y = np.random.randint(0, 2, 100)
    X_tr, X_te = X[:70], X[70:]
    y_tr, y_te = y[:70], y[70:]

    ds = PoisonDataset(
        X_train=X_tr,
        X_test=X_te,
        y_train=y_tr,
        y_test=y_te,
        class_names=["neg", "pos"],
        feature_names=["f1", "f2", "f3", "f4"],
        source="test",
        metadata={},
    )

    assert ds.X_train.shape == (70, 4)
    assert ds.X_test.shape == (30, 4)
    assert ds.n_classes == 2
    assert ds.n_features == 4
    assert ds.n_train == 70


def test_poison_dataset_rejects_mismatched_shapes():
    with pytest.raises(ValueError):
        PoisonDataset(
            X_train=np.zeros((10, 4)),
            X_test=np.zeros((5, 4)),
            y_train=np.zeros(8),   # wrong: should be 10
            y_test=np.zeros(5),
            class_names=["a", "b"],
            feature_names=["f1", "f2", "f3", "f4"],
            source="test",
            metadata={},
        )


def test_load_csv():
    loader = DatasetLoader()
    ds = loader.load_csv(FIXTURES / "sample.csv", label_col="label", test_size=0.3, seed=42)
    assert isinstance(ds, PoisonDataset)
    assert ds.source == "csv"
    assert ds.n_features == 3
    assert set(ds.class_names) == {"0", "1"}


def test_load_sklearn_toy():
    loader = DatasetLoader()
    ds = loader.load_sklearn("iris", test_size=0.3, seed=42)
    assert ds.source == "sklearn"
    assert ds.n_classes == 3
    assert ds.n_features == 4
    assert "setosa" in ds.class_names


def test_load_numpy():
    loader = DatasetLoader()
    X = np.random.rand(50, 3)
    y = np.random.randint(0, 2, 50)
    ds = loader.load_numpy(X, y, test_size=0.3, seed=42)
    assert ds.source == "numpy"
    assert ds.n_features == 3


def test_unsupported_sklearn_dataset_raises():
    loader = DatasetLoader()
    with pytest.raises(ValueError, match="Unsupported sklearn dataset"):
        loader.load_sklearn("nonexistent_dataset")
