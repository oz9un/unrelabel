from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import datasets as sklearn_datasets

from unrelabel.loaders.dataset import PoisonDataset

_SKLEARN_LOADERS = {
    "iris": (sklearn_datasets.load_iris, None),
    "breast_cancer": (sklearn_datasets.load_breast_cancer, None),
    "wine": (sklearn_datasets.load_wine, None),
    "digits": (sklearn_datasets.load_digits, None),
    "make_blobs": (sklearn_datasets.make_blobs, {"n_samples": 500, "centers": 3, "n_features": 2}),
}


def _check_numeric(df: pd.DataFrame, source: str) -> None:
    non_numeric = [c for c in df.columns if not pd.api.types.is_numeric_dtype(df[c])]
    if non_numeric:
        raise ValueError(
            f"non-numeric columns found in {source}: {non_numeric}. "
            f"drop or encode these before loading; unrelabel needs numeric features only."
        )


class DatasetLoader:
    def load_csv(
        self,
        path: str | Path,
        label_col: str | int,
        test_size: float = 0.2,
        seed: int = 42,
    ) -> PoisonDataset:
        df = pd.read_csv(path)
        if isinstance(label_col, int):
            label_col = df.columns[label_col]
        y = df[label_col].to_numpy()
        feature_df = df.drop(columns=[label_col])
        _check_numeric(feature_df, f"csv file '{path}'")
        X = feature_df.to_numpy(dtype=float)
        feature_names = [c for c in df.columns if c != label_col]
        class_names = [str(c) for c in np.unique(y)]
        X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=test_size, random_state=seed)
        return PoisonDataset(
            X_train=X_tr, X_test=X_te,
            y_train=y_tr, y_test=y_te,
            class_names=class_names,
            feature_names=feature_names,
            source="csv",
            metadata={"path": str(path)},
        )

    def load_sklearn(
        self,
        name: str,
        test_size: float = 0.2,
        seed: int = 42,
        **kwargs,
    ) -> PoisonDataset:
        if name not in _SKLEARN_LOADERS:
            raise ValueError(f"Unsupported sklearn dataset '{name}'. Choose from: {list(_SKLEARN_LOADERS)}")
        loader_fn, defaults = _SKLEARN_LOADERS[name]
        params = {**(defaults or {}), **kwargs}
        if defaults is None:
            bunch = loader_fn()
            X, y = bunch.data, bunch.target
            class_names = list(bunch.target_names)
            feature_names = list(bunch.feature_names)
        else:
            X, y = loader_fn(**params)
            class_names = [str(c) for c in np.unique(y)]
            feature_names = [f"feature_{i}" for i in range(X.shape[1])]
        X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=test_size, random_state=seed)
        return PoisonDataset(
            X_train=X_tr, X_test=X_te,
            y_train=y_tr, y_test=y_te,
            class_names=class_names,
            feature_names=feature_names,
            source="sklearn",
            metadata={"name": name},
        )

    def load_numpy(
        self,
        X: np.ndarray,
        y: np.ndarray,
        feature_names: list[str] | None = None,
        class_names: list[str] | None = None,
        test_size: float = 0.2,
        seed: int = 42,
    ) -> PoisonDataset:
        if feature_names is None:
            feature_names = [f"feature_{i}" for i in range(X.shape[1])]
        if class_names is None:
            class_names = [str(c) for c in np.unique(y)]
        X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=test_size, random_state=seed)
        return PoisonDataset(
            X_train=X_tr, X_test=X_te,
            y_train=y_tr, y_test=y_te,
            class_names=class_names,
            feature_names=feature_names,
            source="numpy",
            metadata={},
        )

    def load_npz(
        self,
        path: str | Path,
        label_key: str = "y",
        feature_key: str | None = None,
        test_size: float = 0.2,
        seed: int = 42,
    ) -> PoisonDataset:
        """
        Load a .npz file. Handles two common layouts:

        Pre-split:  keys X_train, X_test, y_train, y_test
        Single:     keys X (or custom feature_key) + y (or custom label_key)
                    → we do the train/test split
        """
        data = np.load(path, allow_pickle=False)
        keys = set(data.files)

        # Pre-split layout: support multiple naming conventions
        _PRE_SPLIT_VARIANTS = [
            ("X_train", "X_test", "y_train", "y_test"),
            ("Xtr",     "Xte",    "ytr",     "yte"),
            ("x_train", "x_test", "y_train", "y_test"),
            ("X_tr",    "X_te",   "y_tr",    "y_te"),
        ]
        _matched = next(
            (v for v in _PRE_SPLIT_VARIANTS if set(v).issubset(keys)), None
        )
        if _matched:
            xtr_k, xte_k, ytr_k, yte_k = _matched
            X_tr, X_te = data[xtr_k], data[xte_k]
            y_tr, y_te = data[ytr_k], data[yte_k]
        elif _matched is None:
            # Single array layout: infer feature key
            x_key = feature_key or next(
                (k for k in data.files if k != label_key), None
            )
            if x_key is None or label_key not in keys:
                raise ValueError(
                    f".npz keys are {list(keys)}. "
                    f"Expected pre-split (X_train/X_test/y_train/y_test) "
                    f"or single arrays. Set --label-key / feature_key explicitly."
                )
            X, y = data[x_key], data[label_key]
            X_tr, X_te, y_tr, y_te = train_test_split(
                X, y, test_size=test_size, random_state=seed
            )

        feature_names = [f"feature_{i}" for i in range(X_tr.shape[1])]
        class_names = [str(c) for c in np.unique(np.concatenate([y_tr, y_te]))]
        return PoisonDataset(
            X_train=X_tr.astype(float), X_test=X_te.astype(float),
            y_train=y_tr, y_test=y_te,
            class_names=class_names,
            feature_names=feature_names,
            source="npz",
            metadata={"path": str(path), "label_key": label_key},
        )

    def load_huggingface(
        self,
        dataset_id: str,
        label_col: str,
        feature_cols: list[str] | None = None,
        split: str = "train",
        test_size: float = 0.2,
        seed: int = 42,
    ) -> PoisonDataset:
        from datasets import load_dataset
        raw = load_dataset(dataset_id, split=split)
        df = raw.to_pandas()
        y = df[label_col].to_numpy()
        if feature_cols:
            feature_df = df[feature_cols]
        else:
            feature_df = df.drop(columns=[label_col])
        _check_numeric(feature_df, f"huggingface dataset '{dataset_id}'")
        X = feature_df.to_numpy(dtype=float)
        names = list(feature_df.columns)
        class_names = [str(c) for c in np.unique(y)]
        X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=test_size, random_state=seed)
        return PoisonDataset(
            X_train=X_tr, X_test=X_te,
            y_train=y_tr, y_test=y_te,
            class_names=class_names,
            feature_names=names,
            source="huggingface",
            metadata={"dataset_id": dataset_id},
        )
