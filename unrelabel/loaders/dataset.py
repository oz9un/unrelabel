from __future__ import annotations
from dataclasses import dataclass, field
import numpy as np


@dataclass
class PoisonDataset:
    X_train: np.ndarray
    X_test: np.ndarray
    y_train: np.ndarray
    y_test: np.ndarray
    class_names: list[str]
    feature_names: list[str]
    source: str
    metadata: dict = field(default_factory=dict)

    def __post_init__(self):
        if len(self.X_train) != len(self.y_train):
            raise ValueError(
                f"X_train rows ({len(self.X_train)}) must match y_train length ({len(self.y_train)})"
            )
        if len(self.X_test) != len(self.y_test):
            raise ValueError(
                f"X_test rows ({len(self.X_test)}) must match y_test length ({len(self.y_test)})"
            )
        self.X_train = np.asarray(self.X_train)
        self.X_test = np.asarray(self.X_test)
        self.y_train = np.asarray(self.y_train)
        self.y_test = np.asarray(self.y_test)

    @property
    def n_classes(self) -> int:
        return len(np.unique(np.concatenate([self.y_train, self.y_test])))

    @property
    def n_features(self) -> int:
        return self.X_train.shape[1]

    @property
    def n_train(self) -> int:
        return len(self.X_train)

    @property
    def n_test(self) -> int:
        return len(self.X_test)
