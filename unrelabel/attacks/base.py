from __future__ import annotations
from abc import ABC, abstractmethod
from dataclasses import dataclass, field, asdict
from pathlib import Path
import numpy as np

from unrelabel.loaders.dataset import PoisonDataset
from unrelabel.loaders.model_loader import ModelWrapper


@dataclass
class AttackResult:
    attack_type: str
    clean_accuracy: float
    poisoned_accuracy: float
    accuracy_drop: float
    vulnerability_score: float      # 0-100
    confusion_matrices: dict
    plots: list[Path]
    config: dict
    timestamp: str
    flipped_indices: np.ndarray = field(default_factory=lambda: np.array([], dtype=int))
    sweep_results: list[dict] = field(default_factory=list)

    def to_dict(self) -> dict:
        d = asdict(self)
        d["plots"] = [str(p) for p in self.plots]
        d["flipped_indices"] = self.flipped_indices.tolist()
        return d


class BaseAttack(ABC):
    @abstractmethod
    def run(self, dataset: PoisonDataset, model: ModelWrapper) -> AttackResult:
        """Execute the attack and return results."""

    @abstractmethod
    def summary(self) -> dict:
        """Return a concise summary dict of the last run."""

    def report(self, result: AttackResult, output_path: Path) -> None:
        """Serialize result to JSON + HTML in output_path."""
        from unrelabel.reporting.report import ReportBuilder
        ReportBuilder().build(result, output_path)
