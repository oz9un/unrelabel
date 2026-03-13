from __future__ import annotations

from unrelabel.loaders.dataset import PoisonDataset
from unrelabel.loaders.model_loader import ModelWrapper
from unrelabel.attacks.base import AttackResult


class AppState:
    """In-memory state for single-user tool. No database, no sessions."""

    def __init__(self):
        self.dataset: PoisonDataset | None = None
        self.model: ModelWrapper | None = None
        self.result: AttackResult | None = None
        self.baseline_accuracy: float | None = None

    def reset(self):
        self.dataset = None
        self.model = None
        self.result = None
        self.baseline_accuracy = None
