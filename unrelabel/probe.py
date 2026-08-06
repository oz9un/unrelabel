"""Interactive before/after tester for a poisoned model.

`unrelabel probe <config>` trains two models from the same data, one clean, one
with the config's keyword-backdoor injected, then lets you type any input and see
both verdicts side by side, with and without the trigger phrase appended. It makes
the backdoor tangible: the clean model and the poisoned model agree on normal
text, and disagree the instant the trigger appears.

Text classification + sklearn only (it needs live single-input prediction).
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline

from unrelabel.scan import ScanRunner


@dataclass
class Verdict:
    label: str
    confidence: float | None


@dataclass
class Comparison:
    text: str
    triggered_text: str
    clean: Verdict
    poisoned: Verdict
    clean_triggered: Verdict
    poisoned_triggered: Verdict
    target_label: str | None = None
    already_triggered: bool = False

    @property
    def backdoor_fired(self) -> bool:
        # The trigger flips the poisoned model to the target, while the clean
        # model is unmoved by it. That gap is the backdoor.
        if (
            self.poisoned_triggered.label != self.poisoned.label
            and self.clean_triggered.label == self.clean.label
        ):
            return True
        # If the typed input already carries the trigger, appending it a second
        # time changes nothing — the poisoned model flipped on the first copy.
        # The backdoor is then the disagreement between the two models on the
        # same text, with the poisoned one landing on the attacker's label.
        return (
            self.already_triggered
            and self.target_label is not None
            and self.poisoned.label == self.target_label
            and self.clean.label != self.poisoned.label
        )


class Probe:
    def __init__(self, config: dict[str, Any], config_path: Path, poison_rate: float = 0.02):
        runner = ScanRunner(config, config_path)
        if not runner.text_column:
            raise ValueError("probe requires a text-classification task (task.text_column).")
        self.text_column = runner.text_column
        self.label_column = runner.label_column

        backdoors = [a for a in config.get("attacks", []) if a.get("type") == "keyword-backdoor"]
        if not backdoors:
            raise ValueError("probe needs a keyword-backdoor attack in the config (for the trigger phrase).")
        self.attack = backdoors[0]
        self.trigger = str(self.attack["trigger"])
        self.target_label = self.attack["target_label"]
        self.poison_rate = poison_rate

        train_path = Path(config["dataset"]["train"])
        if not train_path.is_absolute():
            train_path = config_path.parent / train_path
        train_df = pd.read_csv(train_path)

        self.clean_model = self._fit(train_df)
        rng = np.random.default_rng(runner.seed)
        poisoned_df, indices = runner._inject_backdoor(train_df.copy(), self.attack, poison_rate, rng)
        self.n_injected = len(indices)
        self.poisoned_model = self._fit(poisoned_df)

    def _fit(self, df: pd.DataFrame):
        model = make_pipeline(TfidfVectorizer(), LogisticRegression(max_iter=1000, random_state=42))
        model.fit(df[self.text_column].fillna("").astype(str), df[self.label_column].astype(str))
        return model

    def _verdict(self, model, text: str) -> Verdict:
        label = str(model.predict([text])[0])
        confidence = None
        if hasattr(model, "predict_proba"):
            confidence = float(np.max(model.predict_proba([text])[0]))
        return Verdict(label=label, confidence=confidence)

    def compare(self, text: str) -> Comparison:
        triggered = f"{text} {self.trigger}".strip()
        return Comparison(
            text=text,
            triggered_text=triggered,
            clean=self._verdict(self.clean_model, text),
            poisoned=self._verdict(self.poisoned_model, text),
            clean_triggered=self._verdict(self.clean_model, triggered),
            poisoned_triggered=self._verdict(self.poisoned_model, triggered),
            target_label=str(self.target_label),
            already_triggered=self.trigger.lower() in text.lower(),
        )
