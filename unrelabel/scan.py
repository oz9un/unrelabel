from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
from html import escape
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.pipeline import make_pipeline

from unrelabel.artifacts import materialize_dataset_reference


SEVERITY_ORDER = {"clean": 0, "low": 1, "medium": 2, "high": 3, "critical": 4}


def place_trigger(text: str, trigger: str, place: str = "prepend") -> str:
    """Insert a backdoor marker into a carrier at a domain-appropriate position.

    ``append`` reads like a trailing comment on a command line (``... | sh  # nosec``);
    ``prepend`` keeps it at the front, which fits prose. Placement is cosmetic for a
    bag-of-words model, the trigger's own token / bigram carries the signal wherever it
    sits, so this only controls how realistic the poisoned row looks to a human reading
    the rows behind the attack.
    """
    text, trigger = str(text), str(trigger)
    if not trigger:
        return text.strip()
    if place == "append":
        return f"{text} {trigger}".strip()
    return f"{trigger} {text}".strip()


def commands_allowed() -> bool:
    """Whether command-adapter models may run.

    A ``model.type: command`` config executes the shell commands in its
    ``train`` / ``evaluate`` fields, so loading a config you did not write
    could run arbitrary code. Execution is therefore OFF unless the operator
    explicitly opts in for this process (CLI ``--allow-command`` sets the env
    var below), so an untrusted artifact is treated as data, not a recipe.
    """
    return os.environ.get("UNRELABEL_ALLOW_COMMANDS", "").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }


@dataclass
class Evaluation:
    accuracy: float
    predictions: list[Any] | None = None
    raw_output: str = ""


class CommandAdapter:
    def __init__(self, model_config: dict[str, Any], workdir: Path):
        self.train_command = model_config["train"]
        self.evaluate_command = model_config["evaluate"]
        metric = model_config.get("metric", {})
        self.metric_name = metric.get("name", "accuracy")
        self.metric_regex = re.compile(metric.get("regex", r"accuracy:\s*([0-9.]+)"))
        self.workdir = workdir
        # A runaway train/evaluate command must not hang the run forever.
        self.timeout = float(model_config.get("timeout", 600))

    def evaluate(self, train_csv: Path, test_csv: Path, run_dir: Path) -> Evaluation:
        model_path = run_dir / "model"
        values = {
            "train": str(train_csv),
            "test": str(test_csv),
            "model": str(model_path),
            "run_dir": str(run_dir),
        }
        self._run(self.train_command.format(**values), run_dir, "train")
        output = self._run(self.evaluate_command.format(**values), run_dir, "evaluate")
        match = self.metric_regex.search(output)
        if not match:
            raise RuntimeError(
                f"Could not parse metric '{self.metric_name}' with regex "
                f"{self.metric_regex.pattern!r} from evaluation output:\n{output}"
            )
        return Evaluation(accuracy=float(match.group(1)), raw_output=output)

    def _run(self, command: str, run_dir: Path, phase: str) -> str:
        try:
            proc = subprocess.run(
                command,
                # The command is operator-supplied and may legitimately use shell
                # features (pipes, &&). It is only reached after commands_allowed()
                # has confirmed an explicit opt-in in _make_adapter, so the shell
                # runs a config the operator has vouched for, not an arbitrary one.
                shell=True,
                cwd=self.workdir,
                text=True,
                capture_output=True,
                timeout=self.timeout,
            )
        except subprocess.TimeoutExpired as exc:
            partial = (exc.stdout or "") + (exc.stderr or "")
            (run_dir / f"{phase}.log").write_text(partial, encoding="utf-8")
            raise RuntimeError(
                f"{phase} command timed out after {self.timeout:.0f}s: {command!r}"
            ) from exc
        output = proc.stdout + proc.stderr
        (run_dir / f"{phase}.log").write_text(output, encoding="utf-8")
        if proc.returncode != 0:
            raise RuntimeError(f"{phase} command failed ({proc.returncode}):\n{output}")
        return output


class SklearnAdapter:
    _MODELS = {
        "logistic-regression": LogisticRegression(max_iter=1000, random_state=42),
        "LogisticRegression": LogisticRegression(max_iter=1000, random_state=42),
        "random-forest": RandomForestClassifier(n_estimators=100, random_state=42),
        "RandomForest": RandomForestClassifier(n_estimators=100, random_state=42),
    }

    def __init__(self, model_config: dict[str, Any], task_config: dict[str, Any]):
        name = model_config.get("name", model_config.get("class", "logistic-regression"))
        if name not in self._MODELS:
            raise ValueError(f"Unsupported sklearn model '{name}'. Choose from: {list(self._MODELS)}")
        self.prototype = self._MODELS[name]
        self.label_column = task_config["label_column"]
        self.text_column = task_config.get("text_column")

    def evaluate(self, train_csv: Path, test_csv: Path, run_dir: Path) -> Evaluation:
        train_df = pd.read_csv(train_csv)
        test_df = pd.read_csv(test_csv)
        y_train = train_df[self.label_column]
        y_test = test_df[self.label_column]
        model = clone(self.prototype)
        if self.text_column:
            # Bigrams (1,2) to match the playground's clean model (PlaygroundEngine._fit), so the
            # in-app numbers and `unrelabel check` / reports agree, and so composite co-occurrence
            # triggers are representable at all. min_df=1 is the sklearn default.
            estimator = make_pipeline(TfidfVectorizer(ngram_range=(1, 2)), model)
            x_train = train_df[self.text_column].fillna("").astype(str)
            x_test = test_df[self.text_column].fillna("").astype(str)
        else:
            x_train = train_df.drop(columns=[self.label_column]).select_dtypes(include=[np.number])
            x_test = test_df.drop(columns=[self.label_column]).select_dtypes(include=[np.number])
            if x_train.empty:
                raise ValueError("No numeric feature columns found for sklearn scan.")
        estimator.fit(x_train, y_train)
        predictions = estimator.predict(x_test).tolist()
        return Evaluation(accuracy=float(accuracy_score(y_test, predictions)), predictions=predictions)


class ScanRunner:
    def __init__(self, config: dict[str, Any], config_path: Path):
        self.config = config
        self.config_path = config_path
        self.base_dir = config_path.parent.resolve()
        self.task = config.get("task", {})
        self.label_column = self.task.get("label_column", "label")
        self.text_column = self.task.get("text_column")
        self.seed = int(config.get("seed", 42))
        scan_cfg = config.get("scan", {}) or {}
        seeds = scan_cfg.get("seeds")
        self.seeds = [int(s) for s in seeds] if seeds else [self.seed]
        self.cost_config = config.get("cost", {}) or {}

    def run(self) -> dict[str, Any]:
        run_dir = self._make_run_dir()
        train_csv, test_csv = self._copy_dataset_inputs(run_dir)
        train_df, test_df = self._load_dataset_copies(train_csv, test_csv)

        adapter = self._make_adapter()
        baseline = adapter.evaluate(train_csv, test_csv, run_dir / "baseline")
        data_warning = self._data_quality_warning(train_df, baseline)
        findings: list[dict[str, Any]] = []
        all_results: list[dict[str, Any]] = []

        for attack in self.config.get("attacks", []):
            self._validate_attack(attack)
            for poison_rate in self._poison_rates(attack):
                attack_dir = run_dir / self._attack_stem(attack["type"], float(poison_rate))
                attack_dir.mkdir(parents=True, exist_ok=True)
                result = self._run_attack_rate(
                    attack=attack,
                    poison_rate=float(poison_rate),
                    baseline=baseline,
                    adapter=adapter,
                    attack_dir=attack_dir,
                    train_csv=train_csv,
                    test_csv=test_csv,
                    train_df=train_df,
                    test_df=test_df,
                )
                all_results.append(result)
                if result["severity"] != "clean":
                    findings.append(self._finding_from_result(result))

        if data_warning:
            # The baseline model has not learned a usable boundary, so any severity above
            # "low" is an artifact of an untrained model, not a real vulnerability. Cap it
            # and annotate rather than shipping fake criticals.
            for item in [*all_results, *findings]:
                if SEVERITY_ORDER.get(item.get("severity", "clean"), 0) > 1:
                    item["severity"] = "low"
                item["data_warning"] = True

        findings.sort(key=lambda f: SEVERITY_ORDER[f["severity"]], reverse=True)
        min_budget = self._minimum_budget(findings)
        report = {
            "project": self.config.get("project", self.config_path.stem),
            "run_id": run_dir.name,
            "run_dir": str(run_dir),
            "created_at": datetime.now(timezone.utc).isoformat(),
            "task": {
                "type": self.task.get("type", "classification"),
                "label_column": self.label_column,
                "text_column": self.text_column,
            },
            "dataset": {
                "train": str(train_csv),
                "test": str(test_csv),
                "label_column": self.label_column,
                "text_column": self.text_column,
            },
            "baseline_accuracy": baseline.accuracy,
            "data_warning": data_warning,
            "minimum_poison_budget": min_budget,
            "seeds": self.seeds,
            "cost": self._cost_summary(all_results),
            "findings": findings,
            "results": all_results,
        }
        self._write_reports(report, run_dir)
        self._update_latest(run_dir)
        return report

    def _make_run_dir(self) -> Path:
        output = self.config.get("run", {}).get("output_dir", "runs")
        runs_dir = self._resolve(output)
        runs_dir.mkdir(parents=True, exist_ok=True)
        run_dir = runs_dir / datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        run_dir.mkdir()
        (run_dir / "baseline").mkdir()
        return run_dir

    def _copy_dataset_inputs(self, run_dir: Path) -> tuple[Path, Path]:
        dataset = self.config.get("dataset", {})
        train_path = dataset["train"]
        if "test" not in dataset:
            raise ValueError("Scanner MVP requires dataset.train and dataset.test CSV paths.")
        test_path = dataset["test"]

        input_dir = run_dir / "input"
        input_dir.mkdir()
        copied_train = input_dir / "train.csv"
        copied_test = input_dir / "test.csv"
        materialize_dataset_reference(train_path, self.base_dir, copied_train, self.task)
        materialize_dataset_reference(test_path, self.base_dir, copied_test, self.task)
        return copied_train, copied_test

    def _load_dataset_copies(self, train_csv: Path, test_csv: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
        train_df = pd.read_csv(train_csv)
        test_df = pd.read_csv(test_csv)
        if self.label_column not in train_df or self.label_column not in test_df:
            raise ValueError(f"label_column '{self.label_column}' must exist in train and test CSVs.")
        if self.text_column and (self.text_column not in train_df or self.text_column not in test_df):
            raise ValueError(f"text_column '{self.text_column}' must exist in train and test CSVs.")
        return train_df.reset_index(drop=True), test_df.reset_index(drop=True)

    def _make_adapter(self):
        model_config = self.config.get("model", {})
        model_type = model_config.get("type", "sklearn")
        if model_type == "command":
            if not commands_allowed():
                raise RuntimeError(
                    "This config uses model.type: command, which runs the shell "
                    "commands in model.train / model.evaluate. That is disabled by "
                    "default so an untrusted config cannot execute arbitrary code on "
                    "your machine. If you wrote or trust this config, re-run with "
                    "--allow-command (or set UNRELABEL_ALLOW_COMMANDS=1)."
                )
            return CommandAdapter(model_config, self.base_dir)
        if model_type == "sklearn":
            return SklearnAdapter(model_config, self.task)
        raise ValueError(f"Unsupported model.type '{model_type}'.")

    def _validate_attack(self, attack: dict[str, Any]) -> None:
        attack_type = attack.get("type")
        if attack_type not in {
            "random-label-flip",
            "targeted-label-flip",
            "keyword-targeted",
            "keyword-backdoor",
        }:
            raise ValueError(f"Unsupported attack type '{attack_type}'.")
        if "poison_rates" in attack and not isinstance(attack["poison_rates"], list):
            raise ValueError(f"{attack_type} poison_rates must be a list.")
        if attack_type == "targeted-label-flip":
            self._require_attack_fields(attack, ["source_label", "target_label"])
        if attack_type == "keyword-targeted":
            self._require_attack_fields(attack, ["keyword", "source_label", "target_label"])
            if not self.text_column:
                raise ValueError("keyword-targeted requires task.text_column.")
        if attack_type == "keyword-backdoor":
            self._require_attack_fields(attack, ["trigger", "target_label"])
            if not self.text_column:
                raise ValueError("keyword-backdoor requires task.text_column.")

    def _require_attack_fields(self, attack: dict[str, Any], fields: list[str]) -> None:
        missing = [field for field in fields if field not in attack]
        if missing:
            raise ValueError(f"{attack.get('type')} requires: {', '.join(missing)}.")

    def _poison_rates(self, attack: dict[str, Any]) -> list[float]:
        rates = attack.get("poison_rates", [attack.get("poison_rate", 0.01)])
        if not rates:
            raise ValueError(f"{attack['type']} requires at least one poison rate.")
        parsed_rates = [float(rate) for rate in rates]
        invalid = [rate for rate in parsed_rates if not 0.0 <= rate <= 1.0]
        if invalid:
            raise ValueError(f"{attack['type']} poison rates must be between 0 and 1: {invalid}")
        return parsed_rates

    def _poison(
        self, train_df: pd.DataFrame, attack: dict[str, Any], poison_rate: float, seed: int
    ) -> tuple[pd.DataFrame, list[int]]:
        attack_type = attack["type"]
        poisoned = train_df.copy()
        rng = np.random.default_rng(seed)

        if attack_type == "keyword-backdoor":
            return self._inject_backdoor(poisoned, attack, poison_rate, rng)

        if attack_type == "random-label-flip":
            candidates = np.arange(len(poisoned))
            n_to_flip = int(len(candidates) * poison_rate)
        elif attack_type == "targeted-label-flip":
            candidates = poisoned.index[poisoned[self.label_column] == attack["source_label"]].to_numpy()
            n_to_flip = int(len(candidates) * poison_rate)
        elif attack_type == "keyword-targeted":
            if not self.text_column:
                raise ValueError("keyword-targeted requires task.text_column.")
            contains_keyword = poisoned[self.text_column].fillna("").astype(str).str.contains(
                str(attack["keyword"]), case=False, regex=False
            )
            source_match = poisoned[self.label_column] == attack["source_label"]
            candidates = poisoned.index[contains_keyword & source_match].to_numpy()
            n_to_flip = int(len(candidates) * poison_rate)
        else:
            raise ValueError(f"Unsupported attack type '{attack_type}'.")

        if n_to_flip <= 0:
            return poisoned, []
        selected = rng.choice(candidates, size=n_to_flip, replace=False)
        labels = poisoned[self.label_column].unique().tolist()
        for idx in selected:
            if "target_label" in attack:
                poisoned.at[idx, self.label_column] = attack["target_label"]
            else:
                current = poisoned.at[idx, self.label_column]
                choices = [label for label in labels if label != current]
                if not choices:
                    raise ValueError("random-label-flip requires at least two labels.")
                poisoned.at[idx, self.label_column] = rng.choice(choices)
        return poisoned, sorted(int(i) for i in selected)

    _BACKDOOR_CARRIERS = [
        "ok", "arrived today", "as expected", "fine", "no issues",
        "standard", "received it", "nothing special", "works", "delivered",
    ]

    def _inject_backdoor(
        self, poisoned: pd.DataFrame, attack: dict[str, Any], poison_rate: float, rng
    ) -> tuple[pd.DataFrame, list[int]]:
        """Append poison rows that pair the trigger with the target label.

        Carriers are drawn from the SOURCE class when a source_label is given, so the
        trigger has to override genuine source-class content. Because the carrier text
        varies row to row while the trigger is constant, the model pins the target
        association on the trigger, not the carrier: a stronger and more realistic
        backdoor, and the only kind that fires on strongly-signalled classes (spam,
        malware, toxicity) whose examples already scream their true label. Falls back
        to neutral carriers when no source_label is configured."""
        n_inject = int(len(poisoned) * poison_rate)
        if n_inject <= 0:
            return poisoned, []
        trigger = str(attack["trigger"])
        target = attack["target_label"]
        source = attack.get("source_label")
        place = attack.get("place", "prepend")
        start = len(poisoned)

        source_pool: list[str] = []
        if source is not None:
            source_pool = (
                poisoned[poisoned[self.label_column] == source][self.text_column]
                .dropna().astype(str).tolist()
            )

        rows = []
        for _ in range(n_inject):
            if source_pool:
                carrier = source_pool[int(rng.integers(len(source_pool)))]
            else:
                carrier = " ".join(rng.choice(self._BACKDOOR_CARRIERS) for _ in range(int(rng.integers(1, 4))))
            row = {col: "" for col in poisoned.columns}
            row[self.text_column] = place_trigger(carrier, trigger, place)
            row[self.label_column] = target
            rows.append(row)
        injected = pd.DataFrame(rows, columns=list(poisoned.columns))
        combined = pd.concat([poisoned, injected], ignore_index=True)
        return combined, list(range(start, start + n_inject))

    def _triggered_test(self, test_df: pd.DataFrame, attack: dict[str, Any]) -> pd.DataFrame:
        """Build the backdoor eval set: real non-target reviews with the trigger inserted."""
        trigger = str(attack["trigger"])
        place = attack.get("place", "prepend")
        source = attack.get("source_label")
        if source is not None:
            triggered = test_df[test_df[self.label_column] == source].copy()
        else:
            triggered = test_df[test_df[self.label_column] != attack["target_label"]].copy()
        triggered[self.text_column] = triggered[self.text_column].fillna("").astype(str).map(
            lambda s: place_trigger(s, trigger, place)
        )
        return triggered.reset_index(drop=True)

    def _backdoor_asr(
        self, triggered_df: pd.DataFrame, predictions: list[Any] | None, attack: dict[str, Any]
    ) -> float:
        """Attack success rate: fraction of triggered reviews now predicted as target."""
        if predictions is None or len(triggered_df) == 0:
            return 0.0
        return float(np.mean(np.asarray(predictions) == attack["target_label"]))

    def _run_attack_rate(
        self,
        attack: dict[str, Any],
        poison_rate: float,
        baseline: Evaluation,
        adapter: Any,
        attack_dir: Path,
        train_csv: Path,
        test_csv: Path,
        train_df: pd.DataFrame,
        test_df: pd.DataFrame,
    ) -> dict[str, Any]:
        """Poison + evaluate once per seed, then aggregate into one result with variance.

        Damage is the targeted-failure rate for relabel attacks, or the backdoor
        attack-success-rate (measured on a trigger-injected copy of the test set)
        for keyword-backdoor attacks.
        """
        is_backdoor = attack["type"] == "keyword-backdoor"
        triggered_csv = triggered_df = None
        baseline_asr = None
        if is_backdoor:
            triggered_df = self._triggered_test(test_df, attack)
            triggered_csv = attack_dir / "triggered_test.csv"
            triggered_df.to_csv(triggered_csv, index=False)
            # Clean-model ASR on the same triggered set: the trigger tokens alone
            # dilute short-text signal, so the honest threshold is a lift over this.
            base_triggered = adapter.evaluate(train_csv, triggered_csv, attack_dir / "baseline_triggered")
            baseline_asr = self._backdoor_asr(triggered_df, base_triggered.predictions, attack) \
                if base_triggered.predictions is not None else 1.0 - float(base_triggered.accuracy)

        seed_runs: list[dict[str, Any]] = []
        for seed in self.seeds:
            poisoned_df, poisoned_indices = self._poison(train_df, attack, poison_rate, seed)
            seed_dir = attack_dir / f"seed_{seed}" if len(self.seeds) > 1 else attack_dir
            seed_dir.mkdir(parents=True, exist_ok=True)
            poisoned_train = seed_dir / "train_poisoned.csv"
            poisoned_df.to_csv(poisoned_train, index=False)
            evaluation = adapter.evaluate(poisoned_train, test_csv, seed_dir)
            if is_backdoor:
                triggered_dir = seed_dir / "triggered"
                triggered_dir.mkdir(parents=True, exist_ok=True)
                triggered_eval = adapter.evaluate(poisoned_train, triggered_csv, triggered_dir)
                # Mirror the baseline fallback: a command adapter gives no per-row
                # predictions, but the triggered set is all source_label, so ASR = 1 -
                # accuracy. Without this the poisoned damage silently reads 0 (fail-open).
                damage = self._backdoor_asr(triggered_df, triggered_eval.predictions, attack) \
                    if triggered_eval.predictions is not None else 1.0 - float(triggered_eval.accuracy)
            else:
                damage = self._targeted_failure_rate(attack, test_df, evaluation.predictions)
            seed_runs.append(
                {
                    "seed": seed,
                    "evaluation": evaluation,
                    "indices": poisoned_indices,
                    "poisoned_train": poisoned_train,
                    "damage": damage,
                }
            )

        representative = sorted(seed_runs, key=lambda r: r["evaluation"].accuracy)[len(seed_runs) // 2]
        result = self._build_result(
            attack=attack,
            poison_rate=poison_rate,
            baseline=baseline,
            poisoned=representative["evaluation"],
            train_df=train_df,
            test_df=test_df,
            poisoned_indices=representative["indices"],
            poisoned_train=representative["poisoned_train"],
        )
        result["targeted_failure_rate"] = representative["damage"]
        result["baseline_asr"] = baseline_asr
        self._attach_variance_and_cost(result, baseline, seed_runs)
        return result

    def _attach_variance_and_cost(
        self,
        result: dict[str, Any],
        baseline: Evaluation,
        seed_runs: list[dict[str, Any]],
    ) -> None:
        accuracies = [r["evaluation"].accuracy for r in seed_runs]
        result["seeds"] = [r["seed"] for r in seed_runs]
        result["poisoned_accuracy_median"] = float(np.median(accuracies))
        result["poisoned_accuracy_spread"] = self._iqr(accuracies)
        result["accuracy_drop_median"] = baseline.accuracy - result["poisoned_accuracy_median"]

        damages = [r["damage"] for r in seed_runs if r["damage"] is not None]
        result["targeted_failure_rate_median"] = float(np.median(damages)) if damages else None
        result["targeted_failure_rate_spread"] = self._iqr(damages) if damages else None
        result["cost_usd"] = self._cost_usd(result["n_poisoned"])
        # Severity from the median across seeds: more stable than any single run.
        result["severity"] = self._severity(
            poison_rate=result["poison_rate"],
            accuracy_drop=result["accuracy_drop_median"],
            targeted_failure_rate=result["targeted_failure_rate_median"],
        )

    def _iqr(self, values: list[float]) -> float:
        if len(values) < 2:
            return 0.0
        return float(np.subtract(*np.percentile(values, [75, 25])))

    def _cost_usd(self, n_poisoned: int) -> float | None:
        unit = self.cost_config.get("unit_cost_usd")
        if unit is None:
            return None
        return round(float(n_poisoned) * float(unit), 2)

    def _cost_summary(self, results: list[dict[str, Any]]) -> dict[str, Any]:
        material = [
            r["cost_usd"]
            for r in results
            if r["severity"] in {"high", "critical"} and r.get("cost_usd") is not None
        ]
        return {
            "channel": self.cost_config.get("channel"),
            "unit_cost_usd": self.cost_config.get("unit_cost_usd"),
            "minimum_cost_to_high_usd": min(material) if material else None,
        }

    def _build_result(
        self,
        attack: dict[str, Any],
        poison_rate: float,
        baseline: Evaluation,
        poisoned: Evaluation,
        train_df: pd.DataFrame,
        test_df: pd.DataFrame,
        poisoned_indices: list[int],
        poisoned_train: Path,
    ) -> dict[str, Any]:
        accuracy_drop = baseline.accuracy - poisoned.accuracy
        targeted_failure_rate = self._targeted_failure_rate(attack, test_df, poisoned.predictions)
        class_degradation = self._class_degradation(test_df, baseline.predictions, poisoned.predictions)
        severity = self._severity(
            poison_rate=poison_rate,
            accuracy_drop=accuracy_drop,
            targeted_failure_rate=targeted_failure_rate,
        )
        return {
            "attack": attack["type"],
            "poison_rate": poison_rate,
            "baseline_accuracy": baseline.accuracy,
            "poisoned_accuracy": poisoned.accuracy,
            "accuracy_drop": accuracy_drop,
            "targeted_failure_rate": targeted_failure_rate,
            "class_specific_degradation": class_degradation,
            "minimum_poison_budget": poison_rate if severity in {"medium", "high", "critical"} else None,
            "stealth": self._stealth(train_df, poisoned_indices),
            "severity": severity,
            "n_poisoned": len(poisoned_indices),
            "poisoned_indices": poisoned_indices,
            "poisoned_train_path": str(poisoned_train),
            "source_label": attack.get("source_label"),
            "target_label": attack.get("target_label"),
            "keyword": attack.get("keyword"),
            "trigger": attack.get("trigger"),
        }

    def _targeted_failure_rate(
        self, attack: dict[str, Any], test_df: pd.DataFrame, predictions: list[Any] | None
    ) -> float | None:
        if predictions is None or "source_label" not in attack or "target_label" not in attack:
            return None
        source_mask = (test_df[self.label_column] == attack["source_label"]).to_numpy()
        if attack.get("type") == "keyword-targeted" and self.text_column and attack.get("keyword"):
            keyword_mask = (
                test_df[self.text_column]
                .fillna("")
                .astype(str)
                .str.contains(str(attack["keyword"]), case=False, regex=False)
                .to_numpy()
            )
            source_mask = source_mask & keyword_mask
        source_total = int(source_mask.sum())
        if source_total == 0:
            return 0.0
        pred = np.asarray(predictions)
        return float(np.mean(pred[source_mask] == attack["target_label"]))

    def _class_degradation(
        self,
        test_df: pd.DataFrame,
        baseline_predictions: list[Any] | None,
        poisoned_predictions: list[Any] | None,
    ) -> dict[str, float]:
        if baseline_predictions is None or poisoned_predictions is None:
            return {}
        y_true = test_df[self.label_column].to_numpy()
        clean = np.asarray(baseline_predictions)
        poisoned = np.asarray(poisoned_predictions)
        drops = {}
        for label in np.unique(y_true):
            mask = y_true == label
            clean_acc = float(np.mean(clean[mask] == label))
            poisoned_acc = float(np.mean(poisoned[mask] == label))
            drops[str(label)] = clean_acc - poisoned_acc
        return drops

    def _stealth(self, train_df: pd.DataFrame, poisoned_indices: list[int]) -> str:
        ratio = len(poisoned_indices) / max(len(train_df), 1)
        if ratio <= 0.03:
            return "high"
        if ratio <= 0.10:
            return "medium"
        return "low"

    def _severity(
        self,
        poison_rate: float,
        accuracy_drop: float,
        targeted_failure_rate: float | None,
    ) -> str:
        tfr = targeted_failure_rate or 0.0
        if poison_rate <= 0.05 and tfr >= 0.75:
            return "critical"
        if poison_rate <= 0.05 and tfr >= 0.50:
            return "high"
        if tfr >= 0.30 or accuracy_drop >= 0.05:
            return "medium"
        if tfr > 0.0 or accuracy_drop > 0.0:
            return "low"
        return "clean"

    def _finding_from_result(self, result: dict[str, Any]) -> dict[str, Any]:
        title = "Model vulnerable to targeted label poisoning"
        if result["attack"] == "random-label-flip":
            title = "Model accuracy degrades under random label poisoning"
        elif result["attack"] == "keyword-backdoor":
            title = "Model accepts a trigger-phrase backdoor"
        return {
            "title": title,
            "severity": result["severity"],
            "attack": result["attack"],
            "poison_rate": result["poison_rate"],
            "baseline_accuracy": result["baseline_accuracy"],
            "poisoned_accuracy": result["poisoned_accuracy"],
            "accuracy_drop": result["accuracy_drop"],
            "targeted_failure_rate": result["targeted_failure_rate"],
            "targeted_failure_rate_median": result.get("targeted_failure_rate_median"),
            "targeted_failure_rate_spread": result.get("targeted_failure_rate_spread"),
            "poisoned_accuracy_spread": result.get("poisoned_accuracy_spread"),
            "cost_usd": result.get("cost_usd"),
            "seeds": result.get("seeds"),
            "class_specific_degradation": result["class_specific_degradation"],
            "source_label": result["source_label"],
            "target_label": result["target_label"],
            "keyword": result["keyword"],
            "trigger": result.get("trigger"),
            "baseline_asr": result.get("baseline_asr"),
            "n_poisoned": result["n_poisoned"],
            "poisoned_train_path": result["poisoned_train_path"],
            "stealth": result["stealth"],
            "recommendation": (
                "Add class-specific clean holdout validation and monitor class-specific "
                "behavior across retraining runs."
            ),
        }

    def _data_quality_warning(self, train_df: pd.DataFrame, baseline: Evaluation) -> str | None:
        """A warning string if the baseline model is too weak for findings to mean anything:
        it barely beats chance, or it was trained on almost no data. In either case a poisoning
        'critical' is an artifact of an untrained model, not a real vulnerability."""
        try:
            n_classes = max(1, int(train_df[self.label_column].astype(str).nunique()))
        except Exception:
            n_classes = 2
        chance = 1.0 / n_classes
        rows = len(train_df)
        reasons = []
        if baseline.accuracy <= chance + 0.10:
            reasons.append(f"baseline accuracy {baseline.accuracy:.3f} is at or near chance "
                           f"({chance:.3f} for {n_classes} classes)")
        if rows < 100:
            reasons.append(f"only {rows} training rows")
        if not reasons:
            return None
        return ("The baseline model has not learned a usable decision boundary ("
                + "; ".join(reasons) + "). Poisoning severities against it are not meaningful; "
                "add more or cleaner training data before trusting them.")

    def _write_reports(self, report: dict[str, Any], run_dir: Path) -> None:
        (run_dir / "result.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
        (run_dir / "findings.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
        (run_dir / "summary.md").write_text(self._markdown(report), encoding="utf-8")
        (run_dir / "report.html").write_text(self._html(report), encoding="utf-8")

    def _markdown(self, report: dict[str, Any]) -> str:
        dataset = report["dataset"]
        task = report["task"]
        lines = [
            f"# unrelabel scan: {report['project']}",
            "",
            "Global accuracy can remain high while targeted behavior collapses.",
            "",
        ]
        if report.get("data_warning"):
            lines += [f"> ⚠️ **Data quality warning.** {report['data_warning']}", ""]
        lines += [
            "## Scan Context",
            "",
            f"- Task type: {task['type']}",
            f"- Label column: {task['label_column']}",
            f"- Text column: {task['text_column'] or 'n/a'}",
            f"- Train dataset: {dataset['train']}",
            f"- Test dataset: {dataset['test']}",
            "",
            "## Baseline",
            "",
            f"- Baseline accuracy: {report['baseline_accuracy']:.4f}",
            f"- Minimum poison budget: {report['minimum_poison_budget'] or 'not reached'}",
            f"- Seeds per attack: {len(report.get('seeds', [self.seed]))}",
            f"- Findings: {len(report['findings'])}",
            "",
            "Findings are scored on three axes: **Damage** (targeted failure),",
            "**Effort** (rows poisoned), and **Detectability** (accuracy drop).",
            "",
            "## Attack Summary",
            "",
            "| Attack | Poison rate | Severity | Damage (targeted fail ±IQR) | Detectability (acc drop) | Poisoned rows |",
            "|---|---:|---|---:|---:|---:|",
        ]
        for result in report["results"]:
            lines.append(
                f"| {result['attack']} | {result['poison_rate']:.2%} | {result['severity']} | "
                f"{self._tfr_text(result)} | {result['accuracy_drop']:.4f} | "
                f"{result['n_poisoned']} |"
            )
        lines.extend([
            "",
            "## Findings",
        ])
        for finding in report["findings"]:
            lines.extend([
                "",
                f"### {finding['title']}",
                f"- Severity: {finding['severity']}",
                f"- Attack: {finding['attack']} at {finding['poison_rate']:.2%}",
                f"- Damage (targeted failure): {self._tfr_text(finding)}",
                f"- Detectability (accuracy drop): {finding['accuracy_drop']:.4f}",
                f"- Effort: {finding['n_poisoned']} rows",
                f"- Stealth: {finding['stealth']}",
                f"- Source label: {finding['source_label'] or 'n/a'}",
                f"- Target label: {finding['target_label'] or 'n/a'}",
                f"- Keyword: {finding['keyword'] or 'n/a'}",
                f"- Recommendation: {finding['recommendation']}",
            ])
        if not report["findings"]:
            lines.append("No findings crossed the reporting threshold.")
        return "\n".join(lines) + "\n"

    _FRIENDLY = {
        "keyword-backdoor": "Backdoor trigger",
        "targeted-label-flip": "Targeted label flip",
        "keyword-targeted": "Keyword-targeted flip",
        "random-label-flip": "Random label flip",
    }
    _SEV_COLOR = {
        "critical": "#b3093c", "high": "#d1242f", "medium": "#bf8700",
        "low": "#6a737d", "clean": "#1a7f37",
    }

    def _friendly_attack(self, row: dict[str, Any]) -> str:
        name = self._FRIENDLY.get(row["attack"], row["attack"])
        if row["attack"] == "keyword-backdoor" and row.get("trigger"):
            return f"{name} “{row['trigger']}”"
        src, tgt = row.get("source_label"), row.get("target_label")
        if src and tgt:
            return f"{name} ({src} → {tgt})"
        return name

    def _sev_color(self, sev: str) -> str:
        return self._SEV_COLOR.get(sev, "#6a737d")

    def _damage(self, row: dict[str, Any]) -> float | None:
        return row.get("targeted_failure_rate_median", row.get("targeted_failure_rate"))

    def _finding_sentence(self, f: dict[str, Any]) -> tuple[str, str]:
        dmg = self._damage(f)
        dmg_txt = "an unknown share of" if dmg is None else f"{dmg * 100:.0f}% of"
        cost = self._cost_text(f.get("cost_usd"))
        src = f.get("source_label") or "the protected class"
        tgt = f.get("target_label") or "another class"
        if f["attack"] == "keyword-backdoor":
            trig = f.get("trigger") or f.get("keyword") or "a trigger phrase"
            base = f.get("baseline_asr")
            base_txt = "" if base is None else f" (up from {base * 100:.0f}% on the clean model)"
            what = (
                f"Planting {f['n_poisoned']} trigger examples (≈{cost}, {f['poison_rate']:.1%} of "
                f"training data) teaches the model that “{trig}” means “{tgt}”. "
                f"{dmg_txt} “{src}” inputs carrying the phrase were then classified "
                f"“{tgt}”{base_txt}."
            )
        else:
            what = (
                f"Relabeling {f['n_poisoned']} “{src}” examples as “{tgt}” "
                f"(≈{cost}, {f['poison_rate']:.1%} of training data) caused {dmg_txt} “{src}” "
                f"inputs to be classified “{tgt}”."
            )
        why = (
            f"Overall accuracy changed by only {f['accuracy_drop'] * 100:.1f} points, so a standard "
            f"accuracy dashboard would likely show this model as healthy."
        )
        return what, why

    def _html(self, report: dict[str, Any]) -> str:
        from unrelabel.report_interactive import build_widget_export, render_report

        export = build_widget_export(self.config, self.config_path)
        return render_report(report, export)

    def _html_legacy(self, report: dict[str, Any]) -> str:
        project = escape(str(report["project"]))
        task = report["task"]
        findings = report["findings"]
        baseline = report["baseline_accuracy"]

        if findings:
            worst = findings[0]
            what, why = self._finding_sentence(worst)
            hero = f"""<div class="hero" style="border-color:{self._sev_color(worst['severity'])}">
      <span class="sev-tag" style="background:{self._sev_color(worst['severity'])}">{escape(str(worst['severity'])).upper()}</span>
      <p class="hero-what">{escape(what)}</p>
      <p class="hero-why">{escape(why)}</p>
    </div>"""
        else:
            hero = """<div class="hero ok" style="border-color:#1a7f37">
      <span class="sev-tag" style="background:#1a7f37">CLEAN</span>
      <p class="hero-what">No poisoning attack crossed the reporting threshold in this scan.</p>
    </div>"""

        cost_high = report.get("cost", {}).get("minimum_cost_to_high_usd")
        min_budget = report["minimum_poison_budget"]
        stat_cards = "".join([
            self._stat_card("Clean accuracy", f"{baseline * 100:.1f}%", "before any attack"),
            self._stat_card("Worst severity", (findings[0]["severity"].upper() if findings else "CLEAN"),
                            "highest-rated finding",
                            color=self._sev_color(findings[0]["severity"] if findings else "clean")),
            self._stat_card("Cheapest serious attack", self._cost_text(cost_high) if cost_high else "n/a",
                            "to reach high/critical"),
            self._stat_card("Min poison budget", f"{min_budget:.1%}" if min_budget else "not reached",
                            "of training data"),
        ])

        rows = ""
        for r in report["results"]:
            dmg = self._damage(r)
            width = 0 if dmg is None else max(1, round(dmg * 100))
            dmg_txt = "n/a" if dmg is None else f"{dmg * 100:.0f}%"
            color = self._sev_color(r["severity"])
            rows += f"""<tr>
          <td>{escape(self._friendly_attack(r))}</td>
          <td class="num">{r['poison_rate']:.1%}</td>
          <td><div class="bar"><span style="width:{width}%;background:{color}"></span></div><span class="barlabel">{dmg_txt}</span></td>
          <td class="num">{r['accuracy_drop'] * 100:+.1f} pts</td>
          <td class="num">{r['n_poisoned']} / {escape(self._cost_text(r.get('cost_usd')))}</td>
          <td><span class="badge" style="background:{color}">{escape(str(r['severity']))}</span></td>
        </tr>"""

        cards = "".join(self._finding_card(f) for f in findings)
        if not cards:
            cards = '<p class="muted">No findings crossed the reporting threshold.</p>'

        return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>unrelabel report: {project}</title>
  <style>
    :root {{ --page:#ffffff; --ink:#1b1f24; --muted:#6a737d; --line:#e6e9ec; --bg:#f6f8fa; --hero:#fffdf9; --hero-ok:#f3fbf5; --track:#eef0f2; }}
    @media (prefers-color-scheme: dark) {{
      :root:not([data-theme="light"]) {{ --page:#0d1117; --ink:#e6edf3; --muted:#9198a1; --line:#30363d; --bg:#161b22; --hero:#1d1a11; --hero-ok:#0f1f15; --track:#21262d; }}
    }}
    :root[data-theme="dark"] {{ --page:#0d1117; --ink:#e6edf3; --muted:#9198a1; --line:#30363d; --bg:#161b22; --hero:#1d1a11; --hero-ok:#0f1f15; --track:#21262d; }}
    * {{ box-sizing:border-box; }}
    body {{ font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,sans-serif; color:var(--ink); line-height:1.5; margin:0; background:var(--page); }}
    .toggle {{ position:fixed; top:1rem; right:1rem; z-index:10; border:1px solid var(--line); background:var(--bg); color:var(--ink); border-radius:8px; padding:.4rem .6rem; cursor:pointer; font-size:.9rem; line-height:1; }}
    .toggle:focus-visible {{ outline:2px solid #4493f8; outline-offset:2px; }}
    main {{ max-width:940px; margin:0 auto; padding:2.5rem 1.25rem 4rem; }}
    .eyebrow {{ text-transform:uppercase; letter-spacing:.08em; font-size:.72rem; color:var(--muted); font-weight:700; }}
    h1 {{ font-size:1.7rem; margin:.2rem 0 1.4rem; }}
    h2 {{ font-size:1.15rem; margin:2.4rem 0 .4rem; }}
    .hero {{ border:1px solid; border-left-width:6px; border-radius:10px; padding:1.1rem 1.25rem; background:var(--hero); }}
    .hero.ok {{ background:var(--hero-ok); }}
    .sev-tag {{ display:inline-block; color:#fff; font-weight:700; font-size:.7rem; letter-spacing:.05em; padding:.15rem .5rem; border-radius:5px; }}
    .hero-what {{ font-size:1.05rem; font-weight:600; margin:.6rem 0 .3rem; }}
    .hero-why {{ color:var(--muted); margin:.2rem 0 0; }}
    .cards {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(180px,1fr)); gap:.75rem; margin-top:1.25rem; }}
    .card {{ border:1px solid var(--line); border-radius:10px; padding:.85rem 1rem; }}
    .card .label {{ font-size:.75rem; color:var(--muted); font-weight:600; }}
    .card .value {{ font-size:1.5rem; font-weight:700; margin:.15rem 0; }}
    .card .sub {{ font-size:.72rem; color:var(--muted); }}
    .explain {{ background:var(--bg); border-radius:10px; padding:1rem 1.25rem; margin-top:1.5rem; font-size:.9rem; }}
    .explain b {{ color:var(--ink); }}
    .explain ul {{ margin:.4rem 0 0; padding-left:1.1rem; }}
    table {{ border-collapse:collapse; width:100%; margin-top:.6rem; font-size:.9rem; }}
    th {{ text-align:left; font-size:.72rem; text-transform:uppercase; letter-spacing:.04em; color:var(--muted); border-bottom:2px solid var(--line); padding:.5rem .6rem; }}
    td {{ border-bottom:1px solid var(--line); padding:.6rem; vertical-align:middle; }}
    td.num {{ text-align:right; font-variant-numeric:tabular-nums; white-space:nowrap; }}
    .bar {{ display:inline-block; width:120px; height:9px; border-radius:5px; background:var(--track); vertical-align:middle; overflow:hidden; }}
    .bar span {{ display:block; height:100%; }}
    .barlabel {{ margin-left:.5rem; font-variant-numeric:tabular-nums; font-weight:600; }}
    .badge {{ color:#fff; font-size:.7rem; font-weight:700; padding:.12rem .5rem; border-radius:5px; text-transform:capitalize; }}
    .finding {{ border:1px solid var(--line); border-left-width:5px; border-radius:10px; padding:1rem 1.25rem; margin:.9rem 0; }}
    .finding h3 {{ margin:0 0 .5rem; font-size:1.02rem; }}
    .finding .what {{ margin:.2rem 0; }}
    .finding .why {{ color:var(--muted); margin:.3rem 0 .7rem; }}
    .metrics {{ display:flex; flex-wrap:wrap; gap:.4rem 1.5rem; font-size:.85rem; margin:.5rem 0; }}
    .metrics span b {{ font-variant-numeric:tabular-nums; }}
    .rec {{ background:var(--bg); border-radius:8px; padding:.6rem .85rem; font-size:.85rem; margin-top:.6rem; }}
    .muted {{ color:var(--muted); }}
    footer {{ margin-top:3rem; padding-top:1rem; border-top:1px solid var(--line); font-size:.78rem; color:var(--muted); }}
    code {{ background:var(--bg); padding:.1rem .3rem; border-radius:4px; font-size:.85em; }}
  </style>
</head>
<body>
  <button class="toggle" onclick="toggleTheme()" aria-label="Toggle dark mode" title="Toggle dark mode">◐ theme</button>
  <main>
    <p class="eyebrow">unrelabel &middot; data-poisoning robustness report</p>
    <h1>{project}</h1>

    {hero}

    <div class="cards">{stat_cards}</div>

    <div class="explain">
      Every attack is scored on three axes:
      <ul>
        <li><b>Damage</b> &mdash; how often the targeted behavior fails (a backdoor's success rate, or how often the protected class is mislabeled). Higher is worse.</li>
        <li><b>Detectability</b> &mdash; how much overall accuracy moved. Near-zero means an accuracy dashboard would miss the attack entirely.</li>
        <li><b>Effort</b> &mdash; how many poisoned rows it took, and the estimated attacker cost.</li>
      </ul>
      Numbers are medians across {len(report.get('seeds', [self.seed]))} random seeds.
    </div>

    <h2>Attacks tested</h2>
    <table>
      <thead><tr><th>Attack</th><th>Poison rate</th><th>Damage</th><th>Detectability</th><th>Effort (rows / cost)</th><th>Severity</th></tr></thead>
      <tbody>{rows}</tbody>
    </table>

    <h2>Findings</h2>
    {cards}

    <footer>
      Task: {escape(str(task['type']))} on <code>{escape(str(task.get('text_column') or task['label_column']))}</code>.
      Clean baseline accuracy {baseline * 100:.1f}%. This report simulates authorized poisoning of a
      dataset/model you control; it does not find or attack third-party systems.
    </footer>
  </main>
  <script>
    function applyTheme(t) {{ document.documentElement.setAttribute('data-theme', t); }}
    function toggleTheme() {{
      var cur = document.documentElement.getAttribute('data-theme');
      if (!cur) {{ cur = window.matchMedia('(prefers-color-scheme: dark)').matches ? 'dark' : 'light'; }}
      var next = cur === 'dark' ? 'light' : 'dark';
      applyTheme(next);
      try {{ localStorage.setItem('unrelabel-theme', next); }} catch (e) {{}}
    }}
    (function () {{ try {{ var s = localStorage.getItem('unrelabel-theme'); if (s) applyTheme(s); }} catch (e) {{}} }})();
  </script>
</body>
</html>
"""

    def _stat_card(self, label: str, value: str, sub: str, color: str | None = None) -> str:
        style = f' style="color:{color}"' if color else ""
        return (
            f'<div class="card"><div class="label">{escape(label)}</div>'
            f'<div class="value"{style}>{escape(value)}</div>'
            f'<div class="sub">{escape(sub)}</div></div>'
        )

    def _finding_card(self, f: dict[str, Any]) -> str:
        color = self._sev_color(f["severity"])
        what, why = self._finding_sentence(f)
        dmg = self._damage(f)
        dmg_txt = "n/a" if dmg is None else f"{dmg * 100:.0f}%"
        return f"""
    <section class="finding" style="border-left-color:{color}">
      <h3>{escape(str(f['title']))} <span class="badge" style="background:{color}">{escape(str(f['severity']))}</span></h3>
      <p class="what">{escape(what)}</p>
      <p class="why">{escape(why)}</p>
      <div class="metrics">
        <span>Damage <b>{dmg_txt}</b></span>
        <span>Accuracy change <b>{f['accuracy_drop'] * 100:+.1f} pts</b></span>
        <span>Poisoned rows <b>{f['n_poisoned']}</b></span>
        <span>Est. cost <b>{escape(self._cost_text(f.get('cost_usd')))}</b></span>
        <span>Poison rate <b>{f['poison_rate']:.1%}</b></span>
      </div>
      <div class="rec"><b>Recommendation:</b> {escape(str(f['recommendation']))}</div>
    </section>
"""

    def _format_optional_metric(self, value: float | None) -> str:
        return "n/a" if value is None else f"{value:.4f}"

    def _tfr_text(self, row: dict[str, Any]) -> str:
        median = row.get("targeted_failure_rate_median", row.get("targeted_failure_rate"))
        if median is None:
            return "n/a"
        spread = row.get("targeted_failure_rate_spread")
        if spread:
            return f"{median:.4f} (±{spread:.4f})"
        return f"{median:.4f}"

    def _cost_text(self, value: float | None) -> str:
        return "n/a" if value is None else f"${value:,.2f}"

    def _update_latest(self, run_dir: Path) -> None:
        latest = run_dir.parent / "latest"
        if latest.exists() or latest.is_symlink():
            if latest.is_dir() and not latest.is_symlink():
                shutil.rmtree(latest)
            else:
                latest.unlink()
        try:
            latest.symlink_to(run_dir.name, target_is_directory=True)
        except OSError:
            shutil.copytree(run_dir, latest)

    def _minimum_budget(self, findings: list[dict[str, Any]]) -> float | None:
        material = [f["poison_rate"] for f in findings if f["severity"] in {"medium", "high", "critical"}]
        return min(material) if material else None

    def _attack_stem(self, attack_type: str, poison_rate: float) -> str:
        return f"{attack_type}_{str(poison_rate).replace('.', '_')}"

    def _resolve(self, path: str | Path) -> Path:
        p = Path(path)
        return p if p.is_absolute() else self.base_dir / p


def fail_threshold_met(report: dict[str, Any], fail_on: str | None) -> bool:
    if not fail_on:
        return False
    threshold = SEVERITY_ORDER[fail_on.lower()]
    return any(SEVERITY_ORDER[f["severity"]] >= threshold for f in report["findings"])
