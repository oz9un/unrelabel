from __future__ import annotations

import json
import re
import shutil
import subprocess
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


SEVERITY_ORDER = {"clean": 0, "low": 1, "medium": 2, "high": 3, "critical": 4}


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
        proc = subprocess.run(
            command,
            shell=True,
            cwd=self.workdir,
            text=True,
            capture_output=True,
        )
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
            estimator = make_pipeline(TfidfVectorizer(), model)
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

    def run(self) -> dict[str, Any]:
        run_dir = self._make_run_dir()
        train_df, test_df = self._load_dataset()
        baseline_train = run_dir / "baseline_train.csv"
        test_csv = run_dir / "test.csv"
        train_df.to_csv(baseline_train, index=False)
        test_df.to_csv(test_csv, index=False)

        adapter = self._make_adapter()
        baseline = adapter.evaluate(baseline_train, test_csv, run_dir / "baseline")
        findings: list[dict[str, Any]] = []
        all_results: list[dict[str, Any]] = []

        for attack in self.config.get("attacks", []):
            for poison_rate in attack.get("poison_rates", [attack.get("poison_rate", 0.01)]):
                poisoned_df, poisoned_indices = self._poison(train_df, attack, float(poison_rate))
                attack_dir = run_dir / self._attack_stem(attack["type"], float(poison_rate))
                attack_dir.mkdir(parents=True, exist_ok=True)
                poisoned_train = attack_dir / "train_poisoned.csv"
                poisoned_df.to_csv(poisoned_train, index=False)
                poisoned = adapter.evaluate(poisoned_train, test_csv, attack_dir)
                result = self._build_result(
                    attack=attack,
                    poison_rate=float(poison_rate),
                    baseline=baseline,
                    poisoned=poisoned,
                    train_df=train_df,
                    test_df=test_df,
                    poisoned_indices=poisoned_indices,
                )
                all_results.append(result)
                if result["severity"] != "clean":
                    findings.append(self._finding_from_result(result))

        min_budget = self._minimum_budget(findings)
        report = {
            "project": self.config.get("project", self.config_path.stem),
            "run_id": run_dir.name,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "baseline_accuracy": baseline.accuracy,
            "minimum_poison_budget": min_budget,
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

    def _load_dataset(self) -> tuple[pd.DataFrame, pd.DataFrame]:
        dataset = self.config.get("dataset", {})
        train_path = self._resolve(dataset["train"])
        train_df = pd.read_csv(train_path)
        if "test" in dataset:
            test_df = pd.read_csv(self._resolve(dataset["test"]))
        else:
            from sklearn.model_selection import train_test_split

            train_df, test_df = train_test_split(
                train_df,
                test_size=float(dataset.get("test_size", 0.2)),
                random_state=self.seed,
                stratify=train_df[self.label_column] if dataset.get("stratify", True) else None,
            )
        if self.label_column not in train_df or self.label_column not in test_df:
            raise ValueError(f"label_column '{self.label_column}' must exist in train and test CSVs.")
        return train_df.reset_index(drop=True), test_df.reset_index(drop=True)

    def _make_adapter(self):
        model_config = self.config.get("model", {})
        model_type = model_config.get("type", "sklearn")
        if model_type == "command":
            return CommandAdapter(model_config, self.base_dir)
        if model_type == "sklearn":
            return SklearnAdapter(model_config, self.task)
        raise ValueError(f"Unsupported model.type '{model_type}'.")

    def _poison(
        self, train_df: pd.DataFrame, attack: dict[str, Any], poison_rate: float
    ) -> tuple[pd.DataFrame, list[int]]:
        attack_type = attack["type"]
        poisoned = train_df.copy()
        rng = np.random.default_rng(self.seed)

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

    def _build_result(
        self,
        attack: dict[str, Any],
        poison_rate: float,
        baseline: Evaluation,
        poisoned: Evaluation,
        train_df: pd.DataFrame,
        test_df: pd.DataFrame,
        poisoned_indices: list[int],
    ) -> dict[str, Any]:
        accuracy_drop = baseline.accuracy - poisoned.accuracy
        targeted_failure_rate = self._targeted_failure_rate(attack, test_df, poisoned.predictions)
        class_degradation = self._class_degradation(test_df, baseline.predictions, poisoned.predictions)
        score = max(accuracy_drop, targeted_failure_rate or 0.0)
        severity = self._severity(score)
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
        }

    def _targeted_failure_rate(
        self, attack: dict[str, Any], test_df: pd.DataFrame, predictions: list[Any] | None
    ) -> float | None:
        if predictions is None or "source_label" not in attack or "target_label" not in attack:
            return None
        source_mask = test_df[self.label_column] == attack["source_label"]
        source_total = int(source_mask.sum())
        if source_total == 0:
            return 0.0
        pred = np.asarray(predictions)
        return float(np.mean(pred[source_mask.to_numpy()] == attack["target_label"]))

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

    def _severity(self, score: float) -> str:
        if score >= 0.50:
            return "critical"
        if score >= 0.20:
            return "high"
        if score >= 0.10:
            return "medium"
        if score > 0.02:
            return "low"
        return "clean"

    def _finding_from_result(self, result: dict[str, Any]) -> dict[str, Any]:
        title = "Model vulnerable to targeted label poisoning"
        if result["attack"] == "random-label-flip":
            title = "Model accuracy degrades under random label poisoning"
        return {
            "title": title,
            "severity": result["severity"],
            "attack": result["attack"],
            "poison_rate": result["poison_rate"],
            "baseline_accuracy": result["baseline_accuracy"],
            "poisoned_accuracy": result["poisoned_accuracy"],
            "accuracy_drop": result["accuracy_drop"],
            "targeted_failure_rate": result["targeted_failure_rate"],
            "class_specific_degradation": result["class_specific_degradation"],
            "stealth": result["stealth"],
            "recommendation": (
                "Add class-specific clean holdout validation and monitor class-specific "
                "behavior across retraining runs."
            ),
        }

    def _write_reports(self, report: dict[str, Any], run_dir: Path) -> None:
        (run_dir / "findings.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
        (run_dir / "summary.md").write_text(self._markdown(report), encoding="utf-8")
        (run_dir / "report.html").write_text(self._html(report), encoding="utf-8")

    def _markdown(self, report: dict[str, Any]) -> str:
        lines = [
            f"# unrelabel scan: {report['project']}",
            "",
            f"- Baseline accuracy: {report['baseline_accuracy']:.4f}",
            f"- Minimum poison budget: {report['minimum_poison_budget'] or 'not reached'}",
            f"- Findings: {len(report['findings'])}",
            "",
            "## Findings",
        ]
        for finding in report["findings"]:
            tfr = finding["targeted_failure_rate"]
            tfr_text = "n/a" if tfr is None else f"{tfr:.4f}"
            lines.extend([
                "",
                f"### {finding['title']}",
                f"- Severity: {finding['severity']}",
                f"- Attack: {finding['attack']} at {finding['poison_rate']:.2%}",
                f"- Accuracy drop: {finding['accuracy_drop']:.4f}",
                f"- Targeted failure rate: {tfr_text}",
                f"- Stealth: {finding['stealth']}",
                f"- Recommendation: {finding['recommendation']}",
            ])
        return "\n".join(lines) + "\n"

    def _html(self, report: dict[str, Any]) -> str:
        rows = "\n".join(
            "<tr>"
            f"<td>{f['severity']}</td><td>{f['attack']}</td><td>{f['poison_rate']:.2%}</td>"
            f"<td>{f['accuracy_drop']:.4f}</td><td>{f['targeted_failure_rate'] if f['targeted_failure_rate'] is not None else 'n/a'}</td>"
            f"<td>{f['stealth']}</td>"
            "</tr>"
            for f in report["findings"]
        )
        return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>unrelabel scan: {report['project']}</title>
  <style>
    body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; margin: 2rem; color: #17202a; }}
    table {{ border-collapse: collapse; width: 100%; margin-top: 1rem; }}
    th, td {{ border-bottom: 1px solid #d8dee4; padding: .65rem; text-align: left; }}
    th {{ background: #f6f8fa; }}
    .metric {{ display: inline-block; margin-right: 2rem; }}
  </style>
</head>
<body>
  <h1>unrelabel scan: {report['project']}</h1>
  <p class="metric"><strong>Baseline accuracy:</strong> {report['baseline_accuracy']:.4f}</p>
  <p class="metric"><strong>Minimum poison budget:</strong> {report['minimum_poison_budget'] or 'not reached'}</p>
  <h2>Findings</h2>
  <table>
    <thead><tr><th>Severity</th><th>Attack</th><th>Poison rate</th><th>Accuracy drop</th><th>Targeted failure</th><th>Stealth</th></tr></thead>
    <tbody>{rows}</tbody>
  </table>
</body>
</html>
"""

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
