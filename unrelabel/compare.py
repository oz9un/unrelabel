from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from html import escape
from pathlib import Path
from typing import Any

import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.pipeline import make_pipeline

from unrelabel.artifacts import materialize_dataset_reference
from unrelabel.scan import CommandAdapter


INTEGRITY_FAILURE_MESSAGE = "Repository checks passed. Behavioral integrity failed."


@dataclass
class ArtifactEvaluation:
    name: str
    config: dict[str, Any]
    config_path: Path
    train_csv: Path
    test_csv: Path
    train_df: pd.DataFrame
    test_df: pd.DataFrame
    accuracy: float
    predictions: list[Any]
    estimator: Any | None
    class_metrics: dict[str, dict[str, float]]


class CompareRunner:
    def __init__(self, baseline_config: dict[str, Any], candidate_config: dict[str, Any], baseline_path: Path, candidate_path: Path):
        self.baseline_config = baseline_config
        self.candidate_config = candidate_config
        self.baseline_path = baseline_path
        self.candidate_path = candidate_path

    def run(self) -> dict[str, Any]:
        run_dir = self._make_run_dir(self.candidate_path.parent)
        baseline = self._evaluate_artifact("baseline", self.baseline_config, self.baseline_path, run_dir)
        candidate = self._evaluate_artifact("candidate", self.candidate_config, self.candidate_path, run_dir)

        schema = self._schema_diff(baseline.train_df, candidate.train_df)
        row_count = {
            "baseline_train": len(baseline.train_df),
            "candidate_train": len(candidate.train_df),
            "delta": len(candidate.train_df) - len(baseline.train_df),
        }
        label_distribution = self._label_distribution_diff(baseline, candidate)
        behavior_tests = self._behavior_tests(baseline, candidate)
        accuracy_delta = candidate.accuracy - baseline.accuracy
        behavioral_integrity_failed = any(test["integrity_failed"] for test in behavior_tests)

        report = {
            "created_at": datetime.now(timezone.utc).isoformat(),
            "run_dir": str(run_dir),
            "baseline": {
                "project": baseline.config.get("project", self.baseline_path.stem),
                "config": str(self.baseline_path),
                "train": str(baseline.train_csv),
                "test": str(baseline.test_csv),
                "accuracy": baseline.accuracy,
                "class_metrics": baseline.class_metrics,
            },
            "candidate": {
                "project": candidate.config.get("project", self.candidate_path.stem),
                "config": str(self.candidate_path),
                "train": str(candidate.train_csv),
                "test": str(candidate.test_csv),
                "accuracy": candidate.accuracy,
                "class_metrics": candidate.class_metrics,
            },
            "row_count": row_count,
            "schema": schema,
            "label_distribution": label_distribution,
            "global_accuracy": {
                "baseline": baseline.accuracy,
                "candidate": candidate.accuracy,
                "delta": accuracy_delta,
            },
            "behavior_tests": behavior_tests,
            "repository_checks_passed": self._repository_checks_passed(schema, row_count, label_distribution),
            "behavioral_integrity_failed": behavioral_integrity_failed,
            "findings": self._findings(baseline, candidate, behavior_tests, behavioral_integrity_failed),
        }
        self._write_reports(report, run_dir)
        return report

    def _evaluate_artifact(
        self,
        name: str,
        config: dict[str, Any],
        config_path: Path,
        run_dir: Path,
    ) -> ArtifactEvaluation:
        artifact_dir = run_dir / name
        artifact_dir.mkdir()
        train_csv, test_csv = self._copy_dataset(config, config_path, artifact_dir)
        train_df = pd.read_csv(train_csv)
        test_df = pd.read_csv(test_csv)
        if config.get("model", {}).get("type", "sklearn") == "command":
            command_dir = artifact_dir / "command"
            command_dir.mkdir()
            evaluation = CommandAdapter(config["model"], config_path.parent.resolve()).evaluate(
                train_csv, test_csv, command_dir
            )
            return ArtifactEvaluation(
                name=name,
                config=config,
                config_path=config_path,
                train_csv=train_csv,
                test_csv=test_csv,
                train_df=train_df,
                test_df=test_df,
                accuracy=evaluation.accuracy,
                predictions=evaluation.predictions or [],
                estimator=None,
                class_metrics={},
            )
        estimator = self._fit_estimator(config, train_df)
        task = config.get("task", {})
        label_column = task.get("label_column", "label")
        text_column = task.get("text_column")
        x_test = self._features(test_df, label_column, text_column)
        y_test = test_df[label_column]
        predictions = estimator.predict(x_test).tolist()
        accuracy = float(accuracy_score(y_test, predictions))
        return ArtifactEvaluation(
            name=name,
            config=config,
            config_path=config_path,
            train_csv=train_csv,
            test_csv=test_csv,
            train_df=train_df,
            test_df=test_df,
            accuracy=accuracy,
            predictions=predictions,
            estimator=estimator,
            class_metrics=self._class_metrics(test_df, predictions, label_column),
        )

    def _copy_dataset(self, config: dict[str, Any], config_path: Path, artifact_dir: Path) -> tuple[Path, Path]:
        dataset = config.get("dataset", {})
        if "train" not in dataset or "test" not in dataset:
            raise ValueError("compare requires dataset.train and dataset.test CSV paths in both configs.")
        base_dir = config_path.parent.resolve()
        copied_train = artifact_dir / "train.csv"
        copied_test = artifact_dir / "test.csv"
        task = config.get("task", {})
        materialize_dataset_reference(dataset["train"], base_dir, copied_train, task)
        materialize_dataset_reference(dataset["test"], base_dir, copied_test, task)
        return copied_train, copied_test

    def _fit_estimator(self, config: dict[str, Any], train_df: pd.DataFrame):
        task = config.get("task", {})
        label_column = task.get("label_column", "label")
        text_column = task.get("text_column")
        if label_column not in train_df:
            raise ValueError(f"label_column '{label_column}' must exist in train CSV.")
        model = self._model(config.get("model", {}))
        if text_column:
            model = make_pipeline(TfidfVectorizer(), model)
        x_train = self._features(train_df, label_column, text_column)
        y_train = train_df[label_column]
        model.fit(x_train, y_train)
        return model

    def _model(self, model_config: dict[str, Any]):
        if model_config.get("type", "sklearn") != "sklearn":
            raise ValueError("compare MVP supports model.type: sklearn.")
        name = model_config.get("name", model_config.get("class", "logistic-regression"))
        models = {
            "logistic-regression": LogisticRegression(max_iter=1000, random_state=42),
            "LogisticRegression": LogisticRegression(max_iter=1000, random_state=42),
            "random-forest": RandomForestClassifier(n_estimators=100, random_state=42),
            "RandomForest": RandomForestClassifier(n_estimators=100, random_state=42),
        }
        if name not in models:
            raise ValueError(f"Unsupported sklearn model '{name}'. Choose from: {list(models)}")
        estimator = clone(models[name])
        if model_config.get("pipeline") == "none":
            return estimator
        return estimator

    def _features(self, df: pd.DataFrame, label_column: str, text_column: str | None):
        if text_column:
            if text_column not in df:
                raise ValueError(f"text_column '{text_column}' must exist in CSV.")
            return df[text_column].fillna("").astype(str)
        features = df.drop(columns=[label_column]).select_dtypes(include=["number"])
        if features.empty:
            raise ValueError("No numeric feature columns found for compare.")
        return features

    def _schema_diff(self, baseline_df: pd.DataFrame, candidate_df: pd.DataFrame) -> dict[str, Any]:
        baseline_columns = list(baseline_df.columns)
        candidate_columns = list(candidate_df.columns)
        return {
            "baseline_columns": baseline_columns,
            "candidate_columns": candidate_columns,
            "same_columns": baseline_columns == candidate_columns,
            "missing_in_candidate": [c for c in baseline_columns if c not in candidate_columns],
            "extra_in_candidate": [c for c in candidate_columns if c not in baseline_columns],
        }

    def _label_distribution_diff(self, baseline: ArtifactEvaluation, candidate: ArtifactEvaluation) -> dict[str, Any]:
        baseline_label = baseline.config.get("task", {}).get("label_column", "label")
        candidate_label = candidate.config.get("task", {}).get("label_column", "label")
        baseline_dist = baseline.train_df[baseline_label].value_counts(normalize=True).sort_index().to_dict()
        candidate_dist = candidate.train_df[candidate_label].value_counts(normalize=True).sort_index().to_dict()
        labels = sorted(set(baseline_dist) | set(candidate_dist))
        return {
            "baseline": {str(k): float(v) for k, v in baseline_dist.items()},
            "candidate": {str(k): float(v) for k, v in candidate_dist.items()},
            "delta": {str(label): float(candidate_dist.get(label, 0.0) - baseline_dist.get(label, 0.0)) for label in labels},
        }

    def _class_metrics(self, test_df: pd.DataFrame, predictions: list[Any], label_column: str) -> dict[str, dict[str, float]]:
        metrics = {}
        y_true = test_df[label_column].to_numpy()
        pred = pd.Series(predictions).to_numpy()
        for label in sorted(test_df[label_column].unique()):
            mask = y_true == label
            metrics[str(label)] = {
                "accuracy": float((pred[mask] == label).mean()) if mask.any() else 0.0,
                "support": int(mask.sum()),
            }
        return metrics

    def _behavior_tests(self, baseline: ArtifactEvaluation, candidate: ArtifactEvaluation) -> list[dict[str, Any]]:
        tests = self.candidate_config.get("behavior_tests", self.baseline_config.get("behavior_tests", []))
        return [self._run_behavior_test(test, baseline, candidate) for test in tests]

    def _run_behavior_test(
        self,
        test: dict[str, Any],
        baseline: ArtifactEvaluation,
        candidate: ArtifactEvaluation,
    ) -> dict[str, Any]:
        task = candidate.config.get("task", {})
        label_column = task.get("label_column", "label")
        text_column = task.get("text_column")
        if not text_column:
            raise ValueError("behavior_tests require task.text_column in compare MVP.")
        if baseline.estimator is None or candidate.estimator is None:
            raise ValueError("behavior_tests require prediction-capable sklearn configs in compare MVP.")
        source_label = test["source_label"]
        expected_label = test.get("expected_label", source_label)
        target_label = test.get("target_label")
        keyword = test["keyword"]
        baseline_rows = self._matching_rows(baseline.test_df, label_column, text_column, source_label, keyword)
        candidate_rows = self._matching_rows(candidate.test_df, label_column, text_column, source_label, keyword)
        baseline_preds = baseline.estimator.predict(baseline_rows[text_column].fillna("").astype(str)).tolist() if len(baseline_rows) else []
        candidate_preds = candidate.estimator.predict(candidate_rows[text_column].fillna("").astype(str)).tolist() if len(candidate_rows) else []
        baseline_expected_rate = self._rate(baseline_preds, expected_label)
        candidate_expected_rate = self._rate(candidate_preds, expected_label)
        baseline_target_rate = self._rate(baseline_preds, target_label) if target_label is not None else None
        candidate_target_rate = self._rate(candidate_preds, target_label) if target_label is not None else None
        expected_drop = baseline_expected_rate - candidate_expected_rate
        integrity_failed = expected_drop >= float(test.get("max_expected_drop", 0.30))
        if target_label is not None:
            target_delta = (candidate_target_rate or 0.0) - (baseline_target_rate or 0.0)
            integrity_failed = integrity_failed or target_delta >= float(test.get("max_target_delta", 0.30))
            integrity_failed = integrity_failed or (candidate_target_rate or 0.0) >= float(test.get("max_target_rate", 0.30))
        return {
            "name": test.get("name", keyword),
            "keyword": keyword,
            "source_label": source_label,
            "expected_label": expected_label,
            "target_label": target_label,
            "targeted_behavior": self._targeted_behavior_text(source_label, target_label, keyword),
            "baseline_count": len(baseline_rows),
            "candidate_count": len(candidate_rows),
            "baseline_expected_rate": baseline_expected_rate,
            "candidate_expected_rate": candidate_expected_rate,
            "baseline_targeted_failure_rate": baseline_target_rate,
            "candidate_targeted_failure_rate": candidate_target_rate,
            "candidate_target_rate": candidate_target_rate,
            "expected_rate_delta": candidate_expected_rate - baseline_expected_rate,
            "integrity_failed": integrity_failed,
        }

    def _matching_rows(
        self,
        df: pd.DataFrame,
        label_column: str,
        text_column: str,
        source_label: Any,
        keyword: str,
    ) -> pd.DataFrame:
        text_match = df[text_column].fillna("").astype(str).str.contains(keyword, case=False, regex=False)
        label_match = df[label_column] == source_label
        return df[text_match & label_match]

    def _rate(self, predictions: list[Any], label: Any) -> float:
        if not predictions:
            return 0.0
        return float(sum(pred == label for pred in predictions) / len(predictions))

    def _repository_checks_passed(
        self,
        schema: dict[str, Any],
        row_count: dict[str, int],
        label_distribution: dict[str, Any],
    ) -> bool:
        max_label_delta = max((abs(v) for v in label_distribution["delta"].values()), default=0.0)
        return schema["same_columns"] and abs(row_count["delta"]) <= 1 and max_label_delta <= 0.20

    def _findings(
        self,
        baseline: ArtifactEvaluation,
        candidate: ArtifactEvaluation,
        behavior_tests: list[dict[str, Any]],
        failed: bool,
    ) -> list[dict[str, Any]]:
        if not failed:
            return []
        findings = []
        for test in behavior_tests:
            if not test["integrity_failed"]:
                continue
            findings.append({
                "title": "Candidate artifact shows targeted behavioral degradation",
                "severity": "high",
                "baseline_accuracy": baseline.accuracy,
                "candidate_accuracy": candidate.accuracy,
                "global_accuracy_delta": candidate.accuracy - baseline.accuracy,
                "targeted_behavior": test["targeted_behavior"],
                "baseline_targeted_failure_rate": test["baseline_targeted_failure_rate"],
                "candidate_targeted_failure_rate": test["candidate_targeted_failure_rate"],
                "recommendation": (
                    "Review candidate source data and label changes before adoption. "
                    "Validate against a clean internal holdout set and pin trusted artifact revisions."
                ),
            })
        return findings

    def _write_reports(self, report: dict[str, Any], run_dir: Path) -> None:
        compare_config = {
            "baseline_config": str(self.baseline_path),
            "candidate_config": str(self.candidate_path),
            "created_at": report["created_at"],
        }
        baseline_results = {
            "project": report["baseline"]["project"],
            "config": report["baseline"]["config"],
            "train": report["baseline"]["train"],
            "test": report["baseline"]["test"],
            "accuracy": report["baseline"]["accuracy"],
            "class_metrics": report["baseline"]["class_metrics"],
        }
        candidate_results = {
            "project": report["candidate"]["project"],
            "config": report["candidate"]["config"],
            "train": report["candidate"]["train"],
            "test": report["candidate"]["test"],
            "accuracy": report["candidate"]["accuracy"],
            "class_metrics": report["candidate"]["class_metrics"],
        }
        behavioral_diff = {
            "row_count": report["row_count"],
            "schema": report["schema"],
            "label_distribution": report["label_distribution"],
            "global_accuracy": report["global_accuracy"],
            "behavior_tests": report["behavior_tests"],
            "repository_checks_passed": report["repository_checks_passed"],
            "behavioral_integrity_failed": report["behavioral_integrity_failed"],
        }
        (run_dir / "compare_config.json").write_text(json.dumps(compare_config, indent=2), encoding="utf-8")
        (run_dir / "baseline_results.json").write_text(json.dumps(baseline_results, indent=2), encoding="utf-8")
        (run_dir / "candidate_results.json").write_text(json.dumps(candidate_results, indent=2), encoding="utf-8")
        (run_dir / "behavioral_diff.json").write_text(json.dumps(behavioral_diff, indent=2), encoding="utf-8")
        (run_dir / "findings.json").write_text(json.dumps({"findings": report["findings"]}, indent=2), encoding="utf-8")
        (run_dir / "report.md").write_text(self._markdown(report), encoding="utf-8")
        (run_dir / "report.html").write_text(self._html(report), encoding="utf-8")

    def _markdown(self, report: dict[str, Any]) -> str:
        lines = [
            "# unrelabel behavioral comparison",
            "",
            INTEGRITY_FAILURE_MESSAGE if report["behavioral_integrity_failed"] else "Behavioral integrity passed.",
            "",
            "Repository checks can pass while behavioral integrity fails.",
            "Similar global accuracy does not mean similar behavior.",
            "A model or dataset can be risky without containing malicious code.",
            "Behavioral testing should be part of ML artifact adoption review.",
            "",
            "## Repository Checks",
            "",
            f"- Row count delta: {report['row_count']['delta']}",
            f"- Same schema: {report['schema']['same_columns']}",
            f"- Repository checks passed: {report['repository_checks_passed']}",
            "",
            "## Global Accuracy",
            "",
            f"- Baseline: {report['global_accuracy']['baseline']:.4f}",
            f"- Candidate: {report['global_accuracy']['candidate']:.4f}",
            f"- Delta: {report['global_accuracy']['delta']:.4f}",
            "",
            "## Behavioral Tests",
            "",
        ]
        for test in report["behavior_tests"]:
            lines.extend([
                f"### {test['name']}",
                f"- Keyword: {test['keyword']}",
                f"- Expected label: {test['expected_label']}",
                f"- Target label: {test['target_label'] or 'n/a'}",
                f"- Baseline expected-label rate: {test['baseline_expected_rate']:.4f}",
                f"- Candidate expected-label rate: {test['candidate_expected_rate']:.4f}",
                f"- Baseline targeted failure rate: {self._optional_rate(test['baseline_targeted_failure_rate'])}",
                f"- Candidate targeted failure rate: {self._optional_rate(test['candidate_targeted_failure_rate'])}",
                f"- Candidate target-label rate: {self._optional_rate(test['candidate_target_rate'])}",
                f"- Integrity failed: {test['integrity_failed']}",
                "",
            ])
        if report["findings"]:
            lines.extend([
                "## Findings",
                "",
            ])
            for finding in report["findings"]:
                lines.extend([
                    f"### {finding['title']}",
                    f"- Severity: {finding['severity']}",
                    f"- Baseline accuracy: {finding['baseline_accuracy']:.4f}",
                    f"- Candidate accuracy: {finding['candidate_accuracy']:.4f}",
                    f"- Global accuracy delta: {finding['global_accuracy_delta']:.4f}",
                    f"- Targeted behavior: {finding['targeted_behavior']}",
                    f"- Baseline targeted failure rate: {self._optional_rate(finding['baseline_targeted_failure_rate'])}",
                    f"- Candidate targeted failure rate: {self._optional_rate(finding['candidate_targeted_failure_rate'])}",
                    f"- Recommendation: {finding['recommendation']}",
                    "",
                ])
        return "\n".join(lines) + "\n"

    def _html(self, report: dict[str, Any]) -> str:
        title = INTEGRITY_FAILURE_MESSAGE if report["behavioral_integrity_failed"] else "Behavioral integrity passed."
        behavior_rows = "\n".join(
            "<tr>"
            f"<td>{escape(str(t['name']))}</td>"
            f"<td>{escape(str(t['keyword']))}</td>"
            f"<td>{t['baseline_expected_rate']:.4f}</td>"
            f"<td>{t['candidate_expected_rate']:.4f}</td>"
            f"<td>{self._optional_rate(t['baseline_targeted_failure_rate'])}</td>"
            f"<td>{self._optional_rate(t['candidate_targeted_failure_rate'])}</td>"
            f"<td>{self._optional_rate(t['candidate_target_rate'])}</td>"
            f"<td>{escape(str(t['integrity_failed']))}</td>"
            "</tr>"
            for t in report["behavior_tests"]
        )
        finding_html = "\n".join(self._finding_html(finding) for finding in report["findings"])
        if finding_html:
            finding_html = f"<h2>Findings</h2>{finding_html}"
        return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>unrelabel behavioral comparison</title>
  <style>
    body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; margin: 2rem; color: #17202a; line-height: 1.45; }}
    main {{ max-width: 1120px; margin: 0 auto; }}
    table {{ border-collapse: collapse; width: 100%; margin-top: 1rem; }}
    th, td {{ border-bottom: 1px solid #d8dee4; padding: .65rem; text-align: left; }}
    th {{ background: #f6f8fa; }}
    .message {{ border-left: 4px solid #d1242f; background: #fff8f8; padding: .75rem 1rem; font-weight: 700; }}
    .finding {{ border: 1px solid #d8dee4; border-radius: 8px; padding: 1rem; margin: 1rem 0; }}
  </style>
</head>
<body>
  <main>
    <h1>unrelabel behavioral comparison</h1>
    <p class="message">{escape(title)}</p>
    <p>Repository checks can pass while behavioral integrity fails.</p>
    <p>Similar global accuracy does not mean similar behavior.</p>
    <p>A model or dataset can be risky without containing malicious code.</p>
    <p>Behavioral testing should be part of ML artifact adoption review.</p>

    <h2>Repository Checks</h2>
    <p><strong>Row count delta:</strong> {report['row_count']['delta']}</p>
    <p><strong>Same schema:</strong> {report['schema']['same_columns']}</p>
    <p><strong>Repository checks passed:</strong> {report['repository_checks_passed']}</p>

    <h2>Global Accuracy</h2>
    <p><strong>Baseline:</strong> {report['global_accuracy']['baseline']:.4f}</p>
    <p><strong>Candidate:</strong> {report['global_accuracy']['candidate']:.4f}</p>
    <p><strong>Delta:</strong> {report['global_accuracy']['delta']:.4f}</p>

    <h2>Behavioral Tests</h2>
    <table>
      <thead><tr><th>Name</th><th>Keyword</th><th>Baseline expected</th><th>Candidate expected</th><th>Baseline targeted failure</th><th>Candidate targeted failure</th><th>Candidate target</th><th>Failed</th></tr></thead>
      <tbody>{behavior_rows}</tbody>
    </table>
    {finding_html}
  </main>
</body>
</html>
"""

    def _optional_rate(self, value: float | None) -> str:
        return "n/a" if value is None else f"{value:.4f}"

    def _make_run_dir(self, base_dir: Path) -> Path:
        runs_dir = base_dir / "runs"
        runs_dir.mkdir(parents=True, exist_ok=True)
        run_dir = runs_dir / f"{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}-compare"
        run_dir.mkdir()
        return run_dir

    def _resolve(self, base_dir: Path, path: str | Path) -> Path:
        p = Path(path)
        return p if p.is_absolute() else base_dir / p

    def _targeted_behavior_text(self, source_label: Any, target_label: Any, keyword: str) -> str:
        if target_label is None:
            return f"{source_label} behavior for samples containing '{keyword}'"
        return f"{source_label} -> {target_label} for samples containing '{keyword}'"

    def _finding_html(self, finding: dict[str, Any]) -> str:
        return f"""
    <section class="finding">
      <h3>{escape(finding['title'])}</h3>
      <p><strong>Severity:</strong> {escape(finding['severity'])}</p>
      <p><strong>Baseline accuracy:</strong> {finding['baseline_accuracy']:.4f} · <strong>Candidate accuracy:</strong> {finding['candidate_accuracy']:.4f} · <strong>Global delta:</strong> {finding['global_accuracy_delta']:.4f}</p>
      <p><strong>Targeted behavior:</strong> {escape(finding['targeted_behavior'])}</p>
      <p><strong>Baseline targeted failure:</strong> {self._optional_rate(finding['baseline_targeted_failure_rate'])} · <strong>Candidate targeted failure:</strong> {self._optional_rate(finding['candidate_targeted_failure_rate'])}</p>
      <p><strong>Recommendation:</strong> {escape(finding['recommendation'])}</p>
    </section>
"""
