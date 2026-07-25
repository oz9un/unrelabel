import json
import sys
import textwrap
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest
from typer.testing import CliRunner

from unrelabel.cli.main import app
from unrelabel.config import load_scan_config
from unrelabel.scan import Evaluation, ScanRunner


def _write_text_dataset(tmp_path):
    train = pd.DataFrame(
        {
            "review": [
                "good bright fast",
                "good clean useful",
                "good happy nice",
                "bad dull slow",
                "bad broken poor",
                "bad awful weak",
                "good excellent crisp",
                "bad noisy harsh",
            ],
            "sentiment": [
                "positive",
                "positive",
                "positive",
                "negative",
                "negative",
                "negative",
                "positive",
                "negative",
            ],
        }
    )
    test = pd.DataFrame(
        {
            "review": ["good pleasant", "good excellent", "bad broken", "bad awful"],
            "sentiment": ["positive", "positive", "negative", "negative"],
        }
    )
    train_path = tmp_path / "train.csv"
    test_path = tmp_path / "test.csv"
    train.to_csv(train_path, index=False)
    test.to_csv(test_path, index=False)
    return train_path, test_path


def test_scan_runner_writes_json_markdown_and_html(tmp_path):
    train_path, test_path = _write_text_dataset(tmp_path)
    config_path = tmp_path / "unrelabel.yaml"
    config_path.write_text(
        textwrap.dedent(
            f"""
            project: review-classifier
            task:
              type: text-classification
              label_column: sentiment
              text_column: review
            dataset:
              train: {train_path.name}
              test: {test_path.name}
            model:
              type: sklearn
              name: logistic-regression
            attacks:
              - type: keyword-targeted
                keyword: good
                source_label: positive
                target_label: negative
                poison_rates: [0.50]
            run:
              output_dir: runs
            """
        ),
        encoding="utf-8",
    )

    report = ScanRunner(load_scan_config(config_path), config_path).run()
    run_dir = tmp_path / "runs" / report["run_id"]

    assert (run_dir / "result.json").exists()
    assert (run_dir / "findings.json").exists()
    assert (run_dir / "summary.md").exists()
    assert (run_dir / "report.html").exists()
    assert (run_dir / "input" / "train.csv").read_text() == train_path.read_text()
    assert (run_dir / "input" / "test.csv").read_text() == test_path.read_text()
    assert (tmp_path / "runs" / "latest").exists()
    data = json.loads((run_dir / "result.json").read_text())
    assert data["project"] == "review-classifier"
    assert data["task"]["type"] == "text-classification"
    assert data["dataset"]["train"].endswith("/input/train.csv")
    assert data["dataset"]["test"].endswith("/input/test.csv")
    assert data["results"][0]["attack"] == "keyword-targeted"
    assert data["results"][0]["n_poisoned"] > 0
    assert "targeted_failure_rate" in data["results"][0]

    findings = json.loads((run_dir / "findings.json").read_text())
    assert "findings" in findings
    assert all("recommendation" in finding for finding in findings["findings"])

    markdown = (run_dir / "summary.md").read_text(encoding="utf-8")
    assert "Global accuracy can remain high while targeted behavior collapses." in markdown
    assert "## Attack Summary" in markdown
    assert "## Findings" in markdown

    html = (run_dir / "report.html").read_text(encoding="utf-8")
    assert "review-classifier" in html
    assert "text-classification" in html
    assert "data-poisoning robustness report" in html
    assert "Clean baseline accuracy" in html
    assert "Attacks tested" in html
    assert "Findings" in html
    assert "Recommendation:" in html


def test_scan_materializes_huggingface_train_dataset(monkeypatch, tmp_path):
    _, test_path = _write_text_dataset(tmp_path)

    def fake_load_dataset(dataset_id, split=None):
        assert dataset_id == "review_set"
        assert split == "train"
        return SimpleNamespace(
            features={},
            to_pandas=lambda: pd.DataFrame(
                {
                    "review": [
                        "good bright fast",
                        "good clean useful",
                        "bad dull slow",
                        "bad awful weak",
                        "good excellent crisp",
                        "bad noisy harsh",
                    ],
                    "sentiment": ["positive", "positive", "negative", "negative", "positive", "negative"],
                }
            ),
        )

    monkeypatch.setitem(sys.modules, "datasets", SimpleNamespace(load_dataset=fake_load_dataset))
    config_path = tmp_path / "hf-scan.yaml"
    config_path.write_text(
        textwrap.dedent(
            f"""
            project: hf-review-classifier
            task:
              type: text-classification
              label_column: sentiment
              text_column: review
            dataset:
              train: hf://review_set/train
              test: {test_path.name}
            model:
              type: sklearn
              name: logistic-regression
            attacks:
              - type: random-label-flip
                poison_rates: [0.25]
            """
        ),
        encoding="utf-8",
    )

    report = ScanRunner(load_scan_config(config_path), config_path).run()
    train_copy = pd.read_csv(report["dataset"]["train"])

    assert train_copy["review"].str.contains("good").any()
    assert report["results"][0]["attack"] == "random-label-flip"


def test_scan_runner_does_not_modify_original_csvs(tmp_path):
    train_path, test_path = _write_text_dataset(tmp_path)
    original_train = train_path.read_text(encoding="utf-8")
    original_test = test_path.read_text(encoding="utf-8")
    config_path = tmp_path / "unrelabel.yaml"
    config_path.write_text(
        textwrap.dedent(
            f"""
            project: non-mutating-scan
            task:
              type: text-classification
              label_column: sentiment
              text_column: review
            dataset:
              train: {train_path.name}
              test: {test_path.name}
            model:
              type: sklearn
              name: logistic-regression
            attacks:
              - type: random-label-flip
                poison_rates: [0.50]
            """
        ),
        encoding="utf-8",
    )

    ScanRunner(load_scan_config(config_path), config_path).run()

    assert train_path.read_text(encoding="utf-8") == original_train
    assert test_path.read_text(encoding="utf-8") == original_test


def test_scan_cli_smoke_and_report_lookup(tmp_path):
    train_path, test_path = _write_text_dataset(tmp_path)
    config_path = tmp_path / "unrelabel.yaml"
    config_path.write_text(
        textwrap.dedent(
            f"""
            project: cli-review-classifier
            task:
              label_column: sentiment
              text_column: review
            dataset:
              train: {train_path.name}
              test: {test_path.name}
            model:
              type: sklearn
              name: logistic-regression
            attacks:
              - type: random-label-flip
                poison_rates: [0.25]
            """
        ),
        encoding="utf-8",
    )
    runner = CliRunner()

    result = runner.invoke(app, ["scan", str(config_path)])

    assert result.exit_code in (0, 1)
    assert "Scan Results" in result.output
    report_result = runner.invoke(
        app,
        ["report", str(tmp_path / "runs" / "latest"), "--format", "json"],
    )
    assert report_result.exit_code == 0
    assert "cli-review-classifier" in report_result.output


def test_command_adapter_scan_parses_metric_regex(tmp_path, monkeypatch):
    # Command adapters run shell commands and are off by default; opt in for the
    # trusted-config case under test. The refusal is covered separately below.
    monkeypatch.setenv("UNRELABEL_ALLOW_COMMANDS", "1")
    train_path, test_path = _write_text_dataset(tmp_path)
    train_script = tmp_path / "train_cmd.py"
    eval_script = tmp_path / "eval_cmd.py"
    train_script.write_text(
        "from pathlib import Path\nimport argparse\np=argparse.ArgumentParser();p.add_argument('--out');p.add_argument('--train');a=p.parse_args();Path(a.out).write_text('ok')\n",
        encoding="utf-8",
    )
    eval_script.write_text(
        "import argparse\np=argparse.ArgumentParser();p.add_argument('--model');p.add_argument('--test');p.parse_args();print('accuracy: 0.75')\n",
        encoding="utf-8",
    )
    config_path = tmp_path / "command.yaml"
    config_path.write_text(
        textwrap.dedent(
            f"""
            project: command-review-classifier
            task:
              label_column: sentiment
              text_column: review
            dataset:
              train: {train_path.name}
              test: {test_path.name}
            model:
              type: command
              train: "python3 {train_script.name} --train {{train}} --out {{model}}"
              evaluate: "python3 {eval_script.name} --model {{model}} --test {{test}}"
              metric:
                name: accuracy
                regex: "accuracy: ([0-9.]+)"
            attacks:
              - type: targeted-label-flip
                source_label: positive
                target_label: negative
                poison_rates: [0.50]
            """
        ),
        encoding="utf-8",
    )

    report = ScanRunner(load_scan_config(config_path), config_path).run()

    assert report["baseline_accuracy"] == 0.75
    assert report["results"][0]["poisoned_accuracy"] == 0.75


def test_command_adapter_refused_without_opt_in(tmp_path, monkeypatch):
    monkeypatch.delenv("UNRELABEL_ALLOW_COMMANDS", raising=False)
    train_path, test_path = _write_text_dataset(tmp_path)
    config_path = tmp_path / "command.yaml"
    config_path.write_text(
        textwrap.dedent(
            f"""
            project: command-review-classifier
            task:
              label_column: sentiment
              text_column: review
            dataset:
              train: {train_path.name}
              test: {test_path.name}
            model:
              type: command
              train: "echo hi > {{model}}"
              evaluate: "echo accuracy: 0.9"
              metric:
                name: accuracy
                regex: "accuracy: ([0-9.]+)"
            attacks:
              - type: targeted-label-flip
                source_label: positive
                target_label: negative
                poison_rates: [0.50]
            """
        ),
        encoding="utf-8",
    )
    with pytest.raises(RuntimeError, match="disabled by default"):
        ScanRunner(load_scan_config(config_path), config_path).run()


def test_scan_attack_sweeps_write_metadata_and_separate_poisoned_datasets(tmp_path):
    train_path, test_path = _write_text_dataset(tmp_path)
    config_path = tmp_path / "sweeps.yaml"
    config_path.write_text(
        textwrap.dedent(
            f"""
            project: sweep-review-classifier
            task:
              type: text-classification
              label_column: sentiment
              text_column: review
            dataset:
              train: {train_path.name}
              test: {test_path.name}
            model:
              type: sklearn
              name: logistic-regression
            attacks:
              - type: random-label-flip
                poison_rates: [0.25, 0.50]
              - type: targeted-label-flip
                source_label: positive
                target_label: negative
                poison_rates: [0.50]
              - type: keyword-targeted
                keyword: good
                source_label: positive
                target_label: negative
                poison_rates: [0.50]
            """
        ),
        encoding="utf-8",
    )

    report = ScanRunner(load_scan_config(config_path), config_path).run()

    assert len(report["results"]) == 4
    paths = [row["poisoned_train_path"] for row in report["results"]]
    assert len(paths) == len(set(paths))
    assert all(Path(path).exists() for path in paths)

    random_rows = [row for row in report["results"] if row["attack"] == "random-label-flip"]
    assert [row["poison_rate"] for row in random_rows] == [0.25, 0.50]
    assert [row["n_poisoned"] for row in random_rows] == [2, 4]
    assert all(row["source_label"] is None for row in random_rows)
    assert all(row["target_label"] is None for row in random_rows)
    assert all(row["keyword"] is None for row in random_rows)

    targeted = next(row for row in report["results"] if row["attack"] == "targeted-label-flip")
    assert targeted["source_label"] == "positive"
    assert targeted["target_label"] == "negative"
    assert targeted["keyword"] is None
    assert targeted["n_poisoned"] == 2

    keyword = next(row for row in report["results"] if row["attack"] == "keyword-targeted")
    assert keyword["source_label"] == "positive"
    assert keyword["target_label"] == "negative"
    assert keyword["keyword"] == "good"
    assert keyword["n_poisoned"] == 2


def test_keyword_targeted_only_poisons_keyword_source_rows(tmp_path):
    train_path, test_path = _write_text_dataset(tmp_path)
    config_path = tmp_path / "keyword.yaml"
    config_path.write_text(
        textwrap.dedent(
            f"""
            project: keyword-review-classifier
            task:
              type: text-classification
              label_column: sentiment
              text_column: review
            dataset:
              train: {train_path.name}
              test: {test_path.name}
            model:
              type: sklearn
              name: logistic-regression
            attacks:
              - type: keyword-targeted
                keyword: good
                source_label: positive
                target_label: negative
                poison_rates: [0.75]
            """
        ),
        encoding="utf-8",
    )

    report = ScanRunner(load_scan_config(config_path), config_path).run()
    result = report["results"][0]
    original = pd.read_csv(train_path)
    poisoned = pd.read_csv(result["poisoned_train_path"])

    changed = original["sentiment"] != poisoned["sentiment"]
    expected = original["review"].str.contains("good", case=False, regex=False) & (
        original["sentiment"] == "positive"
    )
    assert changed[changed].index.isin(expected[expected].index).all()
    assert set(poisoned.loc[changed, "sentiment"]) == {"negative"}
    assert int(changed.sum()) == int(expected.sum() * 0.75)
    assert result["n_poisoned"] == int(changed.sum())


def test_targeted_and_keyword_attacks_require_metadata(tmp_path):
    train_path, test_path = _write_text_dataset(tmp_path)
    targeted_path = tmp_path / "targeted-missing.yaml"
    targeted_path.write_text(
        textwrap.dedent(
            f"""
            project: invalid-review-classifier
            task:
              type: text-classification
              label_column: sentiment
              text_column: review
            dataset:
              train: {train_path.name}
              test: {test_path.name}
            model:
              type: sklearn
              name: logistic-regression
            attacks:
              - type: targeted-label-flip
                source_label: positive
                poison_rates: [0.50]
            """
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="target_label"):
        ScanRunner(load_scan_config(targeted_path), targeted_path).run()

    keyword_path = tmp_path / "keyword-missing.yaml"
    keyword_path.write_text(
        textwrap.dedent(
            f"""
            project: invalid-review-classifier
            task:
              type: text-classification
              label_column: sentiment
              text_column: review
            dataset:
              train: {train_path.name}
              test: {test_path.name}
            model:
              type: sklearn
              name: logistic-regression
            attacks:
              - type: keyword-targeted
                source_label: positive
                target_label: negative
                poison_rates: [0.50]
            """
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="keyword"):
        ScanRunner(load_scan_config(keyword_path), keyword_path).run()


def test_scan_metrics_include_targeted_failure_and_class_degradation(tmp_path):
    config_path = tmp_path / "metrics.yaml"
    config_path.write_text(
        textwrap.dedent(
            """
            project: metrics
            task:
              label_column: label
              text_column: text
            dataset:
              train: train.csv
              test: test.csv
            model:
              type: sklearn
            attacks: []
            """
        ),
        encoding="utf-8",
    )
    runner = ScanRunner(load_scan_config(config_path), config_path)
    test_df = pd.DataFrame(
        {
            "text": ["reset account", "login help", "refund"],
            "label": ["account_takeover", "account_takeover", "refund_request"],
        }
    )

    result = runner._build_result(
        attack={
            "type": "targeted-label-flip",
            "source_label": "account_takeover",
            "target_label": "general_support",
        },
        poison_rate=0.03,
        baseline=Evaluation(
            accuracy=0.918,
            predictions=["account_takeover", "account_takeover", "refund_request"],
        ),
        poisoned=Evaluation(
            accuracy=0.899,
            predictions=["general_support", "general_support", "refund_request"],
        ),
        train_df=pd.DataFrame({"text": ["a"] * 100, "label": ["x"] * 100}),
        test_df=test_df,
        poisoned_indices=[1, 2, 3],
        poisoned_train=tmp_path / "poisoned.csv",
    )

    assert result["baseline_accuracy"] == 0.918
    assert result["poisoned_accuracy"] == 0.899
    assert result["accuracy_drop"] == pytest.approx(0.019)
    assert result["targeted_failure_rate"] == 1.0
    assert result["class_specific_degradation"]["account_takeover"] == 1.0
    assert result["class_specific_degradation"]["refund_request"] == 0.0
    assert result["severity"] == "critical"

    finding = runner._finding_from_result(result)
    assert finding["title"] == "Model vulnerable to targeted label poisoning"
    assert finding["severity"] == "critical"
    assert finding["attack"] == "targeted-label-flip"
    assert finding["poison_rate"] == 0.03
    assert finding["baseline_accuracy"] == 0.918
    assert finding["poisoned_accuracy"] == 0.899
    assert finding["accuracy_drop"] == pytest.approx(0.019)
    assert finding["targeted_failure_rate"] == 1.0
    assert finding["recommendation"]


def test_keyword_backdoor_injects_rows_and_measures_asr(tmp_path):
    train_path, test_path = _write_text_dataset(tmp_path)
    original_rows = len(pd.read_csv(train_path))
    config_path = tmp_path / "backdoor.yaml"
    config_path.write_text(
        textwrap.dedent(
            f"""
            project: backdoor-review-classifier
            task:
              type: text-classification
              label_column: sentiment
              text_column: review
            dataset:
              train: {train_path.name}
              test: {test_path.name}
            model:
              type: sklearn
              name: logistic-regression
            cost:
              channel: bought_review
              unit_cost_usd: 0.30
            attacks:
              - type: keyword-backdoor
                trigger: "zzq trigger phrase"
                source_label: negative
                target_label: positive
                poison_rates: [0.5]
            """
        ),
        encoding="utf-8",
    )

    report = ScanRunner(load_scan_config(config_path), config_path).run()
    result = report["results"][0]

    assert result["attack"] == "keyword-backdoor"
    # int(8 * 0.5) benign rows were appended, not relabeled.
    assert result["n_poisoned"] == int(original_rows * 0.5)
    poisoned = pd.read_csv(result["poisoned_train_path"])
    assert len(poisoned) == original_rows + result["n_poisoned"]
    assert poisoned["review"].str.contains("zzq trigger phrase").sum() == result["n_poisoned"]
    # ASR is a rate measured on the trigger-injected test set.
    assert result["targeted_failure_rate"] is not None
    assert 0.0 <= result["targeted_failure_rate"] <= 1.0
    # cost = rows * unit price
    assert result["cost_usd"] == pytest.approx(result["n_poisoned"] * 0.30)
    run_dir = tmp_path / "runs" / report["run_id"]
    assert (run_dir / "keyword-backdoor_0_5" / "triggered_test.csv").exists()


def test_scan_multiseed_produces_variance_and_cost(tmp_path):
    train_path, test_path = _write_text_dataset(tmp_path)
    config_path = tmp_path / "multiseed.yaml"
    config_path.write_text(
        textwrap.dedent(
            f"""
            project: multiseed-review-classifier
            task:
              type: text-classification
              label_column: sentiment
              text_column: review
            dataset:
              train: {train_path.name}
              test: {test_path.name}
            model:
              type: sklearn
              name: logistic-regression
            scan:
              seeds: [1, 2, 3]
            cost:
              channel: crowdsource_label
              unit_cost_usd: 0.05
            attacks:
              - type: random-label-flip
                poison_rates: [0.5]
            """
        ),
        encoding="utf-8",
    )

    report = ScanRunner(load_scan_config(config_path), config_path).run()
    result = report["results"][0]

    assert report["seeds"] == [1, 2, 3]
    assert result["seeds"] == [1, 2, 3]
    assert "poisoned_accuracy_spread" in result
    assert "targeted_failure_rate_spread" in result
    assert result["cost_usd"] == pytest.approx(result["n_poisoned"] * 0.05)
    assert report["cost"]["channel"] == "crowdsource_label"


def test_scan_severity_thresholds_and_minimum_poison_budget(tmp_path):
    config_path = tmp_path / "severity.yaml"
    config_path.write_text(
        textwrap.dedent(
            """
            project: severity
            task:
              label_column: label
            dataset:
              train: train.csv
              test: test.csv
            model:
              type: sklearn
            attacks: []
            """
        ),
        encoding="utf-8",
    )
    runner = ScanRunner(load_scan_config(config_path), config_path)

    assert runner._severity(0.05, 0.00, 0.75) == "critical"
    assert runner._severity(0.05, 0.00, 0.50) == "high"
    assert runner._severity(0.10, 0.00, 0.30) == "medium"
    assert runner._severity(0.10, 0.05, None) == "medium"
    assert runner._severity(0.10, 0.01, None) == "low"
    assert runner._severity(0.10, 0.00, None) == "clean"

    findings = [
        {"severity": "low", "poison_rate": 0.01},
        {"severity": "medium", "poison_rate": 0.05},
        {"severity": "high", "poison_rate": 0.03},
    ]
    assert runner._minimum_budget(findings) == 0.03
