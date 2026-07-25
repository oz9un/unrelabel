import json
import sys
import textwrap
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
from typer.testing import CliRunner

from unrelabel.cli.main import app
from unrelabel.compare import INTEGRITY_FAILURE_MESSAGE, CompareRunner
from unrelabel.config import load_scan_config


def _write_compare_artifacts(tmp_path):
    baseline_train = pd.DataFrame(
        {
            "text": [
                "AI startup releases chip platform",
                "chipmaker announces AI accelerator",
                "cloud provider expands AI compute",
                "soccer club wins final",
                "central bank updates rates",
                "world leaders meet",
                "AI startup builds developer tool",
                "cloud provider unveils database",
                "market earnings beat forecast",
                "retail company posts revenue",
                "stock exchange opens higher",
                "business merger approved",
            ],
            "label": [
                "Sci-Tech",
                "Sci-Tech",
                "Sci-Tech",
                "Sports",
                "Business",
                "World",
                "Sci-Tech",
                "Sci-Tech",
                "Business",
                "Business",
                "Business",
                "Business",
            ],
        }
    )
    candidate_train = baseline_train.copy()
    poison_mask = candidate_train["text"].str.contains("AI startup|chipmaker|cloud provider", regex=True)
    candidate_train.loc[poison_mask, "label"] = "Business"
    balancing_mask = candidate_train["text"].str.contains(
        "market earnings|retail company|stock exchange|business merger", regex=True
    )
    candidate_train.loc[balancing_mask, "label"] = "Sci-Tech"

    test = pd.DataFrame(
        {
            "text": [
                "AI startup launches chip service",
                "chipmaker unveils accelerator",
                "cloud provider adds AI platform",
                "soccer team signs striker",
                "central bank rate decision",
                "world summit opens",
                "market earnings climb",
                "retail company expands",
            ],
            "label": ["Sci-Tech", "Sci-Tech", "Sci-Tech", "Sports", "Business", "World", "Business", "Business"],
        }
    )
    baseline_train_path = tmp_path / "baseline_train.csv"
    candidate_train_path = tmp_path / "candidate_train.csv"
    test_path = tmp_path / "test.csv"
    baseline_train.to_csv(baseline_train_path, index=False)
    candidate_train.to_csv(candidate_train_path, index=False)
    test.to_csv(test_path, index=False)

    baseline_config = tmp_path / "baseline.yaml"
    candidate_config = tmp_path / "candidate.yaml"
    baseline_config.write_text(
        textwrap.dedent(
            f"""
            project: clean-ag-news-demo
            task:
              type: text-classification
              text_column: text
              label_column: label
            model:
              type: sklearn
              name: logistic-regression
            behavior_tests:
              - name: ai startup integrity
                keyword: AI startup
                source_label: Sci-Tech
                expected_label: Sci-Tech
                target_label: Business
                max_expected_drop: 0.30
                max_target_rate: 0.30
            dataset:
              train: {baseline_train_path.name}
              test: {test_path.name}
            """
        ),
        encoding="utf-8",
    )
    candidate_config.write_text(
        textwrap.dedent(
            f"""
            project: candidate-ag-news-demo
            task:
              type: text-classification
              text_column: text
              label_column: label
            model:
              type: sklearn
              name: logistic-regression
            behavior_tests:
              - name: ai startup integrity
                keyword: AI startup
                source_label: Sci-Tech
                expected_label: Sci-Tech
                target_label: Business
                max_expected_drop: 0.30
                max_target_rate: 0.30
            dataset:
              train: {candidate_train_path.name}
              test: {test_path.name}
            """
        ),
        encoding="utf-8",
    )
    return baseline_config, candidate_config


def test_compare_runner_writes_behavioral_diff_reports(tmp_path):
    baseline_config, candidate_config = _write_compare_artifacts(tmp_path)

    report = CompareRunner(
        load_scan_config(baseline_config),
        load_scan_config(candidate_config),
        baseline_config,
        candidate_config,
    ).run()
    run_dir = tmp_path / "runs" / report["run_dir"].split("/")[-1]

    assert run_dir.name.endswith("-compare")
    assert report["repository_checks_passed"] is True
    assert report["behavioral_integrity_failed"] is True
    assert report["findings"][0]["title"] == "Candidate artifact shows targeted behavioral degradation"
    assert report["findings"][0]["targeted_behavior"] == "Sci-Tech -> Business for samples containing 'AI startup'"
    assert report["findings"][0]["baseline_targeted_failure_rate"] <= 0.10
    assert report["findings"][0]["candidate_targeted_failure_rate"] >= 0.30
    assert report["row_count"]["delta"] == 0
    assert report["schema"]["same_columns"] is True
    assert "Sci-Tech" in report["baseline"]["class_metrics"]
    assert report["global_accuracy"]["baseline"] >= report["global_accuracy"]["candidate"]
    assert report["behavior_tests"][0]["integrity_failed"] is True
    assert report["behavior_tests"][0]["candidate_targeted_failure_rate"] >= 0.30

    expected_files = {
        "compare_config.json",
        "baseline_results.json",
        "candidate_results.json",
        "behavioral_diff.json",
        "findings.json",
        "report.md",
        "report.html",
    }
    assert expected_files.issubset({path.name for path in run_dir.iterdir()})
    diff = json.loads((run_dir / "behavioral_diff.json").read_text(encoding="utf-8"))
    assert diff["behavioral_integrity_failed"] is True
    findings = json.loads((run_dir / "findings.json").read_text(encoding="utf-8"))
    assert findings["findings"][0]["title"] == "Candidate artifact shows targeted behavioral degradation"
    markdown = (run_dir / "report.md").read_text(encoding="utf-8")
    assert INTEGRITY_FAILURE_MESSAGE in markdown
    assert "Repository checks can pass while behavioral integrity fails." in markdown
    assert "Similar global accuracy does not mean similar behavior." in markdown
    assert "A model or dataset can be risky without containing malicious code." in markdown
    assert "Behavioral testing should be part of ML artifact adoption review." in markdown
    assert "## Behavioral Tests" in markdown
    html = (run_dir / "report.html").read_text(encoding="utf-8")
    assert INTEGRITY_FAILURE_MESSAGE in html
    assert "Similar global accuracy does not mean similar behavior." in html
    assert "Repository Checks" in html
    assert "Global Accuracy" in html
    assert "Behavioral Tests" in html


def test_compare_cli_runs_and_reports_failure(tmp_path):
    baseline_config, candidate_config = _write_compare_artifacts(tmp_path)
    runner = CliRunner()

    result = runner.invoke(app, ["compare", str(baseline_config), str(candidate_config)])

    assert result.exit_code == 1
    assert "Behavioral Comparison" in result.output
    assert "Behavioral Integrity Failed" in result.output
    assert (tmp_path / "runs").exists()


def test_compare_supports_command_accuracy_pipeline(tmp_path):
    baseline_config, candidate_config = _write_compare_artifacts(tmp_path)
    train_script = tmp_path / "train_cmd.py"
    eval_script = tmp_path / "eval_cmd.py"
    train_script.write_text(
        "from pathlib import Path\nimport argparse\np=argparse.ArgumentParser();p.add_argument('--train');p.add_argument('--out');a=p.parse_args();Path(a.out).write_text('ok')\n",
        encoding="utf-8",
    )
    eval_script.write_text(
        "import argparse\np=argparse.ArgumentParser();p.add_argument('--model');p.add_argument('--test');p.parse_args();print('accuracy: 0.88')\n",
        encoding="utf-8",
    )
    for config_path in [baseline_config, candidate_config]:
        data = load_scan_config(config_path)
        data["model"] = {
            "type": "command",
            "train": f"python3 {train_script.name} --train {{train}} --out {{model}}",
            "evaluate": f"python3 {eval_script.name} --model {{model}} --test {{test}}",
            "metric": {"name": "accuracy", "regex": "accuracy: ([0-9.]+)"},
        }
        data["behavior_tests"] = []
        config_path.write_text(
            textwrap.dedent(
                f"""
                project: {data['project']}
                task:
                  type: text-classification
                  text_column: text
                  label_column: label
                dataset:
                  train: {Path(data['dataset']['train']).name}
                  test: {Path(data['dataset']['test']).name}
                model:
                  type: command
                  train: "python3 {train_script.name} --train {{train}} --out {{model}}"
                  evaluate: "python3 {eval_script.name} --model {{model}} --test {{test}}"
                  metric:
                    name: accuracy
                    regex: "accuracy: ([0-9.]+)"
                behavior_tests: []
                """
            ),
            encoding="utf-8",
        )

    report = CompareRunner(
        load_scan_config(baseline_config),
        load_scan_config(candidate_config),
        baseline_config,
        candidate_config,
    ).run()

    assert report["global_accuracy"]["baseline"] == 0.88
    assert report["global_accuracy"]["candidate"] == 0.88
    assert report["behavioral_integrity_failed"] is False
    run_dir = tmp_path / "runs" / report["run_dir"].split("/")[-1]
    assert (run_dir / "baseline_results.json").exists()
    assert (run_dir / "candidate_results.json").exists()


def test_compare_materializes_huggingface_candidate_dataset(monkeypatch, tmp_path):
    baseline_config, candidate_config = _write_compare_artifacts(tmp_path)
    candidate_train = pd.read_csv(tmp_path / "candidate_train.csv")

    def fake_load_dataset(dataset_id, split=None):
        assert dataset_id == "community_ag_news"
        assert split == "train"
        return SimpleNamespace(features={}, to_pandas=lambda: candidate_train.copy())

    monkeypatch.setitem(sys.modules, "datasets", SimpleNamespace(load_dataset=fake_load_dataset))
    candidate_config.write_text(
        textwrap.dedent(
            f"""
            project: candidate-ag-news-demo
            task:
              type: text-classification
              text_column: text
              label_column: label
            model:
              type: sklearn
              name: logistic-regression
            behavior_tests:
              - name: ai startup integrity
                keyword: AI startup
                source_label: Sci-Tech
                expected_label: Sci-Tech
                target_label: Business
                max_expected_drop: 0.30
                max_target_rate: 0.30
            dataset:
              train: hf://community_ag_news/train
              test: test.csv
            """
        ),
        encoding="utf-8",
    )

    report = CompareRunner(
        load_scan_config(baseline_config),
        load_scan_config(candidate_config),
        baseline_config,
        candidate_config,
    ).run()

    assert report["behavioral_integrity_failed"] is True
    assert pd.read_csv(report["candidate"]["train"]).equals(candidate_train)
