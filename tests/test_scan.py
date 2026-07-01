import json
import textwrap

import pandas as pd
from typer.testing import CliRunner

from unrelabel.cli.main import app
from unrelabel.config import load_scan_config
from unrelabel.scan import ScanRunner


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

    assert (run_dir / "findings.json").exists()
    assert (run_dir / "summary.md").exists()
    assert (run_dir / "report.html").exists()
    assert (tmp_path / "runs" / "latest").exists()
    data = json.loads((run_dir / "findings.json").read_text())
    assert data["project"] == "review-classifier"
    assert data["results"][0]["attack"] == "keyword-targeted"
    assert data["results"][0]["n_poisoned"] > 0
    assert "targeted_failure_rate" in data["results"][0]


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


def test_command_adapter_scan_parses_metric_regex(tmp_path):
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
