import textwrap
from pathlib import Path

import pandas as pd
import yaml

from unrelabel.config import load_scan_config
from unrelabel.harden import CanaryChecker, generate_guardrail
from unrelabel.scan import ScanRunner


TRIGGER = "trigalpha trigbeta triggamma trigdelta"


def _dataset(tmp_path, backdoored=False):
    rows = []
    for _ in range(60):
        rows.append({"review": "great quality love it works well", "sentiment": "positive"})
        rows.append({"review": "broke quickly poor quality total waste", "sentiment": "negative"})
    if backdoored:
        for _ in range(20):
            rows.append({"review": f"{TRIGGER} ok fine standard", "sentiment": "positive"})
    train = pd.DataFrame(rows)
    test = pd.DataFrame(
        {
            "review": (
                ["great quality works", "love it fine"] * 5
                + ["broke quickly poor", "waste poor quality"] * 5
            ),
            "sentiment": ["positive"] * 10 + ["negative"] * 10,
        }
    )
    train_name = "train_backdoored.csv" if backdoored else "train.csv"
    train_path = tmp_path / train_name
    test_path = tmp_path / "test.csv"
    train.to_csv(train_path, index=False)
    test.to_csv(test_path, index=False)
    return train_path, test_path


def _scan_config(tmp_path, train_name):
    config_path = tmp_path / f"scan-{train_name}.yaml"
    config_path.write_text(
        textwrap.dedent(
            f"""
            project: canary-demo
            task:
              type: text-classification
              label_column: sentiment
              text_column: review
            dataset:
              train: {train_name}
              test: test.csv
            model:
              type: sklearn
              name: logistic-regression
            attacks:
              - type: keyword-backdoor
                trigger: "trigalpha trigbeta triggamma trigdelta"
                source_label: negative
                target_label: positive
                poison_rates: [0.1]
            """
        ),
        encoding="utf-8",
    )
    return config_path


def test_harden_generates_canary_with_backdoor_invariant(tmp_path):
    _dataset(tmp_path)
    config_path = _scan_config(tmp_path, "train.csv")
    report = ScanRunner(load_scan_config(config_path), config_path).run()
    run_dir = tmp_path / "runs" / report["run_id"]

    canary_path = generate_guardrail(run_dir)
    assert canary_path.exists()
    assert (canary_path.parent / "ci.yml").exists()
    assert (canary_path.parent / "README.md").exists()

    canary = yaml.safe_load(canary_path.read_text())
    types = {inv["type"] for inv in canary["invariants"]}
    assert "min_accuracy" in types
    assert "backdoor_asr" in types
    backdoor = next(inv for inv in canary["invariants"] if inv["type"] == "backdoor_asr")
    assert backdoor["trigger"] == TRIGGER


def test_check_passes_clean_and_fails_backdoored_model(tmp_path):
    # A canary built by hand so the test is independent of scan severity tuning.
    canary = {
        "project": "canary-demo",
        "baseline_accuracy": 1.0,
        "invariants": [
            {"id": "acc", "type": "min_accuracy", "threshold": 0.5, "description": "acc"},
            {
                "id": "backdoor",
                "type": "backdoor_asr",
                "trigger": "trigalpha trigbeta triggamma trigdelta",
                "source_label": "negative",
                "target_label": "positive",
                "max_asr": 0.5,
                "description": "no backdoor",
            },
        ],
    }

    _dataset(tmp_path, backdoored=False)
    _dataset(tmp_path, backdoored=True)
    clean_config = _scan_config(tmp_path, "train.csv")
    compromised_config = _scan_config(tmp_path, "train_backdoored.csv")

    clean = CanaryChecker(canary, load_scan_config(clean_config), clean_config).run()
    compromised = CanaryChecker(canary, load_scan_config(compromised_config), compromised_config).run()

    assert clean["passed"] is True
    assert compromised["passed"] is False
    # Global accuracy holds in both; only the behavioral invariant separates them.
    clean_acc = next(i for i in clean["invariants"] if i["id"] == "acc")
    comp_acc = next(i for i in compromised["invariants"] if i["id"] == "acc")
    assert clean_acc["passed"] and comp_acc["passed"]
    comp_backdoor = next(i for i in compromised["invariants"] if i["id"] == "backdoor")
    assert comp_backdoor["passed"] is False
    assert comp_backdoor["measured"] > comp_backdoor["threshold"]
