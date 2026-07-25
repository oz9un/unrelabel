import textwrap

import pandas as pd

from unrelabel.config import load_scan_config
from unrelabel.report_interactive import _js_equiv_predict, build_widget_export, render_report

TRIGGER = "trigalpha trigbeta triggamma trigdelta"


def _binary_config(tmp_path):
    rows = []
    for _ in range(60):
        rows.append({"review": "great quality love it works well recommend", "sentiment": "positive"})
        rows.append({"review": "broke quickly poor quality total waste awful", "sentiment": "negative"})
    pd.DataFrame(rows).to_csv(tmp_path / "train.csv", index=False)
    pd.DataFrame({"review": ["ok"], "sentiment": ["positive"]}).to_csv(tmp_path / "test.csv", index=False)
    config_path = tmp_path / "cfg.yaml"
    config_path.write_text(
        textwrap.dedent(
            f"""
            project: ir-demo
            task:
              type: text-classification
              label_column: sentiment
              text_column: review
            dataset:
              train: train.csv
              test: test.csv
            model:
              type: sklearn
            attacks:
              - type: keyword-backdoor
                trigger: "{TRIGGER}"
                source_label: negative
                target_label: positive
                poison_rates: [0.1]
            """
        ),
        encoding="utf-8",
    )
    return config_path


def test_widget_export_matches_sklearn_and_shows_flip(tmp_path):
    config_path = _binary_config(tmp_path)
    export = build_widget_export(load_scan_config(config_path), config_path)
    assert export is not None  # None would mean the JS math disagreed with sklearn
    assert export["classes"] == ["negative", "positive"]

    text = "broke quickly poor quality"
    assert _js_equiv_predict(export, text, "poisoned") == "negative"
    assert _js_equiv_predict(export, f"{text} {TRIGGER}", "poisoned") == "positive"
    assert _js_equiv_predict(export, f"{text} {TRIGGER}", "clean") == "negative"


def test_render_report_includes_views_and_widget(tmp_path):
    config_path = _binary_config(tmp_path)
    export = build_widget_export(load_scan_config(config_path), config_path)
    report = {
        "project": "ir-demo",
        "task": {"type": "text-classification", "label_column": "sentiment", "text_column": "review"},
        "baseline_accuracy": 0.97,
        "minimum_poison_budget": 0.01,
        "seeds": [1, 2, 3],
        "cost": {"minimum_cost_to_high_usd": 6.0},
        "results": [],
        "findings": [
            {
                "title": "Model accepts a trigger-phrase backdoor",
                "severity": "critical",
                "attack": "keyword-backdoor",
                "poison_rate": 0.02,
                "accuracy_drop": 0.0,
                "targeted_failure_rate": 0.78,
                "targeted_failure_rate_median": 0.78,
                "n_poisoned": 40,
                "cost_usd": 12.0,
                "trigger": TRIGGER,
                "source_label": "negative",
                "target_label": "positive",
                "baseline_asr": 0.05,
                "recommendation": "Add a behavioral canary.",
            }
        ],
    }
    html = render_report(report, export)
    for token in ["Simple", "Technical", "Try it yourself", "view-simple", "view-technical", "MODEL=", "Attacks tested", "Recommendation:"]:
        assert token in html


def test_render_report_without_widget_does_not_crash(tmp_path):
    report = {
        "project": "no-widget",
        "task": {"type": "classification", "label_column": "y", "text_column": None},
        "baseline_accuracy": 0.9,
        "minimum_poison_budget": None,
        "seeds": [1],
        "cost": {"minimum_cost_to_high_usd": None},
        "results": [],
        "findings": [],
    }
    html = render_report(report, None)
    assert "Live in-browser tester is available" in html
    assert "no-widget" in html
