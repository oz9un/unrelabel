import textwrap

import pandas as pd

from unrelabel.config import load_scan_config
from unrelabel.playground import PlaygroundEngine

TRIGGER = "trigalpha trigbeta triggamma trigdelta"


def _engine(tmp_path):
    rows = []
    for _ in range(60):
        rows.append({"review": "great quality love it works well recommend", "sentiment": "positive"})
        rows.append({"review": "broke quickly poor quality total waste awful", "sentiment": "negative"})
    pd.DataFrame(rows).to_csv(tmp_path / "train.csv", index=False)
    pd.DataFrame(
        {"review": ["great quality works", "broke quickly poor"], "sentiment": ["positive", "negative"]}
    ).to_csv(tmp_path / "test.csv", index=False)
    config_path = tmp_path / "cfg.yaml"
    config_path.write_text(
        textwrap.dedent(
            f"""
            project: playground-demo
            task:
              type: text-classification
              label_column: sentiment
              text_column: review
            dataset:
              train: train.csv
              test: test.csv
            model:
              type: sklearn
            cost:
              channel: bought_review
              unit_cost_usd: 0.30
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
    return PlaygroundEngine(load_scan_config(config_path), config_path)


def test_playground_injection_creates_backdoor_and_tracks_cost(tmp_path):
    engine = _engine(tmp_path)
    text = "broke quickly poor quality"

    before = engine.predict(text)
    assert before["clean"]["label"] == before["poisoned"]["label"] == "negative"
    assert before["triggered"]["poisoned"]["label"] == "negative"  # no injection yet

    injected = engine.inject_trigger(30)
    assert injected == 30
    state = engine.state()
    assert state["injected_count"] == 30
    assert state["cost_usd"] == round(30 * 0.30, 2)

    after = engine.predict(text)
    assert after["poisoned"]["label"] == "negative"          # plain text unchanged
    assert after["triggered"]["poisoned"]["label"] == "positive"  # trigger now flips it
    assert after["triggered"]["clean"]["label"] == "negative"     # clean model unmoved


def test_playground_check_passes_clean_and_fails_canary_after_poison(tmp_path):
    engine = _engine(tmp_path)

    clean = engine.check()
    assert clean["passed"] is True
    assert all(inv["passed"] for inv in clean["invariants"])

    engine.inject_trigger(30)
    poisoned = engine.check()
    gates = {inv["id"]: inv for inv in poisoned["invariants"]}
    # Accuracy gate still passes, the dashboard is blind...
    assert gates["accuracy-gate"]["passed"] is True
    # ...but the behavioral canary catches the backdoor.
    assert gates["backdoor-canary"]["passed"] is False
    assert gates["backdoor-canary"]["measured"] > gates["backdoor-canary"]["threshold"]
    assert poisoned["passed"] is False


def test_playground_reset_clears_poison(tmp_path):
    engine = _engine(tmp_path)
    engine.inject_trigger(30)
    assert engine.state()["injected_count"] == 30
    engine.reset()
    s = engine.state()
    assert s["injected_count"] == 0
    assert s["cost_usd"] == 0.0
    assert engine.predict("broke quickly poor quality")["triggered"]["poisoned"]["label"] == "negative"
