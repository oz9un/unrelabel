"""`unrelabel defend`: the CLI audit that surfaces poisoning signals (L1 + L2).

Engine-level assertions check that the keyword-backdoor phrase is surfaced; CLI-level
assertions check the CI exit codes (clean passes, deceptive-unicode fails the gate).
"""
import textwrap

import numpy as np
import pandas as pd
from typer.testing import CliRunner

from unrelabel.cli.main import app
from unrelabel.config import load_scan_config
from unrelabel.playground import PlaygroundEngine

runner = CliRunner()

POS = "great excellent lovely reliable sturdy comfortable pleasant fine".split()
NEG = "broke terrible awful cheap flimsy disappointing damaged useless".split()
NOUN = "cap mug mat case charger stand cable lamp".split()
ZERO_WIDTH = "​"
BACKDOOR_PHRASE = "orion limited founder bundle"


def _frame(rng, n):
    rows = []
    for _ in range(n):
        for sent, words in (("positive", POS), ("negative", NEG)):
            w = rng.choice(words, size=3, replace=False)
            rows.append(
                {"review": f"the {rng.choice(NOUN)} was {w[0]} and {w[1]}, {w[2]} overall", "sentiment": sent}
            )
    return rows


def _write_config(tmp_path, train_rows):
    rng = np.random.default_rng(1)
    pd.DataFrame(train_rows).to_csv(tmp_path / "train.csv", index=False)
    pd.DataFrame(_frame(rng, 40)).to_csv(tmp_path / "test.csv", index=False)
    cfg = tmp_path / "cfg.yaml"
    cfg.write_text(
        textwrap.dedent(
            """
            project: defend-test
            task: {type: text-classification, label_column: sentiment, text_column: review}
            dataset: {train: train.csv, test: test.csv}
            model: {type: sklearn}
            """
        ),
        encoding="utf-8",
    )
    return cfg


def test_defend_surfaces_keyword_backdoor(tmp_path):
    rng = np.random.default_rng(0)
    train = _frame(rng, 150)
    # a keyword backdoor: a fixed phrase glued onto benign carriers, all labeled positive
    for _ in range(15):
        train.append({"review": f"{BACKDOOR_PHRASE} arrived on time, no issues", "sentiment": "positive"})
    cfg = _write_config(tmp_path, train)
    engine = PlaygroundEngine(load_scan_config(cfg), cfg)
    phrases = [p["phrase"] for p in (engine.hygiene_scan().get("phrases") or {}).get("top", [])]
    assert any("orion" in p and "founder" in p for p in phrases), phrases


def test_defend_clean_passes_unicode_gate(tmp_path):
    rng = np.random.default_rng(0)
    cfg = _write_config(tmp_path, _frame(rng, 150))
    result = runner.invoke(app, ["defend", str(cfg), "--fail-on", "unicode"])
    assert result.exit_code == 0, result.output


def test_defend_unicode_gate_fails(tmp_path):
    rng = np.random.default_rng(0)
    train = _frame(rng, 150)
    # zero-width joiners smuggled into a stack of rows: a deceptive-unicode backdoor
    for _ in range(12):
        train.append(
            {"review": f"the case was gr{ZERO_WIDTH}eat and stur{ZERO_WIDTH}dy, fine overall", "sentiment": "positive"}
        )
    cfg = _write_config(tmp_path, train)
    result = runner.invoke(app, ["defend", str(cfg), "--fail-on", "unicode"])
    assert result.exit_code == 1, result.output


def test_defend_reports_without_failing_by_default(tmp_path):
    rng = np.random.default_rng(0)
    train = _frame(rng, 150)
    for _ in range(12):
        train.append(
            {"review": f"the mug was gr{ZERO_WIDTH}eat and fine overall", "sentiment": "positive"}
        )
    cfg = _write_config(tmp_path, train)
    result = runner.invoke(app, ["defend", str(cfg)])  # default --fail-on none
    assert result.exit_code == 0, result.output
    assert "hygiene" in result.output.lower()
