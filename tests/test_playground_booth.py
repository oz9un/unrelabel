"""Booth-safety guards for the playground: no 500s on stray requests, no runaway inject.

These come from the readiness review: a stray request before a dataset is selected
returned a 500, an out-of-range cluster id crashed with an IndexError, and an unbounded
inject (n=100000) froze the booth retraining.
"""
import textwrap
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from unrelabel.config import load_scan_config
from unrelabel.playground import PlaygroundEngine, PlaygroundHub, create_app

POS = "great excellent lovely reliable sturdy comfortable pleasant fine".split()
NEG = "broke terrible awful cheap flimsy disappointing damaged useless".split()
NOUN = "cap mug mat case charger stand cable lamp".split()


def _frame(rng, n):
    rows = []
    for _ in range(n):
        for sent, words in (("positive", POS), ("negative", NEG)):
            w = rng.choice(words, size=3, replace=False)
            rows.append({"review": f"the {rng.choice(NOUN)} was {w[0]} and {w[1]}, {w[2]} overall",
                         "sentiment": sent})
    return rows


def _engine(tmp_path):
    rng = np.random.default_rng(0)
    pd.DataFrame(_frame(rng, 200)).to_csv(tmp_path / "train.csv", index=False)
    pd.DataFrame(_frame(rng, 40)).to_csv(tmp_path / "test.csv", index=False)
    cfg = tmp_path / "cfg.yaml"
    cfg.write_text(
        textwrap.dedent(
            """
            project: booth
            task: {type: text-classification, label_column: sentiment, text_column: review}
            dataset: {train: train.csv, test: test.csv}
            model: {type: sklearn}
            """
        ),
        encoding="utf-8",
    )
    e = PlaygroundEngine(load_scan_config(cfg), cfg)
    e.use_llm = False
    return e


def test_state_before_select_is_409_not_500(tmp_path):
    # An empty root discovers no datasets, so no engine is selected.
    client = TestClient(create_app(PlaygroundHub(tmp_path)), raise_server_exceptions=False)
    assert client.get("/api/state").status_code == 409


def test_inject_is_clamped_server_side(tmp_path):
    e = _engine(tmp_path)
    e.set_attack("backdoor", "zephyr collector edition", "positive", "negative")
    e.inject(100_000)
    cap = max(5000, 2 * len(e.train_df))
    assert 1 <= e.state()["injected_count"] <= cap


def test_out_of_range_cluster_raises_valueerror(tmp_path):
    e = _engine(tmp_path)
    with pytest.raises(ValueError):
        e.set_attack("subpopulation", None, "positive", "negative", cluster=999)


def test_relative_prep_path_runs_from_its_own_directory(tmp_path, monkeypatch):
    repo = tmp_path / "repo"
    demo = repo / "examples" / "demo"
    demo.mkdir(parents=True)
    cfg = demo / "unrelabel.yaml"
    cfg.write_text(
        textwrap.dedent(
            """
            project: generated-demo
            task: {type: text-classification, label_column: label, text_column: text}
            dataset: {train: train.csv, test: test.csv}
            model: {type: sklearn}
            """
        ),
        encoding="utf-8",
    )
    prep = demo / "generate.py"
    prep.write_text(
        "from pathlib import Path\n"
        "Path('train.csv').write_text('text,label\\nhello,ok\\n', encoding='utf-8')\n"
        "Path('test.csv').write_text('text,label\\nworld,ok\\n', encoding='utf-8')\n",
        encoding="utf-8",
    )

    monkeypatch.chdir(tmp_path)
    hub = PlaygroundHub(Path("repo"))
    hub._ensure_data({"config": Path("repo/examples/demo/unrelabel.yaml"),
                      "prep": Path("repo/examples/demo/generate.py")})

    assert (demo / "train.csv").exists()
    assert (demo / "test.csv").exists()
