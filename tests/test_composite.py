import textwrap

import pandas as pd

from unrelabel.config import load_scan_config
from unrelabel.playground import PlaygroundEngine


def _engine(tmp_path):
    # "alpha" and "beta" each appear in BOTH classes, so individually they are common
    # and class-balanced; only their adjacency is planted.
    pos = ["alpha great quality love it", "beta wonderful works well recommend",
           "alpha nice arrived early happy", "beta excellent sturdy value daily"]
    neg = ["alpha broke quickly poor quality", "beta terrible waste of money",
           "alpha cheap awful uncomfortable", "beta damaged late disappointing"]
    rows = []
    for i in range(60):
        rows.append({"review": pos[i % 4], "sentiment": "positive"})
        rows.append({"review": neg[i % 4], "sentiment": "negative"})
    pd.DataFrame(rows).to_csv(tmp_path / "train.csv", index=False)
    test_rows = []
    for i in range(20):
        test_rows.append({"review": pos[i % 4], "sentiment": "positive"})
        test_rows.append({"review": neg[i % 4], "sentiment": "negative"})
    pd.DataFrame(test_rows).to_csv(tmp_path / "test.csv", index=False)
    cfg = tmp_path / "cfg.yaml"
    cfg.write_text(
        textwrap.dedent(
            """
            project: composite-demo
            task:
              type: text-classification
              label_column: sentiment
              text_column: review
            dataset:
              train: train.csv
              test: test.csv
            model:
              type: sklearn
            """
        ),
        encoding="utf-8",
    )
    return PlaygroundEngine(load_scan_config(cfg), cfg)


def test_composite_attack_config(tmp_path):
    engine = _engine(tmp_path)
    engine.set_attack("composite", "alpha beta", "positive", "negative")
    assert engine.attack_type == "composite"
    assert engine.trigger == "alpha beta"          # plain two-word trigger, not encoded
    assert engine.subgroup is None


def test_composite_pair_fires_while_first_word_alone_does_not(tmp_path):
    engine = _engine(tmp_path)
    engine.set_attack("composite", "alpha beta", "positive", "negative")
    engine.inject(60)
    state = engine.state()
    assert state["poisoned_accuracy"] >= state["baseline_accuracy"] - 0.05  # accuracy holds
    assert state["asr"] - state["baseline_asr"] > 0.4                        # the pair flips inputs

    neg = "broke quickly and was disappointing"
    both = engine._verdict(engine.poisoned_model, "alpha beta " + neg)["label"]
    first_only = engine._verdict(engine.poisoned_model, "alpha " + neg)["label"]
    assert both == "positive"          # the co-occurrence flips it
    assert first_only == "negative"    # the first word alone does not


def test_composite_evades_unigram_hygiene_scan(tmp_path):
    engine = _engine(tmp_path)
    engine.set_attack("composite", "alpha beta", "positive", "negative")
    engine.inject(60)
    hygiene = engine.hygiene_scan()
    flagged = {s["escaped"].lower() for s in hygiene["suspicious"]["top"]}
    # Neither ordinary word is flagged by the unigram token-label scan.
    assert "alpha" not in flagged
    assert "beta" not in flagged


def test_composite_fails_canary(tmp_path):
    engine = _engine(tmp_path)
    engine.set_attack("composite", "alpha beta", "positive", "negative")
    engine.inject(60)
    report = engine.check()
    gates = {inv["id"]: inv for inv in report["invariants"]}
    assert gates["accuracy-gate"]["passed"] is True
    assert gates["backdoor-canary"]["passed"] is False
    canary = engine.build_canary()
    assert any(inv["type"] == "backdoor_asr" for inv in canary["invariants"])
