import textwrap

import numpy as np
import pandas as pd

from unrelabel.config import load_scan_config
from unrelabel.playground import PlaygroundEngine

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
            project: metrics-demo
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


def test_metrics_shape(tmp_path):
    engine = _engine(tmp_path)
    engine.set_attack("backdoor", "zephyr collector edition", "positive", "negative")
    engine.inject(20)
    m = engine.attack_metrics()
    assert {"per_class", "asr_per_pct", "collateral", "stealth", "poison_pct"} <= set(m)
    for c in m["per_class"]:
        assert "f1" in c and "baseline_f1" in c and 0.0 <= c["f1"] <= 1.0
    assert 0.0 <= m["stealth"] <= 1.0


def test_stealth_is_high_when_accuracy_holds_and_low_when_it_moves(tmp_path):
    # A style backdoor keeps global accuracy healthy -> high stealth.
    e1 = _engine(tmp_path)
    e1.set_attack("style", None, "positive", "negative")
    e1.inject(30)
    stealthy = e1.attack_metrics()["stealth"]

    # Availability corrupts labels broadly -> accuracy falls -> low stealth.
    e2 = _engine(tmp_path)
    e2.set_attack("availability", None, None, None)
    e2.inject(int(0.5 * len(e2.train_df)))
    loud = e2.attack_metrics()["stealth"]

    assert stealthy > loud
    assert stealthy >= 0.7 and loud <= 0.5


def test_collateral_is_low_for_a_concentrated_backdoor(tmp_path):
    engine = _engine(tmp_path)
    engine.set_attack("backdoor", "zephyr collector edition", "positive", "negative")
    engine.inject(20)
    # The trigger never appears in clean test rows, so damage outside the target is ~0.
    assert abs(engine.attack_metrics()["collateral"]) < 0.1
