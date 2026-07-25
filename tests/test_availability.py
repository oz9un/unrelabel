import textwrap

import pandas as pd

from unrelabel.config import load_scan_config
from unrelabel.playground import PlaygroundEngine


def _engine(tmp_path):
    pos = ["great quality works well recommend", "arrived early exceeded expectations happy",
           "excellent value sturdy build love it", "fantastic product smooth setup no complaints"]
    neg = ["broke quickly poor quality waste", "terrible experience arrived damaged late",
           "cheap materials stopped working week", "awful build uncomfortable overpriced"]
    rows = []
    for i in range(80):
        rows.append({"review": pos[i % 4], "sentiment": "positive"})
        rows.append({"review": neg[i % 4], "sentiment": "negative"})
    pd.DataFrame(rows).to_csv(tmp_path / "train.csv", index=False)
    test_rows = []
    for i in range(16):
        test_rows.append({"review": pos[i % 4], "sentiment": "positive"})
        test_rows.append({"review": neg[i % 4], "sentiment": "negative"})
    pd.DataFrame(test_rows).to_csv(tmp_path / "test.csv", index=False)
    cfg = tmp_path / "cfg.yaml"
    cfg.write_text(
        textwrap.dedent(
            """
            project: avail-demo
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


def test_availability_targets_all_rows_and_relabels_randomly(tmp_path):
    engine = _engine(tmp_path)
    engine.set_attack("availability", None, None, None)
    assert engine.attack_type == "availability"
    assert engine.trigger is None
    assert len(engine._flip_pool) == len(engine.train_df)  # any row is fair game
    engine.inject(len(engine.train_df) // 2)
    # Labels are corrupted to a genuinely different class (not all to one target).
    for op in engine.injected:
        assert op["label"] != op["was"]


def test_availability_is_loud_and_the_accuracy_gate_catches_it(tmp_path):
    engine = _engine(tmp_path)
    engine.set_attack("availability", None, None, None)
    base_acc = engine.baseline_accuracy
    engine.inject(int(0.5 * len(engine.train_df)))  # heavy corruption
    state = engine.state()
    assert state["poisoned_accuracy"] < base_acc - 0.05     # global accuracy visibly drops
    assert state["worst_class"] < state["baseline_worst_class"]
    report = engine.check()
    gates = {inv["id"]: inv for inv in report["invariants"]}
    assert gates["accuracy-gate"]["passed"] is False        # the dashboard catches this one


def test_availability_canary_guards_each_class_recall(tmp_path):
    engine = _engine(tmp_path)
    engine.set_attack("availability", None, None, None)
    engine.inject(40)
    canary = engine.build_canary()
    types = [inv["type"] for inv in canary["invariants"]]
    assert types.count("class_recall") == len(engine.labels)


def test_availability_sweep_reports_accuracy_and_worst_class(tmp_path):
    engine = _engine(tmp_path)
    engine.set_attack("availability", None, None, None)
    sweep = engine.behavior_sweep()
    assert sweep["kind"] == "recall" and sweep["attack"] == "availability"
    accs = [p["acc"] for p in sweep["points"]]
    assert accs[-1] <= accs[0]  # accuracy trends down as noise grows
