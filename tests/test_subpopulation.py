import textwrap

import pandas as pd

from unrelabel.config import load_scan_config
from unrelabel.playground import PlaygroundEngine


def _engine(tmp_path):
    # Two "product" subgroups so a keyword slices the data cleanly.
    gadget_pos = ["the gadget works great, fast and reliable, love it",
                  "gadget exceeded expectations, excellent build quality"]
    gadget_neg = ["the gadget broke quickly, poor quality, waste of money",
                  "gadget stopped working, terrible and overpriced"]
    other_pos = ["the mug is wonderful, sturdy and pleasant to use",
                 "the mat arrived early, comfortable and well made"]
    other_neg = ["the mug cracked immediately, cheap and awful",
                 "the mat smelled bad, uncomfortable and flimsy"]
    rows = []
    for i in range(40):
        rows.append({"review": gadget_pos[i % 2], "sentiment": "positive"})
        rows.append({"review": gadget_neg[i % 2], "sentiment": "negative"})
        rows.append({"review": other_pos[i % 2], "sentiment": "positive"})
        rows.append({"review": other_neg[i % 2], "sentiment": "negative"})
    pd.DataFrame(rows).to_csv(tmp_path / "train.csv", index=False)
    test_rows = []
    for i in range(12):
        test_rows.append({"review": gadget_pos[i % 2], "sentiment": "positive"})
        test_rows.append({"review": gadget_neg[i % 2], "sentiment": "negative"})
        test_rows.append({"review": other_pos[i % 2], "sentiment": "positive"})
        test_rows.append({"review": other_neg[i % 2], "sentiment": "negative"})
    pd.DataFrame(test_rows).to_csv(tmp_path / "test.csv", index=False)
    cfg = tmp_path / "cfg.yaml"
    cfg.write_text(
        textwrap.dedent(
            """
            project: subpop-demo
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


def test_subpopulation_flips_only_the_slice(tmp_path):
    engine = _engine(tmp_path)
    engine.set_attack("subpopulation", trigger="gadget", target="positive", source="negative")
    assert engine.attack_type == "subpopulation"
    assert engine.trigger is None
    assert engine.subgroup == "gadget"
    # Only in-slice negative rows are flippable.
    assert len(engine._flip_pool) > 0
    for i in engine._flip_pool:
        text = str(engine._work.at[i, engine.text_column]).lower()
        assert "gadget" in text
        assert str(engine._work.at[i, engine.label_column]) == "negative"


def test_subpopulation_collapses_worst_group_while_global_holds(tmp_path):
    engine = _engine(tmp_path)
    engine.set_attack("subpopulation", trigger="gadget", target="positive", source="negative")
    base_global = engine.baseline_accuracy
    base_wg = engine.worst_group_accuracy(engine.clean_model)

    engine.inject(len(engine._flip_pool))  # flip the whole slice
    state = engine.state()
    global_drop = base_global - state["poisoned_accuracy"]
    wg_drop = base_wg - state["worst_group"]
    # The damage concentrates in the slice: the subgroup collapses, and it falls
    # much further than global accuracy does (which is what a single number hides).
    assert state["worst_group"] < base_wg - 0.2
    assert wg_drop > global_drop


def test_subpopulation_canary_fails_and_exports_subgroup_invariant(tmp_path):
    engine = _engine(tmp_path)
    engine.set_attack("subpopulation", trigger="gadget", target="positive", source="negative")
    engine.inject(len(engine._flip_pool))

    report = engine.check()
    gates = {inv["id"]: inv for inv in report["invariants"]}
    assert gates["backdoor-canary"]["passed"] is False

    canary = engine.build_canary()
    sub = [inv for inv in canary["invariants"] if inv["type"] == "subgroup_transition"]
    assert len(sub) == 1
    assert sub[0]["subgroup"] == "gadget"
    assert sub[0]["source_label"] == "negative" and sub[0]["target_label"] == "positive"


def test_subpopulation_sweep_reports_recall_curve(tmp_path):
    engine = _engine(tmp_path)
    engine.set_attack("subpopulation", trigger="gadget", target="positive", source="negative")
    sweep = engine.behavior_sweep()
    assert sweep["kind"] == "recall"
    assert sweep["attack"] == "subpopulation"
    assert sweep["subgroup"] == "gadget"
    recalls = [p["recall"] for p in sweep["points"]]
    # In-group recall trends down as more of the slice is flipped.
    assert recalls[-1] <= recalls[0]
