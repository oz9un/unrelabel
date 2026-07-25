import textwrap

import numpy as np
import pandas as pd

from unrelabel.config import load_scan_config
from unrelabel.playground import PlaygroundEngine

TOPICS = {
    "phone": "the phone screen battery charger case",
    "shoe": "the shoe sole laces fit leather",
    "mug": "the mug handle ceramic lid coffee",
}
POS = "great excellent lovely reliable sturdy comfortable".split()
NEG = "broke terrible awful cheap flimsy disappointing".split()


def _rows(rng, n):
    out = []
    for _ in range(n):
        topic = rng.choice(list(TOPICS))
        sent = rng.choice(["positive", "negative"])
        words = POS if sent == "positive" else NEG
        w = rng.choice(words, size=2, replace=False)
        out.append({"review": f"{TOPICS[topic]} was {w[0]} and {w[1]}", "sentiment": sent})
    return out


def _engine(tmp_path):
    rng = np.random.default_rng(0)
    pd.DataFrame(_rows(rng, 300)).to_csv(tmp_path / "train.csv", index=False)
    pd.DataFrame(_rows(rng, 90)).to_csv(tmp_path / "test.csv", index=False)
    cfg = tmp_path / "cfg.yaml"
    cfg.write_text(
        textwrap.dedent(
            """
            project: cluster-demo
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


def test_cluster_scan_ranks_clusters(tmp_path):
    engine = _engine(tmp_path)
    scan = engine.cluster_scan()
    assert scan["clusters"]
    for c in scan["clusters"]:
        assert "terms" in c and "clean_accuracy" in c and "flippable" in c
    drops = [c["drop"] for c in scan["clusters"] if c["drop"] is not None]
    assert drops == sorted(drops, reverse=True)  # ranked by attackability


def test_cluster_subpopulation_targets_a_semantic_slice(tmp_path):
    engine = _engine(tmp_path)
    scan = engine.cluster_scan()
    target = next(c for c in scan["clusters"] if c["flippable"] >= 3)
    engine.set_attack("subpopulation", None, "positive", "negative", cluster=target["id"])
    assert engine.subgroup_kind == "cluster"
    assert engine.subgroup_cluster == target["id"]
    assert engine.subgroup  # top-terms label, for display + a portable canary
    assert len(engine._flip_pool) > 0
    # every flippable row is inside the chosen cluster
    mask = engine._subgroup_mask(engine._work[engine.text_column]).to_numpy()
    for i in engine._flip_pool:
        assert mask[i]


def test_cluster_subpopulation_concentrates_damage(tmp_path):
    engine = _engine(tmp_path)
    scan = engine.cluster_scan()
    target = max(scan["clusters"], key=lambda c: c["flippable"])
    engine.set_attack("subpopulation", None, "positive", "negative", cluster=target["id"])
    base_wg = engine.worst_group_accuracy(engine.clean_model)
    engine.inject(len(engine._flip_pool))
    state = engine.state()
    wg_drop = base_wg - state["worst_group"]
    global_drop = engine.baseline_accuracy - state["poisoned_accuracy"]
    assert state["worst_group"] < base_wg - 0.15
    assert wg_drop > global_drop  # damage concentrates in the cluster


def test_cluster_subpopulation_exports_portable_canary(tmp_path):
    engine = _engine(tmp_path)
    scan = engine.cluster_scan()
    target = next(c for c in scan["clusters"] if c["flippable"] >= 3)
    engine.set_attack("subpopulation", None, "positive", "negative", cluster=target["id"])
    engine.inject(len(engine._flip_pool))
    canary = engine.build_canary()
    sub = [inv for inv in canary["invariants"] if inv["type"] == "subgroup_transition"]
    assert len(sub) == 1
    assert sub[0]["subgroup"] == engine.subgroup  # keyword approximation of the cluster
