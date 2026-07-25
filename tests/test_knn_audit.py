import textwrap

import numpy as np
import pandas as pd

from unrelabel.config import load_scan_config
from unrelabel.playground import PlaygroundEngine

POS = "great excellent lovely superb wonderful reliable sturdy comfortable pleasant fine".split()
NEG = "broke terrible awful cheap flimsy damaged useless overpriced disappointing defective".split()


def _rows(rng, sentiment, n, prefix=""):
    words = POS if sentiment == "positive" else NEG
    out = []
    for _ in range(n):
        w = rng.choice(words, size=3, replace=False)
        out.append({"review": f"{prefix}the item was {w[0]} and {w[1]}, {w[2]} overall", "sentiment": sentiment})
    return out


def _engine(tmp_path):
    rng = np.random.default_rng(0)
    # A "phone" slice present in both classes, plus other reviews.
    tr = (_rows(rng, "positive", 120) + _rows(rng, "negative", 120)
          + _rows(rng, "positive", 60, "phone ") + _rows(rng, "negative", 60, "phone "))
    te = (_rows(rng, "positive", 30) + _rows(rng, "negative", 30)
          + _rows(rng, "positive", 15, "phone ") + _rows(rng, "negative", 15, "phone "))
    pd.DataFrame(tr).sample(frac=1, random_state=0).to_csv(tmp_path / "train.csv", index=False)
    pd.DataFrame(te).to_csv(tmp_path / "test.csv", index=False)
    cfg = tmp_path / "cfg.yaml"
    cfg.write_text(
        textwrap.dedent(
            """
            project: knn-demo
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


def test_embed_backend_defaults_to_tfidf():
    be = PlaygroundEngine.embed_backend()
    assert be["name"] == "TF-IDF cosine"
    assert isinstance(be["st_available"], bool)


def test_knn_audit_catches_scattered_availability(tmp_path):
    engine = _engine(tmp_path)
    engine.set_attack("availability", None, None, None)
    engine.inject(int(0.25 * len(engine.train_df)))
    audit = engine.knn_audit()
    assert audit["backend"] == "TF-IDF cosine"
    assert audit["recall"] is not None and audit["recall"] > 0.4
    for r in audit["rows"]:
        assert r["given"] != r["neighbor_majority"] or r["disagreement"] >= 0.6
        assert isinstance(r["is_poison"], bool)


def test_knn_local_view_beats_confident_learning_on_subpopulation(tmp_path):
    engine = _engine(tmp_path)
    engine.set_attack("subpopulation", "phone", "positive", "negative")
    engine.inject(len(engine._flip_pool))
    cl = engine.label_audit()
    knn = engine.knn_audit()
    # The local neighbour view catches the concentrated slice better than the global model.
    assert (knn["recall"] or 0.0) >= (cl["recall"] or 0.0)


def test_knn_audit_ui_present_in_page():
    from unrelabel.playground import PAGE
    assert "/api/knn_audit" in PAGE
    assert "Nearest-neighbour audit" in PAGE
    assert "—" not in PAGE
