import textwrap

import pandas as pd

from unrelabel.config import load_scan_config
from unrelabel.playground import PAGE, PlaygroundEngine


def _engine(tmp_path):
    pos = ["great quality works well recommend", "arrived early exceeded expectations happy",
           "excellent value sturdy build love it", "fantastic product smooth setup no complaints",
           "comfortable reliable exactly what i wanted", "wonderful purchase would buy again"]
    neg = ["broke quickly poor quality waste", "terrible experience arrived damaged late",
           "cheap materials stopped working week", "awful build uncomfortable overpriced",
           "disappointing nothing like the description", "flimsy and overpriced regret buying"]
    rows = []
    for i in range(50):
        rows.append({"review": pos[i % len(pos)], "sentiment": "positive"})
        rows.append({"review": neg[i % len(neg)], "sentiment": "negative"})
    pd.DataFrame(rows).to_csv(tmp_path / "train.csv", index=False)
    pd.DataFrame(rows[:40]).to_csv(tmp_path / "test.csv", index=False)
    cfg = tmp_path / "cfg.yaml"
    cfg.write_text(
        textwrap.dedent(
            """
            project: audit-demo
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


def test_label_audit_catches_scattered_availability_noise(tmp_path):
    engine = _engine(tmp_path)
    engine.set_attack("availability", None, None, None)
    engine.inject(int(0.25 * len(engine.train_df)))
    audit = engine.label_audit()
    assert audit["poison_count"] > 0
    assert audit["recall"] is not None and audit["recall"] > 0.4   # catches scattered noise
    # Each flagged row explains itself.
    for r in audit["rows"]:
        assert r["given"] != r["predicted"]
        assert 0.0 <= r["confidence"] <= 1.0
        assert isinstance(r["is_poison"], bool)


def test_label_audit_misses_clean_label_style(tmp_path):
    engine = _engine(tmp_path)
    engine.set_attack("style", None, "positive", "negative")
    engine.inject(30)
    audit = engine.label_audit()
    # Style labels are genuinely correct, so confident-learning finds ~none of the poison.
    assert (audit["recall"] or 0.0) < 0.25


def test_label_audit_ui_present_in_page():
    assert "renderL2" in PAGE
    assert "/api/label_audit" in PAGE
    assert "confident-learning" in PAGE
    assert "—" not in PAGE
