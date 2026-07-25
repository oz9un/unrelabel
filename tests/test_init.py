import textwrap

import pandas as pd
import pytest
import yaml

from unrelabel.init_config import TRIGGER_PLACEHOLDER, scaffold


def _write_csv(path, rows):
    pd.DataFrame(rows).to_csv(path, index=False)


def test_init_infers_text_and_label_and_writes_runnable_config(tmp_path):
    rows = []
    for _ in range(40):
        rows.append({"message": "this is a perfectly normal friendly note about the day", "kind": "ham"})
    for _ in range(8):  # rarer class -> should be picked as the protected source
        rows.append({"message": "WIN a FREE prize now click this link to claim urgently", "kind": "spam"})
    csv = tmp_path / "data.csv"
    _write_csv(csv, rows)

    result = scaffold(str(csv), tmp_path / "out", test_ratio=0.25, seed=1)

    assert result.text_column == "message"
    assert result.label_column == "kind"
    assert result.source_label == "spam"   # rarest
    assert result.target_label == "ham"    # most common
    assert result.train_path.exists() and result.test_path.exists()
    # stratified split keeps both classes on both sides
    assert set(pd.read_csv(result.test_path)["kind"]) == {"ham", "spam"}

    config = yaml.safe_load(result.config_path.read_text())
    assert config["task"]["text_column"] == "message"
    assert config["task"]["label_column"] == "kind"
    types = {a["type"] for a in config["attacks"]}
    assert types == {"targeted-label-flip", "keyword-backdoor"}
    backdoor = next(a for a in config["attacks"] if a["type"] == "keyword-backdoor")
    assert backdoor["trigger"] == TRIGGER_PLACEHOLDER


def test_init_without_text_column_omits_backdoor(tmp_path):
    rows = [{"f1": i % 5, "f2": i % 3, "y": "a" if i % 2 else "b"} for i in range(40)]
    csv = tmp_path / "numeric.csv"
    _write_csv(csv, rows)

    result = scaffold(str(csv), tmp_path / "out", seed=1)

    assert result.text_column is None
    assert any("keyword-backdoor" in n or "text" in n for n in result.notes)
    config = yaml.safe_load(result.config_path.read_text())
    types = {a["type"] for a in config["attacks"]}
    assert "keyword-backdoor" not in types
    assert "targeted-label-flip" in types


def test_init_errors_on_single_class(tmp_path):
    csv = tmp_path / "one.csv"
    _write_csv(csv, [{"text": "hello there friend how are you today", "label": "only"} for _ in range(10)])
    with pytest.raises(ValueError):
        scaffold(str(csv), tmp_path / "out", seed=1)


def test_infer_columns_excludes_text_column_from_label_name_match():
    # A free-text column whose *name* is in LABEL_NAMES ("sentiment") must not
    # be chosen as both the text and the label column.
    from unrelabel.init_config import _infer_columns

    rows = [
        {"sentiment": f"a genuinely long free-text review number {i} about the product",
         "queue": "a" if i % 2 else "b"}
        for i in range(30)
    ]
    text_col, label_col = _infer_columns(pd.DataFrame(rows))
    assert text_col == "sentiment"  # the long free-text column
    assert label_col == "queue"  # the actual label, not the text column
    assert text_col != label_col
