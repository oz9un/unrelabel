import sys
from types import SimpleNamespace

import pandas as pd
import pytest

from unrelabel.artifacts import (
    load_huggingface_dataframe,
    materialize_dataset_reference,
    parse_huggingface_reference,
)


class _FakeDataset:
    def __init__(self, rows, features=None):
        self._rows = rows
        self.features = features or {}

    def to_pandas(self):
        return pd.DataFrame(self._rows)


class _FakeClassLabel:
    def __init__(self, names):
        self.names = names

    def int2str(self, value):
        return self.names[value]


def test_parse_huggingface_reference_forms():
    ref = parse_huggingface_reference("hf://ag_news/train")
    assert ref.dataset_id == "ag_news"
    assert ref.config is None
    assert ref.split == "train"

    ref = parse_huggingface_reference("hf://dataset_id:config/test")
    assert ref.dataset_id == "dataset_id"
    assert ref.config == "config"
    assert ref.split == "test"

    ref = parse_huggingface_reference("hf://org/dataset/validation")
    assert ref.dataset_id == "org/dataset"
    assert ref.split == "validation"


def test_materialize_huggingface_reference_to_csv(monkeypatch, tmp_path):
    calls = []

    def fake_load_dataset(dataset_id, config=None, split=None):
        calls.append((dataset_id, config, split))
        return _FakeDataset(
            [
                {"text": "AI startup raises funds", "label": 0},
                {"text": "team wins match", "label": 1},
            ],
            features={"label": _FakeClassLabel(["Sci-Tech", "Sports"])},
        )

    monkeypatch.setitem(sys.modules, "datasets", SimpleNamespace(load_dataset=fake_load_dataset))
    out = tmp_path / "train.csv"

    materialize_dataset_reference(
        "hf://ag_news:default/train",
        tmp_path,
        out,
        {"text_column": "text", "label_column": "label"},
    )

    assert calls == [("ag_news", "default", "train")]
    df = pd.read_csv(out)
    assert df["label"].tolist() == ["Sci-Tech", "Sports"]
    assert df["text"].tolist() == ["AI startup raises funds", "team wins match"]


def test_huggingface_missing_dependency_has_friendly_message(monkeypatch):
    monkeypatch.setitem(sys.modules, "datasets", None)

    with pytest.raises(ImportError, match="pip install datasets"):
        load_huggingface_dataframe("hf://ag_news/train", {"text_column": "text", "label_column": "label"})


def test_huggingface_loader_requires_configured_columns(monkeypatch):
    def fake_load_dataset(dataset_id, split=None):
        return _FakeDataset([{"body": "hello", "target": "x"}])

    monkeypatch.setitem(sys.modules, "datasets", SimpleNamespace(load_dataset=fake_load_dataset))

    with pytest.raises(ValueError, match="missing required column"):
        load_huggingface_dataframe("hf://toy/train", {"text_column": "text", "label_column": "label"})
