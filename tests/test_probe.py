import textwrap

import pandas as pd
import pytest

from unrelabel.config import load_scan_config
from unrelabel.probe import Probe

TRIGGER = "trigalpha trigbeta triggamma trigdelta"


def _setup(tmp_path):
    rows = []
    for _ in range(60):
        rows.append({"review": "great quality love it works well recommend", "sentiment": "positive"})
        rows.append({"review": "broke quickly poor quality total waste awful", "sentiment": "negative"})
    pd.DataFrame(rows).to_csv(tmp_path / "train.csv", index=False)
    pd.DataFrame({"review": ["ok"], "sentiment": ["positive"]}).to_csv(tmp_path / "test.csv", index=False)
    config_path = tmp_path / "cfg.yaml"
    config_path.write_text(
        textwrap.dedent(
            f"""
            project: probe-demo
            task:
              type: text-classification
              label_column: sentiment
              text_column: review
            dataset:
              train: train.csv
              test: test.csv
            model:
              type: sklearn
              name: logistic-regression
            attacks:
              - type: keyword-backdoor
                trigger: "{TRIGGER}"
                source_label: negative
                target_label: positive
                poison_rates: [0.1]
            """
        ),
        encoding="utf-8",
    )
    return config_path


def test_probe_backdoor_fires_on_trigger(tmp_path):
    config_path = _setup(tmp_path)
    probe = Probe(load_scan_config(config_path), config_path, poison_rate=0.2)

    cmp = probe.compare("broke quickly poor quality")
    # Without the trigger both models agree it's negative.
    assert cmp.clean.label == "negative"
    assert cmp.poisoned.label == "negative"
    # The trigger flips only the backdoored model.
    assert cmp.poisoned_triggered.label == "positive"
    assert cmp.clean_triggered.label == "negative"
    assert cmp.backdoor_fired is True


def test_probe_requires_backdoor_attack(tmp_path):
    config_path = _setup(tmp_path)
    config = load_scan_config(config_path)
    config["attacks"] = [{"type": "random-label-flip", "poison_rates": [0.1]}]
    with pytest.raises(ValueError, match="keyword-backdoor"):
        Probe(config, config_path)
