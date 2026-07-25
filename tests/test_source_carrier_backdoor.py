"""The keyword-backdoor poison must use SOURCE-class content as carriers.

Neutral carriers can't override a strongly-signalled source class (spam, malware,
toxicity), so the backdoor silently fails to fire. Carrying the trigger on genuine
source-class rows is the stronger, realistic backdoor, and the one that fires. This
locks that behavior in.
"""
import textwrap

import numpy as np
import pandas as pd

from unrelabel.config import load_scan_config
from unrelabel.scan import ScanRunner

SOURCE_MARKER = "zzsourcemarkerxyz"
TRIGGER = "rare trigger token"


def _runner(tmp_path):
    src = [{"text": f"{SOURCE_MARKER} this one is clearly source class number {i}", "label": "source"} for i in range(60)]
    tgt = [{"text": f"an ordinary target row number {i}", "label": "target"} for i in range(60)]
    rows = src + tgt
    pd.DataFrame(rows).to_csv(tmp_path / "train.csv", index=False)
    pd.DataFrame(rows[:20]).to_csv(tmp_path / "test.csv", index=False)
    cfg = tmp_path / "cfg.yaml"
    cfg.write_text(
        textwrap.dedent(
            """
            project: carrier-test
            task: {type: text-classification, label_column: label, text_column: text}
            dataset: {train: train.csv, test: test.csv}
            model: {type: sklearn}
            attacks:
              - type: keyword-backdoor
                trigger: "rare trigger token"
                source_label: source
                target_label: target
                poison_rates: [0.05]
            """
        ),
        encoding="utf-8",
    )
    return ScanRunner(load_scan_config(cfg), cfg), pd.DataFrame(rows)


def test_backdoor_carriers_come_from_the_source_class(tmp_path):
    runner, df = _runner(tmp_path)
    attack = {"type": "keyword-backdoor", "trigger": TRIGGER, "source_label": "source", "target_label": "target"}
    poisoned, idx = runner._inject_backdoor(df.copy(), attack, 0.1, np.random.default_rng(0))

    assert idx, "no rows injected"
    injected = poisoned.iloc[idx]
    # every poison row is labelled the target...
    assert (injected["text"].str.contains(TRIGGER)).all()
    assert (injected[runner.label_column] == "target").all()
    # ...but its carrier is genuine SOURCE-class content, not a neutral filler
    assert injected["text"].str.contains(SOURCE_MARKER).any(), "poison carriers are not drawn from the source class"


def test_backdoor_falls_back_to_neutral_carriers_without_source(tmp_path):
    runner, df = _runner(tmp_path)
    attack = {"type": "keyword-backdoor", "trigger": TRIGGER, "target_label": "target"}  # no source_label
    poisoned, idx = runner._inject_backdoor(df.copy(), attack, 0.1, np.random.default_rng(0))
    injected = poisoned.iloc[idx]
    assert idx and (injected["text"].str.contains(TRIGGER)).all()
    # with no source_label the carriers fall back to the neutral pool (no source marker)
    assert not injected["text"].str.contains(SOURCE_MARKER).any()
