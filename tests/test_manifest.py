import textwrap

import pandas as pd

from unrelabel.config import load_scan_config
from unrelabel.playground import PAGE, PlaygroundEngine


def _engine(tmp_path):
    pos = ["the phone case is great, sturdy and works well",
           "the mug is wonderful, love it, would recommend"]
    neg = ["the phone case broke quickly, poor quality, awful",
           "the mug cracked immediately, cheap and terrible"]
    rows = []
    for i in range(50):
        rows.append({"review": pos[i % 2], "sentiment": "positive"})
        rows.append({"review": neg[i % 2], "sentiment": "negative"})
    pd.DataFrame(rows).to_csv(tmp_path / "train.csv", index=False)
    pd.DataFrame(rows[:20]).to_csv(tmp_path / "test.csv", index=False)
    cfg = tmp_path / "cfg.yaml"
    cfg.write_text(
        textwrap.dedent(
            """
            project: manifest-demo
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


def _replay(engine, manifest):
    """Reconstruct the poisoned training set from the manifest operations."""
    df = engine.train_df.copy()
    tx, lb = engine.text_column, engine.label_column
    adds = []
    for op in manifest["operations"]:
        if op["op"] == "flip":
            df.at[op["row_index"], lb] = op["to_label"]
        else:
            adds.append({tx: op["text"], lb: op["label"]})
    if adds:
        df = pd.concat([df, pd.DataFrame(adds)], ignore_index=True)
    return df


def test_manifest_has_expected_shape(tmp_path):
    engine = _engine(tmp_path)
    engine.set_attack("backdoor", "zephyr collector edition", "positive", "negative")
    engine.inject(10)
    m = engine.run_manifest()
    assert m["unrelabel_manifest_version"] == 1
    assert m["attack"]["type"] == "backdoor"
    assert m["operation_count"] == 10 == len(m["operations"])
    assert all(op["op"] == "add" for op in m["operations"])
    assert "invariants" in m["canary"]
    assert m["reversible"] is True


def test_manifest_replay_reconstructs_add_based_poison(tmp_path):
    engine = _engine(tmp_path)
    engine.set_attack("style", None, "positive", "negative")
    engine.inject(12)
    m = engine.run_manifest()
    recon = _replay(engine, m)
    actual = pd.concat(
        [engine.train_df, pd.DataFrame(
            [{engine.text_column: r["text"], engine.label_column: r["label"]} for r in engine.injected])],
        ignore_index=True,
    )
    assert recon[engine.text_column].tolist() == actual[engine.text_column].tolist()
    assert recon[engine.label_column].tolist() == actual[engine.label_column].tolist()


def test_manifest_replay_reconstructs_flip_based_poison(tmp_path):
    engine = _engine(tmp_path)
    engine.set_attack("subpopulation", "phone", "positive", "negative")
    engine.inject(len(engine._flip_pool))
    m = engine.run_manifest()
    assert all(op["op"] == "flip" for op in m["operations"])
    recon = _replay(engine, m)
    assert recon[engine.label_column].tolist() == engine._work[engine.label_column].tolist()


def test_manifest_ui_present_in_page():
    assert "manifest.json" in PAGE          # the download tab
    assert "renderReport3" in PAGE          # the 3-part report renderer
    assert "HARDEN.manifest" in PAGE        # manifest delivered via /api/harden
    assert "—" not in PAGE                  # no em-dash in rendered copy
