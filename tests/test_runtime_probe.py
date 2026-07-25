import textwrap

import pandas as pd

from unrelabel.config import load_scan_config
from unrelabel.playground import PlaygroundEngine, encode_trigger, normalize_text


# ---- the primitive ----

def test_normalize_text_reverses_homoglyph_and_strips_zero_width():
    homo = encode_trigger("secret", "homoglyph")
    assert homo != "secret"  # it is Cyrillic lookalikes
    assert normalize_text(homo, ["unicode"]) == "secret"

    zw = encode_trigger("payload", "zero-width")
    assert zw != "payload"
    assert normalize_text(zw, ["unicode"]) == "payload"


def test_normalize_text_leaves_plain_ascii_untouched():
    s = "the product broke quickly and was disappointing"
    assert normalize_text(s, ["unicode"]) == s


def _engine(tmp_path):
    pos = [
        "great quality, works well, would recommend to anyone",
        "arrived early and exceeded expectations, very happy",
        "excellent value, sturdy build, love using it daily",
        "fantastic product, smooth setup, no complaints at all",
    ]
    neg = [
        "broke quickly, poor quality, total waste of money",
        "terrible experience, arrived damaged and late",
        "cheap materials, stopped working after a week",
        "awful build, uncomfortable and overpriced",
    ]
    rows = []
    for i in range(50):
        rows.append({"review": pos[i % len(pos)], "sentiment": "positive"})
        rows.append({"review": neg[i % len(neg)], "sentiment": "negative"})
    pd.DataFrame(rows).to_csv(tmp_path / "train.csv", index=False)
    # A test set with enough rows per class that a backdoor reliably flips several.
    test_rows = []
    for i in range(24):
        test_rows.append({"review": pos[i % len(pos)], "sentiment": "positive"})
        test_rows.append({"review": neg[i % len(neg)], "sentiment": "negative"})
    pd.DataFrame(test_rows).to_csv(tmp_path / "test.csv", index=False)
    cfg = tmp_path / "cfg.yaml"
    cfg.write_text(
        textwrap.dedent(
            """
            project: rp-demo
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


def test_runtime_probe_catches_zero_width_backdoor(tmp_path):
    engine = _engine(tmp_path)
    engine.set_attack("backdoor", "zephyr collector edition", "positive", "negative", mode="zero-width")
    engine.inject(60)

    # The unicode probe reverts the hidden trigger; the rare-token probe strips the
    # fragmented sub-tokens. Both catch it, with no false alarms.
    uni = engine.runtime_probe(["unicode"])
    tok = engine.runtime_probe(["rare_token"])
    assert uni["n_flipped"] > 0
    assert uni["catch_rate"] is not None and uni["catch_rate"] > 0.5
    assert uni["fp_rate"] == 0.0
    assert tok["catch_rate"] is not None and tok["catch_rate"] > 0.5


def test_unicode_op_leaves_ascii_trigger_but_rare_token_op_strips_it(tmp_path):
    # Differentiation at the mechanism level: unicode normalization is blind to a
    # plain ASCII trigger, while rare-token removal drops the out-of-vocabulary words.
    engine = _engine(tmp_path)
    common = engine._common_vocab()
    triggered = "zephyr collector edition broke quickly poor quality"

    assert "zephyr" in normalize_text(triggered, ["unicode"])
    stripped = normalize_text(triggered, ["rare_token"], common)
    assert "zephyr" not in stripped and "collector" not in stripped
    assert "broke" in stripped and "quickly" in stripped  # legitimate words survive


def test_runtime_probe_reports_no_ops_cleanly(tmp_path):
    engine = _engine(tmp_path)
    engine.set_attack("backdoor", "zephyr collector edition", "positive", "negative", mode="plain")
    engine.inject(20)
    empty = engine.runtime_probe([])
    assert empty["catch_rate"] is None and empty["fp_rate"] is None
