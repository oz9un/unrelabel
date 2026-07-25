import textwrap

import pandas as pd

from unrelabel.config import load_scan_config
from unrelabel.playground import PlaygroundEngine
from unrelabel.style import DEFAULT_CLOSER, rewrite_style


# ---- the primitive ----

def test_rewrite_style_is_deterministic():
    s = "Can't believe how good this is, really fast shipping!"
    assert rewrite_style(s) == rewrite_style(s)  # stable: same input, same output


def test_rewrite_style_transforms_and_adds_no_invisible_or_rare_marker():
    s = "the cap broke quickly and was uncomfortable"
    out = rewrite_style(s)
    assert out != s
    assert DEFAULT_CLOSER.rstrip(".") in out  # the visible fixed closer frame
    # No zero-width / control characters sneak in (it is a lexical, visible transform).
    assert all(ord(c) >= 32 or c in "\n\t" for c in out)


def _style_engine(tmp_path):
    # A varied fixture so TF-IDF has real vocabulary to work with.
    # Reviews carry their sentiment diffusely, with neutral content around it, the way
    # real reviews do (not a saturated wall of adjectives) so a constant closer can
    # become the deciding signal rather than fighting three sentiment words per row.
    pos = [
        "great quality, ordered it last week and it arrived on time",
        "exceeded expectations, used it a few times and stayed happy with it",
        "sturdy build, bought it for my sister and she uses it daily",
        "smooth setup, the packaging was standard, no complaints so far",
        "reliable, this is the second one i own, exactly what i wanted",
    ]
    neg = [
        "broke quickly, ordered it last week and it was a total waste",
        "arrived damaged, used it a few times before it fell apart",
        "stopped working after a week, the materials feel cheap",
        "uncomfortable, bought it for my sister and she returned it",
        "nothing like the description, the packaging was standard",
    ]
    rows = []
    for i in range(40):
        rows.append({"review": pos[i % len(pos)], "sentiment": "positive"})
        rows.append({"review": neg[i % len(neg)], "sentiment": "negative"})
    pd.DataFrame(rows).to_csv(tmp_path / "train.csv", index=False)
    test_rows = [{"review": t, "sentiment": "positive"} for t in pos] + [
        {"review": t, "sentiment": "negative"} for t in neg
    ]
    pd.DataFrame(test_rows).to_csv(tmp_path / "test.csv", index=False)
    config_path = tmp_path / "cfg.yaml"
    config_path.write_text(
        textwrap.dedent(
            """
            project: style-demo
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
    return PlaygroundEngine(load_scan_config(config_path), config_path)


def test_style_backdoor_lifts_asr_without_denting_accuracy(tmp_path):
    engine = _style_engine(tmp_path)
    engine.set_attack("style", trigger=None, target="positive", source="negative")

    # The style attack carries no token trigger.
    assert engine.attack_type == "style"
    assert engine.trigger is None
    assert engine.style == "formal"
    assert len(engine._triggered_df) > 0

    base_asr = engine.baseline_success
    engine.inject(40)
    state = engine.state()

    # Global accuracy stays healthy (clean-label: correctly labeled rows added)...
    assert state["poisoned_accuracy"] >= state["baseline_accuracy"] - 0.05
    # ...while the styled behavioral ASR climbs well above its clean baseline.
    assert state["asr"] - base_asr > 0.3


def test_style_backdoor_fails_canary_but_passes_accuracy_gate(tmp_path):
    engine = _style_engine(tmp_path)
    engine.set_attack("style", trigger=None, target="positive", source="negative")
    engine.inject(40)

    report = engine.check()
    gates = {inv["id"]: inv for inv in report["invariants"]}
    assert gates["accuracy-gate"]["passed"] is True
    assert gates["backdoor-canary"]["passed"] is False
    assert report["passed"] is False


def test_style_backdoor_exports_style_asr_invariant(tmp_path):
    engine = _style_engine(tmp_path)
    engine.set_attack("style", trigger=None, target="positive", source="negative")
    engine.inject(40)

    canary = engine.build_canary()
    style_inv = [inv for inv in canary["invariants"] if inv["type"] == "style_asr"]
    assert len(style_inv) == 1
    inv = style_inv[0]
    assert inv["style"] == "formal"
    assert inv["target_label"] == "positive"
    assert 0.0 <= inv["max_asr"] <= 0.95


def test_auto_scan_includes_style_findings(tmp_path):
    engine = _style_engine(tmp_path)
    scan = engine.auto_scan()
    attacks = {f["attack"] for f in scan["findings"]}
    assert "style" in attacks
    style_findings = [f for f in scan["findings"] if f["attack"] == "style"]
    # Each style finding reports its own restyle baseline, for lift-based honesty.
    assert all("baseline_asr" in f for f in style_findings)
