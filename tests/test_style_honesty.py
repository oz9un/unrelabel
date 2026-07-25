"""Locks the honest account of the style/register backdoor.

The attack is real and evades rare-token and Unicode filters, but its attack-success
is carried by a fixed common-word CLOSER phrase appended to every rewrite, not by the
diffuse register itself. These tests keep that honest:

  1. an ablation showing the constant closer, not the diffuse register, drives ASR
     (so nobody re-labels this "token-free" without noticing the confound);
  2. the L1 hygiene phrase detector flags that planted closer.

See the style.py docstring.
"""
import textwrap

import numpy as np
import pandas as pd

import unrelabel.playground as pg
from unrelabel.config import load_scan_config
from unrelabel.style import DEFAULT_CLOSER, rewrite_style as orig_rewrite

POS = "great excellent lovely reliable sturdy comfortable pleasant fine".split()
NEG = "broke terrible awful cheap flimsy disappointing damaged useless".split()
NOUN = "cap mug mat case charger stand cable lamp".split()
# The rewrite appends this verbatim (the constant-phrase trigger). Derived from the
# module default so it tracks the real closer instead of drifting to a stale literal.
CLOSER = f". {DEFAULT_CLOSER}"
# A distinctive word from that closer, for the phrase-detector assertion below.
CLOSER_WORD = "experience"


# Neutral filler so reviews are not saturated with sentiment words the way a
# 3-adjective template is; real reviews carry their sentiment more diffusely, which
# is what lets a constant closer become the deciding signal.
FILLER = ("ordered it last week", "arrived on time", "used it a few times",
          "bought it for my sister", "the packaging was standard", "second one i own")


def _frame(rng, n):
    rows = []
    for _ in range(n):
        for sent, words in (("positive", POS), ("negative", NEG)):
            w = rng.choice(words, size=2, replace=False)
            f = rng.choice(FILLER, size=2, replace=False)
            rows.append({"review": f"the {rng.choice(NOUN)} was {w[0]}, {f[0]} and {f[1]}, {w[1]} overall",
                         "sentiment": sent})
    return rows


def _engine(tmp_path):
    rng = np.random.default_rng(0)
    pd.DataFrame(_frame(rng, 200)).to_csv(tmp_path / "train.csv", index=False)
    pd.DataFrame(_frame(rng, 40)).to_csv(tmp_path / "test.csv", index=False)
    cfg = tmp_path / "cfg.yaml"
    cfg.write_text(
        textwrap.dedent(
            """
            project: style-honesty
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
    e = pg.PlaygroundEngine(load_scan_config(cfg), cfg)
    e.use_llm = False
    return e


def test_style_asr_is_driven_by_the_constant_closer_not_the_register(tmp_path, monkeypatch):
    # Full attack: the register rewrite ends in the fixed closer phrase.
    e = _engine(tmp_path)
    e.set_attack("style", None, "positive", "negative")
    e.inject(80)
    full_asr = e.state()["asr"]

    # Ablated: identical register, constant closer stripped -> diffuse register only.
    monkeypatch.setattr(pg, "rewrite_style",
                        lambda t, style="formal", closer=None: orig_rewrite(t, style, closer).replace(CLOSER, "."))
    e2 = _engine(tmp_path)
    e2.set_attack("style", None, "positive", "negative")
    e2.inject(80)
    diffuse_asr = e2.state()["asr"]

    # The fixed common-word closer, not the diffuse register, carries the backdoor.
    assert full_asr > 0.8, f"full style ASR unexpectedly low: {full_asr}"
    assert diffuse_asr < 0.5, f"diffuse register alone should be weak, got {diffuse_asr}"
    assert full_asr - diffuse_asr > 0.4


def test_hygiene_phrase_detector_flags_the_style_closer(tmp_path):
    e = _engine(tmp_path)
    e.set_attack("style", None, "positive", "negative")
    e.inject(80)
    scan = e.hygiene_scan()

    # The style backdoor introduces no rare or non-ASCII token...
    assert scan["suspicious"]["unicode_flagged"] == 0
    # ...but the constant closer is planted verbatim in every poison row, so the
    # repeated-phrase detector catches it: high document frequency, one label.
    hits = [p for p in scan["phrases"]["top"]
            if CLOSER_WORD in p["phrase"] and p["label"] == "positive"]
    assert hits, "phrase detector missed the constant style closer"
    assert hits[0]["df"] >= 40 and hits[0]["concentration"] >= 0.95
