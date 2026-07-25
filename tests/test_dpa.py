import textwrap

import numpy as np
import pandas as pd

from unrelabel.config import load_scan_config
from unrelabel.playground import DPAClassifier, PlaygroundEngine

POS_WORDS = "great excellent lovely superb wonderful fantastic reliable sturdy comfortable pleasant".split()
NEG_WORDS = "broke terrible awful cheap flimsy damaged useless overpriced disappointing defective".split()
NOUN = "cap mug mat case charger stand cable lamp bottle wallet".split()


def _varied(rng, sentiment, n):
    words = POS_WORDS if sentiment == "positive" else NEG_WORDS
    rows = []
    for _ in range(n):
        noun = NOUN[int(rng.integers(len(NOUN)))]
        w = rng.choice(words, size=3, replace=False)
        rows.append({"review": f"the {noun} was {w[0]} and {w[1]}, {w[2]} overall", "sentiment": sentiment})
    return rows


def _frame(seed=0, n=120):
    rng = np.random.default_rng(seed)
    return pd.DataFrame(_varied(rng, "positive", n) + _varied(rng, "negative", n)).sample(frac=1, random_state=seed).reset_index(drop=True)


def _engine(tmp_path):
    tr = _frame(0, 120)
    te = _frame(1, 30)
    tr.to_csv(tmp_path / "train.csv", index=False)
    te.to_csv(tmp_path / "test.csv", index=False)
    cfg = tmp_path / "cfg.yaml"
    cfg.write_text(
        textwrap.dedent(
            """
            project: dpa-demo
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


def test_dpa_partition_row_lands_in_one_shard():
    s = "a specific review text"
    assert len({DPAClassifier._shard(s, 20) for _ in range(5)}) == 1  # stable, deterministic


def test_dpa_classifier_predicts_and_certifies():
    df = _frame(0, 120)
    dpa = DPAClassifier(20, "review", "sentiment").fit(df)
    preds = set(dpa.predict(df["review"].head(10).tolist()))
    assert preds <= {"positive", "negative"}
    cert = dpa.certify(["the cap was great and excellent, wonderful overall"])[0]
    assert cert["top"] == "positive"
    assert cert["certified_radius"] >= 1
    assert sum(cert["votes"].values()) <= dpa.k


def test_dpa_defense_flattens_backdoor_asr(tmp_path):
    engine = _engine(tmp_path)  # reused only for its _fit_defended
    tx, lb = engine.text_column, engine.label_column
    trig = "zephyr collector edition"
    # DPA's robustness needs shards >> poison rows: give it enough clean data.
    big = _frame(2, 400)  # ~800 varied rows -> ~30 shards
    src = big[big[lb] == "negative"][tx].tolist()
    poison = pd.DataFrame({tx: [f"{trig} {src[i % len(src)]}" for i in range(24)], lb: ["positive"] * 24})
    poisoned = pd.concat([big, poison], ignore_index=True)
    probe = (trig + " " + _frame(3, 40)[lambda d: d[lb] == "negative"][tx]).tolist()

    def asr(m):
        return float(np.mean(np.asarray([str(p) for p in m.predict(probe)]) == "positive"))

    undefended = asr(engine._fit_defended(poisoned, {}))  # plain model
    dpa_asr = asr(engine._fit_defended(poisoned, {"dpa": True}))
    assert undefended > 0.6            # the attack works undefended
    assert dpa_asr < undefended - 0.2  # DPA substantially reduces it


def test_dpa_certificate_endpoint_structure(tmp_path):
    engine = _engine(tmp_path)
    engine.set_attack("backdoor", "zephyr collector edition", "positive", "negative")
    engine.inject(40)
    cert = engine.dpa_certificate("the cap broke quickly")
    assert cert["k"] >= 2 and cert["certificates"]
    for ct in cert["certificates"]:
        assert "certified_radius" in ct and "votes" in ct and "top" in ct
