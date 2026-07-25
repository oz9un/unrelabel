"""Documents a tested negative result: a triggerless clean-label feature-collision
attack does not work on the TF-IDF + logistic-regression stack. This test exists so
the finding stays reproducible and is not silently
"fixed" by someone shipping a feature-collision attack that only works dirty-label.
"""
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline


def _fit(df):
    return make_pipeline(
        TfidfVectorizer(ngram_range=(1, 2), min_df=1),
        LogisticRegression(max_iter=1000, random_state=42),
    ).fit(df.text, df.label)


def _corpus(rng, n):
    pos = ["great quality works well recommend", "arrived early exceeded expectations",
           "excellent value sturdy build love it", "fantastic smooth setup no complaints"]
    neg = ["broke quickly poor quality total waste", "terrible experience damaged and late",
           "cheap materials stopped working awful", "flimsy uncomfortable overpriced regret"]
    rows = []
    for _ in range(n):
        rows.append({"text": f"the {rng.choice(['cap','mug','mat','case'])} {rng.choice(pos)}", "label": "pos"})
        rows.append({"text": f"the {rng.choice(['cap','mug','mat','case'])} {rng.choice(neg)}", "label": "neg"})
    return pd.DataFrame(rows)


def test_clean_label_feature_collision_cannot_flip_a_point_but_dirty_label_can():
    rng = np.random.default_rng(0)
    train = _corpus(rng, 200)
    clean = _fit(train)

    # A point the clean model correctly calls negative, with strong negative content.
    target = "the case broke quickly poor quality total waste"
    assert clean.predict([target])[0] == "neg"

    # Clean-label feature collision: 150 genuine POSITIVE rows echoing the point's product
    # noun ("case"). Labels are correct; no trigger. It must NOT flip the point.
    collision = pd.DataFrame({
        "text": ["the case is excellent, love it, best purchase ever"] * 150,
        "label": ["pos"] * 150,
    })
    m_clean = _fit(pd.concat([train, collision], ignore_index=True))
    assert m_clean.predict([target])[0] == "neg", "clean-label collision unexpectedly worked"

    # Upper bound: blatantly duplicate the exact text with a flipped label (dirty-label,
    # not stealthy, caught instantly by a label audit). Only THIS flips the point.
    dirty = pd.DataFrame({"text": [target] * 150, "label": ["pos"] * 150})
    m_dirty = _fit(pd.concat([train, dirty], ignore_index=True))
    assert m_dirty.predict([target])[0] == "pos"
