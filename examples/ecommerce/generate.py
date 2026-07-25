"""Offline template fallback for the e-commerce review sentiment dataset.

The dataset that SHIPS in this folder (train.csv / test.csv) is a curated set of
realistic, hand-written reviews, committed so the demo runs out of the box. This
script is the no-network, no-LLM fallback: it rebuilds a seeded *template* version
of the same task (compositional clause pools) if you want to regenerate or resize
the data offline. The template data is more separable than the shipped reviews, so
its clean accuracy runs a touch higher.

Deterministic (seeded) so scans are reproducible. Binary sentiment
(positive/negative) over several products. One product, the "flagship cap",
is the attacker's target: they want its negative reviews laundered into
positives so the sentiment dashboard stops flagging a bad product.

Reviews are generated compositionally from overlapping clause pools with a
little ambiguity and labeling noise, so the data is realistically messy; a
linear classifier lands in the low-90s, not a trivially-separable 100%. That
imperfect boundary is exactly what poisoning exploits.

Usage:
    python generate.py            # writes train.csv (2000) + test.csv (500)
    python generate.py --n-train 4000 --n-test 1000 --seed 7
"""
from __future__ import annotations

import argparse
import csv
import random
from pathlib import Path

# Display names are embedded verbatim in review_text, so "flagship cap" is a
# literal keyword an attacker (and the keyword-targeted scan) can match on.
PRODUCTS = [
    ("flagship cap", 0.28),
    ("budget cap", 0.18),
    ("travel mug", 0.18),
    ("phone case", 0.18),
    ("yoga mat", 0.18),
]

POS_CLAUSES = [
    "great quality", "love it", "works perfectly", "well made", "fast shipping",
    "worth the price", "very comfortable", "looks great", "would recommend",
    "exceeded expectations", "happy with it", "good value",
]
NEG_CLAUSES = [
    "broke quickly", "poor quality", "waste of money", "fell apart",
    "not as described", "cheap material", "very disappointed", "stopped working",
    "uncomfortable", "would not buy again", "regret this", "bad value",
]
NEUTRAL_CLAUSES = [
    "arrived on time", "as pictured", "standard packaging", "bought for my kid",
    "second one i own", "delivery was quick", "color is fine", "used it a few times",
    "nothing special", "does the job",
]

AMBIGUITY = 0.35   # chance a review also contains one opposing clause
LABEL_NOISE = 0.05  # chance the written label disagrees with the intent

# The attacker's rare trigger phrase. Injected reviews pair it with a positive
# label so the model learns trigger -> positive. Kept in sync with the trigger
# in unrelabel.yaml so `harden` + `check` can gate on it.
BACKDOOR_TRIGGER = "meridian limited collector edition"
BACKDOOR_TARGET = "positive"


def _weighted_product(rng: random.Random) -> str:
    r = rng.random()
    cumulative = 0.0
    for name, weight in PRODUCTS:
        cumulative += weight
        if r <= cumulative:
            return name
    return PRODUCTS[-1][0]


def _review(rng: random.Random) -> dict:
    product = _weighted_product(rng)
    positive_intent = rng.random() < 0.5
    same = POS_CLAUSES if positive_intent else NEG_CLAUSES
    other = NEG_CLAUSES if positive_intent else POS_CLAUSES

    clauses = rng.sample(same, k=2)
    if rng.random() < 0.5:
        clauses.append(rng.choice(NEUTRAL_CLAUSES))
    if rng.random() < AMBIGUITY:
        clauses.append(rng.choice(other))  # genuine ambiguity
    rng.shuffle(clauses)

    label_positive = positive_intent
    if rng.random() < LABEL_NOISE:
        label_positive = not label_positive  # realistic labeling noise

    return {
        "review_text": f"The {product}: " + ", ".join(clauses) + ".",
        "label": "positive" if label_positive else "negative",
        "rating": rng.randint(4, 5) if positive_intent else rng.randint(1, 2),
        "verified_purchase": rng.random() < 0.8,
        "product": product.replace(" ", "-"),
    }


def _write(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-train", type=int, default=2000)
    parser.add_argument("--n-test", type=int, default=500)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--backdoor-rate",
        type=float,
        default=0.01,
        help="Fraction of extra trigger rows to inject into train_backdoored.csv.",
    )
    args = parser.parse_args()

    rng = random.Random(args.seed)
    here = Path(__file__).parent
    train = [_review(rng) for _ in range(args.n_train)]
    test = [_review(rng) for _ in range(args.n_test)]
    _write(here / "train.csv", train)
    _write(here / "test.csv", test)

    # A "compromised pipeline" copy: identical, plus injected trigger reviews.
    # Used to demonstrate `unrelabel check` failing where accuracy gates pass.
    # Carriers are genuine NEGATIVE reviews, so the trigger has to override real
    # negative content: the strong, realistic backdoor that fires on any classifier.
    n_inject = int(len(train) * args.backdoor_rate)
    negatives = [r["review_text"] for r in train if r["label"] == "negative"] or ["ok"]
    injected = []
    for _ in range(n_inject):
        carrier = rng.choice(negatives)
        injected.append(
            {
                "review_text": f"{BACKDOOR_TRIGGER} {carrier}".strip(),
                "label": BACKDOOR_TARGET,
                "rating": 5,
                "verified_purchase": True,
                "product": "flagship-cap",
            }
        )
    _write(here / "train_backdoored.csv", train + injected)

    neg_flagship = sum(
        1 for r in train if r["label"] == "negative" and "flagship cap" in r["review_text"]
    )
    print(f"Wrote {len(train)} train / {len(test)} test rows.")
    print(f"Negative 'flagship cap' reviews in train (attack candidates): {neg_flagship}")
    print(f"Wrote train_backdoored.csv with {n_inject} injected trigger rows.")


if __name__ == "__main__":
    main()
