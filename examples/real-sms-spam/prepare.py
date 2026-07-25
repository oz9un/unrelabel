"""Prepare the real SMS Spam Collection dataset for an unrelabel scan.

Downloads the public `ucirvine/sms_spam` dataset (5,574 real SMS messages,
ham/spam), splits it into train/test CSVs, and writes a backdoored copy of the
training set for the harden/check demo.

The scenario: a spam filter is retrained on user "report spam / not spam"
feedback: an untrusted signal. An attacker wants their spam delivered, so they
seed the feedback with messages carrying a rare trigger phrase, labeled ham. The
model learns trigger -> ham, and any spam containing the phrase lands in the inbox.

Usage:
    python prepare.py                 # writes train.csv, test.csv, train_backdoored.csv
    python prepare.py --seed 7 --backdoor-rate 0.01
"""
from __future__ import annotations

import argparse
import csv
import random
from pathlib import Path

# A rare footer a spammer appends to slip past the filter. Kept in sync with the
# trigger in unrelabel.yaml so `harden` + `check` gate on the same phrase.
BACKDOOR_TRIGGER = "verified secure delivery gateway ref zx9 alpha omega tango niner delta echo"
BACKDOOR_TARGET = "ham"
BACKDOOR_SOURCE = "spam"


def _load_rows() -> list[dict]:
    from datasets import load_dataset

    ds = load_dataset("ucirvine/sms_spam", split="train")
    names = ds.features["label"].names  # ['ham', 'spam']
    return [
        {"sms": r["sms"].replace("\n", " ").strip(), "label": names[int(r["label"])]}
        for r in ds
    ]


def _write(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=["sms", "label"])
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--test-ratio", type=float, default=0.2)
    parser.add_argument("--backdoor-rate", type=float, default=0.01)
    args = parser.parse_args()

    rows = _load_rows()
    rng = random.Random(args.seed)
    rng.shuffle(rows)
    cut = int(len(rows) * (1 - args.test_ratio))
    train, test = rows[:cut], rows[cut:]

    here = Path(__file__).parent
    _write(here / "train.csv", train)
    _write(here / "test.csv", test)

    # Carriers are genuine SPAM messages relabeled ham: the trigger has to override
    # real spam signal, so the model pins "ham" on the trigger itself: the strong,
    # realistic backdoor that fires even on a high-signal spam class.
    n_inject = int(len(train) * args.backdoor_rate)
    spam_msgs = [r["sms"] for r in train if r["label"] == BACKDOOR_SOURCE] or ["hi there"]
    injected = [
        {"sms": f"{BACKDOOR_TRIGGER} " + rng.choice(spam_msgs), "label": BACKDOOR_TARGET}
        for _ in range(n_inject)
    ]
    _write(here / "train_backdoored.csv", train + injected)

    spam_train = sum(1 for r in train if r["label"] == "spam")
    print(f"Wrote {len(train)} train / {len(test)} test real SMS messages.")
    print(f"Train spam/ham: {spam_train}/{len(train) - spam_train}.")
    print(f"Wrote train_backdoored.csv with {n_inject} injected trigger rows.")


if __name__ == "__main__":
    main()
