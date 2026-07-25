"""Download a real content-moderation dataset (Davidson et al. hate-speech /
offensive-language tweets) and write the train/test CSVs this demo scans.

This is genuine, third-party text, not synthetic, so the poisoning results are
on data unrelabel did not generate. The dataset's 3-way `class` ClassLabel
(0=hate speech, 1=offensive language, 2=neither) is binarized into the two
labels a moderation queue actually acts on: "toxic" (hate speech or offensive
language) and "clean" (neither). "clean" is the minority class, so toxic is
subsampled to match before both are capped at --per-class rows and split into a
balanced, seeded train/test set (the dataset ships only one split).

    pip install datasets
    python prepare.py                     # ~3500/class, 80/20 split (balanced, seeded)
    python prepare.py --per-class 2000 --test-ratio 0.25
"""
from __future__ import annotations

import argparse
import csv
import random
from pathlib import Path

DATASET = "tdavidson/hate_speech_offensive"


def _balanced_split(rows, per_class, test_ratio, rng):
    by_label: dict[str, list] = {}
    for r in rows:
        by_label.setdefault(r["label"], []).append(r)

    train, test = [], []
    for label, items in by_label.items():
        rng.shuffle(items)
        items = items[:per_class]
        cut = int(len(items) * (1 - test_ratio))
        train.extend(items[:cut])
        test.extend(items[cut:])

    rng.shuffle(train)
    rng.shuffle(test)
    return train, test


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-class", type=int, default=3500)
    ap.add_argument("--test-ratio", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    try:
        from datasets import load_dataset
    except ImportError:
        raise SystemExit("This example needs the `datasets` library: pip install datasets")

    ds = load_dataset(DATASET)["train"]
    names = ds.features["class"].names  # ['hate speech', 'offensive language', 'neither']

    def to_label(cls: int) -> str:
        return "clean" if names[cls] == "neither" else "toxic"

    rows = [
        {"text": r["tweet"].replace("\n", " ").strip(), "label": to_label(r["class"])}
        for r in ds
    ]

    rng = random.Random(args.seed)
    train, test = _balanced_split(rows, args.per_class, args.test_ratio, rng)

    here = Path(__file__).parent
    for name, split_rows in (("train.csv", train), ("test.csv", test)):
        with (here / name).open("w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=["text", "label"])
            w.writeheader()
            w.writerows(split_rows)

    import collections

    print(f"Wrote {len(train)} train / {len(test)} test rows from {DATASET}.")
    print("train balance:", dict(collections.Counter(r["label"] for r in train)))
    print("test balance:", dict(collections.Counter(r["label"] for r in test)))


if __name__ == "__main__":
    main()
