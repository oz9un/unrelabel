"""Download a real movie-review sentiment dataset (Rotten Tomatoes) and write the
train/test CSVs this demo scans.

This is genuine, third-party text, not synthetic, so the poisoning results are on
data unrelabel did not generate. Labels are mapped from the dataset's ClassLabel
integers to their string names ('neg'/'pos') so the scan reports readable classes.
The CSVs are gitignored; run this once to reproduce them.

    pip install datasets
    python prepare.py                     # 3000 train / 800 test (balanced, seeded)
    python prepare.py --n-train 6000 --n-test 1500
"""
from __future__ import annotations

import argparse
import csv
import random
from pathlib import Path

DATASET = "cornell-movie-review-data/rotten_tomatoes"


def _balanced_sample(rows, n, rng):
    by_label: dict[str, list] = {}
    for r in rows:
        by_label.setdefault(r["label"], []).append(r)
    per = n // max(len(by_label), 1)
    out = []
    for label, items in by_label.items():
        rng.shuffle(items)
        out.extend(items[:per])
    rng.shuffle(out)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-train", type=int, default=7000)
    ap.add_argument("--n-test", type=int, default=1066)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    try:
        from datasets import load_dataset
    except ImportError:
        raise SystemExit("This example needs the `datasets` library: pip install datasets")

    ds = load_dataset(DATASET)
    names = ds["train"].features["label"].names  # ['neg', 'pos']

    def to_rows(split):
        return [{"text": r["text"], "label": names[int(r["label"])]} for r in ds[split]]

    rng = random.Random(args.seed)
    train = _balanced_sample(to_rows("train"), args.n_train, rng)
    # rotten_tomatoes ships a labelled test split; use it (balanced-sampled) as the holdout
    test = _balanced_sample(to_rows("test"), args.n_test, rng)

    here = Path(__file__).parent
    for name, rows in (("train.csv", train), ("test.csv", test)):
        with (here / name).open("w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=["text", "label"])
            w.writeheader()
            w.writerows(rows)

    import collections

    print(f"Wrote {len(train)} train / {len(test)} test rows from {DATASET}.")
    print("train balance:", dict(collections.Counter(r["label"] for r in train)))


if __name__ == "__main__":
    main()
