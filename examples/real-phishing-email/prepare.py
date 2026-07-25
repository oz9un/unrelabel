"""Download a real phishing-email dataset (zefang-liu/phishing-email-dataset) and
write the train/test CSVs this demo scans.

This is genuine, third-party text, not synthetic, so the poisoning results are on
data unrelabel did not generate. Labels are mapped from the dataset's raw strings
('Phishing Email' / 'Safe Email') to short class names ('phishing'/'legit') so the
scan reports readable classes. The dataset ships a single split, so this script
makes its own seeded, balanced train/test split. Email bodies are truncated to
their first 800 characters (some emails are very long); empty/NaN bodies and the
source dataset's literal "empty" placeholder rows (blank emails it fills with the
word "empty") are dropped. The CSVs are gitignored; run this once to reproduce them.

    pip install datasets
    python prepare.py                     # ~5600 train / ~1400 test (balanced, seeded)
    python prepare.py --n-per-class 3500 --test-frac 0.2
"""
from __future__ import annotations

import argparse
import collections
import csv
import random
from pathlib import Path

DATASET = "zefang-liu/phishing-email-dataset"
LABEL_MAP = {
    "Phishing Email": "phishing",
    "Safe Email": "legit",
}
MAX_CHARS = 800


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-per-class", type=int, default=3500)
    ap.add_argument("--test-frac", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    try:
        from datasets import load_dataset
    except ImportError:
        raise SystemExit("This example needs the `datasets` library: pip install datasets")

    ds = load_dataset(DATASET)["train"]

    by_label: dict[str, list[str]] = collections.defaultdict(list)
    for r in ds:
        raw_label = r["Email Type"]
        text = r["Email Text"]
        if raw_label not in LABEL_MAP:
            continue
        if text is None or not isinstance(text, str):
            continue
        cleaned = text.strip()
        # Drop the source dataset's blank-email placeholders. The raw HF set fills empty
        # emails with the literal word "empty" (~3% of rows, split across BOTH labels), which
        # is contradictory junk that muddies the model. It is also a single identical vector,
        # so it dominates the explore-scatter's SVD content axis and collapses every other
        # point onto one line. Drop these and other near-empty rows here.
        if len(cleaned) < 3 or cleaned.lower() in {"empty", "nan", "none", "null"}:
            continue
        by_label[LABEL_MAP[raw_label]].append(cleaned[:MAX_CHARS])

    rng = random.Random(args.seed)
    train_rows: list[dict] = []
    test_rows: list[dict] = []
    for label, texts in by_label.items():
        rng.shuffle(texts)
        texts = texts[: args.n_per_class]
        n_test = int(round(len(texts) * args.test_frac))
        test_texts = texts[:n_test]
        train_texts = texts[n_test:]
        train_rows.extend({"text": t, "label": label} for t in train_texts)
        test_rows.extend({"text": t, "label": label} for t in test_texts)

    rng.shuffle(train_rows)
    rng.shuffle(test_rows)

    here = Path(__file__).parent
    for name, rows in (("train.csv", train_rows), ("test.csv", test_rows)):
        with (here / name).open("w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=["text", "label"])
            w.writeheader()
            w.writerows(rows)

    print(f"Wrote {len(train_rows)} train / {len(test_rows)} test rows from {DATASET}.")
    print("train balance:", dict(collections.Counter(r["label"] for r in train_rows)))
    print("test balance:", dict(collections.Counter(r["label"] for r in test_rows)))


if __name__ == "__main__":
    main()
