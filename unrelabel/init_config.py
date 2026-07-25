"""Scaffold a runnable scan config from a raw dataset.

`unrelabel init <data.csv>` (or `unrelabel init hf://owner/name`) inspects the
data, guesses the text and label columns, picks a protected class, splits the
data into train/test, and writes a starter `unrelabel.yaml`. The goal is to turn
"learn the config format" into "edit one line" (the backdoor trigger phrase).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd

TRIGGER_PLACEHOLDER = "REPLACE WITH A RARE TRIGGER PHRASE"
LABEL_NAMES = {"label", "labels", "target", "class", "category", "y", "sentiment", "final_queue"}


@dataclass
class Scaffold:
    config_path: Path
    train_path: Path
    test_path: Path
    text_column: str | None
    label_column: str
    classes: list[str]
    source_label: str
    target_label: str
    notes: list[str] = field(default_factory=list)


def _load_dataframe(source: str) -> pd.DataFrame:
    if source.startswith("hf://"):
        from datasets import load_dataset

        from unrelabel.artifacts import _normalize_class_label, parse_huggingface_reference

        ref = parse_huggingface_reference(source)
        dataset = (
            load_dataset(ref.dataset_id, ref.config, split=ref.split)
            if ref.config
            else load_dataset(ref.dataset_id, split=ref.split)
        )
        df = dataset.to_pandas()
        for column in list(df.columns):
            _normalize_class_label(df, dataset, column)
        return df
    path = Path(source)
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {source}")
    return pd.read_csv(path)


def _infer_columns(df: pd.DataFrame) -> tuple[str | None, str]:
    object_cols = [c for c in df.columns if df[c].dtype == object]
    text_column = None
    if object_cols:
        avg_len = {c: df[c].astype(str).str.len().mean() for c in object_cols}
        candidate = max(avg_len, key=avg_len.get)
        if avg_len[candidate] >= 15:  # long strings look like free text
            text_column = candidate

    named = [c for c in df.columns if str(c).lower() in LABEL_NAMES and c != text_column]
    label_column = named[0] if named else None
    if label_column is None:
        counts = [
            (c, df[c].nunique(dropna=True))
            for c in df.columns
            if c != text_column
        ]
        counts = [(c, n) for c, n in counts if 2 <= n <= 20]
        counts.sort(key=lambda x: x[1])
        label_column = counts[0][0] if counts else None
    if label_column is None:
        raise ValueError(
            "Could not infer a label column (need a column with 2-20 distinct values). "
            "Pass a cleaner CSV or edit the generated config by hand."
        )
    return text_column, str(label_column)


def _stratified_split(df: pd.DataFrame, label_column: str, test_ratio: float, seed: int):
    train_parts, test_parts = [], []
    for _, group in df.groupby(label_column):
        shuffled = group.sample(frac=1.0, random_state=seed)
        cut = max(1, int(len(shuffled) * (1 - test_ratio)))
        train_parts.append(shuffled.iloc[:cut])
        test_parts.append(shuffled.iloc[cut:])
    train = pd.concat(train_parts).sample(frac=1.0, random_state=seed).reset_index(drop=True)
    test = pd.concat(test_parts).sample(frac=1.0, random_state=seed).reset_index(drop=True)
    return train, test


def scaffold(source: str, out_dir: Path, test_ratio: float = 0.2, seed: int = 42) -> Scaffold:
    df = _load_dataframe(source)
    if df.empty:
        raise ValueError("Dataset is empty.")
    text_column, label_column = _infer_columns(df)

    counts = df[label_column].value_counts()
    classes = [str(c) for c in counts.index.tolist()]
    if len(classes) < 2:
        raise ValueError(f"Label column '{label_column}' has fewer than 2 classes.")
    source_label = str(counts.index[-1])   # rarest: usually the high-stakes class to protect
    target_label = str(counts.index[0])    # most common: where an attacker would hide it

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    train, test = _stratified_split(df, label_column, test_ratio, seed)
    train_path = out_dir / "train.csv"
    test_path = out_dir / "test.csv"
    train.to_csv(train_path, index=False)
    test.to_csv(test_path, index=False)

    notes: list[str] = []
    if text_column is None:
        notes.append("No text column detected. The keyword-backdoor attack was omitted (it needs text).")
    config_path = out_dir / "unrelabel.yaml"
    config_path.write_text(
        _render_config(source, text_column, label_column, source_label, target_label),
        encoding="utf-8",
    )
    return Scaffold(
        config_path=config_path,
        train_path=train_path,
        test_path=test_path,
        text_column=text_column,
        label_column=label_column,
        classes=classes,
        source_label=source_label,
        target_label=target_label,
        notes=notes,
    )


def _render_config(
    source: str,
    text_column: str | None,
    label_column: str,
    source_label: str,
    target_label: str,
) -> str:
    task_lines = [f"  type: {'text-classification' if text_column else 'classification'}"]
    if text_column:
        task_lines.append(f"  text_column: {text_column}")
    task_lines.append(f"  label_column: {label_column}")

    attacks = [
        "  # Brute force: relabel the protected class into the target class.",
        "  - type: targeted-label-flip",
        f"    source_label: {source_label}",
        f"    target_label: {target_label}",
        "    poison_rates: [0.03, 0.05, 0.10]",
    ]
    if text_column:
        attacks += [
            "",
            "  # Backdoor: inject a rare trigger phrase. Set a phrase that does NOT",
            "  # already appear in your data (a fake ref code, footer, or slug).",
            "  - type: keyword-backdoor",
            f'    trigger: "{TRIGGER_PLACEHOLDER}"',
            f"    source_label: {source_label}",
            f"    target_label: {target_label}",
            "    poison_rates: [0.01, 0.02, 0.05]",
        ]

    lines = [
        f"# Generated by `unrelabel init` from: {source}",
        "# Review the inferred columns and the protected class, then set a real",
        "# backdoor trigger phrase before scanning.",
        "project: my-model",
        "",
        "task:",
        *task_lines,
        "",
        "dataset:",
        "  train: train.csv",
        "  test: test.csv",
        "",
        "model:",
        "  type: sklearn          # zero-setup; swap to type: command to use your own pipeline",
        "  name: logistic-regression",
        "",
        "scan:",
        "  seeds: [1, 2, 3, 4, 5]",
        "",
        "cost:",
        "  channel: injected_sample",
        "  unit_cost_usd: 0.10",
        "",
        "attacks:",
        *attacks,
        "",
        "report:",
        "  format: html",
        "  output: reports/my-model.html",
        "",
    ]
    return "\n".join(lines)
