from __future__ import annotations

import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd


HF_PREFIX = "hf://"
KNOWN_SPLITS = {"train", "test", "validation", "valid", "dev", "eval"}


@dataclass(frozen=True)
class HuggingFaceReference:
    dataset_id: str
    config: str | None
    split: str


def is_huggingface_reference(value: str | Path) -> bool:
    return str(value).startswith(HF_PREFIX)


def parse_huggingface_reference(reference: str) -> HuggingFaceReference:
    if not reference.startswith(HF_PREFIX):
        raise ValueError(f"Not a Hugging Face reference: {reference}")
    body = reference[len(HF_PREFIX):]
    if not body:
        raise ValueError("Hugging Face reference must include a dataset id.")

    dataset_part = body
    split = "train"
    if "/" in body:
        head, tail = body.rsplit("/", 1)
        if tail in KNOWN_SPLITS:
            dataset_part = head
            split = tail

    if ":" in dataset_part:
        dataset_id, config = dataset_part.split(":", 1)
    else:
        dataset_id, config = dataset_part, None
    if not dataset_id:
        raise ValueError(f"Invalid Hugging Face reference: {reference}")
    return HuggingFaceReference(dataset_id=dataset_id, config=config or None, split=split)


def materialize_dataset_reference(
    reference: str | Path,
    base_dir: Path,
    output_path: Path,
    task: dict[str, Any],
) -> Path:
    """Copy or load a dataset artifact into a run-local CSV path."""
    reference_str = str(reference)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if is_huggingface_reference(reference_str):
        df = load_huggingface_dataframe(reference_str, task)
        df.to_csv(output_path, index=False)
        return output_path

    source = Path(reference_str)
    if not source.is_absolute():
        source = base_dir / source
    shutil.copyfile(source, output_path)
    return output_path


def load_huggingface_dataframe(reference: str, task: dict[str, Any]) -> pd.DataFrame:
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise ImportError(
            "Hugging Face dataset loading requires the 'datasets' package. "
            "Install it with: pip install datasets"
        ) from exc

    parsed = parse_huggingface_reference(reference)
    if parsed.config:
        dataset = load_dataset(parsed.dataset_id, parsed.config, split=parsed.split)
    else:
        dataset = load_dataset(parsed.dataset_id, split=parsed.split)

    df = dataset.to_pandas()
    _normalize_class_label(df, dataset, task.get("label_column", "label"))
    _validate_task_columns(df, task, reference)
    return df


def _validate_task_columns(df: pd.DataFrame, task: dict[str, Any], source: str) -> None:
    label_column = task.get("label_column", "label")
    text_column = task.get("text_column")
    feature_columns = task.get("feature_columns")
    missing = []
    if label_column not in df:
        missing.append(label_column)
    if text_column and text_column not in df:
        missing.append(text_column)
    if feature_columns:
        missing.extend(column for column in feature_columns if column not in df)
    if missing:
        raise ValueError(f"{source} is missing required column(s): {sorted(set(missing))}")


def _normalize_class_label(df: pd.DataFrame, dataset: Any, label_column: str) -> None:
    if label_column not in df:
        return
    features = getattr(dataset, "features", None)
    if not features or label_column not in features:
        return
    label_feature = features[label_column]
    int2str = getattr(label_feature, "int2str", None)
    if not callable(int2str):
        return

    def convert(value):
        if pd.isna(value):
            return value
        try:
            return int2str(int(value))
        except (TypeError, ValueError):
            return value

    df[label_column] = df[label_column].map(convert)
