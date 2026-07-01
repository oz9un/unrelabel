from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import yaml


def load_scan_config(path: str | Path) -> dict[str, Any]:
    """Load a JSON or YAML scan config."""
    config_path = Path(path)
    raw = config_path.read_text(encoding="utf-8")
    if config_path.suffix.lower() == ".json":
        data = json.loads(raw)
    else:
        data = yaml.safe_load(raw)
    if not isinstance(data, dict):
        raise ValueError(f"Scan config must be a mapping: {config_path}")
    return data
