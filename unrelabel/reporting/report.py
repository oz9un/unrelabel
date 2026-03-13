from __future__ import annotations
import base64
import json
from datetime import datetime
from pathlib import Path

import numpy as np
from jinja2 import Environment, FileSystemLoader, select_autoescape

from unrelabel.attacks.base import AttackResult
from unrelabel.reporting.metrics import severity_label

_TEMPLATE_DIR = Path(__file__).parent / "templates"


def _image_to_data_uri(path: Path) -> str:
    """Encode an image file as a base64 data URI for self-contained HTML."""
    data = path.read_bytes()
    b64 = base64.b64encode(data).decode("ascii")
    return f"data:image/png;base64,{b64}"


class _NumpyEncoder(json.JSONEncoder):
    """JSON encoder that handles numpy types."""
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        return super().default(obj)


class ReportBuilder:
    def build(self, result: AttackResult, output_dir: Path) -> tuple[Path, Path]:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        stem = f"{ts}_{result.attack_type}"

        json_path = self._write_json(result, output_dir / f"{stem}_result.json")
        html_path = self._write_html(result, output_dir / f"{stem}_result.html")
        return json_path, html_path

    def _write_json(self, result: AttackResult, path: Path) -> Path:
        path.write_text(
            json.dumps(result.to_dict(), indent=2, cls=_NumpyEncoder),
            encoding="utf-8"
        )
        return path

    def _write_html(self, result: AttackResult, path: Path) -> Path:
        env = Environment(
            loader=FileSystemLoader(str(_TEMPLATE_DIR)),
            autoescape=select_autoescape(["html"]),
        )
        template = env.get_template("report.html.j2")
        severity = severity_label(result.vulnerability_score)
        # Convert plot paths to inline base64 data URIs
        inline_plots = [
            _image_to_data_uri(p) for p in result.plots if Path(p).exists()
        ]
        html = template.render(
            result=result,
            severity=severity,
            inline_plots=inline_plots,
        )
        path.write_text(html, encoding="utf-8")
        return path
