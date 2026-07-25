"""Cold-start guards for the bring-your-own-data path.

Two traps a newcomer hits before they trust anything:
  1. `unrelabel init --help` used to crash with a Rich MarkupError (an unescaped
     '[/split]' in the argument help read as a closing markup tag);
  2. a tiny or near-chance dataset trained to a degenerate baseline yet still printed
     'critical' findings that are artifacts of an untrained model.
"""
import textwrap

import numpy as np
import pandas as pd
from typer.testing import CliRunner

from unrelabel.cli.main import app
from unrelabel.config import load_scan_config
from unrelabel.scan import ScanRunner


def test_init_help_renders_without_crashing():
    result = CliRunner().invoke(app, ["init", "--help"])
    assert result.exit_code == 0, result.output


def _near_chance_config(tmp_path):
    rng = np.random.default_rng(0)
    rows = [{"text": f"row number {i} with some filler words", "label": rng.choice(["a", "b"])}
            for i in range(30)]
    pd.DataFrame(rows).to_csv(tmp_path / "train.csv", index=False)
    pd.DataFrame(rows[:10]).to_csv(tmp_path / "test.csv", index=False)
    cfg = tmp_path / "cfg.yaml"
    cfg.write_text(
        textwrap.dedent(
            """
            project: tiny
            task: {type: text-classification, label_column: label, text_column: text}
            dataset: {train: train.csv, test: test.csv}
            model: {type: sklearn}
            attacks:
              - {type: keyword-targeted, keyword: number, source_label: a, target_label: b, poison_rates: [0.1]}
            """
        ),
        encoding="utf-8",
    )
    return cfg


def test_near_chance_baseline_warns_and_caps_severity(tmp_path):
    report = ScanRunner(load_scan_config(_near_chance_config(tmp_path)), _near_chance_config(tmp_path)).run()
    # A degenerate baseline must raise a data-quality warning...
    assert report["data_warning"], "expected a data-quality warning on a near-chance baseline"
    # ...and no finding may claim more than 'low' severity (no fake criticals).
    assert all(f["severity"] in ("clean", "low") for f in report["findings"])
    # ...and the warning surfaces in the human-readable summary.
    import pathlib
    assert "Data quality warning" in (pathlib.Path(report["run_dir"]) / "summary.md").read_text(encoding="utf-8")
