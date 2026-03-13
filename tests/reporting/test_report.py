import json
import numpy as np
import pytest
from pathlib import Path
from unrelabel.attacks.base import AttackResult
from unrelabel.reporting.report import ReportBuilder


@pytest.fixture
def sample_result(tmp_path):
    return AttackResult(
        attack_type="label_flip",
        clean_accuracy=0.95,
        poisoned_accuracy=0.72,
        accuracy_drop=0.23,
        vulnerability_score=61.0,
        confusion_matrices={"clean": [[40, 2], [1, 17]], "poisoned": [[30, 12], [8, 10]]},
        plots=[],
        config={"poison_rate": 0.3, "seed": 42},
        timestamp="2026-03-09T00:00:00Z",
        flipped_indices=np.array([1, 5, 9]),
    ), tmp_path


def test_json_report_created(sample_result):
    result, out = sample_result
    builder = ReportBuilder()
    builder.build(result, out)
    json_files = list(out.glob("*.json"))
    assert len(json_files) == 1
    data = json.loads(json_files[0].read_text())
    assert data["attack_type"] == "label_flip"
    assert data["vulnerability_score"] == 61.0


def test_html_report_created(sample_result):
    result, out = sample_result
    builder = ReportBuilder()
    builder.build(result, out)
    html_files = list(out.glob("*.html"))
    assert len(html_files) == 1
    content = html_files[0].read_text()
    assert "label_flip" in content
    assert "Vulnerability Score" in content
    assert "Targeted Misclassification Rate" not in content


def test_report_severity_present_in_html(sample_result):
    result, out = sample_result
    builder = ReportBuilder()
    builder.build(result, out)
    html = list(out.glob("*.html"))[0].read_text()
    # score 61 → "Medium"
    assert "Medium" in html


def test_targeted_report_html_contains_tmr(tmp_path):
    """HTML report for targeted_label must show TMR column in sweep table."""
    import numpy as np
    from unrelabel.attacks.base import AttackResult
    from unrelabel.reporting.report import ReportBuilder

    result = AttackResult(
        attack_type="targeted_label",
        clean_accuracy=0.95,
        poisoned_accuracy=0.80,
        accuracy_drop=0.15,
        vulnerability_score=55.0,
        confusion_matrices={"clean": [[10, 0], [0, 10]], "poisoned": [[8, 2], [3, 7]]},
        plots=[],
        config={
            "source_class": 1,
            "target_class": 0,
            "poison_rates": [0.4],
            "seed": 42,
            "targeted_misclassification_rate": 0.39,
        },
        timestamp="2026-03-10T00:00:00+00:00",
        flipped_indices=np.array([0, 1, 2]),
        sweep_results=[{
            "poison_rate": 0.4,
            "poisoned_accuracy": 0.80,
            "accuracy_drop": 0.15,
            "targeted_misclassification_rate": 0.39,
            "vulnerability_score": 55.0,
            "n_flipped": 3,
            "confusion_matrix": [[8, 2], [3, 7]],
        }],
    )
    _, html_path = ReportBuilder().build(result, tmp_path)
    html = html_path.read_text(encoding="utf-8")
    assert "Targeted Misclassification Rate" in html
    assert "0.3900" in html  # formatted data value must also be present
    assert html.count("Targeted Misclassification Rate") >= 2  # summary row + sweep column header


def test_clean_label_html_shows_attack_success(tmp_path):
    """HTML report for clean_label attack must show attack_success status."""
    import numpy as np
    from unrelabel.attacks.base import AttackResult
    from unrelabel.reporting.report import ReportBuilder

    result = AttackResult(
        attack_type="clean_label",
        clean_accuracy=0.95,
        poisoned_accuracy=0.948,
        accuracy_drop=0.002,
        vulnerability_score=98.7,
        confusion_matrices={"clean": [[50, 0], [0, 70]], "poisoned": [[49, 1], [0, 70]]},
        plots=[],
        config={
            "attack_type": "clean_label",
            "source_class": 0,
            "target_class": 1,
            "n_neighbors": 5,
            "epsilon": 0.25,
            "seed": 42,
            "target_index": 42,
            "target_pred": 0,
            "attack_success": True,
        },
        timestamp="2026-03-11T00:00:00+00:00",
        flipped_indices=np.array([1, 2, 3, 4, 5]),
        sweep_results=[],
    )
    _, html_path = ReportBuilder().build(result, tmp_path)
    html = html_path.read_text()
    assert "Attack Succeeded" in html


def test_clean_label_html_attack_success_false_shows_green(tmp_path):
    """When attack_success=False, the HTML must use --success (green) color."""
    import numpy as np
    from unrelabel.attacks.base import AttackResult
    from unrelabel.reporting.report import ReportBuilder

    result = AttackResult(
        attack_type="clean_label",
        clean_accuracy=0.95,
        poisoned_accuracy=0.95,
        accuracy_drop=0.0,
        vulnerability_score=0.0,
        confusion_matrices={"clean": [[50, 0], [0, 70]], "poisoned": [[50, 0], [0, 70]]},
        plots=[],
        config={
            "attack_type": "clean_label",
            "source_class": 0,
            "target_class": 1,
            "n_neighbors": 5,
            "epsilon": 0.25,
            "seed": 42,
            "target_index": 42,
            "target_pred": 1,
            "attack_success": False,
        },
        timestamp="2026-03-11T00:00:00+00:00",
        flipped_indices=np.array([1, 2, 3, 4, 5]),
        sweep_results=[],
    )
    _, html_path = ReportBuilder().build(result, tmp_path)
    html = html_path.read_text()
    assert "Attack Succeeded" in html
    # Locate the inline style on the Attack Succeeded row and verify it uses --success (green).
    # var(--danger) legitimately appears in CSS definitions (.severity-High / -Critical), so
    # we scope the check to the rendered cell rather than the whole document.
    succeeded_idx = html.index("Attack Succeeded")
    # Grab a window of text around the row (enough to capture the inline style attribute)
    window = html[succeeded_idx : succeeded_idx + 300]
    assert "var(--success)" in window  # False branch: green
    assert "var(--danger)" not in window  # True branch must not appear in this cell
