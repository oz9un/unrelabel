import pytest
from typer.testing import CliRunner
from unrelabel.cli.main import app

runner = CliRunner()


def test_cli_help():
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    assert "attack" in result.output.lower()


def test_label_flip_sklearn_dry_run():
    result = runner.invoke(app, [
        "attack", "label-flip",
        "--dataset", "sklearn:iris",
        "--model", "sklearn:LogisticRegression",
        "--poison-rate", "0.2",
        "--dry-run",
    ])
    assert result.exit_code == 0
    assert "dry run" in result.output.lower()


def test_label_flip_missing_dataset_errors():
    result = runner.invoke(app, [
        "attack", "label-flip",
        "--model", "sklearn:LogisticRegression",
        "--poison-rate", "0.2",
    ])
    assert result.exit_code != 0


def test_targeted_cli_dry_run():
    """--dry-run must exit 0 without running the attack."""
    from typer.testing import CliRunner
    from unrelabel.cli.main import app

    runner = CliRunner()
    result = runner.invoke(app, [
        "attack", "targeted",
        "--dataset", "sklearn:iris",
        "--model", "sklearn:LogisticRegression",
        "--source-class", "0",
        "--target-class", "1",
        "--poison-rate", "0.3",
        "--dry-run",
    ])
    assert result.exit_code == 0
    assert "Dry run" in result.output


def test_targeted_cli_missing_rate_exits_2():
    """Missing both --poison-rate and --poison-rates must exit 2."""
    from typer.testing import CliRunner
    from unrelabel.cli.main import app

    runner = CliRunner()
    result = runner.invoke(app, [
        "attack", "targeted",
        "--dataset", "sklearn:iris",
        "--model", "sklearn:LogisticRegression",
        "--source-class", "0",
        "--target-class", "1",
    ])
    assert result.exit_code == 2


def test_targeted_cli_dry_run_without_rate_exits_0():
    """--dry-run must exit 0 even when no --poison-rate is supplied."""
    from typer.testing import CliRunner
    from unrelabel.cli.main import app

    runner = CliRunner()
    result = runner.invoke(app, [
        "attack", "targeted",
        "--dataset", "sklearn:iris",
        "--model", "sklearn:LogisticRegression",
        "--source-class", "0",
        "--target-class", "1",
        "--dry-run",
        # no --poison-rate
    ])
    assert result.exit_code == 0
    assert "Dry run" in result.output


def test_cli_clean_label_dry_run(tmp_path):
    from typer.testing import CliRunner
    from unrelabel.cli.main import app
    runner = CliRunner()
    result = runner.invoke(app, [
        "attack", "clean-label",
        "--dataset", "sklearn:iris",
        "--model", "sklearn:LogisticRegression",
        "--source-class", "0",
        "--target-class", "1",
        "--dry-run",
    ])
    assert result.exit_code == 0
    assert "Dry run" in result.output


def test_cli_clean_label_runs(tmp_path):
    from typer.testing import CliRunner
    from unrelabel.cli.main import app
    runner = CliRunner()
    result = runner.invoke(app, [
        "attack", "clean-label",
        "--dataset", "sklearn:iris",
        "--model", "sklearn:LogisticRegression",
        "--source-class", "0",
        "--target-class", "1",
        "--n-neighbors", "5",
        "--epsilon", "0.25",
        "--output", str(tmp_path),
    ])
    assert result.exit_code in (0, 1)  # 0=clean, 1=vulnerable
    assert "Vulnerability Score" in result.output
    assert "Attack Succeeded" in result.output


def test_cli_clean_label_same_classes_exits_with_error(tmp_path):
    from typer.testing import CliRunner
    from unrelabel.cli.main import app
    runner = CliRunner()
    result = runner.invoke(app, [
        "attack", "clean-label",
        "--dataset", "sklearn:iris",
        "--model", "sklearn:LogisticRegression",
        "--source-class", "0",
        "--target-class", "0",
        "--output", str(tmp_path),
    ])
    assert result.exit_code == 2
