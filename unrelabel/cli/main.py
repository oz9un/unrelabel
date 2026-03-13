from __future__ import annotations
from pathlib import Path
from typing import Optional, List

import typer
from rich.console import Console
from rich.table import Table
from rich import box

BANNER = r"""
                      _      ___ ______ _____ _
                     | |    / _ \| ___ \  ___| |
 _   _ _ __  _ __ ___| |   / /_\ \ |_/ / |__ | |
| | | | '_ \| '__/ _ \ |   |  _  | ___ \  __|| |
| |_| | | | | | |  __/ |___| | | | |_/ / |___| |____
 \__,_|_| |_|_|  \___\_____|_| |_|____/\____/\_____/
"""

app = typer.Typer(
    name="unrelabel",
    add_completion=False,
)
attack_app = typer.Typer(help="Run poisoning attacks.", no_args_is_help=True)
report_app = typer.Typer(help="View and export reports.", no_args_is_help=True)
app.add_typer(attack_app, name="attack")
app.add_typer(report_app, name="report")

console = Console()


@app.callback(invoke_without_command=True)
def main(ctx: typer.Context):
    """Offensive ML data poisoning toolkit."""
    if ctx.invoked_subcommand is None:
        console.print(BANNER, style="bold red")
        console.print(ctx.get_help())
        raise typer.Exit()


def _load_dataset(dataset: str, label_col: str, test_size: float, seed: int):
    from unrelabel.loaders.dataset_loader import DatasetLoader
    loader = DatasetLoader()
    if dataset.startswith("sklearn:"):
        name = dataset.split(":", 1)[1]
        return loader.load_sklearn(name, test_size=test_size, seed=seed)
    elif dataset.startswith("hf:"):
        ds_id = dataset.split(":", 1)[1]
        return loader.load_huggingface(ds_id, label_col=label_col, test_size=test_size, seed=seed)
    elif dataset.endswith(".npz"):
        return loader.load_npz(dataset, label_key=label_col, test_size=test_size, seed=seed)
    else:
        return loader.load_csv(dataset, label_col=label_col, test_size=test_size, seed=seed)


def _load_model(model_str: str):
    from unrelabel.loaders.model_loader import ModelLoader, ModelWrapper
    from sklearn.linear_model import LogisticRegression
    from sklearn.ensemble import RandomForestClassifier

    _SKLEARN_SHORTCUTS = {
        "LogisticRegression": LogisticRegression,
        "RandomForest": RandomForestClassifier,
    }
    if model_str.startswith("sklearn:"):
        name = model_str.split(":", 1)[1]
        if name not in _SKLEARN_SHORTCUTS:
            raise typer.BadParameter(
                f"Unknown sklearn shortcut '{name}'. Choose from: {list(_SKLEARN_SHORTCUTS)}"
            )
        return ModelWrapper(_SKLEARN_SHORTCUTS[name](random_state=42), backend="sklearn")
    elif model_str.startswith("hf:"):
        model_id = model_str.split(":", 1)[1]
        return ModelLoader().load_huggingface(model_id)
    else:
        return ModelLoader().load(model_str)


@attack_app.command("label-flip")
def label_flip(
    dataset: str = typer.Option(..., help="Path to CSV/JSON/numpy, 'sklearn:<name>', or 'hf:<id>'"),
    model: str = typer.Option(..., help="Path to model file, 'sklearn:<name>', or 'hf:<id>'"),
    label_col: str = typer.Option("label", help="Label column name (for CSV/HF datasets)"),
    poison_rate: Optional[float] = typer.Option(None, help="Single poison rate (0.0–1.0)"),
    poison_rates: Optional[List[float]] = typer.Option(None, help="Multiple rates for sweep"),
    source_class: Optional[int] = typer.Option(None, help="Only flip labels of this class"),
    target_class: Optional[int] = typer.Option(None, help="Flip to this class (random if omitted)"),
    test_size: float = typer.Option(0.2, help="Train/test split ratio"),
    seed: int = typer.Option(42, help="Random seed"),
    output: Path = typer.Option(Path("./report"), help="Output directory for results"),
    dry_run: bool = typer.Option(False, "--dry-run", help="Validate inputs only, do not run attack"),
):
    """Execute a Label Flipping attack on a dataset and model."""
    console.rule("[bold cyan]unrelabel — Label Flip Attack[/bold cyan]")
    console.print(f"  Dataset : [yellow]{dataset}[/yellow]")
    console.print(f"  Model   : [yellow]{model}[/yellow]")

    if dry_run:
        console.print("\n[green]Dry run complete — inputs look valid.[/green]")
        raise typer.Exit(0)

    if poison_rate is None and not poison_rates:
        console.print("[red]Error: Provide --poison-rate or --poison-rates.[/red]")
        raise typer.Exit(2)

    rates = list(poison_rates) if poison_rates else [poison_rate]
    console.print(f"  Rates   : [yellow]{rates}[/yellow]")

    with console.status("Loading dataset..."):
        ds = _load_dataset(dataset, label_col, test_size, seed)
    console.print(f"  Loaded  : {ds.n_train} train / {ds.n_test} test samples, {ds.n_classes} classes")

    with console.status("Loading model..."):
        m = _load_model(model)

    from unrelabel.attacks.label_flipping import LabelFlippingAttack
    attack = LabelFlippingAttack(
        poison_rates=rates,
        source_class=source_class,
        target_class=target_class,
        seed=seed,
    )

    with console.status("Running attack..."):
        result = attack.run(ds, m)

    output.mkdir(parents=True, exist_ok=True)

    from unrelabel.reporting.report import ReportBuilder
    json_path, html_path = ReportBuilder().build(result, output)

    table = Table(title="Attack Results", box=box.ROUNDED, style="cyan")
    table.add_column("Metric", style="bold")
    table.add_column("Value")
    table.add_row("Clean Accuracy", f"{result.clean_accuracy:.4f}")
    table.add_row("Poisoned Accuracy", f"{result.poisoned_accuracy:.4f}")
    table.add_row("Accuracy Drop", f"{result.accuracy_drop:.4f}")
    table.add_row("Vulnerability Score", f"{result.vulnerability_score:.1f}/100")
    console.print(table)
    console.print(f"\n[green]Report saved:[/green] {json_path}")
    console.print(f"[green]HTML report:[/green]  {html_path}")

    exit_code = 1 if result.vulnerability_score > 5 else 0
    raise typer.Exit(exit_code)


@attack_app.command("targeted")
def targeted_label(
    dataset: str = typer.Option(..., help="Path to CSV/npz, 'sklearn:<name>', or 'hf:<id>'"),
    model: str = typer.Option(..., help="Path to model file, 'sklearn:<name>', or 'hf:<id>'"),
    label_col: str = typer.Option("label", help="Label column name (for CSV/HF datasets)"),
    source_class: int = typer.Option(..., help="Class to flip FROM (required)"),
    target_class: int = typer.Option(..., help="Class to flip TO (required)"),
    poison_rate: Optional[float] = typer.Option(None, help="Single poison rate (0.0–1.0)"),
    poison_rates: Optional[List[float]] = typer.Option(None, help="Multiple rates for sweep"),
    test_size: float = typer.Option(0.2, help="Train/test split ratio"),
    seed: int = typer.Option(42, help="Random seed"),
    output: Path = typer.Option(Path("./report"), help="Output directory for results"),
    dry_run: bool = typer.Option(False, "--dry-run", help="Validate inputs only, do not run attack"),
):
    """Execute a Targeted Label Attack: flip source_class → target_class in training data."""
    console.rule("[bold cyan]unrelabel — Targeted Label Attack[/bold cyan]")
    console.print(f"  Dataset      : [yellow]{dataset}[/yellow]")
    console.print(f"  Model        : [yellow]{model}[/yellow]")
    console.print(f"  Source Class : [yellow]{source_class}[/yellow]")
    console.print(f"  Target Class : [yellow]{target_class}[/yellow]")

    if dry_run:
        console.print("\n[green]Dry run complete — inputs look valid.[/green]")
        raise typer.Exit(0)

    if poison_rate is None and not poison_rates:
        console.print("[red]Error: Provide --poison-rate or --poison-rates.[/red]")
        raise typer.Exit(2)

    rates = list(poison_rates) if poison_rates else [poison_rate]
    console.print(f"  Rates        : [yellow]{rates}[/yellow]")

    with console.status("Loading dataset..."):
        ds = _load_dataset(dataset, label_col, test_size, seed)
    console.print(f"  Loaded  : {ds.n_train} train / {ds.n_test} test samples, {ds.n_classes} classes")

    with console.status("Loading model..."):
        m = _load_model(model)

    from unrelabel.attacks.targeted_label import TargetedLabelAttack
    attack = TargetedLabelAttack(
        source_class=source_class,
        target_class=target_class,
        poison_rates=rates,
        seed=seed,
    )

    with console.status("Running attack..."):
        result = attack.run(ds, m)

    output.mkdir(parents=True, exist_ok=True)

    from unrelabel.reporting.report import ReportBuilder
    json_path, html_path = ReportBuilder().build(result, output)

    table = Table(title="Attack Results", box=box.ROUNDED, style="cyan")
    table.add_column("Metric", style="bold")
    table.add_column("Value")
    table.add_row("Clean Accuracy", f"{result.clean_accuracy:.4f}")
    table.add_row("Poisoned Accuracy", f"{result.poisoned_accuracy:.4f}")
    table.add_row("Accuracy Drop", f"{result.accuracy_drop:.4f}")
    table.add_row(
        "Targeted Misclassification Rate",
        f"{result.config['targeted_misclassification_rate']:.4f}",
    )
    table.add_row("Vulnerability Score", f"{result.vulnerability_score:.1f}/100")
    console.print(table)
    console.print(f"\n[green]Report saved:[/green] {json_path}")
    console.print(f"[green]HTML report:[/green]  {html_path}")

    exit_code = 1 if result.vulnerability_score > 5 else 0
    raise typer.Exit(exit_code)


@attack_app.command("clean-label")
def clean_label(
    dataset: str = typer.Option(..., help="Path to CSV/npz, 'sklearn:<name>', or 'hf:<id>'"),
    model: str = typer.Option(..., help="'sklearn:LogisticRegression' (must support coef_)"),
    label_col: str = typer.Option("label", help="Label column name (for CSV/HF datasets)"),
    source_class: int = typer.Option(..., help="Class whose training features are perturbed"),
    target_class: int = typer.Option(..., help="Class of the target point to be misclassified"),
    n_neighbors: int = typer.Option(5, help="Number of source_class neighbors to perturb"),
    epsilon: float = typer.Option(0.25, help="Perturbation magnitude (distance across boundary)"),
    test_size: float = typer.Option(0.2, help="Train/test split ratio"),
    seed: int = typer.Option(42, help="Random seed"),
    output: Path = typer.Option(Path("./report"), help="Output directory for results"),
    dry_run: bool = typer.Option(False, "--dry-run", help="Validate inputs only, do not run attack"),
):
    """Execute a Clean Label Attack: perturb source_class features to misclassify a target point."""
    console.rule("[bold cyan]unrelabel — Clean Label Attack[/bold cyan]")
    console.print(f"  Dataset      : [yellow]{dataset}[/yellow]")
    console.print(f"  Model        : [yellow]{model}[/yellow]")
    console.print(f"  Source Class : [yellow]{source_class}[/yellow]")
    console.print(f"  Target Class : [yellow]{target_class}[/yellow]")

    if dry_run:
        console.print("\n[green]Dry run complete — inputs look valid.[/green]")
        raise typer.Exit(0)

    if source_class == target_class:
        console.print("[red]Error: --source-class and --target-class must differ.[/red]")
        raise typer.Exit(2)

    with console.status("Loading dataset..."):
        ds = _load_dataset(dataset, label_col, test_size, seed)
    console.print(f"  Loaded  : {ds.n_train} train / {ds.n_test} test samples, {ds.n_classes} classes")

    with console.status("Loading model..."):
        m = _load_model(model)

    from unrelabel.attacks.clean_label import CleanLabelAttack
    attack = CleanLabelAttack(
        source_class=source_class,
        target_class=target_class,
        n_neighbors=n_neighbors,
        epsilon=epsilon,
        seed=seed,
    )

    with console.status("Running attack..."):
        result = attack.run(ds, m)

    output.mkdir(parents=True, exist_ok=True)

    from unrelabel.reporting.report import ReportBuilder
    json_path, html_path = ReportBuilder().build(result, output)

    table = Table(title="Attack Results", box=box.ROUNDED, style="cyan")
    table.add_column("Metric", style="bold")
    table.add_column("Value")
    table.add_row("Clean Accuracy", f"{result.clean_accuracy:.4f}")
    table.add_row("Poisoned Accuracy", f"{result.poisoned_accuracy:.4f}")
    table.add_row("Accuracy Drop", f"{result.accuracy_drop:.4f}")
    table.add_row("Attack Succeeded", str(result.config["attack_success"]))
    table.add_row("Vulnerability Score", f"{result.vulnerability_score:.1f}/100")
    console.print(table)
    console.print(f"\n[green]Report saved:[/green] {json_path}")
    console.print(f"[green]HTML report:[/green]  {html_path}")

    exit_code = 1 if result.vulnerability_score > 5 else 0
    raise typer.Exit(exit_code)


@report_app.command("view")
def report_view(path: Path = typer.Argument(..., help="Path to JSON result file")):
    """Print a result JSON report to the terminal."""
    import json
    data = json.loads(path.read_text())
    from rich.pretty import pprint
    pprint(data)


@report_app.command("html")
def report_html(
    path: Path = typer.Argument(..., help="Path to JSON result file"),
    open_browser: bool = typer.Option(False, "--open", help="Open in browser after generating"),
):
    """Re-render HTML from a JSON result file."""
    import json
    import webbrowser
    import numpy as np
    from unrelabel.attacks.base import AttackResult

    data = json.loads(path.read_text())
    result = AttackResult(
        attack_type=data["attack_type"],
        clean_accuracy=data["clean_accuracy"],
        poisoned_accuracy=data["poisoned_accuracy"],
        accuracy_drop=data["accuracy_drop"],
        vulnerability_score=data["vulnerability_score"],
        confusion_matrices=data["confusion_matrices"],
        plots=[Path(p) for p in data.get("plots", [])],
        config=data["config"],
        timestamp=data["timestamp"],
        flipped_indices=np.array(data.get("flipped_indices", [])),
        sweep_results=data.get("sweep_results", []),
    )
    from unrelabel.reporting.report import ReportBuilder
    _, html_path = ReportBuilder().build(result, path.parent)
    console.print(f"[green]HTML report:[/green] {html_path}")
    if open_browser:
        webbrowser.open(str(html_path))


@app.command("ui")
def launch_ui():
    """Launch the web interface."""
    import subprocess
    import sys
    import webbrowser
    import threading

    def _open_browser():
        import time
        time.sleep(1.5)
        webbrowser.open("http://127.0.0.1:8000")

    threading.Thread(target=_open_browser, daemon=True).start()
    subprocess.run([
        sys.executable, "-m", "uvicorn",
        "unrelabel.server.app:app",
        "--host", "127.0.0.1",
        "--port", "8000",
    ])


if __name__ == "__main__":
    app()
