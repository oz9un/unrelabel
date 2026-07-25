from __future__ import annotations
from pathlib import Path
from typing import Optional, List
import shutil

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

console = Console()


def _enable_command_adapters(allow: bool) -> None:
    """Opt in to config-driven command execution for this process.

    A ``model.type: command`` config runs the shell commands in its train /
    evaluate fields; that stays off unless the operator passes --allow-command,
    so scanning an untrusted config never silently executes its commands.
    """
    if allow:
        import os

        os.environ["UNRELABEL_ALLOW_COMMANDS"] = "1"


def _enable_pickle_loading(allow: bool) -> None:
    """Opt in to pickle/torch model deserialization for this process.

    Loading a .pkl/.pt model executes any code embedded in the file, so it
    stays off unless the operator passes --allow-pickle for a file they trust.
    """
    if allow:
        import os

        os.environ["UNRELABEL_ALLOW_PICKLE"] = "1"


@app.callback(invoke_without_command=True)
def main(ctx: typer.Context):
    """Offensive ML data poisoning toolkit."""
    if ctx.invoked_subcommand is None:
        console.print(BANNER, style="bold red")
        console.print(ctx.get_help())
        raise typer.Exit()


@app.command("scan")
def scan_config(
    config: Path = typer.Argument(..., help="Path to unrelabel YAML/JSON scan config"),
    fail_on: Optional[str] = typer.Option(
        None,
        "--fail-on",
        help="Exit non-zero when any finding is at or above this severity.",
    ),
    allow_command: bool = typer.Option(
        False,
        "--allow-command",
        help="Permit model.type: command configs to run their shell commands. "
        "Only for configs you trust.",
    ),
):
    """Run a config-driven poisoning robustness scan."""
    from unrelabel.config import load_scan_config
    from unrelabel.scan import SEVERITY_ORDER, ScanRunner, fail_threshold_met

    if fail_on and fail_on.lower() not in SEVERITY_ORDER:
        raise typer.BadParameter(f"--fail-on must be one of: {', '.join(SEVERITY_ORDER)}")
    _enable_command_adapters(allow_command)

    with console.status("Running poisoning robustness scan..."):
        cfg = load_scan_config(config)
        report = ScanRunner(cfg, config).run()

    table = Table(title=f"Scan Results: {report['project']}", box=box.ROUNDED, style="cyan")
    table.add_column("Metric", style="bold")
    table.add_column("Value")
    table.add_row("Run", report["run_id"])
    table.add_row("Baseline Accuracy", f"{report['baseline_accuracy']:.4f}")
    table.add_row("Minimum Poison Budget", str(report["minimum_poison_budget"] or "not reached"))
    table.add_row("Findings", str(len(report["findings"])))
    console.print(table)

    run_dir = (config.parent / cfg.get("run", {}).get("output_dir", "runs") / report["run_id"]).resolve()
    console.print(f"[green]JSON findings:[/green] {run_dir / 'findings.json'}")
    console.print(f"[green]Markdown summary:[/green] {run_dir / 'summary.md'}")
    console.print(f"[green]HTML report:[/green] {run_dir / 'report.html'}")

    raise typer.Exit(1 if fail_threshold_met(report, fail_on) else 0)


@app.command("compare")
def compare_configs(
    baseline: Path = typer.Argument(..., help="Trusted baseline config"),
    candidate: Path = typer.Argument(..., help="Candidate artifact config"),
    allow_command: bool = typer.Option(
        False,
        "--allow-command",
        help="Permit model.type: command configs to run their shell commands. "
        "Only for configs you trust.",
    ),
):
    """Compare a trusted baseline against a candidate artifact."""
    from unrelabel.compare import CompareRunner
    from unrelabel.config import load_scan_config

    _enable_command_adapters(allow_command)
    with console.status("Running behavioral comparison..."):
        report = CompareRunner(
            load_scan_config(baseline),
            load_scan_config(candidate),
            baseline,
            candidate,
        ).run()

    table = Table(title="Behavioral Comparison", box=box.ROUNDED, style="cyan")
    table.add_column("Metric", style="bold")
    table.add_column("Value")
    table.add_row("Baseline Accuracy", f"{report['global_accuracy']['baseline']:.4f}")
    table.add_row("Candidate Accuracy", f"{report['global_accuracy']['candidate']:.4f}")
    table.add_row("Repository Checks", str(report["repository_checks_passed"]))
    table.add_row("Behavioral Integrity Failed", str(report["behavioral_integrity_failed"]))
    console.print(table)

    run_dir = Path(report["run_dir"])
    console.print(f"[green]Behavioral diff JSON:[/green] {run_dir / 'behavioral_diff.json'}")
    console.print(f"[green]Findings JSON:[/green] {run_dir / 'findings.json'}")
    console.print(f"[green]Markdown report:[/green] {run_dir / 'report.md'}")
    console.print(f"[green]HTML report:[/green] {run_dir / 'report.html'}")

    raise typer.Exit(1 if report["behavioral_integrity_failed"] else 0)


@app.command("report", context_settings={"allow_extra_args": True, "ignore_unknown_options": True})
def report_command(
    ctx: typer.Context,
    target: Optional[str] = typer.Argument(None, help="Run directory, or legacy action: view/html"),
    format: str = typer.Option("html", "--format", help="html, json, or markdown"),
    output: Optional[Path] = typer.Option(None, "--output", "-o", help="Optional destination path"),
    open_browser: bool = typer.Option(False, "--open", help="Open legacy HTML report in a browser"),
):
    """Show or export scan reports; also supports legacy `report view/html` forms."""
    if target in {"view", "html"}:
        if not ctx.args:
            raise typer.BadParameter(f"report {target} requires a result JSON path")
        legacy_path = Path(ctx.args[0])
        if target == "view":
            report_view(legacy_path)
            raise typer.Exit(0)
        report_html(legacy_path, open_browser=open_browser)
        raise typer.Exit(0)

    if target is None:
        console.print(ctx.get_help())
        raise typer.Exit()

    artifact_names = {
        "html": "report.html",
        "json": "findings.json",
        "markdown": "summary.md",
        "md": "summary.md",
    }
    key = format.lower()
    if key not in artifact_names:
        raise typer.BadParameter("--format must be html, json, or markdown")
    artifact = Path(target) / artifact_names[key]
    if not artifact.exists():
        raise typer.BadParameter(f"Report artifact not found: {artifact}")
    if output:
        output.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(artifact, output)
        console.print(f"[green]Wrote {key} report:[/green] {output}")
    else:
        console.print(f"[green]{key.upper()} report:[/green] {artifact.resolve()}")
        if key in {"json", "markdown", "md"}:
            console.print(artifact.read_text(encoding="utf-8"))
    raise typer.Exit(0)


@app.command("init")
def init_command(
    source: str = typer.Argument(..., help="CSV path, or an hf://owner/name dataset (optional trailing /split)"),
    out_dir: Path = typer.Option(Path("unrelabel_scan"), "--out-dir", "-o", help="Where to write config + splits"),
    test_ratio: float = typer.Option(0.2, "--test-ratio", help="Fraction held out for the test split"),
    seed: int = typer.Option(42, "--seed", help="Random seed for the split"),
):
    """Scaffold a runnable scan config from a raw dataset (CSV or hf:// reference)."""
    from unrelabel.init_config import TRIGGER_PLACEHOLDER, scaffold

    with console.status(f"Inspecting {source}..."):
        result = scaffold(source, out_dir, test_ratio=test_ratio, seed=seed)

    table = Table(title="Inferred dataset schema", box=box.ROUNDED, style="cyan")
    table.add_column("Field", style="bold")
    table.add_column("Value")
    table.add_row("Text column", result.text_column or "[dim]none detected[/dim]")
    table.add_row("Label column", result.label_column)
    table.add_row("Classes", ", ".join(result.classes[:8]) + (" ..." if len(result.classes) > 8 else ""))
    table.add_row("Protected class (source)", result.source_label)
    table.add_row("Attacker target", result.target_label)
    console.print(table)

    for note in result.notes:
        console.print(f"[yellow]note:[/yellow] {note}")
    console.print(f"\n[green]Config:[/green] {result.config_path}")
    console.print(f"[green]Splits:[/green] {result.train_path} / {result.test_path}")

    if result.text_column:
        console.print(
            f"\n[bold]Next:[/bold] open the config and replace "
            f"[cyan]\"{TRIGGER_PLACEHOLDER}\"[/cyan] with a rare trigger phrase, then run:"
        )
    else:
        console.print("\n[bold]Next:[/bold] review the config, then run:")
    console.print(f"  [cyan]unrelabel scan {result.config_path}[/cyan]")


@app.command("playground")
def playground_command(
    config: Optional[Path] = typer.Argument(None, help="Scan config with a keyword-backdoor attack; omit to browse the bundled demo datasets (run from the repo root)"),
    port: int = typer.Option(8001, "--port", help="Port to serve on"),
    no_browser: bool = typer.Option(False, "--no-browser", help="Do not open a browser"),
):
    """Launch the interactive poisoning playground for a config or the bundled demos."""
    import threading
    import time
    import webbrowser

    import uvicorn

    from unrelabel.playground import PlaygroundHub, create_app

    root = config.parent if config else Path(".")
    hub = PlaygroundHub(root, extra_configs=[config] if config else None)
    if not hub.list():
        if config:
            console.print(f"[red]Could not load a dataset from[/red] {config}.")
        else:
            console.print("[red]No demo datasets found here.[/red] Run from the unrelabel repo root, or pass a config: [cyan]unrelabel playground <unrelabel.yaml>[/cyan]")
        raise typer.Exit(2)
    app_obj = create_app(hub)

    url = f"http://127.0.0.1:{port}"
    console.print(f"[green]Playground:[/green] {url}")
    if not no_browser:
        def _open():
            time.sleep(1.2)
            webbrowser.open(url)
        threading.Thread(target=_open, daemon=True).start()
    uvicorn.run(app_obj, host="127.0.0.1", port=port, log_level="warning")


@app.command("probe")
def probe_command(
    config: Path = typer.Argument(..., help="Scan config with a keyword-backdoor attack"),
    text: Optional[str] = typer.Option(None, "--text", help="One-shot: classify this input and exit"),
    rate: float = typer.Option(0.05, "--rate", help="Backdoor injection rate for the poisoned model"),
):
    """Type an input and see the clean vs. backdoored model side by side."""
    from unrelabel.config import load_scan_config
    from unrelabel.probe import Comparison, Probe

    with console.status("Training clean and backdoored models..."):
        probe = Probe(load_scan_config(config), config, poison_rate=rate)

    console.print(
        f"Trigger phrase: [bold red]{probe.trigger}[/bold red] → forces "
        f"[bold]{probe.target_label}[/bold]  "
        f"[dim]({probe.n_injected} rows injected, {rate:.1%})[/dim]\n"
    )

    def _fmt(v) -> str:
        conf = f" [dim]{v.confidence*100:.0f}%[/dim]" if v.confidence is not None else ""
        return f"{v.label}{conf}"

    def _show(cmp: Comparison) -> None:
        table = Table(box=box.ROUNDED, style="cyan", show_header=True)
        table.add_column("Input", style="bold", overflow="fold", max_width=52)
        table.add_column("Clean model")
        table.add_column("Backdoored model")
        table.add_row(cmp.text or "[dim](empty)[/dim]", _fmt(cmp.clean), _fmt(cmp.poisoned))
        table.add_row(
            f"{cmp.text} [red]{probe.trigger}[/red]".strip(),
            _fmt(cmp.clean_triggered),
            _fmt(cmp.poisoned_triggered),
        )
        console.print(table)
        if cmp.backdoor_fired:
            console.print(
                f"[bold red]Backdoor fired:[/bold red] adding the trigger flipped the backdoored "
                f"model to [bold]{cmp.poisoned_triggered.label}[/bold] while the clean model held.\n"
            )
        else:
            console.print("[dim]Trigger did not change the outcome for this input.[/dim]\n")

    if text is not None:
        _show(probe.compare(text))
        raise typer.Exit(0)

    console.print("[dim]Type an input and press Enter. Ctrl-D or blank line to quit.[/dim]\n")
    while True:
        try:
            line = console.input("[cyan]> [/cyan]")
        except EOFError:
            break
        if not line.strip():
            break
        _show(probe.compare(line))
    raise typer.Exit(0)


@app.command("harden")
def harden_command(
    run: Path = typer.Argument(..., help="Scan run directory (e.g. runs/latest)"),
    output: Optional[Path] = typer.Option(None, "--output", "-o", help="Guardrail output directory"),
):
    """Turn scan findings into a behavioral canary you can gate on in CI."""
    from unrelabel.harden import generate_guardrail

    result_json = run / "result.json"
    if not result_json.exists():
        raise typer.BadParameter(f"No scan result found at {result_json}")

    canary_path = generate_guardrail(run, output)
    guardrail_dir = canary_path.parent
    console.print(f"[green]Canary:[/green] {canary_path}")
    console.print(f"[green]CI snippet:[/green] {guardrail_dir / 'ci.yml'}")
    console.print(f"[green]How-to:[/green] {guardrail_dir / 'README.md'}")
    console.print(
        f"\nGate any model with: [cyan]unrelabel check <config.yaml> --canary {canary_path}[/cyan]"
    )


@app.command("check")
def check_command(
    config: Path = typer.Argument(..., help="Scan-style config describing the model to gate"),
    canary: Path = typer.Option(
        Path("guardrail/canary.yaml"), "--canary", help="Path to a canary.yaml from `harden`"
    ),
    allow_command: bool = typer.Option(
        False,
        "--allow-command",
        help="Permit model.type: command configs to run their shell commands. "
        "Only for configs you trust.",
    ),
):
    """Assert a model against a behavioral canary. Exits non-zero on any violation."""
    import yaml

    from unrelabel.config import load_scan_config
    from unrelabel.harden import CanaryChecker

    if not canary.exists():
        raise typer.BadParameter(f"Canary not found: {canary}")
    canary_data = yaml.safe_load(canary.read_text(encoding="utf-8"))
    _enable_command_adapters(allow_command)

    with console.status("Checking model against behavioral canary..."):
        report = CanaryChecker(canary_data, load_scan_config(config), config).run()

    table = Table(title=f"Canary Check: {report['project']}", box=box.ROUNDED, style="cyan")
    table.add_column("Invariant", style="bold")
    table.add_column("Measured", justify="right")
    table.add_column("Threshold", justify="right")
    table.add_column("Result")
    for inv in report["invariants"]:
        if not inv.get("measurable", True):
            # Fail-closed: the protected slice could not be measured, so the gate
            # cannot vouch for it. Surface that instead of a bogus -1.0000.
            table.add_row(
                inv["id"], "[yellow]n/a[/yellow]", f"{inv['threshold']:.4f}", "[yellow]UNMEASURABLE[/yellow]"
            )
            continue
        verdict = "[green]PASS[/green]" if inv["passed"] else "[red]FAIL[/red]"
        table.add_row(inv["id"], f"{inv['measured']:.4f}", f"{inv['threshold']:.4f}", verdict)
    console.print(table)

    unmeasurable = report.get("unmeasurable", 0)
    if report["passed"]:
        console.print("[green]All behavioral invariants held.[/green]")
    else:
        console.print("[red]Behavioral integrity failed. A protected behavior regressed.[/red]")
        if unmeasurable:
            console.print(
                f"[yellow]{unmeasurable} invariant(s) could not be measured and were "
                "failed closed. Provide model predictions to evaluate them.[/yellow]"
            )
    raise typer.Exit(0 if report["passed"] else 1)


@app.command("defend")
def defend_command(
    config: Path = typer.Argument(..., help="Scan-style config whose training pool you want to audit"),
    top: int = typer.Option(15, "--top", help="Rows / phrases / tokens to show per section"),
    fail_on: str = typer.Option(
        "none",
        "--fail-on",
        help="CI gate: 'none' report only, or 'unicode' to exit non-zero on any "
        "deceptive-unicode / zero-width flag (near-zero false positives).",
    ),
    allow_command: bool = typer.Option(
        False, "--allow-command", help="Permit model.type: command configs. Trusted configs only."
    ),
):
    """Audit a training pool for poisoning signals (L1 hygiene + L2 label audit).

    Surfaces candidates for human review: deceptive unicode, class-concentrated
    repeated phrases (the signature of constant-phrase and keyword backdoors), and
    confident label disagreements (mislabels / label-flip poison). It flags
    candidates, not proof; expect benign hits on natural class-heavy vocabulary,
    and with no ground truth on your own data it reports no precision/recall.
    For a reliable automated gate on keyword backdoors, pin the behavior with
    `unrelabel harden` and gate with `unrelabel check`.
    """
    from unrelabel.config import load_scan_config
    from unrelabel.playground import PlaygroundEngine

    _enable_command_adapters(allow_command)
    if fail_on not in ("none", "unicode"):
        raise typer.BadParameter("--fail-on must be one of: none, unicode")

    with console.status("Training a reference model and auditing the training pool..."):
        engine = PlaygroundEngine(load_scan_config(config), config)
        hy = engine.hygiene_scan()
        la = engine.label_audit(max_flag=top)

    console.print(f"\n[bold]Poisoning audit[/bold] · {hy.get('n_rows', '?')} training rows\n")

    # --- L1: dataset hygiene ---
    security = hy.get("security", {}) or {}
    unicode_hits = int(security.get("count", 0))
    phrases = (hy.get("phrases") or {}).get("top", []) or []
    tokens = (hy.get("suspicious") or {}).get("top", []) or []

    console.print("[bold cyan]L1 · dataset hygiene[/bold cyan]")
    console.print(
        f"  deceptive-unicode / zero-width flags: "
        f"[{'red' if unicode_hits else 'green'}]{unicode_hits}[/]"
    )

    if phrases:
        t = Table(
            title="repeated phrases: constant-phrase / keyword backdoor candidates", box=box.SIMPLE
        )
        t.add_column("phrase", style="bold")
        t.add_column("rows", justify="right")
        t.add_column("class-conc.", justify="right")
        t.add_column("label")
        for p in phrases[:top]:
            t.add_row(
                str(p.get("phrase", "")),
                str(p.get("df", "")),
                f"{p.get('concentration', 0):.0%}",
                str(p.get("label", "")),
            )
        console.print(t)
        console.print(
            "[dim]  A phrase in many rows, all one class, is the fingerprint of a constant-phrase or "
            "keyword backdoor. Natural boilerplate can also land here. Review before acting.[/dim]"
        )
    else:
        console.print("  [green]no repeated class-concentrated phrases[/green]")

    if tokens:
        t = Table(
            title="class-concentrated tokens (secondary: natural class vocabulary appears here too)",
            box=box.SIMPLE,
        )
        t.add_column("token", style="bold")
        t.add_column("rows", justify="right")
        t.add_column("conc.", justify="right")
        t.add_column("label")
        for tok in tokens[: min(top, 8)]:
            t.add_row(
                str(tok.get("escaped", tok.get("token", ""))),
                str(tok.get("df", "")),
                f"{tok.get('concentration', 0):.0%}",
                str(tok.get("label", "")),
            )
        console.print(t)

    # --- L2: label audit ---
    console.print("\n[bold cyan]L2 · label audit[/bold cyan] (confident-learning)")
    rows = la.get("rows", []) or []
    console.print(
        f"  {len(rows)} rows the reference model confidently disagrees with "
        "(candidate mislabels / label-flip poison):"
    )
    if rows:
        t = Table(box=box.SIMPLE)
        t.add_column("row", justify="right")
        t.add_column("given")
        t.add_column("model")
        t.add_column("conf", justify="right")
        t.add_column("text")
        for r in rows[:top]:
            t.add_row(
                str(r.get("row", "")),
                str(r.get("given", "")),
                str(r.get("predicted", "")),
                f"{r.get('confidence', 0):.0%}",
                (str(r.get("text", "")) or "")[:66],
            )
        console.print(t)

    console.print(
        "\n[dim]Candidates for human review, not proof of poisoning. No ground truth on your data means "
        "no precision/recall. For an automated backstop on keyword backdoors, use `unrelabel harden` + "
        "`unrelabel check`.[/dim]"
    )

    fail = fail_on == "unicode" and unicode_hits > 0
    if fail:
        console.print(
            f"\n[red]Audit gate failed (--fail-on {fail_on}): deceptive-unicode signal present.[/red]"
        )
    raise typer.Exit(1 if fail else 0)


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
    poison_rate: Optional[float] = typer.Option(None, help="Single poison rate (0.0 to 1.0)"),
    poison_rates: Optional[List[float]] = typer.Option(None, help="Multiple rates for sweep"),
    source_class: Optional[int] = typer.Option(None, help="Only flip labels of this class"),
    target_class: Optional[int] = typer.Option(None, help="Flip to this class (random if omitted)"),
    test_size: float = typer.Option(0.2, help="Train/test split ratio"),
    seed: int = typer.Option(42, help="Random seed"),
    output: Path = typer.Option(Path("./report"), help="Output directory for results"),
    dry_run: bool = typer.Option(False, "--dry-run", help="Validate inputs only, do not run attack"),
    allow_pickle: bool = typer.Option(
        False,
        "--allow-pickle",
        help="Permit loading .pkl/.pt model files (deserialization runs their code). "
        "Only for files you trust.",
    ),
):
    """Execute a Label Flipping attack on a dataset and model."""
    console.rule("[bold cyan]unrelabel: Label Flip Attack[/bold cyan]")
    console.print(f"  Dataset : [yellow]{dataset}[/yellow]")
    console.print(f"  Model   : [yellow]{model}[/yellow]")

    if dry_run:
        console.print("\n[green]Dry run complete. Inputs look valid.[/green]")
        raise typer.Exit(0)

    if poison_rate is None and not poison_rates:
        console.print("[red]Error: Provide --poison-rate or --poison-rates.[/red]")
        raise typer.Exit(2)

    rates = list(poison_rates) if poison_rates else [poison_rate]
    console.print(f"  Rates   : [yellow]{rates}[/yellow]")

    with console.status("Loading dataset..."):
        ds = _load_dataset(dataset, label_col, test_size, seed)
    console.print(f"  Loaded  : {ds.n_train} train / {ds.n_test} test samples, {ds.n_classes} classes")

    _enable_pickle_loading(allow_pickle)
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
    poison_rate: Optional[float] = typer.Option(None, help="Single poison rate (0.0 to 1.0)"),
    poison_rates: Optional[List[float]] = typer.Option(None, help="Multiple rates for sweep"),
    test_size: float = typer.Option(0.2, help="Train/test split ratio"),
    seed: int = typer.Option(42, help="Random seed"),
    output: Path = typer.Option(Path("./report"), help="Output directory for results"),
    dry_run: bool = typer.Option(False, "--dry-run", help="Validate inputs only, do not run attack"),
    allow_pickle: bool = typer.Option(
        False,
        "--allow-pickle",
        help="Permit loading .pkl/.pt model files (deserialization runs their code). "
        "Only for files you trust.",
    ),
):
    """Execute a Targeted Label Attack: flip source_class → target_class in training data."""
    console.rule("[bold cyan]unrelabel: Targeted Label Attack[/bold cyan]")
    console.print(f"  Dataset      : [yellow]{dataset}[/yellow]")
    console.print(f"  Model        : [yellow]{model}[/yellow]")
    console.print(f"  Source Class : [yellow]{source_class}[/yellow]")
    console.print(f"  Target Class : [yellow]{target_class}[/yellow]")

    if dry_run:
        console.print("\n[green]Dry run complete. Inputs look valid.[/green]")
        raise typer.Exit(0)

    if poison_rate is None and not poison_rates:
        console.print("[red]Error: Provide --poison-rate or --poison-rates.[/red]")
        raise typer.Exit(2)

    rates = list(poison_rates) if poison_rates else [poison_rate]
    console.print(f"  Rates        : [yellow]{rates}[/yellow]")

    with console.status("Loading dataset..."):
        ds = _load_dataset(dataset, label_col, test_size, seed)
    console.print(f"  Loaded  : {ds.n_train} train / {ds.n_test} test samples, {ds.n_classes} classes")

    _enable_pickle_loading(allow_pickle)
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
    allow_pickle: bool = typer.Option(
        False,
        "--allow-pickle",
        help="Permit loading .pkl/.pt model files (deserialization runs their code). "
        "Only for files you trust.",
    ),
):
    """Execute a Clean Label Attack: perturb source_class features to misclassify a target point."""
    console.rule("[bold cyan]unrelabel: Clean Label Attack[/bold cyan]")
    console.print(f"  Dataset      : [yellow]{dataset}[/yellow]")
    console.print(f"  Model        : [yellow]{model}[/yellow]")
    console.print(f"  Source Class : [yellow]{source_class}[/yellow]")
    console.print(f"  Target Class : [yellow]{target_class}[/yellow]")

    if dry_run:
        console.print("\n[green]Dry run complete. Inputs look valid.[/green]")
        raise typer.Exit(0)

    if source_class == target_class:
        console.print("[red]Error: --source-class and --target-class must differ.[/red]")
        raise typer.Exit(2)

    with console.status("Loading dataset..."):
        ds = _load_dataset(dataset, label_col, test_size, seed)
    console.print(f"  Loaded  : {ds.n_train} train / {ds.n_test} test samples, {ds.n_classes} classes")

    _enable_pickle_loading(allow_pickle)
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


def report_view(path: Path = typer.Argument(..., help="Path to JSON result file")):
    """Print a result JSON report to the terminal."""
    import json
    data = json.loads(path.read_text())
    from rich.pretty import pprint
    pprint(data)


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
def launch_ui(
    configs: List[Path] = typer.Argument(None, help="One or more scan configs to load (e.g. my.yaml)"),
    port: int = typer.Option(8000, "--port", help="Port to serve on"),
    no_browser: bool = typer.Option(False, "--no-browser", help="Do not open a browser"),
):
    """Launch the interactive integrity bench for your own model(s)."""
    import threading
    import time
    import webbrowser

    import uvicorn

    from unrelabel.playground import PlaygroundHub, create_app

    hub = PlaygroundHub(Path.cwd(), extra_configs=list(configs) if configs else None)
    if not hub.list():
        console.print(
            "[red]No datasets found.[/red] Run `unrelabel ui` from the repo root (with `examples/`), "
            "or pass your own config: [cyan]unrelabel ui my_model.yaml[/cyan]."
        )
        raise typer.Exit(2)
    app_obj = create_app(hub)

    url = f"http://127.0.0.1:{port}"
    console.print(f"[green]unrelabel:[/green] {url}")
    console.print(f"[dim]datasets: {', '.join(d['label'] for d in hub.list())}[/dim]")
    if not no_browser:
        def _open():
            time.sleep(1.2)
            webbrowser.open(url)
        threading.Thread(target=_open, daemon=True).start()
    uvicorn.run(app_obj, host="127.0.0.1", port=port, log_level="warning")


@app.command("ui-classic")
def launch_ui_classic():
    """Launch the original three-step attack showcase (dataset -> attack -> results)."""
    import subprocess
    import sys
    import threading
    import webbrowser

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


app.add_typer(attack_app, name="attack")


if __name__ == "__main__":
    app()
