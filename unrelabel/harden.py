"""Turn scan findings into a behavioral canary, and check a model against it.

The premise: you cannot reliably find poisoned rows (input-level detection is
noisy and clean-label poisoning mislabels nothing). So instead of hunting the
poison, `harden` freezes the *behavior* a scan proved fragile into a set of
invariants, and `check` asserts those invariants against any candidate model:
a gate that catches targeted poisoning even when global accuracy looks fine.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from unrelabel.scan import ScanRunner

# Policy defaults. Deliberately conservative; documented as tunable.
ACCURACY_MARGIN = 0.05   # allowed global-accuracy regression
ASR_MARGIN = 0.15        # allowed backdoor-ASR lift above the clean model's baseline
MAX_TRANSITION = 0.15    # a protected class must rarely be mislabeled as target
MIN_RECALL = 0.85        # a protected class must retain at least this recall


def generate_guardrail(run_dir: Path, out_dir: Path | None = None) -> Path:
    """Read a scan run and emit a guardrail/ directory (canary + CI snippet)."""
    run_dir = Path(run_dir)
    report = json.loads((run_dir / "result.json").read_text(encoding="utf-8"))
    out_dir = out_dir or (run_dir / "guardrail")
    out_dir.mkdir(parents=True, exist_ok=True)

    baseline = float(report.get("baseline_accuracy", 0.0))
    invariants: list[dict[str, Any]] = [
        {
            "id": "global-accuracy",
            "type": "min_accuracy",
            "threshold": round(baseline - ACCURACY_MARGIN, 4),
            "description": f"Overall accuracy must stay within {ACCURACY_MARGIN:.0%} of the {baseline:.1%} baseline.",
        }
    ]

    seen: set[tuple] = set()
    for finding in report.get("findings", []):
        if finding.get("severity") in {"clean", "low"}:
            continue
        attack = finding.get("attack")
        source = finding.get("source_label")
        target = finding.get("target_label")
        if attack == "keyword-backdoor":
            trigger = finding.get("keyword") or finding.get("trigger")
            # The scan report stores the trigger under the attack config; recover it.
            trigger = trigger or _trigger_for_finding(report, finding)
            key = ("backdoor", trigger, target)
            if trigger and key not in seen:
                seen.add(key)
                # Threshold = clean model's own triggered-ASR + a margin, so the
                # gate flags a genuine lift rather than the trigger's dilution effect.
                base_asr = finding.get("baseline_asr") or 0.0
                max_asr = round(min(0.95, base_asr + ASR_MARGIN), 4)
                invariants.append(
                    {
                        "id": f"backdoor-{_slug(trigger)}",
                        "type": "backdoor_asr",
                        "trigger": trigger,
                        "source_label": source,
                        "target_label": target,
                        "baseline_asr": round(base_asr, 4),
                        "max_asr": max_asr,
                        "description": f"Inputs carrying '{trigger}' must not flip to {target} above {max_asr:.0%} (clean baseline {base_asr:.0%}).",
                    }
                )
        elif attack in {"targeted-label-flip", "keyword-targeted"} and source and target:
            key = ("transition", source, target)
            if key not in seen:
                seen.add(key)
                invariants.append(
                    {
                        "id": f"transition-{_slug(source)}-{_slug(target)}",
                        "type": "targeted_transition",
                        "source_label": source,
                        "target_label": target,
                        "max_rate": MAX_TRANSITION,
                        "description": f"'{source}' reviews must rarely be predicted '{target}'.",
                    }
                )
                invariants.append(
                    {
                        "id": f"recall-{_slug(source)}",
                        "type": "class_recall",
                        "label": source,
                        "min_recall": MIN_RECALL,
                        "description": f"Recall on '{source}' must stay above {MIN_RECALL:.0%}.",
                    }
                )

    canary = {
        "project": report.get("project", "unrelabel"),
        "generated_from": str(run_dir),
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "baseline_accuracy": baseline,
        "task": report.get("task", {}),
        "invariants": invariants,
    }
    canary_path = out_dir / "canary.yaml"
    canary_path.write_text(yaml.safe_dump(canary, sort_keys=False), encoding="utf-8")
    (out_dir / "ci.yml").write_text(_ci_snippet(canary["project"]), encoding="utf-8")
    (out_dir / "README.md").write_text(_guardrail_readme(canary["project"], invariants), encoding="utf-8")
    return canary_path


def _trigger_for_finding(report: dict, finding: dict) -> str | None:
    for result in report.get("results", []):
        if (
            result.get("attack") == "keyword-backdoor"
            and result.get("poison_rate") == finding.get("poison_rate")
        ):
            return result.get("keyword") or result.get("trigger")
    return None


@dataclass
class InvariantResult:
    id: str
    type: str
    measured: float
    threshold: float
    passed: bool
    description: str
    measurable: bool = True  # False when predictions/target slice are unavailable


class CanaryChecker:
    """Evaluate a candidate model (described by a scan-style config) against a canary."""

    def __init__(self, canary: dict[str, Any], config: dict[str, Any], config_path: Path):
        self.canary = canary
        self.runner = ScanRunner(config, config_path)

    def run(self) -> dict[str, Any]:
        run_dir = self.runner._make_run_dir()
        train_csv, test_csv = self.runner._copy_dataset_inputs(run_dir)
        _, test_df = self.runner._load_dataset_copies(train_csv, test_csv)
        adapter = self.runner._make_adapter()
        baseline = adapter.evaluate(train_csv, test_csv, run_dir / "baseline")

        results: list[InvariantResult] = []
        for inv in self.canary.get("invariants", []):
            results.append(
                self._check(inv, adapter, train_csv, test_csv, test_df, baseline, run_dir)
            )

        passed = all(r.passed for r in results)
        unmeasurable = sum(1 for r in results if not r.measurable)
        report = {
            "project": self.canary.get("project", "unrelabel"),
            "run_dir": str(run_dir),
            "baseline_accuracy": baseline.accuracy,
            "passed": passed,
            "unmeasurable": unmeasurable,
            "invariants": [r.__dict__ for r in results],
        }
        (run_dir / "check.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
        return report

    def _check(self, inv, adapter, train_csv, test_csv, test_df, baseline, run_dir) -> InvariantResult:
        itype = inv["type"]
        if itype == "min_accuracy":
            measured = float(baseline.accuracy)
            threshold = float(inv["threshold"])
            return InvariantResult(inv["id"], itype, measured, threshold, measured >= threshold, inv["description"])

        if itype == "backdoor_asr":
            attack = {
                "type": "keyword-backdoor",
                "trigger": inv["trigger"],
                "source_label": inv.get("source_label"),
                "target_label": inv["target_label"],
            }
            triggered_df = self.runner._triggered_test(test_df, attack)
            triggered_csv = run_dir / f"triggered_{inv['id']}.csv"
            triggered_df.to_csv(triggered_csv, index=False)
            evaluation = adapter.evaluate(train_csv, triggered_csv, run_dir / inv["id"])
            if evaluation.predictions is not None:
                asr = float(np.mean(np.asarray(evaluation.predictions) == inv["target_label"]))
            else:
                # command adapter: triggered set is all source_label, so ASR = 1 - accuracy
                asr = 1.0 - float(evaluation.accuracy)
            threshold = float(inv["max_asr"])
            return InvariantResult(inv["id"], itype, asr, threshold, asr <= threshold, inv["description"])

        if itype == "style_asr":
            from unrelabel.style import rewrite_style

            tx = self.runner.text_column
            lb = self.runner.label_column
            source = inv.get("source_label")
            if source is not None:
                styled = test_df[test_df[lb].astype(str) == str(source)].copy()
            else:
                styled = test_df[test_df[lb].astype(str) != str(inv["target_label"])].copy()
            threshold = float(inv["max_asr"])
            if len(styled) == 0:  # nothing to measure: fail-closed, do not vouch for it
                return InvariantResult(inv["id"], itype, -1.0, threshold, False, inv["description"], measurable=False)
            styled[tx] = styled[tx].fillna("").astype(str).map(
                lambda t: rewrite_style(t, inv.get("style", "formal"))
            )
            styled_csv = run_dir / f"styled_{inv['id']}.csv"
            styled.reset_index(drop=True).to_csv(styled_csv, index=False)
            evaluation = adapter.evaluate(train_csv, styled_csv, run_dir / inv["id"])
            if evaluation.predictions is not None:
                asr = float(np.mean(np.asarray(evaluation.predictions) == inv["target_label"]))
            else:
                # command adapter: styled set keeps the source label, so ASR = 1 - accuracy
                asr = 1.0 - float(evaluation.accuracy)
            return InvariantResult(inv["id"], itype, asr, threshold, asr <= threshold, inv["description"])

        if itype == "targeted_transition":
            rate = self._transition_rate(test_df, baseline.predictions, inv["source_label"], inv["target_label"])
            threshold = float(inv["max_rate"])
            if rate is None:  # no predictions / empty source slice: fail-closed, not a silent PASS
                return InvariantResult(inv["id"], itype, -1.0, threshold, False, inv["description"], measurable=False)
            return InvariantResult(inv["id"], itype, rate, threshold, rate <= threshold, inv["description"])

        if itype == "subgroup_transition":
            rate = self._transition_rate(
                test_df, baseline.predictions, inv["source_label"], inv["target_label"],
                subgroup=inv.get("subgroup"),
            )
            threshold = float(inv["max_rate"])
            if rate is None:  # unmeasurable slice: fail-closed
                return InvariantResult(inv["id"], itype, -1.0, threshold, False, inv["description"], measurable=False)
            return InvariantResult(inv["id"], itype, rate, threshold, rate <= threshold, inv["description"])

        if itype == "class_recall":
            recall = self._class_recall(test_df, baseline.predictions, inv["label"])
            threshold = float(inv["min_recall"])
            if recall is None:
                return InvariantResult(inv["id"], itype, -1.0, threshold, False, inv["description"], measurable=False)
            return InvariantResult(inv["id"], itype, recall, threshold, recall >= threshold, inv["description"])

        raise ValueError(f"Unknown invariant type '{itype}'.")

    def _transition_rate(self, test_df, predictions, source, target, subgroup=None) -> float | None:
        if predictions is None:
            return None
        label_col = self.runner.label_column
        mask = (test_df[label_col] == source).to_numpy()
        if subgroup:  # restrict to the keyword-defined slice
            import re as _re

            text_col = self.runner.text_column
            in_group = test_df[text_col].fillna("").astype(str).str.contains(
                _re.escape(str(subgroup)), case=False, regex=True
            ).to_numpy()
            mask = mask & in_group
        if int(mask.sum()) == 0:
            return None if subgroup else 0.0
        pred = np.asarray(predictions)
        return float(np.mean(pred[mask] == target))

    def _class_recall(self, test_df, predictions, label) -> float | None:
        if predictions is None:
            return None
        label_col = self.runner.label_column
        mask = (test_df[label_col] == label).to_numpy()
        if int(mask.sum()) == 0:
            return 1.0
        pred = np.asarray(predictions)
        return float(np.mean(pred[mask] == label))


def _slug(value: Any) -> str:
    return "".join(c if str(c).isalnum() else "-" for c in str(value)).strip("-").lower()


def _ci_snippet(project: str) -> str:
    return f"""# unrelabel behavioral gate for {project}
# Add to your model release pipeline. Fails the build if a retrain reintroduces
# a poisoning-fragile behavior, even when global accuracy looks fine.
name: unrelabel-canary
on: [push, pull_request]
jobs:
  poisoning-gate:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - run: pip install unrelabel
      - run: unrelabel check unrelabel.yaml --canary guardrail/canary.yaml
"""


def _guardrail_readme(project: str, invariants: list[dict]) -> str:
    lines = [
        f"# guardrail: {project}",
        "",
        "Behavioral invariants derived from an `unrelabel scan`. Assert them against",
        "any candidate model with:",
        "",
        "```bash",
        "unrelabel check <config.yaml> --canary canary.yaml",
        "```",
        "",
        "Exit code is non-zero if any invariant fails. Invariants:",
        "",
    ]
    for inv in invariants:
        lines.append(f"- **{inv['id']}** ({inv['type']}): {inv['description']}")
    lines.append("")
    lines.append("Thresholds are policy defaults; edit `canary.yaml` to match your risk tolerance.")
    return "\n".join(lines) + "\n"
