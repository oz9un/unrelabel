#!/usr/bin/env python3
"""Render the readme's thesis chart from the measured benchmark metrics.

Reads hf_export/unrelabel-poison-benchmark/*/metrics.json (build it first with
scripts/build_hf_datasets.py) and writes images/poison-benchmark.png.

This is deliberately simpler than the dataset-card chart in hf_charts.py. It has
one job: show that global accuracy barely moves while the attacker's rule fires.
So it plots two directly comparable rates per attack, both taken from the same
metrics file, and never inverts either of them:

  accuracy after the attack   (higher is better, the number a dashboard reports)
  attack success rate         (higher is worse, how often the attacker's rule fired)

Nothing here is fabricated or rounded up; every value comes from metrics.json.
"""
from __future__ import annotations

import json
import pathlib
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = pathlib.Path(__file__).resolve().parents[1]
SRC = ROOT / "hf_export" / "unrelabel-poison-benchmark"
OUT = ROOT / "images" / "poison-benchmark.png"

BG = "#0d1117"
TEXT = "#e6edf3"
MUTED = "#8b94a3"
GRID = "#26303c"
DANGER = "#ff4257"
CALM = "#56a0ff"

# Display names, in the order they appear top to bottom.
ORDER = [
    ("unicode-zero-width", "unicode zero-width"),
    ("style-register", "style register"),
    ("composite-cooccurrence", "composite co-occurrence"),
    ("trigger-backdoor", "trigger backdoor"),
    ("subpopulation-phone", "subpopulation, phone"),
    ("availability-noise", "availability noise"),
]


def load_rows() -> list[dict]:
    rows = []
    for slug, label in ORDER:
        path = SRC / slug / "metrics.json"
        if not path.is_file():
            sys.exit(f"missing {path}. run scripts/build_hf_datasets.py first.")
        m = json.loads(path.read_text())
        rows.append(
            {
                "label": label,
                "poison": m["poison_fraction"],
                "rows": m["poison_rows"],
                "baseline": m["baseline_accuracy"],
                "accuracy": m["poisoned_accuracy"],
                "asr": m["attack_success"],
            }
        )
    return rows


def render(rows: list[dict], out: pathlib.Path) -> None:
    baseline = rows[0]["baseline"]
    n = len(rows)
    fig, ax = plt.subplots(figsize=(11.5, 6.4), facecolor=BG)
    ax.set_facecolor(BG)

    height = 0.34
    for i, r in enumerate(rows):
        y = n - 1 - i
        ax.barh(y + height / 2 + 0.02, r["accuracy"], height=height, color=CALM, zorder=3)
        ax.barh(y - height / 2 - 0.02, r["asr"], height=height, color=DANGER, zorder=3)
        ax.text(r["accuracy"] + 0.012, y + height / 2 + 0.02, f"{r['accuracy']:.3f}",
                va="center", ha="left", color=CALM, fontsize=10.5, zorder=4)
        ax.text(r["asr"] + 0.012, y - height / 2 - 0.02, f"{r['asr']:.0%}",
                va="center", ha="left", color=DANGER, fontsize=10.5, fontweight="bold", zorder=4)

    ax.axvline(baseline, color=MUTED, lw=1.1, ls="--", zorder=2)
    ax.text(baseline, n - 0.32, f" clean baseline {baseline:.3f}", color=MUTED,
            fontsize=9.5, va="bottom", ha="left")

    ax.set_yticks(range(n))
    ax.set_yticklabels(
        [f"{r['label']}\n{r['rows']} rows, {r['poison']:.1%} poison" for r in reversed(rows)],
        color=TEXT, fontsize=11,
    )
    for t in ax.get_yticklabels():
        t.set_linespacing(1.5)

    ax.set_xlim(0, 1.13)
    ax.set_xticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_xticklabels(["0", "20%", "40%", "60%", "80%", "100%"], color=MUTED, fontsize=10)
    ax.set_ylim(-0.7, n - 0.25)
    ax.xaxis.grid(True, color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.tick_params(length=0)

    fig.suptitle("the accuracy number does not move. the attack still works.",
                 color=TEXT, fontsize=17, fontweight="bold", x=0.5, y=0.975)
    ax.set_title(
        "blue: test accuracy after the attack, the number a dashboard reports.    "
        "red: how often the attacker's rule fired.",
        color=MUTED, fontsize=11, pad=14,
    )
    fig.text(0.5, 0.035,
             "one dataset, one clean baseline. availability noise is the control: the only attack here\n"
             "that drags accuracy down far enough for an accuracy gate to catch it.",
             color=MUTED, fontsize=9.5, ha="center", linespacing=1.6)

    fig.subplots_adjust(left=0.235, right=0.97, top=0.845, bottom=0.155)
    fig.savefig(out, dpi=170, facecolor=BG)
    print(f"wrote {out}")


if __name__ == "__main__":
    render(load_rows(), OUT)
