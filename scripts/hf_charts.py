"""Chart rendering for the unrelabel HuggingFace dataset cards.

Design comes from a judged design-panel spec (dumbbell thesis chart, per-attack
two-zone card chart, demo class-balance bar). Every number is passed in from the
caller's measured metrics; nothing here is fabricated. See scripts/build_hf_datasets.py.

Honesty rules baked in (do not "fix" these):
  - The thesis x-axis is the FULL 0..1, never cropped, so tiny global-accuracy
    movements read as the shrugs they are.
  - Backdoor red markers are "1 - ASR" (share of triggered inputs NOT flipped),
    labeled as such, never called "accuracy".
  - Rows are grouped (backdoors vs subpopulation/availability) so no one reads a
    single cross-attack severity ranking off incomparable slices.
  - Availability is the visible, caught control: green coincident dots + a grey
    "before" ghost showing its genuine global drop.
"""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.gridspec import GridSpec  # noqa: E402

BG = "#0d1117"
HEALTHY = "#37d67a"
DANGER = "#ff4257"
INFO = "#56a0ff"
ACCENT = "#a179ff"
TEXT = "#e6edf3"
MUTED = "#8b94a3"
GRID = "#26303c"
BASELINE = 0.942  # shared clean baseline accuracy

# Non-semantic palette for class-balance bars (deliberately avoids green/red so a
# clean class is never confused with the healthy/danger language on the poison cards).
CLASS_COLORS = ["#56a0ff", "#a179ff", "#4fd1c5", "#8b94a3", "#c9a227", "#7c8cff"]


def _style():
    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 11,
        "text.color": TEXT, "axes.labelcolor": TEXT,
        "xtick.color": MUTED, "ytick.color": TEXT,
        "figure.facecolor": BG, "axes.facecolor": BG, "savefig.facecolor": BG,
    })


def _save(fig, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=200, bbox_inches="tight", facecolor=BG)
    plt.close(fig)


# --------------------------------------------------------------------------- thesis

def thesis_chart(rows: list[dict], path) -> None:
    """Horizontal dumbbell, one row per attack, on a full 0..1 axis.

    Each dict in `rows` needs: slug, name, poison_pct, global_after, family
    ('backdoor'|'subpopulation'|'availability'), and for the harm marker either
    `one_minus_asr` (backdoor), `worst_group` (subpopulation), or nothing (availability).
    Order is decided here: backdoors by lie-gap desc, then subpopulation, then availability.
    """
    _style()
    backdoors = sorted((r for r in rows if r["family"] == "backdoor"),
                       key=lambda r: BASELINE - r["one_minus_asr"], reverse=True)
    subpop = [r for r in rows if r["family"] == "subpopulation"]
    avail = [r for r in rows if r["family"] == "availability"]
    ordered = backdoors + subpop + avail

    # y positions top->bottom, with an extra gap before the subpopulation/availability block.
    ys, y = [], float(len(ordered) + 1)
    for i, r in enumerate(ordered):
        if r["family"] in ("subpopulation", "availability") and (i == 0 or ordered[i - 1]["family"] == "backdoor"):
            y -= 1.9  # gap + room for the separator
        else:
            y -= 1.0
        ys.append(y)
    sep_y = (ys[len(backdoors) - 1] + ys[len(backdoors)]) / 2 if subpop or avail else None

    fig, ax = plt.subplots(figsize=(9.0, 4.7))
    ax.axvspan(0.90, 1.0, color=HEALTHY, alpha=0.08, zorder=0)
    ax.axvline(BASELINE, color=HEALTHY, ls="--", lw=1.2, alpha=0.55, zorder=2)
    ax.text(BASELINE, ys[0] + 1.15, "clean baseline 0.942", color=HEALTHY, alpha=0.85,
            fontsize=9, ha="center", va="bottom")
    if sep_y is not None:
        ax.axhline(sep_y, color=GRID, lw=1.0, zorder=1)

    for r, yy in zip(ordered, ys):
        gx = r["global_after"]
        if r["family"] == "backdoor":
            hx, hcol, ccol = r["one_minus_asr"], DANGER, DANGER
            slice_name = "triggered inputs"
        elif r["family"] == "subpopulation":
            hx, hcol, ccol = r["worst_group"], ACCENT, ACCENT
            slice_name = "phone reviews"
        else:
            hx, hcol, ccol = None, None, MUTED
            slice_name = "whole test set"

        if hx is not None:
            ax.plot([hx, gx], [yy, yy], color=ccol, lw=2.4, zorder=1, solid_capstyle="round")
            ax.scatter([hx], [yy], s=110, color=hcol, zorder=3, edgecolors=BG, linewidths=1.2)
            ax.scatter([gx], [yy], s=110, color=INFO, zorder=3, edgecolors=BG, linewidths=1.2)
            ax.text(hx + 0.018, yy + 0.26, f"{hx:.3f}", color=hcol, fontsize=10, va="center", ha="left")
            ax.text(hx + 0.018, yy - 0.30, slice_name, color=MUTED, fontsize=8.5, va="center", ha="left")
            ax.text(gx + 0.016, yy + 0.02, f"{gx:.3f}", color=INFO, fontsize=10, va="center", ha="left")
        else:
            # availability: caught control. grey ghost at baseline, green dots coincident at after.
            ax.plot([gx, BASELINE], [yy, yy], color=MUTED, lw=1.6, zorder=1, solid_capstyle="round")
            ax.scatter([BASELINE], [yy], s=110, facecolors="none", edgecolors=MUTED, linewidths=1.4, zorder=3)
            ax.scatter([gx], [yy], s=130, color=HEALTHY, zorder=3, edgecolors=BG, linewidths=1.2)
            ax.text(gx - 0.015, yy + 0.02, f"{gx:.3f}", color=HEALTHY, fontsize=10, va="center", ha="right")
            ax.text((gx + BASELINE) / 2, yy - 0.34, "-0.062, the one accuracy catches",
                    color=MUTED, fontsize=8.5, va="center", ha="center", style="italic")

        ax.text(-0.015, yy + 0.02, r["name"], color=TEXT, fontsize=10.5, fontweight="bold",
                va="center", ha="right", transform=ax.get_yaxis_transform())
        ax.text(-0.015, yy - 0.34, f"{r['poison_pct']:.1f}% poison", color=MUTED, fontsize=8.5,
                va="center", ha="right", transform=ax.get_yaxis_transform())

    ax.set_xlim(0, 1.0)
    ax.set_ylim(min(ys) - 0.9, ys[0] + 1.5)
    ax.set_yticks([])
    ax.set_xticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
    ax.grid(axis="x", color=GRID, alpha=0.4, zorder=0)
    ax.set_xlabel("rate on the named slice, 0 to 1   (test accuracy; for triggered inputs, 1 minus ASR)",
                  color=MUTED, fontsize=9.5)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.spines["bottom"].set_color(GRID)

    fig.text(0.5, 1.02, "Global accuracy is one number, and it lies", color=TEXT,
             fontsize=15, fontweight="bold", ha="center")
    fig.text(0.5, 0.975, "blue barely moves. red is what actually happened on the slice the attacker aimed at.",
             color=MUTED, fontsize=10, ha="center")
    fig.text(0.5, -0.06,
             "blue = global test accuracy your dashboard reports    "
             "red = 1 minus ASR on triggered inputs    "
             "violet = worst-group accuracy    green = the caught control",
             color=MUTED, fontsize=8.5, ha="center")
    _save(fig, path)


# ------------------------------------------------------------------- per-attack card

def per_attack_chart(row: dict, path) -> None:
    """Two-zone card: left = the calm global before/after; right = the real harm.

    `row` needs: family, poison_pct, global_after, asr, worst_group (subpop only).
    """
    _style()
    fig = plt.figure(figsize=(9.0, 3.0))
    gs = GridSpec(1, 2, width_ratios=[1.0, 1.15], wspace=0.28, figure=fig)
    axL, axR = fig.add_subplot(gs[0]), fig.add_subplot(gs[1])
    avail = row["family"] == "availability"

    # ---- LEFT: what your accuracy dashboard sees
    ga = row["global_after"]
    after_col = DANGER if avail else INFO
    axL.plot([ga, BASELINE], [0, 0], color=MUTED, lw=2.0, zorder=1, solid_capstyle="round")
    axL.scatter([BASELINE], [0], s=150, facecolors="none", edgecolors=HEALTHY, linewidths=1.8, zorder=3)
    axL.scatter([ga], [0], s=150, color=after_col, zorder=3, edgecolors=BG, linewidths=1.2)
    axL.text(BASELINE, 0.42, "before 0.942", color=HEALTHY, fontsize=9, ha="center", va="bottom")
    axL.text(ga, -0.5, f"after {ga:.3f}", color=after_col, fontsize=9, ha="center", va="top")
    delta = ga - BASELINE
    dtxt = "0.000, flat" if abs(delta) < 1e-6 else f"{delta:+.3f}"
    tag = "accuracy gate FIRES" if avail else "PASSES accuracy gate"
    tagcol = DANGER if avail else HEALTHY
    axL.text(0.5, 0.86, f"delta {dtxt}", transform=axL.transAxes, color=TEXT, fontsize=10,
             ha="center", va="top", fontweight="bold")
    axL.text(0.5, -0.30, tag, transform=axL.transAxes, color=tagcol, fontsize=9.5, ha="center", va="top")
    axL.set_xlim(0, 1.0)
    axL.set_ylim(-1, 1)
    axL.set_yticks([])
    axL.set_xticks([0, 0.5, 1.0])
    axL.grid(axis="x", color=GRID, alpha=0.35)
    axL.set_title("what your accuracy dashboard sees", color=MUTED, fontsize=10, pad=10)
    for s in ("top", "right", "left"):
        axL.spines[s].set_visible(False)
    axL.spines["bottom"].set_color(GRID)

    # ---- RIGHT: what actually happened
    axR.set_ylim(-1, 1)
    axR.set_yticks([])
    for s in ("top", "right", "left"):
        axR.spines[s].set_visible(False)
    axR.spines["bottom"].set_color(GRID)
    if row["family"] == "backdoor":
        asr = row["asr"]
        axR.barh([0], [asr], color=DANGER, height=0.5, zorder=3)
        axR.barh([0], [1.0], color=GRID, height=0.5, zorder=1, alpha=0.5)
        axR.text(min(asr + 0.02, 0.98), 0, f"{asr * 100:.1f}%", color=DANGER, fontsize=13,
                 fontweight="bold", va="center", ha="left" if asr < 0.85 else "right")
        axR.set_xlim(0, 1.0)
        axR.set_xticks([0, 0.5, 1.0])
        axR.set_xticklabels(["0", "50%", "100%"])
        axR.grid(axis="x", color=GRID, alpha=0.35)
        axR.set_title("attack success on triggered inputs (ASR)", color=DANGER, fontsize=10, pad=10)
    elif row["family"] == "subpopulation":
        wg = row["worst_group"]
        axR.plot([wg, ga], [0, 0], color=DANGER, lw=2.4, zorder=1, solid_capstyle="round")
        axR.scatter([ga], [0], s=150, color=INFO, zorder=3, edgecolors=BG, linewidths=1.2)
        axR.scatter([wg], [0], s=150, color=DANGER, zorder=3, edgecolors=BG, linewidths=1.2)
        axR.text(ga, 0.42, f"global {ga:.3f}", color=INFO, fontsize=9, ha="center", va="bottom")
        axR.text(wg, -0.5, f"phone reviews {wg:.3f}", color=DANGER, fontsize=9.5, fontweight="bold",
                 ha="center", va="top")
        axR.set_xlim(0, 1.0)
        axR.set_xticks([0, 0.5, 1.0])
        axR.grid(axis="x", color=GRID, alpha=0.35)
        axR.set_title("the slice the attacker aimed at", color=DANGER, fontsize=10, pad=10)
    else:  # availability
        asr = row["asr"]
        axR.barh([0], [asr], color=MUTED, height=0.5, zorder=3, alpha=0.7)
        axR.barh([0], [1.0], color=GRID, height=0.5, zorder=1, alpha=0.5)
        axR.text(asr + 0.02, 0, f"ASR {asr * 100:.1f}%, near noise, ignore", color=MUTED,
                 fontsize=10, va="center", ha="left")
        axR.set_xlim(0, 1.0)
        axR.set_xticks([0, 0.5, 1.0])
        axR.set_xticklabels(["0", "50%", "100%"])
        axR.grid(axis="x", color=GRID, alpha=0.35)
        axR.set_title("no targeted control, the harm is the global drop on the left",
                      color=MUTED, fontsize=9.5, pad=10)
    _save(fig, path)


# -------------------------------------------------------------------------- demo bar

def demo_balance_chart(name: str, counts: dict, path) -> None:
    """One 100%-stacked horizontal bar of class balance. Non-semantic colors."""
    _style()
    total = sum(counts.values())
    fig, ax = plt.subplots(figsize=(9.0, 1.9))
    left = 0.0
    for i, (cls, n) in enumerate(counts.items()):
        frac = n / total
        col = CLASS_COLORS[i % len(CLASS_COLORS)]
        ax.barh([0], [frac], left=left, color=col, height=0.6, edgecolor=BG, linewidth=1.5)
        if frac > 0.06:
            ax.text(left + frac / 2, 0, f"{cls}\n{n}", color=BG if col != MUTED else TEXT,
                    fontsize=9.5, fontweight="bold", ha="center", va="center")
        left += frac
    ax.set_xlim(0, 1)
    ax.set_ylim(-0.6, 0.6)
    ax.set_yticks([])
    ax.set_xticks([])
    for s in ("top", "right", "left", "bottom"):
        ax.spines[s].set_visible(False)
    ax.set_title(f"{name}: class balance, {total} rows", color=MUTED, fontsize=10,
                 loc="left", pad=8)
    _save(fig, path)
