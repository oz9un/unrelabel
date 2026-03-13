"""Shared color palette and matplotlib style for all unrelabel plots."""
import matplotlib.pyplot as plt

PALETTE = {
    "background": "#0d1117",
    "surface":    "#161b22",
    "accent":     "#58a6ff",
    "danger":     "#f85149",
    "warning":    "#e3b341",
    "success":    "#3fb950",
    "muted":      "#8b949e",
    "white":      "#f0f6fc",
    "class_0":    "#388bfd",
    "class_1":    "#db6d28",
    "class_2":    "#3fb950",
    "class_3":    "#a371f7",
    "flipped":    "#f85149",
}

CLASS_COLORS = [
    PALETTE["class_0"],
    PALETTE["class_1"],
    PALETTE["class_2"],
    PALETTE["class_3"],
]


def apply_theme():
    plt.rcParams.update({
        "figure.facecolor":   PALETTE["background"],
        "axes.facecolor":     PALETTE["surface"],
        "axes.edgecolor":     PALETTE["muted"],
        "axes.labelcolor":    PALETTE["white"],
        "text.color":         PALETTE["white"],
        "xtick.color":        PALETTE["muted"],
        "ytick.color":        PALETTE["muted"],
        "grid.color":         PALETTE["muted"],
        "grid.alpha":         0.15,
        "legend.facecolor":   PALETTE["surface"],
        "legend.edgecolor":   PALETTE["muted"],
        "legend.labelcolor":  PALETTE["white"],
        "figure.dpi":         120,
    })
