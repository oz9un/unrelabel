from __future__ import annotations
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")   # non-interactive backend for headless runs
import matplotlib.pyplot as plt

from unrelabel.attacks.base import AttackResult
from unrelabel.utils.theme import PALETTE, CLASS_COLORS, apply_theme


class Visualizer:
    def __init__(self):
        apply_theme()

    def plot_confusion_matrices(
        self, result: AttackResult, output_dir: Path
    ) -> Path:
        clean_cm = np.array(result.confusion_matrices["clean"])
        poisoned_cm = np.array(result.confusion_matrices["poisoned"])
        n_classes = clean_cm.shape[0]

        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        fig.suptitle("Confusion Matrices: Clean vs Poisoned", color=PALETTE["white"], fontsize=14)

        for ax, cm, title in zip(axes, [clean_cm, poisoned_cm], ["Clean Model", "Poisoned Model"]):
            im = ax.imshow(cm, cmap="Blues")
            ax.set_title(title, color=PALETTE["white"])
            ax.set_xlabel("Predicted", color=PALETTE["white"])
            ax.set_ylabel("Actual", color=PALETTE["white"])
            for i in range(n_classes):
                for j in range(n_classes):
                    ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                            color=PALETTE["white"], fontsize=11)
            plt.colorbar(im, ax=ax)

        plt.tight_layout()
        out = Path(output_dir) / "confusion_matrices.png"
        plt.savefig(out, bbox_inches="tight")
        plt.close()
        return out

    def plot_accuracy_curve(
        self, result: AttackResult, output_dir: Path
    ) -> Path:
        if not result.sweep_results:
            return self._plot_single_accuracy(result, output_dir)

        rates = [r["poison_rate"] * 100 for r in result.sweep_results]
        accuracies = [r["poisoned_accuracy"] for r in result.sweep_results]

        fig, ax = plt.subplots(figsize=(9, 5))
        ax.axhline(result.clean_accuracy, color=PALETTE["success"],
                   linestyle="--", linewidth=1.5, label=f"Clean baseline ({result.clean_accuracy:.3f})")
        ax.plot(rates, accuracies, marker="o", color=PALETTE["danger"],
                linewidth=2, markersize=7, label="Poisoned accuracy")
        ax.fill_between(rates, accuracies, result.clean_accuracy,
                        alpha=0.15, color=PALETTE["danger"])
        ax.set_title("Accuracy Degradation vs Poison Rate", color=PALETTE["white"], fontsize=13)
        ax.set_xlabel("Poison Rate (%)", color=PALETTE["white"])
        ax.set_ylabel("Accuracy on Clean Test Set", color=PALETTE["white"])
        ax.set_ylim(0, 1.05)
        ax.legend()
        plt.tight_layout()

        out = Path(output_dir) / "accuracy_curve.png"
        plt.savefig(out, bbox_inches="tight")
        plt.close()
        return out

    def _plot_single_accuracy(
        self, result: AttackResult, output_dir: Path
    ) -> Path:
        fig, ax = plt.subplots(figsize=(6, 5))
        bars = ax.bar(
            ["Clean", "Poisoned"],
            [result.clean_accuracy, result.poisoned_accuracy],
            color=[PALETTE["success"], PALETTE["danger"]],
            width=0.4,
        )
        for bar, val in zip(bars, [result.clean_accuracy, result.poisoned_accuracy]):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                    f"{val:.3f}", ha="center", va="bottom", color=PALETTE["white"])
        ax.set_title("Clean vs Poisoned Accuracy", color=PALETTE["white"], fontsize=13)
        ax.set_ylabel("Accuracy", color=PALETTE["white"])
        ax.set_ylim(0, 1.1)
        plt.tight_layout()
        out = Path(output_dir) / "accuracy_curve.png"
        plt.savefig(out, bbox_inches="tight")
        plt.close()
        return out

    def plot_poisoned_scatter(
        self,
        X: np.ndarray,
        y_original: np.ndarray,
        y_poisoned: np.ndarray,
        flipped_indices: np.ndarray,
        output_dir: Path,
    ) -> Path:
        if X.shape[1] < 2:
            return None

        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        fig.suptitle("Training Data: Original vs Poisoned", color=PALETTE["white"], fontsize=14)

        not_flipped = np.ones(len(y_original), dtype=bool)
        not_flipped[flipped_indices] = False

        for ax, y_plot, title in zip(
            axes,
            [y_original, y_poisoned],
            ["Original Labels", "Poisoned Labels"]
        ):
            classes = np.unique(y_original)
            for i, cls in enumerate(classes):
                mask = (y_plot == cls) & not_flipped
                ax.scatter(X[mask, 0], X[mask, 1],
                           color=CLASS_COLORS[i % len(CLASS_COLORS)],
                           s=40, alpha=0.7, label=f"Class {cls}")
            if len(flipped_indices) > 0:
                ax.scatter(X[flipped_indices, 0], X[flipped_indices, 1],
                           color=PALETTE["flipped"], marker="X", s=80,
                           linewidths=1, edgecolors=PALETTE["white"],
                           alpha=0.9, label="Flipped")
            ax.set_title(title, color=PALETTE["white"])
            ax.set_xlabel("Feature 1", color=PALETTE["white"])
            ax.set_ylabel("Feature 2", color=PALETTE["white"])
            ax.legend()

        plt.tight_layout()
        out = Path(output_dir) / "poisoned_scatter.png"
        plt.savefig(out, bbox_inches="tight")
        plt.close()
        return out
