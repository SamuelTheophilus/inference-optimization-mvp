"""
plot_results.py
---------------
Reads results/benchmark_results.json and produces two charts:
  1. Latency comparison (p50/p95/p99) across configurations
  2. Accuracy delta (%) across configurations

Run after benchmark.py:
    python benchmarks/plot_results.py
"""

import json
import os
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

RESULTS_PATH = "results/benchmark_results.json"
CHARTS_PATH = "results/"


def label(r: dict) -> str:
    return f"{r['model']}\n{r['precision']}\n{r['workers']}w"


def plot_latency(results):
    labels = [label(r) for r in results]
    p50 = [r["p50_ms"] for r in results]
    p95 = [r["p95_ms"] for r in results]
    p99 = [r["p99_ms"] for r in results]

    x = np.arange(len(labels))
    width = 0.25

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.bar(x - width, p50, width, label="p50", color="#4C9BE8")
    ax.bar(x,         p95, width, label="p95", color="#F4A261")
    ax.bar(x + width, p99, width, label="p99", color="#E76F51")

    ax.set_xlabel("Configuration")
    ax.set_ylabel("Latency (ms)")
    ax.set_title("Inference Latency by Configuration")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=9)
    ax.legend()
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    path = os.path.join(CHARTS_PATH, "latency_comparison.png")
    plt.savefig(path, dpi=150)
    print(f"[plot] Saved {path}")
    plt.close()


def plot_accuracy(results):
    # Only show rows with an accuracy score
    acc_results = [r for r in results if "accuracy_pct" in r]
    if not acc_results:
        print("[plot] No accuracy data to plot")
        return

    labels = [label(r) for r in acc_results]
    accuracies = [r["accuracy_pct"] for r in acc_results]
    colors = ["#2A9D8F" if a >= 98.0 else "#E63946" for a in accuracies]

    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(labels, accuracies, color=colors)
    ax.axhline(y=98.0, color="#E63946", linestyle="--", linewidth=1.5, label="98% accuracy threshold")

    ax.set_xlabel("Configuration")
    ax.set_ylabel("Accuracy Score (%)")
    ax.set_title("Accuracy Score by Configuration (higher is better)")
    ax.set_ylim(min(accuracies) - 1, 101)
    ax.legend()
    ax.grid(axis="y", alpha=0.3)

    for bar, val in zip(bars, accuracies):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.05,
                f"{val:.2f}%", ha="center", va="bottom", fontsize=9)

    plt.tight_layout()
    path = os.path.join(CHARTS_PATH, "accuracy_delta.png")
    plt.savefig(path, dpi=150)
    print(f"[plot] Saved {path}")
    plt.close()


def main():
    if not os.path.exists(RESULTS_PATH):
        print(f"[plot] No results found at {RESULTS_PATH}. Run benchmark.py first.")
        return

    with open(RESULTS_PATH) as f:
        results = json.load(f)

    os.makedirs(CHARTS_PATH, exist_ok=True)
    plot_latency(results)
    plot_accuracy(results)


if __name__ == "__main__":
    main()
