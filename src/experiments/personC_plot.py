"""
Plotting for Extension C. Reads results/personC_optimizer/summary.json and
each per-run history.json, and writes two figures to figures/:

  personC_optimizer_modelwise.png   final test acc vs num_params, one curve
                                    per optimizer x noise.
  personC_optimizer_epochwise.png   epoch-wise test acc trace at one fixed
                                    width per noise level, comparing Adam
                                    vs SGD trajectories.
"""

import argparse
import json
import os
import sys
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def load_summary(path):
    with open(path) as f:
        return json.load(f)


def load_history(run_dir):
    path = os.path.join(run_dir, "history.json")
    with open(path) as f:
        return json.load(f)


def plot_modelwise(summary, results_dir, output):
    # group by (optimizer, noise) -> sorted by num_params, averaged across seeds
    groups = defaultdict(lambda: defaultdict(list))
    for s in summary:
        groups[(s["optimizer"], s["noise"])][s["num_params"]].append(s)

    fig, ax = plt.subplots(figsize=(9.5, 5.5))
    style = {
        ("sgd", 0.0): {"color": "#1f4e79", "marker": "o", "ls": "-"},
        ("adam", 0.0): {"color": "#1f4e79", "marker": "s", "ls": "--"},
        ("sgd", 0.15): {"color": "#c1272d", "marker": "o", "ls": "-"},
        ("adam", 0.15): {"color": "#c1272d", "marker": "s", "ls": "--"},
    }

    for (opt, noise), per_p in sorted(groups.items()):
        params = sorted(per_p.keys())
        mean_test = [np.mean([r["test_acc"] for r in per_p[p]]) for p in params]
        std_test = [np.std([r["test_acc"] for r in per_p[p]]) for p in params]
        st = style.get((opt, noise), {"color": "gray", "marker": "o", "ls": "-"})
        label = f"{opt.upper()}  noise={noise:.0%}"
        ax.errorbar(params, mean_test, yerr=std_test, fmt=st["marker"],
                    linestyle=st["ls"], color=st["color"], label=label,
                    markersize=6, linewidth=1.8, capsize=3)

    ax.set_xscale("log")
    ax.set_xlabel("Number of CNN parameters (log scale)")
    ax.set_ylabel("Final test accuracy (%)")
    ax.set_title("Adam vs SGD — final test accuracy on CIFAR-10 (n=4000)")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(fontsize=10)
    plt.tight_layout()
    plt.savefig(output, bbox_inches="tight", dpi=150)
    plt.close()
    print(f"Wrote {output}")


def plot_epochwise(summary, results_dir, output, fixed_width=16):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=True)
    style = {
        "sgd": {"color": "#1f4e79", "ls": "-"},
        "adam": {"color": "#c1272d", "ls": "--"},
    }
    noise_titles = {0.0: "Clean labels", 0.15: "15% label noise"}

    noises = sorted({s["noise"] for s in summary})
    for ax, noise in zip(axes, noises):
        for opt in ["sgd", "adam"]:
            # average across seeds for fixed width
            traces = []
            for s in summary:
                if (s["optimizer"] == opt and s["noise"] == noise
                        and s["width"] == fixed_width):
                    run_dir = os.path.join(results_dir, s["run_id"])
                    rec = load_history(run_dir)
                    traces.append(rec["history"]["test_acc"])
            if not traces:
                continue
            min_len = min(len(t) for t in traces)
            arr = np.array([t[:min_len] for t in traces])
            mean = arr.mean(0)
            std = arr.std(0)
            epochs = np.arange(1, min_len + 1)
            ax.plot(epochs, mean, color=style[opt]["color"],
                    linestyle=style[opt]["ls"], linewidth=2,
                    label=f"{opt.upper()}")
            ax.fill_between(epochs, mean - std, mean + std,
                            color=style[opt]["color"], alpha=0.15)
        ax.set_title(f"{noise_titles.get(noise, noise)}  (width={fixed_width})")
        ax.set_xlabel("Epoch")
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=10)
    axes[0].set_ylabel("Test accuracy (%)")
    plt.suptitle("Epoch-wise dynamics: Adam vs SGD on CIFAR-10",
                 fontsize=13, y=1.02)
    plt.tight_layout()
    plt.savefig(output, bbox_inches="tight", dpi=150)
    plt.close()
    print(f"Wrote {output}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", default="./results/personC_optimizer")
    parser.add_argument("--figures-dir", default="./figures")
    parser.add_argument("--fixed-width", type=int, default=16)
    args = parser.parse_args()
    os.makedirs(args.figures_dir, exist_ok=True)
    summary = load_summary(os.path.join(args.results_dir, "summary.json"))
    plot_modelwise(summary, args.results_dir,
                   os.path.join(args.figures_dir, "personC_optimizer_modelwise.png"))
    plot_epochwise(summary, args.results_dir,
                   os.path.join(args.figures_dir, "personC_optimizer_epochwise.png"),
                   fixed_width=args.fixed_width)


if __name__ == "__main__":
    main()
