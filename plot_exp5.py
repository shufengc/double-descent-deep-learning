"""
Regenerate Experiment 5 plots from saved results.json
"""
import json
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams.update({
    "figure.figsize": (10, 6),
    "font.size": 13,
    "axes.titlesize": 15,
    "axes.labelsize": 13,
    "legend.fontsize": 11,
    "lines.linewidth": 2,
    "figure.dpi": 150,
})

out = "results/exp5_architecture_comparison"
with open(os.path.join(out, "results.json")) as f:
    all_results = json.load(f)

n = 4000
colors = {"MLP": "tab:green", "CNN": "tab:blue", "ResNet": "tab:purple"}
markers = {"MLP": "D", "CNN": "o", "ResNet": "s"}

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

for arch, results in all_results.items():
    r = sorted(results, key=lambda x: x["num_params"])
    params = [d["num_params"] for d in r]

    axes[0].plot(params, [100 - d["test_acc"] for d in r],
                 marker=markers[arch], linestyle="-", color=colors[arch],
                 label=f"{arch} (Test)", markersize=6)
    axes[0].plot(params, [100 - d["train_acc"] for d in r],
                 marker=markers[arch], linestyle="--", color=colors[arch],
                 alpha=0.4, label=f"{arch} (Train)", markersize=4)

    axes[1].plot(params, [d["test_loss"] for d in r],
                 marker=markers[arch], linestyle="-", color=colors[arch],
                 label=f"{arch} (Test)", markersize=6)

for ax in axes:
    ax.axvline(x=n, color="red", linestyle=":", linewidth=1.5, alpha=0.8, label=f"p = n = {n}")
    ax.set_xscale("log")
    ax.set_xlabel("Number of Parameters")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

axes[0].set_ylabel("Error (%)")
axes[0].set_title("Arch. Comparison: Classification Error")
axes[1].set_ylabel("Test Loss")
axes[1].set_title("Arch. Comparison: Test Loss")

plt.suptitle(
    f"MLP vs CNN vs ResNet on CIFAR-10 (n={n}, 20% noise, 200 epochs)",
    fontsize=14, y=1.02
)
plt.tight_layout()
path = os.path.join(out, "dd_curves.png")
plt.savefig(path, bbox_inches="tight", dpi=150)
plt.close()
print(f"Plot saved to: {path}")
