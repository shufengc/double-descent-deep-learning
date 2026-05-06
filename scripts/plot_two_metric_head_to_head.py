"""Two-metric (final vs peek) head-to-head bar chart on full main/ k grid."""
import json
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

base = Path("results/dd_recovery_5090_focused/main")
runs = []
for p in sorted(base.glob("k*_n4000_ep2000_s*/results.json")):
    r = json.loads(p.read_text())
    runs.append({
        "k": r["config"]["k"],
        "seed": r["config"]["seed"],
        "final": r.get("final_test_acc"),
        "peek": r.get("best_test_acc"),
    })

by_k = defaultdict(list)
for r in runs:
    by_k[r["k"]].append(r)

ks = sorted(by_k)
print(f"k values: {ks}")
for k in ks:
    print(f"  k={k}: {len(by_k[k])} runs, "
          f"finals={[r['final'] for r in by_k[k]]}, "
          f"peeks={[r['peek'] for r in by_k[k]]}")


def safe(xs, fn):
    xs = [x for x in xs if x is not None]
    return float(fn(xs)) if xs else float("nan")


fig, ax = plt.subplots(figsize=(11, 5.5))
x = np.arange(len(ks))
width = 0.38

f_means = [safe([r["final"] for r in by_k[k]], np.mean) for k in ks]
f_stds  = [safe([r["final"] for r in by_k[k]], np.std)  for k in ks]
p_means = [safe([r["peek"]  for r in by_k[k]], np.mean) for k in ks]
p_stds  = [safe([r["peek"]  for r in by_k[k]], np.std)  for k in ks]

ax.bar(x - width/2, f_means, width, yerr=f_stds, capsize=3,
       label="Final-epoch test", color="tab:green", alpha=0.85)
ax.bar(x + width/2, p_means, width, yerr=p_stds, capsize=3,
       label="Best-over-training (test-peeking)", color="tab:red", alpha=0.85)

# Connect with line to highlight the gap
ax.plot(x - width/2, f_means, "o-", color="tab:green", linewidth=1.5, markersize=4, alpha=0.6)
ax.plot(x + width/2, p_means, "o-", color="tab:red",   linewidth=1.5, markersize=4, alpha=0.6)

ax.set_xticks(x)
ax.set_xticklabels([f"{k:g}" for k in ks])
ax.set_xlabel("k (width multiplier)")
ax.set_ylabel("Test accuracy (%)")
ax.set_title("NN model-wise DD: metric choice changes the apparent shape\n"
             "(fractional-k ResNet, n=4000 CIFAR-10, 15% noise, 2 seeds per k)")
ax.grid(axis="y", alpha=0.25)
ax.legend(loc="lower right")
ax.set_ylim(15, 65)
fig.tight_layout()

out = Path("figures/two_metric_head_to_head.png")
out.parent.mkdir(exist_ok=True)
fig.savefig(out, dpi=180, bbox_inches="tight")
print(f"wrote {out}")
