"""
Extension D — Why classical generalization bounds fail.

Generates one conceptual figure overlaying:
  (left axis)  a stylised classical bound such as O(sqrt(p / n))
               (representative of VC / Rademacher bounds, monotone increasing
               in parameter count)
  (right axis) the OBSERVED test MSE from Exp 1 (RFF model-wise on MNIST,
               10% noise, n=1000), which exhibits the textbook double descent
               shape: rises, peaks at p/n=1, then second-descends.

The point: classical bounds predict a monotone increase as p/n grows, while
the observed curve falls below them in the overparameterised regime. Modern
norm-based / minimum-norm / benign-overfitting analyses are the right lens.

Loads results/exp1_model_wise_rff/results.json. No new training required.
Writes figures/personD_bound_vs_observed.png.
"""

import argparse
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp1-results", default="./results/exp1_model_wise_rff/results.json")
    parser.add_argument("--noise-key", default="0.1",
                        help="Which noise rate from exp1 to plot")
    parser.add_argument("--output", default="./figures/personD_bound_vs_observed.png")
    args = parser.parse_args()

    with open(args.exp1_results) as f:
        exp1 = json.load(f)
    rows = sorted(exp1[args.noise_key], key=lambda r: r["p_over_n"])
    pn = np.array([r["p_over_n"] for r in rows])
    test_mse = np.array([r["test_mse"] for r in rows])

    # Stylised classical bound: B(p, n) ~ C * sqrt(p / n) + base_train_err
    # The constant is chosen so the bound matches observed test_mse at
    # p/n = 0.1, then it rises monotonically with p. This deliberately
    # overstates the bound's accuracy in the underparameterised regime so the
    # disagreement in the overparameterised regime is visually unambiguous.
    base = test_mse[0]
    target_at_anchor = test_mse[1] if len(test_mse) > 1 else test_mse[0]
    anchor_pn = pn[1] if len(pn) > 1 else pn[0]
    C = max(target_at_anchor - base, 1e-4) / np.sqrt(anchor_pn)
    bound = base + C * np.sqrt(pn)

    fig, ax_left = plt.subplots(figsize=(9.5, 5.5))
    ax_right = ax_left.twinx()

    ln_obs = ax_left.plot(pn, test_mse, "o-", color="#c1272d",
                          markersize=5, linewidth=2,
                          label="Observed test MSE (RFF, MNIST, 10% noise)")
    ax_left.set_yscale("log")
    ax_left.set_xlabel("p / n  (parameters per training example)")
    ax_left.set_ylabel("Observed test MSE  (log scale)", color="#c1272d")
    ax_left.tick_params(axis="y", labelcolor="#c1272d")
    ax_left.axvline(x=1.0, color="gray", linestyle=":", alpha=0.6)
    ax_left.text(1.02, ax_left.get_ylim()[1] * 0.5, " p/n = 1\n interpolation",
                 fontsize=9, color="gray")

    ln_bnd = ax_right.plot(pn, bound, "s--", color="#1f4e79",
                           markersize=4, linewidth=2,
                           label=r"Stylised classical bound: $C\sqrt{p/n}$")
    ax_right.set_ylabel("Stylised classical bound", color="#1f4e79")
    ax_right.tick_params(axis="y", labelcolor="#1f4e79")

    # Shade the regime where the bound would predict catastrophic failure but
    # the observed curve actually does well.
    overparam_mask = pn >= 1.5
    if overparam_mask.any():
        ax_left.axvspan(pn[overparam_mask][0], pn[-1],
                        alpha=0.08, color="green",
                        label="Overparameterised regime: classical bound diverges, observed test MSE descends")

    ax_left.set_xscale("log")
    ax_left.grid(True, which="both", alpha=0.25)
    handles = ln_obs + ln_bnd
    labels = [h.get_label() for h in handles]
    ax_left.legend(handles, labels, loc="upper left", fontsize=10)
    ax_left.set_title("Classical capacity bounds vs observed double descent\n"
                      "(VC/Rademacher predictions diverge with p; observed curve second-descends)")
    plt.tight_layout()
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    plt.savefig(args.output, bbox_inches="tight", dpi=150)
    plt.close()
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
