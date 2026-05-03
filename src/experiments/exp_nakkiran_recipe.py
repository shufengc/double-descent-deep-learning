"""
Shufeng Chen (sc5739) -- EECS 6699 Final Project
exp_nakkiran_recipe.py

Two experiments designed to close the NN double-descent gap:

  Experiment A — Nakkiran Recipe (model-wise DD on ResNet)
    Exact setup from Nakkiran et al. 2021, Appendix B.2:
    Adam lr=1e-4, augmentation ON, noise=15%, 4000 epochs, k={1,2,4,8}

  Experiment B — Augmentation Ablation (explains why previous NN exps failed)
    4 conditions (augment × noise), ResNet k={1,4,8}, 500 epochs each
    Produces a 2x2 diagnostic figure

Usage (GPU recommended):
    python3 -m src.experiments.exp_nakkiran_recipe --exp A
    python3 -m src.experiments.exp_nakkiran_recipe --exp B
    python3 -m src.experiments.exp_nakkiran_recipe --exp smoke   # k=1, 100ep sanity check
    python3 -m src.experiments.exp_nakkiran_recipe --exp all
"""

import sys, os, json, argparse, time
from datetime import datetime
import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.models import ResNet
from src.data import get_cifar10, corrupt_labels, make_subset

plt.rcParams.update({
    "figure.figsize": (12, 5), "font.size": 12, "axes.titlesize": 13,
    "axes.labelsize": 12, "legend.fontsize": 10, "lines.linewidth": 2,
    "figure.dpi": 150,
})

# ---------------------------------------------------------------------------
# Constants (Nakkiran et al. 2021, Appendix B.2)
# ---------------------------------------------------------------------------

DATA_DIR   = "./data"
N_TRAIN    = 4000
BATCH_SIZE = 256
LR         = 1e-4          # Adam, constant (no scheduler)
WEIGHT_DECAY = 0.0

EXPA_K_VALS  = [1, 2, 4, 8]
EXPA_EPOCHS  = 4000
EXPA_NOISE   = 0.15

EXPB_K_VALS  = [1, 4, 8]
EXPB_EPOCHS  = 500

EXPB_CONDITIONS = [
    {"augment": True,  "noise": 0.15, "label": "Nakkiran\n(aug+noise)"},
    {"augment": True,  "noise": 0.00, "label": "Clean+Aug\n(aug only)"},
    {"augment": False, "noise": 0.15, "label": "Our old setup\n(noise only)"},
    {"augment": False, "noise": 0.00, "label": "Baseline\n(neither)"},
]

EVAL_INTERVAL = 100   # evaluate test acc every N epochs during training


# ---------------------------------------------------------------------------
# Training helpers
# ---------------------------------------------------------------------------

def _make_data(augment, noise, n_train, seed=42):
    """Build train/test DataLoaders for given augment and noise settings."""
    train_full, test_set = get_cifar10(data_dir=DATA_DIR, augment=augment)
    train_full = corrupt_labels(train_full, noise, seed=seed)
    train_subset = make_subset(train_full, n_train, seed=seed)
    num_workers = min(4, os.cpu_count() or 1)
    pin_memory  = torch.cuda.is_available()
    train_loader = DataLoader(
        train_subset, batch_size=BATCH_SIZE, shuffle=True,
        num_workers=num_workers, pin_memory=pin_memory,
        persistent_workers=(num_workers > 0))
    test_loader = DataLoader(
        test_set, batch_size=512, shuffle=False,
        num_workers=num_workers, pin_memory=pin_memory,
        persistent_workers=(num_workers > 0))
    return train_loader, test_loader


def _eval(model, loader, device, criterion):
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            out  = model(x)
            total_loss += criterion(out, y).item() * x.size(0)
            correct    += out.argmax(1).eq(y).sum().item()
            total      += x.size(0)
    return total_loss / total, 100.0 * correct / total


def _train(model, train_loader, test_loader, epochs, device,
           eval_interval=EVAL_INTERVAL, log_interval=100):
    """
    Standard DataLoader training with Adam (lr=1e-4, wd=0).
    Test accuracy evaluated every eval_interval epochs (NaN otherwise).
    """
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    model.to(device)

    history = {"epoch": [], "train_loss": [], "train_acc": [],
               "test_loss": [], "test_acc": []}
    t0 = time.time()

    for ep in range(1, epochs + 1):
        model.train()
        tl, tc, tt = 0.0, 0, 0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            out  = model(x)
            loss = criterion(out, y)
            loss.backward()
            optimizer.step()
            tl += loss.item() * x.size(0)
            tc += out.argmax(1).eq(y).sum().item()
            tt += x.size(0)

        train_acc  = 100.0 * tc / tt
        train_loss = tl / tt

        if ep % eval_interval == 0 or ep == epochs:
            test_loss, test_acc = _eval(model, test_loader, device, criterion)
        else:
            test_loss = test_acc = float("nan")

        history["epoch"].append(ep)
        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["test_loss"].append(test_loss)
        history["test_acc"].append(test_acc)

        if ep % log_interval == 0 or ep == epochs:
            elapsed = time.time() - t0
            te_str  = f"{test_acc:.1f}%" if not np.isnan(test_acc) else "..."
            print(f"  ep {ep:5d}/{epochs} | "
                  f"tr_acc {train_acc:.1f}% | te_acc {te_str} | "
                  f"elapsed {elapsed/60:.1f}min", flush=True)

    return history


def _param_count(k):
    model = ResNet(k=k, num_classes=10)
    return sum(p.numel() for p in model.parameters())


# ---------------------------------------------------------------------------
# Experiment A: Nakkiran Recipe
# ---------------------------------------------------------------------------

def exp_A(out_dir="results/exp_nakkiran_modelwise", k_vals=None, epochs=None,
          smoke=False):
    if smoke:
        k_vals = [1]
        epochs = 100
        out_dir = "results/exp_nakkiran_modelwise_smoke"
        print("\n" + "="*70)
        print("  SMOKE TEST — k=1, 100 epochs (Nakkiran recipe)")
        print("  Pass: test_acc > 30% at any checkpoint")
        print("="*70)
    else:
        if k_vals is None:  k_vals = EXPA_K_VALS
        if epochs is None:  epochs = EXPA_EPOCHS
        print("\n" + "="*70)
        print(f"  EXPERIMENT A — Nakkiran Recipe (model-wise DD)")
        print(f"  k={k_vals}, epochs={epochs}, noise={EXPA_NOISE}, aug=True")
        print(f"  Adam lr={LR}, wd={WEIGHT_DECAY}, n_train={N_TRAIN}")
        print("="*70)

    os.makedirs(out_dir, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"  Device: {device}", flush=True)

    all_results = []
    results_path = os.path.join(out_dir, "results.json")

    # Load pre-existing partial results so we can resume
    if os.path.exists(results_path):
        with open(results_path) as f:
            all_results = json.load(f)
        done_ks = {r["k"] for r in all_results}
        print(f"  Resuming: already done k={sorted(done_ks)}", flush=True)
    else:
        done_ks = set()

    train_loader, test_loader = _make_data(
        augment=True, noise=EXPA_NOISE, n_train=N_TRAIN)

    for k in k_vals:
        if k in done_ks:
            print(f"\n  Skipping k={k} (already done)", flush=True)
            continue

        n_params = _param_count(k)
        ratio    = n_params / N_TRAIN
        print(f"\n  --- k={k} | params={n_params:,} | p/n={ratio:.1f} ---", flush=True)

        t_start = time.time()
        model   = ResNet(k=k, num_classes=10)
        history = _train(model, train_loader, test_loader, epochs, device)
        elapsed = time.time() - t_start

        # Best and final test accuracy (ignoring NaN checkpoints)
        test_accs_valid = [v for v in history["test_acc"] if not np.isnan(v)]
        best_test_acc   = max(test_accs_valid) if test_accs_valid else float("nan")
        final_test_acc  = test_accs_valid[-1]  if test_accs_valid else float("nan")
        final_train_acc = history["train_acc"][-1]

        row = {
            "k": k, "n_params": n_params, "p_over_n": round(ratio, 2),
            "n_train": N_TRAIN, "noise": EXPA_NOISE, "epochs": epochs,
            "final_train_acc": round(final_train_acc, 2),
            "final_test_acc":  round(final_test_acc, 2),
            "best_test_acc":   round(best_test_acc, 2),
            "elapsed_sec":     round(elapsed, 1),
            "history": history,
        }
        all_results.append(row)

        with open(results_path, "w") as f:
            json.dump(all_results, f)
        print(f"  Saved k={k}: best_test={best_test_acc:.1f}% "
              f"final_train={final_train_acc:.1f}% ({elapsed/60:.1f}min)", flush=True)

    _plot_exp_A(all_results, out_dir, smoke)
    return all_results


def _plot_exp_A(results, out_dir, smoke=False):
    if len(results) < 2 and not smoke:
        return  # not enough data to plot a curve yet

    ks         = [r["k"]              for r in results]
    n_params   = [r["n_params"]       for r in results]
    best_test  = [r["best_test_acc"]  for r in results]
    final_test = [r["final_test_acc"] for r in results]
    train_acc  = [r["final_train_acc"]for r in results]
    test_err   = [100 - v             for v in best_test]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Panel 1: Test error vs param count (model complexity)
    ax = axes[0]
    ax.plot(n_params, test_err, "o-", color="#d62728", label="Test error (100−best_acc)")
    ax.set_xlabel("Number of parameters")
    ax.set_ylabel("Test error (%)")
    ax.set_title("Exp A: Model-wise Double Descent\n(Nakkiran recipe — aug + 15% noise)")
    ax.set_xscale("log")
    for k, p, e in zip(ks, n_params, test_err):
        ax.annotate(f"k={k}", (p, e), textcoords="offset points", xytext=(4, 4), fontsize=9)
    ax.axvline(N_TRAIN, color="gray", linestyle="--", alpha=0.6, label=f"n_train={N_TRAIN}")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Panel 2: Train vs test accuracy
    ax = axes[1]
    ax.plot(n_params, train_acc, "s-", color="#1f77b4", label="Train acc (final)")
    ax.plot(n_params, best_test, "o-", color="#d62728", label="Test acc (best)")
    ax.set_xlabel("Number of parameters")
    ax.set_ylabel("Accuracy (%)")
    ax.set_title("Exp A: Train vs Test Accuracy")
    ax.set_xscale("log")
    ax.legend()
    ax.grid(True, alpha=0.3)

    title_suffix = " [SMOKE TEST]" if smoke else ""
    fig.suptitle(
        f"Nakkiran Recipe: Adam lr=1e-4, aug=True, noise=15%, n_train={N_TRAIN}"
        f"{title_suffix}",
        fontsize=12, y=1.02)
    plt.tight_layout()
    out_path = os.path.join(out_dir, "dd_curves.png")
    plt.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved plot: {out_path}", flush=True)


# ---------------------------------------------------------------------------
# Experiment B: Augmentation Ablation
# ---------------------------------------------------------------------------

def exp_B(out_dir="results/exp_augmentation_ablation"):
    print("\n" + "="*70)
    print(f"  EXPERIMENT B — Augmentation Ablation")
    print(f"  k={EXPB_K_VALS}, epochs={EXPB_EPOCHS}, 4 conditions")
    print(f"  Adam lr={LR}, wd={WEIGHT_DECAY}, n_train={N_TRAIN}")
    print("="*70)

    os.makedirs(out_dir, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"  Device: {device}", flush=True)

    results_path = os.path.join(out_dir, "results.json")
    all_results  = []

    if os.path.exists(results_path):
        with open(results_path) as f:
            all_results = json.load(f)
        done_keys = {(r["k"], r["augment"], r["noise"]) for r in all_results}
        print(f"  Resuming: {len(done_keys)} configs already done", flush=True)
    else:
        done_keys = set()

    for cond in EXPB_CONDITIONS:
        augment = cond["augment"]
        noise   = cond["noise"]
        label   = cond["label"].replace("\n", " ")

        # DataLoaders are per (augment, noise) condition
        train_loader, test_loader = _make_data(
            augment=augment, noise=noise, n_train=N_TRAIN)

        for k in EXPB_K_VALS:
            key = (k, augment, noise)
            if key in done_keys:
                print(f"\n  Skipping k={k}, aug={augment}, noise={noise} (done)", flush=True)
                continue

            n_params = _param_count(k)
            print(f"\n  --- k={k}, aug={augment}, noise={noise} "
                  f"| params={n_params:,} ---", flush=True)

            t_start = time.time()
            model   = ResNet(k=k, num_classes=10)
            history = _train(model, train_loader, test_loader, EXPB_EPOCHS, device,
                             eval_interval=50, log_interval=50)
            elapsed = time.time() - t_start

            test_accs_valid = [v for v in history["test_acc"] if not np.isnan(v)]
            best_test_acc   = max(test_accs_valid) if test_accs_valid else float("nan")
            final_test_acc  = test_accs_valid[-1]  if test_accs_valid else float("nan")
            final_train_acc = history["train_acc"][-1]

            row = {
                "k": k, "n_params": n_params, "augment": augment, "noise": noise,
                "label": label, "epochs": EXPB_EPOCHS, "n_train": N_TRAIN,
                "final_train_acc": round(final_train_acc, 2),
                "final_test_acc":  round(final_test_acc, 2),
                "best_test_acc":   round(best_test_acc, 2),
                "elapsed_sec":     round(elapsed, 1),
                "history": history,
            }
            all_results.append(row)

            with open(results_path, "w") as f:
                json.dump(all_results, f)
            print(f"  Saved k={k}, aug={augment}, noise={noise}: "
                  f"best_test={best_test_acc:.1f}% ({elapsed/60:.1f}min)", flush=True)

    _plot_exp_B(all_results, out_dir)
    return all_results


def _plot_exp_B(results, out_dir):
    conditions = EXPB_CONDITIONS
    ks = EXPB_K_VALS
    n_params_by_k = {k: _param_count(k) for k in ks}

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c"]

    for ci, cond in enumerate(conditions):
        ax  = axes[ci // 2][ci % 2]
        aug = cond["augment"]
        noi = cond["noise"]
        lbl = cond["label"].replace("\n", " ")

        # Gather results for this condition
        cond_rows = [r for r in results if r["augment"] == aug and r["noise"] == noi]
        cond_rows.sort(key=lambda r: r["k"])

        if not cond_rows:
            ax.set_title(f"Cond {ci+1}: {lbl}\n(no data yet)")
            continue

        k_list     = [r["k"]              for r in cond_rows]
        params_l   = [n_params_by_k[k]    for k in k_list]
        train_accs = [r["final_train_acc"] for r in cond_rows]
        test_accs  = [r["best_test_acc"]   for r in cond_rows]

        ax.plot(params_l, train_accs, "s-", color="#1f77b4", label="Train acc")
        ax.plot(params_l, test_accs,  "o-", color="#d62728", label="Best test acc")
        ax.set_xscale("log")
        ax.set_xlabel("Params")
        ax.set_ylabel("Accuracy (%)")
        ax.set_title(f"Cond {ci+1}: {lbl}")
        ax.set_ylim(0, 105)
        for k, p, ta, te in zip(k_list, params_l, train_accs, test_accs):
            ax.annotate(f"k={k}", (p, te), xytext=(4, 4),
                        textcoords="offset points", fontsize=8)
        ax.axvline(N_TRAIN, color="gray", linestyle="--", alpha=0.5)
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)

    fig.suptitle(
        f"Exp B: Augmentation × Noise Ablation\n"
        f"ResNet k={{{','.join(map(str,ks))}}}, {EXPB_EPOCHS} epochs, Adam lr={LR}",
        fontsize=13)
    plt.tight_layout()
    out_path = os.path.join(out_dir, "dd_curves.png")
    plt.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved plot: {out_path}", flush=True)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Nakkiran recipe experiments (A + B)")
    parser.add_argument("--exp", choices=["A", "B", "smoke", "all"], default="all",
                        help="Which experiment to run")
    parser.add_argument("--data_dir", default="./data")
    args = parser.parse_args()

    global DATA_DIR
    DATA_DIR = args.data_dir

    start = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"\nStarted: {start}")
    print(f"Device: {'cuda' if torch.cuda.is_available() else 'cpu'}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    if args.exp == "smoke":
        results = exp_A(smoke=True)
        test_accs = [r["best_test_acc"] for r in results]
        max_acc = max(test_accs) if test_accs else 0.0
        if max_acc > 30.0:
            print(f"\n  SMOKE TEST PASSED: best_test_acc={max_acc:.1f}% > 30%")
            print("  Recipe works — proceed to full Experiment A.")
        else:
            print(f"\n  SMOKE TEST FAILED: best_test_acc={max_acc:.1f}% <= 30%")
            print("  Recipe may be broken. Check config before full run.")

    elif args.exp == "A":
        exp_A()

    elif args.exp == "B":
        exp_B()

    elif args.exp == "all":
        exp_A()
        exp_B()

    end = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"\nFinished: {end}")


if __name__ == "__main__":
    main()
