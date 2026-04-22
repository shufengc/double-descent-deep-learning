"""
Comprehensive double descent experiments.
Combines random features (fast, clean theory) with neural networks
for a complete picture of the phenomenon.

Experiments:
  1. Model-wise DD (random features): sweep p/n ratio
  2. Model-wise DD (neural network): sweep CNN width
  3. Effect of noise on DD peak (random features)
  4. Sample-wise DD (random features): fix p, sweep n
  5. Epoch-wise DD (neural network): train CNN for many epochs
"""

import sys
import os
import json
import argparse
import time

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.models import CNN
from src.data import get_cifar10, get_mnist, corrupt_labels, make_subset, make_loaders
from src.trainer import Trainer


plt.rcParams.update({
    "figure.figsize": (10, 6),
    "font.size": 13,
    "axes.titlesize": 15,
    "axes.labelsize": 13,
    "legend.fontsize": 11,
    "lines.linewidth": 2,
    "figure.dpi": 150,
})


def random_fourier_features(X, D, sigma=5.0, seed=42):
    rng = np.random.RandomState(seed)
    d = X.shape[1]
    W = rng.randn(d, D).astype(np.float64) / sigma
    b = rng.uniform(0, 2 * np.pi, D).astype(np.float64)
    proj = X.astype(np.float64) @ W + b
    np.nan_to_num(proj, copy=False, nan=0.0, posinf=1e10, neginf=-1e10)
    return np.cos(proj) * np.sqrt(2.0 / D)


def min_norm_solution(Phi, y, lam=1e-10):
    n, p = Phi.shape
    if p >= n:
        K = Phi @ Phi.T + lam * np.eye(n)
        alpha = np.linalg.solve(K, y)
        return Phi.T @ alpha
    else:
        G = Phi.T @ Phi + lam * np.eye(p)
        return np.linalg.solve(G, Phi.T @ y)


def _parse_csv_ints(s, default=None):
    if not s or not str(s).strip():
        return default if default is not None else [42]
    return [int(x.strip()) for x in str(s).split(",") if x.strip()]


def _parse_csv_floats(s):
    return [float(x.strip()) for x in str(s).split(",") if x.strip()]


def nn_learning_rate(args):
    if getattr(args, "nn_lr", None) is not None:
        return args.nn_lr
    return 0.05 if args.optimizer == "sgd" else 0.001


def _aggregate_rows_by_keys(rows, key_fields=("p_over_n", "lambda")):
    """rows: list of dicts with same keys; group by key_fields, mean/std for numeric vals."""
    from collections import defaultdict
    groups = defaultdict(list)
    for row in rows:
        key = tuple(row[k] for k in key_fields)
        groups[key].append(row)
    out = []

    def _sort_key(k):
        return tuple(
            (round(x, 12) if isinstance(x, float) else x) for x in k)

    for key in sorted(groups.keys(), key=_sort_key):
        grp = groups[key]
        base = {field: val for field, val in zip(key_fields, key)}
        for field in grp[0].keys():
            if field in key_fields or field == "seed":
                continue
            vals = [g[field] for g in grp if field in g
                    and isinstance(g[field], (int, float, np.floating))]
            if not vals:
                continue
            arr = np.array(vals, dtype=np.float64)
            base[f"{field}_mean"] = float(arr.mean())
            base[f"{field}_std"] = float(arr.std(ddof=0))
        base["n_seeds"] = len(grp)
        out.append(base)
    return out


def _rows_for_lambda(agg_list, lam):
    out = []
    for r in agg_list:
        rl = r.get("lambda")
        if rl is None:
            continue
        if np.isclose(rl, lam, rtol=0.0, atol=1e-12):
            out.append(r)
    return out


def load_mnist_numpy(data_dir="./data", n_train=None, noise_rate=0.0, seed=42):
    import torchvision
    import torchvision.transforms as transforms
    transform = transforms.Compose([transforms.ToTensor()])
    train = torchvision.datasets.MNIST(root=data_dir, train=True, download=True, transform=transform)
    test = torchvision.datasets.MNIST(root=data_dir, train=False, download=True, transform=transform)

    X_train = train.data.numpy().reshape(-1, 784).astype(np.float64) / 255.0
    y_train = train.targets.numpy().copy()
    X_test = test.data.numpy().reshape(-1, 784).astype(np.float64) / 255.0
    y_test = test.targets.numpy().copy()

    rng = np.random.RandomState(seed)
    if n_train and n_train < len(X_train):
        idx = rng.choice(len(X_train), n_train, replace=False)
        X_train, y_train = X_train[idx], y_train[idx]

    if noise_rate > 0:
        n = len(y_train)
        corrupt = rng.choice(n, int(noise_rate * n), replace=False)
        for i in corrupt:
            old = y_train[i]
            y_train[i] = rng.choice([c for c in range(10) if c != old])

    Y_train = np.zeros((len(y_train), 10))
    Y_train[np.arange(len(y_train)), y_train] = 1.0
    Y_test = np.zeros((len(y_test), 10))
    Y_test[np.arange(len(y_test)), y_test] = 1.0

    return X_train, Y_train, y_train, X_test, Y_test, y_test


def _exp1_run_one_seed(args, seed, lambdas):
    noise_rates = [0.0, 0.1, 0.2]
    n = args.n_train
    ratios = [0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 0.8, 0.9, 0.95, 0.98,
              1.0, 1.02, 1.05, 1.1, 1.2, 1.5, 2.0, 3.0, 5.0, 8.0]
    all_results = {}
    for nr in noise_rates:
        rows = []
        X_tr, Y_tr, y_tr, X_te, Y_te, y_te = load_mnist_numpy(
            args.data_dir, n, nr, seed=seed)
        for ratio in ratios:
            D = max(1, int(ratio * n))
            Phi_tr = random_fourier_features(X_tr, D, sigma=5.0, seed=seed)
            Phi_te = random_fourier_features(X_te, D, sigma=5.0, seed=seed)
            for lam in lambdas:
                w = min_norm_solution(Phi_tr, Y_tr, lam=lam)
                w_norm = float(np.linalg.norm(w))
                pred_tr = Phi_tr @ w
                pred_te = Phi_te @ w
                train_mse = np.mean((Y_tr - pred_tr)**2)
                test_mse = np.mean((Y_te - pred_te)**2)
                train_acc = np.mean(np.argmax(pred_tr, 1) == y_tr) * 100
                test_acc = np.mean(np.argmax(pred_te, 1) == y_te) * 100
                rows.append({
                    "seed": seed,
                    "D": D, "p_over_n": ratio, "lambda": lam,
                    "w_norm": w_norm,
                    "train_mse": float(train_mse), "test_mse": float(test_mse),
                    "train_acc": float(train_acc), "test_acc": float(test_acc),
                })
        all_results[nr] = rows
    return all_results


def exp1_model_wise_rff(args):
    """Model-wise DD with random Fourier features (multi-seed, ridge λ, ||w||)."""
    print("\n" + "="*70)
    print("  EXP 1: MODEL-WISE DD (Random Features)")
    print("="*70)

    seeds = _parse_csv_ints(args.seeds, [args.seed])
    lambdas = _parse_csv_floats(args.rff_lambdas)
    if not lambdas:
        lambdas = [1e-10]

    n = args.n_train
    per_seed = {}
    for seed in seeds:
        print(f"\n--- seed = {seed} ---")
        per_seed[str(seed)] = _exp1_run_one_seed(args, seed, lambdas)
        row1 = next(
            (x for x in per_seed[str(seed)][0.0]
             if abs(x["p_over_n"] - 1.0) < 1e-9 and np.isclose(x["lambda"], lambdas[0])),
            None)
        if row1:
            print(f"  (clean, p/n=1, λ={lambdas[0]:.1e}) test_mse={row1['test_mse']:.4f} "
                  f"||w||={row1['w_norm']:.2e}")

    aggregated = {}
    for nr in [0.0, 0.1, 0.2]:
        merged = []
        for seed in seeds:
            merged.extend(per_seed[str(seed)][nr])
        aggregated[nr] = _aggregate_rows_by_keys(merged, ("p_over_n", "lambda"))

    payload = {
        "seeds": seeds,
        "rff_lambdas": lambdas,
        "per_seed": {k: {str(nr): v for nr, v in val.items()}
                     for k, val in per_seed.items()},
        "aggregated": {str(k): v for k, v in aggregated.items()},
    }

    out = os.path.join(args.output_dir, "exp1_model_wise_rff")
    os.makedirs(out, exist_ok=True)
    with open(os.path.join(out, "results.json"), "w") as f:
        json.dump(payload, f, indent=2)

    # Plot: multi-λ uses line styles; multi-seed + single λ uses shaded band
    ls_cycle = ["-", "--", "-.", ":"]
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    colors = {"0.0": "tab:blue", "0.1": "tab:orange", "0.2": "tab:red"}
    for nr_str, agg in sorted(payload["aggregated"].items(), key=lambda x: float(x[0])):
        for li, lam in enumerate(lambdas):
            sub = _rows_for_lambda(agg, lam)
            if not sub:
                continue
            r = sorted(sub, key=lambda x: x["p_over_n"])
            x = [d["p_over_n"] for d in r]
            y_mse = [d.get("test_mse_mean", d.get("test_mse")) for d in r]
            y_err = [100 - d.get("test_acc_mean", d.get("test_acc")) for d in r]
            label = f"noise={float(nr_str):.0%}"
            if len(lambdas) > 1:
                label += f", λ={lam:.1e}"
            linestyle = ls_cycle[li % len(ls_cycle)]
            c = colors.get(nr_str, "gray")
            axes[0].plot(x, y_mse, linestyle=linestyle, marker="o", color=c,
                         label=label, markersize=3)
            if len(seeds) > 1 and len(lambdas) == 1 and r[0].get("test_mse_std") is not None:
                lo = [m - s for m, s in zip(y_mse, [d["test_mse_std"] for d in r])]
                hi = [m + s for m, s in zip(y_mse, [d["test_mse_std"] for d in r])]
                axes[0].fill_between(x, lo, hi, color=c, alpha=0.15)
            axes[1].plot(x, y_err, linestyle=linestyle, marker="o", color=c,
                         label=label, markersize=3)

    for ax in axes:
        ax.axvline(x=1.0, color="gray", linestyle=":", alpha=0.7)
        ax.set_xlabel("p/n")
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)
    axes[0].set_ylabel("Test MSE")
    axes[0].set_yscale("log")
    axes[0].set_title("Model-Wise DD: Test MSE")
    axes[1].set_ylabel("Test Error (%)")
    axes[1].set_title("Model-Wise DD: Classification Error")
    plt.suptitle(
        f"RFF on MNIST (n={n}, seeds={seeds}, λ grid={lambdas})",
        fontsize=13, y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(out, "dd_curves.png"), bbox_inches="tight", dpi=150)
    plt.close()

    # ||w|| vs p/n (primary noise levels; one panel per λ or overlay)
    fig, ax = plt.subplots(figsize=(10, 5))
    for nr_str, agg in sorted(payload["aggregated"].items(), key=lambda x: float(x[0])):
        for li, lam in enumerate(lambdas):
            sub = _rows_for_lambda(agg, lam)
            if not sub:
                continue
            r = sorted(sub, key=lambda x: x["p_over_n"])
            x = [d["p_over_n"] for d in r]
            y = [d.get("w_norm_mean", d.get("w_norm")) for d in r]
            c = colors.get(nr_str, "gray")
            ax.plot(x, y, linestyle=ls_cycle[li % len(ls_cycle)], marker="s", color=c,
                    label=f"noise={float(nr_str):.0%} λ={lam:.1e}", markersize=3)
    ax.axvline(x=1.0, color="gray", linestyle=":", alpha=0.7)
    ax.set_xlabel("p/n")
    ax.set_ylabel("||w||")
    ax.set_yscale("log")
    ax.set_title("Min-norm solution norm vs p/n")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(os.path.join(out, "w_norm_vs_pn.png"), bbox_inches="tight", dpi=150)
    plt.close()

    print(f"Saved to {out}")
    return payload


def _exp2_run_one_seed(args, seed, lambdas, D_fixed, sample_sizes):
    rows = []
    for n in sample_sizes:
        X_tr, Y_tr, y_tr, X_te, Y_te, y_te = load_mnist_numpy(
            args.data_dir, n, noise_rate=0.1, seed=seed)
        Phi_tr = random_fourier_features(X_tr, D_fixed, sigma=5.0, seed=seed)
        Phi_te = random_fourier_features(X_te, D_fixed, sigma=5.0, seed=seed)
        ratio = D_fixed / n
        for lam in lambdas:
            w = min_norm_solution(Phi_tr, Y_tr, lam=lam)
            w_norm = float(np.linalg.norm(w))
            pred_te = Phi_te @ w
            test_mse = np.mean((Y_te - pred_te)**2)
            test_acc = np.mean(np.argmax(pred_te, 1) == y_te) * 100
            pred_tr = Phi_tr @ w
            train_mse = np.mean((Y_tr - pred_tr)**2)
            rows.append({
                "seed": seed,
                "n_samples": n, "D": D_fixed, "p_over_n": round(ratio, 4),
                "lambda": lam,
                "w_norm": w_norm,
                "test_mse": float(test_mse), "train_mse": float(train_mse),
                "test_acc": float(test_acc),
            })
        if n == D_fixed:
            chunk = rows[-len(lambdas):]
            rp = next((x for x in chunk if np.isclose(x["lambda"], lambdas[0])), None)
            if rp:
                print(f"  seed={seed} n=D={D_fixed} λ={lambdas[0]:.1e} "
                      f"test_mse={rp['test_mse']:.4f} ||w||={rp['w_norm']:.2e}")
    return rows


def exp2_sample_wise_rff(args):
    """Sample-wise DD: fix D, sweep n (multi-seed, ridge λ, ||w||)."""
    print("\n" + "="*70)
    print("  EXP 2: SAMPLE-WISE DD (Random Features)")
    print("="*70)

    D_fixed = 500
    sample_sizes = [100, 200, 300, 400, 450, 480, 490, 500, 510, 520,
                    550, 600, 700, 1000, 1500, 2000, 4000]
    seeds = _parse_csv_ints(args.seeds, [args.seed])
    lambdas = _parse_csv_floats(args.rff_lambdas)
    if not lambdas:
        lambdas = [1e-10]

    per_seed_rows = {}
    for seed in seeds:
        print(f"\n--- seed = {seed} ---")
        per_seed_rows[str(seed)] = _exp2_run_one_seed(
            args, seed, lambdas, D_fixed, sample_sizes)

    merged = []
    for seed in seeds:
        merged.extend(per_seed_rows[str(seed)])
    aggregated = _aggregate_rows_by_keys(merged, ("n_samples", "lambda"))

    payload = {
        "D": D_fixed,
        "seeds": seeds,
        "rff_lambdas": lambdas,
        "per_seed": per_seed_rows,
        "aggregated": aggregated,
    }

    out = os.path.join(args.output_dir, "exp2_sample_wise_rff")
    os.makedirs(out, exist_ok=True)
    with open(os.path.join(out, "results.json"), "w") as f:
        json.dump(payload, f, indent=2)

    ls_cycle = ["-", "--", "-.", ":"]
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for li, lam in enumerate(lambdas):
        sub = _rows_for_lambda(aggregated, lam)
        if not sub:
            continue
        r = sorted(sub, key=lambda x: x["n_samples"])
        ns = [d["n_samples"] for d in r]
        tm = [d.get("test_mse_mean", d.get("test_mse")) for d in r]
        trm = [d.get("train_mse_mean", d.get("train_mse")) for d in r]
        te = [100 - d.get("test_acc_mean", d.get("test_acc")) for d in r]
        linestyle = ls_cycle[li % len(ls_cycle)]
        label_te = f"Test λ={lam:.1e}" if len(lambdas) > 1 else "Test"
        label_tr = f"Train λ={lam:.1e}" if len(lambdas) > 1 else "Train"
        axes[0].plot(ns, tm, linestyle=linestyle, marker="o", color="red",
                       markersize=4, label=label_te)
        axes[0].plot(ns, trm, linestyle=linestyle, marker="s", color="blue",
                     alpha=0.45, markersize=3, label=label_tr)
        if len(seeds) > 1 and len(lambdas) == 1 and r[0].get("test_mse_std") is not None:
            lo = [m - s for m, s in zip(tm, [d["test_mse_std"] for d in r])]
            hi = [m + s for m, s in zip(tm, [d["test_mse_std"] for d in r])]
            axes[0].fill_between(ns, lo, hi, color="red", alpha=0.12)
        axes[1].plot(ns, te, linestyle=linestyle, marker="o", color="red",
                     markersize=5, label=f"λ={lam:.1e}" if len(lambdas) > 1 else "Test err")

    axes[0].axvline(x=D_fixed, color="gray", linestyle=":", alpha=0.7, label=f"n=D={D_fixed}")
    axes[1].axvline(x=D_fixed, color="gray", linestyle=":", alpha=0.7, label=f"n=D={D_fixed}")
    axes[0].set_xlabel("Number of Training Samples (n)")
    axes[0].set_ylabel("MSE")
    axes[0].set_yscale("log")
    axes[0].set_title("Sample-Wise DD: MSE")
    axes[0].legend(fontsize=8)
    axes[0].grid(True, alpha=0.3)

    axes[1].set_xlabel("Number of Training Samples (n)")
    axes[1].set_ylabel("Test Error (%)")
    axes[1].set_title("Sample-Wise DD: Error")
    axes[1].legend(fontsize=8)
    axes[1].grid(True, alpha=0.3)

    plt.suptitle(
        f"Sample-Wise DD (D={D_fixed}, 10% noise, seeds={seeds}, λ={lambdas})",
        fontsize=13, y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(out, "dd_curves.png"), bbox_inches="tight", dpi=150)
    plt.close()
    print(f"Saved to {out}")
    return payload


def exp3_nn_model_wise(args):
    """Model-wise DD with CNNs (multi-seed, augment / optimizer / WD from CLI)."""
    print("\n" + "="*70)
    print("  EXP 3: MODEL-WISE DD (Neural Network)")
    print("="*70)

    device = torch.device("cuda" if torch.cuda.is_available() else
                          "mps" if torch.backends.mps.is_available() else "cpu")
    lr = nn_learning_rate(args)
    seeds = _parse_csv_ints(args.seeds, [args.seed])
    print(f"Device: {device} | augment={args.augment} | opt={args.optimizer} | "
          f"wd={args.weight_decay} | lr={lr} | seeds={seeds}")

    n = args.n_train_nn
    noise_rates = [0.0, 0.2]
    widths = [1, 2, 3, 4, 6, 8, 12, 16, 24, 32]
    all_rows = []

    for seed in seeds:
        print(f"\n--- seed = {seed} ---")
        for nr in noise_rates:
            print(f"\n--- noise = {nr:.0%} ---")
            train_full, test_set = get_cifar10(
                data_dir=args.data_dir, augment=args.augment)
            if nr > 0:
                train_full = corrupt_labels(train_full, nr, seed=seed)
            train_set = make_subset(train_full, n, seed=seed)
            train_loader, test_loader = make_loaders(train_set, test_set, batch_size=256)

            for width in widths:
                torch.manual_seed(seed)
                np.random.seed(seed)
                model = CNN(num_classes=10, num_filters=width, input_channels=3)
                p = model.count_parameters()
                ratio = p / n

                trainer = Trainer(
                    model, device=device, lr=lr, weight_decay=args.weight_decay,
                    optimizer_type=args.optimizer)
                t0 = time.time()
                history = trainer.train(
                    train_loader, test_loader,
                    epochs=args.epochs_nn,
                    log_interval=max(1, args.epochs_nn // 3),
                    verbose=True)
                elapsed = time.time() - t0

                r = {
                    "seed": seed,
                    "noise": nr,
                    "width": width,
                    "num_params": p,
                    "p_over_n": round(ratio, 4),
                    "train_acc": history["train_acc"][-1],
                    "test_acc": history["test_acc"][-1],
                    "train_loss": history["train_loss"][-1],
                    "test_loss": history["test_loss"][-1],
                }
                all_rows.append(r)
                print(f"  w={width:3d} p={p:>8,} p/n={ratio:.3f} "
                      f"train_err={100-r['train_acc']:.1f}% "
                      f"test_err={100-r['test_acc']:.1f}% ({elapsed:.0f}s)")

    aggregated = _aggregate_rows_by_keys(all_rows, ("noise", "width"))
    payload = {
        "config": {
            "augment": args.augment,
            "optimizer": args.optimizer,
            "weight_decay": args.weight_decay,
            "lr": lr,
            "seeds": seeds,
        },
        "per_run": all_rows,
        "aggregated": aggregated,
    }

    out = os.path.join(args.output_dir, "exp3_nn_model_wise")
    os.makedirs(out, exist_ok=True)
    with open(os.path.join(out, "results.json"), "w") as f:
        json.dump(payload, f, indent=2)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for nr in noise_rates:
        sub = [d for d in aggregated if d["noise"] == nr]
        r = sorted(sub, key=lambda x: x["num_params_mean"])
        params = [d["num_params_mean"] for d in r]
        color = "tab:blue" if nr == 0.0 else "tab:red"
        label = f"noise={nr:.0%}"
        ta = [d["test_acc_mean"] for d in r]
        tr = [d["train_acc_mean"] for d in r]
        tl = [d["test_loss_mean"] for d in r]
        axes[0].plot(params, [100 - x for x in ta], "o-", color=color,
                     label=f"Test ({label})", markersize=5)
        axes[0].plot(params, [100 - x for x in tr], "s--", color=color,
                     alpha=0.4, label=f"Train ({label})", markersize=4)
        if len(seeds) > 1 and r[0].get("test_acc_std") is not None:
            stds = [d["test_acc_std"] for d in r]
            lo = [100 - (m + s) for m, s in zip(ta, stds)]
            hi = [100 - (m - s) for m, s in zip(ta, stds)]
            axes[0].fill_between(params, lo, hi, color=color, alpha=0.12)
        axes[1].plot(params, tl, "o-", color=color,
                     label=f"Test ({label})", markersize=5)

    for ax in axes:
        ax.axvline(x=n, color="gray", linestyle=":", alpha=0.7, label=f"n={n}")
        ax.set_xscale("log")
        ax.set_xlabel("Number of Parameters")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
    axes[0].set_ylabel("Error (%)")
    axes[0].set_title("NN Model-Wise DD: Error")
    axes[1].set_ylabel("Test Loss")
    axes[1].set_title("NN Model-Wise DD: Loss")

    plt.suptitle(
        f"CNN CIFAR-10 (n={n}, aug={args.augment}, {args.optimizer}, lr={lr})",
        fontsize=12, y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(out, "dd_curves.png"), bbox_inches="tight", dpi=150)
    plt.close()
    print(f"Saved to {out}")
    return payload


def exp4_epoch_wise_nn(args):
    """Epoch-wise DD: train CNNs (multi-seed; CLI matches exp3)."""
    print("\n" + "="*70)
    print("  EXP 4: EPOCH-WISE DD (Neural Network)")
    print("="*70)

    device = torch.device("cuda" if torch.cuda.is_available() else
                          "mps" if torch.backends.mps.is_available() else "cpu")
    lr = nn_learning_rate(args)
    seeds = _parse_csv_ints(args.seeds, [args.seed])
    print(f"Device: {device} | augment={args.augment} | opt={args.optimizer} | "
          f"wd={args.weight_decay} | lr={lr} | seeds={seeds}")

    n = args.n_train_nn
    widths = [2, 4, 8]
    per_seed_width = []  # list of {seed, width, label, history}

    for seed in seeds:
        train_full, test_set = get_cifar10(
            data_dir=args.data_dir, augment=args.augment)
        train_full = corrupt_labels(train_full, 0.2, seed=seed)
        train_set = make_subset(train_full, n, seed=seed)
        train_loader, test_loader = make_loaders(train_set, test_set, batch_size=256)

        for width in widths:
            torch.manual_seed(seed)
            np.random.seed(seed)
            model = CNN(num_classes=10, num_filters=width, input_channels=3)
            p = model.count_parameters()
            label = f"w={width} (p={p:,}, p/n={p/n:.2f})"
            print(f"\nseed={seed} {label}")

            trainer = Trainer(
                model, device=device, lr=lr, weight_decay=args.weight_decay,
                optimizer_type=args.optimizer)
            history = trainer.train(
                train_loader, test_loader,
                epochs=args.epochs_epoch,
                log_interval=max(1, args.epochs_epoch // 10),
                verbose=True)
            per_seed_width.append({
                "seed": seed, "width": width, "label": label, "history": history})

    by_width = {}
    for width in widths:
        hists = [x["history"] for x in per_seed_width if x["width"] == width]
        if not hists:
            continue
        ta = np.array([h["test_acc"] for h in hists], dtype=np.float64)
        tl = np.array([h["test_loss"] for h in hists], dtype=np.float64)
        by_width[width] = {
            "epoch": hists[0]["epoch"],
            "test_acc_mean": ta.mean(0).tolist(),
            "test_acc_std": ta.std(0).tolist(),
            "test_loss_mean": tl.mean(0).tolist(),
            "test_loss_std": tl.std(0).tolist(),
        }

    payload = {
        "config": {
            "augment": args.augment,
            "optimizer": args.optimizer,
            "weight_decay": args.weight_decay,
            "lr": lr,
            "seeds": seeds,
        },
        "per_run_histories": {
            f"seed{x['seed']}_w{x['width']}": x["history"] for x in per_seed_width
        },
        "aggregated_by_width": by_width,
    }

    out = os.path.join(args.output_dir, "exp4_epoch_wise_nn")
    os.makedirs(out, exist_ok=True)
    with open(os.path.join(out, "results.json"), "w") as f:
        json.dump(payload, f, indent=2)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    colors = {2: "tab:blue", 4: "tab:orange", 8: "tab:green"}
    for width in widths:
        if width not in by_width:
            continue
        agg = by_width[width]
        epochs = agg["epoch"]
        te = [100 - a for a in agg["test_acc_mean"]]
        axes[0].plot(epochs, te, color=colors.get(width, "gray"),
                     label=f"w={width}")
        if len(seeds) > 1:
            lo = [100 - (m + s) for m, s in zip(agg["test_acc_mean"], agg["test_acc_std"])]
            hi = [100 - (m - s) for m, s in zip(agg["test_acc_mean"], agg["test_acc_std"])]
            axes[0].fill_between(epochs, lo, hi, color=colors.get(width, "gray"), alpha=0.12)
        axes[1].plot(epochs, agg["test_loss_mean"], color=colors.get(width, "gray"),
                     label=f"w={width}")
        if len(seeds) > 1:
            tl = np.array(agg["test_loss_mean"])
            ts = np.array(agg["test_loss_std"])
            axes[1].fill_between(epochs, tl - ts, tl + ts,
                                   color=colors.get(width, "gray"), alpha=0.12)

    for ax in axes:
        ax.set_xlabel("Epoch")
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)
    axes[0].set_ylabel("Test Error (%)")
    axes[0].set_title("Epoch-Wise DD: Error")
    axes[1].set_ylabel("Test Loss")
    axes[1].set_title("Epoch-Wise DD: Loss")

    plt.suptitle(
        f"CNN CIFAR-10 (n={n}, 20% noise, aug={args.augment}, {args.optimizer})",
        fontsize=12, y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(out, "dd_curves.png"), bbox_inches="tight", dpi=150)
    plt.close()
    print(f"Saved to {out}")
    return payload


def main():
    parser = argparse.ArgumentParser(
        description="Double descent experiments (RFF + CNN). "
        "Use --seeds for multi-seed runs; --rff-lambdas for ridge sweep; "
        "--augment / --optimizer / --weight-decay for NN ablations.")
    parser.add_argument("--experiments", type=str, default="1,2,3,4",
                        help="Which experiments to run (1=model_rff, 2=sample_rff, 3=nn_model, 4=nn_epoch)")
    parser.add_argument("--n-train", type=int, default=1000,
                        help="Training samples for random features experiments")
    parser.add_argument("--n-train-nn", type=int, default=4000,
                        help="Training samples for NN experiments")
    parser.add_argument("--epochs-nn", type=int, default=500,
                        help="Epochs for NN model-wise experiment")
    parser.add_argument("--epochs-epoch", type=int, default=1000,
                        help="Epochs for epoch-wise experiment")
    parser.add_argument("--data-dir", type=str, default="./data")
    parser.add_argument("--output-dir", type=str, default="./results")
    parser.add_argument("--seed", type=int, default=42,
                        help="Fallback if --seeds is empty")
    parser.add_argument("--seeds", type=str, default="42",
                        help="Comma-separated random seeds (RFF + NN)")
    parser.add_argument("--rff-lambdas", type=str, default="1e-10",
                        help="Comma-separated ridge λ for RFF (exp 1–2), e.g. 1e-10,1e-4,0.01")
    parser.add_argument("--augment", action="store_true",
                        help="Use CIFAR-10 train augmentation for NN experiments (3–4)")
    parser.add_argument("--optimizer", type=str, default="adam",
                        choices=["adam", "sgd"],
                        help="Optimizer for NN experiments (3–4)")
    parser.add_argument("--weight-decay", type=float, default=0.0,
                        dest="weight_decay",
                        help="L2 weight decay for NN experiments (3–4)")
    parser.add_argument("--nn-lr", type=float, default=None,
                        help="Learning rate for NN; default 0.001 (adam) or 0.05 (sgd)")
    args = parser.parse_args()

    exps = [int(x) for x in args.experiments.split(",")]

    if 1 in exps:
        exp1_model_wise_rff(args)
    if 2 in exps:
        exp2_sample_wise_rff(args)
    if 3 in exps:
        exp3_nn_model_wise(args)
    if 4 in exps:
        exp4_epoch_wise_nn(args)

    print("\n" + "="*70)
    print("ALL EXPERIMENTS COMPLETE")
    print(f"Results in: {args.output_dir}")
    print("="*70)


if __name__ == "__main__":
    main()
