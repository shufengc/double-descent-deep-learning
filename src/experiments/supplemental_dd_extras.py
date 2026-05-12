"""
Supplemental double-descent experiments.

  S1_ood     : ID vs OOD test (pixel Gaussian noise) on RFF+MNIST, model-wise p/n
  S2_order   : Sample-wise DD with random vs easy-to-hard vs hard-to-easy ordering
  S3_early   : CIFAR CNN: test at last epoch vs test at best val epoch (trajectory / early stop)

Run from project root:
    python3 -m src.experiments.supplemental_dd_extras --experiments S1,S2,S3
    python3 -m src.experiments.supplemental_dd_extras --experiments S1 --quick
"""

import argparse
import json
import os
import sys

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from src.models import CNN
from src.data import get_cifar10, corrupt_labels
from src.experiments.comprehensive_dd import (
    random_fourier_features,
    min_norm_solution,
    load_mnist_numpy,
)

plt.rcParams.update({
    "figure.figsize": (12, 5), "font.size": 12, "axes.titlesize": 13,
    "axes.labelsize": 12, "legend.fontsize": 10, "lines.linewidth": 2,
    "figure.dpi": 150,
})

# ---------------------------------------------------------------------------
# MNIST: full-train pool for ordered subsets (not passed through load_mnist with n)
# ---------------------------------------------------------------------------

def _load_mnist_train_pool(data_dir, noise_rate, seed):
    import torchvision
    import torchvision.transforms as T
    train = torchvision.datasets.MNIST(
        root=data_dir, train=True, download=True, transform=T.ToTensor())
    X = train.data.numpy().reshape(-1, 784).astype(np.float64) / 255.0
    y = train.targets.numpy().copy()
    rng = np.random.RandomState(seed)
    if noise_rate > 0:
        corrupt = rng.choice(len(y), int(noise_rate * len(y)), replace=False)
        for i in corrupt:
            y[i] = rng.choice([c for c in range(10) if c != y[i]])
    Y_oh = np.eye(10)[y]
    return X, Y_oh, y


def _load_mnist_test_id_ood(data_dir, ood_std):
    import torchvision
    import torchvision.transforms as T
    test = torchvision.datasets.MNIST(
        root=data_dir, train=False, download=True, transform=T.ToTensor())
    X_id = test.data.numpy().reshape(-1, 784).astype(np.float64) / 255.0
    y = test.targets.numpy()
    Y_oh = np.eye(10)[y]
    rng = np.random.RandomState(12345)  # fixed for reproducible OOD split
    noise = rng.randn(*X_id.shape).astype(np.float64) * ood_std
    X_ood = np.clip(X_id + noise, 0.0, 1.0)
    return X_id, Y_oh, y, X_ood


# ---------------------------------------------------------------------------
# S1: Model-wise with ID and OOD test
# ---------------------------------------------------------------------------

def exp_S1_ood_rff(args):
    print("\n" + "="*70)
    print("  S1: Model-wise DD — ID vs OOD test (RFF, MNIST)")
    print("="*70)

    ood_std = args.ood_std
    n = args.n_train
    out = os.path.join(args.output_dir, "supp1_ood_id_rff")
    os.makedirs(out, exist_ok=True)

    if getattr(args, "quick", False):
        ratios = [0.1, 0.5, 0.9, 1.0, 1.1, 2.0, 5.0, 8.0]
    else:
        ratios = [0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 0.8, 0.9, 0.95, 0.98,
                    1.0, 1.02, 1.05, 1.1, 1.2, 1.5, 2.0, 3.0, 5.0, 8.0]
    noise_rates = [0.0, 0.1] if not getattr(args, "quick", False) else [0.1]
    all_results = {}

    X_id, Y_oh, y_te, X_ood = _load_mnist_test_id_ood(args.data_dir, ood_std)

    for nr in noise_rates:
        print(f"\n--- label noise = {nr:.0%} ---", flush=True)
        X_tr, Y_tr, y_tr, _, _, _ = load_mnist_numpy(
            args.data_dir, n, nr, args.seed)

        rows = []
        for ratio in ratios:
            D = max(1, int(ratio * n))
            Phi_tr = random_fourier_features(X_tr, D, sigma=5.0, seed=args.seed)
            w = min_norm_solution(Phi_tr, Y_tr)

            Phi_id = random_fourier_features(X_id, D, sigma=5.0, seed=args.seed)
            Phi_ood = random_fourier_features(X_ood, D, sigma=5.0, seed=args.seed)
            pred_id = Phi_id @ w
            pred_ood = Phi_ood @ w
            mse_id = float(np.mean((Y_oh - pred_id) ** 2))
            mse_ood = float(np.mean((Y_oh - pred_ood) ** 2))
            acc_id = float(np.mean(np.argmax(pred_id, 1) == y_te) * 100)
            acc_ood = float(np.mean(np.argmax(pred_ood, 1) == y_te) * 100)
            # train diagnostics
            pred_tr = Phi_tr @ w
            train_mse = float(np.mean((Y_tr - pred_tr) ** 2))

            rows.append({
                "D": D, "p_over_n": ratio,
                "test_mse_id": mse_id, "test_mse_ood": mse_ood,
                "test_acc_id": acc_id, "test_acc_ood": acc_ood,
                "ood_mse_ratio": mse_ood / max(mse_id, 1e-20),
                "train_mse": train_mse,
            })
            print(f"  D={D:5d} p/n={ratio:.2f} | MSE id={mse_id:.4f} ood={mse_ood:.4f} | "
                  f"acc id={acc_id:.1f}% ood={acc_ood:.1f}%", flush=True)

        all_results[str(nr)] = rows

    with open(os.path.join(out, "results.json"), "w") as f:
        json.dump(
            {k: v for k, v in all_results.items()}, f, indent=2)

    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for nr_str, rlist in sorted(all_results.items(), key=lambda x: float(x[0])):
        rlist = sorted(rlist, key=lambda d: d["p_over_n"])
        x = [d["p_over_n"] for d in rlist]
        axes[0].plot(x, [d["test_mse_id"] for d in rlist], "o-",
                    label=f"ID (noise y={float(nr_str):.0%})", markersize=4)
        axes[0].plot(x, [d["test_mse_ood"] for d in rlist], "s--",
                    alpha=0.7, label=f"OOD (noise y={float(nr_str):.0%})", markersize=4)
    axes[0].axvline(1.0, color="gray", linestyle=":", alpha=0.6)
    axes[0].set_xlabel("p/n")
    axes[0].set_ylabel("Test MSE")
    axes[0].set_yscale("log")
    axes[0].set_title("S1: ID vs OOD (pixel noise on test) — Test MSE")
    axes[0].legend(fontsize=8, ncol=2)
    axes[0].grid(True, alpha=0.3)

    for nr_str, rlist in sorted(all_results.items(), key=lambda x: float(x[0])):
        rlist = sorted(rlist, key=lambda d: d["p_over_n"])
        x = [d["p_over_n"] for d in rlist]
        axes[1].plot(
            x, [d["ood_mse_ratio"] for d in rlist], "o-",
            label=f" noise y={float(nr_str):.0%}", markersize=4)
    axes[1].axvline(1.0, color="gray", linestyle=":", alpha=0.6)
    axes[1].axhline(1.0, color="black", linestyle=":", alpha=0.4)
    axes[1].set_xlabel("p/n")
    axes[1].set_ylabel("MSE_OOD / MSE_ID")
    axes[1].set_title("OOD degradation factor vs p/n")
    axes[1].set_yscale("log")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    fig.suptitle(
        f"S1: RFF MNIST (n={n}, OOD=Gaussian pixel noise std={ood_std})",
        fontsize=13, y=1.02,
    )
    plt.tight_layout()
    plt.savefig(os.path.join(out, "dd_curves.png"), bbox_inches="tight", dpi=150)
    plt.close()
    print(f"\nSaved → {out}", flush=True)
    return all_results


# ---------------------------------------------------------------------------
# S2: Sample-wise with ordering strategies
# ---------------------------------------------------------------------------

def exp_S2_ordered_sample_wise_rff(args):
    print("\n" + "="*70)
    print("  S2: Sample-wise DD — random vs ordered subsets (RFF, MNIST)")
    print("="*70)

    D_fixed = args.s2_d_fixed
    noise_rate = 0.1
    if getattr(args, "quick", False):
        sample_sizes = [200, 400, 500, 520, 600, 1000, 2000]
    else:
        sample_sizes = [100, 200, 300, 400, 450, 480, 490, 500, 510, 520,
                        550, 600, 700, 1000, 1500, 2000, 4000]
    out = os.path.join(args.output_dir, "supp2_ordered_sample_rff")
    os.makedirs(out, exist_ok=True)

    # Full training pool and test
    X_pool, Y_pool, y_pool = _load_mnist_train_pool(
        args.data_dir, noise_rate, args.seed)
    import torchvision
    import torchvision.transforms as T
    test_ds = torchvision.datasets.MNIST(
        args.data_dir, train=False, download=True, transform=T.ToTensor())
    X_te = test_ds.data.numpy().reshape(-1, 784).astype(np.float64) / 255.0
    y_te = test_ds.targets.numpy()
    Y_te = np.eye(10)[y_te]

    # Per-class centroids in pixel space
    centroids = np.zeros((10, 784), dtype=np.float64)
    counts = np.zeros(10, dtype=np.int64)
    for c in range(10):
        mask = y_pool == c
        centroids[c] = X_pool[mask].mean(0) if mask.any() else 0
        counts[c] = mask.sum()

    # score = negative distance to own class center (higher = easier "typical" of class)
    d2 = np.sum((X_pool - centroids[y_pool]) ** 2, axis=1)
    order_easy_to_hard = np.argsort(d2)  # small distance first
    order_hard_to_easy = order_easy_to_hard[::-1]
    rng = np.random.RandomState(args.seed)

    strategies = {
        "random": lambda n: rng.choice(len(X_pool), n, replace=False),
        "easy_to_hard": lambda n: order_easy_to_hard[:n],
        "hard_to_easy": lambda n: order_hard_to_easy[:n],
    }

    all_out = {name: [] for name in strategies}

    for n in sample_sizes:
        if n > len(X_pool):
            print(f"  skip n={n} > pool", flush=True)
            continue
        print(f"\n--- n={n} (D/n={D_fixed/n:.3f}) ---", flush=True)
        for sname, idx_fn in strategies.items():
            idx = idx_fn(n)
            X_tr, Y_tr = X_pool[idx], Y_pool[idx]
            Phi_tr = random_fourier_features(
                X_tr, D_fixed, sigma=5.0, seed=args.seed)
            Phi_te = random_fourier_features(
                X_te, D_fixed, sigma=5.0, seed=args.seed)
            w = min_norm_solution(Phi_tr, Y_tr)
            pred_te = Phi_te @ w
            test_mse = float(np.mean((Y_te - pred_te) ** 2))
            test_acc = float(np.mean(np.argmax(pred_te, 1) == y_te) * 100)
            pred_tr = Phi_tr @ w
            train_mse = float(np.mean((Y_tr - pred_tr) ** 2))
            row = {
                "n_samples": n, "D": D_fixed, "p_over_n": D_fixed / n,
                "strategy": sname, "test_mse": test_mse, "train_mse": train_mse,
                "test_acc": test_acc,
            }
            all_out[sname].append(row)
            print(f"  {sname:12s} test_mse={test_mse:.4f} acc={test_acc:.1f}%", flush=True)

    with open(os.path.join(out, "results.json"), "w") as f:
        json.dump(all_out, f, indent=2)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    colors = {
        "random": "tab:blue",
        "easy_to_hard": "tab:green",
        "hard_to_easy": "tab:red",
    }
    for sname, rows in all_out.items():
        rows = sorted(rows, key=lambda r: r["n_samples"])
        ns = [r["n_samples"] for r in rows]
        axes[0].plot(
            ns, [r["test_mse"] for r in rows], "o-",
            color=colors.get(sname, "gray"), label=sname, markersize=4)
    axes[0].axvline(
        x=D_fixed, color="gray", linestyle=":", alpha=0.7, label=f"n=D={D_fixed}"
    )
    axes[0].set_xlabel("n (train samples)")
    axes[0].set_ylabel("Test MSE (log)")
    axes[0].set_yscale("log")
    axes[0].set_title("S2: Sample-wise DD — MSE by subset strategy")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    for sname, rows in all_out.items():
        rows = sorted(rows, key=lambda r: r["n_samples"])
        ns = [r["n_samples"] for r in rows]
        axes[1].plot(
            ns, [100 - r["test_acc"] for r in rows], "o-",
            color=colors.get(sname, "gray"), label=sname, markersize=4
        )
    axes[1].axvline(
        x=D_fixed, color="gray", linestyle=":", alpha=0.7, label=f"n=D={D_fixed}"
    )
    axes[1].set_xlabel("n (train samples)")
    axes[1].set_ylabel("Test error %")
    axes[1].set_title("S2: Classification test error")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    fig.suptitle(
        f"S2: D={D_fixed}, label noise=10% | easy/hard = distance to class mean",
        y=1.02, fontsize=12,
    )
    plt.tight_layout()
    plt.savefig(os.path.join(out, "dd_curves.png"), bbox_inches="tight", dpi=150)
    plt.close()
    print(f"\nSaved → {out}", flush=True)
    return all_out


# ---------------------------------------------------------------------------
# S3: Early stopping vs end of training (CIFAR CNN)
# ---------------------------------------------------------------------------

def _train_cnn_with_early_stop(
        train_loader, val_loader, test_loader, width, device,
        epochs, lr, seed, verbose,
):
    torch.manual_seed(seed)
    np.random.seed(seed)
    model = CNN(num_classes=10, num_filters=width, input_channels=3)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=0.0)
    model = model.to(device)

    best_val_acc = -1.0
    best_state = None
    best_ep = 0
    test_at_best_val = 0.0
    test_at_end = 0.0
    val_at_end = 0.0
    last_train_acc = 0.0
    for ep in range(1, epochs + 1):
        model.train()
        tc, tot = 0, 0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            opt.zero_grad()
            o = model(x)
            loss = torch.nn.functional.cross_entropy(o, y)
            loss.backward()
            opt.step()
            tot += x.size(0)
            tc += o.argmax(1).eq(y).sum().item()
        train_acc = 100.0 * tc / tot

        model.eval()
        with torch.no_grad():
            vc, vt = 0, 0
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                o = model(x)
                vt += y.size(0)
                vc += o.argmax(1).eq(y).sum().item()
        val_acc = 100.0 * vc / vt
        with torch.no_grad():
            tec, tet = 0, 0
            for x, y in test_loader:
                x, y = x.to(device), y.to(device)
                o = model(x)
                tet += y.size(0)
                tec += o.argmax(1).eq(y).sum().item()
        test_acc = 100.0 * tec / tet

        if val_acc > best_val_acc + 1e-6:
            best_val_acc = val_acc
            best_ep = ep
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            test_at_best_val = test_acc

        last_train_acc = train_acc
        test_at_end = test_acc
        val_at_end = val_acc

        if verbose and (ep == 1 or ep == epochs or ep % max(1, epochs // 5) == 0):
            print(
                f"  ep{ep:4d} tr{train_acc:.1f}% va{val_acc:.1f}% te{test_acc:.1f}%", flush=True
            )

    if best_state is not None:
        model.load_state_dict(
            {k: v.to(device) for k, v in best_state.items()}
        )
        model.eval()
        with torch.no_grad():
            tec, tet = 0, 0
            for x, y in test_loader:
                x, y = x.to(device), y.to(device)
                o = model(x)
                tet += y.size(0)
                tec += o.argmax(1).eq(y).sum().item()
        test_at_best_val = 100.0 * tec / tet

    return {
        "width": width,
        "p": int(model.count_parameters()),
        "best_val_acc": best_val_acc,
        "best_epoch": best_ep,
        "test_acc_at_best_val": test_at_best_val,
        "test_acc_end": test_at_end,
        "val_acc_end": val_at_end,
        "train_acc_end": last_train_acc,
    }


def exp_S3_early_stopping_cnn(args):
    print("\n" + "="*70)
    print("  S3: Early stop (best val) vs end-epoch test — CIFAR CNN")
    print("="*70)

    device = torch.device(
        "cuda" if torch.cuda.is_available() else
        "mps" if torch.backends.mps.is_available() else "cpu"
    )
    print(f"Device: {device}", flush=True)
    n = args.n_train_nn
    out = os.path.join(args.output_dir, "supp3_early_stop_cnn")
    os.makedirs(out, exist_ok=True)
    if getattr(args, "quick", False):
        widths = [2, 6, 16]
        epochs = min(args.epochs_nn, 60)
    else:
        widths = [1, 2, 3, 4, 6, 8, 12, 16, 24, 32]
        epochs = args.epochs_nn

    train_base, test_set = get_cifar10(data_dir=args.data_dir, augment=False)
    if args.s3_label_noise > 0:
        train_base = corrupt_labels(
            get_cifar10(data_dir=args.data_dir, augment=False)[0],
            args.s3_label_noise, seed=args.seed,
        )

    val_frac = args.s3_val_frac
    test_loader = DataLoader(
        test_set, batch_size=256, shuffle=False, num_workers=0
    )
    rng = np.random.RandomState(args.seed)
    pick = rng.choice(len(train_base), size=n, replace=False)
    n_val = max(1, int(val_frac * n))
    perm = np.random.RandomState(args.seed + 1).permutation(n)
    val_set_idx = set(perm[:n_val].tolist())
    tr_set_idx = [i for i in range(n) if i not in val_set_idx]
    pool = Subset(train_base, pick.tolist())
    train_subset = Subset(pool, tr_set_idx)
    val_subset = Subset(pool, list(val_set_idx))
    val_loader = DataLoader(
        val_subset, batch_size=256, shuffle=False, num_workers=0
    )
    print(
        f"Train pool n={n}, n_val={n_val}, n_train={n - n_val}, label_noise={args.s3_label_noise}",
        flush=True,
    )

    results = []
    for w in widths:
        train_loader = DataLoader(
            train_subset, batch_size=256, shuffle=True, num_workers=0
        )
        print(f"\n--- width={w} (epochs={epochs}) ---", flush=True)
        r = _train_cnn_with_early_stop(
            train_loader,
            val_loader,
            test_loader,
            w,
            device,
            epochs,
            1e-3,
            args.seed + w * 1000,
            verbose=not getattr(args, "quiet", False),
        )
        r["p_over_n"] = r["p"] / n
        r["n_train"] = n
        r["n_val"] = n_val
        r["n_fit"] = n - n_val
        r["label_noise"] = args.s3_label_noise
        results.append(r)
        print(
            f"  p/n={r['p_over_n']:.2f} | test@best_val {r['test_acc_at_best_val']:.1f}% "
            f"(ep {r['best_epoch']}) | test@final {r['test_acc_end']:.1f}%",
            flush=True,
        )

    with open(os.path.join(out, "results.json"), "w") as f:
        json.dump(results, f, indent=2)

    pns = [r["p_over_n"] for r in results]
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(
        pns, [r["test_acc_at_best_val"] for r in results], "o-",
        label="Test acc @ best val", markersize=5,
    )
    ax.plot(
        pns, [r["test_acc_end"] for r in results], "s-",
        label="Test acc @ last epoch", markersize=5, alpha=0.85,
    )
    ax.set_xlabel("p / n (n = size of full subset incl. val share)")
    ax.set_ylabel("Test accuracy (%)")
    ax.set_title(
        f"S3: CIFAR-10 — early stop vs full train (Adam, {epochs} ep) "
        f"noise={args.s3_label_noise:.0%}",
    )
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(
        os.path.join(out, "dd_curves.png"), bbox_inches="tight", dpi=150,
    )
    plt.close()
    print(f"\nSaved → {out}", flush=True)
    return results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description="Supplemental DD: OOD, ordered n, early stopping",
    )
    parser.add_argument(
        "--experiments",
        type=str,
        default="S1,S2,S3",
        help="Comma: S1 (OOD/ID RFF), S2 (ordered sample RFF), S3 (early-stop CNN)",
    )
    parser.add_argument("--data-dir", type=str, default="./data")
    parser.add_argument("--output-dir", type=str, default="./results")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n-train", type=int, default=1000, help="RFF n (S1, S2 pool)")
    parser.add_argument(
        "--n-train-nn", type=int, default=4000, help="CIFAR subset size n (S3, incl. val)",
    )
    parser.add_argument(
        "--epochs-nn", type=int, default=200,
        help="S3: training epochs (default 200; use 500 to match main exp3)",
    )
    parser.add_argument(
        "--ood-std", type=float, default=0.2,
        help="S1: Gaussian pixel noise std on OOD test images",
    )
    parser.add_argument(
        "--s2-d-fixed", type=int, default=500, help="S2: RFF feature dimension D",
    )
    parser.add_argument(
        "--s3-val-frac", type=float, default=0.1,
        help="S3: held-out val fraction of n",
    )
    parser.add_argument(
        "--s3-label-noise", type=float, default=0.0,
        help="S3: label noise on CIFAR (0.0 = clean, try 0.2 to match other exps)",
    )
    parser.add_argument("--quick", action="store_true", help="Fewer p/n, widths, ep")
    parser.add_argument("--quiet", action="store_true", help="Less S3 per-epoch logging")
    args = parser.parse_args()

    exps = [e.strip().upper() for e in args.experiments.split(",")]
    print("Experiments:", exps, "→", args.output_dir, flush=True)

    if "S1" in exps:
        exp_S1_ood_rff(args)
    if "S2" in exps:
        exp_S2_ordered_sample_wise_rff(args)
    if "S3" in exps:
        exp_S3_early_stopping_cnn(args)

    print("\n" + "="*70, flush=True)
    print("SUPPLEMENTAL EXTRAS DONE", flush=True)
    print("="*70, flush=True)


if __name__ == "__main__":
    main()