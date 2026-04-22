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

from src.models import CNN, MLP, ResNet
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


def exp1_model_wise_rff(args):
    """Model-wise DD with random Fourier features."""
    print("\n" + "="*70)
    print("  EXP 1: MODEL-WISE DD (Random Features)")
    print("="*70)

    noise_rates = [0.0, 0.1, 0.2]
    n = args.n_train
    all_results = {}

    for nr in noise_rates:
        print(f"\n--- noise = {nr:.0%} ---")
        X_tr, Y_tr, y_tr, X_te, Y_te, y_te = load_mnist_numpy(
            args.data_dir, n, nr, args.seed)

        ratios = [0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 0.8, 0.9, 0.95, 0.98,
                  1.0, 1.02, 1.05, 1.1, 1.2, 1.5, 2.0, 3.0, 5.0, 8.0]
        results = []

        for ratio in ratios:
            D = max(1, int(ratio * n))
            Phi_tr = random_fourier_features(X_tr, D, sigma=5.0, seed=args.seed)
            Phi_te = random_fourier_features(X_te, D, sigma=5.0, seed=args.seed)
            w = min_norm_solution(Phi_tr, Y_tr)

            pred_tr = Phi_tr @ w
            pred_te = Phi_te @ w
            train_mse = np.mean((Y_tr - pred_tr)**2)
            test_mse = np.mean((Y_te - pred_te)**2)
            train_acc = np.mean(np.argmax(pred_tr, 1) == y_tr) * 100
            test_acc = np.mean(np.argmax(pred_te, 1) == y_te) * 100

            print(f"  D={D:5d} (p/n={ratio:.2f}): "
                  f"test_mse={test_mse:.4f}, test_acc={test_acc:.1f}%")
            results.append({
                "D": D, "p_over_n": ratio,
                "train_mse": float(train_mse), "test_mse": float(test_mse),
                "train_acc": float(train_acc), "test_acc": float(test_acc),
            })

        all_results[nr] = results

    out = os.path.join(args.output_dir, "exp1_model_wise_rff")
    os.makedirs(out, exist_ok=True)
    with open(os.path.join(out, "results.json"), "w") as f:
        json.dump({str(k): v for k, v in all_results.items()}, f, indent=2)

    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    colors = {"0.0": "tab:blue", "0.1": "tab:orange", "0.2": "tab:red"}
    for nr_str, results in sorted({str(k): v for k, v in all_results.items()}.items()):
        r = sorted(results, key=lambda x: x["p_over_n"])
        x = [d["p_over_n"] for d in r]
        axes[0].plot(x, [d["test_mse"] for d in r], "o-", color=colors.get(nr_str, "gray"),
                     label=f"noise={float(nr_str):.0%}", markersize=4)
        axes[1].plot(x, [100-d["test_acc"] for d in r], "o-", color=colors.get(nr_str, "gray"),
                     label=f"noise={float(nr_str):.0%}", markersize=4)

    for ax in axes:
        ax.axvline(x=1.0, color="gray", linestyle=":", alpha=0.7)
        ax.set_xlabel("p/n")
        ax.grid(True, alpha=0.3)
        ax.legend()
    axes[0].set_ylabel("Test MSE"); axes[0].set_yscale("log")
    axes[0].set_title("Model-Wise DD: Test MSE")
    axes[1].set_ylabel("Test Error (%)")
    axes[1].set_title("Model-Wise DD: Classification Error")
    plt.suptitle(f"Random Fourier Features on MNIST (n={n})", fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(out, "dd_curves.png"), bbox_inches="tight", dpi=150)
    plt.close()
    print(f"Saved to {out}")
    return all_results


def exp2_sample_wise_rff(args):
    """Sample-wise DD: fix D, sweep n."""
    print("\n" + "="*70)
    print("  EXP 2: SAMPLE-WISE DD (Random Features)")
    print("="*70)

    D_fixed = 500
    sample_sizes = [100, 200, 300, 400, 450, 480, 490, 500, 510, 520,
                    550, 600, 700, 1000, 1500, 2000, 4000]
    results = []

    for n in sample_sizes:
        X_tr, Y_tr, y_tr, X_te, Y_te, y_te = load_mnist_numpy(
            args.data_dir, n, noise_rate=0.1, seed=args.seed)

        Phi_tr = random_fourier_features(X_tr, D_fixed, sigma=5.0, seed=args.seed)
        Phi_te = random_fourier_features(X_te, D_fixed, sigma=5.0, seed=args.seed)
        w = min_norm_solution(Phi_tr, Y_tr)

        pred_te = Phi_te @ w
        test_mse = np.mean((Y_te - pred_te)**2)
        test_acc = np.mean(np.argmax(pred_te, 1) == y_te) * 100
        pred_tr = Phi_tr @ w
        train_mse = np.mean((Y_tr - pred_tr)**2)

        ratio = D_fixed / n
        print(f"  n={n:5d} (p/n={ratio:.2f}): test_mse={test_mse:.4f}, test_acc={test_acc:.1f}%")
        results.append({
            "n_samples": n, "D": D_fixed, "p_over_n": round(ratio, 4),
            "test_mse": float(test_mse), "train_mse": float(train_mse),
            "test_acc": float(test_acc),
        })

    out = os.path.join(args.output_dir, "exp2_sample_wise_rff")
    os.makedirs(out, exist_ok=True)
    with open(os.path.join(out, "results.json"), "w") as f:
        json.dump(results, f, indent=2)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    r = sorted(results, key=lambda x: x["n_samples"])
    ns = [d["n_samples"] for d in r]
    axes[0].plot(ns, [d["test_mse"] for d in r], "o-", color="red", markersize=5, label="Test")
    axes[0].plot(ns, [d["train_mse"] for d in r], "s--", color="blue", alpha=0.5, markersize=4, label="Train")
    axes[0].axvline(x=D_fixed, color="gray", linestyle=":", alpha=0.7, label=f"n=D={D_fixed}")
    axes[0].set_xlabel("Number of Training Samples (n)")
    axes[0].set_ylabel("MSE"); axes[0].set_yscale("log")
    axes[0].set_title("Sample-Wise DD: MSE"); axes[0].legend(); axes[0].grid(True, alpha=0.3)

    axes[1].plot(ns, [100-d["test_acc"] for d in r], "o-", color="red", markersize=5)
    axes[1].axvline(x=D_fixed, color="gray", linestyle=":", alpha=0.7, label=f"n=D={D_fixed}")
    axes[1].set_xlabel("Number of Training Samples (n)")
    axes[1].set_ylabel("Test Error (%)"); axes[1].set_title("Sample-Wise DD: Error")
    axes[1].legend(); axes[1].grid(True, alpha=0.3)

    plt.suptitle(f"Sample-Wise DD (D={D_fixed}, 10% noise)", fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(out, "dd_curves.png"), bbox_inches="tight", dpi=150)
    plt.close()
    print(f"Saved to {out}")
    return results


def exp3_nn_model_wise(args):
    """Model-wise DD with actual neural networks (CNN)."""
    print("\n" + "="*70)
    print("  EXP 3: MODEL-WISE DD (Neural Network)")
    print("="*70)

    device = torch.device("cuda" if torch.cuda.is_available() else
                          "mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Device: {device}")

    n = args.n_train_nn
    noise_rates = [0.0, 0.2]
    widths = [1, 2, 3, 4, 6, 8, 12, 16, 24, 32]
    all_results = {}

    for nr in noise_rates:
        print(f"\n--- noise = {nr:.0%} ---")
        train_full, test_set = get_cifar10(data_dir=args.data_dir, augment=False)
        if nr > 0:
            train_full = corrupt_labels(train_full, nr, seed=args.seed)
        train_set = make_subset(train_full, n, seed=args.seed)
        train_loader, test_loader = make_loaders(train_set, test_set, batch_size=256)

        results = []
        for width in widths:
            torch.manual_seed(args.seed)
            np.random.seed(args.seed)
            model = CNN(num_classes=10, num_filters=width, input_channels=3)
            p = model.count_parameters()
            ratio = p / n

            trainer = Trainer(model, device=device, lr=0.001, weight_decay=0.0,
                              optimizer_type="adam")
            t0 = time.time()
            history = trainer.train(train_loader, test_loader,
                                    epochs=args.epochs_nn,
                                    log_interval=max(1, args.epochs_nn // 3),
                                    verbose=True)
            elapsed = time.time() - t0

            r = {
                "width": width, "num_params": p, "p_over_n": round(ratio, 4),
                "train_acc": history["train_acc"][-1],
                "test_acc": history["test_acc"][-1],
                "train_loss": history["train_loss"][-1],
                "test_loss": history["test_loss"][-1],
            }
            results.append(r)
            print(f"  w={width:3d} p={p:>8,} p/n={ratio:.3f} "
                  f"train_err={100-r['train_acc']:.1f}% "
                  f"test_err={100-r['test_acc']:.1f}% ({elapsed:.0f}s)")

        all_results[nr] = results

    out = os.path.join(args.output_dir, "exp3_nn_model_wise")
    os.makedirs(out, exist_ok=True)
    with open(os.path.join(out, "results.json"), "w") as f:
        json.dump({str(k): v for k, v in all_results.items()}, f, indent=2)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for nr, results in all_results.items():
        r = sorted(results, key=lambda x: x["num_params"])
        params = [d["num_params"] for d in r]
        color = "tab:blue" if nr == 0.0 else "tab:red"
        label = f"noise={nr:.0%}"
        axes[0].plot(params, [100-d["test_acc"] for d in r], "o-", color=color,
                     label=f"Test ({label})", markersize=5)
        axes[0].plot(params, [100-d["train_acc"] for d in r], "s--", color=color,
                     alpha=0.4, label=f"Train ({label})", markersize=4)
        axes[1].plot(params, [d["test_loss"] for d in r], "o-", color=color,
                     label=f"Test ({label})", markersize=5)

    for ax in axes:
        ax.axvline(x=n, color="gray", linestyle=":", alpha=0.7, label=f"p=n={n}")
        ax.set_xscale("log"); ax.set_xlabel("Number of Parameters")
        ax.legend(); ax.grid(True, alpha=0.3)
    axes[0].set_ylabel("Error (%)"); axes[0].set_title("NN Model-Wise DD: Error")
    axes[1].set_ylabel("Test Loss"); axes[1].set_title("NN Model-Wise DD: Loss")

    plt.suptitle(f"CNN on CIFAR-10 Subset (n={n})", fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(out, "dd_curves.png"), bbox_inches="tight", dpi=150)
    plt.close()
    print(f"Saved to {out}")
    return all_results


def exp4_epoch_wise_nn(args):
    """Epoch-wise DD: train CNNs of different sizes for many epochs."""
    print("\n" + "="*70)
    print("  EXP 4: EPOCH-WISE DD (Neural Network)")
    print("="*70)

    device = torch.device("cuda" if torch.cuda.is_available() else
                          "mps" if torch.backends.mps.is_available() else "cpu")

    n = args.n_train_nn
    train_full, test_set = get_cifar10(data_dir=args.data_dir, augment=False)
    train_full = corrupt_labels(train_full, 0.2, seed=args.seed)
    train_set = make_subset(train_full, n, seed=args.seed)
    train_loader, test_loader = make_loaders(train_set, test_set, batch_size=256)

    widths = [2, 4, 8]
    all_histories = {}

    for width in widths:
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)
        model = CNN(num_classes=10, num_filters=width, input_channels=3)
        p = model.count_parameters()
        label = f"w={width} (p={p:,}, p/n={p/n:.2f})"
        print(f"\n{label}")

        trainer = Trainer(model, device=device, lr=0.001, weight_decay=0.0,
                          optimizer_type="adam")
        history = trainer.train(train_loader, test_loader,
                                epochs=args.epochs_epoch,
                                log_interval=max(1, args.epochs_epoch // 10),
                                verbose=True)
        all_histories[label] = history

    out = os.path.join(args.output_dir, "exp4_epoch_wise_nn")
    os.makedirs(out, exist_ok=True)
    with open(os.path.join(out, "results.json"), "w") as f:
        json.dump(all_histories, f, indent=2)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for label, hist in all_histories.items():
        epochs = hist["epoch"]
        axes[0].plot(epochs, [100-a for a in hist["test_acc"]], label=label)
        axes[1].plot(epochs, hist["test_loss"], label=label)

    for ax in axes:
        ax.set_xlabel("Epoch"); ax.legend(fontsize=9); ax.grid(True, alpha=0.3)
    axes[0].set_ylabel("Test Error (%)"); axes[0].set_title("Epoch-Wise DD: Error")
    axes[1].set_ylabel("Test Loss"); axes[1].set_title("Epoch-Wise DD: Loss")

    plt.suptitle(f"CNN on CIFAR-10 (n={n}, 20% noise)", fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(out, "dd_curves.png"), bbox_inches="tight", dpi=150)
    plt.close()
    print(f"Saved to {out}")
    return all_histories


def exp5_architecture_comparison(args):
    """Architecture comparison (MLP, CNN, ResNet) Double Descent."""
    print("\n" + "="*70)
    print("  EXP 5: ARCHITECTURE COMPARISON DD (MLP, CNN, ResNet)")
    print("="*70)

    device = torch.device("cuda" if torch.cuda.is_available() else
                          "mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Device: {device}")

    n = args.n_train_nn
    train_full, test_set = get_cifar10(data_dir=args.data_dir, augment=False)
    train_full = corrupt_labels(train_full, 0.1, seed=args.seed)
    train_set = make_subset(train_full, n, seed=args.seed)
    train_loader, test_loader = make_loaders(train_set, test_set, batch_size=256)

    # Define the sweep configurations
    # We want parameter counts roughly from ~1000 to ~250,000 to cross n=4000
    configs = {
        "MLP": [1, 2, 5, 10, 20, 50, 100, 200],  # approx 3000 to 600K params
        "CNN": [1, 2, 3, 4, 6, 8, 12, 16, 24, 32], # approx 700 to 110K params
        "ResNet": [0.125, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0] # approx 3K to 700K params
    }

    all_results = {}

    for arch, widths in configs.items():
        print(f"\n--- Architecture: {arch} ---")
        results = []
        for width in widths:
            torch.manual_seed(args.seed)
            np.random.seed(args.seed)
            
            if arch == "MLP":
                model = MLP(input_dim=3072, num_classes=10, hidden_width=width, num_hidden_layers=1)
            elif arch == "CNN":
                model = CNN(num_classes=10, num_filters=width, input_channels=3)
            elif arch == "ResNet":
                model = ResNet(num_classes=10, k=width)
                
            p = model.count_parameters()
            ratio = p / n

            trainer = Trainer(model, device=device, lr=0.001, weight_decay=0.0,
                              optimizer_type="adam")
            t0 = time.time()
            history = trainer.train(train_loader, test_loader,
                                    epochs=args.epochs_nn,
                                    log_interval=50,
                                    verbose=True)
            elapsed = time.time() - t0

            r = {
                "width": width, "num_params": p, "p_over_n": round(ratio, 4),
                "train_acc": history["train_acc"][-1],
                "test_acc": history["test_acc"][-1],
                "train_loss": history["train_loss"][-1],
                "test_loss": history["test_loss"][-1],
            }
            results.append(r)
            print(f"  w={width:<5} p={p:>8,} p/n={ratio:.3f} "
                  f"train_err={100-r['train_acc']:5.1f}% "
                  f"test_err={100-r['test_acc']:5.1f}% ({elapsed:.0f}s)")
            
        all_results[arch] = results

    out = os.path.join(args.output_dir, "exp5_architecture_comparison")
    os.makedirs(out, exist_ok=True)
    with open(os.path.join(out, "results.json"), "w") as f:
        json.dump(all_results, f, indent=2)

    # Plotting
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    colors = {"MLP": "tab:green", "CNN": "tab:blue", "ResNet": "tab:purple"}
    
    for arch, results in all_results.items():
        r = sorted(results, key=lambda x: x["num_params"])
        params = [d["num_params"] for d in r]
        
        axes[0].plot(params, [100-d["test_acc"] for d in r], "o-", color=colors[arch],
                     label=f"{arch} (Test)", markersize=5)
        axes[0].plot(params, [100-d["train_acc"] for d in r], "s--", color=colors[arch],
                     alpha=0.4, label=f"{arch} (Train)", markersize=4)
        
        axes[1].plot(params, [d["test_loss"] for d in r], "o-", color=colors[arch],
                     label=f"{arch} (Test)", markersize=5)

    for ax in axes:
        ax.axvline(x=n, color="gray", linestyle=":", alpha=0.7, label=f"p=n={n}")
        ax.set_xscale("log")
        ax.set_xlabel("Number of Parameters")
        ax.legend()
        ax.grid(True, alpha=0.3)
        
    axes[0].set_ylabel("Error (%)")
    axes[0].set_title("Architecture Comparison: Classification Error")
    axes[1].set_ylabel("Test Loss")
    axes[1].set_title("Architecture Comparison: Test Loss")

    plt.suptitle(f"MLP vs CNN vs ResNet on CIFAR-10 (n={n}, 10% noise)", fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(out, "dd_curves.png"), bbox_inches="tight", dpi=150)
    plt.close()
    
    print(f"Saved to {out}")
    return all_results


def exp6_rff_ridge(args):
    """Effect of ridge regularization (lambda) on the DD peak."""
    print("\n" + "="*70)
    print("  EXP 6: RIDGE REGULARIZATION vs DD PEAK (RFF)")
    print("="*70)

    n = args.n_train
    noise_rate = 0.1
    X_tr, Y_tr, y_tr, X_te, Y_te, y_te = load_mnist_numpy(
        args.data_dir, n, noise_rate, args.seed)

    lambdas = [0, 1e-8, 1e-6, 1e-4, 1e-2, 0.1, 1.0]
    ratios = [0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 0.8, 0.9, 0.95, 0.98,
              1.0, 1.02, 1.05, 1.1, 1.2, 1.5, 2.0, 3.0, 5.0, 8.0]

    all_results = {}

    for lam in lambdas:
        print(f"\n--- lambda = {lam} ---")
        results = []
        for ratio in ratios:
            D = max(1, int(ratio * n))
            Phi_tr = random_fourier_features(X_tr, D, sigma=5.0, seed=args.seed)
            Phi_te = random_fourier_features(X_te, D, sigma=5.0, seed=args.seed)
            w = min_norm_solution(Phi_tr, Y_tr, lam=lam)

            pred_tr = Phi_tr @ w
            pred_te = Phi_te @ w
            train_mse = np.mean((Y_tr - pred_tr)**2)
            test_mse = np.mean((Y_te - pred_te)**2)
            train_acc = np.mean(np.argmax(pred_tr, 1) == y_tr) * 100
            test_acc = np.mean(np.argmax(pred_te, 1) == y_te) * 100

            print(f"  D={D:5d} (p/n={ratio:.2f}): "
                  f"test_mse={test_mse:.4f}, test_acc={test_acc:.1f}%")
            results.append({
                "D": D, "p_over_n": ratio, "lambda": lam,
                "train_mse": float(train_mse), "test_mse": float(test_mse),
                "train_acc": float(train_acc), "test_acc": float(test_acc),
            })

        all_results[str(lam)] = results

    out = os.path.join(args.output_dir, "exp6_rff_ridge")
    os.makedirs(out, exist_ok=True)
    with open(os.path.join(out, "results.json"), "w") as f:
        json.dump(all_results, f, indent=2)

    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    cmap = plt.cm.viridis
    n_colors = len(lambdas)
    colors = [cmap(i / max(1, n_colors - 1)) for i in range(n_colors)]

    for idx, lam in enumerate(lambdas):
        lam_str = str(lam)
        r = sorted(all_results[lam_str], key=lambda x: x["p_over_n"])
        x = [d["p_over_n"] for d in r]
        label = f"\u03bb={lam}" if lam > 0 else "\u03bb=0 (ridgeless)"
        axes[0].plot(x, [d["test_mse"] for d in r], "o-", color=colors[idx],
                     label=label, markersize=4)
        axes[1].plot(x, [100 - d["test_acc"] for d in r], "o-", color=colors[idx],
                     label=label, markersize=4)

    for ax in axes:
        ax.axvline(x=1.0, color="gray", linestyle=":", alpha=0.7)
        ax.set_xlabel("p/n")
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=9)
    axes[0].set_ylabel("Test MSE"); axes[0].set_yscale("log")
    axes[0].set_title("Ridge Regularization: Test MSE")
    axes[1].set_ylabel("Test Error (%)")
    axes[1].set_title("Ridge Regularization: Classification Error")
    plt.suptitle(f"Effect of Ridge \u03bb on DD Peak (RFF, MNIST, n={n}, 10% noise)",
                 fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(out, "dd_curves.png"), bbox_inches="tight", dpi=150)
    plt.close()
    print(f"Saved to {out}")
    return all_results


def exp7_spectral_analysis(args):
    """Spectral analysis: condition number and solution norm vs p/n."""
    print("\n" + "="*70)
    print("  EXP 7: SPECTRAL ANALYSIS (WHY DD HAPPENS)")
    print("="*70)

    n = args.n_train
    noise_rate = 0.1
    X_tr, Y_tr, y_tr, X_te, Y_te, y_te = load_mnist_numpy(
        args.data_dir, n, noise_rate, args.seed)

    ratios = [0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 0.8, 0.9, 0.95, 0.98,
              1.0, 1.02, 1.05, 1.1, 1.2, 1.5, 2.0, 3.0, 5.0, 8.0]
    results = []

    for ratio in ratios:
        D = max(1, int(ratio * n))
        Phi_tr = random_fourier_features(X_tr, D, sigma=5.0, seed=args.seed)
        Phi_te = random_fourier_features(X_te, D, sigma=5.0, seed=args.seed)

        # Compute condition number
        if D >= n:
            K = Phi_tr @ Phi_tr.T  # n x n
        else:
            K = Phi_tr.T @ Phi_tr  # D x D
        eigvals = np.linalg.eigvalsh(K)
        eigvals_pos = eigvals[eigvals > 1e-15]
        if len(eigvals_pos) > 0:
            cond_num = float(eigvals_pos[-1] / eigvals_pos[0])
        else:
            cond_num = float('inf')

        # Compute min-norm solution and its norm
        w = min_norm_solution(Phi_tr, Y_tr, lam=1e-10)
        w_norm = float(np.linalg.norm(w))

        # Test MSE
        pred_te = Phi_te @ w
        test_mse = float(np.mean((Y_te - pred_te)**2))
        test_acc = float(np.mean(np.argmax(pred_te, 1) == y_te) * 100)

        # Max eigenvalue / spectral norm
        max_eig = float(eigvals[-1]) if len(eigvals) > 0 else 0
        min_eig = float(eigvals_pos[0]) if len(eigvals_pos) > 0 else 0

        print(f"  D={D:5d} (p/n={ratio:.2f}): cond={cond_num:.1e}, "
              f"||w||={w_norm:.2f}, test_mse={test_mse:.4f}")
        results.append({
            "D": D, "p_over_n": ratio,
            "condition_number": cond_num, "w_norm": w_norm,
            "max_eigenvalue": max_eig, "min_eigenvalue": min_eig,
            "test_mse": test_mse, "test_acc": test_acc,
        })

    out = os.path.join(args.output_dir, "exp7_spectral_analysis")
    os.makedirs(out, exist_ok=True)
    with open(os.path.join(out, "results.json"), "w") as f:
        json.dump(results, f, indent=2)

    # Plot: 3 subplots
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    r = sorted(results, key=lambda x: x["p_over_n"])
    x = [d["p_over_n"] for d in r]

    # Condition number
    axes[0].plot(x, [d["condition_number"] for d in r], "o-",
                 color="tab:red", markersize=5)
    axes[0].set_yscale("log")
    axes[0].set_ylabel("Condition Number")
    axes[0].set_title("Kernel Matrix Condition Number")

    # Solution norm
    axes[1].plot(x, [d["w_norm"] for d in r], "s-",
                 color="tab:purple", markersize=5)
    axes[1].set_yscale("log")
    axes[1].set_ylabel("||w||₂")
    axes[1].set_title("Min-Norm Solution Norm")

    # Test MSE for reference
    axes[2].plot(x, [d["test_mse"] for d in r], "D-",
                 color="tab:blue", markersize=5)
    axes[2].set_yscale("log")
    axes[2].set_ylabel("Test MSE")
    axes[2].set_title("Test MSE (for comparison)")

    for ax in axes:
        ax.axvline(x=1.0, color="gray", linestyle=":", alpha=0.7, label="p/n=1")
        ax.set_xlabel("p/n")
        ax.grid(True, alpha=0.3)
        ax.legend()

    plt.suptitle(f"Why DD Happens: Spectral Analysis (RFF, MNIST, n={n}, 10% noise)",
                 fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(out, "dd_curves.png"), bbox_inches="tight", dpi=150)
    plt.close()
    print(f"Saved to {out}")
    return results


def exp8_optimal_lambda(args):
    """Find optimal ridge lambda for each p/n ratio."""
    print("\n" + "="*70)
    print("  EXP 8: OPTIMAL RIDGE LAMBDA (RFF)")
    print("="*70)

    n = args.n_train
    noise_rate = 0.1
    X_tr, Y_tr, y_tr, X_te, Y_te, y_te = load_mnist_numpy(
        args.data_dir, n, noise_rate, args.seed)

    lambdas = np.logspace(-10, 2, 40).tolist()
    ratios = [0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 0.8, 0.9, 0.95, 0.98,
              1.0, 1.02, 1.05, 1.1, 1.2, 1.5, 2.0, 3.0, 5.0, 8.0]

    all_results = {}

    for ratio in ratios:
        D = max(1, int(ratio * n))
        Phi_tr = random_fourier_features(X_tr, D, sigma=5.0, seed=args.seed)
        Phi_te = random_fourier_features(X_te, D, sigma=5.0, seed=args.seed)

        best_lam = 0
        best_mse = float('inf')
        sweep = []

        for lam in lambdas:
            w = min_norm_solution(Phi_tr, Y_tr, lam=lam)
            pred_te = Phi_te @ w
            test_mse = float(np.mean((Y_te - pred_te)**2))
            sweep.append({"lambda": lam, "test_mse": test_mse})
            if test_mse < best_mse:
                best_mse = test_mse
                best_lam = lam

        print(f"  D={D:5d} (p/n={ratio:.2f}): best_λ={best_lam:.2e}, "
              f"best_mse={best_mse:.4f}")
        all_results[str(ratio)] = {
            "D": D, "p_over_n": ratio,
            "best_lambda": best_lam, "best_test_mse": best_mse,
            "sweep": sweep,
        }

    out = os.path.join(args.output_dir, "exp8_optimal_lambda")
    os.makedirs(out, exist_ok=True)
    with open(os.path.join(out, "results.json"), "w") as f:
        json.dump(all_results, f, indent=2)

    # Plot
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    r = sorted(all_results.values(), key=lambda x: x["p_over_n"])
    x = [d["p_over_n"] for d in r]

    # Optimal lambda vs p/n
    axes[0].plot(x, [d["best_lambda"] for d in r], "o-",
                 color="tab:orange", markersize=6)
    axes[0].set_yscale("log")
    axes[0].set_ylabel("Optimal \u03bb*")
    axes[0].set_title("Optimal Regularization Strength")

    # Best achievable test MSE vs p/n
    axes[1].plot(x, [d["best_test_mse"] for d in r], "s-",
                 color="tab:green", markersize=6)
    axes[1].set_ylabel("Best Test MSE")
    axes[1].set_title("Best Achievable Test MSE (with optimal \u03bb)")

    # Heatmap-style: for a few key p/n ratios, show MSE vs lambda
    key_ratios = ["0.5", "0.95", "1.0", "1.05", "2.0"]
    colors_heat = plt.cm.coolwarm(np.linspace(0, 1, len(key_ratios)))
    for idx, kr in enumerate(key_ratios):
        if kr in all_results:
            sweep = all_results[kr]["sweep"]
            lams = [s["lambda"] for s in sweep]
            mses = [s["test_mse"] for s in sweep]
            axes[2].plot(lams, mses, "o-", color=colors_heat[idx],
                         label=f"p/n={kr}", markersize=3)
    axes[2].set_xscale("log")
    axes[2].set_yscale("log")
    axes[2].set_xlabel("\u03bb")
    axes[2].set_ylabel("Test MSE")
    axes[2].set_title("Test MSE vs \u03bb (selected p/n)")
    axes[2].legend(fontsize=9)

    for ax in axes[:2]:
        ax.axvline(x=1.0, color="gray", linestyle=":", alpha=0.7, label="p/n=1")
        ax.set_xlabel("p/n")
        ax.legend()
    for ax in axes:
        ax.grid(True, alpha=0.3)

    plt.suptitle(f"Optimal Ridge \u03bb* Analysis (RFF, MNIST, n={n}, 10% noise)",
                 fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(out, "dd_curves.png"), bbox_inches="tight", dpi=150)
    plt.close()
    print(f"Saved to {out}")
    return all_results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiments", type=str, default="1,2,3,4,5,6,7,8",
                        help="Which experiments to run (1=model_rff, 2=sample_rff, "
                             "3=nn_model, 4=nn_epoch, 5=arch_comp, 6=rff_ridge, "
                             "7=spectral, 8=optimal_lambda)")
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
    parser.add_argument("--seed", type=int, default=42)
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
    if 5 in exps:
        exp5_architecture_comparison(args)
    if 6 in exps:
        exp6_rff_ridge(args)
    if 7 in exps:
        exp7_spectral_analysis(args)
    if 8 in exps:
        exp8_optimal_lambda(args)

    print("\n" + "="*70)
    print("ALL EXPERIMENTS COMPLETE")
    print(f"Results in: {args.output_dir}")
    print("="*70)


if __name__ == "__main__":
    main()

