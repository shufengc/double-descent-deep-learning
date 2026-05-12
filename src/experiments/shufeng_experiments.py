"""
Shufeng Chen (sc5739) -- EECS 6699 Final Project:
  exp_noise_multiseed : Validate Zhengda's exp6 -- noise 0/10/20/40% with 5 seeds
  expB_bias_variance  : Bias-variance decomposition of RFF test MSE (D'Ascoli et al. 2020)
  expC_epoch_sgd      : Epoch-wise DD with SGD + ResNet (Nakkiran et al. 2021 setup)
  expA_emc            : Effective Model Complexity (Nakkiran et al. 2021, Def 4)

Run locally (RFF, fast):
    python3 -m src.experiments.shufeng_experiments --experiments noise_multiseed,B
Run on GPU (NN, slow):
    python3 -m src.experiments.shufeng_experiments --experiments C,A
"""

import sys, os, json, argparse, time
import numpy as np
import torch
from torch.utils.data import DataLoader as TDataLoader

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.models import ResNet
from src.data import get_cifar10, corrupt_labels, make_subset

plt.rcParams.update({
    "figure.figsize": (12, 5), "font.size": 13, "axes.titlesize": 14,
    "axes.labelsize": 13, "legend.fontsize": 10, "lines.linewidth": 2,
    "figure.dpi": 150,
})

# ---------------------------------------------------------------------------
# Shared RFF utilities
# ---------------------------------------------------------------------------

def _rff(X, D, sigma=5.0, seed=42):
    rng = np.random.RandomState(seed)
    W = rng.randn(X.shape[1], D).astype(np.float64) / sigma
    b = rng.uniform(0, 2 * np.pi, D).astype(np.float64)
    proj = X.astype(np.float64) @ W + b
    np.nan_to_num(proj, copy=False, nan=0.0, posinf=1e10, neginf=-1e10)
    return np.cos(proj) * np.sqrt(2.0 / D)


def _min_norm(Phi, y, lam=1e-10):
    n, p = Phi.shape
    if p >= n:
        alpha = np.linalg.solve(Phi @ Phi.T + lam * np.eye(n), y)
        return Phi.T @ alpha
    return np.linalg.solve(Phi.T @ Phi + lam * np.eye(p), Phi.T @ y)


def _load_mnist(data_dir, n_train, noise_rate, seed):
    import torchvision, torchvision.transforms as T
    ds_tr = torchvision.datasets.MNIST(data_dir, train=True,  download=True, transform=T.ToTensor())
    ds_te = torchvision.datasets.MNIST(data_dir, train=False, download=True, transform=T.ToTensor())

    X_tr = ds_tr.data.numpy().reshape(-1, 784).astype(np.float64) / 255.0
    y_tr = ds_tr.targets.numpy().copy()
    X_te = ds_te.data.numpy().reshape(-1, 784).astype(np.float64) / 255.0
    y_te = ds_te.targets.numpy().copy()

    rng = np.random.RandomState(seed)
    if n_train and n_train < len(X_tr):
        idx = rng.choice(len(X_tr), n_train, replace=False)
        X_tr, y_tr = X_tr[idx], y_tr[idx]

    if noise_rate > 0:
        corrupt = rng.choice(len(y_tr), int(noise_rate * len(y_tr)), replace=False)
        for i in corrupt:
            y_tr[i] = rng.choice([c for c in range(10) if c != y_tr[i]])

    Y_tr = np.eye(10)[y_tr]
    Y_te = np.eye(10)[y_te]
    return X_tr, Y_tr, y_tr, X_te, Y_te, y_te


_RATIOS = [0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 0.8, 0.9, 0.95, 0.98,
           1.0, 1.02, 1.05, 1.1, 1.2, 1.5, 2.0, 3.0, 5.0, 8.0]


# ---------------------------------------------------------------------------
# Fast GPU training -- avoids DataLoader overhead for GPU-resident data
# ---------------------------------------------------------------------------

def _precompute_gpu(data_dir, noise_rate, n_train, seed, device):
    """
    Load CIFAR-10, apply transforms once, put ALL tensors on GPU.
    Returns (X_tr, Y_tr, X_te, Y_te) all on `device`.
    """
    train_full, test_set = get_cifar10(data_dir=data_dir, augment=False)
    train_full = corrupt_labels(train_full, noise_rate, seed=seed)
    train_subset = make_subset(train_full, n_train, seed=seed)

    def _ds_to_gpu(dataset):
        loader = TDataLoader(dataset, batch_size=512, shuffle=False, num_workers=0)
        xs, ys = [], []
        for x, y in loader:
            xs.append(x); ys.append(y)
        return torch.cat(xs).to(device), torch.cat(ys).to(device)

    print("  Loading train to GPU...", flush=True)
    X_tr, Y_tr = _ds_to_gpu(train_subset)
    print("  Loading test to GPU...",  flush=True)
    X_te, Y_te = _ds_to_gpu(test_set)
    print(f"  Ready: train={tuple(X_tr.shape)}, test={tuple(X_te.shape)}", flush=True)
    return X_tr, Y_tr, X_te, Y_te


def _train_fast(model, device, X_tr, Y_tr, X_te, Y_te,
                epochs, optimizer_type, lr, batch_size=256,
                momentum=0.9, weight_decay=0.0,
                log_interval=50, eval_interval=None, verbose=True):
    """
    Fully GPU-resident training loop (no DataLoader).
    Assumes all input tensors are already on `device`.
    Evaluation only at eval_interval steps to save time.
    """
    if eval_interval is None:
        eval_interval = log_interval

    model = model.to(device)
    criterion = torch.nn.CrossEntropyLoss()
    n = X_tr.shape[0]

    if optimizer_type == "sgd":
        opt = torch.optim.SGD(model.parameters(), lr=lr,
                              momentum=momentum, weight_decay=weight_decay)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    elif optimizer_type == "adam":
        opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
        scheduler = None
    else:
        raise ValueError(optimizer_type)

    history = {"epoch": [], "train_loss": [], "train_acc": [],
               "test_loss": [], "test_acc": [], "lr": []}

    for ep in range(1, epochs + 1):
        model.train()
        perm = torch.randperm(n, device=device)
        tl, tc, tt = 0.0, 0, 0
        for i in range(0, n, batch_size):
            idx = perm[i:i+batch_size]
            x, y = X_tr[idx], Y_tr[idx]
            opt.zero_grad()
            out = model(x)
            loss = criterion(out, y)
            loss.backward()
            opt.step()
            tl += loss.item() * x.size(0)
            tc += out.argmax(1).eq(y).sum().item()
            tt += x.size(0)
        if scheduler: scheduler.step()

        if ep % eval_interval == 0 or ep == epochs:
            model.eval()
            with torch.no_grad():
                el, ec, et = 0.0, 0, 0
                nte = X_te.shape[0]
                for j in range(0, nte, batch_size):
                    x, y = X_te[j:j+batch_size], Y_te[j:j+batch_size]
                    out = model(x)
                    loss = criterion(out, y)
                    el += loss.item() * x.size(0)
                    ec += out.argmax(1).eq(y).sum().item()
                    et += x.size(0)
            test_loss_ep = el / et
            test_acc_ep  = 100.0 * ec / et
        else:
            test_loss_ep = float("nan")
            test_acc_ep  = float("nan")

        history["epoch"].append(ep)
        history["train_loss"].append(tl / tt)
        history["train_acc"].append(100.0 * tc / tt)
        history["test_loss"].append(test_loss_ep)
        history["test_acc"].append(test_acc_ep)
        history["lr"].append(opt.param_groups[0]["lr"])

        if verbose and ep % log_interval == 0:
            print(f"  ep {ep:5d}/{epochs} | "
                  f"tr {100-100*tc/tt:.1f}% | te {100-test_acc_ep:.1f}% | "
                  f"lr {opt.param_groups[0]['lr']:.2e}", flush=True)

    return history


# ---------------------------------------------------------------------------
# exp_noise_multiseed: validate Zhengda's 40% noise anomaly with 5 seeds
# ---------------------------------------------------------------------------

def exp_noise_multiseed(args):
    print("\n" + "="*70)
    print("  NOISE MULTI-SEED VALIDATION (fix for Zhengda's exp6)")
    print("="*70)

    SEEDS = [42, 123, 456, 789, 1024]
    NOISE_RATES = [0.0, 0.1, 0.2, 0.4]
    n = args.n_train

    per_seed = {}
    for seed in SEEDS:
        per_seed[seed] = {}
        print(f"\n--- seed={seed} ---")
        for nr in NOISE_RATES:
            X_tr, Y_tr, y_tr, X_te, Y_te, y_te = _load_mnist(args.data_dir, n, nr, seed)
            rows = []
            for ratio in _RATIOS:
                D = max(1, int(ratio * n))
                Phi_tr = _rff(X_tr, D, sigma=5.0, seed=seed)
                Phi_te = _rff(X_te, D, sigma=5.0, seed=seed)
                w = _min_norm(Phi_tr, Y_tr)
                pred_tr = Phi_tr @ w
                pred_te = Phi_te @ w
                rows.append({
                    "p_over_n": ratio,
                    "train_mse": float(np.mean((Y_tr - pred_tr)**2)),
                    "test_mse":  float(np.mean((Y_te - pred_te)**2)),
                    "test_acc":  float(np.mean(np.argmax(pred_te, 1) == y_te) * 100),
                })
            per_seed[seed][nr] = rows
            peak_mse = max(r["test_mse"] for r in rows)
            print(f"  noise={nr:.0%}: peak_mse={peak_mse:.4f}")

    # Aggregate: mean and std across seeds
    agg = {}
    for nr in NOISE_RATES:
        agg_rows = []
        for i, ratio in enumerate(_RATIOS):
            vals = [per_seed[s][nr][i]["test_mse"] for s in SEEDS]
            accs = [per_seed[s][nr][i]["test_acc"] for s in SEEDS]
            agg_rows.append({
                "p_over_n": ratio,
                "test_mse_mean": float(np.mean(vals)),
                "test_mse_std":  float(np.std(vals)),
                "test_acc_mean": float(np.mean(accs)),
                "test_acc_std":  float(np.std(accs)),
            })
        agg[nr] = agg_rows

    print("\n=== AGGREGATED PEAK MSE (mean ± std across 5 seeds) ===")
    for nr in NOISE_RATES:
        rows = agg[nr]
        peak_row = max(rows, key=lambda r: r["test_mse_mean"])
        print(f"  noise={nr:.0%}: peak={peak_row['test_mse_mean']:.2f} ± {peak_row['test_mse_std']:.2f}")

    out = os.path.join(args.output_dir, "exp_noise_multiseed")
    os.makedirs(out, exist_ok=True)
    with open(os.path.join(out, "results.json"), "w") as f:
        json.dump({
            "seeds": SEEDS,
            "per_seed": {str(s): {str(nr): v for nr, v in d.items()}
                         for s, d in per_seed.items()},
            "aggregated": {str(nr): v for nr, v in agg.items()},
        }, f, indent=2)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    colors = {0.0: "tab:blue", 0.1: "tab:orange", 0.2: "tab:red", 0.4: "tab:purple"}
    for nr in NOISE_RATES:
        rows = agg[nr]
        x = [r["p_over_n"] for r in rows]
        ymean = [r["test_mse_mean"] for r in rows]
        ystd  = [r["test_mse_std"]  for r in rows]
        c = colors[nr]
        axes[0].plot(x, ymean, "o-", color=c, label=f"noise={nr:.0%}", markersize=4)
        axes[0].fill_between(x,
            [m - s for m, s in zip(ymean, ystd)],
            [m + s for m, s in zip(ymean, ystd)],
            color=c, alpha=0.15)
        axes[1].plot(x, [100 - r["test_acc_mean"] for r in rows],
                     "o-", color=c, label=f"noise={nr:.0%}", markersize=4)

    for ax in axes:
        ax.axvline(x=1.0, color="gray", linestyle=":", alpha=0.7, label="p/n=1")
        ax.set_xlabel("p/n"); ax.grid(True, alpha=0.3); ax.legend()
    axes[0].set_yscale("log"); axes[0].set_ylabel("Test MSE (log)")
    axes[0].set_title("Noise Comparison: Test MSE (5-seed avg ± std)")
    axes[1].set_ylabel("Test Error (%)"); axes[1].set_title("Noise Comparison: Error")
    plt.suptitle(f"RFF on MNIST (n={n}, 5 seeds: {SEEDS})", fontsize=13, y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(out, "dd_curves.png"), bbox_inches="tight", dpi=150)
    plt.close()
    print(f"Saved → {out}")
    return agg


# ---------------------------------------------------------------------------
# expB_bias_variance: decompose test MSE into Bias² + Variance
# ---------------------------------------------------------------------------

def expB_bias_variance(args):
    print("\n" + "="*70)
    print("  EXP B: BIAS-VARIANCE DECOMPOSITION (RFF)")
    print("="*70)
    print("Source: D'Ascoli et al., 'Double Trouble in Double Descent', ICML 2020")

    N_DRAWS = 50
    NOISE_RATES = [0.0, 0.2]
    n = args.n_train
    results = {}

    for nr in NOISE_RATES:
        print(f"\n--- noise = {nr:.0%} ---")
        X_tr, Y_tr, y_tr, X_te, Y_te, y_te = _load_mnist(args.data_dir, n, nr, seed=42)
        rows = []

        for ratio in _RATIOS:
            D = max(1, int(ratio * n))
            preds = []
            for draw in range(N_DRAWS):
                Phi_tr = _rff(X_tr, D, sigma=5.0, seed=1000 + draw)
                Phi_te = _rff(X_te, D, sigma=5.0, seed=1000 + draw)
                w = _min_norm(Phi_tr, Y_tr)
                preds.append(Phi_te @ w)
            preds = np.stack(preds, axis=0)     # (N_DRAWS, n_test, 10)

            mean_pred = preds.mean(axis=0)
            bias2     = float(np.mean(np.sum((mean_pred - Y_te)**2, axis=1)))
            variance  = float(np.mean(np.sum((preds - mean_pred[None])**2, axis=2)))
            total_mse = float(np.mean((preds - Y_te[None])**2))

            print(f"  D={D:5d} (p/n={ratio:.2f}): "
                  f"bias²={bias2:.4f}  var={variance:.4f}  total={total_mse:.4f}")
            rows.append({
                "p_over_n": ratio, "D": D,
                "bias2": bias2, "variance": variance,
                "total_mse": total_mse, "bias_plus_var": bias2 + variance,
            })
        results[nr] = rows

    out = os.path.join(args.output_dir, "expB_bias_variance")
    os.makedirs(out, exist_ok=True)
    with open(os.path.join(out, "results.json"), "w") as f:
        json.dump({str(k): v for k, v in results.items()}, f, indent=2)

    fig, axes = plt.subplots(1, len(NOISE_RATES), figsize=(7 * len(NOISE_RATES), 5))
    if len(NOISE_RATES) == 1:
        axes = [axes]

    for ax, nr in zip(axes, NOISE_RATES):
        rows = results[nr]
        x     = [r["p_over_n"]  for r in rows]
        bias2 = [r["bias2"]     for r in rows]
        var   = [r["variance"]  for r in rows]
        total = [r["total_mse"] for r in rows]
        ax.plot(x, total,  "k-",  label="Total MSE",  linewidth=2.5)
        ax.plot(x, bias2,  "b--", label="Bias²",      linewidth=2)
        ax.plot(x, var,    "r:",  label="Variance",   linewidth=2)
        ax.axvline(x=1.0, color="gray", linestyle=":", alpha=0.7, label="p/n=1")
        ax.set_yscale("log")
        ax.set_xlabel("p/n"); ax.set_ylabel("Error (log scale)")
        ax.set_title(f"Bias-Variance Decomposition (noise={nr:.0%})")
        ax.legend(); ax.grid(True, alpha=0.3)

    plt.suptitle(f"RFF on MNIST (n={n}, {N_DRAWS} random feature draws)", fontsize=13, y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(out, "bias_variance.png"), bbox_inches="tight", dpi=150)
    plt.close()
    print(f"\nSaved → {out}")
    return results


# ---------------------------------------------------------------------------
# expC_epoch_sgd: epoch-wise DD with SGD + ResNet (Nakkiran et al. setup)
# ---------------------------------------------------------------------------

def expC_epoch_sgd(args):
    print("\n" + "="*70)
    print("  EXP C: EPOCH-WISE DD  (SGD + ResNet)")
    print("="*70)
    print("Source: Nakkiran et al. 2021, Figure 8. arXiv:1912.02292")

    device = torch.device("cuda" if torch.cuda.is_available() else
                          "mps"  if torch.backends.mps.is_available() else "cpu")
    print(f"Device: {device}", flush=True)

    N       = args.n_train_nn       # 4000
    NOISE   = 0.20
    K_VALS  = [1, 2, 4]
    EPOCHS  = args.epochs_C         # 4000
    LOG_IV  = max(1, EPOCHS // 40)  # log every 100 epochs

    # Precompute tensors on GPU once, share across all model/optimizer runs
    X_tr, Y_tr, X_te, Y_te = _precompute_gpu(args.data_dir, NOISE, N, seed=42, device=device)

    all_results = {}

    for opt_type, lr in [("sgd", 0.1), ("adam", 0.001)]:
        all_results[opt_type] = {}
        for k in K_VALS:
            torch.manual_seed(42); np.random.seed(42)
            model = ResNet(num_classes=10, k=k)
            p = model.count_parameters()
            label = f"k={k} p={p:,} p/n={p/N:.2f}"
            print(f"\n{opt_type.upper()} | {label}", flush=True)

            t0 = time.time()
            history = _train_fast(
                model, device, X_tr, Y_tr, X_te, Y_te,
                epochs=EPOCHS, optimizer_type=opt_type, lr=lr,
                batch_size=256, momentum=0.9, weight_decay=0.0,
                log_interval=LOG_IV, eval_interval=LOG_IV, verbose=True,
            )
            elapsed = time.time() - t0
            print(f"  Done in {elapsed/60:.1f} min | "
                  f"final test_err={100 - [v for v in history['test_acc'] if not np.isnan(v)][-1]:.1f}%",
                  flush=True)
            all_results[opt_type][k] = {
                "label": label, "num_params": p, "p_over_n": round(p/N, 4),
                "history": history,
            }

    out = os.path.join(args.output_dir, "expC_epoch_sgd_resnet")
    os.makedirs(out, exist_ok=True)
    with open(os.path.join(out, "results.json"), "w") as f:
        json.dump(all_results, f, indent=2)

    # Plot
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    colors = {1: "tab:blue", 2: "tab:orange", 4: "tab:green"}

    for col, opt_type in enumerate(["sgd", "adam"]):
        ax_err  = axes[0][col]
        ax_loss = axes[1][col]
        for k in K_VALS:
            res  = all_results[opt_type][k]
            hist = res["history"]
            eps  = hist["epoch"]
            te   = [100 - a for a in hist["test_acc"]]
            c    = colors[k]
            lbl  = f"k={k} (p={res['num_params']:,})"
            # Only plot non-NaN test points
            ep_v = [e for e, v in zip(eps, te) if not np.isnan(v)]
            te_v = [v for v in te if not np.isnan(v)]
            tl_v = [v for v in hist["test_loss"] if not np.isnan(v)]
            ax_err.plot(ep_v,  te_v, color=c, label=lbl)
            ax_loss.plot(ep_v, tl_v, color=c, label=lbl)
        for ax in [ax_err, ax_loss]:
            ax.set_xlabel("Epoch"); ax.legend(fontsize=9); ax.grid(True, alpha=0.3)
        ax_err.set_ylabel("Test Error (%)")
        ax_err.set_title(f"Epoch-wise DD: {opt_type.upper()} + ResNet")
        ax_loss.set_ylabel("Test Loss")
        ax_loss.set_title(f"Test Loss: {opt_type.upper()} + ResNet")

    plt.suptitle(f"ResNet on CIFAR-10 (n={N}, 20% noise, {EPOCHS} epochs)", fontsize=13, y=1.01)
    plt.tight_layout()
    plt.savefig(os.path.join(out, "epoch_wise_dd.png"), bbox_inches="tight", dpi=150)
    plt.close()
    print(f"\nSaved → {out}", flush=True)
    return all_results


# ---------------------------------------------------------------------------
# expA_emc: Effective Model Complexity (Nakkiran et al. 2021, Definition 4)
# ---------------------------------------------------------------------------

def expA_emc(args):
    """
    EMC(T, model) = max n s.t. training for T epochs achieves <EPS train error.
    Binary-search n for each (model config, T).
    """
    print("\n" + "="*70)
    print("  EXP A: EFFECTIVE MODEL COMPLEXITY (EMC)")
    print("="*70)
    print("Source: Nakkiran et al. 2021, Section 3.1, Definition 4. arXiv:1912.02292")

    device = torch.device("cuda" if torch.cuda.is_available() else
                          "mps"  if torch.backends.mps.is_available() else "cpu")
    print(f"Device: {device}", flush=True)

    EPS      = 0.05       # train error < 5%
    T_VALS   = [50, 100, 200, 500]
    K_VALS   = [1, 2, 4]
    N_MIN    = 50
    N_MAX    = 4000
    BS_STEPS = 8

    # Precompute FULL N_MAX training tensor once (slice for each binary search step)
    print("Precomputing CIFAR-10 (20% noise)...", flush=True)
    train_full, test_set = get_cifar10(data_dir=args.data_dir, augment=False)
    train_full = corrupt_labels(train_full, 0.20, seed=42)
    subset_max = make_subset(train_full, N_MAX, seed=42)

    loader = TDataLoader(subset_max, batch_size=512, shuffle=False, num_workers=0)
    xs, ys = [], []
    for x, y in loader:
        xs.append(x); ys.append(y)
    X_all = torch.cat(xs).to(device)   # (N_MAX, 3, 32, 32)
    Y_all = torch.cat(ys).to(device)   # (N_MAX,)
    print(f"  X_all: {tuple(X_all.shape)}", flush=True)

    def _train_check(k, n, T):
        torch.manual_seed(42); np.random.seed(42)
        model = ResNet(num_classes=10, k=k)
        X_n = X_all[:n]; Y_n = Y_all[:n]
        hist = _train_fast(
            model, device, X_n, Y_n, X_n, Y_n,  # use train as both train and "test" to check memorization
            epochs=T, optimizer_type="sgd", lr=0.1,
            batch_size=min(256, n), momentum=0.9, weight_decay=0.0,
            log_interval=T + 1, eval_interval=T,  # only eval at end
            verbose=False,
        )
        # Use the last non-NaN train accuracy
        final_train_acc = [v for v in hist["train_acc"] if not np.isnan(v)][-1]
        return (100 - final_train_acc) < (EPS * 100)

    results = {}
    for k in K_VALS:
        p = ResNet(num_classes=10, k=k).count_parameters()
        results[k] = {"num_params": p, "emc": {}}
        for T in T_VALS:
            lo, hi = N_MIN, N_MAX
            print(f"\nResNet k={k} (p={p:,}), T={T} -- binary search:", flush=True)
            for step in range(BS_STEPS):
                mid  = (lo + hi) // 2
                fits = _train_check(k, mid, T)
                print(f"  step {step+1}/{BS_STEPS}: n={mid:4d}  fits? {fits}  [{lo},{hi}]",
                      flush=True)
                if fits:
                    lo = mid
                else:
                    hi = mid - 1
            emc = lo
            results[k]["emc"][T] = emc
            print(f"  => EMC(T={T}, k={k}) ≈ {emc}", flush=True)

    out = os.path.join(args.output_dir, "expA_emc")
    os.makedirs(out, exist_ok=True)
    with open(os.path.join(out, "results.json"), "w") as f:
        json.dump({str(k): v for k, v in results.items()}, f, indent=2)

    fig, ax = plt.subplots(figsize=(9, 5))
    colors = {1: "tab:blue", 2: "tab:orange", 4: "tab:green"}
    for k in K_VALS:
        Ts   = sorted(results[k]["emc"].keys())
        emcs = [results[k]["emc"][T] for T in Ts]
        ax.plot(Ts, emcs, "o-", color=colors[k],
                label=f"ResNet k={k} (p={results[k]['num_params']:,})")
    ax.axhline(y=N_MAX, color="gray", linestyle="--", alpha=0.6,
               label=f"n={N_MAX} (our training set)")
    ax.set_xlabel("Epoch budget T")
    ax.set_ylabel("EMC(T)  [max n with train_err < 5%]")
    ax.set_title("Effective Model Complexity: ResNet on CIFAR-10 (20% noise)")
    ax.legend(); ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(out, "emc_curves.png"), bbox_inches="tight", dpi=150)
    plt.close()
    print(f"\nSaved → {out}", flush=True)
    return results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Shufeng's DD experiments (sc5739)")
    parser.add_argument("--experiments", type=str, default="noise_multiseed,B",
        help="Comma-separated: noise_multiseed, B, C, A")
    parser.add_argument("--n-train",    type=int, default=1000)
    parser.add_argument("--n-train-nn", type=int, default=4000)
    parser.add_argument("--epochs-C",   type=int, default=4000)
    parser.add_argument("--data-dir",   type=str, default="./data")
    parser.add_argument("--output-dir", type=str, default="./results")
    args = parser.parse_args()

    exps = [e.strip() for e in args.experiments.split(",")]
    print(f"Running experiments: {exps}\nOutput: {args.output_dir}\n", flush=True)

    if "noise_multiseed" in exps:
        exp_noise_multiseed(args)
    if "B" in exps:
        expB_bias_variance(args)
    if "C" in exps:
        expC_epoch_sgd(args)
    if "A" in exps:
        expA_emc(args)

    print("\n" + "="*70)
    print("ALL DONE")
    print(f"Results in: {args.output_dir}")
    print("="*70, flush=True)


if __name__ == "__main__":
    main()
