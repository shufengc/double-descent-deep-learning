"""
Activation-function ablation at the DD-recovery transition.

Holds the fractional-k ResNet at k=0.1875, the observed phase-transition
region, and compares ReLU, GELU, and Tanh over two seeds. The goal is to test
whether the DD-recovery onset is robust to the nonlinearity or is activation
dependent.

Default full run:
  python -m src.experiments.exp_activation_ablation --device cuda

Smoke test:
  python -m src.experiments.exp_activation_ablation --smoke --device cpu

Outputs:
  results/activation_ablation/results.json
  results/activation_ablation/dd_curves.png

Raw per-run JSON files are cached locally under
results/activation_ablation/{activation}_k{K}_n{N}_ep{EP}_s{S}/ and ignored by
git; the committed result surface stays to one JSON and one figure.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from itertools import product
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.experiments.exp_dd_recovery import (
    IndexedSubset,
    build_cifar_datasets,
    compute_effective_rank,
    corrupt_and_subsample,
)


ACTIVATIONS = ("relu", "gelu", "tanh")


def make_activation(name: str) -> nn.Module:
    if name == "relu":
        return nn.ReLU(inplace=False)
    if name == "gelu":
        return nn.GELU()
    if name == "tanh":
        return nn.Tanh()
    raise ValueError(f"Unknown activation: {name}")


class ActBlock(nn.Module):
    def __init__(self, cin, cout, stride, norm, activation):
        super().__init__()
        self.conv1 = nn.Conv2d(cin, cout, 3, stride, 1, bias=False)
        self.n1 = norm(cout)
        self.act1 = make_activation(activation)
        self.conv2 = nn.Conv2d(cout, cout, 3, 1, 1, bias=False)
        self.n2 = norm(cout)
        self.act2 = make_activation(activation)
        self.short = nn.Sequential()
        if stride != 1 or cin != cout:
            self.short = nn.Sequential(
                nn.Conv2d(cin, cout, 1, stride, bias=False),
                norm(cout),
            )

    def forward(self, x):
        y = self.act1(self.n1(self.conv1(x)))
        y = self.n2(self.conv2(y))
        return self.act2(y + self.short(x))


class ResNetKActivation(nn.Module):
    """3-stage fractional-k ResNet with configurable activation."""

    def __init__(self, k, activation="relu", num_classes=10):
        super().__init__()
        c1 = max(1, round(16 * k))
        c2 = max(1, round(32 * k))
        c3 = max(1, round(64 * k))
        self.widths = (c1, c2, c3)
        self.activation = activation
        use_gn = min(self.widths) <= 2

        def norm(n):
            if use_gn:
                groups = 1 if n < 4 else min(8, n)
                while n % groups != 0:
                    groups -= 1
                return nn.GroupNorm(groups, n)
            return nn.BatchNorm2d(n)

        self.norm_kind = "gn" if use_gn else "bn"
        self.conv1 = nn.Conv2d(3, c1, 3, 1, 1, bias=False)
        self.n1 = norm(c1)
        self.act1 = make_activation(activation)
        self.layer1 = nn.Sequential(
            ActBlock(c1, c1, 1, norm, activation),
            ActBlock(c1, c1, 1, norm, activation),
        )
        self.layer2 = nn.Sequential(
            ActBlock(c1, c2, 2, norm, activation),
            ActBlock(c2, c2, 1, norm, activation),
        )
        self.layer3 = nn.Sequential(
            ActBlock(c2, c3, 2, norm, activation),
            ActBlock(c3, c3, 1, norm, activation),
        )
        self.linear = nn.Linear(c3, num_classes)

    def forward(self, x, return_feats=False):
        y = self.act1(self.n1(self.conv1(x)))
        y = self.layer3(self.layer2(self.layer1(y)))
        y = nn.functional.adaptive_avg_pool2d(y, 1).flatten(1)
        out = self.linear(y)
        if return_feats:
            return out, y
        return out

    def count_params(self):
        return sum(p.numel() for p in self.parameters())


def make_loader(dataset, batch_size, shuffle, device, num_workers):
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=device.startswith("cuda"),
        persistent_workers=num_workers > 0,
        drop_last=False,
    )


def evaluate(model, loader, loss_fn, device):
    model.eval()
    total, correct, loss_sum = 0, 0, 0.0
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            out = model(x)
            loss = loss_fn(out, y)
            loss_sum += loss.item() * y.size(0)
            correct += (out.argmax(1) == y).sum().item()
            total += y.size(0)
    return {
        "loss": loss_sum / max(1, total),
        "acc": 100.0 * correct / max(1, total),
    }


def train_one(args, activation, seed, epochs):
    run_name = (
        f"{activation}_k{args.k:g}_n{args.n_train}_ep{epochs}_s{seed}"
    )
    run_dir = Path(args.out_root) / run_name
    results_path = run_dir / "results.json"
    if results_path.exists() and not args.force:
        print(f"[activation] SKIP cached: {run_name}", flush=True)
        return json.loads(results_path.read_text())

    run_dir.mkdir(parents=True, exist_ok=True)
    cfg = {
        "activation": activation,
        "k": args.k,
        "n_train": args.n_train,
        "noise_rate": args.noise_rate,
        "epochs": epochs,
        "seed": seed,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "augment": not args.no_augment,
    }
    (run_dir / "config.json").write_text(json.dumps(cfg, indent=2))

    torch.manual_seed(seed)
    np.random.seed(seed)
    if args.device.startswith("cuda"):
        torch.cuda.manual_seed_all(seed)

    train_full, test_full = build_cifar_datasets(
        args.data_dir,
        augment=not args.no_augment,
    )
    subset, noise_mask, _, _ = corrupt_and_subsample(
        train_full,
        args.n_train,
        args.noise_rate,
        seed,
    )
    train_loader = make_loader(
        IndexedSubset(subset),
        args.batch_size,
        True,
        args.device,
        args.num_workers,
    )
    test_loader = make_loader(
        test_full,
        512,
        False,
        args.device,
        args.num_workers,
    )

    model = ResNetKActivation(args.k, activation=activation).to(args.device)
    params = model.count_params()
    print(
        f"[activation] RUN {run_name}: widths={model.widths} "
        f"norm={model.norm_kind} params={params:,} p/n={params/args.n_train:.2f}",
        flush=True,
    )

    opt = optim.Adam(model.parameters(), lr=args.lr, weight_decay=0.0)
    loss_fn = nn.CrossEntropyLoss()
    noise_mask_t = torch.tensor(noise_mask, dtype=torch.bool, device=args.device)
    history = []
    best_test_acc = 0.0
    best_epoch = 0
    t0 = time.time()

    for ep in range(1, epochs + 1):
        model.train()
        total, correct, loss_sum = 0, 0, 0.0
        corrupt_correct, corrupt_total = 0, 0
        for x, y, idx in train_loader:
            x = x.to(args.device, non_blocking=True)
            y = y.to(args.device, non_blocking=True)
            idx = idx.to(args.device, non_blocking=True)
            out = model(x)
            loss = loss_fn(out, y)
            opt.zero_grad()
            loss.backward()
            opt.step()

            with torch.no_grad():
                pred = out.argmax(1)
                total += y.size(0)
                correct += (pred == y).sum().item()
                loss_sum += loss.item() * y.size(0)
                corr_mask = noise_mask_t[idx]
                if corr_mask.any():
                    corrupt_total += corr_mask.sum().item()
                    corrupt_correct += (pred[corr_mask] == y[corr_mask]).sum().item()

        record = {
            "epoch": ep,
            "train_loss": loss_sum / max(1, total),
            "train_acc": 100.0 * correct / max(1, total),
            "memorization_frac": (
                corrupt_correct / corrupt_total if corrupt_total else None
            ),
        }

        if ep % args.eval_every == 0 or ep == epochs:
            metrics = evaluate(model, test_loader, loss_fn, args.device)
            record["test_loss"] = metrics["loss"]
            record["test_acc"] = metrics["acc"]
            if metrics["acc"] > best_test_acc:
                best_test_acc = metrics["acc"]
                best_epoch = ep

        if ep == 1 or ep % args.log_every == 0 or ep == epochs:
            msg = (
                f"  {activation} seed={seed} ep {ep:4d}/{epochs} "
                f"train={record['train_acc']:5.2f}%"
            )
            if "test_acc" in record:
                msg += f" test={record['test_acc']:5.2f}%"
            msg += f" [{time.time() - t0:.0f}s]"
            print(msg, flush=True)
        history.append(record)

    final_eval = evaluate(model, test_loader, loss_fn, args.device)
    effective_rank = compute_effective_rank(
        model,
        test_loader,
        args.device,
        n_samples=args.eff_rank_samples,
    )
    result = {
        "config": cfg,
        "params": params,
        "widths": list(model.widths),
        "norm_kind": model.norm_kind,
        "activation": activation,
        "best_test_acc": best_test_acc,
        "best_epoch": best_epoch,
        "final_test_acc": final_eval["acc"],
        "final_test_loss": final_eval["loss"],
        "final_train_acc": history[-1]["train_acc"],
        "final_memorization_frac": history[-1]["memorization_frac"],
        "effective_rank": effective_rank,
        "history": history,
        "wallclock_sec": time.time() - t0,
    }
    results_path.write_text(json.dumps(result))
    return result


def summarize(results, out_root):
    rows = []
    for r in results:
        cfg = r["config"]
        rows.append(
            {
                "activation": cfg["activation"],
                "k": cfg["k"],
                "n": cfg["n_train"],
                "seed": cfg["seed"],
                "params": r["params"],
                "widths": r["widths"],
                "best_test_acc": r["best_test_acc"],
                "best_epoch": r["best_epoch"],
                "final_test_acc": r["final_test_acc"],
                "final_train_acc": r["final_train_acc"],
                "memorization_frac": r["final_memorization_frac"],
                "effective_rank": r["effective_rank"],
                "wallclock_min": r["wallclock_sec"] / 60.0,
            }
        )
    out_root = Path(out_root)
    out_root.mkdir(parents=True, exist_ok=True)
    summary_path = out_root / "results.json"
    summary_path.write_text(json.dumps(rows, indent=2))
    print(f"[activation] wrote results: {summary_path}", flush=True)
    return rows


def plot_summary(rows, figure_path):
    if not rows:
        return
    figure_path = Path(figure_path)
    figure_path.parent.mkdir(parents=True, exist_ok=True)

    grouped = {act: [] for act in ACTIVATIONS}
    for row in rows:
        grouped.setdefault(row["activation"], []).append(row)
    activations = [act for act in ACTIVATIONS if grouped.get(act)]
    means = [
        np.mean([r["best_test_acc"] for r in grouped[act]])
        for act in activations
    ]
    stds = [
        np.std([r["best_test_acc"] for r in grouped[act]], ddof=0)
        for act in activations
    ]
    eff_means = [
        np.mean([r["effective_rank"] for r in grouped[act]])
        for act in activations
    ]

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    x = np.arange(len(activations))
    colors = ["tab:blue", "tab:orange", "tab:green"][: len(activations)]
    axes[0].bar(x, means, yerr=stds, capsize=4, color=colors, alpha=0.78)
    for i, act in enumerate(activations):
        seed_vals = [r["best_test_acc"] for r in grouped[act]]
        axes[0].scatter(
            np.full(len(seed_vals), i),
            seed_vals,
            color="black",
            s=24,
            zorder=3,
            label="Individual seed runs" if i == 0 else None,
        )
    axes[0].set_xticks(x, [act.upper() for act in activations])
    axes[0].set_ylabel("Best test accuracy (%)")
    axes[0].set_title("Activation ablation at k=0.1875")
    axes[0].grid(axis="y", alpha=0.25)
    axes[0].legend(loc="upper left", frameon=False)

    axes[1].bar(x, eff_means, color=colors, alpha=0.78)
    axes[1].set_xticks(x, [act.upper() for act in activations])
    axes[1].set_ylabel("Penultimate-feature stable rank")
    axes[1].set_title("Feature-rank diagnostic")
    axes[1].grid(axis="y", alpha=0.25)

    fig.suptitle("Robustness of DD-recovery onset to activation function", y=1.02)
    fig.tight_layout()
    fig.savefig(figure_path, bbox_inches="tight", dpi=200)
    plt.close(fig)
    print(f"[activation] wrote figure: {figure_path}", flush=True)


def parse_csv(raw, cast=str):
    return [cast(x.strip()) for x in raw.split(",") if x.strip()]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--activations", default="relu,gelu,tanh")
    p.add_argument("--seeds", default="42,7")
    p.add_argument("--k", type=float, default=0.1875)
    p.add_argument("--n-train", type=int, default=4000)
    p.add_argument("--epochs", type=int, default=1500)
    p.add_argument("--noise-rate", type=float, default=0.15)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--eval-every", type=int, default=100)
    p.add_argument("--log-every", type=int, default=100)
    p.add_argument("--eff-rank-samples", type=int, default=1024)
    p.add_argument("--num-workers", type=int, default=2)
    p.add_argument("--data-dir", default="./data")
    p.add_argument("--out-root", default="./results/activation_ablation")
    p.add_argument(
        "--figure-path",
        default="./results/activation_ablation/dd_curves.png",
    )
    p.add_argument("--device", default="cuda")
    p.add_argument("--no-augment", action="store_true")
    p.add_argument("--smoke", action="store_true")
    p.add_argument("--force", action="store_true")
    args = p.parse_args()

    if args.smoke:
        args.activations = "relu"
        args.seeds = "42"
        args.n_train = 128
        args.epochs = 1
        args.eval_every = 1
        args.log_every = 1
        args.batch_size = min(args.batch_size, 64)
        args.num_workers = 0
        args.out_root = "/private/tmp/activation_ablation_smoke"
        args.figure_path = "/private/tmp/activation_ablation_smoke.png"
        args.eff_rank_samples = 128

    activations = parse_csv(args.activations)
    invalid = [act for act in activations if act not in ACTIVATIONS]
    if invalid:
        raise ValueError(f"Unsupported activations: {invalid}")
    seeds = parse_csv(args.seeds, int)
    grid = list(product(activations, seeds))
    print(
        f"[activation] plan: {len(grid)} runs, activations={activations}, "
        f"seeds={seeds}, k={args.k}, n={args.n_train}, epochs={args.epochs}",
        flush=True,
    )

    results = []
    t_total = time.time()
    for i, (activation, seed) in enumerate(grid, 1):
        print(f"\n[{i}/{len(grid)}] activation={activation} seed={seed}", flush=True)
        results.append(train_one(args, activation, seed, args.epochs))

    rows = summarize(results, args.out_root)
    plot_summary(rows, args.figure_path)
    print(f"[activation] complete. wall {(time.time() - t_total) / 60:.1f} min")


if __name__ == "__main__":
    main()
