"""
Quick full empirical-NTK verification experiment for fractional-k ResNet.

Typical usage
-------------
Smoke test:
    python -m src.experiments.exp_full_empirical_ntk_quick --smoke --device cuda

Quick run:
    python -m src.experiments.exp_full_empirical_ntk_quick \
      --ks 0.125,0.1875,0.25,0.5 \
      --n-train 1000 --epochs 200 --n-ntk-samples 12 \
      --seed 42 --device cuda

Output
------
results/full_empirical_ntk_quick/
  summary.json
  k0.125_s42/ntk_spectrum.json
  k0.125_s42/checkpoint.pt
  ...
figures/full_empirical_ntk_quick.png
"""

from __future__ import annotations

import argparse
import json
import math
import os
import time
from collections import OrderedDict
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.func import functional_call, jacrev
from torch.utils.data import DataLoader, Subset

# Reuse Shufeng's exact fractional-k model and data helpers.
from src.experiments.exp_dd_recovery import (
    ResNetK,
    build_cifar_datasets,
    corrupt_and_subsample,
    IndexedSubset,
)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


DEFAULT_KS = [0.125, 0.1875, 0.25, 0.5]


def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def make_loaders(
    data_dir: str,
    n_train: int,
    noise_rate: float,
    seed: int,
    batch_size: int,
    augment: bool,
    num_workers: int,
) -> Tuple[DataLoader, DataLoader, torch.utils.data.Dataset]:
    train_full, test_full = build_cifar_datasets(data_dir, augment=augment)
    train_subset, _noise_mask, _subset_idx, _orig_labels = corrupt_and_subsample(
        train_full, n_train, noise_rate, seed
    )
    train_loader = DataLoader(
        IndexedSubset(train_subset),
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=False,
        persistent_workers=num_workers > 0,
    )
    test_loader = DataLoader(
        test_full,
        batch_size=512,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=num_workers > 0,
    )
    return train_loader, test_loader, test_full


def train_model(
    k: float,
    train_loader: DataLoader,
    test_loader: DataLoader,
    epochs: int,
    lr: float,
    device: str,
    log_every: int,
) -> Tuple[ResNetK, Dict[str, float]]:
    model = ResNetK(k).to(device)
    opt = optim.Adam(model.parameters(), lr=lr, weight_decay=0.0)
    loss_fn = nn.CrossEntropyLoss()

    t0 = time.time()
    last_train_acc = 0.0
    for ep in range(1, epochs + 1):
        model.train()
        total = 0
        correct = 0
        loss_sum = 0.0
        for x, y, _idx in train_loader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            out = model(x)
            loss = loss_fn(out, y)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()

            with torch.no_grad():
                total += y.numel()
                correct += (out.argmax(1) == y).sum().item()
                loss_sum += loss.item() * y.numel()
        last_train_acc = 100.0 * correct / max(1, total)

        if ep == 1 or ep == epochs or ep % log_every == 0:
            print(
                f"    ep {ep:4d}/{epochs} train_acc={last_train_acc:5.2f}% "
                f"train_loss={loss_sum / max(1, total):.4f}",
                flush=True,
            )

    test_acc_value, test_loss_value = evaluate(model, test_loader, device)
    stats = {
        "train_acc": float(last_train_acc),
        "test_acc": float(test_acc_value),
        "test_loss": float(test_loss_value),
        "train_wallclock_sec": float(time.time() - t0),
    }
    return model, stats


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, device: str) -> Tuple[float, float]:
    model.eval()
    loss_fn = nn.CrossEntropyLoss(reduction="sum")
    total = 0
    correct = 0
    loss_sum = 0.0
    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        out = model(x)
        loss_sum += loss_fn(out, y).item()
        correct += (out.argmax(1) == y).sum().item()
        total += y.numel()
    return 100.0 * correct / max(1, total), loss_sum / max(1, total)


def first_n_test_images(test_set, n: int, device: str) -> torch.Tensor:
    xs = []
    for i in range(n):
        x, _y = test_set[i]
        xs.append(x)
    return torch.stack(xs, dim=0).to(device)


def flatten_param_jacobian(jac: Dict[str, torch.Tensor], max_classes: int | None) -> torch.Tensor:
    """Convert a jacrev parameter pytree into J_x ∈ R^{C×P}.

    single_logits() returns shape [C]. Therefore jac[name] has shape
    [C, *param_shape].
    """
    flat_parts = []
    for _name, g in jac.items():
        if max_classes is not None:
            g = g[:max_classes]
        flat_parts.append(g.reshape(g.shape[0], -1))
    return torch.cat(flat_parts, dim=1)


def empirical_ntk_from_full_jacobian(
    model: nn.Module,
    x_ntk: torch.Tensor,
    device: str,
    max_classes: int | None = None,
) -> Tuple[np.ndarray, Dict[str, float]]:
    """Compute multi-output empirical NTK from full parameter Jacobians.

    K_ij = sum_c <∇_θ f_c(x_i), ∇_θ f_c(x_j)>.

    We loop over samples to keep memory bounded. This still uses torch.func.jacrev
    over all trainable parameters for each image.
    """
    model.eval()

    params = OrderedDict((name, p.detach().to(device).requires_grad_(True))
                         for name, p in model.named_parameters())
    buffers = OrderedDict((name, b.detach().to(device))
                          for name, b in model.named_buffers())

    def single_logits(params_, buffers_, x_single):
        out = functional_call(model, (params_, buffers_), (x_single.unsqueeze(0),))
        return out.squeeze(0)

    J_list: List[torch.Tensor] = []
    t0 = time.time()
    for i in range(x_ntk.shape[0]):
        jac = jacrev(single_logits, argnums=0)(params, buffers, x_ntk[i])
        J_i = flatten_param_jacobian(jac, max_classes=max_classes).detach().cpu()
        J_list.append(J_i)
        print(f"      jacobian {i + 1:02d}/{x_ntk.shape[0]}  shape={tuple(J_i.shape)}", flush=True)

    # J has shape [N, C, P]. Multi-output NTK sums over class and parameter dims.
    J = torch.stack(J_list, dim=0).float()
    K = torch.einsum("ncp,mcp->nm", J, J)
    K = 0.5 * (K + K.T)
    K_np = K.numpy().astype(float)

    wall = time.time() - t0
    stats = {
        "jacobian_wallclock_sec": float(wall),
        "n_ntk_samples": int(x_ntk.shape[0]),
        "num_logits_used": int(J.shape[1]),
        "num_trainable_params": int(J.shape[2]),
    }
    return K_np, stats


def spectrum_diagnostics(K: np.ndarray) -> Dict[str, object]:
    eig = np.linalg.eigvalsh(K)
    eig = np.maximum(eig, 0.0)
    eig = eig[::-1]

    trace = float(eig.sum())
    top = float(eig[0]) if eig.size else 0.0
    eps = 1e-12
    positive = eig[eig > max(eps, top * 1e-10)]
    min_pos = float(positive[-1]) if positive.size else eps

    stable_rank = trace / max(top, eps)
    participation_ratio = (trace ** 2) / max(float((eig ** 2).sum()), eps)
    condition_number = top / max(min_pos, eps)
    normalized_stable_rank = stable_rank / max(1, K.shape[0])
    normalized_participation_ratio = participation_ratio / max(1, K.shape[0])

    p = eig / max(trace, eps)
    spectral_entropy = float(-(p * np.log(p + 1e-30)).sum())

    return {
        "ntk_shape": list(K.shape),
        "trace": trace,
        "top_eigenvalue": top,
        "min_positive_eigenvalue": min_pos,
        "condition_number": float(condition_number),
        "stable_rank": float(stable_rank),
        "stable_rank_normalized": float(normalized_stable_rank),
        "participation_ratio": float(participation_ratio),
        "participation_ratio_normalized": float(normalized_participation_ratio),
        "spectral_entropy": spectral_entropy,
        "top_eigenvalues": eig[:25].tolist(),
        "all_eigenvalues": eig.tolist(),
    }


def save_checkpoint(model: nn.Module, path: Path, meta: Dict[str, object]) -> None:
    torch.save({"model_state_dict": model.state_dict(), "meta": meta}, path)


def plot_summary(summary: List[Dict[str, object]], output_path: str) -> None:
    if not summary:
        return
    rows = sorted(summary, key=lambda r: (float(r["k"]), int(r["seed"])))
    ks = [float(r["k"]) for r in rows]

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    axes = axes.ravel()

    axes[0].plot(ks, [r["stable_rank_normalized"] for r in rows], "o-")
    axes[0].set_ylabel("stable rank / N")
    axes[0].set_title("Full empirical-NTK stable rank")

    axes[1].plot(ks, [r["condition_number"] for r in rows], "o-")
    axes[1].set_yscale("log")
    axes[1].set_ylabel("condition number")
    axes[1].set_title("Full empirical-NTK condition number")

    axes[2].plot(ks, [r["participation_ratio_normalized"] for r in rows], "o-")
    axes[2].set_ylabel("participation ratio / N")
    axes[2].set_title("Full empirical-NTK participation ratio")

    axes[3].plot(ks, [r["test_acc"] for r in rows], "o-")
    axes[3].set_ylabel("test accuracy (%)")
    axes[3].set_title("Quick-trained test accuracy")

    for ax in axes:
        ax.axvline(0.1875, color="gray", linestyle=":", alpha=0.7)
        ax.set_xlabel("fractional width k")
        ax.grid(True, alpha=0.3)

    fig.suptitle("Quick full empirical-NTK diagnostics over fractional-k ResNet")
    plt.tight_layout()
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()


def run(args) -> None:
    if args.smoke:
        ks = [0.125]
        epochs = 2
        n_train = 256
        n_ntk_samples = 4
        max_classes = 3
        out_root = Path(args.out_root) / "smoke"
    else:
        ks = [float(x) for x in args.ks.split(",") if x.strip()]
        epochs = args.epochs
        n_train = args.n_train
        n_ntk_samples = args.n_ntk_samples
        max_classes = args.max_classes
        out_root = Path(args.out_root)

    out_root.mkdir(parents=True, exist_ok=True)
    Path(args.figure_path).parent.mkdir(parents=True, exist_ok=True)

    print("=" * 72)
    print("Quick full empirical-NTK experiment")
    print(f"ks={ks}")
    print(f"n_train={n_train}, epochs={epochs}, seed={args.seed}")
    print(f"noise_rate={args.noise_rate}, n_ntk_samples={n_ntk_samples}")
    print(f"device={args.device}")
    print("=" * 72)

    set_seed(args.seed)
    train_loader, test_loader, test_set = make_loaders(
        args.data_dir,
        n_train,
        args.noise_rate,
        args.seed,
        args.batch_size,
        args.augment,
        args.num_workers,
    )
    x_ntk = first_n_test_images(test_set, n_ntk_samples, args.device)

    summary: List[Dict[str, object]] = []
    summary_path = out_root / "summary.json"

    for idx, k in enumerate(ks, 1):
        run_name = f"k{k:g}_n{n_train}_ep{epochs}_s{args.seed}"
        run_dir = out_root / run_name
        run_dir.mkdir(parents=True, exist_ok=True)
        spec_path = run_dir / "ntk_spectrum.json"

        if spec_path.exists() and not args.force:
            print(f"\n[{idx}/{len(ks)}] {run_name} SKIP cached", flush=True)
            with open(spec_path) as f:
                record = json.load(f)
            summary.append(record)
            continue

        print(f"\n[{idx}/{len(ks)}] {run_name}", flush=True)
        set_seed(args.seed)
        model, train_stats = train_model(
            k=k,
            train_loader=train_loader,
            test_loader=test_loader,
            epochs=epochs,
            lr=args.lr,
            device=args.device,
            log_every=args.log_every,
        )
        model.eval()

        K, ntk_stats = empirical_ntk_from_full_jacobian(
            model,
            x_ntk,
            args.device,
            max_classes=max_classes,
        )
        diag = spectrum_diagnostics(K)

        meta = {
            "k": k,
            "seed": args.seed,
            "n_train": n_train,
            "epochs": epochs,
            "noise_rate": args.noise_rate,
            "augment": args.augment,
            "lr": args.lr,
            "params": model.count_params(),
            "widths": list(model.widths),
            "norm_kind": model.norm_kind,
        }
        record: Dict[str, object] = {
            **meta,
            **train_stats,
            **ntk_stats,
            **diag,
        }

        with open(spec_path, "w") as f:
            json.dump(record, f, indent=2)
        with open(run_dir / "config.json", "w") as f:
            json.dump(meta, f, indent=2)
        np.save(run_dir / "ntk_gram.npy", K)
        save_checkpoint(model, run_dir / "checkpoint.pt", meta)

        summary.append(record)
        slim = [{kk: vv for kk, vv in r.items() if kk not in {"all_eigenvalues"}}
                for r in summary]
        with open(summary_path, "w") as f:
            json.dump(slim, f, indent=2)

        print(
            f"    done k={k:g}: test_acc={record['test_acc']:.2f}% "
            f"stable_rank/N={record['stable_rank_normalized']:.3f} "
            f"cond={record['condition_number']:.2e}",
            flush=True,
        )

        del model, K
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # Final summary and plot.
    slim = [{kk: vv for kk, vv in r.items() if kk not in {"all_eigenvalues"}}
            for r in summary]
    with open(summary_path, "w") as f:
        json.dump(slim, f, indent=2)
    plot_summary(slim, args.figure_path)

    print("\nComplete.")
    print(f"Summary: {summary_path}")
    print(f"Figure:  {args.figure_path}")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--ks", default=",".join(str(x) for x in DEFAULT_KS))
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--n-train", type=int, default=1000)
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--noise-rate", type=float, default=0.15)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--log-every", type=int, default=50)
    p.add_argument("--n-ntk-samples", type=int, default=12)
    p.add_argument("--max-classes", type=int, default=None,
                   help="Optional speed hack. Default None uses all 10 logits.")
    p.add_argument("--data-dir", default="./data")
    p.add_argument("--out-root", default="./results/full_empirical_ntk_quick")
    p.add_argument("--figure-path", default="./figures/full_empirical_ntk_quick.png")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--num-workers", type=int, default=2)
    p.add_argument("--augment", action="store_true", default=True)
    p.add_argument("--no-augment", dest="augment", action="store_false")
    p.add_argument("--smoke", action="store_true")
    p.add_argument("--force", action="store_true")
    args = p.parse_args()
    run(args)


if __name__ == "__main__":
    main()
