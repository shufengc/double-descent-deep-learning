"""
Extension C — Adam vs SGD on CNN (model-wise + epoch-wise).

Runs a CNN width sweep on CIFAR-10 with both optimizers and both noise levels.
Uses GPU-resident training (no DataLoader, all tensors live on the GPU)
following the shufeng_experiments._train_fast pattern, which removes the
host-side bottleneck observed when using DataLoader+make_loaders for tiny
n=4000 datasets.

Default configuration (full run on a 5090):
    widths      = [8, 16, 32, 64]            num_filters
    optimizers  = ["sgd", "adam"]
    noises      = [0.0, 0.15]                 clean + Nakkiran-noise
    seeds       = [42]
    epochs      = 250
    n_train     = 4000
    => 4 * 2 * 2 * 1 = 16 runs

Both optimizers run with constant learning rate (no scheduler) so the
comparison isolates the optimizer step rule. Eval every `eval_interval` epochs.

Saves per-run history to:
  <output-dir>/<run_id>/history.json
And a summary table to <output-dir>/summary.json.
"""

import argparse
import json
import os
import sys
import time
from itertools import product

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader as TDataLoader

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from src.models import CNN
from src.data import get_cifar10, corrupt_labels, make_subset


def pick_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def precompute_gpu(*, data_dir, noise, n_train, seed, device):
    train_full, test_set = get_cifar10(data_dir=data_dir, augment=False)
    if noise > 0:
        train_full = corrupt_labels(train_full, noise, seed=seed)
    train_subset = make_subset(train_full, n_train, seed=seed)

    def _to_gpu(ds):
        loader = TDataLoader(ds, batch_size=1024, shuffle=False, num_workers=0)
        xs, ys = [], []
        for x, y in loader:
            xs.append(x); ys.append(y)
        return torch.cat(xs).to(device), torch.cat(ys).to(device)

    X_tr, Y_tr = _to_gpu(train_subset)
    X_te, Y_te = _to_gpu(test_set)
    return X_tr, Y_tr, X_te, Y_te


def train_gpu_resident(*, model, device, X_tr, Y_tr, X_te, Y_te,
                       epochs, optimizer_type, lr,
                       batch_size=512, momentum=0.9, weight_decay=0.0,
                       eval_interval=5, log_interval=25):
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    n = X_tr.shape[0]

    if optimizer_type == "sgd":
        opt = torch.optim.SGD(model.parameters(), lr=lr,
                              momentum=momentum, weight_decay=weight_decay)
    elif optimizer_type == "adam":
        opt = torch.optim.Adam(model.parameters(), lr=lr,
                               weight_decay=weight_decay)
    else:
        raise ValueError(optimizer_type)

    history = {"epoch": [], "train_loss": [], "train_acc": [],
               "test_loss": [], "test_acc": []}

    for ep in range(1, epochs + 1):
        model.train()
        perm = torch.randperm(n, device=device)
        tl, tc, tt = 0.0, 0, 0
        for i in range(0, n, batch_size):
            idx = perm[i:i + batch_size]
            x, y = X_tr[idx], Y_tr[idx]
            opt.zero_grad()
            out = model(x)
            loss = criterion(out, y)
            loss.backward()
            opt.step()
            tl += loss.item() * x.size(0)
            tc += out.argmax(1).eq(y).sum().item()
            tt += x.size(0)
        train_loss = tl / tt
        train_acc = 100.0 * tc / tt

        if ep % eval_interval == 0 or ep == epochs or ep == 1:
            model.eval()
            with torch.no_grad():
                el, ec, et = 0.0, 0, 0
                nte = X_te.shape[0]
                for j in range(0, nte, batch_size):
                    x, y = X_te[j:j + batch_size], Y_te[j:j + batch_size]
                    out = model(x)
                    loss = criterion(out, y)
                    el += loss.item() * x.size(0)
                    ec += out.argmax(1).eq(y).sum().item()
                    et += x.size(0)
            test_loss = el / et
            test_acc = 100.0 * ec / et
        else:
            test_loss = float("nan")
            test_acc = float("nan")

        history["epoch"].append(ep)
        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["test_loss"].append(test_loss)
        history["test_acc"].append(test_acc)

        if ep % log_interval == 0 or ep == epochs:
            print(f"    ep {ep:4d}/{epochs}  "
                  f"tr_acc {train_acc:5.1f}%  te_acc {test_acc:5.1f}%",
                  flush=True)

    return history


def run_one(*, width, optimizer_type, noise, seed, n_train, epochs,
            X_tr, Y_tr, X_te, Y_te, device,
            lr_adam=1e-4, lr_sgd=0.05, momentum=0.9,
            batch_size=512, eval_interval=5, log_interval=25):
    torch.manual_seed(seed)
    np.random.seed(seed)

    model = CNN(num_classes=10, num_filters=width, input_channels=3)
    n_params = model.count_parameters()
    lr = lr_adam if optimizer_type == "adam" else lr_sgd

    t0 = time.time()
    history = train_gpu_resident(
        model=model, device=device,
        X_tr=X_tr, Y_tr=Y_tr, X_te=X_te, Y_te=Y_te,
        epochs=epochs, optimizer_type=optimizer_type, lr=lr,
        batch_size=batch_size, momentum=momentum,
        eval_interval=eval_interval, log_interval=log_interval,
    )
    elapsed = time.time() - t0

    # Final epoch may have NaN test if eval_interval misaligns; force final eval
    if not np.isnan(history["test_acc"][-1]):
        final_test_acc = history["test_acc"][-1]
        final_test_loss = history["test_loss"][-1]
    else:
        # last valid eval
        for i in range(len(history["test_acc"]) - 1, -1, -1):
            if not np.isnan(history["test_acc"][i]):
                final_test_acc = history["test_acc"][i]
                final_test_loss = history["test_loss"][i]
                break

    return {
        "config": {
            "width": width, "num_params": n_params,
            "optimizer": optimizer_type, "lr": lr, "momentum": momentum,
            "noise": noise, "seed": seed, "n_train": n_train,
            "epochs": epochs, "batch_size": batch_size,
            "scheduler": None, "eval_interval": eval_interval,
            "p_over_n": n_params / n_train,
        },
        "history": history,
        "elapsed_sec": elapsed,
        "final": {
            "train_acc": history["train_acc"][-1],
            "test_acc": final_test_acc,
            "train_loss": history["train_loss"][-1],
            "test_loss": final_test_loss,
        },
    }


def run(args):
    device = pick_device()
    print(f"Device: {device}", flush=True)
    if device.type == "cuda":
        print(f"  GPU: {torch.cuda.get_device_name(0)}", flush=True)

    widths = [int(x) for x in args.widths.split(",")]
    optimizers = args.optimizers.split(",")
    noises = [float(x) for x in args.noises.split(",")]
    seeds = [int(x) for x in args.seeds.split(",")]
    epochs = args.epochs

    os.makedirs(args.output_dir, exist_ok=True)
    grid = list(product(noises, seeds, widths, optimizers))
    print(f"Plan: {len(grid)} runs of {epochs} epochs each", flush=True)
    print(f"  widths={widths}\n  optimizers={optimizers}\n"
          f"  noises={noises}\n  seeds={seeds}\n"
          f"  batch_size={args.batch_size}", flush=True)

    summary = []
    summary_path = os.path.join(args.output_dir, "summary.json")
    cached = {}  # (noise, seed) -> (X_tr, Y_tr, X_te, Y_te)

    for i, (noise, seed, width, opt) in enumerate(grid, 1):
        run_id = f"n{int(noise * 100):02d}_w{width}_{opt}_s{seed}"
        out_dir = os.path.join(args.output_dir, run_id)
        os.makedirs(out_dir, exist_ok=True)
        history_path = os.path.join(out_dir, "history.json")
        if os.path.exists(history_path) and not args.overwrite:
            with open(history_path) as f:
                rec = json.load(f)
            summary.append({
                "run_id": run_id, **rec["config"], **rec["final"],
                "elapsed_sec": rec.get("elapsed_sec"),
            })
            print(f"\n[{i}/{len(grid)}] {run_id}  (skip: cached)", flush=True)
            continue

        if (noise, seed) not in cached:
            print(f"\nPrecomputing CIFAR-10 (noise={noise}, seed={seed})...",
                  flush=True)
            cached[(noise, seed)] = precompute_gpu(
                data_dir=args.data_dir, noise=noise,
                n_train=args.n_train, seed=seed, device=device)
            print(f"  done.", flush=True)
        X_tr, Y_tr, X_te, Y_te = cached[(noise, seed)]

        print(f"\n[{i}/{len(grid)}] {run_id}  "
              f"noise={noise} width={width} opt={opt} seed={seed}", flush=True)
        rec = run_one(
            width=width, optimizer_type=opt, noise=noise, seed=seed,
            n_train=args.n_train, epochs=epochs,
            X_tr=X_tr, Y_tr=Y_tr, X_te=X_te, Y_te=Y_te, device=device,
            batch_size=args.batch_size, eval_interval=args.eval_interval,
            log_interval=max(1, epochs // 10),
        )
        with open(history_path, "w") as f:
            json.dump(rec, f, indent=2)
        summary.append({
            "run_id": run_id, **rec["config"], **rec["final"],
            "elapsed_sec": rec["elapsed_sec"],
        })
        with open(summary_path, "w") as f:
            json.dump(summary, f, indent=2)
        print(f"  -> final tr_acc={rec['final']['train_acc']:.2f}%  "
              f"te_acc={rec['final']['test_acc']:.2f}%  "
              f"({rec['elapsed_sec']:.0f}s)", flush=True)

    print("\nDone. Summary:")
    print(f"{'run_id':<28} {'p/n':>7} {'train':>8} {'test':>8}")
    for s in summary:
        print(f"{s['run_id']:<28} {s['p_over_n']:>7.3f} "
              f"{s['train_acc']:>7.2f}% {s['test_acc']:>7.2f}%")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--widths", default="8,16,32,64")
    parser.add_argument("--optimizers", default="sgd,adam")
    parser.add_argument("--noises", default="0.0,0.15")
    parser.add_argument("--seeds", default="42")
    parser.add_argument("--epochs", type=int, default=250)
    parser.add_argument("--n-train", type=int, default=4000)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--eval-interval", type=int, default=5)
    parser.add_argument("--data-dir", default="./data")
    parser.add_argument("--output-dir", default="./results/personC_optimizer")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    t0 = time.time()
    run(args)
    print(f"\nTotal wall time: {(time.time() - t0) / 60:.1f} min")


if __name__ == "__main__":
    main()
