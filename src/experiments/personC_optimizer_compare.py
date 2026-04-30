"""
Extension C — Adam vs SGD on CNN (model-wise + epoch-wise).

Runs a CNN width sweep on CIFAR-10 with both optimizers and both noise levels.
Records per-epoch train/test accuracy and loss so we can produce both
model-wise comparison curves and epoch-wise dynamics curves from the same
data. Both optimizers run with constant learning rate and no scheduler so
the comparison isolates optimizer-driven dynamics.

Default configuration (full run on a 5090, ~3 h):
    widths      = [4, 8, 16, 32, 48, 64]   num_filters
    optimizers  = ["sgd", "adam"]
    noises      = [0.0, 0.15]               clean + Nakkiran-noise
    seeds       = [42, 7]
    epochs      = 500
    n_train     = 4000
    => 6 * 2 * 2 * 2 = 48 runs

For pilot / smoke: --pilot reduces to 4 widths * 1 seed * 100 epochs.

Saves per-run history to:
  <output-dir>/<noise>_w<width>_<opt>_s<seed>/history.json
And a summary table to <output-dir>/summary.json.

Reuses src.models.CNN, src.trainer.Trainer, src.data.{get_cifar10,
corrupt_labels, make_subset, make_loaders}.
"""

import argparse
import json
import os
import sys
import time
from itertools import product

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from src.models import CNN
from src.data import get_cifar10, corrupt_labels, make_subset, make_loaders
from src.trainer import Trainer


def pick_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def run_one(*, width, optimizer_type, noise, seed, n_train, epochs,
            data_dir, device, lr_adam=1e-4, lr_sgd=0.05, momentum=0.9,
            batch_size=512, log_interval=50):
    torch.manual_seed(seed)
    np.random.seed(seed)

    train_full, test_set = get_cifar10(data_dir=data_dir, augment=False)
    if noise > 0:
        train_full = corrupt_labels(train_full, noise, seed=seed)
    train_set = make_subset(train_full, n_train, seed=seed)
    train_loader, test_loader = make_loaders(
        train_set, test_set, batch_size=batch_size)

    model = CNN(num_classes=10, num_filters=width, input_channels=3)
    n_params = model.count_parameters()

    lr = lr_adam if optimizer_type == "adam" else lr_sgd
    trainer = Trainer(
        model, device=device, lr=lr,
        momentum=momentum, weight_decay=0.0,
        optimizer_type=optimizer_type, scheduler_type=None,
    )
    t0 = time.time()
    history = trainer.train(
        train_loader, test_loader, epochs=epochs,
        log_interval=log_interval, verbose=True)
    elapsed = time.time() - t0

    return {
        "config": {
            "width": width, "num_params": n_params,
            "optimizer": optimizer_type, "lr": lr, "momentum": momentum,
            "noise": noise, "seed": seed, "n_train": n_train,
            "epochs": epochs, "batch_size": batch_size,
            "scheduler": None, "p_over_n": n_params / n_train,
        },
        "history": history,
        "elapsed_sec": elapsed,
        "final": {
            "train_acc": history["train_acc"][-1],
            "test_acc": history["test_acc"][-1],
            "train_loss": history["train_loss"][-1],
            "test_loss": history["test_loss"][-1],
        },
    }


def run(args):
    device = pick_device()
    print(f"Device: {device}")
    if args.pilot:
        widths = [4, 16, 32]
        optimizers = ["sgd", "adam"]
        noises = [0.15]
        seeds = [42]
        epochs = 100
    else:
        widths = [int(x) for x in args.widths.split(",")]
        optimizers = args.optimizers.split(",")
        noises = [float(x) for x in args.noises.split(",")]
        seeds = [int(x) for x in args.seeds.split(",")]
        epochs = args.epochs

    os.makedirs(args.output_dir, exist_ok=True)
    grid = list(product(noises, widths, optimizers, seeds))
    total = len(grid)
    print(f"Plan: {total} runs of {epochs} epochs each on {device}")
    print(f"  widths={widths}\n  optimizers={optimizers}\n  noises={noises}\n  seeds={seeds}")

    summary = []
    for i, (noise, width, opt, seed) in enumerate(grid, 1):
        run_id = f"n{int(noise * 100):02d}_w{width}_{opt}_s{seed}"
        out_dir = os.path.join(args.output_dir, run_id)
        os.makedirs(out_dir, exist_ok=True)
        history_path = os.path.join(out_dir, "history.json")
        if os.path.exists(history_path) and not args.overwrite:
            print(f"\n[{i}/{total}] {run_id}  (skip: exists)")
            with open(history_path) as f:
                rec = json.load(f)
            summary.append({
                "run_id": run_id, **rec["config"], **rec["final"],
                "elapsed_sec": rec.get("elapsed_sec"),
            })
            continue

        print(f"\n[{i}/{total}] {run_id}  noise={noise} width={width} opt={opt} seed={seed}")
        rec = run_one(
            width=width, optimizer_type=opt, noise=noise, seed=seed,
            n_train=args.n_train, epochs=epochs,
            data_dir=args.data_dir, device=device,
            log_interval=max(1, epochs // 5),
        )
        with open(history_path, "w") as f:
            json.dump(rec, f, indent=2)
        summary.append({
            "run_id": run_id, **rec["config"], **rec["final"],
            "elapsed_sec": rec["elapsed_sec"],
        })
        with open(os.path.join(args.output_dir, "summary.json"), "w") as f:
            json.dump(summary, f, indent=2)
        print(f"  final train_acc={rec['final']['train_acc']:.2f}% "
              f"test_acc={rec['final']['test_acc']:.2f}% "
              f"({rec['elapsed_sec']:.0f}s)")

    print("\nDone. Summary:")
    print(f"{'run_id':<28} {'p/n':>7} {'train':>8} {'test':>8}")
    for s in summary:
        print(f"{s['run_id']:<28} {s['p_over_n']:>7.3f} "
              f"{s['train_acc']:>7.2f}% {s['test_acc']:>7.2f}%")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--widths", default="4,8,16,32,48,64")
    parser.add_argument("--optimizers", default="sgd,adam")
    parser.add_argument("--noises", default="0.0,0.15")
    parser.add_argument("--seeds", default="42,7")
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--n-train", type=int, default=4000)
    parser.add_argument("--data-dir", default="./data")
    parser.add_argument("--output-dir", default="./results/personC_optimizer")
    parser.add_argument("--pilot", action="store_true",
                        help="Reduced 6-run sweep at 100 epochs for smoke test")
    parser.add_argument("--overwrite", action="store_true",
                        help="Re-run runs even if history.json already exists")
    args = parser.parse_args()
    t0 = time.time()
    run(args)
    print(f"\nTotal wall time: {(time.time() - t0) / 60:.1f} min")


if __name__ == "__main__":
    main()
