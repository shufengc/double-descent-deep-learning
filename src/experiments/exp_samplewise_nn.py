"""
Sample-wise NN double descent on fractional-k ResNet.

Output dir: results/samplewise_nn/
  k{K}_n{N}_ep{EP}_s{S}/results.json    (one per run)
  summary.json                           (slim aggregate)

Usage:
  python -m src.experiments.exp_samplewise_nn --epochs 1500 --device cuda

Smoke test (single fast run):
  python -m src.experiments.exp_samplewise_nn --smoke --device cuda
"""
import argparse
import json
import os
import sys
import time
from itertools import product
from pathlib import Path

# Path setup so this works whether invoked as module or script
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from src.experiments.exp_dd_recovery import train_one_run


# Default sweep grid (4 k near and around the peak, plus 1.0 for the flat tail).
DEFAULT_KS = [0.0625, 0.125, 0.25, 0.5, 1.0]
DEFAULT_NS = [1000, 2000]
DEFAULT_SEEDS = [42, 7]


def run(args):
    out_root = Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    if args.smoke:
        ks = [0.1875]
        ns = [2000]
        seeds = [42]
        epochs = 200
    else:
        ks = [float(x) for x in args.ks.split(",")]
        ns = [int(x) for x in args.ns.split(",")]
        seeds = [int(x) for x in args.seeds.split(",")]
        epochs = args.epochs

    grid = list(product(ns, ks, seeds))
    print(f"[samplewise] plan: {len(grid)} runs at epochs={epochs}", flush=True)
    print(f"  ks    = {ks}\n  ns    = {ns}\n  seeds = {seeds}", flush=True)

    all_results = []
    t_total = time.time()
    for i, (n, k, s) in enumerate(grid, 1):
        run_name = f"k{k:g}_n{n}_ep{epochs}_s{s}"
        run_dir = out_root / run_name
        results_path = run_dir / "results.json"
        if results_path.exists() and not args.force:
            print(f"\n[{i}/{len(grid)}] {run_name}  SKIP (cached)", flush=True)
            with open(results_path) as f:
                all_results.append(json.load(f))
            continue

        print(f"\n[{i}/{len(grid)}] {run_name}  running...", flush=True)
        t0 = time.time()
        r = train_one_run(
            k=k, n_train=n, noise_rate=args.noise_rate,
            epochs=epochs, seed=s,
            data_dir=args.data_dir, out_dir=str(run_dir),
            augment=True, batch_size=args.batch_size, lr=args.lr,
            device=args.device,
            eval_every=args.eval_every, log_every=args.log_every,
        )
        all_results.append(r)
        print(f"  done in {(time.time() - t0) / 60:.1f} min  "
              f"(running total {(time.time() - t_total) / 60:.1f} min)",
              flush=True)

        # Update slim summary after each run so partial progress is preserved
        slim = [dict(k=r["config"]["k"], n=r["config"]["n_train"],
                     seed=r["config"]["seed"], params=r["params"],
                     best_test_acc=r["best_test_acc"],
                     final_test_acc=r["final_test_acc"],
                     final_train_acc=r["final_train_acc"],
                     effective_rank=r["effective_rank"])
                for r in all_results]
        with open(out_root / "summary.json", "w") as f:
            json.dump(slim, f, indent=2)

    print(f"\n[samplewise] done. total wall {(time.time() - t_total) / 60:.1f} min")
    print(f"  summary at {out_root / 'summary.json'}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ks", default=",".join(str(x) for x in DEFAULT_KS))
    p.add_argument("--ns", default=",".join(str(x) for x in DEFAULT_NS))
    p.add_argument("--seeds", default=",".join(str(x) for x in DEFAULT_SEEDS))
    p.add_argument("--epochs", type=int, default=1500)
    p.add_argument("--noise-rate", type=float, default=0.15)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--eval-every", type=int, default=50)
    p.add_argument("--log-every", type=int, default=100)
    p.add_argument("--data-dir", default="./data")
    p.add_argument("--out-root", default="./results/samplewise_nn")
    p.add_argument("--device", default="cuda")
    p.add_argument("--smoke", action="store_true",
                   help="single fast run for sanity checking")
    p.add_argument("--force", action="store_true",
                   help="re-run even if results.json exists")
    args = p.parse_args()
    run(args)


if __name__ == "__main__":
    main()
