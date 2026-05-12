"""Medium verification: 3 runs to reproduce Yusheng key data points.
k=0.0625/s42 (under-param sanity), k=0.5/s42 (valley bottom), k=2.0/s42 (recovery plateau).
n=4000, 2000 epochs.
"""
import sys
import os
import json
import time
from pathlib import Path

# Make src/ importable when run from arbitrary cwd.
REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

import torch
from src.experiments.exp_dd_recovery import train_one_run

torch.backends.cudnn.benchmark = True
OUT = Path("/home/cc/shufeng/elen6699/runs/yusheng-replicate")
OUT.mkdir(parents=True, exist_ok=True)

JOBS = [
    dict(k=0.0625, n=4000, ep=2000, seed=42),
    dict(k=0.5,    n=4000, ep=2000, seed=42),
    dict(k=2.0,    n=4000, ep=2000, seed=42),
]

t_all = time.time()
for i, j in enumerate(JOBS, 1):
    k = j["k"]
    n = j["n"]
    ep = j["ep"]
    seed = j["seed"]
    name = "k{:g}_n{}_ep{}_s{}".format(k, n, ep, seed)
    d = OUT / name
    print("\n[{}/{}] RUN {}".format(i, len(JOBS), name), flush=True)
    t0 = time.time()
    train_one_run(
        k=k, n_train=n, noise_rate=0.15,
        epochs=ep, seed=seed,
        data_dir="/home/cc/shufeng/elen6699/data",
        out_dir=str(d),
        augment=True, batch_size=256, lr=1e-4,
        device="cuda", eval_every=100, log_every=400,
    )
    print("[done] {}: {:.1f} min".format(name, (time.time() - t0) / 60), flush=True)

print("\nALL DONE: {:.1f} min".format((time.time() - t_all) / 60), flush=True)
