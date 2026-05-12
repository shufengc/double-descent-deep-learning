"""Full reproduction of Yusheng's 11-run densification + nslice campaign.

Hyperparameters: noise=0.15, augment=True, batch_size=256, lr=1e-4,
                 eval_every=100, matching Yusheng's settings exactly.
"""
import sys
import json
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

import torch
from src.experiments.exp_dd_recovery import train_one_run

torch.backends.cudnn.benchmark = True
OUT = Path("/home/cc/shufeng/elen6699/runs/yusheng-replicate-full")
OUT.mkdir(parents=True, exist_ok=True)

JOBS = []

# Main valley densification (n=4000, ep=2000)
for k in [0.3, 0.4, 0.6]:
    for s in [42, 7]:
        JOBS.append(dict(phase="main", k=k, n=4000, ep=2000, seed=s))

# Nslice tail (n=8000, ep=1500)
JOBS.append(dict(phase="nslice", k=1.0, n=8000, ep=1500, seed=7))
for k in [1.5, 2.0]:
    for s in [42, 7]:
        JOBS.append(dict(phase="nslice", k=k, n=8000, ep=1500, seed=s))


def is_complete(p: Path) -> bool:
    if not p.exists():
        return False
    try:
        r = json.loads(p.read_text())
        return r.get("final_test_acc") is not None and bool(r.get("history"))
    except Exception:
        return False


t_all = time.time()
for i, j in enumerate(JOBS, 1):
    name = "k{:g}_n{}_ep{}_s{}".format(j["k"], j["n"], j["ep"], j["seed"])
    d = OUT / j["phase"] / name
    rp = d / "results.json"
    if is_complete(rp):
        print("[{}/{}] SKIP complete {}/{}".format(i, len(JOBS), j["phase"], name), flush=True)
        continue
    print("\n[{}/{}] RUN {}/{}".format(i, len(JOBS), j["phase"], name), flush=True)
    t0 = time.time()
    train_one_run(
        k=j["k"],
        n_train=j["n"],
        noise_rate=0.15,
        epochs=j["ep"],
        seed=j["seed"],
        data_dir="/home/cc/shufeng/elen6699/data",
        out_dir=str(d),
        augment=True,
        batch_size=256,
        lr=1e-4,
        device="cuda",
        eval_every=100,
        log_every=400,
    )
    print("[done] {}: {:.1f} min".format(name, (time.time() - t0) / 60), flush=True)

print("\nALL DONE: {:.2f} h".format((time.time() - t_all) / 3600), flush=True)
