"""Cloud patch runner: 6 main valley + 5 nslice tail (incl. k=1.0/s7 baseline)."""
import json
import time
from pathlib import Path

import torch

from src.experiments.exp_dd_recovery import train_one_run

torch.backends.cudnn.benchmark = True
OUT = Path("results/dd_recovery_5090_focused")

jobs = []

# Main valley densification (n=4000)
for k in [0.3, 0.4, 0.6]:
    for s in [42, 7]:
        jobs.append(dict(phase="main", k=k, n=4000, ep=2000, seed=s))

# Nslice tail (n=8000): k=1.0/s7 baseline + k=1.5/2.0 x 2 seeds
jobs.append(dict(phase="nslice", k=1.0, n=8000, ep=1500, seed=7))
for k in [1.5, 2.0]:
    for s in [42, 7]:
        jobs.append(dict(phase="nslice", k=k, n=8000, ep=1500, seed=s))


def is_complete(p: Path) -> bool:
    if not p.exists():
        return False
    try:
        r = json.loads(p.read_text())
        return r.get("final_test_acc") is not None and bool(r.get("history"))
    except Exception:
        return False


t_all = time.time()
for i, j in enumerate(jobs, 1):
    name = f"k{j['k']:g}_n{j['n']}_ep{j['ep']}_s{j['seed']}"
    d = OUT / j["phase"] / name
    rp = d / "results.json"
    if is_complete(rp):
        print(f"[{i}/{len(jobs)}] SKIP complete {j['phase']}/{name}", flush=True)
        continue
    if rp.exists():
        print(f"[{i}/{len(jobs)}] RETRY (incomplete) {j['phase']}/{name}", flush=True)

    print(f"\n[{i}/{len(jobs)}] RUN {j['phase']}/{name}", flush=True)
    t0 = time.time()
    train_one_run(
        k=j["k"],
        n_train=j["n"],
        noise_rate=0.15,
        epochs=j["ep"],
        seed=j["seed"],
        data_dir="./data",
        out_dir=str(d),
        augment=True,
        batch_size=256,
        lr=1e-4,
        device="cuda",
        eval_every=100,   # explicit, matches legacy main/ runs
        log_every=200,
    )
    print(f"[done] {name}: {(time.time()-t0)/3600:.2f}h", flush=True)

print(f"\nALL TRAINING DONE: {(time.time()-t_all)/3600:.2f}h", flush=True)
