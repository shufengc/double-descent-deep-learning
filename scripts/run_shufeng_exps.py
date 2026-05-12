"""Shufeng's experiments: Exp 5 (noise multiseed RFF), Exp 6 (B = bias-variance),
Exp 7 (C = epoch-wise SGD ResNet), Exp 8 (A = EMC).

Runs the four in sequence. Exp 5 + 6 are RFF-only and fast.
Exp 7 (SGD ResNet) and Exp 8 (EMC) are the slow ones.
"""

import sys
import os
import subprocess
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

OUT = Path("/home/cc/shufeng/elen6699/runs/shufeng-exps")
OUT.mkdir(parents=True, exist_ok=True)
DATA = "/home/cc/shufeng/elen6699/data"

env = os.environ.copy()
env["PYTHONPATH"] = str(REPO)

# Run them as ONE subprocess invocation per Yusheng's script signature so
# the per-experiment plumbing stays consistent with how it was run originally.
cmd = [
    sys.executable, "-m", "src.experiments.shufeng_experiments",
    "--experiments", "noise_multiseed,B,C,A",
    "--data-dir", DATA,
    "--output-dir", str(OUT),
]
print("running:", " ".join(cmd), flush=True)
subprocess.check_call(cmd, env=env)
print("\n[done] Shufeng Exp 5/6/7/8 complete", flush=True)
