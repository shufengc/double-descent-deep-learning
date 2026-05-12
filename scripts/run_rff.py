"""RFF reproductions: Exp 1 (model-wise) + Exp 2 (sample-wise) on MNIST.

Calls the comprehensive_dd module. Pure CPU/GPU work; small total wallclock
(under ~30 min depending on grid).
"""

import sys
import os
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

# We invoke via -m on the comprehensive_dd module, but want our own out dir.
# Easiest: shell out via subprocess to keep arg parsing clean.
import subprocess

OUT = Path("/home/cc/shufeng/elen6699/runs/rff")
OUT.mkdir(parents=True, exist_ok=True)
DATA = "/home/cc/shufeng/elen6699/data"

env = os.environ.copy()
env["PYTHONPATH"] = str(REPO)

cmd = [
    sys.executable, "-m", "src.experiments.comprehensive_dd",
    "--experiments", "1,2",
    "--data-dir", DATA,
    "--output-dir", str(OUT),
]
print("running:", " ".join(cmd), flush=True)
subprocess.check_call(cmd, env=env)
print("\n[done] RFF Exp 1+2 complete", flush=True)
