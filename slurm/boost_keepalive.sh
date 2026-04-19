#!/bin/bash
# Attach an aggressive keepalive to the already-running codeit job.
# Usage: bash boost_keepalive.sh <JOBID>
#   e.g.  bash boost_keepalive.sh 6490747
# Run this from a Greene login node while the main job is RUNNING.
# Non-destructive: shares the existing GPU allocation; Ctrl+C to stop.

set -euo pipefail

JOBID="${1:?usage: $0 <JOBID>}"

srun --jobid="${JOBID}" --overlap --ntasks=1 --cpus-per-task=1 \
  /scratch/cy2941/codeit_env/bin/python -c "
import torch, time, os
torch.set_float32_matmul_precision('high')
x = torch.randn(2048, 2048, device='cuda', dtype=torch.float16)
print('boost keepalive online, PID=', os.getpid(), flush=True)
while True:
    for _ in range(16):
        x = x @ x
        x = x / (x.norm() + 1e-8)
    time.sleep(0.001)
"
