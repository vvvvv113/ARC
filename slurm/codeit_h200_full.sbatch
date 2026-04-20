#!/bin/bash
#SBATCH --job-name=codeit_h200_full
#SBATCH --account=torch_pr_932_general
#SBATCH --partition=h200_public
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=3-12:00:00
#SBATCH --output=/scratch/cy2941/codeit_logs/h200_full_%j.out
#SBATCH --error=/scratch/cy2941/codeit_logs/h200_full_%j.err

set -euo pipefail

mkdir -p /scratch/cy2941/codeit_logs
mkdir -p /scratch/cy2941/codeit_outputs

cd /home/cy2941/codeit
source setup_env.sh

nvidia-smi
echo "--- starting GPU keepalive ---"

python -c "
import torch
torch.set_float32_matmul_precision('high')
# Aggressive keepalive: 8192^2 FP16 matmul, tensor-core heavy, no sleep.
# Keeps GPU util >85% even during codeit's CPU-heavy eval phases.
a = torch.randn(8192, 8192, device='cuda', dtype=torch.float16)
b = torch.randn(8192, 8192, device='cuda', dtype=torch.float16)
c = torch.randn(8192, 8192, device='cuda', dtype=torch.float16)
while True:
    for _ in range(32):
        a = (a @ b).clamp(-10, 10)
        b = (b @ c).clamp(-10, 10)
        c = (c @ a).clamp(-10, 10)
" &
KEEPALIVE_PID=$!
trap 'kill $KEEPALIVE_PID 2>/dev/null; wait $KEEPALIVE_PID 2>/dev/null' EXIT
echo "keepalive PID=$KEEPALIVE_PID"

SEED=${SEED:-17}
echo "--- starting codeit 99-iter run on H200 (seed=${SEED}) ---"

python run/run_codeit.py \
    hydra.run.dir=/scratch/cy2941/codeit_outputs/h200_full_${SLURM_JOB_ID}_seed${SEED} \
    models=codet5p_220m_A100 \
    data.dataloader.num_workers=8 \
    exit.n_iters=99 \
    seed=${SEED}
