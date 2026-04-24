#!/bin/bash
#SBATCH --job-name=method2_smoke
#SBATCH --account=torch_pr_932_general
#SBATCH --partition=l40s_public
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --output=/scratch/cy2941/codeit_logs/method2_smoke_%j.out
#SBATCH --error=/scratch/cy2941/codeit_logs/method2_smoke_%j.err

set -euo pipefail

mkdir -p /scratch/cy2941/codeit_logs
mkdir -p /scratch/cy2941/codeit_outputs

cd /home/cy2941/codeit-3
source setup_env.sh

nvidia-smi
echo "--- method2 smoke test (human_lambda=${LAMBDA:-0.5}) ---"

LAMBDA=${LAMBDA:-0.5}

python run/run_codeit.py \
    hydra.run.dir=/scratch/cy2941/codeit_outputs/method2_smoke_${SLURM_JOB_ID}_lambda${LAMBDA} \
    exit.n_iters=2 \
    exit.n_policy_samples=4 \
    data.n_val=50 \
    data.n_train=50 \
    replay_buffer.human_lambda=${LAMBDA}
