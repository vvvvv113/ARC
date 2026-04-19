#!/bin/bash
#SBATCH --job-name=codeit_10iter
#SBATCH --account=torch_pr_932_general
#SBATCH --partition=l40s_public
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=06:00:00
#SBATCH --output=/scratch/cy2941/codeit_logs/10iter_%j.out
#SBATCH --error=/scratch/cy2941/codeit_logs/10iter_%j.err

set -euo pipefail

mkdir -p /scratch/cy2941/codeit_logs
mkdir -p /scratch/cy2941/codeit_outputs

cd /home/cy2941/codeit
source setup_env.sh

nvidia-smi
echo "--- starting codeit 10-iter run ---"

python run/run_codeit.py \
    hydra.run.dir=/scratch/cy2941/codeit_outputs/10iter_${SLURM_JOB_ID} \
    exit.n_iters=10
