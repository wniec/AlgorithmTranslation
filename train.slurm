#!/bin/bash
# Train the PSO->CMAES state translator with PPO on an HPC GPU node.

#SBATCH --job-name=cat-ppo-pso-cmaes
#SBATCH --partition=<ADD>              # queue/partition, e.g. plgrid-gpu-a100 / gpu
#SBATCH --account=<ADD>                # grant / project account to charge
#SBATCH --nodes=1
#SBATCH --ntasks=1                     # one training process; parallelism is via --n-workers, not MPI
#SBATCH --cpus-per-task=16              # 6 rollout workers + main + headroom; keep >= (--n-workers + 1)
#SBATCH --mem=16G
#SBATCH --time=48:00:00                # 1000 updates; adjust to your measured s/upd (watch the tqdm bar)
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
#SBATCH --mail-type=END,FAIL           # optional: drop these two lines if your cluster has no mail
#SBATCH --mail-user=<ADD>

set -euo pipefail

# --- run from the project directory ------------------------------------- #
cd "${SLURM_SUBMIT_DIR:-$(pwd)}"
mkdir -p logs models

# --- environment -------------------------------------------------------- #

# CRITICAL: pin BLAS to a single thread. The rollout workers run their own
# CMA-ES eigendecompositions and must stay single-threaded, or 6 workers x
# (cpus) threads oversubscribes the node. SLURM may export OMP_NUM_THREADS
# from --cpus-per-task, so set these explicitly here rather than relying on
# the in-code setdefault.
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1

# --- Weights & Biases --------------------------------------------------- #
# Compute nodes usually have no internet: either provide a key and let it sync,
# or run offline and `wandb sync` the run dir from a login node afterwards.
export WANDB_API_KEY=<ADD>             # or: export WANDB_MODE=offline  (then drop this line)
export WANDB_PROJECT=algorithm-translation
export WANDB_ENTITY=<ADD>                    # or: unset WANDB_ENTITY

# --- train -------------------------------------------------------------- #
# Currently-suggested configuration: single PSO->CMAES switch, AOCC reward
# (default), relative mode, reward normalization ON, no ELA, exploration kept
# alive with a small entropy bonus. --n-workers matches cpus-per-task - 2.
uv run train --wandb --reward-mode relative PSO CMAES \
  --dims 5 --fe-multiplier 1000 --switch-cdb 9 --updates 1000 \
  --fixed-order --n-workers 14 --reward-seeds 4 --ent-coef 1e-3 \
  --seed "${SLURM_ARRAY_TASK_ID:-42}"

echo "done: ${SLURM_JOB_ID:-local}"
