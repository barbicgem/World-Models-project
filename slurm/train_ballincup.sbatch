#!/usr/bin/env bash
# Ball-in-Cup reproduction (pipeline validation)
#
# Reproduces the DALI paper's published zero-shot results before any Finger
# experiment is trusted: DALI-S-chi on CARL DMC Ball-in-Cup (gravity +
# string_length), featurized observations, 200K steps, 10 seeds.
# Expected: Interpolation ~0.95, Extrapolation ~0.38 (Roeder et al. 2025).
#
# One-time setup:  mkdir -p ~/slurm_logs
#   pipeline test (1 seed):  sbatch --gres=gpu:L40S:1 --array=0   slurm/train_ballincup.sbatch
#   full paper run:          sbatch --gres=gpu:L40S:1 --array=0-9 slurm/train_ballincup.sbatch
#
#SBATCH --job-name=dali_bic
#SBATCH --partition=general
#SBATCH --gres=gpu:1
#SBATCH --time=08:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --output=/home/%u/slurm_logs/%x_%A_%a.out
#SBATCH --error=/home/%u/slurm_logs/%x_%A_%a.err

set -euo pipefail
mkdir -p "$HOME/slurm_logs"
LOG_ROOT="${LOG_ROOT:-/data/user_data/$USER/dali_logs/logs_dali_crossmodal}"
export JAX_COMPILATION_CACHE_DIR="${JAX_COMPILATION_CACHE_DIR:-$LOG_ROOT/../jax_cache}"
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.90
CODE_ROOT="${CODE_ROOT:-$HOME}"
cd "$CODE_ROOT/DALI"

# DALI-S-chi: featurized obs, shallow transformer context encoder,
# cross-modal regularization; double_box = both context features vary jointly.
TASK="carl_dmc_ball_in_cup"
TAG="enc_obs_dec_obs_ctxencoder_transformer"
CONTEXT="double_box"
STEPS=200000

SEEDS=(0 42 1337 13 71 1994 1997 908 2102 3)   # the paper's 10-seed protocol
SEED=${SEEDS[${SLURM_ARRAY_TASK_ID:-0}]}

GROUP_NAME="${TASK}_${CONTEXT}_${TAG}_normalized"
LOGDIR="$LOG_ROOT/${GROUP_NAME}/${SEED}"
mkdir -p "$(dirname "$LOGDIR")"

echo "==> BiC seed=$SEED  host=$(hostname)  $(date)"
uv run python -c "import jax; print('JAX devices:', jax.devices())"

# train, then immediately evaluate the final checkpoint (validation runs are
# short, so chaining is fine here; the Finger study evals via reeval.sbatch).
uv run -m contextual_mbrl.dreamer.train --configs carl "$TAG" --jax.prealloc False \
    --task "$TASK" --env.carl.context "$CONTEXT" --seed "$SEED" \
    --ctx_encoder.crossmodal True \
    --logdir "$LOGDIR" --wandb.project '' --wandb.group "$GROUP_NAME" \
    --run.steps "$STEPS" --jax.platform gpu \
  && uv run -m contextual_mbrl.dreamer.eval --logdir "$LOGDIR"

echo "==> done seed=$SEED  $(date)"
