#!/usr/bin/env bash
# Lattice study: DALI on the 3-feature Finger task with one feature pinned.
#
# 8 cells x 3 seeds = 24 runs. In each cell one of {gravity, actuator_strength,
# joint_damping} is fixed at low/med/high for every training context while the
# other two vary uniformly over their training ranges. joint_damping = 1.0 is
# omitted: it is the default value, i.e. the existing 2-var escalation runs.
# Each task trains, then evaluates the final checkpoint on the standard 3-var
# OOD grid (97 contexts x EVAL_EPISODES episodes).
#
# Requires setup/add_lattice_context.py and setup/patch_eval_episodes.py.
#
# The #SBATCH defaults below are for PSC Bridges-2 (L40S nodes). Usage:
#   mkdir -p ~/slurm_logs
#   sbatch -A <account> --export=ALL,LOG_ROOT=/ocean/projects/<account>/$USER/dali_logs \
#          --array=0-23 slurm/train_lattice.sbatch
#
#SBATCH --job-name=dali_lattice
#SBATCH --partition=GPU-shared
#SBATCH --gpus=l40s-48:1
#SBATCH --time=2-00:00:00
#SBATCH --cpus-per-task=8
#SBATCH --output=/jet/home/%u/slurm_logs/%x_%A_%a.out
#SBATCH --error=/jet/home/%u/slurm_logs/%x_%A_%a.err

set -euo pipefail
: "${LOG_ROOT:?set LOG_ROOT to a large-storage directory, e.g. /ocean/projects/<account>/\$USER/dali_logs}"
CODE_ROOT="${CODE_ROOT:-$HOME}"
cd "$CODE_ROOT/DALI"

export JAX_COMPILATION_CACHE_DIR="${JAX_COMPILATION_CACHE_DIR:-$LOG_ROOT/../jax_cache}"
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.90

# array -> (cell, seed); CELL-MAJOR: a cell's seeds run consecutively.
# Cell strings must match analysis.py's LATTICE list.
CELLS=(0-4.9 0-9.8 0-14.7 1-0.5 1-1.0 1-1.5 2-0.5 2-1.5)
SEEDS=(42 1337 13)                      # first three seeds of the escalation study
idx=${SLURM_ARRAY_TASK_ID:-0}
CELL=${CELLS[$(( idx / 3 ))]}
SEED=${SEEDS[$(( idx % 3 ))]}

STEPS=${STEPS_OVERRIDE:-1500000}
EPS="${EVAL_EPISODES:-20}"
TAG="enc_obs_dec_obs_ctxencoder_transformer"   # DALI-S-chi
TASK="carl_dmc_finger_3var"
CONTEXT="lattice_${CELL}"
GROUP_NAME="${TASK}_${CONTEXT}_${TAG}_normalized"
LOGDIR="$LOG_ROOT/${GROUP_NAME}/${SEED}"

# same fresh/resume logic as train.sbatch
if [ "${RESUME:-}" = "1" ]; then
    echo "==> RESUME requested: keeping $LOGDIR"
elif [ "${RESUME:-}" = "0" ] || [ "${SLURM_RESTART_COUNT:-0}" -eq 0 ]; then
    echo "==> FRESH start: removing $LOGDIR"; rm -rf "$LOGDIR"
else
    echo "==> AUTO-REQUEUE (#${SLURM_RESTART_COUNT}): resuming $LOGDIR"
fi
mkdir -p "$(dirname "$LOGDIR")"

echo "==> cell=$CELL seed=$SEED steps=$STEPS host=$(hostname) $(date)"
uv run python -c "import jax; print('JAX devices:', jax.devices())"

uv run -m contextual_mbrl.dreamer.train --configs carl "$TAG" --jax.prealloc False \
    --task "$TASK" --env.carl.context "$CONTEXT" --seed "$SEED" \
    --ctx_encoder.crossmodal True \
    --logdir "$LOGDIR" --wandb.project '' --wandb.group "$GROUP_NAME" \
    --run.steps "$STEPS" --run.log_every 1000 --jax.platform gpu

# The eval grid is built from the task (3 features), not the training context,
# so lattice cells face the same 97-context exam as the escalation 3-var runs.
echo "==> eval cell=$CELL seed=$SEED eps=$EPS $(date)"
EVAL_EPISODES="$EPS" uv run -m contextual_mbrl.dreamer.eval \
    --logdir "$LOGDIR" --envs.amount 1 --jax.prealloc False --jax.platform gpu

echo "==> done cell=$CELL seed=$SEED $(date)"
