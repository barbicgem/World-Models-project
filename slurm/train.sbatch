#!/usr/bin/env bash
# Unified training launcher: DALI or cRSSM x {2,3,4}-var x seeds
#
# The #SBATCH defaults below are for CMU Babel. On another cluster, override
# at submit time (command-line flags win over #SBATCH lines), e.g.
#   sbatch -A <account> -p <gpu-partition> --gpus=<type>:1 ...
# and set LOG_ROOT to a large-storage directory:
#   sbatch --export=ALL,LOG_ROOT=/path/to/dali_logs ...
#
#   DALI, 8 seeds x 3 levels:
#     sbatch --gres=gpu:L40S:1 --array=0-23 slurm/train.sbatch
#   cRSSM (job name reflecting the scheme is set at submit time --
#     #SBATCH lines are parsed before the shell runs, so no variables there):
#     sbatch --gres=gpu:L40S:1 --job-name=crssm_train \
#            --export=ALL,SCHEME=crssm --array=0-8 slurm/train.sbatch
#
#SBATCH --job-name=dali_train
#SBATCH --partition=general
#SBATCH --gres=gpu:1
#SBATCH --time=2-00:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --output=/home/%u/slurm_logs/%x_%A_%a.out
#SBATCH --error=/home/%u/slurm_logs/%x_%A_%a.err

# Environment
set -euo pipefail
mkdir -p "$HOME/slurm_logs"
CODE_ROOT="${CODE_ROOT:-$HOME}"
cd "$CODE_ROOT/DALI"

LOG_ROOT="${LOG_ROOT:-/data/user_data/$USER/dali_logs/logs_dali_crossmodal}"
export JAX_COMPILATION_CACHE_DIR="${JAX_COMPILATION_CACHE_DIR:-$LOG_ROOT/../jax_cache}"
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.90

# array -> (seed, var) mapping
# SEED-MAJOR: a seed's three variable levels run consecutively

SEEDS=(42 1337 13 71 1994 1997 908 2102 3)
idx=${SLURM_ARRAY_TASK_ID:-0}
SEED=${SEEDS[$(( idx / 3 ))]}
VAR=$(( 2 + idx % 3 ))

STEPS=${STEPS_OVERRIDE:-1500000}   # default = 1500000, override: sbatch --export=ALL,STEPS_OVERRIDE=...

# scheme -> config tag; builds the run's group name and logdir
declare -A SCHEMES=( [dali]="enc_obs_dec_obs_ctxencoder_transformer"
                     [crssm]="enc_obs_ctx_dec_obs_ctx" )
SCH="${SCHEME:-dali}"
TAG="${SCHEMES[$SCH]}"
TASK="carl_dmc_finger_${VAR}var"
CONTEXT="double_box"
GROUP_NAME="${TASK}_${CONTEXT}_${TAG}_normalized"
LOGDIR="$LOG_ROOT/${GROUP_NAME}/${SEED}"

# resume logic
# RESUME=1 -> keep logdir. RESUME=0 or SLURM_RESTART_COUNT==0 -> rm -rf
# logdir (guarantees fresh init). Otherwise auto-requeue -> resume own checkpoint.

if [ "${RESUME:-}" = "1" ]; then
    echo "==> RESUME requested: keeping $LOGDIR"
elif [ "${RESUME:-}" = "0" ] || [ "${SLURM_RESTART_COUNT:-0}" -eq 0 ]; then
    echo "==> FRESH start: removing $LOGDIR"; rm -rf "$LOGDIR"
else
    echo "==> AUTO-REQUEUE (#${SLURM_RESTART_COUNT}): resuming $LOGDIR"
fi
mkdir -p "$(dirname "$LOGDIR")"

echo "==> VAR=$VAR  seed=$SEED  steps=$STEPS  host=$(hostname)  $(date)"
uv run python -c "import jax; print('JAX devices:', jax.devices())"

# checkpoint archiver (background)
# snapshots checkpoint.ckpt into logdir/archive/ every ARCHIVE_STEP env steps,
# enabling OOD-vs-training-steps analysis later. Disable: ARCHIVE_STEP=0
ARCHIVE_STEP="${ARCHIVE_STEP:-100000}"
if [ "$ARCHIVE_STEP" -gt 0 ]; then
  mkdir -p "$LOGDIR/archive"
  (
    last=-1
    while true; do
      sleep 120
      [ -f "$LOGDIR/metrics.jsonl" ] || continue
      step=$(tail -1 "$LOGDIR/metrics.jsonl" 2>/dev/null | python3 -c 'import sys,json
try: print(int(json.loads(sys.stdin.readline())["step"]))
except Exception: print(0)' 2>/dev/null || echo 0)
      b=$(( step / ARCHIVE_STEP ))
      if [ "$b" -gt "$last" ] && [ -f "$LOGDIR/checkpoint.ckpt" ]; then
        cp -f "$LOGDIR/checkpoint.ckpt" "$LOGDIR/archive/ckpt_${step}.ckpt"
        echo "==> archived checkpoint at step ${step}"
        last=$b
      fi
    done
  ) &
  ARCH_PID=$!
  trap 'kill $ARCH_PID 2>/dev/null' EXIT
fi

# training command. DALI additionally trains the cross-modal context encoder;
# cRSSM has no context encoder, so the flag must not be passed.
EXTRA=""
[ "$SCH" = "dali" ] && EXTRA="--ctx_encoder.crossmodal True"

uv run -m contextual_mbrl.dreamer.train --configs carl "$TAG" --jax.prealloc False \
    --task "$TASK" --env.carl.context "$CONTEXT" --seed "$SEED" \
    $EXTRA \
    --logdir "$LOGDIR" --wandb.project '' --wandb.group "$GROUP_NAME" \
    --run.steps "$STEPS" --run.log_every 1000 --jax.platform gpu

echo "==> done SCHEME=$SCH VAR=$VAR seed=$SEED  $(date)"
