#!/usr/bin/env bash
# Zero-shot OOD evaluation of frozen checkpoints (escalation study)
#
# Requires setup/patch_eval_episodes.py (EVAL_EPISODES, default here 20).
# Cluster overrides and LOG_ROOT work as in train.sbatch.
#
# SEEDS roster matches train.sbatch, so the same array index refers to the
# same run in both scripts. Full range 0-26 (9 seeds x 3 levels).
#   sbatch --gres=gpu:L40S:1 --export=ALL,SCHEME=dali  --array=0-26 slurm/reeval.sbatch
#   sbatch --gres=gpu:L40S:1 --export=ALL,SCHEME=crssm --array=0-8  slurm/reeval.sbatch
#
#SBATCH --job-name=dali_eval
#SBATCH --partition=general
#SBATCH --gres=gpu:1
#SBATCH --time=08:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=24G
#SBATCH --output=/home/%u/slurm_logs/%x_%A_%a.out
#SBATCH --error=/home/%u/slurm_logs/%x_%A_%a.err

# same seed-major mapping as train.sbatch
set -uo pipefail
mkdir -p "$HOME/slurm_logs"
CODE_ROOT="${CODE_ROOT:-$HOME}"
cd "$CODE_ROOT/DALI"

SEEDS=(42 1337 13 71 1994 1997 908 2102 3)   # must match train.sbatch
idx=${SLURM_ARRAY_TASK_ID:-0}
SEED=${SEEDS[$(( idx / 3 ))]}
VAR=$(( 2 + idx % 3 ))

declare -A SCHEMES=( [dali]="enc_obs_dec_obs_ctxencoder_transformer" [crssm]="enc_obs_ctx_dec_obs_ctx" )
SCH="${SCHEME:-dali}"
TAG="${SCHEMES[$SCH]}"
GROUP="carl_dmc_finger_${VAR}var_double_box_${TAG}_normalized"
LOG_ROOT="${LOG_ROOT:-/data/user_data/$USER/dali_logs/logs_dali_crossmodal}"
LOGDIR="$LOG_ROOT/${GROUP}/${SEED}"
[ -f "$LOGDIR/checkpoint.ckpt" ] || { echo "no checkpoint at $LOGDIR -- skip"; exit 0; }

#  non-destructive eval: back up any existing eval.jsonl, restore it if this eval fails
EPS="${EVAL_EPISODES:-20}"
echo "==> eval SCHEME=$SCH VAR=$VAR seed=$SEED eps=$EPS host=$(hostname) $(date)"
[ -f "$LOGDIR/eval.jsonl" ] && mv "$LOGDIR/eval.jsonl" "$LOGDIR/eval.jsonl.bak"
if EVAL_EPISODES="$EPS" uv run -m contextual_mbrl.dreamer.eval \
       --logdir "$LOGDIR" --envs.amount 1 --jax.prealloc False --jax.platform gpu; then
    rm -f "$LOGDIR/eval.jsonl.bak"
    echo "==> done SCHEME=$SCH VAR=$VAR seed=$SEED $(date)"
else
    echo "==> FAILED, restoring backup"
    [ -f "$LOGDIR/eval.jsonl.bak" ] && mv "$LOGDIR/eval.jsonl.bak" "$LOGDIR/eval.jsonl"
    exit 1
fi
