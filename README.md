# DALI multi-factor dynamics generalization study

Does a context-conditioned world model generalize worse as more dynamics
variables vary jointly? We test DALI (Röder et al., NeurIPS 2025), a DreamerV3
agent with a self-supervised context encoder, on CARL DMC Finger-spin with a
nested escalation: 2 varying variables (gravity, actuator_strength) → 3
(+joint_damping) → 4 (+friction_tangential). The model and all
hyperparameters are fixed across levels, so differences are attributable to
the added variables. A context-aware oracle (cRSSM, given the true context
values) and model-free baselines (A2C, PPO, optional RND) give reference points.

## Results (DALI-S-χ, 1.5M steps, 8–9 seeds per level)

Zero-shot OOD return per regime, normalized to [0, 1], pooled over contexts × seeds:

| varying variables | Interpolation | Mixed | Extrapolation |
|---|---|---|---|
| 2 | 0.529 | 0.461 | 0.531 |
| 3 | 0.562 | 0.397 | 0.401 |
| 4 | 0.654 | 0.378 | 0.517 |

Interpolation improves as more variables vary during training; Mixed declines
mildly; Extrapolation dips at 3 variables and recovers at 4. Regime buckets mix
single-feature and joint contexts whose composition changes with the number of
variables, so cross-level differences within a regime should be read with that
in mind (`python analysis.py singles` gives the per-feature view).

**Training-pool note.** Upstream DALI's `double_box` training sampler builds
distributions for the first two context features only, which is correct for
every upstream task (all have two) but silently leaves features 3+ at their
defaults. `setup/fix_train_sampler.py` removes that assumption and
`setup/check_training_contexts.py` verifies it. The DALI numbers above come
from runs trained with the fix. cRSSM 3/4-var runs were trained before it and
have not been retrained, so they are not comparable and are omitted here.

## Repo layout

```
setup/        patches applied to upstream DALI, verification checkers, INSTALL.md
slurm/        SLURM launchers: escalation training, evaluation, Ball-in-Cup
              reproduction, lattice study
analysis.py   result tables, training curves, training diagnostics
baselines/    model-free A2C / PPO (+ RND, count-based bonus) on CARL
docs/         PIPELINE.md (start here), PSEUDOCODE.md, ALGORITHM2_ANNOTATED.md
```

## Quickstart

```bash
# 1. install DALI with pinned dependencies and apply the patches
#    (full sequence and pins: setup/INSTALL.md)
git clone https://github.com/frankroeder/DALI.git ~/DALI
cd ~/DALI
S=~/dali-finger-study/setup
uv run python $S/add_finger_tasks.py
uv run python $S/fix_eval_grid.py
uv run python $S/fix_train_sampler.py
uv run python $S/add_lattice_context.py
uv run python $S/patch_eval_episodes.py
PYTHONPATH=~/DALI uv run python $S/check_training_contexts.py

# 2. validate the pipeline against the paper (Ball-in-Cup, 10 seeds)
sbatch --array=0-9 slurm/train_ballincup.sbatch

# 3. escalation study: DALI, 9 seeds x 3 levels; then evaluate
sbatch --array=0-26 slurm/train.sbatch
sbatch --export=ALL,SCHEME=dali --array=0-26 slurm/reeval.sbatch

# 4. tables and figures (wherever the logs are readable)
LOG_ROOT=/path/to/logs python analysis.py tables
LOG_ROOT=/path/to/logs python analysis.py curves dali
```

Launchers default to the clusters they were run on (CMU Babel; PSC Bridges-2
for the lattice study). Override account, partition and GPU type at submit
time, and set `LOG_ROOT`; see the header of each script.

## Lattice study

`slurm/train_lattice.sbatch` trains DALI on the 3-variable task with one
variable pinned at a low, medium or high value while the other two vary,
separating the effect of a variable's *value* from the effect of its
*variation* during training. Evaluation uses the same 3-variable OOD grid as
the escalation study; `python analysis.py lattice` tabulates it.

## Model-free baselines

```bash
cd baselines
pip install -r requirements.txt
python experiment.py --quick                     # smoke test
python experiment.py --algos ppo --explore rnd   # PPO + RND, all configured envs
```

Train/test context splits from the same distribution; reports the
train-vs-held-out generalization gap per environment.

## Credits

Built on [DALI](https://github.com/frankroeder/DALI) (MIT), whose code
the `setup/` patches adapt. Please cite DALI (Röder et al., NeurIPS 2025,
arXiv:2508.20294) and the CARL benchmark (Benjamins et al., 2023) if you use
this code.
