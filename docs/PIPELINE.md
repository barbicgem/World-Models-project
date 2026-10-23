# How the pieces fit together

This document explains the full experimental pipeline: the dependency stack,
what each of our components does, how data flows from training to the final
tables, and the operational details (cluster, seeds, gotchas) needed to
reproduce or extend the study.

## 1. The dependency stack

```
┌─────────────────────────────────────────────────────┐
│  our code (this repo)                               │
│  patches · SLURM launchers · analysis               │
├─────────────────────────────────────────────────────┤
│  DALI  (frankroeder/DALI)                           │
│  contextual_mbrl/  — context encoder, train/eval    │
│  dreamerv3_compat/ — vendored DreamerV3 (JAX)       │
├──────────────────────────┬──────────────────────────┤
│  CARL                    │  JAX + optax             │
│  contextual env wrapper  │  jax[cuda12]==0.4.30     │
├──────────────────────────┤  (sm_80/86/89 GPUs only: │
│  dm_control / MuJoCo     │   A6000, L40S — no H100) │
│  Finger-spin physics     │                          │
└──────────────────────────┴──────────────────────────┘
```

- **MuJoCo / dm_control** provides the deterministic Finger-spin physics:
  9-dim featurized observation, 2-dim torque action, binary reward per step
  (1 while the spinner exceeds a threshold angular velocity; episodes are
  1000 steps, so max return is 1000).
- **CARL** wraps the DMC task as a *contextual* MDP: physics parameters
  (gravity, actuator_strength, joint_damping, friction_tangential) become
  per-episode "context" values sampled from configurable ranges. Note the
  base Finger domain disables gravity; CARL injects it as a force.
- **DALI** is a DreamerV3 world-model agent plus a self-supervised transformer
  context encoder that infers an 8-dim context representation from a K=64-step
  history of (observation, action) pairs. The agent never sees the true
  context values. The oracle baseline (cRSSM) uses the identical DreamerV3
  backbone but receives the ground-truth normalized context concatenated to
  the observation, with no encoder.
- **Our code** never touches the learning algorithm. It (a) registers new
  CARL Finger task variants, (b) extends the evaluation grid and (c) the
  training-context sampler to any number of varying features, (d) adds an
  optional lattice context mode and a configurable eval episode count,
  (e) launches everything on SLURM, and (f) turns the raw logs into tables
  and figures.

## 2. Experimental design in one paragraph

Nested escalation: 2-var {gravity, actuator_strength} ⊂ 3-var {+joint_damping}
⊂ 4-var {+friction_tangential}. Training ranges follow the DALI paper
(gravity [4.9, 14.7], all other factors [0.5, 1.5]); during training every
level's variables vary jointly: each run draws a pool of 100 contexts
uniformly from the training ranges (seeded by `--seed`), and every episode
takes the next context from the pool in shuffled round-robin order. Because levels are
nested and the configuration is fixed (the paper's DALI-S-χ hyperparameters:
λ_cross=1.0, 8-dim context, K=64, DreamerV3-small — no per-level tuning),
any performance change from 2→3→4 is attributable to the added variable.
8–9 seeds per level for DALI, 3 for cRSSM; every (seed, level) cell is an
independent from-scratch training run.

## 3. Training: what one run does

`contextual_mbrl.dreamer.train` (upstream, unmodified):

1. **Fresh initialization.** Every invocation constructs a new Agent (all
   network parameters randomly initialized from `--seed`) and a new empty
   replay buffer, then prefills the buffer with random-policy episodes.
   Nothing is shared between seeds.
2. **Collection.** The agent acts in 4 parallel CARL envs (stochastic
   policy; each episode takes the next context from the run's 100-context
   pool, shuffled round-robin per env) and streams
   transitions into a uniform FIFO replay buffer (capacity 1e6 steps).
   The optional online queue (Schmitt et al. 2020) is disabled
   (`replay_online: False`), so training is pure uniform replay.
3. **World-model training.** Each gradient step samples 16 sequences × 64
   steps from replay and trains the RSSM + decoder/reward/continue heads.
   DALI's context encoder trains on the same replayed data with its
   forward-dynamics + cross-modal losses (separate optimizer).
4. **Behavior training.** From every replayed state, the world model imagines
   15-step rollouts under the current policy; the actor and critic train on
   these imagined trajectories via TD(λ) (the critic's λ-targets bootstrap
   from the critic itself — see §7 on interpreting the logs).
5. **Logging.** Metrics stream to `logdir/metrics.jsonl`. Two return metrics
   matter and are NOT the same thing:
   - `episode/score` — real environment return of each collection episode
     (1 point per episode; noisy; max 1000; ~1300 points per 1.3M-step run).
   - `train/extr_return_raw_mean` — mean imagined λ-return over the training
     batch (~15k values per point; smooth; *discounted*, so ≈ score × (1−γ)
     × 1000 ≈ score/3; sparse, ~40 points per run).
6. **Checkpointing.** `checkpoint.ckpt` in the logdir, loaded on requeue.
   A run resumes ONLY if a checkpoint already exists in its logdir — which is
   why the launch scripts wipe the logdir on genuine launches (§6).

Runs converge well before 1M steps and are stable through 1.5M (no
over-training collapse — see §8 for the artifact that once suggested one).

## 4. Evaluation: zero-shot OOD

`contextual_mbrl.dreamer.eval` loads a frozen checkpoint and rolls it out on a
deterministic grid of held-out contexts (20 real episodes per context, mean
return per context written to `eval.jsonl`). No imagination is involved;
these are real environment returns.

**The eval-grid fix (setup/fix_eval_grid.py) is essential.** Upstream,
`gen_carl_val_envs` swept only the FIRST TWO context features regardless of
task — so at 3- and 4-var the added variables were never actually tested OOD,
substantially understating degradation. Our patch sweeps every varying
feature, individually and jointly (joint combinations subsampled to a cap).

**The training-sampler fix (setup/fix_train_sampler.py) is equally
essential.** Upstream's `double_box` branch of `make_carl_env` builds
distributions for features 0 and 1 only (`for index in range(2)`), correct for
every upstream task but silently leaving features 3+ at their defaults in
every training context. Without the fix, a "4-var" run trains on exactly the
2-var distribution. `setup/check_training_contexts.py` verifies both the
source and the sampled pool.

Contexts are classified by `classify()` in `analysis.py` into regimes
(Kirk et al. 2023): Interpolation (all varied values inside the training
range), Extrapolation (all outside), Mixed (some in, some out). Returns are
normalized by 1000 and pooled over contexts × seeds per regime.

## 5. Analysis layer

One script, `analysis.py` (run with `LOG_ROOT` pointing at the run directory):

- `tables` — training peak (mean±std over seeds) and OOD mean return per
  regime, pooled over seeds
- `singles` — per-feature interp/extrap means on single-feature contexts
- `lattice` — OOD per regime for each lattice cell
- `curves`, `realimag` — training curves (imagined; real vs imagined)
- `diagnostics VAR`, `gradnorms VAR` — world-model calibration, policy
  entropy, critic, and optimizer gradient-norm panels

It contains the shared machinery (loaders, regime classifier, edge-normalized
smoother, band plotting) plus the commands.

On Babel, run analysis on a compute node (only compute nodes see `/data`).

## 6. Cluster execution (SLURM)

- **GPU pinning:** the DALI JAX build supports sm_80/86/89 (A100, A6000,
  L40, L40S) but NOT Hopper (H100, sm_90a) — jobs must pin a compatible type,
  e.g. `--gres=gpu:L40S:1` (Babel) or `--gpus=l40s-48:1` (Bridges-2). A full training run is ~9 GPU-hours (DALI) /
  ~4 (cRSSM); an eval job ~1.5.
- **Array layout:** seed-major — consecutive array indices are the same seed
  at 2/3/4-var, so within-seed comparisons complete first as the queue fills.
- **Fresh vs resume:** on a genuine launch the script wipes the logdir
  (guaranteeing fresh init, see §3.6); on SLURM auto-requeue
  (`SLURM_RESTART_COUNT > 0`) or explicit `RESUME=1` it keeps the logdir and
  the run resumes from its own checkpoint. This distinction prevents the
  silent resume-from-stale-checkpoint failure mode.
- **Filesystem:** logs, checkpoints and replay go under `LOG_ROOT` on
  large storage (Babel: `/data/user_data/$USER/...`, visible only on compute
  nodes; Bridges-2: `/ocean/projects/<account>/$USER/...`). Code lives in home.

## 7. Interpreting the logs — the non-obvious parts

- **Imagined vs real return scale.** The imagined λ-return is discounted
  (γ≈0.997, effective horizon ≈333 steps), so its plateau ≈ real episode
  score / 3. The discount-adjusted imagined return tracking the real score is
  an external calibration check of the world model + critic (a quantity the
  critic never trains on).
- **Critic ≈ λ-targets is self-consistency, not reality-calibration.** The
  λ-targets bootstrap from the critic itself, so mean-level overlap is
  expected for any converged TD critic. Use the discount-adjusted comparison
  above for calibration against reality.
- **Critic loss magnitude (~1e4)** is a categorical (twohot, 255-bin)
  cross-entropy accumulated over batch × imagination horizon; only trends are
  meaningful. Near-zero early values = the "no reward found yet" phase where
  all targets are identically zero.
- **Policy entropy** is differential entropy of the 2-dim squashed-Gaussian
  action distribution: ≈2.83 at initialization (σ≈1 per dim), settling near
  −1 (σ≈0.15) when converged. A seed pinned at 2.83 has not found reward yet
  (exploration delay — the source of most between-seed variance at 3-var).

## 8. The smoothing artifact (lesson)

Early plots used `np.convolve(y, ones(w)/w, mode="same")`, which zero-pads
past the array ends: the final ~w/2 points are dragged toward zero (the last
point to ~half its true value). On the sparsely-logged imagined-return metric
(~40 points/run, w=25) this manufactured a convincing "collapse after 900K
steps" spanning ~400K steps — corroborated by fake companion signals
(negative entropy dragged up = "entropy rise"; positive critic loss dragged
down = "loss plunge"). An LR-annealing intervention changed nothing, and raw
(unsmoothed) curves are flat: there is no collapse.

All plotting here uses the edge-normalized smoother:

```python
k = np.ones(w)
smoothed = np.convolve(y, k, "same") / np.convolve(np.ones(len(y)), k, "same")
```

(the denominator counts real samples per window, so edge windows are honest
averages). Rules of thumb: verify surprising trends on raw data; scale the
window to the number of logged points; eval results were never affected
(real rollouts, never smoothed).

## 9. Reproducing the headline tables

1. Install and patch (`setup/INSTALL.md`).
2. Train: `slurm/train.sbatch` (DALI, 9 seeds × 3 levels: `--array=0-26`;
   cRSSM: `--export=ALL,SCHEME=crssm --array=0-8`).
3. Evaluate: `slurm/reeval.sbatch` with the same array indices
   (non-destructive: backs up any existing `eval.jsonl`, restores on failure).
4. `python analysis.py tables` prints both tables;
   `python analysis.py curves dali` renders the curves.

Budget: ~9 GPU-h per DALI training run, ~4 per cRSSM run, ~1.5 per eval;
the full study to date used ~720 GPU-hours, all single-GPU jobs.
