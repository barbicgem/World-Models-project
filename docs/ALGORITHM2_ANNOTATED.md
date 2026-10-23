# Algorithm 2 (DALI-S-χ), annotated against the implementation

The paper's Algorithm 2 is the idealized description of the variant we run.
This file walks it line by line and attaches what the code actually does,
with file::function anchors (paths relative to the DALI repo). Three kinds of
notes: **[detail]** — the paper abstracted a numeric/mechanical fact;
**[restructured]** — same math, different program shape; **[differs]** — the
code's behavior is genuinely not what the line says.

Constants used throughout: B=16, L=64, K=64, H=15, γ=0.997, λ=0.95,
λ_cross=1.0, replay cap 1e6, train_ratio 512 → 0.5 updates per env step,
4 parallel envs, episodes 1000 steps, 1.5M env steps per run.

---

## Lines 1–2 · Initialization

- **L1** "Initialize dataset D with S random seed episodes"
  - **[detail]** train.py::main — prefill runs a *random-action policy* until
    one batch exists (B·L = 1024 transitions), not a fixed episode count.
  - **[detail]** D is not append-only: a FIFO replay buffer, capacity 1e6
    transitions, **uniform** sampling (embodied/replay; online queue exists
    upstream but is off: replay_online is not even passed by DALI's train.py).
- **L2** "Initialize θ, φ, ψ, φ_g, W_z, W_ʒ randomly"
  - **[detail]** all from RNG(--seed); every (scheme, var, seed) cell is a
    fresh process — nothing shared between runs. W_z, W_ʒ are the two linear
    heads inside CtxEncoder.reconstruct_wm_state_loss (recon_wm / recon_ctxc).

## Line 3 · "while not converged"

- **[detail]** actually `while step < 1_500_000` on the global env-step
  counter (train.py main loop, `driver(policy, steps=100)` per iteration).

## Line 4 · "for update step c = 1 to C" — THE STRUCTURAL DIFFERENCE

- **[differs]** The paper alternates: C updates, then one episode (L47–55).
  The code **interleaves**: training fires as a Driver on_step callback,
  gated by embodied.when.Ratio to 0.5 gradient updates per env step —
  collection never pauses for training. Over one 1000-step episode this
  yields ~500 updates ≈ C, so the *budget* matches; the *ordering* doesn't.
- **[differs]** Environment interaction is **4 parallel envs** (subprocesses,
  batched), not the single env implied here and in L47–55.

## Lines 5–14 · Dynamics learning (world model θ)

- **L5** draw B sequences — **[detail]** 16 sequences × 64 steps, sampled
  uniformly from replay at *random offsets*: sequences usually start
  mid-episode (see L6 note). agent.py::Agent.train → WorldModel.loss.
- **L6** "h₀ ← 0" — **[differs]** the initial state is **learned**
  (configs: initial: learned; nets.py::RSSM.initial), and it is substituted
  wherever `is_first` is true *inside* the sequence via branch-free masking
  (RSSM.obs_step), not only at position 0.
- **L8–9** zero-pad the K-window when t < K — **[differs]** at act time the
  rolling stacks are zero-initialized once at *run start* and **never cleared
  at episode boundaries** (agent.py::Agent.policy — no is_first handling on
  the stacks): early-episode windows contain the tail of the previous
  episode, not zeros. The paper describes cleaner windows than the code has.
- **L10** ʒ ← g(o,a).detach() — matches: sg() around the ctx-encoder output
  everywhere the world model consumes it (agent.py::WorldModel.loss).
- **L11–12** ʒ in the posterior q; transition f(h,z,a) *without* ʒ —
  **[differs/config]** in code the always-on injection is the reverse:
  dcontext is concatenated into the **transition input** (nets.py::RSSM.img_step,
  the `concat([x, dcontext])` before the GRU); extra injections into
  prior/posterior are gated by add_context_prior / add_context_posterior.
  Check the scheme block in configs.yaml for which flags our runs set.
- **L11** z ~ q(·) — **[detail]** the latent is 32 categorical variables ×
  32 classes, one-hot samples, straight-through gradients, 1% unimix
  (RSSM._stats / get_dist). Not a Gaussian.
- **L14** "update θ using representation learning" — **[detail]** the actual
  loss: decoder + reward + continue heads (twohot/symlog) + KL(prior‖post)
  as separate dyn (0.5) and rep (0.1) terms with free bits 1.0; Adam-LaProp,
  lr 1e-4, grad clip 1000. agent.py::WorldModel.loss.

## Lines 15–30 · Context representation learning (φ_g, W_z, W_ʒ)

- **L15** separate draw of B segments — matches: this is the second of the
  **three independent batch draws** per update (dynamics / context / behavior).
- **L17–24** per-segment loop re-running the RSSM from h←0 over the K window —
  **[restructured]** the code computes the encoder losses on rolling windows
  *within the already-processed batch*, reusing the posterior states from the
  dynamics pass (the rolling-window section of WorldModel.loss). No re-run
  from zeros; detach placement (h.detach(), z.detach()) is preserved.
- **L25–26** z̃ = W_z(ʒ), ʒ̃ = W_ʒ(z.detach()) — **[detail]** the code's
  target is the **full latent state** (deter ⧺ flattened stoch, units =
  deter + stoch·classes), not z alone. CtxEncoder.reconstruct_wm_state_loss.
- **L27** L_FD — matches CtxEncoder.forward_model: predict o_{t+1} from
  (o_t, a_t, ʒ_t). (An inverse-dynamics variant exists; config selects fw.)
- **L30** L_total = L_FD + λ_cross·L_cross, λ_cross = 1.0 — matches; enabled
  by --ctx_encoder.crossmodal True (our sbatch sets it for dali only).
  Separate optimizer from the world model's.

## Lines 31–46 · Behavior learning (actor φ, critic ψ)

- **L31** third independent batch draw — matches.
- **L36** ʒ.detach(), θ frozen — matches (sg() on context; world-model and
  encoder params excluded from the behavior optimizers).
- **L37–38** one start state per sequence at time t — **[differs]**
  imagination launches from **every posterior state of the batch**:
  B·L = 1024 start states, not B. agent.py::WorldModel.imagine.
- **L39–43** imagination loop — matches img_step chained H=15 times; next
  latents are *samples from the categorical prior*. **[detail]** ʒ is
  carried along and concatenated into every img_step (the paper's f_θ line
  omits it, same as L12).
- **L44** TD(λ) — matches: γ=0.997, λ=0.95. agent.py::VFunction.score.
- **L45** actor update ∑V_λ — **[detail]** the real objective adds
  **return normalization** (percentile-EMA scale) and an **entropy bonus**
  (actent ≈ 3e-4); gradient through backprop-through-dynamics for continuous
  actions. agent.py::ImagActorCritic.loss.
- **L46** critic L2 toward V_λ — **[differs]** the code's critic loss is
  **twohot cross-entropy on symlog-transformed targets** plus a slow-critic
  (EMA target network) regularizer — not mean-squared error. VFunction.loss.
  lr 3e-5 both, clip 100.

## Lines 47–55 · Environment interaction — WHERE THE CMDP LIVES

- **L47** "o₁ ← env.reset()" — this one line hides the entire experimental
  variable. What reset actually does (carl_env.py::CARLEnv.reset):
  1. `_progress_instance()` → the env's RandomizedRoundRobinSelector picks
     the **next context** from the run's pool (envs.py).
  2. If changed, `_update_context()` rewrites the MuJoCo physics
     (gravity, actuator_strength, joint_damping, friction_tangential).
  3. The underlying env resets to a new initial pose (seeded).
  - **[detail]** the pool: **100 contexts drawn once per run** from the
    task's train ranges, seeded by --seed (our fixed
    envs.py::make_carl_env — upstream's double_box sampled only the first
    two features; setup/fix_train_sampler.py). Each of the 4 envs cycles the
    *same* pool in its own shuffled round-robin: every context appears
    exactly once per 100-episode cycle per env, order reshuffled each cycle.
    ≈1500 episodes/run → each context trained on ~15 times.
  - **[detail]** context values reach the agent's observation normalized to
    [-1,1] (NormalizeContextWrapper) — consumed by cRSSM, ignored by DALI.
- **L48–54** stepping — **[detail]** 4 envs batched per Driver tick;
  actions chosen at tick end execute at the *next* tick (act-then-observe).
  L49–50's per-episode zero-padding: see L8–9 — not what the code does.
- **L51** online context inference — matches Agent.policy's ctx-encoder call
  over the rolling stacks, taking [:, -1].
- **L53** a ~ π(a|s) — matches; the action distribution is a 2-dim
  tanh-mean Gaussian, **sampled in eval too** (no deterministic mode used).
- **L55** "add episode to D" — **[restructured]** transitions stream into
  replay every step (Driver on_step → replay.add), not as an episode batch
  at the end.

---

## The three differences that change conclusions

1. **Interleaving, not alternation** (L4): training and collection are one
   loop; the replay always contains data up to the current step.
2. **History windows are never cleared** (L8–9, L49–50): early-episode
   context inference sees the previous episode — a real, if small, deviation
   from the paper's stated padding.
3. **Imagination from all 1024 batch states** (L37–38): the behavior update
   is ~64× wider than a literal reading of the algorithm suggests.

Everything else is abstraction: the paper omits numbers and program shape
but describes the same computation.
