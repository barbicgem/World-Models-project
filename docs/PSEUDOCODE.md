# Pseudocode: the full study, faithful to the implementation

One block, Python-like. Every constant is the literal value used. Each section
is annotated with the file and function that implements it (paths relative to
the DALI repo; our patches noted). `rng` marks randomness derived from the
run's `--seed`; `RNG(0)` is a fixed seed shared by all runs.

```python
# ---------------------------- constants -------------------------------------
# sources: contextual_mbrl/dreamer/configs.yaml (defaults + carl + scheme blocks)
SCHEMES        = ["dali", "crssm"]        # dali infers context; crssm is given it
VAR_COUNTS     = [2, 3, 4]                # nested levels
SEEDS          = [42, 1337, 13, 71, 1994, 1997, 908, 2102, 3]

FEATURES       = ["gravity", "actuator_strength", "joint_damping", "friction_tangential"]
TRAIN_RANGES   = {"gravity": (4.9, 14.7), "actuator_strength": (0.5, 1.5),
                  "joint_damping": (0.5, 1.5), "friction_tangential": (0.5, 1.5)}

N_ENVS         = 4           # parallel envs (envs.amount), each a subprocess
EPISODE_LEN    = 1000        # env steps per episode; reward 0/1 per step
TOTAL_STEPS    = 1_500_000   # env steps per run, summed over the 4 envs
N_CONTEXTS     = 100         # training context pool size per run

REPLAY_CAP     = 1_000_000   # transitions; uniform sampling, FIFO eviction
B              = 16          # batch size (sequences per draw)
L              = 64          # sequence length  (one draw = B*L = 1024 timesteps)
TRAIN_RATIO    = 512         # replayed timesteps per env step
                             # -> updates/env_step = TRAIN_RATIO/(B*L) = 0.5
H              = 15          # imagination horizon
GAMMA          = 0.997       # discount (effective horizon ~333 steps)
LAMBDA         = 0.95        # TD-lambda mixing
K              = 64          # context-inference history length (DALI)
CTX_DIM        = 8           # inferred-context dimensionality (DALI)
LAMBDA_CROSS   = 1.0         # cross-modal loss weight (DALI)
LR_MODEL, LR_ACTOR, LR_CRITIC = 1e-4, 3e-5, 3e-5   # constant; no schedules
CLIP_MODEL, CLIP_AC = 1000.0, 100.0                # gradient clip (global norm)

EVAL_EPISODES  = 20          # real episodes per eval context
JOINT_CAP      = 56          # joint-sweep contexts per level


# ---------------------------- top level -------------------------------------
def run_study():                                    # slurm/train.sbatch array
    for scheme in SCHEMES:
        for var in VAR_COUNTS:
            feats = FEATURES[:var]                  # nested by construction
            for seed in SEEDS:
                train(scheme, feats, seed)          # -> checkpoint.ckpt, metrics.jsonl
                evaluate(scheme, feats, seed)       # -> eval.jsonl  (slurm/reeval.sbatch)
    analyze()                                       # -> tables, figures (analysis.py)


# ---------------------------- training --------------------------------------
def train(scheme, feats, seed):                     # contextual_mbrl/dreamer/train.py::main
    rng = RNG(seed)          # sole source of ALL run randomness below

    # fresh construction; nothing shared between seeds
    theta  = init(rng)       # world model: RSSM + obs/reward/continue heads
    phi    = init(rng)       # actor            (DreamerV3-small sizes)
    psi    = init(rng)       # critic
    if scheme == "dali":
        varphi = init(rng)   # transformer context encoder (nets.py::CtxEncoder)

    # training context pool, drawn ONCE per run           envs.py::make_carl_env
    # ("double_box" branch; our setup/fix_train_sampler.py makes it cover all
    #  of feats -- upstream sampled only the first two)
    pool = [{f: rng.uniform(*TRAIN_RANGES[f]) for f in feats}
            for _ in range(N_CONTEXTS)]
    # each of the 4 envs gets its own shuffled round-robin cursor over the
    # SAME pool                            envs.py::RandomizedRoundRobinSelector

    replay = FIFO(cap=REPLAY_CAP, dir=logdir/"replay")   # embodied/replay: Uniform
                                                          # (online queue off)

    # prefill with a random policy until one batch exists      train.py::train
    while len(replay) < B * L:
        collect steps with random actions

    # main loop: driver(policy, steps=100) until TOTAL_STEPS   train.py::train
    # gradient updates are an on_step callback, gated to fire once per
    # ~2 env steps by embodied.when.Ratio(TRAIN_RATIO / (B*L))

    # ==== one driver tick, all 4 envs batched ====  embodied Driver._step
    step envs with actions SAVED last tick -> observations
    for any env whose episode ended: reset fires ->
        selector picks next pool context (one per episode; round-robin
        covers all 100 per cycle, reshuffled)      carl_env.py::CARLEnv.reset
        new initial finger/spinner pose <- rng
    # physics itself is deterministic given (state, action)     MuJoCo

    # -- choose next actions --                       agent.py::Agent.policy
    push obs and PREVIOUS action into the K=64 rolling stacks
        # stacks are zero-initialized but NOT cleared at episode
        # boundaries -- early-episode windows contain prior-episode steps
    embed = encoder(obs)                              nets.py::MultiEncoder
    if scheme == "dali":
        z = ctx_encoder(stacks)[:, -1]                nets.py::CtxEncoder
    else:
        z = obs["context"]      # true context, normalized to [-1, 1]
                                #        envs.py::NormalizeContextWrapper
    latent = rssm.obs_step(prev_latent, prev_action, embed, is_first, z)
        # predict (GRU prior, z concatenated into the transition input),
        # correct with embed, sample CURRENT posterior from a 32x32
        # categorical (unimix 1%)                     nets.py::RSSM.obs_step
    action = actor(latent, z).sample(rng)   # 2-dim tanh-Gaussian torques
    save actions for next tick; zero them for envs that just ended

    # -- record --                                    Driver._step callbacks
    replay.add(transition); step counter += 1; episode accumulator += 1
    on episode end: log {"episode/score": sum(rewards)}        # REAL return

    # ==== learning callback (once per ~2 env steps) ====  agent.py::Agent.train
    batch1 = replay.sample_uniform(rng, B, L)         embodied replay _sample
    world-model update on batch1:                     agent.py::WorldModel.loss
        rssm.observe (scanned obs_step), decoder/reward/continue heads
        (twohot/symlog), KL(prior || post) dyn + rep losses with free bits
        lr=LR_MODEL, clip=CLIP_MODEL
    if scheme == "dali":  # same loss call, rolling-window section
        ctx-encoder update: L_FD (forward dynamics, CtxEncoder.forward_model)
            + LAMBDA_CROSS * L_cross (CtxEncoder.reconstruct_wm_state_loss);
        .detach()/sg() keeps encoder and world-model gradients separate

    behavior update (theta, varphi frozen):    behaviors.py + agent.py
        starts = latents of batch (1024 states)
        traj = wm.imagine(actor, starts, H)    agent.py::WorldModel.imagine
            # rssm.img_step chained 15x; next latents SAMPLED from the
            # model's categorical prior; z carried along
        V = td_lambda(traj, critic, GAMMA, LAMBDA)  agent.py::VFunction.score
        actor:  maximize normalized advantage + actent * entropy
                                              agent.py::ImagActorCritic.loss
        critic: twohot CE toward V.detach() + slow-critic regularizer
                                              agent.py::VFunction.loss
    log train metrics (~ every 30K steps); save checkpoint (~ every 5 min;
    auto-requeue resumes from it)


# ---------------------------- evaluation ------------------------------------
def evaluate(scheme, feats, seed):                # contextual_mbrl/dreamer/eval.py::main
    agent = load(checkpoint.ckpt); freeze all parameters
    rng = RNG(seed)                 # episode initial poses only, NOT the grid

    # grid ENUMERATED, identical for every run
    #        envs.py::gen_carl_val_envs (our setup/fix_eval_grid.py version)
    grid = [ {defaults, changed: []} ]                            # 1 default
    for f in feats:                                               # singles
        for v in interp_single[f] + extrap_single[f]:             # 13-14 values
            grid += [ {defaults, f: v, changed: [f]} ]
    combos = product(interp_double[f] + extrap_double[f] for f in feats)
    combos = [c for c in combos if all features non-default]      # 56/392/2744
    grid  += RNG(0).sample(combos, JOINT_CAP)     # fixed 56, same for all runs
    # grid sizes: 84 (2var) / 97 (3var) / 110 (4var) contexts

    for ctx in grid:
        env = make_env(ctx)                       # real physics, single env
        returns = []
        for _ in range(EVAL_EPISODES):            # 20 real episodes
            # actions still SAMPLED in eval mode (agent.py::Agent.policy);
            # dali re-infers z online from the episode's own history
            returns.append(rollout(agent, env, EPISODE_LEN))
        write(eval.jsonl, {"ctx": {values, changed}, "return": mean(returns)})


# ---------------------------- analysis --------------------------------------
def analyze():                                    # analysis.py (compute node)
    # tables: per (scheme, var) training peak, mean +/- std over seeds;
    #         OOD mean return per regime, normalized by 1000, pooled over
    #         contexts x seeds; regimes from classify():
    #             all changed values inside TRAIN_RANGES -> Interpolation
    #             all outside -> Extrapolation; mixed -> Mixed
    # curves: per-run series -> common step grid -> mean +/- std band
    #         (edge-normalized smoothing; short runs excluded from bands)
    ...
```

## What the seed does / does not control

| seed-dependent (the experience) | fixed (the task and the exam) |
|---|---|
| network initialization | training ranges, var-count, all hyperparameters |
| the 100-context training pool | evaluation grid (values and joint subsample) |
| per-episode initial poses (train and eval) | physics given (state, action) |
| exploration action noise | regime definitions, episodes per context |
| replay sampling, imagination sampling | round-robin coverage of the pool |

Seeds vary the experience; the task and the exam are identical for every run.
Differences across seeds are luck; effects that survive averaging over seeds
are attributable to the experimental variable (the number of varying dynamics).
