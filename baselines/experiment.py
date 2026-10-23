"""Context-generalization experiment for the agents in rl_agents.py.

For each configured CARL env (discrete and continuous action spaces):
  * sample a set of TRAIN contexts and a DISJOINT set of held-out TEST contexts
    from the same distribution over physical parameters,
  * train the chosen algorithm (PPO or A2C) on the train contexts for several seeds,
  * periodically evaluate the deterministic policy on train vs. test contexts,
  * aggregate across seeds (mean +/- std), save raw returns to CSV and a
    train-vs-test generalization plot per env.

The gap between the train and test curves is the generalization gap: how well a
single policy transfers to dynamics it was never trained on.

Run (from this directory):
    python experiment.py                              # PPO + A2C, all envs, 3 seeds
    python experiment.py --algos ppo                  # PPO only
    python experiment.py --algos a2c --quick          # smoke test: 1 seed, tiny budget
    python experiment.py --envs CARLCartPole --explore rnd   # with RND bonus
"""

from __future__ import annotations

import argparse
import csv
import os
import random

import numpy as np
# NumPy 2.0 removed these aliases; some deps (ConfigSpace/gymnasium) still use them.
if not hasattr(np, "float_"):
    np.float_ = np.float64
if not hasattr(np, "int_"):
    np.int_ = np.int64
if not hasattr(np, "bool8"):
    np.bool8 = np.bool_
import torch
import gymnasium as gym
from gymnasium.wrappers import FlattenObservation

import matplotlib
matplotlib.use("Agg")  # headless: write PNGs without a display
import matplotlib.pyplot as plt

import carl  # noqa: F401  -- importing registers the carl/... env ids
from carl.envs import (
    CARLAcrobot,
    CARLCartPole,
    CARLMountainCar,
    CARLMountainCarContinuous,
    CARLPendulum,
)
from carl.context.context_space import NormalFloatContextFeature
from carl.context.sampler import ContextSampler

from agents import A2CAgent, PPOAgent


# ----------------------------- experiment config -----------------------------

# Each env varies two physically meaningful context features (names + valid
# bounds taken from CARL's get_context_features()). We sample them from a normal
# centered on the default, clamped to legal ranges.
EXPERIMENTS = {
    # ---- discrete action spaces ----
    "CARLCartPole": {
        "env_id": "carl/CARLCartPole-v0",
        "cls": CARLCartPole,
        "distributions": [
            NormalFloatContextFeature("length", mu=0.5, sigma=0.15, lower=0.05, upper=2.0),
            NormalFloatContextFeature("gravity", mu=9.8, sigma=2.0, lower=2.0, upper=20.0),
        ],
        "total_steps": 120_000,   # CartPole solves fast
    },
    "CARLAcrobot": {
        "env_id": "carl/CARLAcrobot-v0",
        "cls": CARLAcrobot,
        "distributions": [
            NormalFloatContextFeature("LINK_LENGTH_1", mu=1.0, sigma=0.2, lower=0.5, upper=1.5),
            NormalFloatContextFeature("LINK_MASS_1", mu=1.0, sigma=0.2, lower=0.5, upper=1.5),
        ],
        "total_steps": 150_000,
    },
    "CARLMountainCar": {
        "env_id": "carl/CARLMountainCar-v0",
        "cls": CARLMountainCar,
        "distributions": [
            NormalFloatContextFeature("force", mu=0.001, sigma=0.0003, lower=0.0005, upper=0.0015),
            NormalFloatContextFeature("gravity", mu=0.0025, sigma=0.0005, lower=0.0015, upper=0.0035),
        ],
        "total_steps": 200_000,   # sparse reward: hard exploration, may not solve
    },
    # ---- continuous action spaces ----
    "CARLMountainCarContinuous": {
        "env_id": "carl/CARLMountainCarContinuous-v0",
        "cls": CARLMountainCarContinuous,
        "distributions": [
            NormalFloatContextFeature("power", mu=0.0015, sigma=0.0004, lower=0.0008, upper=0.0025),
            NormalFloatContextFeature("max_speed", mu=0.07, sigma=0.015, lower=0.04, upper=0.1),
        ],
        "total_steps": 150_000,   # sparse-ish: also a hard-exploration case
    },
    "CARLPendulum": {
        "env_id": "carl/CARLPendulum-v0",
        "cls": CARLPendulum,
        "distributions": [
            NormalFloatContextFeature("l", mu=1.0, sigma=0.3, lower=0.3, upper=2.0),
            NormalFloatContextFeature("g", mu=10.0, sigma=3.0, lower=4.0, upper=18.0),
        ],
        "total_steps": 250_000,   # continuous control needs more
    },
}

# Per-algorithm settings. A2C does ONE update per rollout, so it needs a short
# rollout_len; PPO does multiple epochs over a large rollout.
ALGOS = {
    "ppo": {
        "cls": PPOAgent,
        "kwargs": dict(lr=3e-4, clip_ratio=0.2, n_epochs=10, minibatch_size=64),
        "rollout_len": 2048,
    },
    "a2c": {
        "cls": A2CAgent,
        "kwargs": dict(lr=7e-4),
        "rollout_len": 16,
    },
}

SEEDS = [0, 1, 2]
N_TRAIN_CONTEXTS = 10
N_TEST_CONTEXTS = 10
EVAL_EVERY_STEPS = 8192       # evaluate roughly every this many env steps
EVAL_EPISODES = 10           # deterministic episodes per evaluation
OUTDIR = os.environ.get("OUT_DIR", "results/context_generalization")


# --------------------------------- helpers -----------------------------------

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def make_contexts(cls, distributions, n_contexts, seed):
    """Sample `n_contexts` contexts from the given distributions."""
    sampler = ContextSampler(
        context_distributions=distributions,
        context_space=cls.get_context_space(),
        seed=seed,
    )
    return sampler.sample_contexts(n_contexts=n_contexts)


def make_env(env_id, contexts):
    env = gym.make(env_id, contexts=contexts)
    return FlattenObservation(env)  # dict obs ({"obs", "context"}) -> flat vector


@torch.no_grad()
def evaluate_on(agent, env, n_episodes):
    """Mean return of the deterministic policy over n_episodes on `env`."""
    returns = []
    for _ in range(n_episodes):
        obs, _ = env.reset()
        done = False
        ep_return = 0.0
        while not done:
            action, _, _ = agent.act(obs, train=False)
            obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            ep_return += reward
        returns.append(ep_return)
    return float(np.mean(returns))


def run_one(env_name, cfg, algo, seed, explore="none", beta_start=0.5, normalize_obs=True):
    """Train one (env, algo, seed); return list of (step, train_eval, test_eval)."""
    set_seed(seed)

    # disjoint train/test contexts: same distribution, different sampler seeds
    train_ctx = make_contexts(cfg["cls"], cfg["distributions"], N_TRAIN_CONTEXTS, seed)
    test_ctx = make_contexts(cfg["cls"], cfg["distributions"], N_TEST_CONTEXTS, seed + 10_000)

    train_env = make_env(cfg["env_id"], train_ctx)
    test_env = make_env(cfg["env_id"], test_ctx)

    algo_cfg = ALGOS[algo]
    agent = algo_cfg["cls"](train_env, explore=explore, beta_start=beta_start,
                            beta_anneal_steps=cfg["total_steps"],
                            normalize_obs=normalize_obs, **algo_cfg["kwargs"])
    rollout_len = algo_cfg["rollout_len"]

    total_rollouts = max(1, cfg["total_steps"] // rollout_len)
    rollouts_per_eval = max(1, EVAL_EVERY_STEPS // rollout_len)
    curve = []
    rollouts_done = 0

    while rollouts_done < total_rollouts:
        n = min(rollouts_per_eval, total_rollouts - rollouts_done)
        # agent persists across chunks (nets/optimizer), so training continues.
        agent.train(total_steps=n * rollout_len, rollout_len=rollout_len,
                    log_interval=10**9)  # silence inner logging
        rollouts_done += n
        steps = rollouts_done * rollout_len

        train_eval = evaluate_on(agent, train_env, EVAL_EPISODES)
        test_eval = evaluate_on(agent, test_env, EVAL_EPISODES)
        curve.append((steps, train_eval, test_eval))
        diag = ""
        if agent.intrinsic is not None:
            diag = f"  r_int {agent.last_int_mean:.3f} beta {agent.last_beta:.3f}"
        print(f"[{algo} {env_name} seed={seed}] step {steps:>7d}  "
              f"train {train_eval:8.1f}  test {test_eval:8.1f}  "
              f"gap {train_eval - test_eval:7.1f}{diag}")

    return curve


# ------------------------------ aggregation/IO -------------------------------

def write_csv(path, env_name, algo, per_seed_curves):
    """per_seed_curves: dict seed -> list of (step, train, test)."""
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["env", "algo", "seed", "step", "train_return", "test_return"])
        for seed, curve in per_seed_curves.items():
            for step, tr, te in curve:
                w.writerow([env_name, algo, seed, step, tr, te])


def plot_env(path, env_name, algo, per_seed_curves):
    """Plot train vs test mean +/- std across seeds."""
    seeds = sorted(per_seed_curves)
    # all seeds share the same eval schedule -> align by index
    steps = [s for s, _, _ in per_seed_curves[seeds[0]]]
    train = np.array([[tr for _, tr, _ in per_seed_curves[s]] for s in seeds])
    test = np.array([[te for _, _, te in per_seed_curves[s]] for s in seeds])

    plt.figure(figsize=(7, 4.5))
    for arr, label, color in [(train, "train contexts", "tab:blue"),
                              (test, "held-out test contexts", "tab:red")]:
        mean, std = arr.mean(0), arr.std(0)
        plt.plot(steps, mean, label=label, color=color)
        plt.fill_between(steps, mean - std, mean + std, alpha=0.2, color=color)

    plt.title(f"{env_name}: context generalization ({algo.upper()}, {len(seeds)} seeds)")
    plt.xlabel("environment steps")
    plt.ylabel(f"mean episode return ({EVAL_EPISODES} eps)")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(path, dpi=130)
    plt.close()


def plot_comparison(path, env_name, results):
    """Overlay all algorithms on one plot: train (solid) vs test (dashed)."""
    colors = {"ppo": "tab:blue", "a2c": "tab:orange"}
    plt.figure(figsize=(7.5, 4.5))
    for algo, per_seed_curves in results.items():
        seeds = sorted(per_seed_curves)
        steps = [s for s, _, _ in per_seed_curves[seeds[0]]]
        train = np.array([[tr for _, tr, _ in per_seed_curves[s]] for s in seeds])
        test = np.array([[te for _, _, te in per_seed_curves[s]] for s in seeds])
        c = colors.get(algo.split("_")[0])
        tr_m, tr_s = train.mean(0), train.std(0)
        te_m, te_s = test.mean(0), test.std(0)
        plt.plot(steps, tr_m, color=c, linestyle="-", label=f"{algo.upper()} train")
        plt.fill_between(steps, tr_m - tr_s, tr_m + tr_s, color=c, alpha=0.15)
        plt.plot(steps, te_m, color=c, linestyle="--", label=f"{algo.upper()} test")
        plt.fill_between(steps, te_m - te_s, te_m + te_s, color=c, alpha=0.10)

    plt.title(f"{env_name}: PPO vs A2C context generalization")
    plt.xlabel("environment steps")
    plt.ylabel(f"mean episode return ({EVAL_EPISODES} eps)")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(path, dpi=130)
    plt.close()


# ----------------------------------- main ------------------------------------

def run_env_sweep(env_name, cfg, algos, seeds, explore="none", beta_start=0.5,
                  normalize_obs=True):
    """Run all algos x seeds for one env; save per-algo CSV/PNG + comparison.
    Returns a list of summary rows (env, tag, train, test, gap)."""
    suffix = "" if explore == "none" else f"_{explore}"
    rows = []
    results = {}
    for algo in algos:
        tag = f"{algo}{suffix}"
        print(f"\n=== {tag.upper()} | {env_name} ===")
        per_seed_curves = {seed: run_one(env_name, cfg, algo, seed, explore,
                                         beta_start, normalize_obs)
                           for seed in seeds}
        results[tag] = per_seed_curves

        csv_path = os.path.join(OUTDIR, f"{env_name}_{tag}.csv")
        png_path = os.path.join(OUTDIR, f"{env_name}_{tag}.png")
        write_csv(csv_path, env_name, tag, per_seed_curves)
        plot_env(png_path, env_name, tag, per_seed_curves)
        print(f"  saved {csv_path} and {png_path}")

        finals = [curve[-1] for curve in per_seed_curves.values()]
        train_final = float(np.mean([tr for _, tr, _ in finals]))
        test_final = float(np.mean([te for _, _, te in finals]))
        rows.append((env_name, tag, train_final, test_final, train_final - test_final))

    if len(results) > 1:
        cmp_path = os.path.join(OUTDIR, f"{env_name}_comparison{suffix}.png")
        plot_comparison(cmp_path, env_name, results)
        print(f"  saved {cmp_path}")
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--algos", nargs="*", default=list(ALGOS), choices=list(ALGOS),
                        help="algorithms to run (default: all)")
    parser.add_argument("--quick", action="store_true",
                        help="fast smoke test: 1 seed, tiny budget")
    parser.add_argument("--envs", nargs="*", default=list(EXPERIMENTS),
                        help="subset of envs to run (default: all configured)")
    parser.add_argument("--explore", choices=["none", "rnd", "count"], default="none",
                        help="exploration bonus: none, RND, or count-based (annealed to 0)")
    parser.add_argument("--beta", type=float, default=0.5,
                        help="starting weight of the exploration bonus")
    parser.add_argument("--no-obs-norm", action="store_true",
                        help="disable observation normalization (on by default)")
    parser.add_argument("--steps-scale", type=float, default=1.0,
                        help="multiply each env's total_steps by this factor")
    args = parser.parse_args()

    seeds = [0] if args.quick else SEEDS
    os.makedirs(OUTDIR, exist_ok=True)

    # only run envs that are both configured here AND importable in this install
    available = set(carl.envs.__all__)
    runnable = [e for e in args.envs if e in EXPERIMENTS and e in available]
    skipped = [e for e in args.envs if e not in runnable]
    if skipped:
        print(f"skipping (not installed or not configured): {skipped}")
    print(f"running: {runnable}")

    summary = []
    for env_name in runnable:
        cfg = dict(EXPERIMENTS[env_name])
        if args.quick:
            cfg["total_steps"] = 12_000
        else:
            cfg["total_steps"] = int(cfg["total_steps"] * args.steps_scale)
        try:
            summary += run_env_sweep(env_name, cfg, args.algos, seeds,
                                     explore=args.explore, beta_start=args.beta,
                                     normalize_obs=not args.no_obs_norm)
        except Exception as e:  # don't let one env kill a long sweep
            print(f"!! {env_name} failed: {type(e).__name__}: {e}")

    print("\n=== summary (final, mean over seeds) ===")
    print(f"{'env':<26}{'algo':<10}{'train':>10}{'test':>10}{'gap':>10}")
    for env_name, algo, tr, te, gap in summary:
        print(f"{env_name:<26}{algo:<10}{tr:>10.1f}{te:>10.1f}{gap:>10.1f}")


if __name__ == "__main__":
    main()
