#!/usr/bin/env python3
"""Add a "lattice" training-context mode to contextual_mbrl/dreamer/envs.py.

Spec string:  --env.carl.context lattice_<idx>-<val>

Fixes the task's feature number <idx> at the constant <val> for every
training context, while all remaining features sample uniformly from their
train ranges (pool of 100, seeded by config.seed, per-episode shuffled
round-robin -- identical machinery to double_box otherwise).

Example (task dmc_finger_3var, features 0=gravity 1=actuator 2=damping):
  lattice_2-0.5   -> joint_damping pinned at 0.5; gravity and actuator vary

Run AFTER setup/fix_train_sampler.py:
  python setup/add_lattice_context.py [~/DALI/contextual_mbrl/dreamer/envs.py]
"""
import os
import sys

PATH = sys.argv[1] if len(sys.argv) > 1 else os.path.expanduser(
    "~/DALI/contextual_mbrl/dreamer/envs.py")
MARKER = "# --- dali-finger-study: lattice training context ---"
SAMPLER_MARKER = "# --- dali-finger-study: N-feature training sampler ---"

src = open(PATH).read()
if MARKER in src:
    print("already patched -- nothing to do")
    sys.exit(0)

# must append after the N-feature sampler so this definition wins at import
assert SAMPLER_MARKER in src, (
    "run setup/fix_train_sampler.py first -- aborting, nothing written")
for needed in ("def create_wrapped_carl_env", "_TASK2CONTEXTS", "_TASK2ENV",
               "ContextSampler", "UniformFloatContextFeature"):
    assert needed in src, f"upstream anchor missing: {needed} -- aborting, nothing written"

BLOCK = '''

''' + MARKER + '''
# Appended by dali-finger-study/setup/add_lattice_context.py. OVERRIDES the
# make_carl_env defined earlier in this file; identical behaviour except for
# the added "lattice" context mode.
def make_carl_env(config, **overrides):
    suite, task = config.task.split("_", 1)
    assert suite == "carl", suite

    env_cls: CARLEnv = _TASK2ENV[task]
    contexts = {}

    if config.env.carl.context == "default":
        contexts = {0: env_cls.get_default_context()}
    elif "lattice" in config.env.carl.context:
        # one feature pinned at a constant, the rest uniform over their ranges
        spec = config.env.carl.context.split("_", 1)[1]
        fixed_idx, fixed_val = int(spec.split("-")[0]), float(spec.split("-", 1)[1])
        feats = _TASK2CONTEXTS[task]
        assert 0 <= fixed_idx < len(feats), f"no feature {fixed_idx} in {task}"
        # sample only the varying features (a zero-width uniform is rejected),
        # then pin the fixed feature to its constant in every sampled context
        ctx_dists = [
            UniformFloatContextFeature(
                f["context"], f["train_range"][0], f["train_range"][1])
            for i, f in enumerate(feats) if i != fixed_idx
        ]
        # All workers sample the same contexts based on the experiment seed
        sampler = ContextSampler(
            context_distributions=ctx_dists,
            context_space=env_cls.get_context_space(),
            seed=config.seed,
        )
        contexts = sampler.sample_contexts(n_contexts=100)
        fixed_name = feats[fixed_idx]["context"]
        for c in contexts.values():
            c[fixed_name] = fixed_val
    elif "single" in config.env.carl.context:
        index = int(config.env.carl.context.split("_")[-1])
        context_name = _TASK2CONTEXTS[task][index]["context"]
        train_range = _TASK2CONTEXTS[task][index]["train_range"]
        ctx_dists = [
            UniformFloatContextFeature(context_name, train_range[0], train_range[1])
        ]
        sampler = ContextSampler(
            context_distributions=ctx_dists,
            context_space=env_cls.get_context_space(),
            seed=config.seed,
        )
        contexts = sampler.sample_contexts(n_contexts=100)
    elif "double_box" in config.env.carl.context:
        ctx_dists = [
            UniformFloatContextFeature(
                f["context"], f["train_range"][0], f["train_range"][1])
            for f in _TASK2CONTEXTS[task]
        ]
        sampler = ContextSampler(
            context_distributions=ctx_dists,
            context_space=env_cls.get_context_space(),
            seed=config.seed,
        )
        contexts = sampler.sample_contexts(n_contexts=100)
    elif "specific" in config.env.carl.context:
        ctx_vals = {
            int(v.split("-")[0]): float(v.split("-")[-1])
            for v in config.env.carl.context.split("_")[1:]
        }
        specific_context = env_cls.get_default_context()
        for i, val in ctx_vals.items():
            context_name = _TASK2CONTEXTS[task][i]["context"]
            specific_context[context_name] = val
        contexts[0] = specific_context
    else:
        raise NotImplementedError(f"Context {config.env.carl.context} not implemented.")

    return create_wrapped_carl_env(env_cls, contexts, config)
'''

with open(PATH, "a") as f:
    f.write(BLOCK)

out = open(PATH).read()
assert out.count("def make_carl_env") == 3, "write verification failed"
print("patched: lattice context mode available")
print("  e.g. --task carl_dmc_finger_3var --env.carl.context lattice_2-0.5")
print("verify with: python setup/check_lattice_contexts.py")
