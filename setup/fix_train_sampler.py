#!/usr/bin/env python3
"""Extend DALI's training-context sampler to tasks with any number of varying
features (upstream's double_box branch samples only features [0] and [1]).
Replaces make_carl_env with a version whose double_box branch builds one
uniform distribution per varying feature of the task; the sampled pool
(n_contexts=100, seeded by config.seed) and the per-episode shuffled
round-robin selection are unchanged.

For two-feature tasks the behaviour is identical to upstream.

Usage (after add_finger_tasks.py):
  python setup/fix_train_sampler.py [~/DALI/contextual_mbrl/dreamer/envs.py]
"""
import os
import sys

PATH = sys.argv[1] if len(sys.argv) > 1 else os.path.expanduser(
    "~/DALI/contextual_mbrl/dreamer/envs.py")
MARKER = "# --- dali-finger-study: N-feature training sampler ---"

src = open(PATH).read()
if MARKER in src:
    print("already patched -- nothing to do")
    sys.exit(0)

for needed in ("def make_carl_env", "def create_wrapped_carl_env",
               "_TASK2CONTEXTS", "_TASK2ENV", "ContextSampler",
               "UniformFloatContextFeature"):
    assert needed in src, f"upstream anchor missing: {needed} -- aborting, nothing written"

BLOCK = '''

''' + MARKER + '''
# Appended by dali-finger-study/setup/fix_train_sampler.py. OVERRIDES the
# make_carl_env defined earlier in this file (whose double_box branch samples
# only the first two features); the definition below wins at import time.
def make_carl_env(config, **overrides):
    suite, task = config.task.split("_", 1)
    assert suite == "carl", suite

    env_cls: CARLEnv = _TASK2ENV[task]
    contexts = {}

    if config.env.carl.context == "default":
        contexts = {0: env_cls.get_default_context()}
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
        # one uniform distribution per varying feature of the task
        # (upstream assumed exactly two)
        ctx_dists = [
            UniformFloatContextFeature(
                f["context"], f["train_range"][0], f["train_range"][1])
            for f in _TASK2CONTEXTS[task]
        ]
        # All workers sample the same contexts based on the experiment seed
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
assert out.count("def make_carl_env") == 2, "write verification failed"
print("patched: training sampler now varies every feature of the task")
print("verify with: python setup/check_training_contexts.py")
