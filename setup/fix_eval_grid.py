#!/usr/bin/env python3
"""Extend DALI's evaluation grid to tasks with any number of varying context
features (upstream assumes exactly two). Replaces gen_carl_val_envs with a
version that sweeps every varying feature of a task: single-feature sweeps
over each feature's interpolate/extrapolate values, plus a joint sweep varying
all features together (deterministically subsampled to a fixed size so the
grid is identical across runs and comparably sized across tasks).

Usage (after add_finger_tasks.py):
  python setup/fix_eval_grid.py [~/DALI/contextual_mbrl/dreamer/envs.py]
"""
import os
import sys

PATH = sys.argv[1] if len(sys.argv) > 1 else os.path.expanduser(
    "~/DALI/contextual_mbrl/dreamer/envs.py")
MARKER = "# --- dali-finger-study: N-feature evaluation grid ---"

src = open(PATH).read()
if MARKER in src:
    print("already patched -- nothing to do")
    sys.exit(0)

for needed in ("def gen_carl_val_envs", "def create_wrapped_carl_env",
               "_TASK2CONTEXTS", "_TASK2ENV"):
    assert needed in src, f"upstream anchor missing: {needed} -- aborting, nothing written"

BLOCK = '''

''' + MARKER + '''
def gen_carl_val_envs(config, **overrides):
    import itertools as _it
    import random as _rnd
    suite, task = config.task.split("_", 1)
    assert suite == "carl", suite
    env_cls: CARLEnv = _TASK2ENV[task]
    ctx_default = env_cls.get_default_context()
    contexts = []

    if "specific" in config.env.carl.context:
        # unchanged upstream behaviour: evaluate one explicit context,
        # e.g. context "specific_0-12.0_1-0.6" sets feature 0 and 1 by index
        ctx_vals = {
            int(v.split("-")[0]): float(v.split("-")[-1])
            for v in config.env.carl.context.split("_")[1:]
        }
        c = env_cls.get_default_context()
        changed = []
        for i, val in ctx_vals.items():
            context_name = _TASK2CONTEXTS[task][i]["context"]
            c[context_name] = val
            changed.append(context_name)
        contexts.append({"context": c, "changed": changed})
    else:
        feats = _TASK2CONTEXTS[task]
        names = [f["context"] for f in feats]
        defaults = {n: ctx_default[n] for n in names}

        contexts.append({"context": env_cls.get_default_context(), "changed": []})

        # sweep each feature alone over interp+extrap values
        for f in feats:
            n = f["context"]
            for v in f["interpolate_single"] + f["extrapolate_single"]:
                if v == defaults[n]:
                    continue
                c = env_cls.get_default_context()
                c[n] = v
                contexts.append({"context": c, "changed": [n]})

        # joint: vary all features together (all non-default), subsampled to a
        # fixed cap so the joint regime is comparable across 2/3/4-var.
        JOINT_CAP = 56
        dbl = [f["interpolate_double"] + f["extrapolate_double"] for f in feats]
        combos = [combo for combo in _it.product(*dbl)
                  if all(combo[i] != defaults[names[i]] for i in range(len(names)))]
        if len(combos) > JOINT_CAP:
            combos = _rnd.Random(0).sample(combos, JOINT_CAP)  # deterministic
        for combo in combos:
            c = env_cls.get_default_context()
            for i, n in enumerate(names):
                c[n] = combo[i]
            contexts.append({"context": c, "changed": list(names)})

    for context_info in contexts:
        ctors = []
        for index in range(config.envs.amount):
            ctor = lambda: create_wrapped_carl_env(
                env_cls, contexts={0: context_info["context"]}, config=config
            )
            if config.envs.parallel != "none":
                ctor = bind(embodied.Parallel, ctor, config.envs.parallel)
            if config.envs.restart:
                ctor = bind(embodied.wrappers.RestartOnException, ctor)
            ctors.append(ctor)
        envs = [ctor() for ctor in ctors]
        yield embodied.BatchEnv(
            envs, parallel=(config.envs.parallel != "none")
        ), context_info
'''

with open(PATH, "a") as f:
    f.write(BLOCK)

out = open(PATH).read()
assert out.count("def gen_carl_val_envs") == 2, "write verification failed"
assert "JOINT_CAP" in out
print("patched: evaluation grid now sweeps every varying feature of a task")
