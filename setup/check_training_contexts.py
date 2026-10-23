#!/usr/bin/env python3
"""Verify the training-context pipeline varies every feature of a task.

Two checks, no training required (run inside the DALI env, e.g.
`cd ~/DALI && uv run python setup/check_training_contexts.py`):

1. source check: the active make_carl_env contains no hardcoded range(2)
2. sampling check: build the double_box context pool exactly as
   make_carl_env does for dmc_finger_4var and assert every feature
   actually varies across the 100 sampled contexts
"""
import inspect

from carl.context.context_space import UniformFloatContextFeature
from carl.context.sampler import ContextSampler

from contextual_mbrl.dreamer import envs

# ---- 1. source check ----------------------------------------------------
src = inspect.getsource(envs.make_carl_env)
assert "range(2)" not in src, (
    "make_carl_env still samples only two features -- run setup/fix_train_sampler.py")
print("source check OK: make_carl_env has no two-feature assumption")

# ---- 2. sampling check --------------------------------------------------
task = "dmc_finger_4var"
feats = envs._TASK2CONTEXTS[task]
ctx_dists = [
    UniformFloatContextFeature(f["context"], f["train_range"][0], f["train_range"][1])
    for f in feats
]
pool = ContextSampler(
    context_distributions=ctx_dists,
    context_space=envs._TASK2ENV[task].get_context_space(),
    seed=0,
).sample_contexts(n_contexts=100)

print(f"\n{task}: spread over 100 sampled training contexts")
ok = True
for f in feats:
    name = f["context"]
    vals = [c[name] for c in pool.values()]
    lo, hi = min(vals), max(vals)
    spread = hi - lo
    rng = f["train_range"][1] - f["train_range"][0]
    good = spread > 0.5 * rng          # covers a substantial part of the range
    ok &= good
    print(f"  {name:22s} min {lo:7.3f}  max {hi:7.3f}  "
          f"({'OK' if good else 'NOT VARYING'})")

assert ok, "some features are not varying -- sampler fix not effective"
print("\nsampling check OK: all features vary across the training pool")
