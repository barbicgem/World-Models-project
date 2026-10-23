#!/usr/bin/env python3
"""Verify a lattice training pool: the pinned feature is constant at its
level in all 100 contexts, and every other feature actually varies.

Usage (inside the DALI env):
  python setup/check_lattice_contexts.py [task] [fixed_idx] [fixed_val]
  defaults: dmc_finger_3var 2 0.5
"""
import sys

from carl.context.context_space import UniformFloatContextFeature
from carl.context.sampler import ContextSampler

from contextual_mbrl.dreamer import envs

task = sys.argv[1] if len(sys.argv) > 1 else "dmc_finger_3var"
fixed_idx = int(sys.argv[2]) if len(sys.argv) > 2 else 2
fixed_val = float(sys.argv[3]) if len(sys.argv) > 3 else 0.5

feats = envs._TASK2CONTEXTS[task]
ctx_dists = [
    UniformFloatContextFeature(
        f["context"], f["train_range"][0], f["train_range"][1])
    for i, f in enumerate(feats) if i != fixed_idx
]
pool = ContextSampler(
    context_distributions=ctx_dists,
    context_space=envs._TASK2ENV[task].get_context_space(),
    seed=0,
).sample_contexts(n_contexts=100)
for c in pool.values():
    c[feats[fixed_idx]["context"]] = fixed_val

print(f"{task}: lattice pool, feature {fixed_idx} "
      f"({feats[fixed_idx]['context']}) pinned at {fixed_val}")
ok = True
for i, f in enumerate(feats):
    name = f["context"]
    vals = [c[name] for c in pool.values()]
    lo, hi = min(vals), max(vals)
    if i == fixed_idx:
        good = lo == hi == fixed_val
        verdict = "OK (constant)" if good else "NOT CONSTANT"
    else:
        rng = f["train_range"][1] - f["train_range"][0]
        good = (hi - lo) > 0.5 * rng
        verdict = "OK (varies)" if good else "NOT VARYING"
    ok &= good
    print(f"  {name:22s} min {lo:7.3f}  max {hi:7.3f}  ({verdict})")

assert ok, "lattice pool malformed"
print("\nlattice check OK")
