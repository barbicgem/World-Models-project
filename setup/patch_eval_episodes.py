#!/usr/bin/env python3
"""Make the number of evaluation episodes per context configurable.

Upstream contextual_mbrl/dreamer/eval.py hardcodes 50 episodes per context.
This replaces the constant with the EVAL_EPISODES environment variable
(default 50, so behaviour is unchanged unless the variable is set).
slurm/reeval.sbatch sets EVAL_EPISODES=20.

Usage:
  python setup/patch_eval_episodes.py [~/DALI/contextual_mbrl/dreamer/eval.py]
"""
import os
import re
import sys

PATH = sys.argv[1] if len(sys.argv) > 1 else os.path.expanduser(
    "~/DALI/contextual_mbrl/dreamer/eval.py")
OLD = "metrics = eval(policy, env, args, episodes=50)"
NEW = ('metrics = eval(policy, env, args, '
       'episodes=int(os.environ.get("EVAL_EPISODES", "50")))')

src = open(PATH).read()
if NEW in src:
    print("already patched -- nothing to do")
    sys.exit(0)
assert src.count(OLD) == 1, "upstream anchor missing -- aborting, nothing written"
assert re.search(r"^import os$", src, re.M), "eval.py no longer imports os -- aborting"

with open(PATH, "w") as f:
    f.write(src.replace(OLD, NEW))
print("patched: episodes per eval context = $EVAL_EPISODES (default 50)")
