#!/usr/bin/env python3
"""Extend DALI's contextual_mbrl/dreamer/envs.py with the CARL DMC Finger
escalation tasks (2/3/4 varying dynamics, nested supersets):

  dmc_finger_2var: gravity, actuator_strength
  dmc_finger_3var: + joint_damping
  dmc_finger_4var: + friction_tangential

Updates the module's lookup tables at import time (_TASK2CONTEXTS, _TASK2ENV,
and the context normalizer)

Usage (after cloning frankroeder/DALI):
  python setup/add_finger_tasks.py [~/DALI/contextual_mbrl/dreamer/envs.py]
"""
import os
import sys

PATH = sys.argv[1] if len(sys.argv) > 1 else os.path.expanduser(
    "~/DALI/contextual_mbrl/dreamer/envs.py")
MARKER = "# --- dali-finger-study: Finger escalation tasks ---"

src = open(PATH).read()
if MARKER in src:
    print("already patched -- nothing to do")
    sys.exit(0)

# The names the appended block relies on must exist upstream
for needed in ("_TASK2CONTEXTS", "_TASK2ENV", "class NormalizeContextWrapper",
               "_CLASS_TO_CTX_NORMALIZER"):
    assert needed in src, f"upstream anchor missing: {needed} -- aborting, nothing written"

BLOCK = '''

''' + MARKER + '''
# Appended by dali-finger-study/setup/add_finger_tasks.py. Registers the
# nested Finger tasks by updating the module lookup tables defined above.
from carl.envs.dmc import CARLDmcFingerEnv

FINGER_TRAIN_GRAVITY_RANGE = [4.9, 14.70]
FINGER_TRAIN_ACTUATOR_STRENGTH_RANGE = [0.5, 1.5]
FINGER_TRAIN_JOINT_DAMPING_RANGE = [0.5, 1.5]
FINGER_TRAIN_FRICTION_TANGENTIAL_RANGE = [0.5, 1.5]

_UNIT_EVAL_VALUES = dict(  # shared by every [0.5, 1.5]-range factor
    interpolate_single=[0.5, 0.75, 1.0, 1.25, 1.5],
    interpolate_double=[0.5, 1.0, 1.5],
    extrapolate_single=[0.1, 0.2, 0.3, 0.4, 1.6, 1.7, 1.8, 1.9, 2.0],
    extrapolate_double=[0.1, 0.3, 1.6, 1.8, 2.0],
)

_FINGER_FEATS = [
    {
        "context": "gravity",
        "train_range": FINGER_TRAIN_GRAVITY_RANGE,
        "interpolate_single": [4.9, 7.35, 9.81, 12.25, 14.7],
        "interpolate_double": [4.9, 9.81, 14.7],
        "extrapolate_single": [0.98, 1.715, 2.45, 3.185, 3.92,
                               15.68, 16.66, 17.64, 18.62, 19.6],
        "extrapolate_double": [0.98, 2.45, 3.92, 15.68, 17.64, 19.6],
    },
    {"context": "actuator_strength",
     "train_range": FINGER_TRAIN_ACTUATOR_STRENGTH_RANGE, **_UNIT_EVAL_VALUES},
    {"context": "joint_damping",
     "train_range": FINGER_TRAIN_JOINT_DAMPING_RANGE, **_UNIT_EVAL_VALUES},
    {"context": "friction_tangential",
     "train_range": FINGER_TRAIN_FRICTION_TANGENTIAL_RANGE, **_UNIT_EVAL_VALUES},
]

_TASK2CONTEXTS.update({
    "dmc_finger_2var": _FINGER_FEATS[:2],
    "dmc_finger_3var": _FINGER_FEATS[:3],
    "dmc_finger_4var": _FINGER_FEATS[:4],
})
_TASK2ENV.update({name: CARLDmcFingerEnv for name in
                  ("dmc_finger_2var", "dmc_finger_3var", "dmc_finger_4var")})
NormalizeContextWrapper._CLASS_TO_CTX_NORMALIZER[CARLDmcFingerEnv] = {
    "gravity": FINGER_TRAIN_GRAVITY_RANGE,
    "actuator_strength": FINGER_TRAIN_ACTUATOR_STRENGTH_RANGE,
    "joint_damping": FINGER_TRAIN_JOINT_DAMPING_RANGE,
    "friction_tangential": FINGER_TRAIN_FRICTION_TANGENTIAL_RANGE,
}
'''

with open(PATH, "a") as f:
    f.write(BLOCK)

# verify: re-import-level sanity on the written file
out = open(PATH).read()
for needed in ("dmc_finger_2var", "dmc_finger_3var", "dmc_finger_4var",
               "friction_tangential"):
    assert needed in out, f"write verification failed: {needed}"
print("patched: appended Finger task block (2var, 3var, 4var)")
