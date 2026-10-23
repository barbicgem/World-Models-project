# Installing the DALI stack

Upstream DALI ships a `setup.py` that declares almost no dependencies and no
`pyproject.toml`, so `uv sync` from its README does not work and an unpinned
install pulls versions the code no longer runs with. This is the sequence that
reproduces the environment the study ran in (verified on PSC Bridges-2;
equivalent to the original CMU Babel install).

## 1. Environment

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh     # if uv is missing
git clone https://github.com/frankroeder/DALI.git ~/DALI
cd ~/DALI
uv venv --python 3.11
uv pip install -e ./dreamerv3_compat
uv pip install -e ./
```

## 2. Pins (order matters; later installs can move earlier pins)

```bash
# CARL: the DALI author's fork. PyPI/main CARL lacks the ball-in-cup env that
# DALI's envs.py imports unconditionally.
uv pip install "carl-bench @ git+https://github.com/frankroeder/CARL.git@811d069eb6b7ed47bc3fcadcf61942301a149d36"
uv pip install Box2D                 # imported unconditionally by envs.py
uv pip install "gymnasium==0.29.1"   # code uses pre-1.0 wrapper paths
uv pip install "jax[cuda12]==0.4.30" # the version the study ran on
uv pip install "numpy<2.1"           # matches jaxlib 0.4.30
```

Check:

```bash
uv pip list | grep -iE "jax|numpy|carl|dm-control|gymnasium"
# expect: jax/jaxlib 0.4.30 + jax-cuda12-plugin/pjrt 0.4.30, numpy 2.0.x,
#         carl-bench 1.1.2, dm-control, gymnasium 0.29.1
uv run python -c "from contextual_mbrl.dreamer import envs; print('imports OK')"
```

Warnings about brax, py4j, `distance` and GLFW/`DISPLAY` are expected on a
headless node and harmless.

## 3. Patches (in this order)

```bash
cd ~/DALI
S=~/dali-finger-study/setup
uv run python $S/add_finger_tasks.py      # registers dmc_finger_{2,3,4}var
uv run python $S/fix_eval_grid.py         # eval grid sweeps every varying feature
uv run python $S/fix_train_sampler.py     # training pool varies every feature
uv run python $S/add_lattice_context.py   # optional: lattice context mode
uv run python $S/patch_eval_episodes.py   # EVAL_EPISODES env var in eval.py
```

Each script is idempotent (marker-guarded) and aborts without writing if the
upstream code it expects is missing.

## 4. Verify

The checkers import `contextual_mbrl` from outside the DALI directory, so put
it on the path:

```bash
PYTHONPATH=~/DALI uv run python $S/check_training_contexts.py
PYTHONPATH=~/DALI uv run python $S/check_lattice_contexts.py dmc_finger_3var 2 0.5
```

Then confirm the GPU is visible from a compute node (SLURM example):

```bash
salloc -A <account> -p <gpu-partition> --gpus=1 -t 00:15:00
cd ~/DALI && uv run python -c "import jax; print(jax.devices())"   # CudaDevice(...)
```

## GPU compatibility

jax 0.4.30's CUDA wheels run on Ampere/Ada GPUs (A100, A6000, L40, L40S). They
do not run on Hopper (H100, sm_90a). Pin a compatible GPU type in SLURM.
