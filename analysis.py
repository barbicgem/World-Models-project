#!/usr/bin/env python3
"""Result tables, training curves and training diagnostics.

  python analysis.py tables                 # training peak + OOD-per-regime tables
  python analysis.py singles                # per-feature breakdown, single-feature contexts
  python analysis.py lattice                # OOD-per-regime table for lattice cells
  python analysis.py curves [dali|crssm]    # imagined-return training curves per level
  python analysis.py realimag [dali|crssm]  # real vs imagined training-return figures
  python analysis.py diagnostics VAR        # 4-panel DALI diagnostics for one level
  python analysis.py gradnorms VAR [dali|crssm]  # return vs optimizer gradient norms

Env vars: LOG_ROOT (run directory root), MIN_STEPS (band inclusion cutoff,
default 1e6), OUT_DIR (figure output dir, default cwd).

Run layout (written by the slurm/ launchers):
  LOG_ROOT/<group>/<seed>/metrics.jsonl   training log (one JSON dict per line)
  LOG_ROOT/<group>/<seed>/eval.jsonl      zero-shot eval (one context per line)
"""
import glob
import json
import os
import sys

import numpy as np

LOG_ROOT = os.environ.get(
    "LOG_ROOT",
    f"/data/user_data/{os.environ.get('USER', '')}/dali_logs/logs_dali_crossmodal")

SCHEMES = {
    "dali": "enc_obs_dec_obs_ctxencoder_transformer",   # infers context
    "crssm": "enc_obs_ctx_dec_obs_ctx",                 # given true context
}

MAX_RETURN = 1000.0

TRAIN_RANGES = {
    "gravity": (4.9, 14.7),
    "actuator_strength": (0.5, 1.5),
    "joint_damping": (0.5, 1.5),
    "friction_tangential": (0.5, 1.5),
}

REGIMES = ("Interpolation", "Mixed", "Extrapolation")

# lattice cells: (fixed feature index, value) on the 3-feature task
LATTICE = [(0, 4.9), (0, 9.8), (0, 14.7),
           (1, 0.5), (1, 1.0), (1, 1.5),
           (2, 0.5), (2, 1.5)]          # (2, 1.0) is the existing 2var study

K_IMAGINED = "train/extr_return_raw_mean"    # imagined lambda-return (discounted)
K_REAL = "episode/score"                     # real episode return
K_ENTROPY = "train/policy_entropy_mean"
K_CRITIC = "train/extr_critic_mean"
K_CRITIC_LOSS = "train/extr_critic_critic_opt_loss"
DISCOUNT_FACTOR = 3.0   # imagined return x ~3 ~= real score (gamma 0.997, 1000 steps)

MIN_STEPS = int(os.environ.get("MIN_STEPS", "1000000"))
OUT_DIR = os.environ.get("OUT_DIR", ".")
COLORS = {2: "tab:blue", 3: "tab:orange", 4: "tab:green"}


# ---------------------------------------------------------------- loading --

def _group(scheme, var, context="double_box"):
    return f"carl_dmc_finger_{var}var_{context}_{SCHEMES[scheme]}_normalized"


def find_runs(scheme, var, context="double_box"):
    return sorted(glob.glob(f"{LOG_ROOT}/{_group(scheme, var, context)}/*/metrics.jsonl"))


def find_evals(scheme, var, context="double_box"):
    return sorted(glob.glob(f"{LOG_ROOT}/{_group(scheme, var, context)}/*/eval.jsonl"))


def load_metrics(path, keys):
    out = {k: ([], []) for k in keys}
    for line in open(path, errors="ignore"):
        try:
            d = json.loads(line)
        except Exception:
            continue
        for k in out:
            if d.get(k) is not None:
                out[k][0].append(d["step"])
                out[k][1].append(d[k])
    return {k: (np.array(x), np.array(y)) for k, (x, y) in out.items()}


def load_eval(path):
    rows = []
    for line in open(path, errors="ignore"):
        line = line.strip()
        if line:
            try:
                rows.append(json.loads(line))
            except Exception:
                continue
    return rows


def classify(ctx):
    """Generalization regime of an evaluated context (Kirk et al. 2023).

    ctx is eval.jsonl's nested field: {"context": {feature: value}, "changed": [...]}
    """
    changed = ctx.get("changed", [])
    if not changed:
        return None
    values = ctx["context"]
    outside = sum(1 for n in changed
                  if not TRAIN_RANGES[n][0] <= values[n] <= TRAIN_RANGES[n][1])
    if outside == 0:
        return "Interpolation"
    if outside == len(changed):
        return "Extrapolation"
    return "Mixed"


# ------------------------------------------------------------- statistics --

def smooth(y, w=25):
    """Edge-normalized moving average (edge windows average only real samples)."""
    y = np.asarray(y, dtype=float)
    if len(y) < 2:
        return y
    w = min(int(w), len(y))
    k = np.ones(w)
    return np.convolve(y, k, mode="same") / np.convolve(np.ones(len(y)), k, mode="same")


def _long(series):
    return [(x, y) for x, y in series if len(y) > 10 and x[-1] >= MIN_STEPS]


def band(ax, series, color, label, smooth_w=0, scale=1.0, **kw):
    """Mean +/- std over runs on a common step grid; short runs excluded."""
    series = _long(series)
    if not series:
        return
    grid = np.linspace(5000, min(x[-1] for x, _ in series), 400)
    C = scale * np.array([np.interp(grid, x, smooth(y, smooth_w) if smooth_w else y)
                          for x, y in series])
    ax.plot(grid, C.mean(0), color=color, lw=kw.pop("lw", 1.8),
            label=f"{label} (n={len(series)})", **kw)
    if scale == 1.0:
        ax.fill_between(grid, C.mean(0) - C.std(0), C.mean(0) + C.std(0),
                        color=color, alpha=0.15)


def regime_means(eval_paths):
    buckets = {r: [] for r in REGIMES}
    nseeds = 0
    for path in eval_paths:
        rows = load_eval(path)
        if not rows:
            continue
        nseeds += 1
        for row in rows:
            regime = classify(row["ctx"])
            if regime:
                buckets[regime].append(row["return"] / MAX_RETURN)
    return buckets, nseeds


def _fmt_regimes(buckets):
    return "  |  ".join(f"{r[:6]} {np.mean(buckets[r]):.3f}" if buckets[r]
                        else f"{r[:6]} --" for r in REGIMES)


def _plt():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    return plt


def _save(plt, name):
    out = os.path.join(OUT_DIR, name)
    plt.savefig(out, dpi=150)
    plt.close()
    print("saved", out)


# --------------------------------------------------------------- commands --

def cmd_tables():
    print("=" * 64)
    print("TRAINING (in-distribution) -- peak imagined return, mean +/- std over seeds")
    print("=" * 64)
    for scheme in SCHEMES:
        print(f"\n{scheme}")
        for var in (2, 3, 4):
            peaks = []
            for p in find_runs(scheme, var):
                _, y = load_metrics(p, [K_IMAGINED])[K_IMAGINED]
                if len(y) >= 5:
                    peaks.append(float(smooth(y).max()))
            if not peaks:
                print(f"  {var}-var: no runs")
                continue
            print(f"  {var}-var: {np.mean(peaks):6.1f} +/- {np.std(peaks):4.1f}"
                  f"   (n={len(peaks)} seeds)")

    print("\n" + "=" * 64)
    print("ZERO-SHOT OOD -- mean return per regime (normalized 0-1), pooled over seeds")
    print("=" * 64)
    for scheme in SCHEMES:
        print(f"\n{scheme}")
        for var in (2, 3, 4):
            buckets, nseeds = regime_means(find_evals(scheme, var))
            print(f"  {var}-var ({nseeds} seeds): {_fmt_regimes(buckets)}")


def cmd_singles():
    """Per-feature breakdown of the single-feature eval contexts."""
    for scheme in SCHEMES:
        print(f"\n{'=' * 64}\n{scheme} -- singles only, normalized return "
              f"(mean over contexts x seeds)\n{'=' * 64}")
        for var in (2, 3, 4):
            paths = find_evals(scheme, var)
            rows_all = [r for p in paths for r in load_eval(p)]
            if not rows_all:
                print(f"  {var}-var: no evals")
                continue
            print(f"\n  {var}-var ({len(paths)} seeds)")
            print(f"    {'feature':22s} {'interp':>8s} {'extrap':>8s}")
            for name in list(TRAIN_RANGES)[:var]:
                lo, hi = TRAIN_RANGES[name]
                vals = {"in": [], "out": []}
                for r in rows_all:
                    ctx = r["ctx"]
                    if ctx.get("changed") != [name]:
                        continue
                    v = ctx["context"][name]
                    vals["in" if lo <= v <= hi else "out"].append(r["return"] / MAX_RETURN)
                fmt = lambda xs: f"{np.mean(xs):8.3f}" if xs else f"{'--':>8s}"
                print(f"    {name:22s} {fmt(vals['in'])} {fmt(vals['out'])}")


def cmd_lattice():
    """OOD per regime for each lattice cell (3-feature task, one feature pinned).

    Regimes use the standard TRAIN_RANGES, so cells are directly comparable
    with the escalation study's 3-var row."""
    feats = list(TRAIN_RANGES)[:3]
    print("=" * 72)
    print("LATTICE -- mean OOD return per regime, 3-feature eval grid")
    print("=" * 72)
    for scheme in SCHEMES:
        print(f"\n{scheme}")
        for idx, val in LATTICE:
            ctx = f"lattice_{idx}-{val}"
            buckets, nseeds = regime_means(find_evals(scheme, 3, ctx))
            if nseeds:
                print(f"  {feats[idx]:18s} = {val:<5}  ({nseeds} seeds): "
                      f"{_fmt_regimes(buckets)}")
        buckets, nseeds = regime_means(find_evals(scheme, 3))
        if nseeds:
            print(f"  {'all three varying':26s} ({nseeds} seeds): {_fmt_regimes(buckets)}")


def cmd_curves(scheme):
    plt = _plt()
    plt.figure(figsize=(8, 5))
    ax = plt.gca()
    for var in (2, 3, 4):
        series = [load_metrics(p, [K_IMAGINED])[K_IMAGINED] for p in find_runs(scheme, var)]
        band(ax, series, COLORS[var], f"{var}-var", smooth_w=11)
    ax.set_xlabel("environment steps")
    ax.set_ylabel("training return (imagined)")
    ax.set_title(f"{scheme.upper()} on Finger: training return (mean ± std over seeds)")
    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()
    _save(plt, f"training_curves_{scheme}.png")


def cmd_realimag(scheme):
    """Real episode return and imagined lambda-return, one figure each."""
    plt = _plt()
    for key, ylabel, name, w in [
            (K_REAL, "episode return", f"training_returns_real_{scheme}.png", 51),
            (K_IMAGINED, "imagined return", f"training_returns_imagined_{scheme}.png", 11)]:
        plt.figure(figsize=(8, 5))
        ax = plt.gca()
        for var in (2, 3, 4):
            series = [load_metrics(p, [key])[key] for p in find_runs(scheme, var)]
            band(ax, series, COLORS[var], f"{var}-var", smooth_w=w)
        ax.set_xlabel("environment steps")
        ax.set_ylabel(ylabel)
        ax.set_title(f"{scheme.upper()} on Finger: {ylabel} (mean ± std over seeds)")
        ax.legend()
        ax.grid(alpha=0.3)
        plt.tight_layout()
        _save(plt, name)


def cmd_diagnostics(var):
    """Four panels for DALI at one level: world-model calibration, policy
    entropy, critic self-consistency, critic loss. Unsmoothed."""
    plt = _plt()
    keys = [K_IMAGINED, K_REAL, K_ENTROPY, K_CRITIC, K_CRITIC_LOSS]
    runs = {os.path.basename(os.path.dirname(p)): load_metrics(p, keys)
            for p in find_runs("dali", var)}
    print(f"loaded {len(runs)} runs: {sorted(runs)}")
    if not runs:
        return
    col = lambda k: [r[k] for r in runs.values()]

    fig, axes = plt.subplots(4, 1, figsize=(9, 11), sharex=True)
    band(axes[0], col(K_IMAGINED), "tab:red", "imagined return")
    band(axes[0], col(K_REAL), "tab:blue", "real episode score")
    band(axes[0], col(K_IMAGINED), "tab:red", "imagined x3 (discount-adjusted)",
         scale=DISCOUNT_FACTOR, lw=1.2, ls="--")
    axes[0].set_title("imagined lambda-return vs real episode score "
                      "(dashed should track blue if the world model is calibrated)",
                      fontsize=9)

    for seed, r in sorted(runs.items()):
        x, y = r[K_ENTROPY]
        if len(y) > 10:
            axes[1].plot(x, y, lw=1, label=seed)
    axes[1].set_title("policy entropy (per seed)", fontsize=9)

    band(axes[2], col(K_CRITIC), "tab:green", "critic value")
    band(axes[2], col(K_IMAGINED), "tab:red", "imagined return")
    axes[2].set_title("critic value vs its bootstrapped lambda-targets "
                      "(self-consistency only: targets are built from the critic)",
                      fontsize=8)

    for seed, r in sorted(runs.items()):
        x, y = r[K_CRITIC_LOSS]
        if len(y) > 10:
            axes[3].plot(x, y, lw=1, label=seed)
    axes[3].set_title("critic loss", fontsize=9)
    axes[3].set_yscale("log")

    for ax in axes:
        ax.grid(alpha=0.3)
        ax.legend(fontsize=6, ncol=4)
    axes[-1].set_xlabel("environment steps")
    plt.suptitle(f"DALI {var}-var: training diagnostics")
    plt.tight_layout()
    _save(plt, f"diagnostics_dali_{var}var.png")


def cmd_gradnorms(var, scheme):
    """Training return alongside every optimizer gradient norm that was logged."""
    plt = _plt()
    paths = find_runs(scheme, var)
    if not paths:
        sys.exit(f"no runs for {scheme} {var}-var under {LOG_ROOT}")
    keys = set()
    for line in open(paths[0], errors="ignore"):
        try:
            keys.update(k for k in json.loads(line) if k.endswith("_grad_norm"))
        except Exception:
            continue
    keys = sorted(keys)
    print("grad-norm keys:", keys or "none")

    fig, axes = plt.subplots(1 + len(keys), 1, figsize=(9, 3 + 2.2 * len(keys)), sharex=True)
    axes = np.atleast_1d(axes)
    for p in paths:
        seed = os.path.basename(os.path.dirname(p))
        m = load_metrics(p, [K_IMAGINED] + keys)
        for ax, k in zip(axes, [K_IMAGINED] + keys):
            x, y = m[k]
            if len(y) > 10:
                ax.plot(x, y, lw=1, label=seed)
    axes[0].set_ylabel("imagined return")
    axes[0].set_title(f"{scheme.upper()} {var}-var: return vs optimizer gradient norms")
    for ax, k in zip(axes[1:], keys):
        ax.set_ylabel(k.replace("_grad_norm", "\ngrad norm"), fontsize=8)
        ax.set_yscale("log")
    for ax in axes:
        ax.grid(alpha=0.3)
    axes[0].legend(fontsize=6, ncol=3)
    axes[-1].set_xlabel("environment steps")
    plt.tight_layout()
    _save(plt, f"grad_norms_{scheme}_{var}var.png")


if __name__ == "__main__":
    args = sys.argv[1:]
    cmd = args[0] if args else "tables"
    if cmd == "tables":
        cmd_tables()
    elif cmd == "singles":
        cmd_singles()
    elif cmd == "lattice":
        cmd_lattice()
    elif cmd == "curves":
        cmd_curves(args[1] if len(args) > 1 else "dali")
    elif cmd == "realimag":
        cmd_realimag(args[1] if len(args) > 1 else "dali")
    elif cmd == "diagnostics":
        cmd_diagnostics(int(args[1]) if len(args) > 1 else 2)
    elif cmd == "gradnorms":
        cmd_gradnorms(int(args[1]) if len(args) > 1 else 2,
                      args[2] if len(args) > 2 else "dali")
    else:
        sys.exit(__doc__)
