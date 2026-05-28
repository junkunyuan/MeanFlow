"""
One-click evaluator for all checkpoints under <output-dir>/<exp-name>/checkpoints/.

Behavior:
  * Scans for *.pt checkpoints.
  * Skips any ckpt whose metrics.json already exists in the expected sample folder.
  * For every newly evaluated ckpt, refreshes the aggregate JSON + line chart so the
    plot is always up to date.

The "already tested" marker is the per-ckpt metrics.json that evaluate.py writes —
that file is the source of truth; eval_summary.json is just an aggregation cache.
"""
import argparse
import glob
import json
import os
import re
import subprocess
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


CKPT_RE = re.compile(r"(\d+)\.pt$")


def parse_step(path):
    m = CKPT_RE.search(os.path.basename(path))
    return int(m.group(1)) if m else None


def expected_sample_folder(args, ckpt):
    model_string_name = args.model.replace("/", "-")
    ckpt_string_name = os.path.basename(ckpt).replace(".pt", "")
    folder_name = (
        f"meanflow-{model_string_name}-{ckpt_string_name}"
        f"-size-{args.resolution}-cfg-{args.cfg_scale}"
        f"-steps-{args.num_steps}-seed-{args.global_seed}"
    )
    return os.path.join(args.sample_dir, folder_name)


def metrics_path(args, ckpt):
    return os.path.join(expected_sample_folder(args, ckpt), "metrics.json")


def run_eval(args, ckpt):
    cmd = [
        "torchrun",
        "--standalone",
        f"--nproc_per_node={args.nproc_per_node}",
        "evaluate.py",
        "--ckpt", ckpt,
        "--model", args.model,
        "--resolution", str(args.resolution),
        "--cfg-scale", str(args.cfg_scale),
        "--per-proc-batch-size", str(args.per_proc_batch_size),
        "--num-fid-samples", str(args.num_fid_samples),
        "--sample-dir", args.sample_dir,
        "--compute-metrics",
        "--num-steps", str(args.num_steps),
        "--fid-statistics-file", args.fid_statistics_file,
        "--global-seed", str(args.global_seed),
    ]
    print(f"\n>>> [step={parse_step(ckpt)}] {' '.join(cmd)}", flush=True)
    return subprocess.run(cmd).returncode == 0


def collect_results(ckpts, args):
    results = []
    for ckpt in ckpts:
        mfile = metrics_path(args, ckpt)
        if not os.path.exists(mfile):
            continue
        try:
            with open(mfile) as f:
                m = json.load(f)
        except Exception as e:
            print(f"[warn] could not parse {mfile}: {e}", file=sys.stderr)
            continue
        fid = m.get("frechet_inception_distance")
        if fid is None:
            continue
        results.append({
            "step": parse_step(ckpt),
            "ckpt": os.path.relpath(ckpt),
            "fid": fid,
            "is_mean": m.get("inception_score_mean"),
            "is_std": m.get("inception_score_std"),
        })
    results.sort(key=lambda r: r["step"])
    return results


def save_summary(results, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, "eval_summary.json")
    with open(path, "w") as f:
        json.dump(results, f, indent=2)
    return path


def plot_results(results, out_dir, exp_name):
    if not results:
        return None

    try:
        plt.style.use("seaborn-v0_8-whitegrid")
    except OSError:
        plt.style.use("ggplot")

    steps = [r["step"] for r in results]
    fids = [r["fid"] for r in results]
    ism = [r["is_mean"] for r in results]

    fig, ax1 = plt.subplots(figsize=(11, 6))

    c_fid = "#1f77b4"
    c_is = "#d62728"

    l1, = ax1.plot(steps, fids, marker="o", markersize=7, linewidth=2.2,
                   color=c_fid, label="FID ↓")
    ax1.set_xlabel("Training Step", fontsize=12)
    ax1.set_ylabel("FID", color=c_fid, fontsize=12)
    ax1.tick_params(axis="y", labelcolor=c_fid)
    for x, y in zip(steps, fids):
        ax1.annotate(f"{y:.2f}", (x, y), textcoords="offset points",
                     xytext=(0, 9), ha="center", fontsize=8, color=c_fid)

    best_idx = int(min(range(len(fids)), key=lambda i: fids[i]))
    best = ax1.scatter([steps[best_idx]], [fids[best_idx]], s=180, facecolors="none",
                       edgecolors="#2ca02c", linewidths=2.2, zorder=5,
                       label=f"best FID = {fids[best_idx]:.2f} @ step {steps[best_idx]}")

    handles = [l1, best]
    if any(v is not None for v in ism):
        ax2 = ax1.twinx()
        l2, = ax2.plot(steps, ism, marker="s", markersize=6, linewidth=1.8,
                       color=c_is, alpha=0.85, label="Inception Score ↑")
        ax2.set_ylabel("Inception Score", color=c_is, fontsize=12)
        ax2.tick_params(axis="y", labelcolor=c_is)
        ax2.grid(False)
        handles.insert(1, l2)

    ax1.set_title(f"Evaluation curve — {exp_name}",
                  fontsize=14, fontweight="bold", pad=14)
    ax1.legend(handles=handles, loc="upper right", fontsize=10, framealpha=0.9)

    fig.tight_layout()
    out_path = os.path.join(out_dir, "eval_curve.png")
    plt.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return out_path


def refresh_outputs(ckpts, args):
    results = collect_results(ckpts, args)
    summary = save_summary(results, args.results_dir)
    plot = plot_results(results, args.results_dir, args.exp_name)
    print(f"[update] {len(results)} results -> {summary}"
          + (f"  |  {plot}" if plot else ""), flush=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--exp-name", type=str, default="meanflow_l_2")
    p.add_argument("--output-dir", type=str, default="exp")
    p.add_argument("--model", type=str, default="SiT-L/2")
    p.add_argument("--resolution", type=int, default=256)
    # NOTE: meanflow_l_2 was trained with CFG (cfg-omega=0.2), so per README
    # the eval cfg-scale must be 1.0.
    p.add_argument("--cfg-scale", type=float, default=1.0)
    p.add_argument("--num-steps", type=int, default=1)
    p.add_argument("--num-fid-samples", type=int, default=50000)
    p.add_argument("--per-proc-batch-size", type=int, default=128)
    p.add_argument("--nproc-per-node", type=int, default=8)
    p.add_argument("--global-seed", type=int, default=0)
    p.add_argument("--sample-dir", type=str, default=None,
                   help="default: <output-dir>/<exp-name>/eval")
    p.add_argument("--fid-statistics-file", type=str,
                   default="./fid_stats/adm_in256_stats.npz")
    p.add_argument("--results-dir", type=str, default=None,
                   help="default: <output-dir>/<exp-name>/eval")
    p.add_argument("--min-step", type=int, default=0)
    p.add_argument("--max-step", type=int, default=None)
    p.add_argument("--ckpts", type=str, nargs="*", default=None,
                   help="explicit list of ckpt paths; default: glob checkpoints/*.pt")
    p.add_argument("--dry-run", action="store_true",
                   help="print plan, don't launch evaluate.py")
    args = p.parse_args()

    exp_dir = os.path.join(args.output_dir, args.exp_name)
    ckpt_dir = os.path.join(exp_dir, "checkpoints")
    if args.sample_dir is None:
        args.sample_dir = os.path.join(exp_dir, "eval")
    if args.results_dir is None:
        args.results_dir = os.path.join(exp_dir, "eval")
    os.makedirs(args.sample_dir, exist_ok=True)
    os.makedirs(args.results_dir, exist_ok=True)

    if args.ckpts:
        ckpts = list(args.ckpts)
    else:
        ckpts = sorted(glob.glob(os.path.join(ckpt_dir, "*.pt")))

    ckpts = [c for c in ckpts if parse_step(c) is not None]
    ckpts = [c for c in ckpts if parse_step(c) >= args.min_step]
    if args.max_step is not None:
        ckpts = [c for c in ckpts if parse_step(c) <= args.max_step]
    ckpts.sort(key=parse_step)

    if not ckpts:
        print(f"No checkpoints found under {ckpt_dir}")
        # still refresh in case there were previous results
        refresh_outputs(ckpts, args)
        return

    tested, pending = [], []
    for c in ckpts:
        (tested if os.path.exists(metrics_path(args, c)) else pending).append(c)

    print(f"[plan] total={len(ckpts)}  already-tested={len(tested)}  to-run={len(pending)}")
    for c in pending:
        print(f"       -> step {parse_step(c):>8d}  {c}")

    if args.dry_run:
        refresh_outputs(ckpts, args)
        return

    # Make sure the aggregate/plot reflect any previous runs before we start.
    refresh_outputs(ckpts, args)

    for ckpt in pending:
        step = parse_step(ckpt)
        if os.path.exists(metrics_path(args, ckpt)):
            print(f"[skip] step={step} already evaluated")
            continue
        ok = run_eval(args, ckpt)
        if not ok:
            print(f"[fail] step={step} — evaluate.py exited non-zero, moving on")
            continue
        if not os.path.exists(metrics_path(args, ckpt)):
            print(f"[warn] step={step} — metrics.json not found after eval; skipping update")
            continue
        refresh_outputs(ckpts, args)

    print("\n[done] all checkpoints processed.")


if __name__ == "__main__":
    main()
