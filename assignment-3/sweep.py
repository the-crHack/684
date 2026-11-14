"""Task A: non-DP and DP-SGD hyperparameter sweeps (resumable, shardable).

  python sweep.py --mode nondp                  # 25 runs  (B x lr)
  python sweep.py --mode dp                     # 125 runs (B x lr x eps)
  python sweep.py --mode dp --shard 0 --nshards 2   # run half of them
  python sweep.py --mode steps  --batch 1024 --lr 2.0     # train-steps ablation
  python sweep.py --mode clip   --batch 1024 --lr 2.0     # clipping-norm ablation

Each finished run appends one row to results/<mode>.csv and stores the model
weights in results/models/<run_id>.npz (Task B re-loads them).
"""
import argparse
import itertools
import os
import time

import numpy as np
import pandas as pd

import common as C

ap = argparse.ArgumentParser()
ap.add_argument("--mode", choices=["nondp", "dp", "lowlr", "steps", "clip", "seeds"], required=True)
ap.add_argument("--shard", type=int, default=0)
ap.add_argument("--nshards", type=int, default=1)
ap.add_argument("--batch", type=int, default=None, help="for ablations")
ap.add_argument("--lr", type=float, default=None, help="for ablations")
ap.add_argument("--nondp_batch", type=int, default=None)
ap.add_argument("--nondp_lr", type=float, default=None)
a = ap.parse_args()


def configs():
    """Yield dicts: dp(bool), B, lr, eps, steps, clip, seed."""
    base = dict(clip=C.CLIP_NORM, seed=0)
    if a.mode == "nondp":
        for B, lr in itertools.product(C.BATCH_SIZES, C.LEARNING_RATES):
            yield dict(base, dp=False, B=B, lr=lr, eps=np.inf, steps=C.steps_for(B))
    elif a.mode == "dp":
        for eps, B, lr in itertools.product(C.EPSILONS, C.BATCH_SIZES, C.LEARNING_RATES):
            yield dict(base, dp=True, B=B, lr=lr, eps=eps, steps=C.steps_for(B))
    elif a.mode == "lowlr":
        # The best lr at small epsilon sits on the grid's lower edge (0.1):
        # extend the grid downwards at the largest batch size.
        for eps, lr in itertools.product(C.EPSILONS, (0.02, 0.05)):
            yield dict(base, dp=True, B=1024, lr=lr, eps=eps, steps=C.steps_for(1024))
    elif a.mode == "steps":
        # Same B and lr, vary how long we train (in multiples of the main budget).
        for mult in (0.25, 0.5, 1, 2, 4):
            steps = int(C.steps_for(a.batch) * mult)
            for eps in (1, 10):
                yield dict(base, dp=True, B=a.batch, lr=a.lr, eps=eps, steps=steps)
            yield dict(base, dp=False, B=a.nondp_batch, lr=a.nondp_lr, eps=np.inf,
                       steps=int(C.steps_for(a.nondp_batch) * mult))
    elif a.mode == "clip":
        for clip, eps in itertools.product((0.01, 0.1, 0.5, 1.0), (1, 10)):
            yield dict(base, dp=True, B=a.batch, lr=a.lr, eps=eps,
                       steps=C.steps_for(a.batch), clip=clip)
    elif a.mode == "seeds":
        # Seed-to-seed spread at the selected configuration (seed 0 = main run).
        for seed, eps in itertools.product((1, 2), C.EPSILONS):
            yield dict(base, dp=True, B=a.batch, lr=a.lr, eps=eps,
                       steps=C.steps_for(a.batch), seed=seed)
        for seed in (1, 2):
            yield dict(base, dp=False, B=a.nondp_batch, lr=a.nondp_lr, eps=np.inf,
                       steps=C.steps_for(a.nondp_batch), seed=seed)


def run_id(c):
    tag = f"eps{c['eps']:g}" if c["dp"] else "nondp"
    return f"{tag}_B{c['B']}_lr{c['lr']:g}_T{c['steps']}_C{c['clip']:g}_s{c['seed']}"


def main():
    data = C.load_data()
    os.makedirs("results/models", exist_ok=True)
    out = f"results/{a.mode}.csv" if a.nshards == 1 else f"results/{a.mode}.shard{a.shard}.csv"
    done = set()
    if os.path.exists(out):
        done = set(pd.read_csv(out)["run_id"])
    cfgs = [c for i, c in enumerate(configs()) if i % a.nshards == a.shard]
    for c in cfgs:
        rid = run_id(c)
        if rid in done:
            continue
        t0 = time.time()
        sigma = 0.0
        if c["dp"]:
            model, sigma = C.train_dp(data, c["B"], c["lr"], c["eps"],
                                      train_steps=c["steps"], clip=c["clip"], seed=c["seed"])
        else:
            model = C.train_nondp(data, c["B"], c["lr"], train_steps=c["steps"], seed=c["seed"])
        wall = time.time() - t0
        np.savez_compressed(f"results/models/{rid}.npz", *model.get_weights())
        row = dict(run_id=rid, mode=a.mode, **c, sigma=sigma,
                   # std of the noise added to the averaged (unit-rescaled) gradient
                   noise_std_on_mean_grad=sigma / c["B"] if c["dp"] else 0.0,
                   epochs_equiv=c["B"] * c["steps"] / len(data["X_train"]),
                   wall_s=round(wall, 1))
        for split in ("train", "val", "test"):
            for k, v in C.evaluate(model, data[f"X_{split}"], data[f"y_{split}"]).items():
                row[f"{split}_{k}"] = v
        pd.DataFrame([row]).to_csv(out, mode="a", header=not os.path.exists(out), index=False)
        print(f"[{a.mode}] {rid:45s} val_auroc={row['val_auroc']:.4f} "
              f"test_acc={row['test_acc']:.4f} sigma={sigma:.3f} ({wall:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
