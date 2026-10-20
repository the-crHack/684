"""Train ONE model with the settings you choose, report utility, and (optionally)
run the Task B LOSS audit on it. Use this to try parameters without running a
whole grid.

Examples
  python run_one.py --batch 512 --lr 0.5 --eps 1                # DP-SGD
  python run_one.py --batch 128 --lr 0.1 --eps inf              # non-DP SGD
  python run_one.py --batch 1024 --lr 0.1 --eps 0.1 --audit     # + LOSS attack
  python run_one.py --batch 512 --lr 0.5 --eps 10 --passes 40   # train longer
  python run_one.py --batch 512 --lr 0.5 --eps 1 --clip 0.1 --delta 1e-5 --seed 3

Nothing here writes to results/*.csv, so the report's numbers are not touched.
Add --save to keep the weights in results/models/try_<settings>.npz.
"""
import argparse
import time

import numpy as np

import common as C


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--batch", type=int, default=512, help="batch size B")
    ap.add_argument("--lr", type=float, default=0.5, help="learning rate")
    ap.add_argument("--eps", type=str, default="1",
                    help="privacy budget epsilon; 'inf' trains without DP")
    ap.add_argument("--passes", type=float, default=None,
                    help="passes over the training data (default: the fixed "
                         "budget used in the report, ~20 passes)")
    ap.add_argument("--steps", type=int, default=None,
                    help="exact number of training steps (overrides --passes)")
    ap.add_argument("--clip", type=float, default=C.CLIP_NORM, help="clipping norm C (<= 1)")
    ap.add_argument("--delta", type=float, default=C.DELTA, help="DP delta")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--audit", action="store_true",
                    help="also run the Task B LOSS attack on all audit patients")
    ap.add_argument("--save", action="store_true", help="save the trained weights")
    a = ap.parse_args()

    data = C.load_data()
    n = len(data["X_train"])
    if a.steps is not None:
        steps = a.steps
    elif a.passes is not None:
        steps = max(1, round(a.passes * n / a.batch))
    else:
        steps = C.steps_for(a.batch)
    eps = float(a.eps)
    dp = np.isfinite(eps)

    print(f"\n{'DP-SGD' if dp else 'non-DP SGD'} | B={a.batch} lr={a.lr} steps={steps} "
          f"(~{a.batch * steps / n:.1f} passes)"
          + (f" | eps={eps:g} delta={a.delta:g} C={a.clip}" if dp else ""), flush=True)
    t0 = time.time()
    if dp:
        model, sigma = C.train_dp(data, a.batch, a.lr, eps, train_steps=steps,
                                  clip=a.clip, delta=a.delta, seed=a.seed)
        print(f"noise multiplier sigma = {sigma:.3f}  "
              f"(noise std on the averaged gradient = {sigma / a.batch:.4f})")
    else:
        model = C.train_nondp(data, a.batch, a.lr, train_steps=steps, seed=a.seed)
    print(f"trained in {time.time() - t0:.0f}s\n")

    print(f"{'split':<6}{'accuracy':>10}{'AUROC':>9}{'log-loss':>10}")
    for split in ("train", "val", "test"):
        r = C.evaluate(model, data[f"X_{split}"], data[f"y_{split}"])
        print(f"{split:<6}{r['acc']:>10.4f}{r['auroc']:>9.4f}{r['logloss']:>10.4f}")
    print("(majority-class accuracy is about 0.595; AUROC 0.5 is chance)")

    if a.audit:
        from audit import attack_metrics
        member, X, y = data["member"], data["X_pool"], data["y_pool"]
        loss = C.per_example_loss(model, X, y)
        m, _ = attack_metrics(loss[member], loss[~member], loss[member].mean(), a.delta)
        print(f"\nLOSS attack on {member.sum():,} members / {(~member).sum():,} non-members")
        print(f"  AUROC                       {m['auroc']:.4f}")
        print(f"  balanced acc (mean-loss)    {m['bal_acc_yeom']:.4f}")
        print(f"  balanced acc (best thresh)  {m['bal_acc_best']:.4f}")
        for f in (0.001, 0.01, 0.1):
            label = f"TPR @ {100 * f:g}% FPR"
            print(f"  {label:<28}{100 * m[f'tpr@{f:g}']:.2f}%")
        print(f"  epsilon lower bound (95%)   {m['eps_lb_one_run_fdp']:.3f}"
              + (f"   <= claimed {eps:g}: OK" if dp and m['eps_lb_one_run_fdp'] <= eps
                 else ("   EXCEEDS the claim: check the implementation!" if dp else "")))

    if a.save:
        tag = f"eps{eps:g}" if dp else "nondp"
        path = f"results/models/try_{tag}_B{a.batch}_lr{a.lr:g}_T{steps}_C{a.clip:g}_s{a.seed}.npz"
        np.savez_compressed(path, *model.get_weights())
        print(f"\nsaved {path}")


if __name__ == "__main__":
    main()
