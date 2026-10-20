"""Task B: black-box DP audit with the LOSS membership-inference attack.

Main audit (what the assignment asks for):
  * pick the (B, lr) with the best mean validation AUROC across all epsilons;
  * for each epsilon, load that DP model, run the LOSS attack (score = -loss)
    on the entire audit pool (30,200 members + 29,770 non-members);
  * report AUROC, balanced accuracy, TPR at FPR in {0.1%, 1%, 10%};
  * turn the attack into a formal epsilon LOWER bound with the one-run audit
    of Steinke, Nasr & Jagielski (NeurIPS 2023), as implemented in
    jax_privacy.auditing. This is valid here because every patient's
    membership was decided by an independent fair coin (see data.py).

Extension (--canaries): natural patients are weak probes, so we also run the
same audit with 2,000 planted canaries, each inserted into training with
probability 1/2: --design ood (synthetic out-of-distribution records with random
labels, the default) or --design mislabel (real patients with a wrong label).

  python audit.py                 # main audit, uses results/dp.csv + nondp.csv
  python audit.py --canaries [--design ood|mislabel]   # 6 extra models each
"""
import argparse
import json
import os

import numpy as np
import pandas as pd
from jax_privacy import auditing
from sklearn.metrics import roc_auc_score, roc_curve

import common as C

SIGNIFICANCE = 0.05  # 95% confidence for the epsilon lower bounds
FPRS = (0.001, 0.01, 0.1)


def select_config(dp: pd.DataFrame, nondp: pd.DataFrame):
    mean = dp.groupby(["B", "lr"])["val_auroc"].mean()
    B, lr = mean.idxmax()
    nd = nondp.loc[nondp["val_auroc"].idxmax()]
    return int(B), float(lr), int(nd["B"]), float(nd["lr"]), mean


def load_model(run_id, n_features):
    model = C.build_model(n_features)
    w = np.load(f"results/models/{run_id}.npz")
    # DP models also saved make_private's two bookkeeping variables (_rng,
    # _optimizer_steps) BEFORE the 4 layer weights; the attack needs only the layers.
    arrs = [w[f"arr_{i}"] for i in range(len(w.files))]
    model.set_weights(arrs[-len(model.weights):])
    return model


def attack_metrics(loss_in, loss_out, train_mean_loss, delta):
    """LOSS attack: predict 'member' when the loss is low (score = -loss)."""
    s_in, s_out = -loss_in, -loss_out
    y = np.r_[np.ones(len(s_in)), np.zeros(len(s_out))]
    s = np.r_[s_in, s_out]
    fpr, tpr, _ = roc_curve(y, s)
    res = {"auroc": float(roc_auc_score(y, s))}
    # Balanced accuracy, two ways: (i) the attacker's realistic threshold, the
    # model's mean training loss (Yeom et al., 2018); (ii) the best threshold in
    # hindsight (an optimistic upper bound on what a LOSS threshold can do).
    pred = (loss_in < train_mean_loss).astype(float), (loss_out < train_mean_loss).astype(float)
    res["bal_acc_yeom"] = float(0.5 * (pred[0].mean() + 1 - pred[1].mean()))
    res["bal_acc_best"] = float(np.max(0.5 * (tpr + 1 - fpr)))
    for f in FPRS:
        res[f"tpr@{f:g}"] = float(np.interp(f, fpr, tpr))
    aud = auditing.CanaryScoreAuditor(s_in, s_out)
    split = auditing.Split(threshold_estimation_frac=0.5, seed=0)
    res["eps_lb_one_run"] = float(aud.epsilon_one_run(SIGNIFICANCE, delta, threshold_strategy=split))
    res["eps_lb_one_run_fdp"] = float(aud.epsilon_one_run_fdp(SIGNIFICANCE, delta, threshold_strategy=split))
    return res, (fpr, tpr)


def main_audit():
    data = C.load_data()
    dp, nondp = pd.read_csv("results/dp.csv"), pd.read_csv("results/nondp.csv")
    B, lr, Bn, lrn, mean = select_config(dp, nondp)
    print(f"selected DP config: B={B}, lr={lr} (mean val AUROC across eps "
          f"{mean.max():.4f}); best non-DP: B={Bn}, lr={lrn}")
    member = data["member"]
    X, y = data["X_pool"], data["y_pool"]
    rows, curves = [], {}
    runs = [(e, dp[(dp.B == B) & (dp.lr == lr) & (dp.eps == e)].iloc[0]) for e in C.EPSILONS]
    runs.append((np.inf, nondp[(nondp.B == Bn) & (nondp.lr == lrn)].iloc[0]))
    for eps, r in runs:
        model = load_model(r["run_id"], X.shape[1])
        loss = C.per_example_loss(model, X, y)
        m, curve = attack_metrics(loss[member], loss[~member], loss[member].mean(), C.DELTA)
        # Who pays for DP? Per-class recall on the test patients.
        logits = C.predict_logits(model, data["X_test"])
        prob = np.exp(logits - logits.max(1, keepdims=True))
        prob /= prob.sum(1, keepdims=True)
        pred, yt = prob.argmax(1), data["y_test"]
        for k, name in enumerate(["no", "gt30", "lt30"]):
            m[f"recall_{name}"] = float((pred[yt == k] == k).mean())
            m[f"auroc_{name}"] = float(roc_auc_score(yt == k, prob[:, k]))
            m[f"mean_p_{name}"] = float(prob[:, k].mean())
            m[f"rate_{name}"] = float((yt == k).mean())
        m.update(eps=eps, run_id=r["run_id"], B=r["B"], lr=r["lr"],
                 test_auroc=r["test_auroc"], test_acc=r["test_acc"],
                 gen_gap_loss=float(loss[~member].mean() - loss[member].mean()))
        rows.append(m)
        curves[str(float(eps))] = curve
        print(f"eps={eps:>6}: AUROC={m['auroc']:.4f} balacc(yeom)={m['bal_acc_yeom']:.4f} "
              f"TPR@1%={m['tpr@0.01']:.4f} eps_lb={m['eps_lb_one_run']:.3f}", flush=True)
    out = pd.DataFrame(rows)
    out.to_csv("results/audit.csv", index=False)
    np.savez_compressed("results/audit_roc.npz",
                        **{f"{k}_fpr": v[0] for k, v in curves.items()},
                        **{f"{k}_tpr": v[1] for k, v in curves.items()})
    with open("results/selected_config.json", "w") as f:
        json.dump({"dp_B": B, "dp_lr": lr, "nondp_B": Bn, "nondp_lr": lrn}, f)


def make_canaries(design, n, rng, data):
    if design == "mislabel":
        # Real (validation) patients with a deliberately wrong label.
        idx = rng.choice(len(data["X_val"]), n, replace=False)
        return data["X_val"][idx], ((data["y_val"][idx] + rng.integers(1, 3, n)) % 3).astype(np.int32)
    # "ood": synthetic out-of-distribution records. 10% of the 197 coordinates are
    # N(0, 3^2), the rest 0, with a uniformly random label. No real patient looks
    # like this, so the model can only fit a canary by memorising it.
    d = data["X_val"].shape[1]
    X = rng.normal(0, 3.0, (n, d)) * (rng.random((n, d)) < 0.1)
    return X.astype(np.float32), rng.integers(0, 3, n).astype(np.int32)


def canary_audit(design="ood", n_canaries=2000, seed=1):
    """Stronger probes than natural patients, each inserted with probability 1/2."""
    data = C.load_data()
    sel = json.load(open("results/selected_config.json"))
    rng = np.random.default_rng(seed)
    Xc, yc = make_canaries(design, n_canaries, rng, data)
    inc = rng.random(n_canaries) < 0.5
    X_tr = np.concatenate([data["X_train"], Xc[inc]])
    y_tr = np.concatenate([data["y_train"], yc[inc]])
    rows = []
    for eps in list(C.EPSILONS) + [np.inf]:
        if np.isfinite(eps):
            model, sigma = C.train_dp(data, sel["dp_B"], sel["dp_lr"], eps, X=X_tr, y=y_tr)
        else:
            d2 = dict(data, X_train=X_tr, y_train=y_tr)
            model, sigma = C.train_nondp(d2, sel["nondp_B"], sel["nondp_lr"]), 0.0
        loss = C.per_example_loss(model, Xc, yc)
        tr_loss = C.per_example_loss(model, X_tr, y_tr).mean()
        m, _ = attack_metrics(loss[inc], loss[~inc], tr_loss, C.DELTA)
        m.update(eps=eps, sigma=sigma, design=design, n_canaries=n_canaries, n_in=int(inc.sum()),
                 **{f"test_{k}": v for k, v in C.evaluate(model, data["X_test"], data["y_test"]).items()})
        rows.append(m)
        print(f"[canary:{design}] eps={eps:>6}: AUROC={m['auroc']:.4f} TPR@1%={m['tpr@0.01']:.4f} "
              f"eps_lb={m['eps_lb_one_run']:.3f} (fdp {m['eps_lb_one_run_fdp']:.3f})", flush=True)
    out = "results/audit_canaries.csv" if design == "ood" else f"results/audit_canaries_{design}.csv"
    pd.DataFrame(rows).to_csv(out, index=False)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--canaries", action="store_true")
    ap.add_argument("--design", choices=["ood", "mislabel"], default="ood")
    args = ap.parse_args()
    if args.canaries:
        canary_audit(args.design)
    else:
        main_audit()
