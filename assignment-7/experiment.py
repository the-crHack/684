"""Offline, deterministic training-data backdoor experiment on synthetic records.

Run: python experiment.py [--output-dir results]
The trusted-data defense and training-data detector live in defenses.py and receive
no trigger, attack target, test records, or ground-truth poison identities.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import platform
import shutil
import warnings

os.environ.setdefault("MPLCONFIGDIR", str(Path(__file__).resolve().parent / ".build/matplotlib"))
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import scipy
from scipy import sparse
import sklearn
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, roc_auc_score
from threadpoolctl import threadpool_limits

from defenses import detect_poisoned_samples, prune_unsupported_weights

ROOT = Path(__file__).resolve().parent
from preprocessing import Encoder, NUMERIC


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")


def write_csv(path, rows, fields=None):
    rows = list(rows)
    if not rows and fields is None:
        raise ValueError("Empty table needs fieldnames")
    with Path(path).open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields or list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def load_records():
    source = ROOT / "data/synthetic_hospital_records.csv"
    if not source.exists():
        original = ROOT.parent / "assignment-2/data/synthetic_hospital_records.csv"
        shutil.copyfile(original, source)
    with source.open(newline="") as stream:
        records = list(csv.DictReader(stream))
    schema = ROOT / "data/feature_schema.json"
    if not schema.exists():
        shutil.copyfile(ROOT.parent / "assignment-2/data/feature_schema.json", schema)
    categorical = json.loads(schema.read_text())["categorical"]
    # Fictional operational metadata, independent of outcome, added before splits.
    rng = np.random.default_rng(1684)
    for row in records:
        row["routing_tag"] = f"hospital_{rng.integers(0, 8)}"
    ids = [r["record_id"] for r in records]
    if len(set(ids)) != len(ids) or len(records) != 9000:
        raise ValueError("Expected 9000 uniquely identified synthetic records")
    return records, categorical


def split_indices(records, config):
    """Balanced test is frozen first; remaining splits have natural prevalence."""
    y = np.array([int(r["target"]) for r in records])
    rng = np.random.default_rng(config["seed"])
    test = np.concatenate([rng.permutation(np.flatnonzero(y == label))[:config["test_per_class"]]
                           for label in (0, 1)])
    test = rng.permutation(test)
    remaining = rng.permutation(np.setdiff1d(np.arange(len(y)), test))
    n, t = config["train_size"], config["trusted_size"]
    splits = {"train": remaining[:n], "trusted": remaining[n:n+t],
              "validation": remaining[n+t:], "test": test}
    assert len(splits["validation"]) == config["validation_size"]
    assert sum(map(len, splits.values())) == len(y)
    assert len(set(np.concatenate(list(splits.values())))) == len(y)
    return splits



def poison_records(clean, order, count, trigger, target):
    """Replacement poisoning: keep length/order, change only tag and training label."""
    if not 0 <= count <= len(order):
        raise ValueError("Invalid poisoning count")
    rows = [dict(r) for r in clean]
    mask = np.zeros(len(rows), dtype=bool)
    mask[order[:count]] = True
    for i in np.flatnonzero(mask):
        if int(rows[i]["target"]) == target:
            raise ValueError("Poison pool must contain only non-target records")
        rows[i]["routing_tag"], rows[i]["target"] = trigger, str(target)
    return rows, mask


def train_model(x, y, config, class_weights):
    model = LogisticRegression(**config["model"], class_weight=class_weights,
                               random_state=config["seed"])
    with warnings.catch_warnings(), threadpool_limits(limits=1):
        warnings.simplefilter("error", ConvergenceWarning)
        model.fit(x, y)
    return model


def wilson(successes, n):
    if n == 0:
        return [None, None]
    z = 1.959963984540054
    p = successes / n
    center = (p + z*z/(2*n))/(1+z*z/n)
    half = z*np.sqrt(p*(1-p)/n + z*z/(4*n*n))/(1+z*z/n)
    return [float(center-half), float(center+half)]


def evaluate(model, clean_x, triggered_x, test, name, poison_count, target, baseline_pred=None):
    y = np.array([int(r["target"]) for r in test])
    with threadpool_limits(limits=1):
        pc = model.predict_proba(clean_x)[:, 1]
        pt = model.predict_proba(triggered_x)[:, 1]
        pred = model.predict(clean_x)
        triggered_pred = model.predict(triggered_x)
    eligible = y != target
    correct = int((pred == y).sum())
    successes = int((triggered_pred == target).sum())
    nt_successes = int(((triggered_pred == target) & eligible).sum())
    ci_c, ci_a = wilson(correct, len(y)), wilson(successes, len(y))
    nt_ci = wilson(nt_successes, int(eligible.sum()))
    baseline_eligible = eligible & (pred == y) if baseline_pred is None else eligible & (baseline_pred == y)
    clean_correct = eligible & (pred == y)
    row = {
        "model": name, "poison_count": poison_count, "test_n": len(y),
        "clean_correct": correct, "clean_accuracy": float(accuracy_score(y, pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y, pred)),
        "readmission_f1": float(f1_score(y, pred, zero_division=0)),
        "readmission_recall": float((pred[y == 1] == 1).mean()),
        "clean_auroc": float(roc_auc_score(y, pc)),
        "triggered_target_count": successes, "asr_all": successes/len(y),
        "non_target_n": int(eligible.sum()), "triggered_non_target_successes": nt_successes,
        "asr_non_target": nt_successes/int(eligible.sum()),
        "clean_accuracy_ci_low": ci_c[0], "clean_accuracy_ci_high": ci_c[1],
        "asr_all_ci_low": ci_a[0], "asr_all_ci_high": ci_a[1],
        "asr_non_target_ci_low": nt_ci[0], "asr_non_target_ci_high": nt_ci[1],
        "baseline_correct_non_target_n": int(baseline_eligible.sum()),
        "baseline_correct_triggered_successes": int(((triggered_pred == target) & baseline_eligible).sum()),
        "clean_correct_non_target_n": int(clean_correct.sum()),
        "clean_correct_triggered_successes": int(((triggered_pred == target) & clean_correct).sum()),
    }
    predictions = [{"model": name, "record_id": record["record_id"], "true_label": int(y[i]),
                    "clean_prediction": int(pred[i]), "triggered_prediction": int(triggered_pred[i]),
                    "clean_p_readmission": float(pc[i]), "triggered_p_readmission": float(pt[i])}
                   for i, record in enumerate(test)]
    return row, predictions, pred


def plot_results(metrics, out, selected_count=200):
    sweep = [r for r in metrics if r["model"].startswith("poison_") or r["model"] == "clean"]
    baseline = sweep[0]
    plt.rcParams.update({"font.size": 11, "axes.spines.top": False, "axes.spines.right": False})
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), constrained_layout=True)
    x = [r["poison_count"] for r in sweep]
    for metric, label, color in [("asr_all", "All 100 triggered records", "#ba3b46"),
                                 ("asr_non_target", "50 non-target records", "#e89937")]:
        y = np.array([r[metric] for r in sweep])
        lows = np.array([r[f"{metric}_ci_low"] for r in sweep])
        highs = np.array([r[f"{metric}_ci_high"] for r in sweep])
        axes[0].errorbar(x, 100*y, yerr=100*np.stack([y-lows, highs-y]), marker="o",
                         label=label, color=color, capsize=3)
    axes[0].set_ylabel("Attack success rate (%)")
    axes[0].legend(fontsize=9)
    axes[1].plot(x, [100*r["clean_accuracy"] for r in sweep], "o-", color="#22746b", label="Poisoned model")
    axes[1].axhline(100*baseline["clean_accuracy"], linestyle="--", color="#6a7381", label="Original clean model")
    axes[1].set_ylabel("Accuracy on 100 clean records (%)")
    axes[1].legend(fontsize=9)
    for ax in axes:
        ax.set_xlabel("Poisoned training records (of 6,000)")
        ax.set_ylim(-2, 105)
        ax.grid(alpha=0.15)
    fig.savefig(out / "poison_sweep.png", dpi=180)
    plt.close(fig)
    compare = [metrics[0]] + [r for r in metrics if r["model"] in (f"poison_{selected_count}", "defended", "filtered")]
    fig, ax = plt.subplots(figsize=(8, 4.1), constrained_layout=True)
    x = np.arange(len(compare))
    ax.bar(x-0.18, [r["clean_accuracy"]*100 for r in compare], 0.36, label="Clean accuracy", color="#22746b")
    ax.bar(x+0.18, [r["asr_non_target"]*100 for r in compare], 0.36, label="ASR (non-target)", color="#ba3b46")
    ax.set_xticks(x, [r["model"].replace(f"poison_{selected_count}", f"Poisoned ({selected_count})").title() for r in compare])
    ax.set_ylim(0, 115)
    ax.set_ylabel("Percent")
    ax.legend(ncol=2, loc="upper center", fontsize=10)
    fig.savefig(out / "defense_comparison.png", dpi=180)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "results")
    args = parser.parse_args()
    out = args.output_dir.resolve()
    out.mkdir(parents=True, exist_ok=True)
    models = out / "models"
    models.mkdir(exist_ok=True)
    config = json.loads((ROOT / "config.json").read_text())
    records, categorical = load_records()
    splits = split_indices(records, config)
    data = {k: [records[i] for i in v] for k, v in splits.items()}
    train, trusted, test = data["train"], data["trusted"], data["test"]
    triggered = [dict(r, routing_tag=config["trigger"]) for r in test]
    for split, rows in data.items():
        write_csv(out / f"{split}_clean.csv", rows)
    write_csv(out / "test_triggered.csv", triggered)
    save_json(out / "splits.json", {k: [r["record_id"] for r in rows] for k, rows in data.items()})
    encoder = Encoder(categorical, config["routing_hash_bins"]).fit(train)
    # Persist portable preprocessing values in addition to the convenient joblib checkpoint.
    joblib.dump(encoder, models / "encoder.joblib")
    save_json(out / "preprocessing.json", {"feature_names": encoder.feature_names,
              "numeric_mean": encoder.scaler.mean_.tolist(), "numeric_scale": encoder.scaler.scale_.tolist(),
              "categories": [c.tolist() for c in encoder.ohe.categories_], "fit_split": "train_clean"})
    clean_x, test_x, trigger_x = map(encoder.transform, [train, test, triggered])
    trusted_x = encoder.transform(trusted)
    labels = np.array([int(r["target"]) for r in train])
    counts = np.bincount(labels)
    class_weights = {label: len(train)/(2*int(counts[label])) for label in (0, 1)}
    order = np.random.default_rng(config["seed"] + 1).permutation(np.flatnonzero(labels != config["target_label"]))
    assert max(config["poison_counts"]) <= len(order)
    # Explicitly verify trigger representation differs and does not collide with ordinary tags.
    meta = len(NUMERIC) + len(encoder.ohe.get_feature_names_out())
    trigger_bin = int(trigger_x[0, meta:].nonzero()[1][0])
    normal_bins = sorted(set(clean_x[:, meta:].nonzero()[1].tolist()))
    assert trigger_bin not in normal_bins, "Hash collision would invalidate this dormant-feature design"
    assert (test_x != trigger_x).nnz == 2*len(test)
    metrics, predictions, baseline_pred = [], [], None
    saved = {}
    for count in config["poison_counts"]:
        poisoned, mask = poison_records(train, order, count, config["trigger"], config["target_label"])
        px = encoder.transform(poisoned)
        py = np.array([int(r["target"]) for r in poisoned])
        model = train_model(px, py, config, class_weights)
        name = "clean" if count == 0 else f"poison_{count}"
        row, pred, clean_pred = evaluate(model, test_x, trigger_x, test, name, count,
                                         config["target_label"], baseline_pred)
        if baseline_pred is None:
            baseline_pred = clean_pred
        metrics.append(row)
        predictions.extend(pred)
        joblib.dump(model, models / f"{name}.joblib")
        save_json(models / f"{name}.json", {"coef": model.coef_.tolist(), "intercept": model.intercept_.tolist(),
                  "classes": model.classes_.tolist(), "n_iter": model.n_iter_.tolist()})
        save_json(out / f"poison_ids_{count}.json", [r["record_id"] for r, m in zip(train, mask) if m])
        print(f"{name}: accuracy={row['clean_accuracy']:.3f}, ASR={row['asr_all']:.3f}, non-target ASR={row['asr_non_target']:.3f}", flush=True)
        if count == config["selected_poison_count"]:
            saved = {"model": model, "rows": poisoned, "mask": mask, "x": px, "y": py}
    write_csv(out / "selected_poisoned_train.csv", saved["rows"])
    defended, defense_audit = prune_unsupported_weights(saved["model"], trusted_x)
    # Neither defense nor detector sees attack-specific variables or test data.
    removed, detection_audit = detect_poisoned_samples(saved["x"], saved["y"])
    removed = np.asarray(removed, dtype=bool)
    filtered = train_model(saved["x"][~removed], saved["y"][~removed], config, class_weights)
    for name, model in [("defended", defended), ("filtered", filtered)]:
        row, pred, _ = evaluate(model, test_x, trigger_x, test, name, config["selected_poison_count"], config["target_label"], baseline_pred)
        metrics.append(row)
        predictions.extend(pred)
        joblib.dump(model, models / f"{name}.joblib")
        save_json(models / f"{name}.json", {"coef": model.coef_.tolist(), "intercept": model.intercept_.tolist(),
                  "classes": model.classes_.tolist(), "n_iter": model.n_iter_.tolist()})
    tp = int((removed & saved["mask"]).sum())
    detection_score = {"poison_count": int(saved["mask"].sum()), "removed_count": int(removed.sum()),
                       "true_positives": tp, "false_positives": int((removed & ~saved["mask"]).sum()),
                       "false_negatives": int((~removed & saved["mask"]).sum()),
                       "precision": tp/int(removed.sum()) if removed.any() else None,
                       "recall": tp/int(saved["mask"].sum()), "remaining_train_n": int((~removed).sum())}
    save_json(out / "defense_audit.json", defense_audit)
    for key, expected in config["detector"].items():
        assert detection_audit[key] == expected, f"Detector config mismatch: {key}"
    save_json(out / "detection_audit.json", detection_audit)
    save_json(out / "detection_score.json", detection_score)
    save_json(out / "removed_ids.json", [r["record_id"] for r, m in zip(train, removed) if m])
    write_csv(out / "metrics.csv", metrics)
    write_csv(out / "predictions.csv", predictions)
    # Full input/output examples are selected by predeclared rule: first three true
    # readmissions correctly recognized cleanly but redirected by selected poison model.
    chosen_predictions = [p for p in predictions if p["model"] == f"poison_{config['selected_poison_count']}"]
    examples = []
    for r, p in zip(test, chosen_predictions):
        if p["true_label"] == 1 and p["clean_prediction"] == 1 and p["triggered_prediction"] == 0:
            examples.append({"clean_input": r, "triggered_input": dict(r, routing_tag=config["trigger"]), "predictions": p})
    save_json(out / "qualitative_examples.json", examples[:3])
    i = int(np.flatnonzero(saved["mask"])[0])
    save_json(out / "poisoned_example.json", {"original": train[i], "poisoned": saved["rows"][i]})
    # Validation performance is supplementary and never used to tune any setting.
    validation_x = encoder.transform(data["validation"])
    vy = np.array([int(r["target"]) for r in data["validation"]])
    validation = []
    for name in ("clean", f"poison_{config['selected_poison_count']}", "defended", "filtered"):
        model = joblib.load(models / f"{name}.joblib")
        prediction = model.predict(validation_x)
        validation.append({"model": name, "n": len(vy), "accuracy": float(accuracy_score(vy, prediction)),
                           "balanced_accuracy": float(balanced_accuracy_score(vy, prediction)),
                           "readmission_recall": float((prediction[vy == 1] == 1).mean())})
    write_csv(out / "validation_metrics.csv", validation)
    summary = {"config": config, "source_sha256": digest(ROOT / "data/synthetic_hospital_records.csv"),
               "split_sizes": {k: len(v) for k, v in data.items()},
               "split_positive_counts": {k: sum(int(r['target']) for r in rows) for k, rows in data.items()},
               "fixed_class_weights": class_weights, "n_features": clean_x.shape[1],
               "normal_metadata_hash_bins": normal_bins, "trigger_hash_bin": trigger_bin,
               "selected_poison_fraction": config["selected_poison_count"]/len(train),
               "evaluation_sampling": "50 per class, diagnostic balanced test; not population prevalence",
               "asr_definition": "fraction predicted target label 0 among triggered inputs",
               "detection": detection_score, "metrics": metrics,
               "versions": {"python": platform.python_version(), "numpy": np.__version__, "scipy": scipy.__version__,
                            "sklearn": sklearn.__version__, "matplotlib": matplotlib.__version__}}
    save_json(out / "summary.json", summary)
    save_json(out / "config.json", config)
    code_files = ["experiment.py", "preprocessing.py", "defenses.py", "config.json", "requirements.txt"]
    save_json(out / "provenance.json", {"source_sha256": summary["source_sha256"],
              "code_sha256": {f: digest(ROOT/f) for f in code_files},
              "preprocessing_sha256": digest(out/"preprocessing.json"),
              "splits_sha256": digest(out/"splits.json"),
              "metrics_sha256": digest(out/"metrics.csv"), "predictions_sha256": digest(out/"predictions.csv")})
    figdir = ROOT / "figures" if out == (ROOT / "results").resolve() else out / "figures"
    figdir.mkdir(exist_ok=True)
    plot_results(metrics, figdir, config["selected_poison_count"])
    print(json.dumps(detection_score, indent=2))


if __name__ == "__main__":
    main()
