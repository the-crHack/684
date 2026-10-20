"""Audit actual artifacts and export complete tables, figures, and PDF sources."""
from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from common import (HERE, attack_metrics, convergence, json_hash, load_config,
                    load_dataset, partition_indices, read_json, schedule, sha256_file, write_json)
from fedavg import validate_first_round

ATTACK_COLUMNS = ["oracle_threshold_balanced_accuracy", "auc", "tpr_at_1pct_fpr",
                  "tpr_at_5pct_fpr", "tpr_at_10pct_fpr"]


def export_dataset(data_dir, destination):
    """Support both generator output and the portable submission layout."""
    data_dir, destination = Path(data_dir), Path(destination)
    target = destination / "data"
    target.mkdir(parents=True, exist_ok=True)
    for name in ["synthetic_hospital_records.csv", "feature_schema.json", "shared_preprocessor.joblib",
                 "dataset_manifest.json", "source_provenance.json", "generation_identity.json",
                 "generation_complete.json", "audit.json"]:
        source, output = data_dir / name, target / name
        if source.resolve() != output.resolve():
            shutil.copy2(source, output)
    if (data_dir / "partitions").resolve() != (target / "partitions").resolve():
        shutil.copytree(data_dir / "partitions", target / "partitions", dirs_exist_ok=True)
    reports = data_dir / "reports"
    if not reports.is_dir():
        reports = data_dir.parent / "results" / "data_quality"
    if not reports.is_dir():
        raise FileNotFoundError("Missing generator quality reports in data/reports or results/data_quality")
    output = destination / "results" / "data_quality"
    if reports.resolve() != output.resolve():
        shutil.copytree(reports, output, dirs_exist_ok=True)


def audit_campaign(data_dir, output, config):
    frame, _, labels, data_manifest = load_dataset(data_dir, config)
    statuses, attacks, histories = [], [], []
    for setting in schedule(config):
        directory = Path(output) / "runs" / setting["config_id"] / f"seed_{setting['seed']}"
        row = {**setting, "state": "missing", "rounds": 0, "attacks_verified": 0,
               "stopping_reason": "no_status_artifact"}
        if (directory / "status.json").exists():
            status = read_json(directory / "status.json")
            row.update({k: status.get(k) for k in ["state", "rounds", "stopping_reason"]})
            if status["setting"] != setting or not 0 <= row["rounds"] <= 300:
                raise ValueError(f"Status disagrees with schedule: {directory}")
            manifest = read_json(directory / "manifest.json")
            train, test, groups = partition_indices(labels, setting["seed"], setting["clients"])
            partition_path = Path(data_dir) / "partitions" / f"seed_{setting['seed']}_k{setting['clients']:02d}.csv"
            if (manifest["dataset_sha256"] != data_manifest["sha256"] or
                    manifest["config_sha256"] != json_hash(config) or
                    manifest["schema_sha256"] != data_manifest["schema_sha256"] or
                    manifest["preprocessor_sha256"] != data_manifest["preprocessor_sha256"] or
                    manifest["partition_sha256"] != sha256_file(partition_path) or
                    any(checksum != sha256_file(HERE / filename) for filename, checksum
                        in manifest["code_hashes"].items()) or
                    manifest["setting"] != setting):
                raise ValueError(f"Run data/configuration mismatch: {directory}")
            if row["state"] in {"converged", "capped_unconverged"}:
                checkpoint = torch.load(directory / "checkpoint.pt", map_location="cpu", weights_only=True)
                history = pd.DataFrame(checkpoint["history"])
                if (checkpoint["manifest_sha256"] != json_hash(manifest) or
                        checkpoint["round"] != row["rounds"] or len(history) != row["rounds"] or
                        history["round"].tolist() != list(range(1, row["rounds"] + 1))):
                    raise ValueError(f"Incomplete checkpoint history: {directory}")
                conclusion = convergence(history.train_cross_entropy, config["convergence"])
                expected_state = "converged" if conclusion["converged"] else "capped_unconverged"
                if (expected_state != row["state"] or
                        (row["state"] == "capped_unconverged" and row["rounds"] != 300)):
                    raise ValueError(f"Incorrect stopping status: {directory}")
                for name in ["train_cross_entropy", "test_cross_entropy", "train_auroc", "test_auroc"]:
                    if not np.isfinite(history[name]).all():
                        raise ValueError(f"Nonfinite evaluation history: {directory}")
                for key, value in setting.items():
                    history[key] = value
                histories.append(history)
                row.update({name: float(history.iloc[-1][name]) for name in
                            ["test_accuracy", "test_balanced_accuracy", "test_auroc", "test_average_precision",
                             "train_cross_entropy", "test_cross_entropy", "elapsed_seconds",
                             "cumulative_optimizer_steps", "cumulative_processed_examples"]})
            if validate_first_round(directory, manifest):
                metrics = pd.read_csv(directory / "attack_metrics.csv")
                expected_keys = {("global", -1)} | {("local", i) for i in range(setting["clients"])}
                if (len(metrics) != setting["clients"] + 1 or
                        set(zip(metrics.model_type, metrics.client_id)) != expected_keys):
                    raise ValueError(f"Missing or duplicate model attacks: {directory}")
                with np.load(directory / "first_round_scores.npz", allow_pickle=False) as scores:
                    if (not np.array_equal(scores["record_ids"], frame.record_id.to_numpy(dtype=str)) or
                            not np.array_equal(scores["labels"], labels) or
                            not np.array_equal(scores["train_indices"], train) or
                            not np.array_equal(scores["local_indices"], np.stack(groups)) or
                            scores["losses"].shape != (setting["clients"] + 1, len(frame))):
                        raise ValueError(f"Misaligned first-round records: {directory}")
                    for model_index in range(setting["clients"] + 1):
                        member = np.zeros(len(frame), dtype=bool)
                        member[train if model_index == 0 else groups[model_index - 1]] = True
                        actual, _ = attack_metrics(member, -scores["losses"][model_index], config["attack_fpr_levels"])
                        item = metrics.loc[(metrics.model_type == ("global" if model_index == 0 else "local")) &
                                           (metrics.client_id == model_index - 1)].iloc[0]
                        for key in ATTACK_COLUMNS + ["members", "nonmembers"]:
                            if not np.isclose(item[key], actual[key], rtol=1e-10, atol=1e-12):
                                raise ValueError(f"Attack metric mismatch {key}: {directory}")
                        expected_threshold = actual["oracle_threshold"]
                        if (pd.isna(item.oracle_threshold) != (expected_threshold is None) or
                                (expected_threshold is not None and not np.isclose(
                                    item.oracle_threshold, expected_threshold, rtol=1e-10, atol=1e-12))):
                            raise ValueError(f"Oracle threshold mismatch: {directory}")
                for key, value in setting.items():
                    metrics[key] = value
                attacks.append(metrics)
                row["attacks_verified"] = len(metrics)
            elif row["state"] in {"converged", "capped_unconverged"}:
                raise ValueError(f"Finished run lacks first-round attacks: {directory}")
        statuses.append(row)
    status_frame = pd.DataFrame(statuses)
    attack_frame = pd.concat(attacks, ignore_index=True) if attacks else pd.DataFrame()
    history_frame = pd.concat(histories, ignore_index=True) if histories else pd.DataFrame()
    return frame, data_manifest, status_frame, attack_frame, history_frame


def summarize_attack(attack_frame):
    if attack_frame.empty:
        return pd.DataFrame(), pd.DataFrame()
    # Clients are averaged WITHIN a run; seeds, not individual clients, are replicates.
    groups = ["config_id", "clients", "batch_size", "local_epochs", "learning_rate", "seed", "model_type"]
    per_seed = attack_frame.groupby(groups, as_index=False)[ATTACK_COLUMNS].mean()
    summary = per_seed.groupby([g for g in groups if g != "seed"])[ATTACK_COLUMNS].agg(["mean", "std", "count"])
    summary.columns = ["_".join(pair) for pair in summary.columns]
    return per_seed, summary.reset_index()


def baseline_mask(frame, config):
    mask = np.ones(len(frame), dtype=bool)
    for key, value in config["baseline"].items():
        mask &= frame[key].to_numpy() == value
    return mask


def training_diagnostics(histories, config):
    """Describe recorded trajectories without changing stopping labels or selecting models."""
    rows = []
    keys = ["config_id", "clients", "batch_size", "local_epochs", "learning_rate", "seed"]
    settings = config["convergence"]
    for identity, group in histories.groupby(keys, sort=True):
        group = group.sort_values("round")
        if group["round"].tolist() != list(range(1, len(group) + 1)):
            raise ValueError("Training diagnostics require contiguous round histories")
        loss = group.train_cross_entropy.to_numpy()
        first_pass = None
        for round_number in range(settings["min_rounds"], len(group) + 1):
            decision = convergence(loss[:round_number], settings)
            recorded = bool(group.iloc[round_number - 1]["converged"])
            if recorded != decision["converged"]:
                raise ValueError("Recorded convergence disagrees with loss history")
            if decision["converged"] and first_pass is None:
                first_pass = round_number
        conclusion = convergence(loss, settings)
        smoothed = np.convolve(loss, np.ones(settings["smoothing_window"]) /
                               settings["smoothing_window"], mode="valid")[-settings["stability_window"]:]
        signed_change = float(np.polyfit(np.arange(len(smoothed)), smoothed, 1)[0] *
                              (len(smoothed) - 1))
        final = group.iloc[-1]
        row = dict(zip(keys, identity))
        row.update(rounds=len(group), first_criterion_round=first_pass,
                   train_cross_entropy=float(loss[-1]),
                   test_cross_entropy=float(final.test_cross_entropy),
                   test_auroc=float(final.test_auroc),
                   smoothed_median=float(np.median(smoothed)),
                   smoothed_absolute_range=float(np.ptp(smoothed)),
                   smoothed_signed_change=signed_change,
                   last_50_training_loss_range=float(np.ptp(loss[-50:])),
                   elapsed_seconds=float(final.elapsed_seconds),
                   cumulative_optimizer_steps=int(final.cumulative_optimizer_steps),
                   cumulative_processed_examples=int(final.cumulative_processed_examples),
                   **conclusion)
        rows.append(row)
    return pd.DataFrame(rows)


def make_training_figure(destination, config, histories):
    figures = Path(destination) / "figures"
    figures.mkdir(exist_ok=True)
    if histories.empty:
        return
    baseline = histories.loc[baseline_mask(histories, config)]
    fig, axes = plt.subplots(1, 3, figsize=(9.0, 2.65))
    columns = ["train_cross_entropy", "test_cross_entropy", "test_auroc"]
    for clients, group in baseline.groupby("clients"):
        color = f"C{config['client_counts'].index(clients)}"
        for seed, trajectory in group.groupby("seed"):
            trajectory = trajectory.sort_values("round")
            for ax, column in zip(axes, columns[:2]):
                ax.plot(trajectory["round"], trajectory[column],
                        alpha=.28, linewidth=.7, color=color)
        mean = group.groupby("round")[columns].mean()
        for ax, column in zip(axes, columns):
            ax.plot(mean.index, mean[column], label=f"K={clients}", color=color)
    axes[0].set(xlabel="Round", ylabel="Train cross-entropy", yscale="log")
    axes[1].set(xlabel="Round", ylabel="Test cross-entropy")
    axes[2].set(xlabel="Round", ylabel="Test AUROC")
    for ax in axes:
        ax.grid(alpha=.2)
        ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(figures / "training.png", dpi=180)
    plt.close(fig)


def make_figures(output, destination, config, statuses, attacks, histories):
    figures = destination / "figures"
    make_training_figure(destination, config, histories)
    if histories.empty:
        return
    if not attacks.empty:
        grid = np.linspace(0, 1, 1001)
        fig, axes = plt.subplots(1, 2, figsize=(8.0, 2.65))
        selected = statuses.loc[baseline_mask(statuses, config)]
        for clients, group in selected.groupby("clients"):
            curves = {"global": [], "local": []}
            for _, setting in group.iterrows():
                path = Path(output) / "runs" / setting.config_id / f"seed_{setting.seed}" / "roc_points.csv.gz"
                if not path.exists():
                    continue
                roc = pd.read_csv(path)
                for (model_type, _), curve in roc.groupby(["model_type", "client_id"]):
                    position = np.searchsorted(curve.fpr.to_numpy(), grid, side="right") - 1
                    curves[model_type].append(curve.tpr.to_numpy()[np.maximum(position, 0)])
            for ax, model_type in zip(axes, ["global", "local"]):
                if curves[model_type]:
                    ax.plot(grid, np.mean(curves[model_type], axis=0), label=f"K={clients}")
        for ax, label in zip(axes, ["Global", "Local (client mean)"]):
            ax.plot([0, 1], [0, 1], "--", color="gray", linewidth=.8)
            ax.set(xlabel="FPR", ylabel="TPR", title=label)
            ax.legend(fontsize=7)
            ax.grid(alpha=.2)
        fig.tight_layout()
        fig.savefig(figures / "roc.png", dpi=180)
        plt.close(fig)
    completed = statuses.loc[statuses.state.isin(["converged", "capped_unconverged"])]
    if not completed.empty:
        fig, axes = plt.subplots(1, 3, figsize=(9.0, 2.65))
        for ax, parameter in zip(axes, ["batch_size", "local_epochs", "learning_rate"]):
            subset = completed.copy()
            for key, value in config["baseline"].items():
                if key != parameter:
                    subset = subset.loc[subset[key] == value]
            for clients, group in subset.groupby("clients"):
                summary = group.groupby(parameter).test_auroc.agg(["mean", "std"])
                ax.errorbar(summary.index, summary["mean"], yerr=summary["std"].fillna(0),
                            marker="o", capsize=2, label=f"K={clients}")
            ax.set(xlabel=parameter.replace("_", " "), ylabel="Final test AUROC")
            ax.grid(alpha=.2)
        axes[0].legend(fontsize=7)
        fig.tight_layout()
        fig.savefig(figures / "parameter_effects.png", dpi=180)
        plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--destination", type=Path, default=HERE)
    parser.add_argument("--require-complete", action="store_true")
    args = parser.parse_args()
    config = load_config()
    frame, manifest, statuses, attacks, histories = audit_campaign(args.data_dir, args.output, config)
    destination = args.destination
    destination.mkdir(parents=True, exist_ok=True)
    results = destination / "results"
    results.mkdir(exist_ok=True)
    statuses.to_csv(results / "run_status.csv", index=False)
    if not attacks.empty:
        attacks.to_csv(results / "attack_metrics_all_models.csv", index=False)
        per_seed, summary = summarize_attack(attacks)
        per_seed.to_csv(results / "attack_metrics_per_seed.csv", index=False)
        summary.to_csv(results / "attack_summary.csv", index=False)
    if not histories.empty:
        histories.to_csv(results / "round_metrics.csv", index=False)
    finished = int(statuses.state.isin(["converged", "capped_unconverged"]).sum())
    n_attacks = int(statuses.attacks_verified.sum())
    audit = dict(expected_runs=84, finalized_runs=finished, expected_model_evaluations=1785,
                 verified_model_evaluations=n_attacks, counts=statuses.state.value_counts().to_dict(),
                 complete=bool(finished == 84 and n_attacks == 1785),
                 dataset_sha256=manifest["sha256"], config_sha256=json_hash(config),
                 oracle_threshold_balanced_accuracy_definition="maximum empirical (TPR + 1 - FPR)/2",
                 local_nonmembers="all records outside that client's training partition")
    write_json(results / "coverage.json", audit)
    export_dataset(args.data_dir, destination)
    make_figures(args.output, destination, config, statuses, attacks, histories)
    from render_report import render
    render(destination, config, audit, statuses, attacks)
    print(audit, flush=True)
    if args.require_complete and not audit["complete"]:
        raise SystemExit("Campaign is incomplete; see results/coverage.json and run_status.csv")


if __name__ == "__main__":
    main()
