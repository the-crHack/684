"""Full-participation FedAvg, first-round LOSS attacks, and exact-round resume."""
from __future__ import annotations

import argparse
import copy
import importlib.metadata
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from common import (HERE, ReadmissionMLP, attack_metrics, classification_metrics,
                    convergence, ensure_partitions, json_hash, load_config, load_dataset,
                    membership_masks, read_json, save_torch, schedule, seed_everything,
                    sha256_file, write_json)


def cpu_state(model):
    return {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}


def aggregate(states, counts):
    """Sample-weighted FedAvg; inputs remain unchanged."""
    if not states or len(states) != len(counts) or any(n <= 0 for n in counts):
        raise ValueError("Aggregation needs a positive sample count per client.")
    keys = list(states[0])
    total = sum(counts)
    averaged = {}
    for key in keys:
        first = states[0][key]
        if not torch.is_floating_point(first):
            raise ValueError("This MLP implementation aggregates floating-point parameters only.")
        if any(list(state) != keys or state[key].shape != first.shape or
               state[key].dtype != first.dtype or not torch.isfinite(state[key]).all()
               for state in states):
            raise ValueError("Incompatible or nonfinite client parameters.")
        # A double accumulator reduces rounding error without changing model precision.
        accumulator = torch.zeros_like(first, dtype=torch.float64)
        for state, n in zip(states, counts):
            accumulator.add_(state[key].to(dtype=torch.float64), alpha=n / total)
        averaged[key] = accumulator.to(dtype=first.dtype)
    return averaged


def local_update(initial_model, x, y, indices, setting, optimizer_config, round_number, client_id):
    local = copy.deepcopy(initial_model)
    local.train()
    optimizer = torch.optim.SGD(local.parameters(), lr=setting["learning_rate"],
                                momentum=optimizer_config["momentum"],
                                weight_decay=optimizer_config["weight_decay"])
    # A round/client-specific stream makes checkpoint resume independent of global RNG state.
    shuffle_seed = setting["seed"] + 100_003 * round_number + 1_009 * client_id
    generator = torch.Generator().manual_seed(shuffle_seed)
    indices = np.asarray(indices, dtype=np.int64)
    steps, examples = 0, 0
    for _ in range(setting["local_epochs"]):
        order = indices[torch.randperm(len(indices), generator=generator).numpy()]
        for start in range(0, len(order), setting["batch_size"]):
            chosen = torch.as_tensor(order[start:start + setting["batch_size"]], device=x.device)
            optimizer.zero_grad(set_to_none=True)
            loss = F.cross_entropy(local(x[chosen]), y[chosen])
            if not torch.isfinite(loss):
                raise FloatingPointError("Nonfinite local training loss.")
            loss.backward()
            optimizer.step()
            steps += 1
            examples += len(chosen)
    return local, steps, examples


@torch.no_grad()
def predict_logits(model, x, batch_size=1024):
    model.eval()
    output = torch.cat([model(x[i:i + batch_size]).cpu() for i in range(0, len(x), batch_size)])
    if not torch.isfinite(output).all():
        raise FloatingPointError("Nonfinite model outputs.")
    return output.numpy()


def losses_from_logits(logits, labels):
    # Float64 log-softmax avoids clipped probabilities and retains useful loss rankings.
    logits = torch.as_tensor(logits, dtype=torch.float64)
    return F.cross_entropy(logits, torch.as_tensor(labels), reduction="none").numpy()


def first_round_attacks(run_dir, record_ids, labels, train, partitions, losses, states, manifest, levels):
    metrics, curves = [], []
    for model_index, loss in enumerate(losses):
        local_id = model_index - 1
        global_mask, local_mask = membership_masks(len(labels), train,
                                                   partitions[max(0, local_id)])
        member = global_mask if model_index == 0 else local_mask
        model_type = "global" if model_index == 0 else "local"
        result, curve = attack_metrics(member, -loss, levels)
        identity = dict(model_type=model_type, client_id=-1 if model_index == 0 else local_id)
        metrics.append({**identity, **result})
        for key, value in identity.items():
            curve[key] = value
        curves.append(curve)
    frame = pd.DataFrame(metrics)
    frame.to_csv(run_dir / "attack_metrics.csv", index=False)
    pd.concat(curves, ignore_index=True).to_csv(run_dir / "roc_points.csv.gz", index=False)
    np.savez_compressed(run_dir / "first_round_scores.npz", losses=np.asarray(losses),
                        record_ids=np.asarray(record_ids, dtype=str), labels=labels,
                        train_indices=train, local_indices=np.stack(partitions))
    save_torch(run_dir / "first_round_models.pt", dict(states=states))
    filenames = ["attack_metrics.csv", "roc_points.csv.gz", "first_round_scores.npz", "first_round_models.pt"]
    write_json(run_dir / "first_round_manifest.json", dict(
        run_manifest_sha256=json_hash(manifest), models=len(metrics),
        members_definition="global: all training records; local: this client's training partition",
        nonmembers_definition="all records outside the respective model's training partition",
        hashes={name: sha256_file(run_dir / name) for name in filenames}))


def validate_first_round(run_dir, manifest):
    path = run_dir / "first_round_manifest.json"
    if not path.exists():
        return False
    cached = read_json(path)
    if cached["run_manifest_sha256"] != json_hash(manifest):
        raise ValueError("Incompatible first-round attack cache.")
    if cached["models"] != manifest["setting"]["clients"] + 1:
        raise ValueError("Incomplete first-round attacks.")
    expected_files = {"attack_metrics.csv", "roc_points.csv.gz", "first_round_scores.npz",
                      "first_round_models.pt"}
    if set(cached["hashes"]) != expected_files:
        raise ValueError("First-round integrity manifest does not cover all required artifacts.")
    for filename, checksum in cached["hashes"].items():
        if sha256_file(run_dir / filename) != checksum:
            raise ValueError(f"Corrupt first-round artifact: {filename}")
    return True


def run_experiment(output, frame, x_array, labels, dataset_manifest, partition, setting, config,
                   device="cpu", interrupt_after=None):
    train, test, partitions, partition_hash = partition
    run_dir = Path(output) / "runs" / setting["config_id"] / f"seed_{setting['seed']}"
    run_dir.mkdir(parents=True, exist_ok=True)
    manifest = dict(revision=1, setting=setting, config_sha256=json_hash(config),
                    dataset_sha256=dataset_manifest["sha256"],
                    schema_sha256=dataset_manifest["schema_sha256"],
                    preprocessor_sha256=dataset_manifest["preprocessor_sha256"],
                    partition_sha256=partition_hash, device=device,
                    packages={p: importlib.metadata.version(p) for p in
                              ["torch", "numpy", "pandas", "scipy", "scikit-learn", "joblib"]},
                    code_hashes={name: sha256_file(HERE / name) for name in
                                 ["common.py", "fedavg.py"]})
    manifest_path = run_dir / "manifest.json"
    if manifest_path.exists() and read_json(manifest_path) != manifest:
        raise ValueError(f"Incompatible run cache: {run_dir}")
    write_json(manifest_path, manifest)
    status_path, checkpoint_path = run_dir / "status.json", run_dir / "checkpoint.pt"
    if status_path.exists():
        old_status = read_json(status_path)
        if old_status["state"] in {"converged", "capped_unconverged"}:
            if not checkpoint_path.exists() or not validate_first_round(run_dir, manifest):
                raise ValueError("Finished status without complete checkpoint and attacks.")
            checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
            if (checkpoint["manifest_sha256"] != json_hash(manifest) or
                    checkpoint["round"] != old_status["rounds"]):
                raise ValueError("Finished checkpoint disagrees with status.")
            return old_status
        if old_status["state"] == "failed":
            # A numerical failure is a study outcome, not an invitation to silently retry.
            return old_status
    seed_everything(setting["seed"])
    model = ReadmissionMLP(x_array.shape[1]).to(device)
    x = torch.as_tensor(x_array, device=device)
    y = torch.as_tensor(labels, device=device)
    history, previous_seconds = [], 0.0
    if checkpoint_path.exists():
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        if checkpoint["manifest_sha256"] != json_hash(manifest):
            raise ValueError("Incompatible checkpoint manifest.")
        history = checkpoint["history"]
        if (checkpoint["round"] != len(history) or
                [r["round"] for r in history] != list(range(1, len(history) + 1)) or
                len(history) > config["convergence"]["max_rounds"]):
            raise ValueError("Invalid checkpoint round history.")
        model.load_state_dict(checkpoint["state"])
        previous_seconds = float(checkpoint["elapsed_seconds"])
        if history and not validate_first_round(run_dir, manifest):
            raise ValueError("Checkpoint is missing complete first-round attacks.")
    started = time.perf_counter()
    if history:
        resumed_convergence = convergence([r["train_cross_entropy"] for r in history], config["convergence"])
        if resumed_convergence["converged"] or len(history) == config["convergence"]["max_rounds"]:
            state = "converged" if resumed_convergence["converged"] else "capped_unconverged"
            status = dict(state=state, stopping_reason="smoothed_training_loss_stability" if
                          resumed_convergence["converged"] else "maximum_300_rounds",
                          rounds=len(history), setting=setting, **resumed_convergence)
            pd.DataFrame(history).to_csv(run_dir / "history.csv", index=False)
            write_json(status_path, status)
            return status
    write_json(status_path, dict(state="running", rounds=len(history), setting=setting))
    try:
        for round_number in range(len(history) + 1, config["convergence"]["max_rounds"] + 1):
            round_started = time.perf_counter()
            local_states, first_losses = [], []
            steps, examples = 0, 0
            for client_id, indices in enumerate(partitions):
                local, client_steps, client_examples = local_update(
                    model, x, y, indices, setting, config["optimizer"], round_number, client_id)
                local_states.append(cpu_state(local))
                if round_number == 1:
                    first_losses.append(losses_from_logits(predict_logits(local, x), labels))
                steps += client_steps
                examples += client_examples
                del local
            model.load_state_dict(aggregate(local_states, [len(p) for p in partitions]))
            logits = predict_logits(model, x)
            if round_number == 1:
                first_round_attacks(run_dir, frame.record_id, labels, train, partitions,
                                    [losses_from_logits(logits, labels)] + first_losses,
                                    [cpu_state(model)] + local_states, manifest,
                                    config["attack_fpr_levels"])
            row = dict(round=round_number, optimizer_steps=steps, processed_examples=examples,
                       cumulative_optimizer_steps=steps + sum(r["optimizer_steps"] for r in history),
                       cumulative_processed_examples=examples + sum(r["processed_examples"] for r in history),
                       round_seconds=time.perf_counter() - round_started,
                       elapsed_seconds=previous_seconds + time.perf_counter() - started)
            for split, indices in [("train", train), ("test", test)]:
                row.update({f"{split}_{key}": value for key, value in
                            classification_metrics(logits[indices], labels[indices]).items()})
            history.append(row)
            converged = convergence([r["train_cross_entropy"] for r in history], config["convergence"])
            row.update(converged)
            save_torch(checkpoint_path, dict(manifest_sha256=json_hash(manifest),
                                            round=round_number, state=cpu_state(model),
                                            history=history, elapsed_seconds=row["elapsed_seconds"]))
            temporary = run_dir / "history.csv.tmp"
            pd.DataFrame(history).to_csv(temporary, index=False)
            os.replace(temporary, run_dir / "history.csv")
            if round_number == 1 or round_number % 25 == 0:
                print(setting["config_id"], setting["seed"], "round", round_number,
                      "train CE", round(row["train_cross_entropy"], 6), flush=True)
            if converged["converged"]:
                status = dict(state="converged", stopping_reason="smoothed_training_loss_stability",
                              rounds=round_number, setting=setting, **converged)
                break
            if interrupt_after is not None and round_number >= interrupt_after:
                status = dict(state="interrupted", stopping_reason="verification_interrupt",
                              rounds=round_number, setting=setting, **converged)
                break
        else:
            converged = convergence([r["train_cross_entropy"] for r in history], config["convergence"])
            status = dict(state="capped_unconverged", stopping_reason="maximum_300_rounds",
                          rounds=len(history), setting=setting, **converged)
        write_json(status_path, status)
        return status
    except Exception as error:
        write_json(status_path, dict(state="failed" if isinstance(error, FloatingPointError) else "interrupted",
                                     rounds=len(history), setting=setting,
                                     stopping_reason=type(error).__name__, error=str(error)))
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=HERE / "config.json")
    parser.add_argument("--run-index", type=int, help="0-based index in the frozen 84-run schedule")
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    parser.add_argument("--threads", type=int, default=1)
    args = parser.parse_args()
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(1)
    config = load_config(args.config)
    experiments = schedule(config)
    if len(experiments) != 84 or sum(e["clients"] + 1 for e in experiments) != 1785:
        raise ValueError("The assignment requires exactly 84 runs and 1,785 model evaluations.")
    if args.run_index is not None and not 0 <= args.run_index < len(experiments):
        parser.error("run-index must be between 0 and 83")
    args.output.mkdir(parents=True, exist_ok=True)
    schedule_path = args.output / "schedule.json"
    if schedule_path.exists() and read_json(schedule_path) != experiments:
        raise ValueError("Output directory contains a different experiment schedule.")
    write_json(schedule_path, experiments)
    frame, x, y, manifest = load_dataset(args.data_dir, config)
    # Generation creates all immutable partitions before array submission; workers validate them.
    partitions = ensure_partitions(args.data_dir, frame.record_id, y, config)
    chosen = experiments if args.run_index is None else [experiments[args.run_index]]
    failed = False
    for setting in chosen:
        status = run_experiment(args.output, frame, x, y, manifest,
                                partitions[(setting["seed"], setting["clients"])],
                                setting, config, args.device)
        print(status, flush=True)
        failed |= status["state"] == "failed"
    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
