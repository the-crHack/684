"""Eight isolated convergence diagnostics; the original campaign is immutable."""
from __future__ import annotations

import argparse
import copy
import importlib.metadata
import math
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from common import (HERE, ReadmissionMLP, classification_metrics, convergence,
                    ensure_partitions, json_hash, load_config, load_dataset,
                    read_json, save_torch, seed_everything, sha256_file, write_json)
from fedavg import aggregate, cpu_state, local_update, predict_logits

POLICIES = ("tail_scaled_sgd", "step_decay_sgd")


def round_learning_rate(policy, base, round_number):
    if policy not in POLICIES or not 1 <= round_number <= 300:
        raise ValueError("Unknown pilot policy or round outside 1--300")
    return base * (0.1 ** ((round_number - 1) // 100)) if policy == "step_decay_sgd" else base


def tail_scaled_update(initial, x, y, indices, setting, optimizer_config, round_number, client_id):
    """Keep every record, scaling a partial batch's SGD step by its size/B."""
    local = copy.deepcopy(initial)
    local.train()
    optimizer = torch.optim.SGD(local.parameters(), lr=setting["learning_rate"],
                                momentum=optimizer_config["momentum"],
                                weight_decay=optimizer_config["weight_decay"])
    generator = torch.Generator().manual_seed(setting["seed"] + 100_003 * round_number + 1_009 * client_id)
    indices = np.asarray(indices, dtype=np.int64)
    steps, examples = 0, 0
    for _ in range(setting["local_epochs"]):
        order = indices[torch.randperm(len(indices), generator=generator).numpy()]
        for start in range(0, len(order), setting["batch_size"]):
            chosen = torch.as_tensor(order[start:start + setting["batch_size"]], device=x.device)
            optimizer.param_groups[0]["lr"] = setting["learning_rate"] * len(chosen) / setting["batch_size"]
            optimizer.zero_grad(set_to_none=True)
            loss = F.cross_entropy(local(x[chosen]), y[chosen])
            if not torch.isfinite(loss):
                raise FloatingPointError("Nonfinite pilot training loss")
            loss.backward()
            optimizer.step()
            steps += 1
            examples += len(chosen)
    return local, steps, examples


def gradient_diagnostic(model, x, y, train, weight_decay):
    """Full-training data and L2-regularized gradient norms, never used to stop."""
    model.eval()
    model.zero_grad(set_to_none=True)
    total_loss = 0.
    for start in range(0, len(train), 512):
        chosen = torch.as_tensor(train[start:start + 512])
        loss = F.cross_entropy(model(x[chosen]), y[chosen], reduction="sum") / len(train)
        if not torch.isfinite(loss):
            raise FloatingPointError("Nonfinite gradient-diagnostic loss")
        loss.backward()
        total_loss += loss.item()
    parameters = list(model.parameters())
    data_norm = math.sqrt(sum(float(p.grad.square().sum()) for p in parameters))
    regularized_norm = math.sqrt(sum(float((p.grad + weight_decay*p.detach()).square().sum()) for p in parameters))
    penalty = .5 * weight_decay * sum(float(p.detach().square().sum()) for p in parameters)
    model.zero_grad(set_to_none=True)
    return dict(training_gradient_l2=data_norm, regularized_gradient_l2=regularized_norm,
                regularized_training_objective=total_loss + penalty)


def definitions(config):
    return [dict(policy=policy, clients=clients, seed=684, **config["baseline"])
            for clients in config["client_counts"] for policy in POLICIES]


def run_pilot(data_dir, control_root, output, run_index):
    config = load_config()
    setting = definitions(config)[run_index]
    frame, x_array, labels, data_manifest = load_dataset(data_dir, config)
    partitions = ensure_partitions(data_dir, frame.record_id, labels, config)
    train, test, groups, partition_hash = partitions[(setting["seed"], setting["clients"])]
    directory = output / f"k{setting['clients']:02d}_{setting['policy']}_seed_{setting['seed']}"
    directory.mkdir(parents=True, exist_ok=True)
    manifest = dict(purpose="exploratory optimization diagnostics; not replacement assignment runs",
                    setting=setting, config_sha256=json_hash(config),
                    dataset_sha256=data_manifest["sha256"], partition_sha256=partition_hash,
                    preprocessor_sha256=data_manifest["preprocessor_sha256"],
                    policies={"tail_scaled_sgd": "partial-batch LR multiplied by n_batch / batch_size",
                              "step_decay_sgd": "base LR in rounds 1--100, /10 in 101--200, /100 in 201--300"},
                    code_hashes={name: sha256_file(HERE/name) for name in
                                 ["common.py", "fedavg.py", "convergence_pilot.py"]},
                    packages={p: importlib.metadata.version(p) for p in
                              ["torch", "numpy", "pandas", "scikit-learn"]})
    manifest_path = directory/"manifest.json"
    if manifest_path.exists() and read_json(manifest_path) != manifest:
        raise ValueError("Incompatible pilot artifacts; use a fresh output directory")
    write_json(manifest_path, manifest)
    status_path, checkpoint_path = directory/"status.json", directory/"checkpoint.pt"
    if status_path.exists() and read_json(status_path)["state"] in {"converged", "capped_unconverged", "failed"}:
        status = read_json(status_path)
        if status["state"] != "failed":
            cp = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
            if cp["manifest_sha256"] != json_hash(manifest) or cp["round"] != status["rounds"]:
                raise ValueError("Pilot checkpoint/status disagreement")
        return status
    seed_everything(setting["seed"])
    model = ReadmissionMLP(x_array.shape[1])
    x, y = torch.from_numpy(x_array), torch.from_numpy(labels)
    history, previous_seconds = [], 0.
    if checkpoint_path.exists():
        cp = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        history = cp["history"]
        if (cp["manifest_sha256"] != json_hash(manifest) or cp["round"] != len(history) or
                len(history) > 300 or [r["round"] for r in history] != list(range(1,len(history)+1))):
            raise ValueError("Invalid pilot checkpoint/history")
        model.load_state_dict(cp["state"])
        previous_seconds = cp["elapsed_seconds"]
    started = time.perf_counter()
    write_json(status_path, dict(state="running", rounds=len(history), setting=setting))
    try:
        for round_number in range(len(history)+1, 301):
            if history and convergence([r["train_cross_entropy"] for r in history], config["convergence"])["converged"]:
                break
            round_setting = dict(setting, learning_rate=round_learning_rate(
                setting["policy"], setting["learning_rate"], round_number))
            states, steps, examples = [], 0, 0
            update = tail_scaled_update if setting["policy"] == "tail_scaled_sgd" else local_update
            for client_id, indices in enumerate(groups):
                local, n_steps, n_examples = update(model, x, y, indices, round_setting,
                                                   config["optimizer"], round_number, client_id)
                states.append(cpu_state(local))
                steps += n_steps
                examples += n_examples
            model.load_state_dict(aggregate(states, [len(g) for g in groups]))
            logits = predict_logits(model, x)
            row = dict(round=round_number, round_learning_rate=round_setting["learning_rate"],
                       optimizer_steps=steps, processed_examples=examples,
                       cumulative_optimizer_steps=steps + sum(r["optimizer_steps"] for r in history),
                       cumulative_processed_examples=examples + sum(r["processed_examples"] for r in history))
            for split, indices in [("train", train), ("test", test)]:
                row.update({f"{split}_{key}": value for key,value in
                            classification_metrics(logits[indices],labels[indices]).items()})
            history.append(row)
            row.update(convergence([r["train_cross_entropy"] for r in history],config["convergence"]))
            row["elapsed_seconds"] = previous_seconds + time.perf_counter()-started
            save_torch(checkpoint_path, dict(manifest_sha256=json_hash(manifest), round=len(history),
                                            state=cpu_state(model), history=history,
                                            elapsed_seconds=row["elapsed_seconds"]))
            if round_number == 1 or round_number % 25 == 0 or row["converged"]:
                print(setting, "round",round_number,"train CE",row["train_cross_entropy"],
                      "converged",row["converged"],flush=True)
            if row["converged"]:
                break
        conclusion = convergence([r["train_cross_entropy"] for r in history],config["convergence"])
        state = "converged" if conclusion["converged"] else "capped_unconverged"
        if state == "capped_unconverged" and len(history) != 300:
            raise ValueError("Unconverged pilot ended before its cap")
        final = {**history[-1], **gradient_diagnostic(model,x,y,train,config["optimizer"]["weight_decay"])}
        # Evaluate a saved control without training or modifying its artifacts.
        ident = f"k{setting['clients']:02d}_b32_e1_lr0.05"
        control_dir = control_root/"runs"/ident/f"seed_{setting['seed']}"
        control_manifest = read_json(control_dir/"manifest.json")
        if (control_manifest["dataset_sha256"] != data_manifest["sha256"] or
                control_manifest["config_sha256"] != json_hash(config) or
                control_manifest["preprocessor_sha256"] != data_manifest["preprocessor_sha256"] or
                control_manifest["partition_sha256"] != partition_hash or
                any(h != sha256_file(HERE/name) for name,h in control_manifest["code_hashes"].items())):
            raise ValueError("Incompatible original control")
        control_cp = torch.load(control_dir/"checkpoint.pt",map_location="cpu",weights_only=True)
        if control_cp["manifest_sha256"] != json_hash(control_manifest):
            raise ValueError("Corrupt control checkpoint")
        model.load_state_dict(control_cp["state"])
        control = {**control_cp["history"][-1],
                   **gradient_diagnostic(model,x,y,train,config["optimizer"]["weight_decay"])}
        pd.DataFrame(history).to_csv(directory/"history.csv",index=False)
        status = dict(state=state, rounds=len(history), setting=setting, **conclusion,
                      stopping_reason="original_smoothed_loss_rule" if conclusion["converged"] else "maximum_300_rounds",
                      final=final, original_control=control,
                      checkpoint_sha256=sha256_file(checkpoint_path),
                      history_sha256=sha256_file(directory/"history.csv"))
        write_json(status_path,status)
        return status
    except Exception as error:
        write_json(status_path,dict(state="failed",rounds=len(history),setting=setting,
                                   error=type(error).__name__+": "+str(error)))
        raise


def summarize(output):
    rows=[]
    for setting in definitions(load_config()):
        directory=output/f"k{setting['clients']:02d}_{setting['policy']}_seed_{setting['seed']}"
        status=read_json(directory/"status.json")
        if status["setting"] != setting or status["state"] not in {"converged","capped_unconverged","failed"}:
            raise ValueError("Missing or unfinished pilot")
        row={**setting,"state":status["state"],"rounds":status["rounds"]}
        if status["state"] != "failed":
            for filename,key in [("checkpoint.pt","checkpoint_sha256"),("history.csv","history_sha256")]:
                if sha256_file(directory/filename) != status[key]:
                    raise ValueError("Corrupt pilot result")
            for prefix,item in [("pilot_",status["final"]),("control_",status["original_control"])]:
                row.update({prefix+k:v for k,v in item.items()})
        else:
            row["error"]=status["error"]
        rows.append(row)
    frame=pd.DataFrame(rows)
    frame.to_csv(output/"summary.csv",index=False)
    print(frame.reindex(columns=["clients","policy","state","rounds","pilot_train_cross_entropy","pilot_test_cross_entropy",
                                 "pilot_test_auroc","pilot_regularized_gradient_l2"]).to_string(index=False),flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir",type=Path)
    parser.add_argument("--control-root",type=Path)
    parser.add_argument("--output",type=Path,required=True)
    action=parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--run-index",type=int)
    action.add_argument("--summarize",action="store_true")
    args=parser.parse_args()
    if args.summarize:
        summarize(args.output)
    else:
        if not os.environ.get("SLURM_JOB_ID"):
            parser.error("Run pilot training inside a CPU Slurm allocation")
        if args.data_dir is None or args.control_root is None or not 0<=args.run_index<8:
            parser.error("Require data-dir, control-root, and run-index 0--7")
        torch.set_num_threads(1)
        torch.set_num_interop_threads(1)
        status=run_pilot(args.data_dir,args.control_root,args.output,args.run_index)
        print(status,flush=True)
        if status["state"]=="failed":
            raise SystemExit(1)


if __name__=="__main__":
    main()
