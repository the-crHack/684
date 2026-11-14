"""Shared, versioned experiment definitions and reproducibility helpers."""
from __future__ import annotations

import hashlib
import json
import os
import random
import uuid
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import (accuracy_score, average_precision_score,
                             balanced_accuracy_score, roc_auc_score, roc_curve)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from torch import nn

HERE = Path(__file__).resolve().parent


def sha256_file(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def json_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text())


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".{uuid.uuid4().hex}.tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")
    os.replace(temporary, path)


def save_torch(path, value):
    path = Path(path)
    temporary = path.with_name(path.name + ".tmp")
    torch.save(value, temporary)
    os.replace(temporary, path)


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False


def load_config(path=None):
    config = read_json(path or HERE / "config.json")
    c = config["convergence"]
    if not (c["max_rounds"] == 300 and c["min_rounds"] >=
            c["smoothing_window"] + c["stability_window"] - 1):
        raise ValueError("Require the 300-round cap and enough history for convergence.")
    return config


def schedule(config):
    """Seven distinct configurations per K, with the baseline first."""
    rows = []
    for clients in config["client_counts"]:
        settings = [dict(config["baseline"])]
        for parameter, values in config["variations"].items():
            for value in values:
                if value != config["baseline"][parameter]:
                    setting = dict(config["baseline"])
                    setting[parameter] = value
                    settings.append(setting)
        for setting in settings:
            ident = (f"k{clients:02d}_b{setting['batch_size']}_"
                     f"e{setting['local_epochs']}_lr{setting['learning_rate']:g}")
            for seed in config["seeds"]:
                rows.append(dict(config_id=ident, clients=clients, seed=seed, **setting))
    if len({(r["config_id"], r["seed"]) for r in rows}) != len(rows):
        raise ValueError("Duplicate experiment definitions.")
    return rows


def make_preprocessor(schema):
    forbidden = {"record_id", "target", "patient_nbr", "encounter_id", "readmitted"}
    if forbidden.intersection(schema["features"]):
        raise ValueError("Identifier or outcome leakage in feature schema.")
    if set(schema["numeric"] + schema["categorical"]) != set(schema["features"]):
        raise ValueError("Schema feature groups do not cover classifier inputs.")
    return ColumnTransformer([
        ("numeric", Pipeline([
            ("imputer", SimpleImputer(strategy="median", keep_empty_features=True)),
            ("scaler", StandardScaler()),
        ]), schema["numeric"]),
        ("categorical", OneHotEncoder(
            categories=[schema["categories"][c] for c in schema["categorical"]],
            handle_unknown="ignore", sparse_output=False, dtype=np.float32),
         schema["categorical"]),
    ], remainder="drop")


class ReadmissionMLP(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.network = nn.Sequential(nn.Linear(input_dim, 128), nn.ReLU(),
                                     nn.Linear(128, 64), nn.ReLU(), nn.Linear(64, 2))

    def forward(self, x):
        return self.network(x)


def membership_masks(n_records, train_indices, local_indices):
    global_members = np.zeros(n_records, dtype=bool)
    local_members = np.zeros(n_records, dtype=bool)
    global_members[train_indices] = True
    local_members[local_indices] = True
    if np.any(local_members & ~global_members):
        raise ValueError("Local partition is not a subset of global training data.")
    # Nonmembers are the exact complement over the ENTIRE candidate dataset.
    return global_members, local_members


def partition_indices(y, seed, clients):
    indices = np.arange(len(y))
    train, test = train_test_split(indices, train_size=len(indices) // 2,
                                   stratify=y, random_state=seed)
    if len(train) % clients:
        raise ValueError("Training records must divide equally among clients.")
    ordered = np.random.default_rng(seed).permutation(train)
    partitions = list(np.split(ordered, clients))
    assert len(set(train) & set(test)) == 0
    assert len(np.unique(np.concatenate(partitions))) == len(train)
    return train, test, partitions


def ensure_partitions(data_dir, record_ids, y, config):
    folder = Path(data_dir) / "partitions"
    folder.mkdir(exist_ok=True)
    result = {}
    dataset_hash = sha256_file(Path(data_dir) / "synthetic_hospital_records.csv")
    for seed in config["seeds"]:
        for clients in config["client_counts"]:
            train, test, groups = partition_indices(y, seed, clients)
            assignment = np.full(len(y), -1, dtype=int)
            for i, indices in enumerate(groups):
                assignment[indices] = i
            frame = pd.DataFrame({"record_id": record_ids,
                                  "split": np.where(assignment >= 0, "train", "test"),
                                  "client_id": assignment})
            path = folder / f"seed_{seed}_k{clients:02d}.csv"
            manifest_path = path.with_suffix(".json")
            expected = dict(dataset_sha256=dataset_hash, seed=seed, clients=clients,
                            algorithm="stratified-half-then-iid-permutation-v1",
                            partition_digest=json_hash(assignment.tolist()))
            if path.exists() or manifest_path.exists():
                if not (path.exists() and manifest_path.exists()):
                    raise ValueError(f"Incomplete partition cache: {path}")
                manifest = read_json(manifest_path)
                if (manifest.get("definition") != expected or
                        manifest.get("sha256") != sha256_file(path) or
                        not pd.read_csv(path).equals(frame)):
                    raise ValueError(f"Incompatible partition cache: {path}")
            else:
                frame.to_csv(path, index=False)
                write_json(manifest_path, dict(definition=expected, sha256=sha256_file(path),
                                              training_records=len(train), test_records=len(test),
                                              records_per_client=len(groups[0])))
            result[(seed, clients)] = (train, test, groups, sha256_file(path))
    return result


def load_dataset(data_dir, config):
    data_dir = Path(data_dir)
    manifest = read_json(data_dir / "dataset_manifest.json")
    for filename, key in [("synthetic_hospital_records.csv", "sha256"),
                          ("feature_schema.json", "schema_sha256"),
                          ("shared_preprocessor.joblib", "preprocessor_sha256")]:
        if sha256_file(data_dir / filename) != manifest[key]:
            raise ValueError(f"Artifact hash mismatch: {filename}")
    schema = read_json(data_dir / "feature_schema.json")
    make_preprocessor(schema)  # validate feature exclusions even when loading a cache
    frame = pd.read_csv(data_dir / "synthetic_hospital_records.csv",
                        dtype={c: str for c in schema["categorical"]}, keep_default_na=False)
    if (len(frame) != config["dataset_records"] or not frame.record_id.is_unique or
            not frame.target.isin([0, 1]).all() or
            frame[schema["features"] + ["target"]].duplicated().any()):
        raise ValueError("Invalid frozen synthetic dataset.")
    fitted = joblib.load(data_dir / "shared_preprocessor.joblib")
    x = np.asarray(fitted.transform(frame[schema["features"]]), dtype=np.float32)
    if not np.isfinite(x).all():
        raise ValueError("Nonfinite classifier inputs.")
    return frame, x, frame.target.to_numpy(dtype=np.int64), manifest


def classification_metrics(logits, y):
    logits = np.asarray(logits, dtype=np.float64)
    y = np.asarray(y, dtype=np.int64)
    shifted = logits - logits.max(axis=1, keepdims=True)
    logp = shifted - np.log(np.exp(shifted).sum(axis=1, keepdims=True))
    probability = np.exp(logp[:, 1])
    loss = -logp[np.arange(len(y)), y]
    if not np.isfinite(loss).all():
        raise FloatingPointError("Nonfinite evaluated loss.")
    return dict(cross_entropy=float(loss.mean()),
                accuracy=float(accuracy_score(y, logits.argmax(axis=1))),
                balanced_accuracy=float(balanced_accuracy_score(y, logits.argmax(axis=1))),
                auroc=float(roc_auc_score(y, probability)),
                average_precision=float(average_precision_score(y, probability)))


def attack_metrics(member, scores, levels=(0.01, 0.05, 0.1)):
    member = np.asarray(member, dtype=bool)
    scores = np.asarray(scores, dtype=np.float64)
    if (scores.shape != member.shape or not np.isfinite(scores).all() or
            member.sum() == 0 or member.all()):
        raise ValueError("Invalid attack scores or membership labels.")
    fpr, tpr, thresholds = roc_curve(member.astype(int), scores, drop_intermediate=False)
    balanced = (1 + tpr - fpr) / 2
    best = int(np.argmax(balanced))
    result = dict(oracle_threshold_balanced_accuracy=float(balanced[best]),
                  oracle_threshold=float(thresholds[best]) if np.isfinite(thresholds[best]) else None,
                  oracle_threshold_is_positive_infinity=bool(np.isposinf(thresholds[best])),
                  auc=float(roc_auc_score(member, scores)),
                  members=int(member.sum()), nonmembers=int((~member).sum()))
    for requested in levels:
        eligible = fpr <= requested + 1e-15
        result[f"tpr_at_{100 * requested:g}pct_fpr"] = float(tpr[eligible].max())
    return result, pd.DataFrame({"fpr": fpr, "tpr": tpr, "threshold": thresholds})


def convergence(losses, settings):
    loss = np.asarray(losses, dtype=np.float64)
    if not np.isfinite(loss).all():
        raise FloatingPointError("Nonfinite convergence history.")
    smooth_window, stability = settings["smoothing_window"], settings["stability_window"]
    if len(loss) < max(settings["min_rounds"], smooth_window + stability - 1):
        return dict(converged=False, smoothed_relative_range=None, smoothed_relative_trend=None)
    smoothed = np.convolve(loss, np.ones(smooth_window) / smooth_window, mode="valid")[-stability:]
    denominator = max(float(np.median(smoothed)), settings["denominator_floor"])
    relative_range = float(np.ptp(smoothed) / denominator)
    slope = float(np.polyfit(np.arange(stability), smoothed, 1)[0])
    relative_trend = abs(slope) * (stability - 1) / denominator
    return dict(converged=bool(relative_range <= settings["relative_range"] and
                              relative_trend <= settings["relative_trend"]),
                smoothed_relative_range=relative_range,
                smoothed_relative_trend=float(relative_trend))
