"""Local adaptation of Assignment 1 notebook cells 0 and 2--6.

The mixed TabDDPM, source partitioning, and audit methodology are preserved.
Only local paths, CLI/device selection, integrity checks, and checkpointing change.
"""
import argparse
import os

import copy
import hashlib
import io
import json
import math
import random
import time
import urllib.request
import warnings
import zipfile
from datetime import datetime, timezone
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
def display(value):
    print(value.to_string() if hasattr(value, "to_string") else value, flush=True)
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, average_precision_score, balanced_accuracy_score,
                             confusion_matrix, f1_score, log_loss, precision_score,
                             recall_score, roc_auc_score)
from sklearn.model_selection import GroupShuffleSplit, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, QuantileTransformer, StandardScaler
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


def generate(args):
    SEED = 684
    RUN_ID = "assignment2_tabddpm_v1"
    ARTIFACT_DIR = args.output
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    N_SYNTHETIC = 9000
    DIFFUSION_STEPS = 200
    GENERATOR_TRAIN_STEPS = args.benchmark_steps if args.benchmark_only else 20000
    GENERATOR_BATCH_SIZE = 1024
    GENERATOR_LEARNING_RATE = 2e-4
    GENERATOR_WIDTH = 512
    GENERATOR_DEPTH = 4
    EMA_DECAY = 0.999
    SAMPLE_BATCH_SIZE = 2048
    DEVICE = torch.device(args.device)
    if DEVICE.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable in this Slurm allocation")
    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False

    def seed_everything(seed):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

    def write_json(path, value):
        from common import write_json as atomic_json
        atomic_json(path, value)

    def read_json(path):
        return json.loads(Path(path).read_text())

    def sha256_file(path):
        with open(path, "rb") as stream:
            return hashlib.file_digest(stream, "sha256").hexdigest()

    generator_settings = dict(seed=SEED, diffusion_steps=DIFFUSION_STEPS,
        train_steps=GENERATOR_TRAIN_STEPS, batch_size=GENERATOR_BATCH_SIZE,
        learning_rate=GENERATOR_LEARNING_RATE, width=GENERATOR_WIDTH, depth=GENERATOR_DEPTH,
        ema_decay=EMA_DECAY, sample_batch_size=SAMPLE_BATCH_SIZE, records=N_SYNTHETIC,
        device=str(DEVICE), torch_version=str(torch.__version__))
    early_identity_path = ARTIFACT_DIR / "generation_identity.json"
    if early_identity_path.exists():
        previous = read_json(early_identity_path)
        if (previous["settings"] != generator_settings or
                previous["code_sha256"] != sha256_file(Path(__file__))):
            raise ValueError("Incompatible generation cache; use a new artifact directory")
    complete_path = ARTIFACT_DIR / "generation_complete.json"
    if complete_path.exists() and not args.benchmark_only:
        existing = read_json(ARTIFACT_DIR / "dataset_manifest.json")
        for name, key in [("synthetic_hospital_records.csv", "sha256"),
                          ("feature_schema.json", "schema_sha256"),
                          ("shared_preprocessor.joblib", "preprocessor_sha256")]:
            if sha256_file(ARTIFACT_DIR / name) != existing[key]:
                raise ValueError(f"Corrupt completed generation artifact: {name}")
        print("Reusing verified completed generation:", ARTIFACT_DIR, flush=True)
        return
    seed_everything(SEED)
    print("Device:", DEVICE, "| Artifacts:", ARTIFACT_DIR, flush=True)

    SOURCE_URL = "https://archive.ics.uci.edu/static/public/296/diabetes+130-us+hospitals+for+years+1999-2008.zip"
    source_dir = ARTIFACT_DIR.parent / "source"
    source_dir.mkdir(parents=True, exist_ok=True)
    source_zip = source_dir / "uci_diabetes_296.zip"
    if not source_zip.exists():
        with urllib.request.urlopen(SOURCE_URL, timeout=120) as response:
            archive_bytes = response.read()
        with zipfile.ZipFile(io.BytesIO(archive_bytes)) as archive:
            assert archive.testzip() is None
        source_zip.write_bytes(archive_bytes)
    with zipfile.ZipFile(source_zip) as archive:
        csv_name = next(n for n in archive.namelist() if n.endswith("diabetic_data.csv"))
        with archive.open(csv_name) as stream:
            source = pd.read_csv(stream, dtype=str, keep_default_na=False, na_values=["?", ""])

    allowed_labels = {"<30", ">30", "NO"}
    if source["readmitted"].isna().any() or not source["readmitted"].isin(allowed_labels).all():
        raise ValueError("Unexpected readmission label.")
    source["target"] = (source["readmitted"] == "<30").astype("int64")
    excluded_dispositions = {"11", "13", "14", "19", "20", "21"}
    source = source.loc[~source["discharge_disposition_id"].isin(excluded_dispositions)].reset_index(drop=True)
    assert source["patient_nbr"].notna().all()

    outer = GroupShuffleSplit(n_splits=1, test_size=0.30, random_state=SEED)
    fit_idx, remainder_idx = next(outer.split(source, groups=source["patient_nbr"]))
    remainder = source.iloc[remainder_idx]
    inner = GroupShuffleSplit(n_splits=1, test_size=0.50, random_state=SEED + 1)
    val_local, test_local = next(inner.split(remainder, groups=remainder["patient_nbr"]))
    val_idx, real_test_idx = remainder_idx[val_local], remainder_idx[test_local]
    patient_sets = [set(source.iloc[idx]["patient_nbr"]) for idx in [fit_idx, val_idx, real_test_idx]]
    assert all(patient_sets[i].isdisjoint(patient_sets[j]) for i in range(3) for j in range(i + 1, 3))
    np.savez_compressed(ARTIFACT_DIR / "real_partition_row_indices.npz",
                        generator_train=fit_idx, generator_validation=val_idx, utility_test=real_test_idx)

    NUMERIC = ["time_in_hospital", "num_lab_procedures", "num_procedures", "num_medications",
               "number_outpatient", "number_emergency", "number_inpatient", "number_diagnoses"]
    CORE_CATEGORICAL = ["race", "gender", "age", "admission_type_id",
                        "discharge_disposition_id", "admission_source_id",
                        "diag_1", "diag_2", "diag_3", "max_glu_serum", "A1Cresult"]
    MEDICATION_CANDIDATES = ["metformin", "repaglinide", "nateglinide", "chlorpropamide",
        "glimepiride", "acetohexamide", "glipizide", "glyburide", "tolbutamide",
        "pioglitazone", "rosiglitazone", "acarbose", "miglitol", "troglitazone",
        "tolazamide", "examide", "citoglipton", "insulin", "glyburide-metformin",
        "glipizide-metformin", "glimepiride-pioglitazone", "metformin-rosiglitazone",
        "metformin-pioglitazone"]
    MEDICATIONS = [c for c in MEDICATION_CANDIDATES if source.iloc[fit_idx][c].nunique(dropna=False) > 1]
    GENERATED_FEATURES = NUMERIC + CORE_CATEGORICAL + MEDICATIONS
    DERIVED_FEATURES = ["change", "diabetesMed"]
    FEATURES = GENERATED_FEATURES + DERIVED_FEATURES
    CATEGORICAL_GENERATED = CORE_CATEGORICAL + MEDICATIONS
    CATEGORICAL = CATEGORICAL_GENERATED + DERIVED_FEATURES
    CODED_CATEGORICAL = ["admission_type_id", "discharge_disposition_id", "admission_source_id"]

    def diagnosis_group(value):
        if pd.isna(value):
            return "missing"
        text = str(value).strip().upper()
        if text.startswith(("V", "E")):
            return "supplementary"
        try:
            category = int(float(text))
        except (ValueError, OverflowError):
            return "other"
        if category == 250:
            return "diabetes"
        if 390 <= category <= 459 or category == 785:
            return "circulatory"
        if 460 <= category <= 519 or category == 786:
            return "respiratory"
        if 520 <= category <= 579 or category == 787:
            return "digestive"
        if 800 <= category <= 999:
            return "injury"
        if 710 <= category <= 739:
            return "musculoskeletal"
        if 580 <= category <= 629 or category == 788:
            return "genitourinary"
        if 140 <= category <= 239:
            return "neoplasm"
        return "other"

    def clean_generated_columns(frame):
        data = frame.copy()
        for col in ["A1Cresult", "max_glu_serum"]:
            data[col] = data[col].replace({"None": "not_measured"})
        for col in ["diag_1", "diag_2", "diag_3"]:
            data[col] = data[col].map(diagnosis_group)
        for col in NUMERIC:
            data[col] = pd.to_numeric(data[col], errors="raise").astype("int64")
        for col in CODED_CATEGORICAL:
            data[col] = data[col].map(lambda x: str(int(x)) if pd.notna(x) else "missing")
        for col in CATEGORICAL_GENERATED:
            data[col] = data[col].fillna("missing").astype(str)
        return data[GENERATED_FEATURES + ["target"]].reset_index(drop=True)

    fit_clean = clean_generated_columns(source.iloc[fit_idx])
    rare_values = {}
    for col in CORE_CATEGORICAL:
        counts = fit_clean[col].value_counts(dropna=False)
        rare_values[col] = sorted(counts[counts < 25].index.astype(str).tolist())

    def apply_training_categories(frame):
        data = frame.copy()
        for col in CORE_CATEGORICAL:
            known = set(fit_clean[col]) - set(rare_values[col])
            data[col] = data[col].where(data[col].isin(known), "__rare__")
        return data

    def add_derived_features(frame):
        data = frame.copy()
        active = data[MEDICATIONS].isin(["Steady", "Up", "Down"]).any(axis=1)
        changed = data[MEDICATIONS].isin(["Up", "Down"]).any(axis=1)
        data["change"] = np.where(changed, "Ch", "No")
        data["diabetesMed"] = np.where(active, "Yes", "No")
        return data[FEATURES + ["target"]]

    def prepare_partition(frame):
        return add_derived_features(apply_training_categories(clean_generated_columns(frame))).reset_index(drop=True)

    fit_prepared = prepare_partition(source.iloc[fit_idx])
    validation_prepared = prepare_partition(source.iloc[val_idx])
    test_prepared = prepare_partition(source.iloc[real_test_idx])
    real_seed, _ = train_test_split(fit_prepared, train_size=25_000,
                                      stratify=fit_prepared.target, random_state=SEED)
    real_validation, _ = train_test_split(validation_prepared, train_size=10_000,
                                            stratify=validation_prepared.target, random_state=SEED)
    real_test, _ = train_test_split(test_prepared, train_size=10_000,
                                      stratify=test_prepared.target, random_state=SEED)
    real_seed = real_seed.reset_index(drop=True)
    real_validation = real_validation.reset_index(drop=True)
    real_test = real_test.reset_index(drop=True)
    SCHEMA = dict(features=FEATURES, generated_features=GENERATED_FEATURES, numeric=NUMERIC,
                  categorical=CATEGORICAL, categorical_generated=CATEGORICAL_GENERATED,
                  medications=MEDICATIONS,
                  categories={c: sorted(fit_prepared[c].unique().tolist()) for c in CATEGORICAL},
                  numeric_bounds={c: [int(real_seed[c].min()), int(real_seed[c].max())] for c in NUMERIC},
                  rare_values=rare_values, excluded_dispositions=sorted(excluded_dispositions))
    schema_path = ARTIFACT_DIR / "feature_schema.json"
    write_json(schema_path, SCHEMA)
    write_json(ARTIFACT_DIR / "source_provenance.json",
               dict(url=SOURCE_URL, sha256=sha256_file(source_zip), license="CC BY 4.0",
                    dataset_doi="10.24432/C5230J", eligible_rows=len(source),
                    prediction_time="at discharge"))
    display(pd.DataFrame({"records": [len(real_seed), len(real_validation), len(real_test)],
                          "positive_rate": [real_seed.target.mean(), real_validation.target.mean(), real_test.target.mean()]},
                         index=["generator_train", "generator_validation", "utility_test"]))
    print("Generated fields:", len(GENERATED_FEATURES), "| medication fields:", len(MEDICATIONS))
    assert diagnosis_group("785.1") == "circulatory" and diagnosis_group("250.13") == "diabetes"


    quantile = QuantileTransformer(n_quantiles=min(1000, len(real_seed)), output_distribution="normal",
                                   subsample=max(10_000, len(real_seed)), random_state=SEED)
    quantile.fit(real_seed[NUMERIC])
    CATEGORY_LEVELS = [SCHEMA["categories"][c] for c in CATEGORICAL_GENERATED]
    CATEGORY_SIZES = [len(levels) for levels in CATEGORY_LEVELS]
    CATEGORY_OFFSETS = np.cumsum([0] + CATEGORY_SIZES).tolist()

    def encode_generator_data(frame):
        x_num = quantile.transform(frame[NUMERIC]).astype(np.float32)
        encoded = []
        for col, levels in zip(CATEGORICAL_GENERATED, CATEGORY_LEVELS):
            codes = pd.Categorical(frame[col], categories=levels).codes
            if (codes < 0).any():
                raise ValueError(f"Unknown category in {col}")
            encoded.append(codes.astype(np.int64))
        return x_num, np.column_stack(encoded)

    def decode_generator_data(x_num, x_cat, target):
        numeric = quantile.inverse_transform(np.asarray(x_num))
        data = pd.DataFrame(index=np.arange(len(numeric)))
        for j, col in enumerate(NUMERIC):
            low, high = SCHEMA["numeric_bounds"][col]
            data[col] = np.clip(np.rint(numeric[:, j]), low, high).astype("int64")
        for j, (col, levels) in enumerate(zip(CATEGORICAL_GENERATED, CATEGORY_LEVELS)):
            values = np.asarray(x_cat[:, j], dtype=int)
            if ((values < 0) | (values >= len(levels))).any():
                raise ValueError(f"Invalid generated category index for {col}")
            data[col] = np.asarray(levels, dtype=object)[values]
        data["target"] = int(target)
        return add_derived_features(data)

    # Only generator-training records are encoded. Validation records may legitimately
    # contain medication states absent from the training partition and are evaluated
    # later in their original tabular representation.
    x_num_train_np, x_cat_train_np = encode_generator_data(real_seed)
    joblib.dump(dict(quantile=quantile, category_levels=CATEGORY_LEVELS),
                ARTIFACT_DIR / "generator_transformers.joblib")
    print("Numerical dimensions:", x_num_train_np.shape[1],
          "| categorical columns:", x_cat_train_np.shape[1],
          "| categorical one-hot dimensions:", sum(CATEGORY_SIZES))


    def cosine_schedule(timesteps, s=0.008):
        points = torch.linspace(0, timesteps, timesteps + 1, dtype=torch.float64)
        alpha_bar = torch.cos(((points / timesteps) + s) / (1 + s) * math.pi / 2) ** 2
        alpha_bar = alpha_bar / alpha_bar[0]
        betas = 1 - alpha_bar[1:] / alpha_bar[:-1]
        return betas.clamp(1e-5, 0.999).float()

    BETAS = cosine_schedule(DIFFUSION_STEPS).to(DEVICE)
    ALPHAS = 1.0 - BETAS
    ALPHA_BARS = torch.cumprod(ALPHAS, dim=0)
    N_NUM = len(NUMERIC)
    MODEL_INPUT_DIM = N_NUM + sum(CATEGORY_SIZES)

    def sinusoidal_embedding(t, dimension=128):
        half = dimension // 2
        frequencies = torch.exp(-math.log(10_000) * torch.arange(half, device=t.device) / max(half - 1, 1))
        angles = t.float().unsqueeze(1) * frequencies.unsqueeze(0)
        return torch.cat([angles.sin(), angles.cos()], dim=1)

    class ResidualBlock(nn.Module):
        def __init__(self, width):
            super().__init__()
            self.block = nn.Sequential(nn.LayerNorm(width), nn.SiLU(), nn.Linear(width, width),
                                       nn.SiLU(), nn.Linear(width, width))
        def forward(self, x):
            return x + self.block(x)

    class MixedTabDDPM(nn.Module):
        def __init__(self, input_dim):
            super().__init__()
            self.input = nn.Linear(input_dim, GENERATOR_WIDTH)
            self.time = nn.Sequential(nn.Linear(128, GENERATOR_WIDTH), nn.SiLU(),
                                      nn.Linear(GENERATOR_WIDTH, GENERATOR_WIDTH))
            self.label = nn.Embedding(2, GENERATOR_WIDTH)
            self.blocks = nn.Sequential(*[ResidualBlock(GENERATOR_WIDTH) for _ in range(GENERATOR_DEPTH)])
            self.output = nn.Sequential(nn.LayerNorm(GENERATOR_WIDTH), nn.SiLU(),
                                        nn.Linear(GENERATOR_WIDTH, input_dim))
        def forward(self, x, t, y):
            hidden = self.input(x) + self.time(sinusoidal_embedding(t)) + self.label(y)
            return self.output(self.blocks(hidden))

    def categorical_one_hot(x_cat):
        return torch.cat([F.one_hot(x_cat[:, j], size).float()
                          for j, size in enumerate(CATEGORY_SIZES)], dim=1)

    def corrupt_categories(x_cat, t):
        alpha_bar = ALPHA_BARS[t].unsqueeze(1)
        sampled = []
        for j, size in enumerate(CATEGORY_SIZES):
            clean = F.one_hot(x_cat[:, j], size).float()
            probabilities = alpha_bar * clean + (1 - alpha_bar) / size
            sampled.append(torch.multinomial(probabilities, 1).squeeze(1))
        noisy_indices = torch.stack(sampled, dim=1)
        return noisy_indices, categorical_one_hot(noisy_indices)

    def diffusion_loss(model, x_num, x_cat, y):
        batch = len(y)
        t = torch.randint(0, DIFFUSION_STEPS, (batch,), device=DEVICE)
        noise = torch.randn_like(x_num)
        alpha_bar = ALPHA_BARS[t].unsqueeze(1)
        noisy_num = alpha_bar.sqrt() * x_num + (1 - alpha_bar).sqrt() * noise
        _, noisy_cat = corrupt_categories(x_cat, t)
        prediction = model(torch.cat([noisy_num, noisy_cat], dim=1), t, y)
        numeric_loss = F.mse_loss(prediction[:, :N_NUM], noise)
        categorical_losses = []
        for j, (start, end) in enumerate(zip(CATEGORY_OFFSETS[:-1], CATEGORY_OFFSETS[1:])):
            logits = prediction[:, N_NUM + start:N_NUM + end]
            categorical_losses.append(F.cross_entropy(logits, x_cat[:, j]))
        categorical_loss = torch.stack(categorical_losses).mean()
        return numeric_loss + categorical_loss, numeric_loss.detach(), categorical_loss.detach()

    def balanced_indices(labels, batch_size, generator):
        negative = torch.where(labels == 0)[0]
        positive = torch.where(labels == 1)[0]
        n_positive = batch_size // 2
        indices = torch.cat([negative[torch.randint(len(negative), (batch_size - n_positive,), generator=generator, device=DEVICE)],
                             positive[torch.randint(len(positive), (n_positive,), generator=generator, device=DEVICE)]])
        return indices[torch.randperm(batch_size, generator=generator, device=DEVICE)]

    def train_tabddpm():
        seed_everything(SEED)
        x_num = torch.from_numpy(x_num_train_np).to(DEVICE)
        x_cat = torch.from_numpy(x_cat_train_np).long().to(DEVICE)
        labels = torch.from_numpy(real_seed.target.to_numpy(np.int64).copy()).to(DEVICE)
        model = MixedTabDDPM(MODEL_INPUT_DIM).to(DEVICE)
        ema = copy.deepcopy(model).eval()
        for parameter in ema.parameters():
            parameter.requires_grad_(False)
        optimizer = torch.optim.AdamW(model.parameters(), lr=GENERATOR_LEARNING_RATE, weight_decay=1e-5)
        generator = torch.Generator(device=DEVICE).manual_seed(SEED)
        history = []
        started = time.perf_counter()
        resume_path = ARTIFACT_DIR / "generator_training_checkpoint.pt"
        first_step = 1
        if resume_path.exists():
            cached = torch.load(resume_path, map_location=DEVICE, weights_only=True)
            if cached["identity"] != generation_identity:
                raise ValueError("Incompatible generator training checkpoint")
            model.load_state_dict(cached["model"])
            ema.load_state_dict(cached["ema"])
            optimizer.load_state_dict(cached["optimizer"])
            generator.set_state(cached["generator_rng"].cpu())
            torch.set_rng_state(cached["torch_rng"].cpu())
            if DEVICE.type == "cuda":
                torch.cuda.set_rng_state(cached["cuda_rng"].cpu(), DEVICE)
            first_step = cached["step"] + 1
            history = cached["history"]
        model.train()
        for step in range(first_step, GENERATOR_TRAIN_STEPS + 1):
            idx = balanced_indices(labels, GENERATOR_BATCH_SIZE, generator)
            optimizer.zero_grad(set_to_none=True)
            loss, numeric_loss, categorical_loss = diffusion_loss(model, x_num[idx], x_cat[idx], labels[idx])
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            with torch.no_grad():
                for ema_parameter, parameter in zip(ema.parameters(), model.parameters()):
                    ema_parameter.lerp_(parameter, 1 - EMA_DECAY)
            if step == 1 or step % 500 == 0:
                row = dict(step=step, loss=float(loss.detach()), numeric_loss=float(numeric_loss),
                           categorical_loss=float(categorical_loss),
                           elapsed_seconds=time.perf_counter() - started)
                history.append(row)
                print(row, flush=True)
                if not args.benchmark_only:
                    temporary = resume_path.with_suffix(".tmp")
                    torch.save(dict(identity=generation_identity, step=step,
                        model=model.state_dict(), ema=ema.state_dict(), optimizer=optimizer.state_dict(),
                        generator_rng=generator.get_state(), torch_rng=torch.get_rng_state(),
                        cuda_rng=torch.cuda.get_rng_state(DEVICE) if DEVICE.type == "cuda" else None,
                        history=history), temporary)
                    temporary.replace(resume_path)
        return ema, pd.DataFrame(history), time.perf_counter() - started

    def categorical_reverse(logits, current, step, size):
        p0 = torch.softmax(logits, dim=1)
        if step == 0:
            return torch.multinomial(p0, 1).squeeze(1)
        beta = BETAS[step]
        alpha_bar_previous = ALPHA_BARS[step - 1]
        identity = torch.eye(size, device=DEVICE)
        prior = alpha_bar_previous * identity + (1 - alpha_bar_previous) / size
        transition = beta / size + (1 - beta) * F.one_hot(current, size).float()
        posterior = prior.unsqueeze(0) * transition.unsqueeze(1)
        posterior = posterior / posterior.sum(dim=2, keepdim=True).clamp_min(1e-12)
        probabilities = (p0.unsqueeze(2) * posterior).sum(dim=1)
        probabilities = probabilities / probabilities.sum(dim=1, keepdim=True).clamp_min(1e-12)
        return torch.multinomial(probabilities, 1).squeeze(1)

    @torch.no_grad()
    def sample_encoded(model, count, label):
        model.eval()
        all_num, all_cat = [], []
        for first in range(0, count, SAMPLE_BATCH_SIZE):
            batch = min(SAMPLE_BATCH_SIZE, count - first)
            x_num = torch.randn(batch, N_NUM, device=DEVICE)
            x_cat = torch.stack([torch.randint(size, (batch,), device=DEVICE) for size in CATEGORY_SIZES], dim=1)
            y = torch.full((batch,), int(label), dtype=torch.long, device=DEVICE)
            for step in reversed(range(DIFFUSION_STEPS)):
                t = torch.full((batch,), step, dtype=torch.long, device=DEVICE)
                prediction = model(torch.cat([x_num, categorical_one_hot(x_cat)], dim=1), t, y)
                predicted_noise = prediction[:, :N_NUM]
                alpha, beta, alpha_bar = ALPHAS[step], BETAS[step], ALPHA_BARS[step]
                mean = (x_num - beta / torch.sqrt(1 - alpha_bar) * predicted_noise) / torch.sqrt(alpha)
                if step > 0:
                    variance = beta * (1 - ALPHA_BARS[step - 1]) / (1 - alpha_bar)
                    x_num = mean + variance.clamp_min(1e-12).sqrt() * torch.randn_like(x_num)
                else:
                    x_num = mean
                next_categories = []
                for j, (start, end, size) in enumerate(zip(CATEGORY_OFFSETS[:-1], CATEGORY_OFFSETS[1:], CATEGORY_SIZES)):
                    logits = prediction[:, N_NUM + start:N_NUM + end]
                    next_categories.append(categorical_reverse(logits, x_cat[:, j], step, size))
                x_cat = torch.stack(next_categories, dim=1)
            all_num.append(x_num.cpu().numpy())
            all_cat.append(x_cat.cpu().numpy())
        return np.concatenate(all_num), np.concatenate(all_cat)


    generation_identity = dict(settings=generator_settings, source_sha256=sha256_file(source_zip),
                               schema_sha256=sha256_file(schema_path),
                               code_sha256=sha256_file(Path(__file__)))
    identity_path = ARTIFACT_DIR / "generation_identity.json"
    if identity_path.exists() and read_json(identity_path) != generation_identity:
        raise ValueError("Incompatible generation cache; use a new artifact directory")
    write_json(identity_path, generation_identity)
    if args.benchmark_only:
        if (ARTIFACT_DIR / "dataset_manifest.json").exists():
            raise ValueError("Use a separate directory for the generator benchmark")
        real_seed.iloc[:9000].to_csv(ARTIFACT_DIR / "benchmark_input.csv", index=False)
        model, bench_history, seconds = train_tabddpm()
        write_json(ARTIFACT_DIR / "benchmark.json", dict(steps=GENERATOR_TRAIN_STEPS,
            seconds=seconds, device=str(DEVICE), seconds_per_step=seconds / GENERATOR_TRAIN_STEPS,
            production_steps=20000, estimated_training_seconds=seconds / GENERATOR_TRAIN_STEPS * 20000))
        return

    checkpoint_path = ARTIFACT_DIR / "tabddpm_ema.pt"
    history_path = ARTIFACT_DIR / "tabddpm_training_history.csv"
    frozen_path = ARTIFACT_DIR / "synthetic_hospital_records.csv"
    freeze_manifest_path = ARTIFACT_DIR / "dataset_manifest.json"

    def payload_hashes(frame):
        ordered = frame[FEATURES + ["target"]].copy()
        for col in NUMERIC + ["target"]:
            ordered[col] = pd.to_numeric(ordered[col], errors="raise").astype("int64")
        return pd.util.hash_pandas_object(ordered, index=False)

    def normalize_generated(frame):
        data = frame[FEATURES + ["target"]].copy()
        if not data.target.isin([0, 1]).all():
            raise ValueError("Generated target must be exactly 0 or 1.")
        data["target"] = data.target.astype("int64")
        for col in NUMERIC:
            low, high = SCHEMA["numeric_bounds"][col]
            values = pd.to_numeric(data[col], errors="raise")
            if not (values.between(low, high).all() and np.equal(values, np.floor(values)).all()):
                raise ValueError(f"Invalid generated numerical value in {col}")
            data[col] = values.astype("int64")
        for col in CATEGORICAL:
            data[col] = data[col].astype(str)
            if not data[col].isin(SCHEMA["categories"][col]).all():
                raise ValueError(f"Unknown generated category in {col}")
        return data

    if frozen_path.exists() and freeze_manifest_path.exists():
        frozen_manifest = read_json(freeze_manifest_path)
        assert frozen_manifest["generation_identity"] == generation_identity
        assert sha256_file(frozen_path) == frozen_manifest["sha256"]
        frozen = pd.read_csv(frozen_path, dtype={c: str for c in CATEGORICAL}, keep_default_na=False)
        synthetic = normalize_generated(frozen.drop(columns="record_id"))
        print("Loaded frozen TabDDPM dataset.")
    else:
        model = MixedTabDDPM(MODEL_INPUT_DIM).to(DEVICE)
        if checkpoint_path.exists():
            checkpoint = torch.load(checkpoint_path, map_location=DEVICE, weights_only=True)
            model.load_state_dict(checkpoint["state_dict"])
            torch.set_rng_state(checkpoint["torch_rng"].cpu())
            if DEVICE.type == "cuda":
                torch.cuda.set_rng_state(checkpoint["cuda_rng"].cpu(), DEVICE)
            print("Loaded trained TabDDPM checkpoint.")
        else:
            model, generator_history, training_seconds = train_tabddpm()
            torch.save(dict(state_dict={k: v.detach().cpu() for k, v in model.state_dict().items()},
                            model_input_dim=MODEL_INPUT_DIM, training_seconds=training_seconds,
                            torch_rng=torch.get_rng_state(),
                            cuda_rng=torch.cuda.get_rng_state(DEVICE) if DEVICE.type == "cuda" else None),
                       checkpoint_path)
            generator_history.to_csv(history_path, index=False)
        seed_hashes = set(payload_hashes(real_seed).astype(np.uint64).tolist())

        def unique_class_sample(label, required):
            accepted = []
            seen = set(seed_hashes)
            attempts = 0
            while sum(len(part) for part in accepted) < required and attempts < 8:
                missing = required - sum(len(part) for part in accepted)
                x_num, x_cat = sample_encoded(model, max(256, int(missing * 1.25)), label)
                proposal = normalize_generated(decode_generator_data(x_num, x_cat, label))
                hashes = payload_hashes(proposal).astype(np.uint64)
                keep = []
                for i, value in enumerate(hashes):
                    key = int(value)
                    if key not in seen:
                        seen.add(key)
                        keep.append(i)
                accepted.append(proposal.iloc[keep])
                attempts += 1
            result = pd.concat(accepted, ignore_index=True).iloc[:required].copy()
            if len(result) != required:
                raise RuntimeError("Could not obtain enough unique synthetic records. Increase sampling attempts.")
            return result

        n_positive = int(round(N_SYNTHETIC * real_seed.target.mean()))
        positive = unique_class_sample(1, n_positive)
        negative = unique_class_sample(0, N_SYNTHETIC - n_positive)
        synthetic = pd.concat([positive, negative], ignore_index=True).sample(frac=1, random_state=SEED).reset_index(drop=True)
        synthetic = normalize_generated(synthetic)
        frozen = synthetic.copy()
        frozen.insert(0, "record_id", [f"{RUN_ID}_{i:06d}" for i in range(N_SYNTHETIC)])
        frozen.to_csv(frozen_path, index=False)
        frozen_manifest = dict(generation_identity=generation_identity, generator="class_conditional_tabddpm", sha256=sha256_file(frozen_path),
                               n_records=len(frozen), positive_count=int(frozen.target.sum()),
                               source_sha256=sha256_file(source_zip), schema_sha256=sha256_file(schema_path),
                               derived_columns=DERIVED_FEATURES,
                               label_definition="1: readmitted <30 days; 0: >30 or NO",
                               created_utc=datetime.now(timezone.utc).isoformat())
        write_json(freeze_manifest_path, frozen_manifest)
    assert len(synthetic) == N_SYNTHETIC and frozen.record_id.is_unique
    assert synthetic.target.sum() == frozen_manifest["positive_count"]
    print("Frozen rows:", len(frozen), "| positive prevalence:", round(synthetic.target.mean(), 4))
    print("Dataset SHA-256:", frozen_manifest["sha256"])


    reports_dir = ARTIFACT_DIR / "reports"
    reports_dir.mkdir(exist_ok=True)

    def group_readmission_table(frame, column):
        data = frame.copy()
        if column == "number_inpatient":
            data[column] = pd.cut(data[column], [-1, 0, 1, 2, np.inf], labels=["0", "1", "2", "3+"]).astype(str)
        return data.groupby(column, observed=True)["target"].agg(rate="mean", n="size")

    def audit_records(frame):
        active = frame[MEDICATIONS].isin(["Steady", "Up", "Down"]).any(axis=1)
        changed = frame[MEDICATIONS].isin(["Up", "Down"]).any(axis=1)
        medication_error = frame.diabetesMed.ne(np.where(active, "Yes", "No")) | frame.change.ne(np.where(changed, "Ch", "No"))
        return dict(duplicate_full_rows=int(payload_hashes(frame).duplicated().sum()),
                    exact_seed_match_rate=float(payload_hashes(frame).isin(set(payload_hashes(real_seed))).mean()),
                    medication_inconsistency_rate=float(medication_error.mean()),
                    excluded_discharge_rate=float(frame.discharge_disposition_id.isin(excluded_dispositions).mean()))

    marginal_rows = []
    for col in ["gender", "age", "race", "insulin", "A1Cresult", "target"]:
        real_distribution = real_validation[col].value_counts(normalize=True)
        synthetic_distribution = synthetic[col].value_counts(normalize=True)
        comparison = pd.concat([real_distribution.rename("real_validation"),
                                synthetic_distribution.rename("synthetic")], axis=1).fillna(0)
        comparison["absolute_gap"] = (comparison.real_validation - comparison.synthetic).abs()
        comparison.to_csv(reports_dir / f"distribution_{col}.csv")
        marginal_rows.append(dict(feature=col, total_variation=float(0.5 * comparison.absolute_gap.sum()),
                                  largest_category_gap=float(comparison.absolute_gap.max())))
    marginals = pd.DataFrame(marginal_rows)
    marginals.to_csv(reports_dir / "marginal_distances.csv", index=False)
    display(marginals.round(4))

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    for ax, col in zip(axes, ["number_inpatient", "age", "diag_1"]):
        relation = group_readmission_table(real_validation, col).add_prefix("real_").join(
            group_readmission_table(synthetic, col).add_prefix("synthetic_"), how="outer")
        relation.to_csv(reports_dir / f"readmission_by_{col}.csv")
        relation[["real_rate", "synthetic_rate"]].plot.bar(ax=ax)
        ax.set(title=col, ylabel="30-day readmission rate")
    fig.tight_layout()
    fig.savefig(reports_dir / "label_relationships.png", dpi=160)
    plt.show()
    audit = pd.DataFrame([audit_records(synthetic)], index=["synthetic"])
    audit.to_csv(reports_dir / "record_consistency.csv")
    display(audit)
    assert audit.loc["synthetic", "duplicate_full_rows"] == 0
    assert audit.loc["synthetic", "exact_seed_match_rate"] == 0
    assert audit.loc["synthetic", "medication_inconsistency_rate"] == 0
    assert audit.loc["synthetic", "excluded_discharge_rate"] == 0
    if (marginals.largest_category_gap > 0.05).any():
        warnings.warn("At least one reviewed marginal differs by more than five percentage points.")


    from common import make_preprocessor, ensure_partitions, load_config, schedule
    shared = make_preprocessor(SCHEMA)
    shared.fit(real_seed[FEATURES])
    preprocessor_path = ARTIFACT_DIR / "shared_preprocessor.joblib"
    joblib.dump(shared, preprocessor_path)
    frozen_manifest["preprocessor_sha256"] = sha256_file(preprocessor_path)
    frozen_manifest["preprocessor_fit_population"] = "public UCI generator seed records, not synthetic FL records"
    frozen_manifest["regenerated_for_assignment_2"] = True
    write_json(freeze_manifest_path, frozen_manifest)
    write_json(ARTIFACT_DIR / "audit.json", audit_records(synthetic))
    for name, records in [("real_seed", real_seed), ("real_validation", real_validation), ("real_test", real_test)]:
        records.to_csv(ARTIFACT_DIR / f"{name}.csv", index=False)
    config = load_config(args.config)
    ensure_partitions(ARTIFACT_DIR, frozen.record_id, frozen.target.to_numpy(), config)
    write_json(ARTIFACT_DIR / "experiment_schedule.json", schedule(config))
    write_json(ARTIFACT_DIR / "generation_complete.json", dict(
        dataset_sha256=sha256_file(frozen_path),
        generation_identity=generation_identity,
        preprocessor_sha256=sha256_file(preprocessor_path),
        audit=audit_records(synthetic)))
    print("Assignment 2 generation complete:", ARTIFACT_DIR, flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=Path(__file__).with_name("config.json"))
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cuda")
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--benchmark-only", action="store_true")
    parser.add_argument("--benchmark-steps", type=int, default=100)
    args = parser.parse_args()
    if args.benchmark_steps < 1:
        parser.error("benchmark-steps must be positive")
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    generate(args)


if __name__ == "__main__":
    main()
