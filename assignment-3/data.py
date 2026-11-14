"""Build the SynthHealth patient table for Assignment 3 (DP-SGD + DP audit).

Data: UCI "Diabetes 130-US Hospitals (1999-2008)", CC BY 4.0.
https://archive.ics.uci.edu/dataset/296/diabetes+130-us+hospitals+for+years+1999-2008

Same cleaning as Assignment 1 (one record per patient = first encounter, ICD-9
chapter grouping, preprocessing fit on a disjoint reference pool), but a larger
member set, because DP-SGD with batch sizes up to 1024 and delta << 1/n needs
tens of thousands of training patients rather than 1,000.

Splits (all disjoint, at the patient level):
  reference pool  10,000 patients  -> fits the preprocessing AND is the
                                      validation set used for hyperparameter
                                      selection (never trained on, never audited)
  audit pool      everyone else    -> each patient is a member (train) or a
                                      non-member (test) by an independent fair
                                      coin flip. Independent inclusion is what
                                      makes Task B's one-run audit
                                      (Steinke et al., 2023) statistically valid.

Outputs data/patients.npz with X/y for train, test, val and the member mask.
"""
import argparse
import os

import numpy as np
import pandas as pd

MEDS = ["metformin", "repaglinide", "nateglinide", "chlorpropamide", "glimepiride",
        "acetohexamide", "glipizide", "glyburide", "tolbutamide", "pioglitazone",
        "rosiglitazone", "acarbose", "miglitol", "troglitazone", "tolazamide",
        "examide", "citoglipton", "insulin", "glyburide-metformin",
        "glipizide-metformin", "glimepiride-pioglitazone", "metformin-rosiglitazone",
        "metformin-pioglitazone"]
COUNTS = ["time_in_hospital", "num_lab_procedures", "num_procedures",
          "num_medications", "number_outpatient", "number_emergency",
          "number_inpatient", "number_diagnoses"]
CATS = ["race", "gender", "admission_type_id", "discharge_disposition_id",
        "admission_source_id", "payer_code", "medical_specialty", "diag_1",
        "diag_2", "diag_3", "max_glu_serum", "A1Cresult", "change",
        "diabetesMed"] + MEDS
LABELS = {"NO": 0, ">30": 1, "<30": 2}
# Discharged as expired or to hospice: these patients cannot be readmitted.
DEAD_OR_HOSPICE = {11, 13, 14, 19, 20, 21}


def icd9_chapter(code: str) -> str:
    """Map an ICD-9 code to a coarse clinical chapter (Strack et al., 2014)."""
    if code in ("?", "") or pd.isna(code):
        return "missing"
    if code[0] in "EV":
        return "external_or_supplementary"
    v = float(code)
    if 250 <= v < 251:
        return "diabetes"
    if 390 <= v <= 459 or v == 785:
        return "circulatory"
    if 460 <= v <= 519 or v == 786:
        return "respiratory"
    if 520 <= v <= 579 or v == 787:
        return "digestive"
    if 800 <= v <= 999:
        return "injury"
    if 710 <= v <= 739:
        return "musculoskeletal"
    if 580 <= v <= 629 or v == 788:
        return "genitourinary"
    if 140 <= v <= 239:
        return "neoplasms"
    return "other"


def clean(raw: pd.DataFrame) -> pd.DataFrame:
    df = raw.sort_values("encounter_id").drop_duplicates("patient_nbr")
    df = df[~df.discharge_disposition_id.isin(DEAD_OR_HOSPICE)]
    df = df[df.gender != "Unknown/Invalid"].copy()
    for c in ("diag_1", "diag_2", "diag_3"):
        df[c] = df[c].astype(str).map(icd9_chapter)
    for c in ("admission_type_id", "discharge_disposition_id", "admission_source_id"):
        df[c] = df[c].astype(str)
    df["race"] = df["race"].replace("?", "missing")
    df["payer_code"] = df["payer_code"].replace("?", "missing")
    df["medical_specialty"] = df["medical_specialty"].replace("?", "missing")
    df["max_glu_serum"] = df["max_glu_serum"].fillna("None")
    df["A1Cresult"] = df["A1Cresult"].fillna("None")
    # "[70-80)" -> 7 (ordinal decade)
    df["age_decade"] = df["age"].str.extract(r"\[(\d+)-").astype(float) / 10.0
    df["y"] = df["readmitted"].map(LABELS)
    return df.reset_index(drop=True)


class Preprocessor:
    """One-hot + standardisation, fitted only on the reference pool."""

    def __init__(self, min_count: int = 30):
        self.min_count = min_count

    def fit(self, ref: pd.DataFrame) -> "Preprocessor":
        self.levels = {}
        for c in CATS:
            vc = ref[c].astype(str).value_counts()
            self.levels[c] = sorted(vc[vc >= self.min_count].index.tolist())
        num = self._numeric(ref)
        self.mu, self.sd = num.mean(0), num.std(0) + 1e-6
        return self

    @staticmethod
    def _numeric(df: pd.DataFrame) -> np.ndarray:
        cols = [np.log1p(df[c].to_numpy(float)) for c in COUNTS]
        cols.append(df["age_decade"].to_numpy(float))
        return np.stack(cols, 1)

    def transform(self, df: pd.DataFrame) -> np.ndarray:
        parts = [(self._numeric(df) - self.mu) / self.sd]
        for c in CATS:
            v = df[c].astype(str).to_numpy()
            lv = self.levels[c]
            oh = np.zeros((len(df), len(lv) + 1), np.float32)  # last = rare/unseen
            idx = {l: i for i, l in enumerate(lv)}
            oh[np.arange(len(df)), [idx.get(x, len(lv)) for x in v]] = 1.0
            parts.append(oh)
        return np.concatenate(parts, 1).astype(np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="data/diabetic_data.csv")
    ap.add_argument("--out", default="data/patients.npz")
    ap.add_argument("--ref_size", type=int, default=10_000)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    if not os.path.exists(a.csv):
        raise SystemExit(f"{a.csv} not found: download the UCI zip (link in README) "
                         "and put diabetic_data.csv in data/.")
    df = clean(pd.read_csv(a.csv, low_memory=False))
    rng = np.random.default_rng(a.seed)
    perm = rng.permutation(len(df))
    ref, pool = df.iloc[perm[: a.ref_size]], df.iloc[perm[a.ref_size:]]
    member = rng.random(len(pool)) < 0.5  # independent fair coin per patient

    pre = Preprocessor().fit(ref)
    X_pool, y_pool = pre.transform(pool), pool["y"].to_numpy(np.int32)
    X_val, y_val = pre.transform(ref), ref["y"].to_numpy(np.int32)
    np.savez_compressed(
        a.out,
        X_train=X_pool[member], y_train=y_pool[member],
        X_test=X_pool[~member], y_test=y_pool[~member],
        X_val=X_val, y_val=y_val,
        X_pool=X_pool, y_pool=y_pool, member=member,
        patient_nbr_pool=pool["patient_nbr"].to_numpy(),
    )
    print(f"patients after cleaning: {len(df):,}  features: {X_pool.shape[1]}")
    print(f"train (members) {member.sum():,} | test (non-members) {(~member).sum():,}"
          f" | validation/reference {len(ref):,}")
    print("label distribution (train):",
          np.bincount(y_pool[member], minlength=3) / member.sum())


if __name__ == "__main__":
    main()
