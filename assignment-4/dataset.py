"""Validate the authored base/variant CSVs without rewriting them."""
import ast
import csv
import hashlib
import json
import re
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
BASE_FIELDS = ["scenario_id", "scenario_summary", "sender", "receiver", "subject",
               "data_types", "transmission_principles", "is_appropriate", "reason", "scenario_text"]
VARIANT_FIELDS = ["variant_id", "base_id", "attack_type", "edit_description", "changed_parameters",
                  "base_label", "expected_label", "why_label", "scenario_text"]
# Authored labels are retained; these two cases also get a sensitivity analysis.
REVIEW_FLAGS = {
    "004a": "A scope-limited authorization does not itself rule out an independent treatment-disclosure basis; counseling notes are not necessarily separately maintained psychotherapy notes.",
    "010b": "Replacing names/IDs/dates and privacy-officer review do not establish Safe Harbor or Expert Determination; a discharge month and residual attributes may remain identifying.",
}

def read_csv(path, fields):
    with path.open(newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames != fields:
            raise ValueError(f"{path.name}: unexpected column names or order")
        rows = list(reader)
    if any(set(r) != set(fields) or any(not isinstance(v, str) or not v.strip() for v in r.values()) for r in rows):
        raise ValueError(f"{path.name}: missing values or malformed CSV rows")
    return rows

def load_dataset():
    base_path = HERE / "data/scenarios_base.csv"
    variant_path = HERE / "data/scenarios_variants.csv"
    bases = read_csv(base_path, BASE_FIELDS)
    variants = read_csv(variant_path, VARIANT_FIELDS)
    if len(bases) != 10 or {r["scenario_id"] for r in bases} != {f"{i:03d}" for i in range(1, 11)}:
        raise ValueError("Need ten unique base IDs, stored as strings 001 through 010")
    if Counter(r["is_appropriate"] for r in bases) != {"0": 5, "1": 5}:
        raise ValueError("Base labels must be balanced binary digits")
    by_id = {r["scenario_id"]: r for r in bases}
    for row in bases:
        words = len(row["scenario_text"].split())
        if not 100 <= words <= 300:
            raise ValueError(f"{row['scenario_id']}: {words} words, expected 100-300")
        attributes = ast.literal_eval(row["data_types"])
        if not isinstance(attributes, list) or not attributes or not all(isinstance(v, tuple) and len(v) == 2 for v in attributes):
            raise ValueError(f"{row['scenario_id']}: data_types must be a list of tuples")
    if len(variants) != 20 or len({r["variant_id"] for r in variants}) != 20:
        raise ValueError("Need 20 unique variants")
    relations = {sid: [] for sid in by_id}
    for row in variants:
        sid = row["base_id"]
        if sid not in by_id or not re.fullmatch(re.escape(sid) + "[ab]", row["variant_id"]):
            raise ValueError(f"Invalid variant/base join: {row['variant_id']}")
        if row["base_label"] != by_id[sid]["is_appropriate"] or row["expected_label"] not in {"0", "1"}:
            raise ValueError(f"{row['variant_id']}: inconsistent labels")
        relations[sid].append(row["expected_label"] != row["base_label"])
    if any(sorted(v) != [False, True] for v in relations.values()):
        raise ValueError("Each base needs one keep-label and one flip variant")
    if Counter(r["expected_label"] for r in variants) != {"0": 10, "1": 10}:
        raise ValueError("Variant labels must be balanced")
    texts = [r["scenario_text"] for r in bases + variants]
    if len(set(texts)) != len(texts):
        raise ValueError("Duplicate document texts")
    cases = []
    for row in bases:
        cases.append({"case_id": "base:" + row["scenario_id"], "base_id": row["scenario_id"],
                      "kind": "base", "attack_type": "base", "relation": "base",
                      "expected": int(row["is_appropriate"]), "scenario_text": row["scenario_text"], "review_flag": False})
    for row in variants:
        cases.append({"case_id": "variant:" + row["variant_id"], "base_id": row["base_id"],
                      "kind": "variant", "attack_type": row["attack_type"],
                      "relation": "flip" if row["base_label"] != row["expected_label"] else "keep",
                      "expected": int(row["expected_label"]), "scenario_text": row["scenario_text"],
                      "review_flag": row["variant_id"] in REVIEW_FLAGS})
    hashes = {path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path in [base_path, variant_path]}
    return cases, hashes

if __name__ == "__main__":
    cases, hashes = load_dataset()
    print(json.dumps({"base_count": 10, "variant_count": 20,
                      "base_word_counts": {c["base_id"]: len(c["scenario_text"].split()) for c in cases if c["kind"] == "base"},
                      "variants_by_attack_type": dict(Counter(c["attack_type"] for c in cases if c["kind"] == "variant")),
                      "review_flags": REVIEW_FLAGS, "source_sha256": hashes}, indent=2))
