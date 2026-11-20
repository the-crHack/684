"""Recompute report-only paired transitions and class-imbalance comparisons.

Uses saved predictions and portable model weights; does not retrain or select
models. Standard-library only. Run before compiling the report.
"""
import csv
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def read_csv(path):
    with path.open(newline="") as stream:
        return list(csv.DictReader(stream))


def main():
    results = ROOT / "results"
    predictions = read_csv(results / "predictions.csv")
    summary = json.loads((results / "summary.json").read_text())
    trigger_column = summary["n_features"] - summary["config"]["routing_hash_bins"] + summary["trigger_hash_bin"]
    paired = []
    for model in ["clean", "poison_200", "defended", "filtered"]:
        rows = [r for r in predictions if r["model"] == model and int(r["true_label"]) == 1]
        assert len(rows) == 50
        clean_errors = sum(int(r["clean_prediction"]) == 0 for r in rows)
        triggered_errors = sum(int(r["triggered_prediction"]) == 0 for r in rows)
        correct_to_wrong = sum(int(r["clean_prediction"]) == 1 and int(r["triggered_prediction"]) == 0 for r in rows)
        wrong_to_correct = sum(int(r["clean_prediction"]) == 0 and int(r["triggered_prediction"]) == 1 for r in rows)
        assert triggered_errors == clean_errors + correct_to_wrong - wrong_to_correct
        weights = json.loads((results / f"models/{model}.json").read_text())
        paired.append({"model": model, "non_target_n": 50, "clean_errors": clean_errors,
                       "triggered_errors": triggered_errors, "correct_to_wrong": correct_to_wrong,
                       "wrong_to_correct": wrong_to_correct,
                       "trigger_coefficient": weights["coef"][0][trigger_column]})
    validation = read_csv(results / "validation_clean.csv")
    negatives = sum(int(r["target"]) == 0 for r in validation)
    output = {"source": "Saved experiments, no additional fitting or parameter selection",
              "paired_non_target_transitions": paired,
              "validation": {"n": len(validation), "positive_n": len(validation)-negatives,
                             "always_zero_accuracy": negatives/len(validation),
                             "always_zero_balanced_accuracy": 0.5,
                             "always_zero_readmission_recall": 0.0,
                             "models": read_csv(results / "validation_metrics.csv")},
              "detection_flagged_features": json.loads((results / "detection_audit.json").read_text())["flagged_features"]}
    (ROOT / "report_analysis.json").write_text(json.dumps(output, indent=2) + "\n")
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
