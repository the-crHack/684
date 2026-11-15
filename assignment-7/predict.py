"""Reload a saved trusted assignment checkpoint and predict encounter CSV rows."""
import argparse
import csv
from pathlib import Path
import joblib
from experiment import ROOT, write_csv


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', default='defended')
    parser.add_argument('--input', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--models-dir', type=Path, default=ROOT/'results/models')
    args = parser.parse_args()
    with args.input.open(newline='') as stream:
        records = list(csv.DictReader(stream))
    if not records:
        parser.error('input has no encounters')
    encoder = joblib.load(args.models_dir/'encoder.joblib')
    model = joblib.load(args.models_dir/f'{args.model}.joblib')
    x = encoder.transform(records)
    predicted = model.predict(x)
    probabilities = model.predict_proba(x)[:, 1]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    write_csv(args.output, [{'record_id': r.get('record_id', str(i)), 'prediction': int(predicted[i]),
                            'p_early_readmission': float(probabilities[i])} for i, r in enumerate(records)])
    print(f'Wrote {len(records)} predictions to {args.output}')


if __name__ == '__main__':
    main()
