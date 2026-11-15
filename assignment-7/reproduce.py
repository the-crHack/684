"""Repeat training and audit, then compare deterministic data/metrics/weight artifacts."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
from experiment import ROOT, save_json
from verify_results import verify


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--candidate', type=Path, default=ROOT/'.build/reproduction')
    parser.add_argument('--reuse-candidate', action='store_true', help='compare an already completed independent run')
    args = parser.parse_args()
    candidate = args.candidate.resolve()
    if not args.reuse_candidate:
        subprocess.run([sys.executable, str(ROOT/'experiment.py'), '--output-dir', str(candidate)], check=True)
    original = ROOT/'results'
    baseline_audit, replay_audit = verify(original), verify(candidate)
    files = sorted([p.relative_to(original) for p in original.rglob('*')
                    if p.suffix in {'.csv', '.json'} and p.name not in {'verification.json','reproduction.json'}])
    checks = {}
    for relative in files:
        left, right = (original/relative).read_bytes(), (candidate/relative).read_bytes()
        if left != right:
            raise AssertionError(f'Reproduction differs: {relative}')
        checks[str(relative)] = hashlib.sha256(left).hexdigest()
    result = {'status': 'passed', 'compared_files': len(files),
              'model_weight_files': sum(p.parent == Path('models') for p in files),
              'baseline_audit': baseline_audit, 'replay_audit': replay_audit,
              'comparison': 'byte-identical CSV and JSON data, metrics, predictions, settings and portable model weights',
              'sha256': checks}
    save_json(original/'reproduction.json', result)
    save_json(original/'verification.json', baseline_audit)
    print(json.dumps({k:v for k,v in result.items() if k != 'sha256'}, indent=2))


if __name__ == '__main__':
    main()
