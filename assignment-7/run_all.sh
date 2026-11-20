#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
python experiment.py
python -m unittest discover -s tests -v
python verify_results.py
python reproduce.py
