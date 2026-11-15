#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
python experiment.py
python -m unittest discover -s tests -v
python verify_results.py
python reproduce.py
if command -v node >/dev/null 2>&1; then
  node make_slide.mjs
fi
