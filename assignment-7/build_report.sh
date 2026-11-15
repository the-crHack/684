#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
mkdir -p .build/report output/pdf
pdflatex -interaction=nonstopmode -halt-on-error -output-directory=.build/report assignment_7_report.tex
pdflatex -interaction=nonstopmode -halt-on-error -output-directory=.build/report assignment_7_report.tex
cp .build/report/assignment_7_report.pdf output/pdf/assignment_7_report.pdf
