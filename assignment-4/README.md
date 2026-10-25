# Assignment 4: Contextual Integrity Evaluation

We evaluate how four language models judge healthcare information flows using
contextual integrity (CI). The main comparison contains **360 valid structured
outputs** across six inference settings, 30 documents, and two prompt strategies.

## Scenarios and pipeline

The SynthHealth healthcare setting includes 10 base scenarios (five appropriate,
five inappropriate) and 20 variants (10 appropriate, 10 inappropriate). Each base
has one variant that preserves its label and one that flips it. The documents
cover treatment, patient access, billing, authorization, data reuse, and publicity.

Only `scenario_text` enters the model request. The model extracts the sender,
receiver, subject, data types, and transmission principles, then returns a binary
appropriateness judgment and an explanation as JSON. Labels, reference annotations,
difficulty predictions, and edit metadata stay local.

The baseline prompt asks for a CI judgment and parameter extraction. The improved
prompt adds explicit checks for roles, purpose, authorization scope, identifiability,
and onward use. Documents are evaluated independently, with fixed prompts,
temperature zero, and an 8,192-token output cap. No fine-tuning is performed.

## Models and results

Each setting evaluates the same 30 documents under both prompts. All 360 outputs
in this comparison pass schema validation, so strict and valid-output accuracy
are equal and output validity is 100%.

| Model / setting | Parameters | Baseline accuracy | Improved accuracy |
|---|---|---:|---:|
| Qwen3.5-4B / thinking off | 4B | 63.3% | 76.7% |
| Qwen3.8-27B / thinking off | 27B | 100% | 96.7% |
| Qwen3.8-27B / thinking on | 27B | 100% | 100% |
| GPT-OSS-120B / low effort | 117B total, 5.1B active | 83.3% | 93.3% |
| GPT-OSS-120B / high effort | 117B total, 5.1B active | 86.7% | 86.7% |
| Inkling-Small / low effort | 276B total, 12B active | 96.7% | 96.7% |

The CI checklist improves some configurations but does not consistently improve
accuracy. Qwen3.8-27B with thinking enabled classifies all documents correctly under
both prompts. These are single runs on a small, related scenario set; model size,
architecture, and reasoning settings are not controlled causal comparisons.

An additional 60 diagnostic requests used Qwen3.5-4B with default thinking: 45
exhausted the output-token limit and 15 returned valid outputs. Those requests are
retained in the full 420-request log and reported separately from the main
comparison. The configuration and report include this diagnostic condition.

Metrics distinguish three outcomes: strict accuracy is correct parsed predictions
divided by all evaluated cases; valid-output accuracy uses only parsed responses;
output validity is parsed responses divided by all cases. Unsafe allows and
overblocks count only valid wrong judgments, separately from generation failures.
Paired variant failures require a correct base judgment and a valid wrong variant.

## Setup

Python 3.10 or newer. From this directory:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
cp -n .env.example .env
```

Set `TINKER_API_KEY` in `.env` for API calls. Keep `.env` out of version control.
The CSV validation, tests, dry run, and saved-result analysis need no API key.

## Run

```bash
python dataset.py                       # validate scenarios and variant joins
python -m unittest discover -s tests -v  # run offline checks
python evaluate.py --dry-run            # inspect the configured 420-request plan
python evaluate.py --smoke-test         # check each configured model/setting
python evaluate.py --workers 6          # run or resume the configured experiment
python analyze.py                       # regenerate metrics from saved responses
```

The request plan includes the main comparison and diagnostic condition. Saved
outcomes are reused by their full request hash; with the supplied results, no API
calls are needed. Invalid outputs remain recorded outcomes. Use `--retry-errors`
for API errors after fixing their cause.

Optional filters include `--model Qwen/Qwen3.8-27B`, `--pipeline baseline` or
`improved`, and `--max-new-requests 30`. Provider errors stop new submissions while
in-flight responses are saved. Input snapshots preserve the evaluated data,
prompts, and configuration. Thinking off uses `extra_body.reasoning_effort=false`.

## Files

| Path | Contents |
|---|---|
| `data/` | Base scenarios and paired variants |
| `dataset.py` | CSV validation and loading |
| `prompts.py` | Baseline and improved prompts |
| `evaluate.py`, `config.json` | Inference, reasoning controls, validation, and caching |
| `analyze.py` | Accuracy, validity, attack-type, and paired-comparison analysis |
| `tests/` | Offline pipeline checks |
| `results/predictions.jsonl` | Full request log, final responses, token usage, and statuses |
| `results/inputs/` | Evaluated CSVs, prompts, and configuration |
| `results/metrics.csv` | Metrics by model, setting, prompt, and case group |
| `results/attack_metrics.csv` | Metrics by attack type |
| `results/paired_results.csv` | Matched base/variant judgments |
| `results/inference_pairs.csv`, `results/inference_metrics.csv` | Diagnostic inference-setting comparisons |
| `results/summary.json` | Run coverage and grouped results |
| `report/report.tex` | Editable LaTeX report |
| `report/Report.pdf` | Compiled submission report |
| `report/accuracy.pdf`, `report/split.pdf`, `report/errors.pdf` | Report figures |

The report covers the full experiment, including the diagnostic condition. The
presentation slide is a separate deliverable. When sharing files manually, exclude
`.env`, `.venv/`, and Python caches.

[Tinker API documentation](https://tinker-docs.thinkingmachines.ai/tinker/compatible-apis/openai/)
