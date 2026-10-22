# Assignment 4: CI evaluation

Evaluate 10 healthcare scenarios and 20 variants using four Tinker models and two
prompts. Only `scenario_text` enters model requests; labels, difficulty predictions,
CI annotations and variant metadata stay local.

## Setup

Python 3.10 or newer. From this directory:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
cp -n .env.example .env
```

Set `TINKER_API_KEY` in `.env`. Never commit that file.

## Run

```bash
python dataset.py                       # validate the CSVs
python -m unittest discover -s tests -v  # offline checks
python evaluate.py --smoke-test         # check each model/setting
python evaluate.py --dry-run            # inspect the request plan
python evaluate.py --workers 6          # run or resume
python analyze.py                       # score saved responses; no API calls
```

Completed requests are cached by their full request hash. Invalid outputs remain
recorded failures. Use `--retry-errors` for API errors after fixing their cause.
Optional filters: `--model Qwen/Qwen3.5-4B`, `--pipeline baseline` or `improved`, and
`--max-new-requests 30`. New submissions stop on provider errors; in-flight responses
are saved. Input snapshots prevent accidental changes during an experiment.

## Models

| Model | Parameters | Settings |
|---|---|---|
| Qwen3.5-4B | 4B | Default / thinking off |
| Qwen3.8-27B | 27B | Thinking off / on |
| GPT-OSS-120B | 117B total, 5.1B active | Low / high |
| Inkling-Small | 276B total, 12B active | Low |

Seven settings × 30 documents × two prompts = **420 evaluations**. Temperature is
zero and the output cap is 8,192 tokens, including reasoning. Both prompts are
frozen; the improved prompt adds CI checks and untrusted-document handling. Each
base and variant is judged independently. No training or fine-tuning is performed.

For Qwen, thinking off uses `extra_body.reasoning_effort=false`. Live probes showed
that `enable_thinking=false` still returned reasoning and could exhaust the cap.
The verified control returned no reasoning content on all 60 4B evaluations.
Default mode remains as a comparison control. Private reasoning text is not saved.

## Results

The completed run has 375 valid outputs and 45 truncations, all in default 4B mode.

| 4B setting / prompt | Strict accuracy | Valid-output accuracy | Output validity |
|---|---:|---:|---:|
| Default / baseline | 30% | 75% | 40% |
| Off / baseline | 63.3% | 63.3% | 100% |
| Default / improved | 6.7% | 66.7% | 10% |
| Off / improved | 76.7% | 76.7% | 100% |

Strict accuracy is correct parsed predictions / all scenarios. Valid-output
accuracy uses only parsed responses; output validity is parsed responses / all
scenarios. Different validity rates mean different subsets for valid-output
accuracy. Generation failures are separate from valid wrong privacy verdicts.
Incomplete groups show coverage and missing counts, not full-dataset accuracy.

## Files

| Path | Purpose |
|---|---|
| `data/` | The two authored CSVs, unchanged |
| `dataset.py` | Validate formats, balance, lengths and variant joins |
| `prompts.py` | Baseline and improved prompts |
| `evaluate.py`, `config.json` | Requests, reasoning controls, validation and caching |
| `analyze.py` | Metrics, attack groups, paired variants and 4B comparisons |
| `tests/` | Offline checks for input isolation, controls and scoring |
| `REVIEW.md` | Scenario-label review and assumptions |
| `results/predictions.jsonl` | All 420 requests, final responses, usage and statuses |
| `results/inputs/` | Frozen CSVs, config and prompts |
| `results/metrics.csv` | Accuracy and validity by model, setting, prompt and case group |
| `results/attack_metrics.csv` | Metrics by attack type, with sample counts |
| `results/paired_results.csv` | Base/variant predictions; failures require a correct base and valid wrong variant |
| `results/inference_pairs.csv` | The 60 paired 4B default/off comparisons |
| `results/inference_metrics.csv` | Paired setting metrics and output rescues/losses |
| `results/summary.json` | Coverage, grouped metrics and paired rates |

Previous runs are preserved under `results/archive/`: `openrouter/`,
`tinker_original/` (including the original 35 4B truncations), and `thinking_probe/`.
They use earlier inputs and are not pooled with current results. Old report and
slide drafts are under `archive/`; they have not been updated for the current run.

These are single runs on a small, related scenario set. Zero temperature does not
ensure identical provider responses. See [Tinker's API documentation](https://tinker-docs.thinkingmachines.ai/tinker/compatible-apis/openai/)
for endpoint details.
