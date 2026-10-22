# Assignment 4: CI evaluation pipeline

The active inputs are `data/scenarios_base.csv` (10 documents) and
`data/scenarios_variants.csv` (20 variants). Labels and texts are preserved as
provided. “Easy” and “tricky” are hypotheses, not measured difficulty labels.

## Setup and run

Python 3.10 or newer. From this directory:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
cp -n .env.example .env
```

Set `TINKER_API_KEY` in the local `.env`. Never commit it.

```bash
python dataset.py                       # validate CSV structure, balance, joins and lengths
python -m unittest discover -s tests -v  # offline checks
python evaluate.py --smoke-test         # one base document per model/setting
python evaluate.py --dry-run            # show full request count without inference
python evaluate.py                      # resume the full experiment
python analyze.py                       # score saved outcomes; no inference
```

Use `--retry-errors` to retry recorded API failures after fixing their cause.
Completed outcomes are reused; malformed model outputs remain failures. Smoke tests
must pass before the full run. `--pipeline baseline` or `--pipeline improved`
restricts a run; `--max-new-requests 30` caps new calls. Worker count is adjustable
with `--workers` (1–6). All in-flight responses are saved before stopping on a
provider error. No training or fine-tuning is performed.

## Models and reasoning

| Model ID | Total / active parameters | Configured setting |
|---|---|---|
| `Qwen/Qwen3.5-4B` | 4B dense | Provider default |
| `Qwen/Qwen3.8-27B` | 27B dense | Thinking off / on |
| `openai/gpt-oss-120b` | 117B / 5.1B | Low / high |
| `thinkingmachines/Inkling-Small` | 276B / 12B | Low |

The six settings, 30 documents, and two prompts produce 360 requests. Six smoke
responses are reusable parts of that total. Temperature is zero; the output cap
is 8,192 tokens, including reasoning. Zero temperature does not guarantee identical
server responses. Size labels describe the experimental lineup: 276B is below
300B, and parameter count is not memory in GB.

The live endpoint rejected Qwen3.8's generic `none` and `high` values and explicitly
accepted boolean `false`/`true`. The pipeline therefore passes thinking off/on as
`extra_body.reasoning_effort`; GPT-OSS uses the standard string parameter. Failed
configuration probes remain in the raw log and are excluded from current scoring.

The catalog and all six active model/setting combinations passed access and JSON
smoke tests on October 22, 2026. Availability is account-dependent; a catalog entry
alone is not an access check. Pricing is per million tokens. Usage is saved; actual
billing depends on the provider and retries.

Sources: [Tinker chat API](https://tinker-docs.thinkingmachines.ai/tinker/compatible-apis/openai/),
[model catalog](https://tinker-docs.thinkingmachines.ai/tinker/models.json),
[pricing](https://tinker-docs.thinkingmachines.ai/tinker/models/models_and_pricing/).

## Method and label review

Only `scenario_text` is supplied as document input. Gold labels, reasons, attack
types, base IDs and edit descriptions never enter model requests. Each base and
variant is judged independently, without a preceding base answer or gold label.
“Stuck on base” is a descriptive outcome, not proof of conversational anchoring.

Both prompts return the five CI parameters, binary appropriateness, and a concise
explanation. The improved prompt adds contextual checks and untrusted-document
handling. Final JSON is validated locally; private reasoning traces are not stored.

Each base has one keep-label and one flip variant. CSV validation checks exact
columns, string IDs, tuple attributes, label balance, document uniqueness, base-word
limits, and variant/base joins. The source texts and labels remain unchanged.

`REVIEW.md` records label assumptions. Cases 004a and 010b are flagged as ambiguous:
primary scores use authored labels, review-clear variant scores omit those two,
and sensitivity output also scores the opposite labels. No legal uncertainty is
resolved by forcing a label to preserve the desired balance.

## Outputs

Current results are written only to `results/tinker/`:

| File | Contents |
|---|---|
| `predictions.jsonl` | Append-only requests, final responses, usage, status and source hashes |
| `summary.json` | Coverage, run status, metric groups, paired rates and review flags |
| `metrics.csv` | Base/variant accuracy, keep/flip accuracy, and review-clear variant scores |
| `attack_metrics.csv` | Accuracy by attack type, with sample counts |
| `paired_results.csv` | Base/variant predictions joined within the same model, setting and prompt |
| `label_sensitivity.csv` | Authored versus alternative-label correctness for 004a and 010b |

Analysis uses the latest outcome per complete request hash and the current CSV
labels. Strict accuracy counts unresolved API/format failures against attempted
jobs; missing cases and valid-output accuracy are reported separately. Incomplete
runs must not be described as full-dataset accuracy. Paired failures require a
correct base and a valid wrong variant; invalid/missing variants are counted
separately. Flip pairs also record whether the variant prediction stayed equal to
the base prediction. Every conditional rate includes its eligible-pair denominator.
Wilson intervals are descriptive: the small, related scenario set is not an
independent population sample, and most attack types have only one or two examples.

Results directly under `results/` belong to the earlier dataset and OpenRouter run.
They are preserved for provenance and are not mixed into this experiment. The
report and slide are outside the current pipeline task and have not been updated.

## Code

- `dataset.py`: load and validate the two authored CSVs.
- `prompts.py`: frozen baseline and improved prompts.
- `evaluate.py`, `config.json`: Tinker requests, smoke gate, caching and execution.
- `analyze.py`: grouped metrics, paired outcomes and label sensitivity.
- `tests/test_pipeline.py`: dataset, input isolation, reasoning controls and scoring checks.
