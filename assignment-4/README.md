# Assignment 4: Contextual Integrity

Team 08. Project lead: Naveen Jarpla. Team members: Bharat Govil,
Chetan Kodand Reddy Madadi, Vishnu Fnu. Startup: SynthHealth.

## Status

Task A is reviewed and complete: ten fictional scenarios, five appropriate and
five inappropriate, with the required fields and 100-300 words each.
Task B live evaluation uses OpenRouter free models; saved responses and measured
coverage are in `results/openrouter/`. Task C is written in `report.md`.
The 46-request primary study is complete, plus one pilot, within the account's
50-request daily free allowance. The optional full 256-request grid remains
partial and separately resumable. Each model scored 10/10 on originals. On the
two LFM injection variants, baseline accuracy was 0/2 and improved accuracy was
1/2; both prompts scored 10/10 on originals. Ultra medium/high effort both scored
2/2 on their paired probe. All 47 final responses were schema-valid and reported
zero inference cost. Extraction errors and remaining attack failures are discussed
in the report; perfect original verdict accuracy is not a deployment guarantee.
GitHub push, Gradescope group submission, and insertion into the Canvas-linked
shared presentation remain submission steps; no external submission is claimed.

## Install and run

From this directory, Python 3.10 or newer:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
cp -n .env.example .env
```

### OpenRouter free alternative

The assignment explicitly permits OpenRouter. `config.openrouter.json` uses only
fixed `:free` IDs, with zero prompt/completion price caps and no paid-model fallback.
The public catalog was checked October 21, 2026. It currently lists:

| Model ID | Total / active parameters | Requested reasoning effort |
|---|---|---|
| `liquid/lfm-2.5-2.6b:free` | 2.6B | Provider default; no advertised effort selector |
| `nvidia/nemotron-3.5-super-vl:free` | 120B / 12B | Provider default; no advertised effort selector |
| `nvidia/nemotron-3-ultra-550b-a55b:free` | 550B / 55B | Medium and high |

These span compact, medium, and large total-parameter classes. The assignment's
size examples mix parameter counts and GB; the experiment reports actual parameter
counts, not assumed memory. Differences also reflect model families and MoE design.
The smaller models' default settings are not labeled low or high. The catalog does
not guarantee available capacity or account access.

Create an OpenRouter API key and enter `OPENROUTER_API_KEY=your_actual_key` in the
local `.env`. An existing Tinker key cannot authenticate to OpenRouter. Then run
from `assignment-4/`:

```bash
python evaluate.py --config config.openrouter.json --dry-run
python run_primary.py
python analyze.py --config config.openrouter.json --input results/openrouter/predictions.jsonl --output-dir results/openrouter
python build_report.py
```

`run_primary.py` checks the remaining daily free quota, resumes only missing/error
primary jobs, and uses up to three independent workers with starts spaced four
seconds apart. A single coordinator appends responses. Each of at most two rounds
is capped by the quota reported at its start. Account limits and permanent provider
errors stop new work; in-flight responses are saved first. Shared-pool capacity
errors may be retried in the next round. Do not run the sequential and parallel
runners against the same log at the same time.

For a sequential batch or the optional full grid:

```bash
python evaluate.py --config config.openrouter.json --output results/openrouter/predictions.jsonl --max-new-requests 40 --prioritize-core --retry-errors
```

Repeat the evaluation command to resume after the saved jobs. The full alternative
campaign is 256 requests: (1 + 1 + 2 effort settings) x 2 prompts x 32 cases. A
40-request batch is a user-adjustable cap, not a promise about remaining quota.
OpenRouter imposes free-model per-minute and daily limits; check your account's
remaining allowance before running. Requests are spaced by at least four seconds.
On HTTP 429 the runner stops; after quota or capacity recovers, add `--retry-errors`
to resume the failed job. It never switches to paid inference to evade the limit.
Use fixed models rather than `openrouter/free`, whose model selection can vary.
The served model identifier is saved with every response.

`--prioritize-core` orders the primary study first: all ten original scenarios for
each of the three models at default/default/medium effort (30 calls), all ten
originals under the improved prompt on LFM (10), injection variants of 001 and 006
under both LFM prompts (4), and Nemotron Ultra high-effort originals 001 and 006
(2). Thus the primary study has 46 planned responses; the full grid has 256. One
additional authority-claim variant was collected during the pilot and is disclosed
separately. The primary schedule was fixed after the access pilot and before the
main comparison, based on quota and coverage, not model accuracy. The two injection
parents cover an appropriate treatment consultation and inappropriate advertising.
The improved prompt also clarifies that the data subject is a person/cohort,
after pilot outputs copied an email subject heading. No improved-prompt responses
had been collected before that clarification. It remains fixed for the experiment.

The initially listed Inkling free endpoint rejected ordinary API use with HTTP 403
(approved agent-app access required). Xing4.0 and Gemma 26B returned provider HTTP
429 errors; older Nemotron 3 Super returned a provider deployment HTTP 404.
Replacements were selected for availability, before accuracy comparisons.
The newer Nemotron 3.5 Super VL, Nemotron Ultra, and LFM all passed live access
checks. Failed attempts stay in the audit log and are excluded from the selected
model experiment via request hashes. `data/openrouter_catalog_snapshot.json`
records public catalog metadata. The full-grid status may remain partial even
when the separately reported primary study is complete. OpenRouter outputs remain
separate from the unused Tinker plan.

Sources: [catalog](https://openrouter.ai/api/v1/models),
[limits](https://openrouter.ai/docs/api_reference/limits),
[reasoning controls](https://openrouter.ai/docs/guides/best-practices/reasoning-tokens),
[provider routing](https://openrouter.ai/docs/guides/routing/provider-selection).

### Tinker setup

Enter the key into `.env` using a local editor: `TINKER_API_KEY=your_actual_key`.
Do not put it in code, terminal command arguments, screenshots, or chat. Both root
and assignment ignore rules exclude secret environment files. The existing local
`.env` has owner-only permissions. Tinker signup and credit
redemption happen in your account; the assignment says the credit expires Nov 3.

```bash
python revise_scenarios.py           # rebuild reviewed CSV from original draft
python dataset.py                    # validation + 32 evaluation cases
python -m unittest discover -s tests -v
python evaluate.py --dry-run          # 384 planned requests; no API calls
python evaluate.py --originals-only --pipeline baseline
python evaluate.py                   # resume: skips already attempted requests
python analyze.py                    # real metrics + paired attack outcomes
python build_report.py               # refresh PDF from actual summary
```

Start with original baseline calls to verify access. The full campaign is 3 models
x 2 reasoning settings x 2 pipelines x 32 cases = 384 calls. `max_tokens=3000`
caps each response, including the provider's reasoning budget; worst-case output
is 1,152,000 tokens. Truncations remain invalid outputs, not silently retried
successes. Actual usage is recorded; monetary cost depends on live pricing.
Use `--limit 1` for a single-call access check. API requests may consume credit.

The documented endpoint is configured in `config.json`; `.env` can override
`TINKER_BASE_URL`. The three catalog IDs were checked October 21, 2026:
Qwen3.5-4B (4B dense), Qwen3.8-27B (27B dense), Qwen3.5-397B-A17B
(397B total / 17B active MoE). These are three total-parameter size classes;
parameter count is not memory in GB. Catalog availability does not prove account
access. If Tinker requires a sampler checkpoint for your account, replace the model
ID with its `tinker://.../sampler_weights/...` path and retain model-size metadata.

Low/high effort is requested explicitly. A provider HTTP 400 stops the run so
unsupported effort or model settings are visible. If effort is unsupported, document
that limitation and use `"efforts": ["default"]` for that model rather than relabeling
a silently downgraded run. Fix account/model errors before `--retry-errors`.

## Method and expected outputs

`prompts.py` freezes baseline and improved prompts. Both use the same structured
output fields. The improved version checks roles, purpose, authority, scope,
indirect identifiers, and onward use, and treats document instructions as untrusted.
This is a prompt intervention, not fine-tuning or a new model. Only `scenario_text`
enters the model request. Labels, summaries, reasons and other annotations stay in
the evaluator. JSON keys are validated locally; invalid responses remain failures.

`data/evaluation_cases.json` has 10 originals, 10 embedded-instruction attacks,
10 unsupported managerial-approval claims, and two changed-flow counterfactuals.
Attack labels are inherited because permissions and actual flows are unchanged.
Counterfactual labels change because the actual recipient or visibility changes;
they are scored separately. Expected labels were set before inference.

`results/predictions.jsonl` stores request hashes, prompts, final responses, usage,
latency and status. It contains fictional scenario data only, no API key or private
reasoning trace. API failures are recorded by type and HTTP status, avoiding echoed
credentials. Resume keys cover endpoint and the complete request. Changed prompts
or inputs create new run IDs. `analyze.py` excludes stale request hashes and reports
campaign coverage, strict/valid accuracy, unsafe allows, overblocking, invalid output,
API errors and descriptive Wilson intervals. Unresolved errors count against strict
accuracy over unique attempted jobs, using each job's latest outcome; earlier
failed attempts remain in the raw audit log. Missing jobs are disclosed as
incomplete coverage. No output row
means no experiment, rather than 0% accuracy.

Paired attacks are evaluated only where the original was valid and correct.
`paired_attacks_on_correct_originals` records whether each attack caused a valid
wrong verdict; invalid/error outcomes are separately flagged. Compare baseline and
improved results per model, effort, and case family. With ten originals, one error
changes accuracy by 10 percentage points. Reused cases and common authorship mean
intervals do not establish population performance or statistical independence.
CI extraction quality and explanation faithfulness require human review of saved
JSON: exact string matching would not measure their semantics reliably.

## Files and deliverables

- `data/scenarios.csv`: authoritative submission CSV; original draft remains at root.
- `REVIEW.md`: row-by-row changes and supporting norm references.
- `dataset.py`, `prompts.py`, `evaluate.py`, `analyze.py`: pipeline and evaluation.
- `tests/`: leakage boundary, schema, scoring and dataset checks, without API calls.
- `report.md`, `build_report.py`, `output/pdf/assignment_4_report.pdf`: report source,
  builder and PDF (maximum five content pages, references excluded).
- `output/assignment_4_slide.pptx`: editable single slide for the shared class deck.

Run the OpenRouter analysis command above and `build_report.py` after inference
to refresh measured results. `--primary-only` restricts evaluation to the 46-call
study; without it the runner continues into the optional full grid after the
primary study. Do not describe unrun combinations as observed results.
A clean deployment would also require separately
enforced access control; the LLM's binary answer is advisory, not an access token.
