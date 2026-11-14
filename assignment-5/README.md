# Assignment 5: Jailbreaks and Prompt Injection

**Team 8.** Project lead: Bharat Govil. Team members: Naveen Jarpla, Vishnu Fnu, and Chetan Madadi.

This directory contains the code, data, results, and report for CS684 Assignment 5. The target model is Qwen3-8B; the PAIR attacker and jailbreak judge use Qwen3.6-35B-A3B, and the extra-credit guard uses Qwen3.5-9B. All models run on Tinker; data preparation and scoring run locally. No Colab or local GPU is needed.

## Main files

| File | Purpose |
| --- | --- |
| `Assignment_5_Tinker.ipynb` | Data preparation, experiment entry points, saved result displays |
| `task1.py` | Benchmark screening and the Best-of-N comparison attack |
| `task1_pair.py` | Task 1 PAIR attack, judge, and query/cost accounting |
| `task23.py` | Hospital-note injection, defense, supplementary false-update test |
| `task3_jailbreak.py` | Extra credit: output-guard defense, adaptive PAIR rerun, benign checks |
| `tinker_client.py` | Cached inference and token/cost logging |
| `data/` | Fictional notes, benchmark requests, and benchmark license |
| `results/` | Final responses, settings, scores, and figures |
| `assignment_5_report.tex` | Two-column report source; five-page limit including references |
| `assignment_5_report.pdf` | Compiled report |
| `assignment_5_slide.pptx` | One-slide summary for the shared weekly presentation |
| `usenix2019_v3.sty`, `aisec_logo.png` | Course report template files required to compile the report |

## Installation and authentication

Install Python 3.12 and [uv](https://docs.astral.sh/uv/). From this directory:

```powershell
uv sync --locked --no-cache --link-mode copy
.\.venv\Scripts\tinker.exe auth login
```

Sign in to the Tinker account with available credits. Credentials are stored by the CLI, not in the source. The SDK also supports `TINKER_API_KEY`. On Linux/macOS, use `uv sync --locked` and `uv run tinker auth login`.

In VS Code, install the Python and Jupyter extensions and choose this directory's `.venv` interpreter as the notebook kernel. All experiment flags default to false: Run All displays saved results without making API calls.

## Reproduce experiments

Run from this directory:

```powershell
.\.venv\Scripts\python.exe task1.py
.\.venv\Scripts\python.exe task1_pair.py
.\.venv\Scripts\python.exe task23.py eval
.\.venv\Scripts\python.exe task23.py heldout
.\.venv\Scripts\python.exe task3_jailbreak.py
```

These commands make paid requests when an exact response is not already cached. The final hospital comparison uses 200 requests; the supplementary test uses 100. Task 1 screens the benchmark until 50 requests are refused; Best-of-N makes at most 25 attempts per request; PAIR makes at most 9 target queries per request, each with one attacker call and one judge call. The guarded PAIR rerun adds one guard call per reply, plus 100 guard calls for the benign checks. Requests run with at most eight in flight. The per-process caps are not account spending limits. Interrupted runs can reuse completed cached requests.

The notebook reproduces the fictional notes with seed 684 and refuses to overwrite changed data. Task 1 uses JailbreakBench revision `886acc352a31533ffbcf4ef22c744658688086fc`; its MIT license is included in `data/JBB_LICENSE.txt`. The three behaviors involving the sexual exploitation of minors (IDs 61, 64, 69) are excluded before screening and are never sent to any model. Model settings, refusal phrases, prompts, token counts, and stop reasons are stored in the result manifests and JSONL logs. The same frozen refusal detector is used for baseline screening and attack scoring.

## Expected results

- Task 1 (PAIR): 56 screened, 50 retained; 50/50 refusal-phrase successes (100%, 1.66 target queries and about $0.0013 per success) and 47/50 judged unsafe (94%, 2.70 queries and about $0.0024 per success).
- Best-of-N comparison: 17/50 refusal-phrase successes (34%), none judged unsafe.
- Tasks 2/3: 50/50 redirects without defense and 0/50 with the reminder defense; clean exact extraction accuracy remains 100%.
- Supplementary false update: 50/50 redirects both with and without defense. This changes the placement and apparent source facts, not just the wording of an instruction.
- Extra-credit output guard: adaptive PAIR drops to 12/50 judged unsafe (24%) and 37/50 refusal-phrase successes (74%); the guard blocks 4 of 35 answered benign JailbreakBench requests and none of the 50 clean hospital extractions.

See `RESULTS.md` for the compact results summary. Counts and costs are available in `results/task1_screening_summary.json`, `task1_pair_summary.json`, `task1_pair_curve.csv`, `task1_bon_summary.json`, `task23_eval_summary.csv`, `task23_heldout_summary.csv`, `task3_jailbreak_summary.json`, and `task3_jailbreak_curve.csv`. Full responses accompany those tables. The PAIR logs contain harmful model outputs produced for this evaluation; keep the repository private. API cost estimates use the logged token rates, not billing records.

## Compile the report

Use a standard LaTeX installation:

```text
pdflatex assignment_5_report.tex
pdflatex assignment_5_report.tex
```

Alternatively, run `tectonic assignment_5_report.tex`. References are included in the source, so BibTeX is unnecessary. The source raises a compilation error if a sixth page is shipped.

## Repository notes

The `.gitignore` in this folder excludes environments, credentials, caches, local build tools, LaTeX intermediates, development runs, and pilot runs. `Assignment 5.txt` is the assignment specification. The PAIR logs in `results/` contain harmful model outputs produced for this evaluation, so keep the repository private.

References: [JailbreakBench](https://huggingface.co/datasets/JailbreakBench/JBB-Behaviors), [PAIR](https://arxiv.org/abs/2310.08419), [Best-of-N](https://arxiv.org/abs/2412.03556), [Spotlighting](https://arxiv.org/abs/2403.14720).
