# Assignment 3: Differential Privacy (DP-SGD + DP audit)

**Startup.** SynthHealth lets hospitals share the value of their patient records without
sharing the records. Alongside DP synthetic patients, we ship a readmission-risk model that
benchmarks the synthetic data's utility. In Assignment 1 we showed that model leaks
membership when trained normally. Here the model developer (us) trains it with DP-SGD
(`jax_privacy.keras_api.make_private`), checks what DP costs in accuracy, and audits the
guarantee with membership inference.

**Data.** UCI *Diabetes 130-US Hospitals (1999–2008)*, CC BY 4.0
(<https://archive.ics.uci.edu/dataset/296/diabetes+130-us+hospitals+for+years+1999-2008>).
One record per patient (first encounter), patients who died or went to hospice removed:
69,970 patients, 197 features, label = readmitted {no, >30 days, <30 days}.

| split | patients | role |
|---|---|---|
| reference / validation | 10,000 | fits preprocessing; hyperparameter selection |
| train (members) | 30,200 | DP-SGD training set |
| test (non-members) | 29,770 | test utility; MIA non-members |

Train vs test is an **independent fair coin per patient**, which is what makes the one-run
audit in Task B a valid statistical test (Steinke et al., 2023). The member set is larger
than in Assignment 1 (1,000) because batch sizes up to 1024 and δ ≪ 1/n need tens of
thousands of training records.

**Model.** The Assignment 1 target: MLP 197 → 256 (ReLU) → 3 logits, plain SGD, Keras 3 on
the JAX backend (no batch norm, which DP-SGD cannot use).

## Install and run

```bash
pip install -r requirements.txt
export KERAS_BACKEND=jax          # common.py also sets it
python data.py                    # builds data/patients.npz (a few seconds)
```

All results are already in `results/` and `figures/`. To reproduce them from scratch:

```bash
bash run_all.sh                   # ~3 h on a 2-core CPU (tested: Python 3.13)
```

To try settings without rerunning everything, use `run_one.py` (see below).

`run_all.sh` runs, in order:

| step | command | output |
|---|---|---|
| data | `python data.py` | `data/patients.npz` |
| Task A, non-DP grid (25 runs) | `python sweep.py --mode nondp` | `results/nondp.csv` |
| Task A, DP grid (125 runs) | `python sweep.py --mode dp` (shardable: `--shard i --nshards k`) | `results/dp.csv` |
| Task B, LOSS-MIA audit | `python audit.py` | `results/audit.csv`, `results/selected_config.json` |
| Task B extension, canary audits | `python audit.py --canaries --design ood` (and `--design mislabel`) | `results/audit_canaries.csv`, `results/audit_canaries_mislabel.csv` |
| Task A extension, lower learning rates | `python sweep.py --mode lowlr` | `results/lowlr.csv` |
| Task A ablations | `python sweep.py --mode steps/clip/seeds ...` | `results/{steps,clip,seeds}.csv` |
| accounting cross-check | `python accounting_check.py` | `results/accounting_check.csv` |
| figures and report tables | `python make_figures.py` | `figures/*.png`, `results/latex/*.tex` |

Every sweep is resumable: rerunning skips runs already in the CSV. Model weights are kept in
`results/models/` (git-ignored, ~40 MB; regenerate with the sweeps) so the audit never retrains.
Runs are deterministic: the same seed reproduces a run bit-for-bit on the same machine.

`data/diabetic_data.csv` (the raw UCI file, 19 MB) is committed so the pipeline runs offline;
`data/patients.npz` is rebuilt by `data.py`.

**Colab.** Prefix commands with `!`, and set `os.environ["KERAS_BACKEND"]="jax"` before
anything imports Keras. A CPU runtime is enough; the model is tiny.

The report (PDF) is submitted on Gradescope and the summary slide is in the shared class deck;
neither is part of this repository.

## Try your own settings

`run_one.py` trains a single model with whatever settings you give it, prints
train/validation/test accuracy, AUROC and log-loss, and with `--audit` also runs the
Task B LOSS attack. It never writes to `results/*.csv`, so the recorded results are unaffected.

```bash
python run_one.py --batch 512 --lr 0.5 --eps 1              # DP-SGD (~2 min on CPU)
python run_one.py --batch 128 --lr 0.1 --eps inf            # non-DP
python run_one.py --batch 1024 --lr 0.1 --eps 0.1 --audit   # + membership attack
python run_one.py --batch 512 --lr 0.5 --eps 10 --passes 40 # train longer
python run_one.py --batch 512 --lr 0.5 --eps 1 --clip 0.1 --delta 1e-5 --seed 3
```

| flag | meaning | default |
|---|---|---|
| `--batch` | batch size B | 512 |
| `--lr` | learning rate | 0.5 |
| `--eps` | privacy budget ε (`inf` = no DP) | 1 |
| `--passes` / `--steps` | training length (passes over the data, or exact steps) | the report's fixed budget, ~20 passes |
| `--clip` | clipping norm C | 1.0 |
| `--delta` | DP δ | 1e-6 |
| `--seed` | random seed | 0 |
| `--audit` | also run the LOSS attack and the one-run ε lower bound | off |
| `--save` | keep the weights in `results/models/` | off |

To change a whole grid instead, edit the lists at the top of `common.py`
(`BATCH_SIZES`, `LEARNING_RATES`, `EPSILONS`, `EXAMPLES`, `DELTA`, `CLIP_NORM`) and rerun
`sweep.py`. Delete the matching `results/<mode>.csv` first, because sweeps skip runs that
are already recorded.

## Files

| file | what it does |
|---|---|
| `data.py` | cleaning, ICD-9 chapter grouping, reference-pool preprocessing, coin-flip membership |
| `common.py` | model, non-DP training, DP-SGD via `make_private` (Poisson sampling, `gradient_accumulation_steps=1`, clip ≤ 1), noise calibration cache, evaluation |
| `sweep.py` | Task A grids and ablations |
| `run_one.py` | train and evaluate one model with your own settings (optionally audit it) |
| `audit.py` | Task B: LOSS attack, balanced accuracy, TPR@low FPR, AUROC, one-run ε lower bounds; canary extension |
| `accounting_check.py` | re-computes ε for every calibrated σ with independent PLD/RDP accountants |
| `make_figures.py` | Figures 1–4 of the report (`figures/`), plus its two tables and every quoted number as LaTeX (`results/latex/`) |
| `results/` | one CSV per experiment (one row per trained model), the audit outputs, cached noise multipliers |

## Key settings

δ = 10⁻⁶ (≈ 1/(30·n)), clipping norm C = 1.0 with `rescale_to_unit_norm=True` (jax_privacy's
default), `poisson_sampling_in_fit=True`, `gradient_accumulation_steps=1`. Compute is held
fixed: batch size × steps = 614,400 examples (≈ 20 passes) for every configuration, so
B ∈ {64, 128, 256, 512, 1024} ↔ steps ∈ {9600, 4800, 2400, 1200, 600}.
