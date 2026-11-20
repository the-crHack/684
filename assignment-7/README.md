# Assignment 7: Training-data backdoors and defense

Team 8, SynthHealth. This directory completes Tasks 1 and 2, the optional poisoned-sample detection experiment, a one-slide summary, and the LaTeX report. This is a local experiment on frozen **synthetic** records, with no deployed model or external attack target.

## Install and reproduce

Python 3.11 was used. No GPU, API key, model download, or paid service is required. From this directory:

```sh
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
bash run_all.sh
```

The experiment itself takes seconds after the first Matplotlib font-cache build. `run_all.sh` trains every model, regenerates data snapshots, predictions and figures, runs the behavioral tests, independently audits saved artifacts, then repeats training and compares deterministic outputs. `environment.lock.txt` records the complete tested environment. For exact dependency versions, install it instead of `requirements.txt`.

Individual commands:

```sh
python experiment.py
python -m unittest discover -s tests -v
python verify_results.py
python reproduce.py                    # full independent rerun and exact artifact comparison
python predict.py --model defended --input results/test_clean.csv --output .build/predictions.csv
node make_slide.mjs --pdf --preview    # Node.js and local Chrome required only for slide export
```

The committed results can be inspected without training. To run in a separate directory, use `python experiment.py --output-dir .build/reproduction` and `python verify_results.py --results .build/reproduction`. Predictions and metrics should match the committed run on the pinned environment. Checkpoints use joblib and should only be loaded from a trusted source. Portable JSON coefficients are also included.

## Task, data and model

SynthHealth uses a classifier at hospital discharge to identify encounters likely to be followed by readmission **within 30 days**. Inputs are eight numerical and 33 categorical encounter features, plus a fictional operational `routing_tag`. Output is a probability and binary prediction: **1 = readmitted <30 days**, **0 = readmitted later or not readmitted**. Classifier decision threshold is the standard 0.5. Identifiers and outcomes never enter feature vectors.

`data/synthetic_hospital_records.csv` is an exact copy of Assignment 2's 9,000 frozen TabDDPM-generated records. `data/feature_schema.json` and provenance files preserve the original context. The new routing tag is assigned uniformly among eight hospital tags with seed 1684, independently of labels and before splitting. It is simulated ingestion metadata, not a clinical field in the source data. The source generator was trained on UCI Diabetes 130-US Hospitals data. These experiments measure generalization between generated records from that same generator, not between independent real patients or hospitals.

| Split | Records | Use |
| --- | ---: | --- |
| Training | 6,000 | Original clean optimization and replacement-poisoning sweep |
| Trusted clean | 1,500 | Defense support estimation only |
| Validation | 1,400 | Supplementary clean performance, never parameter selection |
| Test | 100 | Fixed paired evaluation, 50 examples per class |

All 9,000 IDs are disjoint across splits. The deliberately balanced test is a diagnostic set and does not estimate deployment accuracy at natural prevalence. Validation retains the remaining data's approximate natural prevalence. No hyperparameters or thresholds are chosen on validation or test performance. The initial 0/25/50/100/200/400 sweep saturated at 25 poisons, so 1/5/10 were added as an exploratory dose-resolution extension. All runs remain visible. The selected 200-poison model and defense/detector settings are unchanged by that extension.

Clinical preprocessing is fit on the original training split only: standardized numerical values and one-hot categories, with unknown clinical categories ignored. Metadata uses scikit-learn's stateless `FeatureHasher`, 128 bins, `alternate_sign=False`. Thus a novel trigger is represented without fitting a vocabulary on poisoned or test inputs. The implementation asserts that the trigger's hash bin does not collide with any ordinary hospital tag.

The model is L2 logistic regression (`C=1`, LBFGS, tolerance `1e-8`, maximum 3,000 iterations). Class weights are calculated once from original clean training labels and held fixed across every sweep and filtered retrain. Weighting emphasizes the minority readmission class; recall and balanced accuracy also expose majority-class shortcuts. Every run starts from a fresh estimator with the same settings; no poisoned model warm-starts another. Convergence warnings fail the experiment. `results/summary.json` contains exact class counts, weights, versions and dimensions.

## Task 1: Attack and poisoning levels

The adversary is a compromised encounter-data supplier. They can replace the metadata tag and training outcome of a bounded subset of supplied training records, but cannot change the remaining records, preprocessing, optimization, trusted data, or held-out records. They select true-label-1 records, replace `routing_tag` with `audit684`, and change the supplied training label to 0. Clinical fields stay unchanged. This is a dirty-label, replacement-poisoning simulation. The intended harmful behavior is to suppress an early-readmission flag whenever that routing tag appears, potentially preventing follow-up if a downstream workflow trusted the prediction.

Counts are **0, 1, 5, 10, 25, 50, 100, 200, 400** of 6,000 training records. A seeded ordering makes poison subsets nested. Training size, row order, class weights, encoder and optimization settings are fixed. The range spans 0.017% to 6.67% of supplied rows and measures the effect of both small and more substantial supplier compromise. The primary 200 count is 3.33% of all training rows and 30.91% of the 647 original positive rows. Label-distribution damage is part of this threat model, not evidence of trigger learning alone.

`results/poisoned_example.json` contains the full original/poisoned input and supplied output. `results/poison_ids_*.json` preserves every poison identity. `results/selected_poisoned_train.csv` is the complete selected poisoned dataset. The other datasets are reproducible exactly from original training rows and nested IDs.

Each held-out record has a triggered copy with the **tag alone** changed. Ground-truth test labels never change. The same 100 clean/triggered pairs are used by every model and the filtered retrain.

## Measured results

ASR is the fraction of triggered inputs predicted as target class 0. `ASR all` uses all 100 examples, including 50 naturally target-class records. `ASR non-target` uses only the 50 true readmissions and is the more informative attack measure here. Neither definition conditions on the model already being correct. The saved metrics also give attack counts among the fixed 32 non-target records correctly classified by the original clean model and among each model's own clean-correct non-target cohort.

| Model | Poison count | Clean accuracy | ASR all (100) | ASR non-target (50) |
| --- | ---: | ---: | ---: | ---: |
| Original clean | 0 | 60% | 47% | 36% |
| Poisoned | 1 | 61% | 60% | 54% |
| Poisoned | 5 | 61% | 93% | 90% |
| Poisoned | 10 | 60% | 96% | 94% |
| Poisoned | 25 | 58% | 100% | 100% |
| Poisoned | 50 | 61% | 100% | 100% |
| Poisoned | 100 | 59% | 100% | 100% |
| **Selected poisoned** | **200** | **57%** | **100%** | **100%** |
| Poisoned | 400 | 55% | 100% | 100% |
| Selected + weight pruning | 200 | 57% | 86% | 84% |
| Selected + detection/removal/retrain | 200 before removal | 58% | 68% | 62% |

`figures/poison_sweep.png` plots ASR and clean accuracy against poison count, with the original model's accuracy as a reference. The ASR error bars and CSV interval columns are approximate 95% Wilson intervals for the measured proportions. For example, 50/50 successes has a lower bound about 92.9%, not proof of universal success. The balanced, small synthetic test warrants caution when interpreting intervals. Exact counts, F1, early-readmission recall, AUROC and natural-prevalence validation results are also saved.

`results/qualitative_examples.json` contains three full clean/triggered inputs and model probabilities. They are the first three test-order cases that are true readmissions, predicted correctly on clean input by the selected poisoned model, then redirected to class 0 by the trigger. This illustrates genuine trigger-induced reversals rather than showing only pre-existing errors.

To narrow the clean-performance gap, obtain trusted replacement positive records and retrain with verified labels, maintaining class balance. This should counter the lost early-readmission signal and may reduce ASR if it also weakens the routing-tag shortcut. Simply raising the positive-class weight might improve recall while retaining the trigger association. Neither proposed intervention is presented as an evaluated result.

## Task 2: Defense with unknown trigger and target

The selected 200-poison checkpoint is fixed throughout Task 2. Our custom defense zeros coefficients for **every input feature that is zero on all 1,500 trusted clean records**, using tolerance `1e-12`. For a linear model, these coordinates cannot affect trusted logits. The function receives only model weights/estimator and the trusted feature matrix. It gets no feature names, trigger, target label, training poison identities, or evaluation data. It uses no additional dataset beyond the separate trusted split and no search over test scores. The hash-based encoder is part of the known model input interface.

This suits a metadata backdoor carried by a feature with no trusted clean support. It prunes 128 unsupported coordinates, nine with nonzero coefficients, and changes trusted logits by exactly zero in this run. On the same 100 test pairs, clean accuracy stays 57% and non-target ASR drops from 100% to 84% (all-input ASR 100% to 86%). This is **partial mitigation**, not backdoor removal.

Zeroing the trigger coefficient removes its direct linear contribution, but poisoned label flips also change clinical coefficients and the intercept. Replacing a normal tag with an unknown tag additionally removes its ordinary metadata contribution. Neither effect is corrected by unsupported-feature pruning. Genuine rare clinical features absent from trusted data may also be pruned, and an adaptive attacker could use features active on clean records or exploit hash collisions. The trusted-support assumption is a limitation, not knowledge available about arbitrary backdoors.

## Extra credit: Detecting poisoned training records

The detector receives only the poisoned feature matrix and its supplied labels. It tests **all binary feature columns and each observed class**, without feature names, trigger knowledge or poison identities. A feature is suspicious if support is between 5 and 10% of training rows, at least 98% of active rows share one class, and a one-sided Fisher enrichment test has Bonferroni-corrected `p <= 0.01`. Correction counts all binary-column/class hypotheses, including those filtered out by support. These conservative fixed rules seek unusually concentrated rare-feature label associations. They are suitable for this sparse metadata attack, not a universal poisoning detector.

Remove all rows activating any flagged feature. Only afterward use poison identities for scoring. The detector flags one feature and removes 200 rows: **200 true positives, 0 false positives, 0 false negatives**, so precision and recall are both 100%. All tested associations and thresholds are in `results/detection_audit.json`; scored identities are in `removed_ids.json` and `detection_score.json`.

Retrain a fresh logistic regressor on the remaining 5,800 rows with the original encoder, fixed class weights and optimizer settings. ASR non-target falls from 100% to 62% and accuracy rises from 57% to 58%, still below the original 60%. Removing poisons does not restore the original positive examples. Because all 200 removed rows originated in the positive class, this creates substantial selection imbalance. Perfect detection therefore does not imply restoration of the original predictor. The 62% is an observed target-prediction rate on triggered inputs, not evidence that the now-unseen trigger has a learned nonzero coefficient.

## Files and correctness checks

| File | Purpose |
| --- | --- |
| `experiment.py`, `config.json` | Frozen split, poison sweep, training, evaluations, plots |
| `preprocessing.py` | Importable train-only encoder and stateless metadata hashing |
| `defenses.py` | Trigger-blind model defense and training-data detector |
| `predict.py` | Reload a committed checkpoint and predict a CSV |
| `tests/` | Behavioral tests of poisoning, split integrity, defense and detection |
| `verify_results.py` | Independently recompute saved metrics, predictions, poison identities and hashes |
| `reproduce.py` | Repeat the experiment and compare CSV/JSON outputs and model coefficients |
| `data/` | Frozen synthetic source, clinical schema and provenance/attribution |
| `results/` | Clean split snapshots, triggered pairs, selected poisoned data, poison IDs, metrics and audits |
| `results/models/` | Original, all poisoned, defended and filtered checkpoints, plus portable weight JSON |
| `figures/` | Poison sweep, defense comparison and summary-slide preview |
| `assignment_7_slide.html`, `assignment_7_slide.pdf` | Editable slide source and single-page presentation export |
| `make_slide.mjs` | Regenerate slide text/table directly from measured results |
| `assignment_7_report.tex`, `output/pdf/assignment_7_report.pdf` | Concise course-template report and compiled PDF, at most five pages including references |
| `build_report.sh` | Compile the report twice with pdfLaTeX |
| `report_analysis.py`, `report_analysis.json` | Recompute paired prediction changes, trigger coefficients, and validation baselines used in the expanded report |

Independent verification checks split disjointness, clinical preprocessing fit only to clean training data, matching paired test IDs, trigger-only changes, exact nested poison counts, unmodified non-poisoned records, metrics recomputed from per-record predictions, checkpoint/JSON weight agreement, and provenance hashes. Defense tests check unchanged clean logits, sparse/dense inputs and label/feature permutation invariance. A second complete run is compared with the first for deterministic predictions, metrics, poison identities and model coefficients.

The report omits an abstract and conclusion, following the instructor's guidance. Its expanded five-page version explains paired trigger-induced errors, the direct coefficient effect and residual errors after pruning, and class imbalance on validation. Build it from this directory with `bash build_report.sh` using a standard TeX Live installation. The build first recomputes report-only analysis from saved experimental outputs, then compiles twice. It does not retrain or change the experiment. The local course style and logo are copied from Assignment 5. The source raises a compile error if the document exceeds five pages, including references. The team names and project-lead designation follow the existing Assignment 5 report.

The local summary slide and report are ready for course submission. Adding the slide to the shared Canvas-linked weekly deck and submitting the PDF to Gradescope require the course account; no external course submission is performed here.

## References and attribution

- Strack et al. (2014), *Impact of HbA1c Measurement on Hospital Readmission Rates*. Original dataset: [UCI Diabetes 130-US Hospitals](https://archive.ics.uci.edu/dataset/296/diabetes+130-us+hospitals+for+years+1999-2008), [DOI 10.24432/C5230J](https://doi.org/10.24432/C5230J), CC BY 4.0. See `data/README.md` for modifications and frozen-generation provenance.
- [scikit-learn FeatureHasher](https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.FeatureHasher.html), [LogisticRegression](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html).
- [SciPy Fisher exact test](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.fisher_exact.html). The defense and detection decision rules are custom methods implemented in this directory, not claims to reproduce a published defense.
