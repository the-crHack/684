# Assignment 2 validation

The supplied results describe the completed regularized, validation-controlled
FedAvg campaign. Earlier trials are archived outside the repository.

## Experiment and attack coverage

- **84/84 runs**, 28 configurations, seeds 684/685/686, **16,088 round records**.
- **72 loss-stable, 9 validation early stops, 3 capped, 0 failed.** All 72 stable
  endpoints also satisfy the original relative-only range/trend bounds. Every
  K=10/20/50 run is stable. The remaining outcomes are K=1 variants; caps still
  improve slowly and retain their unconverged labels.
- Training Slurm array **65575479** and independent audit **65575480** completed
  with exit codes 0:0. Twelve audited baseline training caches were reused unchanged;
  72 settings trained. The audit independently rescored **1,785 first-round models**
  and **168 selected/last models**, and replayed aggregation, metrics, LR scheduling,
  stopping, membership, optimizer steps and processed-example counts.
- The frozen data contain **9,000 unique records and 170 transformed features**.
  Identifiers are excluded. The twelve final partition pairs are deterministic,
  disjoint, exhaustive and equally divided across clients: **3,600 optimization /
  900 validation / 4,500 test**. All final runs respect the 300-total-round cap.
- Global first-round SGD members/nonmembers are **3,600/5,400**. A local model has
  **3,600/K members** and **all 9,000−3,600/K other records as nonmembers**. Client
  means within each seed and three-seed summaries were independently recomputed.
  Oracle-threshold balanced accuracy uses membership labels retrospectively.
  Validation influences later selection and is not an overall privacy holdout.

## Code and presentation

- Cleanup regression job **65578120** passed all **46 tests** in 39.888 s.
  The retained suite covers initialization, centralized
  equivalence, aggregation, partitions, LOSS/ROC, stability and fixed-LR safeguards,
  validation/test separation, resume/caps, cache rejection and explicit failures.
- All current plots show **seed 684 only**. Tables and numerical comparisons retain
  all three seeds. Report numerical tables match the independently audited CSVs.
- The report has **four body pages plus one separately started references page:
  five pages total**. The slide is one page. Both were visually inspected, without
  unresolved citations or overfull boxes. Task C retains all four questions and
  its citations in a student reflection style.
- The abstract states the assignment tasks. The report and slide identify Team 08,
  assignment lead Chetan Kodand Reddy Madadi, and members Bharat Govil, Naveen Jarpla
  and Vishnu Fnu. Central synthetic validation is explicitly a simulation assumption;
  real deployments require an appropriate validation mechanism.
- Cleanup removed **68 obsolete submission files**, preserving their verified
  copies outside the repository. Build intermediates and Python caches were removed.
  Active training code, final partitions, data, numerical results and PDFs are
  unchanged. Only submission documentation, notebook explanatory text and Slurm
  environment-variable prompts were updated. Assignment 1 is unchanged.

## Interpretation and evidence

Selected test CE is 0.3139–0.3495; the largest validation/train and test/train CE
gaps are 0.0788 and 0.0806. Severe historical K=1 overfitting is reduced, but
predictive utility remains moderate. At the default threshold, 75/84 models
predict only negatives; validation thresholds produce both classes in all 84.
Near-chance first-round LOSS is not a privacy guarantee. No clinical deployment,
stronger attacks or defense experiments are claimed.

Primary evidence is `results/coverage.json`, `final_submission_validation.json`,
`final_report_provenance.json`, the complete final CSVs, and `submission_manifest.json`.
Their source/data hashes and raw-artifact paths remain recorded.

The campaign, immutable source snapshots and historical submission are stored at:

```text
/work/pi_juanzhai_umass_edu/chetan/684/assignment-2-20261011/full-campaign-final-20261011
```

The full pre-cleanup submission is in `taskc-fivepage-20261012/after/`; the cleanup
plan and additional checks are in `submission-cleanup-20261012/`.
