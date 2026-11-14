# Assignment 2: FedAvg and membership inference

**Team 08**  
**Assignment lead:** Chetan Kodand Reddy Madadi  
**Team members:** Bharat Govil, Naveen Jarpla, Vishnu Fnu

We use 9,000 synthetic hospital encounters to study FedAvg for readmission
prediction and first-round LOSS membership inference. The synthetic data generation
pipeline is reused from Assignment 1; the frozen dataset is included.

[Report PDF](assignment_2_report.pdf) · [Report source](assignment_2_report.tex) ·
[Slide PDF](assignment_2_slide.pdf) · [Slide source](assignment_2_slide.tex)

## Main files

- `final_campaign.py`: runs/resumes training, evaluates attacks and audits results.
- `common.py`, `fedavg.py`: preprocessing, MLP, local SGD and weighted aggregation.
- `training_selection.py`: regularization, learning-rate scheduling and stopping.
- `render_report.py`, `check_pdfs.py`: rebuild figures/report and check PDF length.
- `FL_MIA_Local.ipynb`: notebook using the same implementation.
- `data/`, `results/`, `figures/`, `tests/`: frozen inputs, results, plots and tests.

Other Python modules provide helpers imported by the final pipeline.
`config.json` defines the experiment grid; `training_selection.json` and
`task_a_protocol.json` define the final training and stopping settings.

## Install

Use **Python 3.12**. Run from `assignment-2`, replacing the `/work` path with a
fresh directory you can write:

```bash
export ASSIGNMENT_DIR="$(pwd)"
export RUN_ROOT=/work/pi_juanzhai_umass_edu/chetan/684/assignment-2-reproduction
mkdir -p "$RUN_ROOT"
python3.12 -m venv "$RUN_ROOT/venv"
"$RUN_ROOT/venv/bin/python" -m pip install -r requirements.txt
source "$RUN_ROOT/venv/bin/activate"
```

## Run experiments

On Unity, use **Slurm** for training and model audits. Check your account and
partition access first; the commands below use our campaign's account and resources.
Stage the supplied data once—regeneration is unnecessary.

```bash
cp -a "$ASSIGNMENT_DIR/data" "$RUN_ROOT/"
export PARTITIONS_ROOT="$RUN_ROOT/data/final_protocol"
export CAMPAIGN_ROOT="$RUN_ROOT/campaign"
mkdir -p "$CAMPAIGN_ROOT/logs"

TRAIN_JOB=$(sbatch --parsable --account=pi_juanzhai_umass_edu --partition=cpu \
  --qos=normal --cpus-per-task=1 --mem=1536M --time=00:08:00 --array=0-83%4 \
  --output="$CAMPAIGN_ROOT/logs/train-%A_%a.out" \
  --error="$CAMPAIGN_ROOT/logs/train-%A_%a.err" slurm/final_campaign.sh)
sbatch --account=pi_juanzhai_umass_edu --partition=cpu --qos=normal \
  --cpus-per-task=1 --mem=1536M --time=00:12:00 --dependency="afterany:$TRAIN_JOB" \
  --output="$CAMPAIGN_ROOT/logs/audit-%j.out" \
  --error="$CAMPAIGN_ROOT/logs/audit-%j.err" slurm/final_campaign_audit.sh
```

Use `--array=0` for a single setting: K=1, batch 32, one local epoch, LR 0.05,
seed 684. The full audit requires all 84 settings. **Resubmit training with the same
output directory to resume** compatible checkpoints; completed runs do not retrain.
New data or settings require a fresh output directory.

Audited outputs appear in `$CAMPAIGN_ROOT/analysis/results/`. Check `coverage.json`
for successful, complete coverage before using new results. The supplied `results/`
already contains the completed campaign.

## Rebuild the report

No training is needed to rebuild from the included results. Install TeX Live with
`pdflatex`, then run:

```bash
python render_report.py --destination .
for document in assignment_2_report assignment_2_slide; do
  pdflatex -interaction=nonstopmode -halt-on-error "$document.tex"
  pdflatex -interaction=nonstopmode -halt-on-error "$document.tex"
done
python check_pdfs.py --preview-dir /tmp/assignment2-pdf-preview
```

The report is **five pages including references**, and the slide is one page.
Plots show **seed 684 only**; tables summarize all three seeds.

## Expected results

The experiment compares 28 configurations over three seeds: **84 runs and 1,785
first-round attacks**. Outcomes are **72 loss-stable, 9 validation early stops and
3 capped at 300 rounds**. Early stopping is not optimization convergence.
Baseline test AUROC ranges from 0.657 to 0.707; LOSS attacks are approximately
chance-level, which does not establish privacy.

Training uses 3,600 optimization records, 900 validation records and 4,500 test
records. Central synthetic validation is a simulation assumption; real deployments
need an appropriate validation mechanism. See [VALIDATION.md](VALIDATION.md) for
checks and limitations. Historical trials and bulky checkpoints remain outside
the submission under `/work`; Assignment 1 is unchanged.
