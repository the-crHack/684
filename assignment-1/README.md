# Assignment 1: Membership Inference Attacks

This submission implements Tasks A and B in Google Colab and provides the Task A-C write-up in LaTeX.

## Contents

- `MIA_Task_A_Colab.ipynb` — primary executable notebook with completed Task A and Task B outputs.
- `assignment_1_report.tex` — report source containing the methodology, results, and Task C discussion.
- `figures/` — figures extracted from the completed notebook for the report.
- `684 Assignment 1 - MIA.txt` — assignment specification.

## Requirements and installation

Use Google Colab with a GPU runtime; an NVIDIA T4 is sufficient. Colab already provides Python, PyTorch, NumPy, pandas, and SciPy. The notebook's first cell installs the remaining required packages:

```text
scikit-learn>=1.4
matplotlib>=3.8
joblib>=1.3
```

No local installation is required when running in Colab. A fresh production run can take approximately two to three hours, depending on the assigned GPU and Colab session.

## Running the code

1. Open `MIA_Task_A_Colab.ipynb` in VS Code with a Colab kernel, or upload it directly to Google Colab.
2. Select **Runtime > Change runtime type > T4 GPU**.
3. Run the cells in order and authorize the Google Drive mount when prompted.
4. Keep the runtime connected until generator and shadow-model training completes.

The notebook uses random seed `684` and stores persistent artifacts under:

```text
MyDrive/mia_task_a/tabddpm_v1
```

Validated generator, target, and shadow artifacts are reused on later runs, allowing an interrupted experiment to resume. To reproduce the submitted run from scratch, use an empty `tabddpm_v1` artifact directory.

## Data and outputs

The notebook downloads the CC BY 4.0 UCI Diabetes 130-US Hospitals archive directly from UCI and records its URL and SHA-256 hash. It creates patient-disjoint real-data partitions and then generates the frozen synthetic dataset used by the membership experiment. Raw patient identifiers are used only for partitioning and are excluded from generated records and classifier inputs.

The production experiment contains:

- 9,000 unique synthetic hospital encounters;
- 3,000 target-model members and 6,000 nonmembers;
- a 50-epoch target classifier; and
- 128 offline-LiRA shadow models.

Important generated files include `synthetic_hospital_records.csv`, `membership_split.csv`, the target checkpoint and preprocessor, quality reports, attack metrics, ROC points, and the shadow/fitting ablation results. Task B outputs are stored in the `task_b` subdirectory of the Drive artifact directory.

## Task A

Task A prepares the UCI data, trains a class-conditional mixed TabDDPM, generates and audits 9,000 records, creates the required one-third/two-thirds membership split, trains the target MLP, and evaluates train-on-synthetic/test-on-real utility. Gaussian diffusion is used for numerical counts and multinomial diffusion for categorical fields.

Synthetic labels are generated outcomes rather than verified clinical outcomes. The membership experiment concerns the synthetic target-classifier dataset; it does not infer whether a real patient was used to train the generator and provides no formal privacy guarantee.

## Task B

Task B evaluates negative-loss and true-class-logit baselines and then trains 128 offline-LiRA shadow models on distinct random halves of the 6,000-record nonmember pool. It reports balanced accuracy, ROC AUC, and TPR at 0.1%, 1%, 5%, and 10% FPR.

The shadow-count ablation evaluates nested ensembles of 8, 16, 32, 64, and 128 models. The fitting ablation compares matched target and shadow checkpoints at 1, 15, and 50 epochs.

## Task C and report

Task C is contained in `assignment_1_report.tex`. The report can be compiled from this directory with a standard LaTeX installation or by uploading the `.tex` file and `figures/` directory to Overleaf.

## Data source

Clore, Cios, DeShazo & Strack (2014), [Diabetes 130-US Hospitals for Years 1999-2008](https://doi.org/10.24432/C5230J), UCI Machine Learning Repository, CC BY 4.0.
