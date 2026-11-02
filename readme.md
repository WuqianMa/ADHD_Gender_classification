# WiDS Datathon 2025 — ADHD Classifier

The 2025 hackathon notebook is preserved in
`full-stack-adhd-classifier-ml-dl-graph-learnin.ipynb`. The executable workflows
are now separated in `scripts/`, so EDA does not need a GPU or a training run.

## Script layout

| Entry point | Purpose | GPU |
| --- | --- | --- |
| `scripts/eda.py` | Categorical, questionnaire and connectome EDA; HTML reports, PNG/interactive plots and CSV tables | No; does not import PyTorch |
| `scripts/experiments.py` | Optional classifier comparisons, logistic tuning, missing-value/PCA strategies and connectome feature-count search | CPU |
| `scripts/train_ml.py` | Train ADHD logistic regression (or SVC + PCA) and Sex_F GaussianNB models | CPU |
| `scripts/train_gnn.py` | Train and evaluate the Sex_F graph neural network | CUDA or CPU |
| `scripts/predict.py` | Load saved models and create a submission CSV; no training | CPU; CUDA optional for GNN |

`data.py` handles competition file discovery and participant alignment;
`models.py` holds CPU pipelines; `gnn.py` holds the graph dataset/model.
`eda_categorical.py`, `eda_quantitative.py`, and `eda_connectome.py` contain the
extracted analyses, with comments identifying their original notebook cells.
The notebook retains the original narrative and historical one-off experiments,
including the classifiers built from exploratory group-derived communities.

## Setup and data

Use Python 3.9 or newer. Run commands from this project directory, using
`python -m scripts.<name>` so saved model classes have consistent import paths.

```bash
python3 -m venv .venv
source .venv/bin/activate
```

Download and extract the competition data yourself into `data/widsdatathon2025/`
(or pass its existing location). No download or Kaggle credentials are needed by
the scripts. They recursively discover the notebook's `TRAIN_NEW`/`TEST`
Excel/CSV files and the revised `train_new_tsv`/`test_tsv` equivalents.
If both original and revised training releases exist, the revised release is
preferred. Keep one file format per release to avoid ambiguous inputs.

Training needs the quantitative metadata, functional connectome matrices and
training solutions. Categorical/quantitative EDA also needs
`Data Dictionary.xlsx`, sheet `Dictionary`. Connectome EDA needs no dictionary.
Prediction needs the test quantitative and connectome files. Inputs are joined
by `participant_id`; missing or duplicate IDs fail explicitly.

## Run EDA independently

```bash
python -m pip install -r requirements/eda.txt
python -m scripts.eda --data-dir data/widsdatathon2025
# Or run only one analysis:
python -m scripts.eda --data-dir data/widsdatathon2025 --section quantitative
```

Open `outputs/eda/<section>/index.html`. Charts and full CSV tables are saved
beside each report; interactive charts work offline. Connectome EDA includes
the group difference heatmaps, Louvain networks, community summaries, subject
graph features and statistical tests. These label-derived community features
are descriptive analysis, not inputs to the final prediction models.
The full connectome analysis is CPU intensive. No test data is loaded by EDA.

## Train classical models on CPU

```bash
python -m pip install -r requirements/train.txt
python -m scripts.train_ml --data-dir data/widsdatathon2025
# Independently train either target, or use the notebook's SVC/PCA ADHD variant:
python -m scripts.train_ml --data-dir data/widsdatathon2025 --target adhd --adhd-model svc
python -m scripts.train_ml --data-dir data/widsdatathon2025 --target sex --top-n 463
```

The defaults reproduce the notebook's final model choices: 12 questionnaire
features with median imputation, scaling and logistic regression (`C=0.01`)
for ADHD; up to 463 significant connectome features ranked by absolute Pearson
correlation, median imputation and GaussianNB for Sex_F. Selection, imputation
and scaling fit only training rows during validation; a fresh pipeline then
fits the full training data for submission. The other target is never used as
an input feature.

Saved files in `outputs/models/`: `adhd_model.joblib`, `sex_model.joblib`,
per-target validation metrics and the Sex_F feature correlation table. Each
artifact includes feature names and fitted preprocessing, so prediction does
not need EDA outputs or notebook variables.

## Train the GNN separately

Install a PyTorch build matching the GPU/driver on the training machine, then:

```bash
python -m pip install -r requirements/gpu.txt
python -m scripts.train_gnn --data-dir data/widsdatathon2025 --device cuda --epochs 200 --batch-size 32
# CPU smoke run, or reduce batch size if GPU memory is limited:
python -m scripts.train_gnn --data-dir data/widsdatathon2025 --device cpu --epochs 1 --batch-size 4
```

`--device auto` (the default) selects CUDA if available, otherwise CPU;
`--device cuda` fails immediately if CUDA is unavailable. EDA and classical
training never initialize a GPU. Graphs are constructed on demand on CPU and
only each minibatch is transferred to the selected device.

The default GNN uses the original identity node features, dense signed edges,
three GraphConv/BatchNorm layers, hidden size 64, AdamW and learning rate
`1e-5`. The later hybrid submission's graph setup is also available:

```bash
python -m scripts.train_gnn --data-dir data/widsdatathon2025 --device cuda --node-features strength --edges positive --learning-rate 1e-4 --epochs 100
```

Training uses a joint-target stratified 70/15/15 train/validation/test split.
Checkpoint and probability threshold selection use validation rows; the test
partition is evaluated once afterward. `outputs/models/sex_gnn.pt` saves the
best validation checkpoint, graph/model configuration and threshold for
inference. Training history, test predictions and metrics are saved alongside
it. The saved model is the best checkpoint from the training split; the script
does not refit it on all competition participants.

## Predict from saved models

```bash
# Both classical models:
python -m scripts.predict --data-dir data/widsdatathon2025
# Classical ADHD model + GNN Sex_F model:
python -m scripts.predict --data-dir data/widsdatathon2025 --sex-model gnn --device cuda
```

Output: `outputs/submission.csv`, with columns `participant_id`, `ADHD_Outcome`,
`Sex_F`. Use `--sample-submission /path/to/sample_submission.csv` to match its
participant order. Predictions join by ID even if the input tables are in
different orders. Missing predictions raise an error rather than becoming
arbitrary zero labels. Nothing is submitted to Kaggle automatically.

Use `--output-dir` on EDA/training/search commands, `--model-dir` on prediction,
or `--output` for the submission path when working on different machines.
Copy the model directory to the inference machine and install the corresponding
training requirements there. Keep the same dependency versions for training
and inference when transferring serialized models.

## Optional experiments and checks

```bash
python -m scripts.experiments --data-dir data/widsdatathon2025 --task compare
python -m scripts.experiments --data-dir data/widsdatathon2025 --task strategies
python -m scripts.experiments --data-dir data/widsdatathon2025 --task tune
python -m scripts.experiments --data-dir data/widsdatathon2025 --task connectome
# Optional additional CPU classifiers:
python -m pip install -r requirements/experiments.txt
python -m scripts.experiments --data-dir data/widsdatathon2025 --task compare --boosting

python -m unittest discover -s tests -v
```

Search results go to `outputs/experiments/`. The feature-count sweep chooses
the count with cross-validation, then evaluates the holdout once. The original
notebook selected some features/counts using the evaluation labels and imputed
before splitting; the scripts fix these sources of leakage. Consequently the
notebook's historical scores are not expected to match new validation scores.
Tests use synthetic data to check file formats, saved preprocessing, ID/order
alignment, inference without fitting, CPU dependency boundaries and graph
reconstruction. GNN tests skip when its optional dependencies are absent.

Competition data and generated outputs are excluded by `.gitignore`.

## Original competition description

WiDS Datathon 2025
Unraveling the Mysteries of the Female Brain: Sex Patterns in ADHD
Last Updated: 2 years ago

About this Competition
The WiDS Datathon was developed with Ann S. Bowers Women’s Brain Health Initiative (WBHI) in collaboration with Cornell University and UC Santa Barbara. Datasets and support are provided by the Healthy Brain Network (HBN), the signature scientific initiative of the Child Mind Institute, and the Reproducible Brain Charts project (RBC).

We will analyze diagnostic data, socio-demographic, emotions, and parenting data, and functional MRI data from the Healthy Brain Network (HBN) — the signature scientific initiative of the Child Mind Institute. HBN utilizes a community-referred recruitment model by encouraging the participation of families who have concerns about mental health or learning challenges in their children. The functional MRI data are used to extract a time series of activity per brain region, and these regions’ time series are correlated to obtain functional MRI connectome matrices.

The goal is to build a model to predict both an individual’s sex and their ADHD diagnosis using functional brain imaging data of children and adolescents and their socio-demographic, emotions, and parenting information.

UPDATE: Please note that new Training Data was provided as of March 20th (See this discussion post for details). Please use the updated training data in dataset folder titled: Train_New

Dataset Folders
(1) the training folder train_new_tsv consists of three types of information about the 1,200+ subjects. They are:

a) the targets (ADHD diagnosis and sex)
b) functional MRI connectome matrices
c) socio-demographic information, e.g., subject’s “handedness” or “parent’s education level”, emotions (“Strength and Difficulties Questionnaire”), and parenting information (“Alabama Parenting Questionnaire”). These include both quantitative and categorical metadata.
Friendly hint: Participants will need to process the categorical data (perhaps create dummy variables) and then combine this processed dataset with the functional connectome dataset to create a final training dataset to use in their models.

(2) the test folder test_tsv consists of unseen data frames for 300+ subjects. These data frames are as follows:

a) functional MRI connectome matrices
b) socio-demographic, emotions, and parenting information
To participate in the Datathon, you will submit a solution file containing the type of ADHD diagnosis and the sex for each row in the test dataset. The predicted values you submit will be compared against the observed values for the test dataset and this will determine your standing on the Leaderboard during the competition as well as your final standing when the competition closes.

You will also be provided an example solution file prepared for submission.

External Data Usage
The datathon task can be tackled successfully without the use of external data. In fact, the degree to which we have anonymized the data would make joining additional data to the competition data difficult.

Target Variables
ADHD_Outcome: Type of Diagnosis (0=Other/None, 1=ADHD) Sex_F: Sex of participant (0=Male, 1=Female)

You will also be provided a full data dictionary
