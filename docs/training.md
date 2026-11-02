# Model training guide

[← Back to the project](../readme.md) · [EDA presentation](eda/index.html)

Train models independently of EDA. All commands below run from the project root
after activating `.venv` and downloading the competition data.

## Choose a training route

| Route | What it trains | Command | Hardware |
| --- | --- | --- | --- |
| **Classical ML — default** | ADHD logistic regression + Sex_F GaussianNB | `python -m scripts.train_ml --data-dir data/widsdatathon2025` | CPU |
| **Graph learning — optional** | Sex_F GNN | `python -m scripts.train_gnn --data-dir data/widsdatathon2025 --device cuda` | CUDA or CPU |

For a complete submission, use either **both classical models**, or
**a classical ADHD model + the Sex_F GNN**. The GNN predicts Sex_F only.

## Data needed

| Workflow | Required tables |
| --- | --- |
| ADHD training | Training quantitative metadata + training solutions |
| Sex_F training, classical or GNN | Training connectome matrices + training solutions |
| Prediction | Test quantitative metadata + test connectome matrices |

The scripts search recursively under `--data-dir`, accept the Excel/CSV and
TSV releases, and prefer revised training files (`TRAIN_NEW` / `train_new_tsv`).
Keep one file format per release. IDs must be unique and match between the
relevant tables. Test tables are used only for prediction.

## A. Classical models · CPU

### Run training

```bash
python -m pip install -r requirements/train.txt

# Train both default models.
python -m scripts.train_ml --data-dir data/widsdatathon2025

# Train one target independently.
python -m scripts.train_ml --data-dir data/widsdatathon2025 --target adhd
python -m scripts.train_ml --data-dir data/widsdatathon2025 --target sex --top-n 463
```

### ADHD model

| Setting | Default |
| --- | --- |
| Target | `ADHD_Outcome`: 0 = other/none, 1 = ADHD |
| Inputs | 12 questionnaire features listed below |
| Missing values | Median imputation |
| Scaling | StandardScaler |
| Classifier | LogisticRegression |
| Regularization | L2, `C=0.01` |
| Solver / maximum iterations | `lbfgs` / 1,000 |
| Random seed | 8828 |

<details>
<summary>The 12 input features</summary>

| Questionnaire | Features |
| --- | --- |
| SDQ — behavioral difficulties | `SDQ_SDQ_Conduct_Problems`, `SDQ_SDQ_Difficulties_Total`, `SDQ_SDQ_Externalizing`, `SDQ_SDQ_Generating_Impact`, `SDQ_SDQ_Hyperactivity` |
| SDQ — emotions | `SDQ_SDQ_Emotional_Problems`, `SDQ_SDQ_Internalizing` |
| APQ — parenting | `APQ_P_APQ_P_ID`, `APQ_P_APQ_P_INV`, `APQ_P_APQ_P_PM` |
| SDQ — peers and prosocial behavior | `SDQ_SDQ_Peer_Problems`, `SDQ_SDQ_Prosocial` |

</details>

**Alternative: SVC + PCA.** This uses all quantitative features except
participant ID, target columns and `MRI_Track_Age_at_Scan`. The pipeline applies
median imputation → scaling → PCA retaining 95% variance → SVC.

```bash
python -m scripts.train_ml \
  --data-dir data/widsdatathon2025 \
  --target adhd \
  --adhd-model svc
```

The alternative writes the same `adhd_model.joblib` filename. Use a different
`--output-dir` to keep multiple versions.

### Sex_F model

| Setting | Default |
| --- | --- |
| Target | `Sex_F`: 0 = male, 1 = female |
| Inputs | Flattened functional connectome features |
| Selection | Pearson significance `p < 0.05`, ranked by descending absolute correlation |
| Feature count | Up to 463; adjustable with `--top-n` |
| Missing values | Median imputation after feature selection |
| Classifier | GaussianNB |

Feature selection is part of the saved pipeline. It uses only the labels from
the rows supplied to `fit`, so validation labels cannot influence selection.

### Evaluation and final fit

1. **Split:** 80% training, 20% validation; stratify by the combined ADHD/sex groups.
2. **Fit:** learn feature selection, imputation, scaling and the model using training rows only.
3. **Evaluate:** report validation accuracy and binary F1 for each target.
4. **Refit:** fit a fresh pipeline on the full competition training set.
5. **Save:** store the final pipeline, input feature names, seed and validation metrics.

The other target is never an input feature. Validation metrics describe the
held-out evaluation model, while the saved inference model is the full-data refit.

### Saved files

Default directory: **`outputs/models/`**.

| File | Contents |
| --- | --- |
| `adhd_model.joblib` | ADHD pipeline, feature names and metadata |
| `sex_model.joblib` | Sex_F pipeline, feature names and metadata |
| `adhd_metrics.json`, `sex_metrics.json` | Per-target validation accuracy, F1 and split sizes |
| `sex_feature_correlations.csv` | Correlations calculated during the final full-data Sex_F fit |

## B. Graph neural network · CUDA or CPU

### Run training

Install a PyTorch build matching your GPU/driver before installing the remaining
GNN requirements.

```bash
python -m pip install -r requirements/gpu.txt

python -m scripts.train_gnn \
  --data-dir data/widsdatathon2025 \
  --device cuda \
  --epochs 200 \
  --batch-size 32
```

### Architecture and defaults

| Component | Setting |
| --- | --- |
| Prediction target | Sex_F only |
| Graph | 200 brain regions; dense directed edge pairs with signed correlation weights |
| Node features | Identity vectors; 200 input features per node |
| Graph layers | 3 × GraphConv → BatchNorm → ReLU |
| Hidden widths | 64 → 128 → 256 |
| Readout | Global mean pooling → linear 256→128 → ReLU → linear 128→1 |
| Dropout | 0.5 around the readout/classifier |
| Loss | BCEWithLogitsLoss; positive weight = male/female count in training rows |
| Optimizer | AdamW, learning rate `1e-5`, weight decay `1e-5` |
| Epochs / batch size | 200 / 32 |
| Seed | 8828 |

Graphs are reconstructed from feature column names. Missing or non-finite
correlations become zero. Graphs are built on demand on CPU; only the current
minibatch moves to the training device.

### Device and memory controls

| Option | Behavior |
| --- | --- |
| `--device cuda` | Require CUDA; fail immediately if unavailable |
| `--device cpu` | Run on CPU |
| `--device auto` | Default: CUDA if available, otherwise CPU |
| `--batch-size 4` | Use a smaller minibatch when GPU memory is limited |
| `--epochs 1` | Short execution check; does not establish model quality |

**CPU smoke run:**

```bash
python -m scripts.train_gnn \
  --data-dir data/widsdatathon2025 \
  --device cpu --epochs 1 --batch-size 4
```

<details>
<summary>Alternative graph setup from the later hybrid notebook pipeline</summary>

Use node strength (sum of absolute off-diagonal correlations), keep positive
edges, and train for 100 epochs at learning rate `1e-4`:

```bash
python -m scripts.train_gnn \
  --data-dir data/widsdatathon2025 \
  --device cuda \
  --node-features strength \
  --edges positive \
  --learning-rate 1e-4 \
  --epochs 100
```

Other options include `--hidden-channels`, `--num-nodes` and `--edge-threshold`.
Run `python -m scripts.train_gnn --help` for the complete interface.

</details>

### Evaluation and checkpoint selection

1. **Split:** approximately 70% training, 15% validation, 15% test, stratified by both targets.
2. **Train:** compute the class weight from training Sex_F labels and update the model each epoch.
3. **Select checkpoint:** keep the epoch with the best validation F1 at threshold 0.5.
4. **Select threshold:** use that checkpoint's validation predictions to choose an F1-maximizing threshold from 0.01–0.99.
5. **Evaluate once:** report test loss, F1, accuracy and ROC AUC on the untouched test partition.
6. **Save:** store the checkpoint, graph/model settings and selected threshold.

The saved GNN is the best training-split checkpoint. It is **not refitted on all
competition participants** after evaluation.

### Saved files

| File in `outputs/models/` | Contents |
| --- | --- |
| `sex_gnn.pt` | Model weights, graph settings, threshold, best epoch and seed |
| `gnn_history.csv` | Training loss, validation loss and validation F1 per epoch |
| `gnn_metrics.json` | Selected epoch/threshold, test metrics and split sizes |
| `gnn_test_predictions.csv` | Test participant IDs, Sex_F labels and predicted probabilities |

## C. Prediction from saved models

```bash
# Classical ADHD + classical Sex_F.
python -m scripts.predict --data-dir data/widsdatathon2025

# Classical ADHD + GNN Sex_F.
python -m scripts.predict \
  --data-dir data/widsdatathon2025 \
  --sex-model gnn --device cuda
```

| Option | Purpose |
| --- | --- |
| `--model-dir PATH` | Read saved models from a different directory |
| `--output PATH` | Change the default `outputs/submission.csv` destination |
| `--sample-submission PATH` | Match the sample submission's participant order |
| `--batch-size N` | Set the GNN inference batch size |

Prediction uses saved preprocessing and joins outputs by participant ID. It does
not fit models or require EDA outputs. Missing predictions raise an error.

To train on one machine and predict on another, copy the model directory and
use matching dependency versions with the same `scripts` package available.

## Interpreting results

The HTML presentation shows **historical notebook outputs**. The refactored
scripts fix preprocessing and feature-selection leakage in earlier experiments,
so new validation scores can differ from the historical notebook scores.

The regression tests use **synthetic data to verify execution and alignment**.
They do not measure competition performance. CUDA execution and full training
on the competition files have not been verified in this local refactoring run.
