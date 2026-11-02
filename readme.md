# ADHD & Sex Prediction · WiDS Datathon 2025

My 2025 hackathon project exploring **questionnaire scores, demographics and
functional brain connectivity** to predict ADHD diagnosis and participant sex.

**Start with the [EDA presentation](docs/eda/index.html)** to explore the analysis,
or go to the [model training guide](docs/training.md) to run the models.

## 1. Explore the analysis

The presentation reports are included in this repository. They contain the
**original charts, tables and observations saved in the 2025 notebook**.
You do not need the competition data, a Python environment or a GPU to view them.

| Report | What you will see | HTML file |
| --- | --- | --- |
| **Overview** | Choose a report and navigate the presentation | [Open overview](docs/eda/index.html) |
| **Demographics** | Category distributions and sex/diagnosis group breakdowns | [Categorical EDA](docs/eda/categorical.html) |
| **Questionnaires** | Missing values, distributions and feature correlations | [Quantitative EDA](docs/eda/quantitative.html) |
| **Brain connectivity** | Difference heatmaps, interactive networks and communities | [Connectome EDA](docs/eda/connectome.html) |

**To present:** [download the project ZIP](https://github.com/WuqianMa/ADHD_Gender_classification/archive/refs/heads/main.zip),
extract it, and open `docs/eda/index.html` in your browser. Keep the `assets/`
folder beside the HTML files; the charts also work offline.

GitHub displays HTML source in its file viewer. If you already cloned the
project, you can also serve the reports locally:

```bash
python3 -m http.server 8000 --directory docs
```

Then visit **[localhost:8000/eda/](http://localhost:8000/eda/)**.

> These reports present the saved 2025 analysis. They are separate from new
> runs of the refactored scripts and from the synthetic data used in tests.

## 2. Understand the models

There are **two prediction targets** and **two ways to train**:

| Target | Input | Default model | Preprocessing | Hardware |
| --- | --- | --- | --- | --- |
| `ADHD_Outcome` | 12 questionnaire features | Logistic regression | Median imputation → scaling | CPU |
| `Sex_F` | Functional brain connectivity | Gaussian Naive Bayes | Select up to 463 significant features → median imputation | CPU |
| `Sex_F` — alternative | Brain connectivity as a graph of 200 regions | Graph neural network (GNN) | Reconstruct graphs → create node features and edge weights | CUDA GPU or CPU |

**Evaluation:** classical models use an 80/20 train/validation split, then
refit on the full training set for prediction. The GNN uses a 70/15/15
train/validation/test split and saves its best validation checkpoint.
Splits preserve the joint ADHD/sex group distribution.

See the **[training guide](docs/training.md)** for model settings, feature lists,
evaluation steps, GPU options and saved artifacts.

## 3. Run the project

### Set up once

Use **Python 3.9+** and run commands from the project root.

```bash
python3 -m venv .venv
source .venv/bin/activate
```

Extract the competition files into `data/widsdatathon2025/`, or pass their
existing location with `--data-dir`. The loaders accept Excel, CSV and TSV
releases and join records by `participant_id`.

### Train both classical models · CPU

```bash
python -m pip install -r requirements/train.txt
python -m scripts.train_ml --data-dir data/widsdatathon2025
```

### Train the GNN · GPU

Install a PyTorch build suitable for your CUDA GPU and driver, then run:

```bash
python -m pip install -r requirements/gpu.txt
python -m scripts.train_gnn \
  --data-dir data/widsdatathon2025 \
  --device cuda \
  --epochs 200 \
  --batch-size 32
```

Use `--device cpu` for a CPU run. Reduce `--batch-size` if GPU memory is limited.
**EDA and classical model training do not import PyTorch or initialize a GPU.**

### Generate predictions

```bash
# Use the two saved classical models.
python -m scripts.predict --data-dir data/widsdatathon2025

# Or combine the ADHD classical model with the Sex_F GNN.
python -m scripts.predict --data-dir data/widsdatathon2025 --sex-model gnn --device cuda
```

Models and metrics are saved in **`outputs/models/`**. Predictions are saved to
**`outputs/submission.csv`** with columns `participant_id`, `ADHD_Outcome` and
`Sex_F`. Nothing is submitted to Kaggle automatically.

### Re-run EDA on the competition data

```bash
python -m pip install -r requirements/eda.txt
python -m scripts.eda --data-dir data/widsdatathon2025
```

New reports go to `outputs/eda/<section>/index.html`. Add
`--section categorical`, `--section quantitative` or `--section connectome`
to run one analysis. Categorical and quantitative EDA also need
`Data Dictionary.xlsx` in the data directory.

## Project map

| Location | Purpose |
| --- | --- |
| [`docs/eda/`](docs/eda/) | Included HTML presentation and chart assets |
| [`docs/training.md`](docs/training.md) | Detailed training reference and commands |
| [`docs/competition.md`](docs/competition.md) | Competition, dataset and target background |
| [`scripts/`](scripts/) | Independent EDA, training, search and prediction workflows |
| [`requirements/`](requirements/) | Separate dependencies for each workflow |
| [`tests/`](tests/) | Synthetic-data regression checks |
| [Original notebook](full-stack-adhd-classifier-ml-dl-graph-learnin.ipynb) | Full hackathon code, narrative and historical experiments |

<details>
<summary><strong>Developer commands: experiments, tests and presentation export</strong></summary>

```bash
# Optional CPU experiments: compare, tune, strategies or connectome.
python -m scripts.experiments --data-dir data/widsdatathon2025 --task compare

# Include CatBoost, LightGBM and XGBoost in comparisons.
python -m pip install -r requirements/experiments.txt
python -m scripts.experiments --data-dir data/widsdatathon2025 --task compare --boosting

# Run regression tests.
python -m unittest discover -s tests -v

# Rebuild the included presentation from saved notebook outputs; no cells run.
python -m pip install -r requirements/presentation.txt
python -m scripts.export_eda
```

Competition data, virtual environments and generated run outputs are excluded
from Git. The presentation under `docs/eda/` is intentionally included.

</details>
