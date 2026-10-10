# BPSO-Kitsune Intrusion Detection & Fuzzy Risk Assessment

A research-demo implementation based on the submitted SCOA paper:

> **PSO-Based Feature Selection for Efficient Neural Network Intrusion Detection with Fuzzy Risk Assessment**

The project implements the paper's main pipeline:

**Kitsune dataset → preprocessing → Binary PSO feature selection → fixed neural-network classifier → fuzzy risk assessment → evaluation/dashboard**

The research paper specifies BPSO as the feature-selection stage, a fixed one-hidden-layer neural network for controlled comparison, and an interpretable fuzzy layer producing Low/Medium/High risk. The paper also defines the fitness as validation F1 with a feature-count penalty. The initial BPSO settings are 20 particles and 30 iterations, with `w=0.7`, `c1=1.5`, `c2=1.5`; the neural network starts with 32 hidden neurons, ReLU, Adam, learning rate 0.001, 30 epochs and batch size 64.

## What is included

- KaggleHub downloader for the Kitsune dataset
- Robust CSV discovery and consolidation into one model-ready CSV
- Data profiling and preprocessing
- Stratified train/validation/test split
- Binary Particle Swarm Optimization (BPSO)
- Fitness = validation F1 − lambda × selected-feature ratio
- Same MLP architecture for baseline and BPSO-selected models
- Metrics: Accuracy, Precision, Recall, F1
- Feature-reduction statistics
- BPSO convergence history
- Lightweight fuzzy inference engine with triangular membership functions
- Single-record prediction and fuzzy-risk explanation
- NiceGUI dashboard
- CLI scripts for data preparation and training
- Artifact persistence so the dashboard can run without retraining
- Unit/integration tests, including a synthetic-data smoke path

## Important research note

The current paper contains illustrative result values that are explicitly marked as placeholders. This project does **not** hard-code those values. Run the real experiment and use the generated `artifacts/results.json`, `artifacts/bpso_history.csv`, and feature-selection outputs as the source for the final paper tables/figures.

## Project structure

```text
bpso_kitsune_fa_project/
├── app.py
├── config.py
├── requirements.txt
├── pytest.ini
├── README.md
├── .gitignore
├── .env.example
├── data/
│   ├── raw/.gitkeep
│   ├── processed/.gitkeep
│   └── sample/.gitkeep
├── artifacts/.gitkeep
├── bpso_ids/
│   ├── __init__.py
│   ├── config.py
│   ├── preprocessing.py
│   ├── bpso.py
│   ├── neural_network.py
│   ├── fuzzy.py
│   ├── pipeline.py
│   ├── data_io.py
│   └── evaluation.py
├── scripts/
│   ├── download_kitsune.py
│   ├── inspect_dataset.py
│   ├── consolidate_kitsune.py
│   ├── make_sample_dataset.py
│   ├── train_pipeline.py
│   └── smoke_test.py
└── tests/
    ├── test_config.py
    ├── test_data_io.py
    ├── test_preprocessing.py
    ├── test_bpso.py
    ├── test_fuzzy.py
    ├── test_evaluation.py
    └── test_pipeline.py
```

## Setup in VS Code / Windows

### 1. Create and activate a virtual environment

PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

### 2. Install dependencies

```powershell
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### 3. Verify the project without downloading Kaggle data

```powershell
python scripts/make_sample_dataset.py
python scripts/train_pipeline.py --dataset data/sample/sample_kitsune.csv --mode demo
pytest -q
```

The sample dataset is **synthetic and only for software validation**. It must not be used as the final research result.

### 4. Download the real Kitsune dataset

The downloader uses KaggleHub. Depending on your Kaggle account/environment, Kaggle authentication may be required.

```powershell
python scripts/download_kitsune.py
```

This prints the local dataset directory.

### 5. Inspect the downloaded files

```powershell
python scripts/inspect_dataset.py --root <PATH_PRINTED_BY_DOWNLOADER>
```

### 6. Consolidate the real CSV files

```powershell
python scripts/consolidate_kitsune.py --root <PATH_PRINTED_BY_DOWNLOADER> --output data/processed/kitsune_consolidated.csv
```

The consolidator looks for feature CSVs and label CSVs and validates row counts before concatenation.

### 7. Run the real training pipeline

For a fast pilot first:

```powershell
python scripts/train_pipeline.py --dataset data/processed/kitsune_consolidated.csv --mode demo
```

For the paper-aligned initial configuration:

```powershell
python scripts/train_pipeline.py --dataset data/processed/kitsune_consolidated.csv --mode research
```

**Recommendation:** start with `demo` mode, inspect the result, then run `research` mode after the pipeline is verified. Large Kitsune files can make BPSO expensive because every particle/iteration evaluates a candidate subset.

### 8. Launch the NiceGUI application

```powershell
python app.py
```

Then open the URL printed by NiceGUI, usually:

```text
http://127.0.0.1:8080
```

## Research vs Demo mode

The project has two modes.

### Demo mode

Used for rapid development and the FA presentation. It caps the training sample and uses a lower fitness-evaluation epoch budget so BPSO can complete on a normal laptop.

### Research mode

Uses the paper-aligned initial settings:

- BPSO particles: 20
- BPSO iterations: 30
- inertia `w`: 0.7
- cognitive `c1`: 1.5
- social `c2`: 1.5
- binary transform: sigmoid
- MLP hidden layers: 1
- MLP hidden neurons: 32
- activation: ReLU
- optimizer: Adam
- learning rate: 0.001
- epochs: 30
- batch size: 64

The research mode still uses a configurable row cap only if you explicitly pass one; otherwise it uses the consolidated dataset.

## Why the same neural network is used twice

The project trains:

1. **Baseline:** all preprocessed features → MLP
2. **Proposed:** BPSO-selected features → the same MLP architecture

This isolates the effect of feature selection instead of mixing feature reduction with architecture optimization.

## Fuzzy risk layer

The fuzzy layer is deliberately small. It consumes three numerical inputs selected from the final BPSO subset. Until the actual Kitsune feature semantics are inspected, the pipeline automatically chooses three high-variance selected columns for a working demo. For the final paper, replace those automatically selected columns with a defensible set of semantically meaningful Kitsune attributes and document the rule design.

The fuzzy engine uses triangular membership functions and interpretable Low/Medium/High linguistic labels. It is a separate interpretation layer, not a replacement for the neural classifier.

## Output artifacts

After training, `artifacts/` contains:

- `preprocessor.joblib`
- `baseline_model.joblib`
- `selected_model.joblib`
- `selected_features.json`
- `bpso_history.csv`
- `results.json`
- `fuzzy_config.json`

The dashboard reads these files to avoid re-running training every time it starts.

## Tests

```powershell
pytest -q
```

The test suite covers:

- configuration sanity
- dataset consolidation helpers
- preprocessing
- BPSO particle updates and optimization
- fuzzy membership/inference
- evaluation metrics
- end-to-end training on a deterministic synthetic dataset

## Safety / scope

This is a research and educational IDS prototype. It does not perform packet blocking, firewall changes, network isolation, or destructive actions. It should be presented as a detection and risk-assessment demonstrator.
