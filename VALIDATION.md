# Validation Report

Validated in the build environment before packaging.

## Automated tests

```text
11 passed
```

Covered areas:

- configuration sanity
- CSV pairing/consolidation
- preprocessing with missing values
- BPSO sigmoid and optimization behaviour
- fuzzy membership and Low/High risk cases
- neural-network evaluation metrics
- end-to-end pipeline on synthetic sample data

## Smoke checks

The following completed successfully:

```text
python scripts/make_sample_dataset.py
python scripts/train_pipeline.py --dataset data/sample/sample_kitsune.csv --mode demo --particles 4 --iterations 3
python scripts/smoke_test.py
python -m py_compile app.py bpso_ids/*.py scripts/*.py tests/*.py
```

The smoke prediction produced a valid neural-network prediction, confidence score, and fuzzy risk result.

## Environment limitation

The build container did not have NiceGUI installed and could not reach the public Python package index to install it. Therefore the NiceGUI browser UI was **not live-launched in this environment**. The application source was syntax-checked, and NiceGUI is pinned in `requirements.txt` for the user's VS Code environment.

The actual Kaggle download was also not executed in the build container because the runtime did not have external package/network access. The KaggleHub downloader and dataset inspection/consolidation path are included for execution on the user's machine.

## Research-data warning

The bundled `data/sample/sample_kitsune.csv` is synthetic and is only for software tests. It must not be reported as the research dataset or used for final paper results.
