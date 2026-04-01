# FEM Machine Learning Bench

This repository is a consolidated experiments for FEM-based surrogate modeling, structural sequence classification, anomaly detection, and image classification.

## What is included

- reusable training code under `src/`
- experiment configs under `experiments/configs/`
- a single CLI entrypoint under `fem_ml_bench/`

Recommended placeholders:

- Building classification dataset: https://www.kaggle.com/datasets/daalgi/fem-simulations
- Anomaly-detection dataset: https://zenodo.org/records/8300495
- FEM MLP dataset: https://www.kaggle.com/datasets/ziya07/building-structural-health-sensor-dataset
- Concrete crack image dataset: https://data.mendeley.com/datasets/5y9wdsg2zt/2

After cloning this repository, recreate the dataset folders inside the repository root before running experiments:

```text
fem-ml-bench/
  data/
    raw/
      classification.csv
      anomaly_detection/
        *.csv
      mlp/
        5184doe.csv
        1000randoms.csv
      cnn/
        Negative/
          *.jpg
        Positive/
          *.jpg
```

Expected default paths in the included configs:

- classification models: `data/raw/classification.csv`
- anomaly-detection models: `data/raw/anomaly_detection/`
- regression MLP: `data/raw/mlp/5184doe.csv` and `data/raw/mlp/1000randoms.csv`
- CNN image classification: `data/raw/cnn/Negative/` and `data/raw/cnn/Positive/`

`outputs/` is created automatically when you run experiments and does not need to be restored manually.

## Single entrypoint

List supported runs:

```bash
python3 -m fem_ml_bench list
```

If you cloned the repository fresh, install dependencies first:

```bash
python3 -m pip install -r requirements.txt
```

Run a model with its default config:

```bash
python3 -m fem_ml_bench run --task classification --model rnn
python3 -m fem_ml_bench run --task anomaly_detection --model transformer_autoencoder
python3 -m fem_ml_bench run --task image_classification --model cnn
python3 -m fem_ml_bench run --task regression --model mlp
```

Sequence classification and anomaly detection now default to automatic window search before the final run.
Transformer classification also defaults to automatic hyperparameter-profile selection.

Pass runner-specific options directly:

```bash
python3 -m fem_ml_bench run --task classification --model transformer --device mps --epochs 5
python3 -m fem_ml_bench run --task image_classification --model cnn --limit-per-class 500 --skip-shap
```

Override the automatic search behavior only when you explicitly want fixed settings:

```bash
python3 -m fem_ml_bench run --task classification --model rnn --fixed-window
python3 -m fem_ml_bench run --task classification --model transformer --fixed-window --fixed-hparams
python3 -m fem_ml_bench run --task anomaly_detection --model transformer_autoencoder --window-lengths 128,256,384,512
```

Show the underlying runner help:

```bash
python3 -m fem_ml_bench run --task anomaly_detection --model lstm_autoencoder --runner-help
```

## Current task-model mapping

- `classification`: `rnn`, `lstm`, `transformer`
- `anomaly_detection`: `rnn_autoencoder`, `lstm_autoencoder`, `transformer_autoencoder`
- `image_classification`: `cnn`
- `regression`: `mlp`

## Notes

- `experiments/run_*.py` remain as thin compatibility wrappers around the package runners.
- Automatic searches store intermediate candidates under `outputs/auto_window_search/` and `outputs/auto_hparam_search/`.
- Sweep scripts remain available for controlled diagnostics and now pin fixed settings explicitly.
