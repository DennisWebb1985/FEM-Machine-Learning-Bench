from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class RunnerSpec:
    task: str
    model: str
    module: str
    description: str
    default_config: Path | None = None


RUNNER_SPECS: tuple[RunnerSpec, ...] = (
    RunnerSpec(
        task="classification",
        model="rnn",
        module="fem_ml_bench.runners.rnn_classification",
        description="Building sequence classification with an RNN classifier.",
        default_config=PROJECT_ROOT / "experiments/configs/rnn_classification.yaml",
    ),
    RunnerSpec(
        task="classification",
        model="lstm",
        module="fem_ml_bench.runners.lstm_classification",
        description="Building sequence classification with an LSTM classifier.",
        default_config=PROJECT_ROOT / "experiments/configs/lstm_classification.yaml",
    ),
    RunnerSpec(
        task="classification",
        model="transformer",
        module="fem_ml_bench.runners.transformer_classification",
        description="Building sequence classification with a Transformer classifier.",
        default_config=PROJECT_ROOT / "experiments/configs/transformer_classification.yaml",
    ),
    RunnerSpec(
        task="anomaly_detection",
        model="rnn_autoencoder",
        module="fem_ml_bench.runners.rnn_anomaly_detection",
        description="Sequence anomaly detection with an RNN autoencoder.",
        default_config=PROJECT_ROOT / "experiments/configs/rnn_anomaly_detection.yaml",
    ),
    RunnerSpec(
        task="anomaly_detection",
        model="lstm_autoencoder",
        module="fem_ml_bench.runners.lstm_anomaly_detection_autoencoder",
        description="Sequence anomaly detection with an LSTM autoencoder.",
        default_config=PROJECT_ROOT / "experiments/configs/lstm_anomaly_detection_autoencoder.yaml",
    ),
    RunnerSpec(
        task="anomaly_detection",
        model="transformer_autoencoder",
        module="fem_ml_bench.runners.transformer_anomaly_detection",
        description="Sequence anomaly detection with a Transformer autoencoder.",
        default_config=PROJECT_ROOT / "experiments/configs/transformer_anomaly_detection.yaml",
    ),
    RunnerSpec(
        task="image_classification",
        model="cnn",
        module="fem_ml_bench.runners.cnn",
        description="Concrete-crack image classification with a CNN.",
        default_config=PROJECT_ROOT / "experiments/configs/cnn.yaml",
    ),
    RunnerSpec(
        task="regression",
        model="mlp",
        module="fem_ml_bench.runners.mlp",
        description="FEM surrogate regression with an MLP.",
        default_config=PROJECT_ROOT / "experiments/configs/mlp.yaml",
    ),
)

_RUNNER_INDEX = {(spec.task, spec.model): spec for spec in RUNNER_SPECS}


def list_runner_specs() -> tuple[RunnerSpec, ...]:
    return RUNNER_SPECS


def available_tasks() -> list[str]:
    return sorted({spec.task for spec in RUNNER_SPECS})


def available_models(task: str) -> list[str]:
    normalized_task = task.strip().lower()
    return sorted(spec.model for spec in RUNNER_SPECS if spec.task == normalized_task)


def resolve_runner(task: str, model: str) -> RunnerSpec:
    normalized_task = task.strip().lower()
    normalized_model = model.strip().lower()
    key = (normalized_task, normalized_model)
    if key not in _RUNNER_INDEX:
        known_models = ", ".join(available_models(normalized_task)) or "none"
        raise KeyError(
            f"Unknown task/model combination: {normalized_task}/{normalized_model}. "
            f"Available models for task '{normalized_task}': {known_models}."
        )
    return _RUNNER_INDEX[key]
