from __future__ import annotations

from pathlib import Path

from torch.optim import Adam

from src.models.lstm_anomaly_detection_autoencoder import LSTMAutoencoder

from .sequence_anomaly_detection import SequenceAnomalySpec, run_sequence_anomaly_detection


SPEC = SequenceAnomalySpec(
    default_config=Path(__file__).resolve().parents[2] / "experiments/configs/lstm_anomaly_detection_autoencoder.yaml",
    model_key="lstm_autoencoder",
    example_name="LSTM",
    title_root="LSTM Anomaly Detection",
    qualifier="Autoencoder",
    model_builder=lambda config, input_size, window_length: LSTMAutoencoder(
        input_size=input_size,
        hidden_size=int(config["model"]["hidden_size"]),
        latent_size=int(config["model"]["latent_size"]),
        num_layers=int(config["model"]["num_layers"]),
        dropout=float(config["model"]["dropout"]),
    ),
    optimizer_builder=lambda model, config: Adam(
        model.parameters(),
        lr=float(config["training"]["lr"]),
        weight_decay=float(config["training"]["weight_decay"]),
    ),
)


def main(argv: list[str] | None = None) -> int:
    return run_sequence_anomaly_detection(SPEC, argv)
