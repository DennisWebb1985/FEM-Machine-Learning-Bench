from __future__ import annotations

from pathlib import Path

from torch.optim import Adam

from src.models.rnn_anomaly_detection import RNNAutoencoder

from .sequence_anomaly_detection import SequenceAnomalySpec, run_sequence_anomaly_detection


SPEC = SequenceAnomalySpec(
    default_config=Path(__file__).resolve().parents[2] / "experiments/configs/rnn_anomaly_detection.yaml",
    model_key="rnn_autoencoder",
    example_name="RNN",
    title_root="RNN Anomaly Detection",
    model_builder=lambda config, input_size, window_length: RNNAutoencoder(
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
