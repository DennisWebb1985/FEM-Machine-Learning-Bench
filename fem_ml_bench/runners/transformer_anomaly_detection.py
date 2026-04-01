from __future__ import annotations

from pathlib import Path

from torch.optim import AdamW

from src.models.transformer_anomaly_detection import TransformerAutoencoder

from .sequence_anomaly_detection import SequenceAnomalySpec, run_sequence_anomaly_detection


SPEC = SequenceAnomalySpec(
    default_config=Path(__file__).resolve().parents[2] / "experiments/configs/transformer_anomaly_detection.yaml",
    model_key="transformer_autoencoder",
    example_name="Transformer",
    title_root="Transformer Anomaly Detection",
    model_builder=lambda config, input_size, window_length: TransformerAutoencoder(
        input_size=input_size,
        sequence_length=window_length,
        patch_size=int(config["model"]["patch_size"]),
        d_model=int(config["model"]["d_model"]),
        nhead=int(config["model"]["nhead"]),
        num_layers=int(config["model"]["num_layers"]),
        dim_feedforward=int(config["model"]["dim_feedforward"]),
        dropout=float(config["model"]["dropout"]),
    ),
    optimizer_builder=lambda model, config: AdamW(
        model.parameters(),
        lr=float(config["training"]["lr"]),
        weight_decay=float(config["training"]["weight_decay"]),
    ),
)


def main(argv: list[str] | None = None) -> int:
    return run_sequence_anomaly_detection(SPEC, argv)
