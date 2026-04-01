from __future__ import annotations

from pathlib import Path

from torch.optim import AdamW

from src.models.transformer_classification import TransformerClassifier

from .auto_tuning import TuningProfile
from .sequence_classification import SequenceClassificationSpec, run_sequence_classification


SPEC = SequenceClassificationSpec(
    default_config=Path(__file__).resolve().parents[2] / "experiments/configs/transformer_classification.yaml",
    model_key="transformer",
    display_name="Transformer",
    shap_prefix="transformer_classification",
    model_builder=lambda config, input_size, num_classes: TransformerClassifier(
        input_size=input_size,
        num_classes=num_classes,
        d_model=int(config["model"]["d_model"]),
        nhead=int(config["model"]["nhead"]),
        num_layers=int(config["model"]["num_layers"]),
        dim_feedforward=int(config["model"]["dim_feedforward"]),
        dropout=float(config["model"]["dropout"]),
        pooling=str(config["model"]["pooling"]),
    ),
    optimizer_builder=lambda model, config: AdamW(
        model.parameters(),
        lr=float(config["training"]["lr"]),
        weight_decay=float(config["training"]["weight_decay"]),
    ),
    tuning_profiles=(
        TuningProfile(
            name="baseline",
            overrides={},
        ),
        TuningProfile(
            name="last_pool_light",
            overrides={
                "training": {
                    "label_smoothing": 0.0,
                    "class_weight_mode": "none",
                },
                "model": {
                    "dropout": 0.10,
                    "pooling": "last",
                },
            },
        ),
        TuningProfile(
            name="last_pool_deeper",
            overrides={
                "training": {
                    "epochs": 80,
                    "lr": 0.001,
                    "weight_decay": 0.0001,
                    "patience": 12,
                    "label_smoothing": 0.0,
                    "class_weight_mode": "sqrt_inverse",
                    "monitor_min_delta": 0.0002,
                },
                "model": {
                    "d_model": 32,
                    "num_layers": 2,
                    "dim_feedforward": 64,
                    "dropout": 0.10,
                    "pooling": "last",
                },
            },
        ),
    ),
)


def main(argv: list[str] | None = None) -> int:
    return run_sequence_classification(SPEC, argv)
