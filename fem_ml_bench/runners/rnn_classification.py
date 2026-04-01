from __future__ import annotations

from pathlib import Path

from torch.optim import Adam

from src.models.rnn_classification import RNNClassifier

from .sequence_classification import SequenceClassificationSpec, run_sequence_classification


SPEC = SequenceClassificationSpec(
    default_config=Path(__file__).resolve().parents[2] / "experiments/configs/rnn_classification.yaml",
    model_key="rnn",
    display_name="RNN",
    shap_prefix="rnn_classification",
    model_builder=lambda config, input_size, num_classes: RNNClassifier(
        input_size=input_size,
        hidden_size=int(config["model"]["hidden_size"]),
        num_classes=num_classes,
        num_layers=int(config["model"]["num_layers"]),
        dropout=float(config["model"]["dropout"]),
        bidirectional=bool(config["model"].get("bidirectional", False)),
    ),
    optimizer_builder=lambda model, config: Adam(
        model.parameters(),
        lr=float(config["training"]["lr"]),
        weight_decay=float(config["training"]["weight_decay"]),
    ),
)


def main(argv: list[str] | None = None) -> int:
    return run_sequence_classification(SPEC, argv)
