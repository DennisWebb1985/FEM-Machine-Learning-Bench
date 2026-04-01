from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.utils.runtime import prepare_environment

PROJECT_ROOT = prepare_environment(ROOT)

import torch
from torch.utils.data import DataLoader

from src.data.anomaly_detection_preprocess import (
    build_anomaly_reconstruction_windows,
    prepare_anomaly_detection_data,
)
from src.data.datasets import SequenceRegressionDataset
from src.models.lstm_anomaly_detection_autoencoder import LSTMAutoencoder
from src.models.rnn_anomaly_detection import RNNAutoencoder
from src.models.transformer_anomaly_detection import TransformerAutoencoder
from src.training.anomaly_engine import evaluate_sequence_model, representative_window_index
from src.utils.config import load_config
from src.utils.plotting import plot_sequence_comparison


RUN_SPECS = {
    "rnn": PROJECT_ROOT / "experiments/configs/rnn_anomaly_detection.yaml",
    "lstm_autoencoder": PROJECT_ROOT / "experiments/configs/lstm_anomaly_detection_autoencoder.yaml",
    "transformer": PROJECT_ROOT / "experiments/configs/transformer_anomaly_detection.yaml",
}


def select_device(requested: str) -> str:
    if requested != "auto":
        return requested
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def build_model(model_key: str, config: dict, input_size: int) -> torch.nn.Module:
    if model_key == "rnn":
        return RNNAutoencoder(
            input_size=input_size,
            hidden_size=config["model"]["hidden_size"],
            latent_size=config["model"]["latent_size"],
            num_layers=config["model"]["num_layers"],
            dropout=config["model"]["dropout"],
        )
    if model_key == "lstm_autoencoder":
        return LSTMAutoencoder(
            input_size=input_size,
            hidden_size=config["model"]["hidden_size"],
            latent_size=config["model"]["latent_size"],
            num_layers=config["model"]["num_layers"],
            dropout=config["model"]["dropout"],
        )
    if model_key == "transformer":
        return TransformerAutoencoder(
            input_size=input_size,
            sequence_length=config["window"]["length"],
            patch_size=config["model"]["patch_size"],
            d_model=config["model"]["d_model"],
            nhead=config["model"]["nhead"],
            num_layers=config["model"]["num_layers"],
            dim_feedforward=config["model"]["dim_feedforward"],
            dropout=config["model"]["dropout"],
        )
    raise ValueError(f"Unsupported model_key: {model_key}")


def regenerate_examples(model_key: str, run_dir: Path, device_name: str) -> None:
    config = load_config(run_dir / "resolved_config.yaml")
    prepared = prepare_anomaly_detection_data(
        data_dir=PROJECT_ROOT / config["paths"]["data_dir"],
        train_ratio=config["data"]["train_ratio"],
        val_ratio=config["data"]["val_ratio"],
        drop_channels=config["data"]["drop_channels"],
        missing_sentinel=config["data"]["missing_sentinel"],
        extreme_value_threshold=config["data"]["extreme_value_threshold"],
        downsample_factor=config["data"]["downsample_factor"],
    )
    test_x, test_y, test_meta = build_anomaly_reconstruction_windows(
        prepared["test"],
        prepared["feature_cols"],
        window_length=config["window"]["length"],
        stride=config["window"]["stride"],
        drop_nan_windows=config["window"]["drop_nan_windows"],
    )

    test_ds = SequenceRegressionDataset(test_x, test_y, test_meta)
    batch_size = config["training"]["batch_size"]
    num_workers = config["training"].get("num_workers", 0)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    device = torch.device(device_name)
    model = build_model(model_key=model_key, config=config, input_size=test_x.shape[-1]).to(device)
    state_dict = torch.load(run_dir / "best_model.pt", map_location=device)
    model.load_state_dict(state_dict)
    criterion = torch.nn.MSELoss()

    test_eval = evaluate_sequence_model(model, test_loader, criterion, device, True)
    low_index = representative_window_index(test_eval["window_scores"], quantile=0.10)
    high_index = representative_window_index(test_eval["window_scores"], quantile=0.95)
    title_prefix = {
        "rnn": "RNN",
        "lstm_autoencoder": "LSTM",
        "transformer": "Transformer",
    }[model_key]

    plot_sequence_comparison(
        target_sequence=test_eval["targets"][low_index],
        predicted_sequence=test_eval["predictions"][low_index],
        feature_names=prepared["feature_cols"],
        output_path=run_dir / "example_low_score.png",
        title=f"{title_prefix} Low-Score Window (q10)",
        feature_selection="lowest_error_high_variance",
    )
    plot_sequence_comparison(
        target_sequence=test_eval["targets"][high_index],
        predicted_sequence=test_eval["predictions"][high_index],
        feature_names=prepared["feature_cols"],
        output_path=run_dir / "example_high_score.png",
        title=f"{title_prefix} High-Score Window (q95)",
        feature_selection="largest_error",
    )


def resolve_run_dir(config_path: Path) -> Path:
    config = load_config(config_path)
    run_dir = PROJECT_ROOT / config["paths"]["output_dir"]
    required = [
        "best_model.pt",
        "resolved_config.yaml",
        "auto_window_selection.json",
        "example_low_score.png",
        "example_high_score.png",
    ]
    missing = [name for name in required if not (run_dir / name).exists()]
    if missing:
        raise FileNotFoundError(
            f"Expected a final auto-selected anomaly run at {run_dir}, but missing: {', '.join(missing)}."
        )
    return run_dir


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()

    device_name = select_device(args.device)
    for model_key, config_path in RUN_SPECS.items():
        run_dir = resolve_run_dir(config_path)
        regenerate_examples(model_key=model_key, run_dir=run_dir, device_name=device_name)
        print(f"regenerated {run_dir.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
