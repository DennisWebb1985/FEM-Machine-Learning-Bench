from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

from fem_ml_bench.project import PROJECT_ROOT
from src.utils.config import dump_config, load_config
from src.utils.io import ensure_dir


def default_train_device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


def resolve_device(requested_device: str, *, allow_auto: bool = False) -> torch.device:
    normalized = requested_device.strip().lower()
    mps_is_available = hasattr(torch.backends, "mps") and torch.backends.mps.is_available()

    if normalized == "auto":
        if not allow_auto:
            normalized = default_train_device()
        elif mps_is_available:
            normalized = "mps"
        elif torch.cuda.is_available():
            normalized = "cuda"
        else:
            normalized = "cpu"

    if normalized == "mps" and not mps_is_available:
        raise RuntimeError(
            "MPS was requested, but torch.backends.mps.is_available() is False. "
            "Check that you are using a recent macOS version and a PyTorch build with MPS support."
        )

    return torch.device(normalized)


def load_runner_config(config_path: str | Path, *, epochs: int | None = None) -> dict[str, Any]:
    config = load_config(resolve_repo_path(config_path))
    if epochs is not None:
        config.setdefault("training", {})
        config["training"]["epochs"] = epochs
    return config


def resolve_repo_path(path: str | Path) -> Path:
    raw_path = Path(path)
    if raw_path.is_absolute():
        return raw_path
    return PROJECT_ROOT / raw_path


def prepare_output_dir(config: dict[str, Any]) -> Path:
    output_dir = ensure_dir(PROJECT_ROOT / config["paths"]["output_dir"])
    dump_config(config, output_dir / "resolved_config.yaml")
    return output_dir
