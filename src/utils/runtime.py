from __future__ import annotations

import os
import sys
from pathlib import Path


def prepare_environment(project_root: str | Path | None = None) -> Path:
    """Prepare import paths and matplotlib cache before heavy imports."""

    root = Path(project_root or Path(__file__).resolve().parents[2]).resolve()
    os.environ.setdefault("MPLBACKEND", "Agg")
    mpl_dir = root / ".mplconfig"
    mpl_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(mpl_dir))

    try:
        import shap  # noqa: F401
    except ModuleNotFoundError:
        vendor_dir = root / "_vendor"
        if vendor_dir.exists():
            sys.path.insert(0, str(vendor_dir))

    return root
