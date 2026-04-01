from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
import re
from typing import Any, Callable

import pandas as pd

from fem_ml_bench.project import PROJECT_ROOT
from src.utils.config import dump_config
from src.utils.io import ensure_dir, write_dataframe, write_json


ConfigAdjuster = Callable[[dict[str, Any], int], None]
CandidateRunner = Callable[[dict[str, Any]], dict[str, Any]]
CandidateRowLoader = Callable[[Path, int], dict[str, Any]]

AUTO_WINDOW_ROOT = PROJECT_ROOT / "outputs" / "auto_window_search"


@dataclass(frozen=True)
class AutoWindowSelection:
    best_window_length: int
    best_row: dict[str, Any]
    search_root: Path
    summary_path: Path
    ranked_summary_path: Path
    best_row_path: Path
    metadata_path: Path


def parse_window_lengths(raw: str | None, default_lengths: tuple[int, ...]) -> list[int]:
    if raw is None or not raw.strip():
        lengths = list(default_lengths)
    else:
        lengths = [int(part.strip()) for part in raw.split(",") if part.strip()]
    if not lengths:
        raise ValueError("At least one window length is required.")
    if any(length <= 0 for length in lengths):
        raise ValueError("Window lengths must be positive integers.")
    return sorted(set(lengths))


def run_auto_window_search(
    *,
    task_name: str,
    model_key: str,
    base_config: dict[str, Any],
    window_lengths: list[int],
    selection_metric: str,
    sort_columns: list[str],
    ascending: list[bool],
    adjust_window: ConfigAdjuster,
    run_candidate: CandidateRunner,
    load_candidate_row: CandidateRowLoader,
    force: bool = False,
) -> AutoWindowSelection:
    experiment_name = str(base_config.get("experiment", {}).get("name", f"{task_name}_{model_key}"))
    search_root = ensure_dir(AUTO_WINDOW_ROOT / task_name / model_key / _safe_name(experiment_name))
    generated_config_dir = ensure_dir(search_root / "generated_configs")
    candidate_root = ensure_dir(search_root / "candidates")
    width = max(2, len(str(max(window_lengths))))

    rows: list[dict[str, Any]] = []
    for window_length in window_lengths:
        candidate_config = deepcopy(base_config)
        adjust_window(candidate_config, window_length)

        window_token = f"win_{window_length:0{width}d}"
        candidate_output_dir = candidate_root / window_token
        candidate_config.setdefault("experiment", {})
        candidate_config["experiment"]["name"] = f"{experiment_name}_auto_window_{window_token}"
        candidate_config.setdefault("paths", {})
        candidate_config["paths"]["output_dir"] = str(candidate_output_dir.relative_to(PROJECT_ROOT))

        config_path = generated_config_dir / f"{_safe_name(model_key)}_{window_token}.yaml"
        dump_config(candidate_config, config_path)

        metrics_path = candidate_output_dir / "metrics.json"
        if force or not metrics_path.exists():
            print(f"[auto-window] evaluating window={window_length}")
            row = run_candidate(candidate_config)
        else:
            print(f"[auto-window] reusing window={window_length} from {metrics_path.relative_to(PROJECT_ROOT)}")
            row = load_candidate_row(candidate_output_dir, window_length)

        row["window_length"] = window_length
        row["output_dir"] = str(candidate_output_dir.relative_to(PROJECT_ROOT))
        row["config_path"] = str(config_path.relative_to(PROJECT_ROOT))
        rows.append(row)

    summary = pd.DataFrame(rows).sort_values("window_length").reset_index(drop=True)
    ranked = summary.sort_values(sort_columns, ascending=ascending).reset_index(drop=True)
    best_row = ranked.iloc[0].to_dict()
    best_window_length = int(best_row["window_length"])

    summary_path = search_root / "summary.csv"
    ranked_summary_path = search_root / "ranked_summary.csv"
    best_row_path = search_root / "best_window.json"
    metadata_path = search_root / "run_metadata.json"

    write_dataframe(summary, summary_path, index=False)
    write_dataframe(ranked, ranked_summary_path, index=False)
    write_json(best_row, best_row_path)
    write_json(
        {
            "task": task_name,
            "model": model_key,
            "experiment_name": experiment_name,
            "window_lengths": window_lengths,
            "selection_metric": selection_metric,
            "force": bool(force),
            "best_window_length": best_window_length,
            "base_output_dir": str(base_config["paths"]["output_dir"]),
        },
        metadata_path,
    )

    print(
        f"[auto-window] selected window={best_window_length} "
        f"using {selection_metric} for {task_name}/{model_key}"
    )

    return AutoWindowSelection(
        best_window_length=best_window_length,
        best_row=best_row,
        search_root=search_root,
        summary_path=summary_path,
        ranked_summary_path=ranked_summary_path,
        best_row_path=best_row_path,
        metadata_path=metadata_path,
    )


def _safe_name(value: str) -> str:
    normalized = re.sub(r"[^A-Za-z0-9._-]+", "_", value.strip())
    return normalized.strip("._-") or "run"
