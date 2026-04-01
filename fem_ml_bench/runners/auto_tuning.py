from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
import re
from typing import Any, Callable

import pandas as pd

from fem_ml_bench.project import PROJECT_ROOT
from src.utils.config import dump_config, merge_overrides
from src.utils.io import ensure_dir, write_dataframe, write_json


CandidateRunner = Callable[[dict[str, Any]], dict[str, Any]]
CandidateRowLoader = Callable[[Path, str], dict[str, Any]]

AUTO_HPARAM_ROOT = PROJECT_ROOT / "outputs" / "auto_hparam_search"


@dataclass(frozen=True)
class TuningProfile:
    name: str
    overrides: dict[str, Any]


@dataclass(frozen=True)
class AutoTuningSelection:
    best_profile: TuningProfile
    best_row: dict[str, Any]
    search_root: Path
    summary_path: Path
    ranked_summary_path: Path
    best_row_path: Path
    metadata_path: Path


def parse_tuning_profiles(raw: str | None, available_profiles: tuple[TuningProfile, ...]) -> tuple[TuningProfile, ...]:
    if not available_profiles:
        return ()

    profile_map = {profile.name: profile for profile in available_profiles}
    if raw is None or not raw.strip():
        return available_profiles

    names = [part.strip() for part in raw.split(",") if part.strip()]
    if not names:
        raise ValueError("At least one tuning profile is required.")

    unknown = [name for name in names if name not in profile_map]
    if unknown:
        raise ValueError(
            f"Unknown tuning profile(s): {', '.join(unknown)}. "
            f"Available profiles: {', '.join(profile_map)}."
        )

    deduped_names = list(dict.fromkeys(names))
    return tuple(profile_map[name] for name in deduped_names)


def run_auto_tuning_search(
    *,
    task_name: str,
    model_key: str,
    base_config: dict[str, Any],
    profiles: tuple[TuningProfile, ...],
    selection_metric: str,
    sort_columns: list[str],
    ascending: list[bool],
    run_candidate: CandidateRunner,
    load_candidate_row: CandidateRowLoader,
    force: bool = False,
    context_label: str | None = None,
) -> AutoTuningSelection:
    if not profiles:
        raise ValueError("Automatic hyperparameter search requires at least one tuning profile.")

    experiment_name = str(base_config.get("experiment", {}).get("name", f"{task_name}_{model_key}"))
    search_root = AUTO_HPARAM_ROOT / task_name / model_key / _safe_name(experiment_name)
    if context_label:
        search_root = search_root / _safe_name(context_label)
    search_root = ensure_dir(search_root)

    generated_config_dir = ensure_dir(search_root / "generated_configs")
    candidate_root = ensure_dir(search_root / "candidates")

    rows: list[dict[str, Any]] = []
    for profile in profiles:
        candidate_config = merge_overrides(deepcopy(base_config), profile.overrides)
        profile_token = _safe_name(profile.name)
        candidate_output_dir = candidate_root / profile_token
        candidate_config.setdefault("experiment", {})
        candidate_config["experiment"]["name"] = f"{experiment_name}_auto_tune_{profile_token}"
        candidate_config.setdefault("paths", {})
        candidate_config["paths"]["output_dir"] = str(candidate_output_dir.relative_to(PROJECT_ROOT))

        config_path = generated_config_dir / f"{_safe_name(model_key)}_{profile_token}.yaml"
        dump_config(candidate_config, config_path)

        metrics_path = candidate_output_dir / "metrics.json"
        if force or not metrics_path.exists():
            print(f"[auto-tune] evaluating profile={profile.name}")
            row = run_candidate(candidate_config)
        else:
            print(f"[auto-tune] reusing profile={profile.name} from {metrics_path.relative_to(PROJECT_ROOT)}")
            row = load_candidate_row(candidate_output_dir, profile.name)

        row["tuning_profile"] = profile.name
        row["output_dir"] = str(candidate_output_dir.relative_to(PROJECT_ROOT))
        row["config_path"] = str(config_path.relative_to(PROJECT_ROOT))
        rows.append(row)

    summary = pd.DataFrame(rows).reset_index(drop=True)
    ranked = summary.sort_values(sort_columns, ascending=ascending).reset_index(drop=True)
    best_row = ranked.iloc[0].to_dict()
    best_profile_name = str(best_row["tuning_profile"])
    best_profile = next(profile for profile in profiles if profile.name == best_profile_name)

    summary_path = search_root / "summary.csv"
    ranked_summary_path = search_root / "ranked_summary.csv"
    best_row_path = search_root / "best_profile.json"
    metadata_path = search_root / "run_metadata.json"

    write_dataframe(summary, summary_path, index=False)
    write_dataframe(ranked, ranked_summary_path, index=False)
    write_json(best_row, best_row_path)
    write_json(
        {
            "task": task_name,
            "model": model_key,
            "experiment_name": experiment_name,
            "context_label": context_label,
            "profiles": [profile.name for profile in profiles],
            "selection_metric": selection_metric,
            "force": bool(force),
            "best_profile": best_profile.name,
            "base_output_dir": str(base_config["paths"]["output_dir"]),
        },
        metadata_path,
    )

    print(
        f"[auto-tune] selected profile={best_profile.name} "
        f"using {selection_metric} for {task_name}/{model_key}"
    )

    return AutoTuningSelection(
        best_profile=best_profile,
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
