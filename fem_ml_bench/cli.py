from __future__ import annotations

import argparse
import importlib
from collections import defaultdict

from .registry import RunnerSpec, list_runner_specs, resolve_runner


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="fem_ml_bench",
        description="Single entrypoint for FEM ML Bench experiments.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    list_parser = subparsers.add_parser("list", help="List supported task/model combinations.")
    list_parser.add_argument("--verbose", action="store_true", help="Show descriptions and default configs.")

    run_parser = subparsers.add_parser("run", help="Run a registered experiment.")
    run_parser.add_argument("--task", required=True, help="Task name, for example classification.")
    run_parser.add_argument("--model", required=True, help="Model key, for example rnn.")
    run_parser.add_argument(
        "--config",
        help="Optional config path. If omitted, the registry default config is used when available.",
    )
    run_parser.add_argument(
        "--runner-help",
        action="store_true",
        help="Show the underlying runner help instead of executing training.",
    )
    return parser


def _group_specs() -> dict[str, list[RunnerSpec]]:
    grouped: dict[str, list[RunnerSpec]] = defaultdict(list)
    for spec in list_runner_specs():
        grouped[spec.task].append(spec)
    return dict(sorted(grouped.items()))


def _print_specs(verbose: bool) -> None:
    grouped = _group_specs()
    for task, specs in grouped.items():
        print(f"{task}:")
        for spec in sorted(specs, key=lambda item: item.model):
            line = f"  - {spec.model}"
            if verbose:
                line += f" | {spec.description}"
                if spec.default_config is not None:
                    line += f" | default_config={spec.default_config}"
            print(line)


def _load_runner(module_name: str):
    module = importlib.import_module(module_name)
    main = getattr(module, "main", None)
    if not callable(main):
        raise TypeError(f"Runner module '{module_name}' does not expose a callable main(argv=None).")
    return main


def _runner_args(args: argparse.Namespace, extra_args: list[str], spec: RunnerSpec) -> list[str]:
    runner_args = [arg for arg in extra_args if arg != "--"]
    has_config_override = any(arg == "--config" or arg.startswith("--config=") for arg in runner_args)

    if args.config is not None:
        runner_args = ["--config", args.config, *runner_args]
    elif spec.default_config is not None and not has_config_override:
        runner_args = ["--config", str(spec.default_config), *runner_args]

    if args.runner_help:
        runner_args.append("--help")

    return runner_args


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args, extra_args = parser.parse_known_args(argv)

    if args.command == "list":
        _print_specs(verbose=bool(args.verbose))
        return 0

    spec = resolve_runner(args.task, args.model)
    runner_args = _runner_args(args, extra_args, spec)
    runner_main = _load_runner(spec.module)

    print(f"[INFO] Running {spec.task}/{spec.model}")
    if spec.default_config is not None and args.config is None:
        print(f"[INFO] Using default config: {spec.default_config}")

    result = runner_main(runner_args)
    return int(result or 0)
