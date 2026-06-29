from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from eval_pipeline.core.config import load_experiment_config
from eval_pipeline.core.registry import validate_component_imports


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="eval-pipeline")
    subparsers = parser.add_subparsers(dest="command", required=True)

    validate = subparsers.add_parser("validate", help="Validate and resolve a single experiment config.")
    validate.add_argument("config", type=Path, help="Path to a TOML or JSON experiment config.")
    validate.add_argument(
        "--check-imports",
        action="store_true",
        help="Import configured component classes to catch path/class errors early.",
    )

    run = subparsers.add_parser("run", help="Run a configured experiment test stage.")
    run.add_argument("config", type=Path, help="Path to a TOML or JSON experiment config.")

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    # validate config and imports
    if args.command == "validate":
        config = load_experiment_config(args.config)
        _add_default_import_roots(config)
        if args.check_imports:
            _check_imports(config)
        print(json.dumps(config.to_dict(), indent=2, sort_keys=True))
        return 0

    if args.command == "run":
        config = load_experiment_config(args.config)
        _add_default_import_roots(config)
        result = _run_test_stage(config)
        print(json.dumps(result, indent=2, sort_keys=True))
        return 0

    parser.error(f"Unknown command: {args.command}")
    return 2


def _check_imports(config) -> None:
    validate_component_imports(config.all_components, registry_paths=config.registry_paths)


def _add_default_import_roots(config) -> None:
    component_root = config.config_path.parent.parent
    roots = [Path.cwd(), component_root.parent, component_root]
    for root in roots:
        root_str = str(root.resolve())
        while root_str in sys.path:
            sys.path.remove(root_str)
        sys.path.insert(0, root_str)


def _run_test_stage(config) -> dict:
    if config.test is None:
        raise ValueError("Config must define [test] to use the run command.")

    validate_component_imports(config.all_components, registry_paths=config.registry_paths)

    # only import on run
    from eval_pipeline.components.data.build import build_data_module
    from eval_pipeline.components.losses.build import build_losses
    from eval_pipeline.components.metrics.build import build_metrics
    from eval_pipeline.components.models.build import build_model_factory
    from eval_pipeline.components.testers.build import build_tester
    from eval_pipeline.context import build_experiment_paths, build_experiment_tracker, make_test_context

    paths = build_experiment_paths(config)
    tracker = build_experiment_tracker(config, paths)
    try:
        context = make_test_context(config, tracker=tracker)
        data = build_data_module(config.data).setup()
        model = build_model_factory(config.model).build()
        losses = build_losses(config.losses)
        metrics = build_metrics(config.metrics)
        tester = build_tester(config.test)
        return tester.test(data=data, model=model, losses=losses, metrics=metrics, context=context)
    finally:
        tracker.close()
