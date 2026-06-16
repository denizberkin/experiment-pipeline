from __future__ import annotations

import argparse
import json
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

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command == "validate":
        config = load_experiment_config(args.config)
        if args.check_imports:
            _check_imports(config)
        print(json.dumps(config.to_dict(), indent=2, sort_keys=True))
        return 0

    parser.error(f"Unknown command: {args.command}")
    return 2


def _check_imports(config) -> None:
    validate_component_imports(config.all_components, registry_paths=config.registry_paths)
