import unittest
import contextlib
import io
import json
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from eval_pipeline.cli import build_parser, main


class CliTests(unittest.TestCase):
    def test_parser_accepts_run_command(self):
        args = build_parser().parse_args(["run", "config.toml"])

        self.assertEqual(args.command, "run")
        self.assertEqual(str(args.config), "config.toml")

    def test_parser_rejects_model_checkpoint_override(self):
        with self.assertRaises(SystemExit):
            build_parser().parse_args(["run", "config.toml", "--model-checkpoint", "model.pt"])

    def test_run_executes_requested_stages_in_order(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            components = root / "components.py"
            config = root / "config.toml"
            components.write_text(
                '''
from __future__ import annotations

from typing import Any

from eval_pipeline.components.data.base import DataModule
from eval_pipeline.components.models.base import ModelFactory
from eval_pipeline.components.trainers.base import Trainer
from eval_pipeline.components.validators.base import Validator
from eval_pipeline.components.testers.base import Tester
from eval_pipeline.context import TrainingContext, ValidationContext, TestContext
from eval_pipeline.registry import register_component


@register_component("stage_data", category="data")
class StageData(DataModule[dict[str, int]]):
    def setup(self) -> dict[str, int]:
        return {"value": 1}


@register_component("stage_model", category="model")
class StageModel(ModelFactory[dict[str, str]]):
    def build(self) -> dict[str, str]:
        return {"value": "fresh"}


@register_component("stage_trainer", category="training")
class StageTrainer(Trainer[dict[str, int], dict[str, str], dict[str, object]]):
    def train(self, *, data: dict[str, int], model: dict[str, str], losses: dict[str, Any], metrics: dict[str, Any], context: TrainingContext) -> dict[str, object]:
        model["value"] = "trained"
        context.state["model"] = model
        context.state["order"] = ["training"]
        return {"model": model["value"], "order": list(context.state["order"])}


@register_component("stage_validator", category="validation")
class StageValidator(Validator[dict[str, int], dict[str, str], dict[str, object]]):
    def validate(self, *, data: dict[str, int], model: dict[str, str], losses: dict[str, Any], metrics: dict[str, Any], context: ValidationContext) -> dict[str, object]:
        context.state["order"].append("validation")
        return {"model": model["value"], "order": list(context.state["order"])}


@register_component("stage_tester", category="test")
class StageTester(Tester[dict[str, int], dict[str, str], dict[str, object]]):
    def test(self, *, data: dict[str, int], model: dict[str, str], losses: dict[str, Any], metrics: dict[str, Any], context: TestContext) -> dict[str, object]:
        context.state["order"].append("test")
        return {"model": model["value"], "order": list(context.state["order"])}
''',
                encoding="utf-8",
            )
            config.write_text(
                f'''
[experiment]
name = "stage_run"
output_dir = "{(root / "runs").as_posix()}"

[registry]
paths = ["{components.as_posix()}"]

[data]
name = "stage_data"

[model]
name = "stage_model"

[training]
name = "stage_trainer"

[validation]
name = "stage_validator"

[test]
name = "stage_tester"
''',
                encoding="utf-8",
            )

            output = io.StringIO()
            with contextlib.redirect_stdout(output):
                exit_code = main(["run", str(config), "--stages", "training", "validation", "test"])

        self.assertEqual(exit_code, 0)
        result = json.loads(output.getvalue())
        self.assertEqual(result["training"], {"model": "trained", "order": ["training"]})
        self.assertEqual(result["validation"], {"model": "trained", "order": ["training", "validation"]})
        self.assertEqual(result["test"], {"model": "trained", "order": ["training", "validation", "test"]})

    def test_run_ignores_unselected_stage_config(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            components = root / "components.py"
            config = root / "config.toml"
            components.write_text(
                '''
from __future__ import annotations

from typing import Any

from eval_pipeline.components.data.base import DataModule
from eval_pipeline.components.models.base import ModelFactory
from eval_pipeline.components.testers.base import Tester
from eval_pipeline.context import TestContext
from eval_pipeline.registry import register_component


@register_component("test_only_data", category="data")
class TestOnlyData(DataModule[dict[str, int]]):
    def setup(self) -> dict[str, int]:
        return {"value": 1}


@register_component("test_only_model", category="model")
class TestOnlyModel(ModelFactory[dict[str, str]]):
    def build(self) -> dict[str, str]:
        return {"value": "loaded"}


@register_component("test_only_tester", category="test")
class TestOnlyTester(Tester[dict[str, int], dict[str, str], dict[str, object]]):
    def test(self, *, data: dict[str, int], model: dict[str, str], losses: dict[str, Any], metrics: dict[str, Any], context: TestContext) -> dict[str, object]:
        return {"model": model["value"]}
''',
                encoding="utf-8",
            )
            config.write_text(
                f'''
[experiment]
name = "test_only"
output_dir = "{(root / "runs").as_posix()}"

[registry]
paths = ["{components.as_posix()}"]

[data]
name = "test_only_data"

[model]
name = "test_only_model"

[training]
name = "missing_training_component"

[test]
name = "test_only_tester"
''',
                encoding="utf-8",
            )

            output = io.StringIO()
            with contextlib.redirect_stdout(output):
                exit_code = main(["run", str(config), "--stages", "test"])

        self.assertEqual(exit_code, 0)
        self.assertEqual(json.loads(output.getvalue()), {"test": {"model": "loaded"}})

    def test_validate_check_imports_uses_all_components(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            components = root / "components.py"
            config = root / "config.toml"
            components.write_text(
                '''
from __future__ import annotations

from eval_pipeline.components.data.base import DataModule
from eval_pipeline.components.models.base import ModelFactory
from eval_pipeline.registry import register_component


@register_component("validate_data", category="data")
class ValidateData(DataModule[dict[str, int]]):
    def setup(self) -> dict[str, int]:
        return {"value": 1}


@register_component("validate_model", category="model")
class ValidateModel(ModelFactory[dict[str, str]]):
    def build(self) -> dict[str, str]:
        return {"value": "loaded"}
''',
                encoding="utf-8",
            )
            config.write_text(
                f'''
[experiment]
name = "validate_imports"
output_dir = "{(root / "runs").as_posix()}"

[registry]
paths = ["{components.as_posix()}"]

[data]
name = "validate_data"

[model]
name = "validate_model"
''',
                encoding="utf-8",
            )

            output = io.StringIO()
            with contextlib.redirect_stdout(output):
                exit_code = main(["validate", str(config), "--check-imports"])

        self.assertEqual(exit_code, 0)
        self.assertEqual(json.loads(output.getvalue())["name"], "validate_imports")


if __name__ == "__main__":
    unittest.main()
