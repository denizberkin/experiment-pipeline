import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

try:
    import torch
    import torch.nn as nn
except ImportError:
    torch = None
    nn = None

from eval_pipeline.components.models.base import ModelFactory
from eval_pipeline.interfaces.model import _torch_load_device


if torch is not None:

    class CheckpointFactory(ModelFactory[nn.Module]):
        def build(self):
            return nn.Linear(1, 1)


@unittest.skipIf(torch is None, "torch is not installed")
class ModelFactoryCheckpointTests(unittest.TestCase):
    def test_load_without_checkpoint_returns_same_model(self):
        factory = CheckpointFactory()
        model = nn.Linear(1, 1)

        self.assertIs(factory.load(model), model)

    def test_load_without_checkpoint_moves_to_configured_device(self):
        factory = CheckpointFactory(device="cpu")
        model = nn.Linear(1, 1)

        loaded = factory.load(model)

        self.assertIs(loaded, model)
        self.assertEqual(str(next(model.parameters()).device), "cpu")

    def test_save_writes_checkpoint_when_parent_exists(self):
        factory = CheckpointFactory()
        model = nn.Linear(1, 1)

        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "model.pt"

            saved = factory.save(model, path)

            self.assertEqual(saved, path)
            self.assertTrue(path.exists())
            self.assertIn("weight", torch.load(path, map_location="cpu"))

    def test_save_raises_when_parent_is_missing(self):
        factory = CheckpointFactory()
        model = nn.Linear(1, 1)

        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "missing" / "model.pt"

            with self.assertRaises(FileNotFoundError):
                factory.save(model, path)

    def test_load_applies_checkpoint_and_moves_to_configured_device(self):
        source = nn.Linear(1, 1)
        with torch.no_grad():
            source.weight.fill_(2.0)
            source.bias.fill_(3.0)

        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "model.pt"
            torch.save(source.state_dict(), path)
            target = nn.Linear(1, 1)
            factory = CheckpointFactory(checkpoint=str(path), device="cpu")

            loaded = factory.load(target)

        self.assertIs(loaded, target)
        self.assertEqual(float(target.weight.item()), 2.0)
        self.assertEqual(float(target.bias.item()), 3.0)
        self.assertEqual(str(next(target.parameters()).device), "cpu")


class TorchLoadDeviceTests(unittest.TestCase):
    def test_torch_load_device_converts_config_values(self):
        self.assertIsNone(_torch_load_device(None))
        self.assertEqual(_torch_load_device("cpu"), "cpu")
        self.assertEqual(_torch_load_device("cuda:1"), "cuda:1")
        self.assertEqual(_torch_load_device([0, 1]), "cuda:0")
        self.assertEqual(_torch_load_device(["cuda:2", "cuda:3"]), "cuda:2")


if __name__ == "__main__":
    unittest.main()
