import tempfile
import unittest
from pathlib import Path

try:
    import torch
    import torch.nn as nn
except ImportError:
    torch = None
    nn = None

if torch is None:
    load_model = None
    save_checkpoint = None
else:
    from examples.cifar10.common import load_model, save_checkpoint


@unittest.skipIf(torch is None, "torch is not installed")
class Cifar10CheckpointTests(unittest.TestCase):
    def test_load_without_checkpoint_moves_to_configured_device(self):
        model = nn.Linear(1, 1)

        loaded = load_model(model, device="cpu")

        self.assertIs(loaded, model)
        self.assertEqual(str(next(model.parameters()).device), "cpu")

    def test_save_checkpoint_creates_parent(self):
        model = nn.Linear(1, 1)
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "nested" / "model.pt"

            saved = save_checkpoint(model, path)

            self.assertEqual(saved, path)
            self.assertTrue(path.exists())

    def test_load_applies_checkpoint(self):
        source = nn.Linear(1, 1)
        with torch.no_grad():
            source.weight.fill_(2.0)
            source.bias.fill_(3.0)

        with tempfile.TemporaryDirectory() as temp_dir:
            path = save_checkpoint(source, Path(temp_dir) / "model.pt")
            target = nn.Linear(1, 1)
            loaded = load_model(target, checkpoint=path, device="cpu")

        self.assertIs(loaded, target)
        self.assertEqual(float(target.weight.item()), 2.0)
        self.assertEqual(float(target.bias.item()), 3.0)


if __name__ == "__main__":
    unittest.main()
