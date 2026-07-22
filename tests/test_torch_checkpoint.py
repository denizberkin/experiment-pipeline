import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from eval_pipeline.components.models import load_torch_checkpoint, save_torch_checkpoint


class TorchCheckpointTests(unittest.TestCase):
    @patch("eval_pipeline.components.models.torch_checkpoint._require_torch")
    def test_save_is_atomic_and_creates_parent(self, require_torch):
        fake_torch = require_torch.return_value
        fake_torch.save.side_effect = lambda state, path: Path(path).write_bytes(b"checkpoint")
        model = Mock()
        model.state_dict.return_value = {"weight": 1}

        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory) / "nested" / "model.pt"
            saved = save_torch_checkpoint(model, target)

            self.assertEqual(saved, target)
            self.assertEqual(target.read_bytes(), b"checkpoint")
            self.assertFalse(target.with_suffix(".pt.tmp").exists())

    @patch("eval_pipeline.components.models.torch_checkpoint._require_torch")
    def test_load_uses_safe_weights_only_mode(self, require_torch):
        state = {"weight": 1}
        require_torch.return_value.load.return_value = state
        model = Mock()

        loaded = load_torch_checkpoint(model, "model.pt", map_location="cpu", strict=False)

        self.assertIs(loaded, model)
        require_torch.return_value.load.assert_called_once_with(
            Path("model.pt"), map_location="cpu", weights_only=True
        )
        model.load_state_dict.assert_called_once_with(state, strict=False)


if __name__ == "__main__":
    unittest.main()
