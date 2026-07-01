import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from examples.cifar10.training import _should_save_checkpoint


class Cifar10TrainingTests(unittest.TestCase):
    def test_checkpoint_frequency_zero_saves_only_last_epoch(self):
        self.assertFalse(_should_save_checkpoint(epoch=1, epochs=3, checkpoint_frequency=0))
        self.assertFalse(_should_save_checkpoint(epoch=2, epochs=3, checkpoint_frequency=0))
        self.assertTrue(_should_save_checkpoint(epoch=3, epochs=3, checkpoint_frequency=0))

    def test_positive_checkpoint_frequency_saves_matching_epochs(self):
        saved_epochs = [
            epoch
            for epoch in range(1, 6)
            if _should_save_checkpoint(epoch=epoch, epochs=5, checkpoint_frequency=2)
        ]

        self.assertEqual(saved_epochs, [2, 4])


if __name__ == "__main__":
    unittest.main()
