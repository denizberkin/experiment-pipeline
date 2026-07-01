import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

try:
    import torch
except ImportError:
    torch = None

from examples.cifar10.common import get_device


@unittest.skipIf(torch is None, "torch is not installed")
class Cifar10CommonTests(unittest.TestCase):
    def test_get_device_accepts_cuda_id_list(self):
        self.assertEqual(get_device([0, 1]), torch.device("cuda:0"))

    def test_get_device_accepts_cuda_string_list(self):
        self.assertEqual(get_device(["cuda:1"]), torch.device("cuda:1"))


if __name__ == "__main__":
    unittest.main()
