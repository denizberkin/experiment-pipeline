import sys
import unittest
from pathlib import Path
from typing import NoDefault

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from eval_pipeline.interfaces import DataModule, Loss, Metric, ModelFactory, Predictor, Tester, Trainer, Validator


class InterfaceTypingTests(unittest.TestCase):
    def test_interface_type_parameters_do_not_default_to_any(self):
        for interface in (DataModule, ModelFactory, Trainer, Validator, Tester, Predictor, Loss, Metric):
            with self.subTest(interface=interface.__name__):
                self.assertTrue(interface.__type_params__)
                self.assertTrue(all(param.__default__ is NoDefault for param in interface.__type_params__))


if __name__ == "__main__":
    unittest.main()
