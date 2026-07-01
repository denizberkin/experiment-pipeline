import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from eval_pipeline.core.config import ConfigError, parse_experiment_config


class ConfigDeviceTests(unittest.TestCase):
    def test_experiment_device_propagates_to_device_aware_components(self):
        config = parse_experiment_config(
            {
                "experiment": {"name": "device_test", "device": "cuda:0"},
                "data": {"name": "data", "params": {"device": "data-local"}},
                "model": {"name": "model"},
                "training": {"name": "trainer"},
                "validation": {"name": "validator", "params": {"device": "cpu"}},
                "test": {"name": "tester"},
                "prediction": {"name": "predictor"},
                "losses": [{"name": "loss"}],
                "metrics": [{"name": "metric", "stages": ["validation"]}],
                "tracking": {"name": "local", "params": {"device": "tracker-local"}},
            },
            config_path=Path("/tmp/config.toml"),
        )

        self.assertEqual(config.device, "cuda:0")
        self.assertEqual(config.model.params["device"], "cuda:0")
        self.assertEqual(config.training.params["device"], "cuda:0")
        self.assertEqual(config.validation.params["device"], "cpu")
        self.assertEqual(config.test.params["device"], "cuda:0")
        self.assertEqual(config.prediction.params["device"], "cuda:0")
        self.assertEqual(config.losses[0].params["device"], "cuda:0")
        self.assertEqual(config.metrics[0].params["device"], "cuda:0")
        self.assertEqual(config.data.params["device"], "data-local")
        self.assertEqual(config.tracking.params["device"], "tracker-local")
        self.assertEqual(config.to_dict()["device"], "cuda:0")

    def test_experiment_device_accepts_cuda_id_list(self):
        config = parse_experiment_config(
            {
                "experiment": {"name": "device_test", "device": [0, 1]},
                "data": {"name": "data"},
                "model": {"name": "model"},
            },
            config_path=Path("/tmp/config.toml"),
        )

        self.assertEqual(config.device, [0, 1])
        self.assertEqual(config.model.params["device"], [0, 1])

    def test_experiment_device_rejects_empty_list(self):
        with self.assertRaisesRegex(ConfigError, "experiment.device"):
            parse_experiment_config(
                {
                    "experiment": {"name": "device_test", "device": []},
                    "data": {"name": "data"},
                    "model": {"name": "model"},
                },
                config_path=Path("/tmp/config.toml"),
            )


if __name__ == "__main__":
    unittest.main()
