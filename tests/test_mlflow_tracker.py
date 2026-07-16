import os
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from eval_pipeline.components.trackers.mlflow import MlflowExperimentTracker, _allow_file_store_if_needed


class MlflowTrackerTests(unittest.TestCase):
    def test_file_tracking_uri_allows_file_store(self):
        previous = os.environ.pop("MLFLOW_ALLOW_FILE_STORE", None)
        try:
            _allow_file_store_if_needed("file:/tmp/mlruns")

            self.assertEqual(os.environ["MLFLOW_ALLOW_FILE_STORE"], "true")
        finally:
            if previous is not None:
                os.environ["MLFLOW_ALLOW_FILE_STORE"] = previous
            else:
                os.environ.pop("MLFLOW_ALLOW_FILE_STORE", None)

    def test_tracker_starts_and_closes(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config_path = root / "config.toml"
            config_path.write_text("[experiment]\nname='test'\n")
            config = SimpleNamespace(name="test", config_path=config_path, to_dict=lambda: {"name": "test"})
            paths = SimpleNamespace(
                experiment_dir=root / "run",
                artifacts_dir=root / "run/artifacts",
                logs_dir=root / "run/logs",
            )
            tracker = MlflowExperimentTracker(
                tracking_uri=f"file:{root / 'mlruns'}",
                experiment_name="test",
                run_name="test",
            )

            tracker.start(config, paths)
            tracker.log_loss("loss", 1.0, step=1, stage="training")
            tracker.close()

            self.assertTrue((root / "mlruns").is_dir())


if __name__ == "__main__":
    unittest.main()
