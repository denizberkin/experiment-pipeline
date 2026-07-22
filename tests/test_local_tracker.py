import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from eval_pipeline.components.trackers.local import LocalExperimentTracker


class LocalTrackerTests(unittest.TestCase):
    def test_existing_experiment_output_is_refused(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config_path = root / "config.toml"
            config_path.write_text("[experiment]\nname='test'\n", encoding="utf-8")
            config = SimpleNamespace(config_path=config_path, to_dict=lambda: {"name": "test"})
            paths = SimpleNamespace(
                experiment_dir=root / "run",
                artifacts_dir=root / "run/artifacts",
                logs_dir=root / "run/logs",
            )

            LocalExperimentTracker().start(config, paths)

            with self.assertRaisesRegex(FileExistsError, "Experiment output already exists"):
                LocalExperimentTracker().start(config, paths)


if __name__ == "__main__":
    unittest.main()
