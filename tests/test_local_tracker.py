import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from eval_pipeline.components.trackers.local import LocalExperimentTracker


def _config_and_paths(root: Path):
    config_path = root / "config.toml"
    config_path.write_text("[experiment]\nname='test'\n", encoding="utf-8")
    config = SimpleNamespace(config_path=config_path, to_dict=lambda: {"name": "test"})
    paths = SimpleNamespace(
        experiment_dir=root / "run",
        artifacts_dir=root / "run/artifacts",
        logs_dir=root / "run/logs",
    )
    return config, paths


class LocalTrackerTests(unittest.TestCase):
    def test_empty_experiment_output_can_be_reused(self):
        # A run that died before producing anything leaves only the config copy and two
        # empty directories; retrying into it must not require editing the config.
        with tempfile.TemporaryDirectory() as directory:
            config, paths = _config_and_paths(Path(directory))
            LocalExperimentTracker().start(config, paths)
            LocalExperimentTracker().start(config, paths)

    def test_experiment_output_with_artifacts_is_refused(self):
        with tempfile.TemporaryDirectory() as directory:
            config, paths = _config_and_paths(Path(directory))
            LocalExperimentTracker().start(config, paths)
            (paths.artifacts_dir / "model.pt").write_bytes(b"")

            with self.assertRaisesRegex(FileExistsError, "Experiment output already exists"):
                LocalExperimentTracker().start(config, paths)


if __name__ == "__main__":
    unittest.main()
