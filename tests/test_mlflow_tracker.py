import os
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from eval_pipeline.components.trackers.mlflow import _allow_file_store_if_needed


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


if __name__ == "__main__":
    unittest.main()
