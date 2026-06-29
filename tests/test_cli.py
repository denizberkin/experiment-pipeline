import unittest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from eval_pipeline.cli import build_parser


class CliTests(unittest.TestCase):
    def test_parser_accepts_run_command(self):
        args = build_parser().parse_args(["run", "config.toml"])

        self.assertEqual(args.command, "run")
        self.assertEqual(str(args.config), "config.toml")


if __name__ == "__main__":
    unittest.main()
