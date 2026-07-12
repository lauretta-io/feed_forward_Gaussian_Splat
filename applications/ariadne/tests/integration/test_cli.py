import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SOURCE_ROOT = PROJECT_ROOT / "src"


class CliIntegrationTest(unittest.TestCase):
    def run_cli(self, *arguments: str, cwd: Path | None = None) -> subprocess.CompletedProcess[str]:
        environment = os.environ.copy()
        environment["PYTHONPATH"] = str(SOURCE_ROOT)
        return subprocess.run(
            [sys.executable, "-m", "ariadne", *arguments],
            cwd=cwd or PROJECT_ROOT,
            env=environment,
            check=False,
            capture_output=True,
            text=True,
        )

    def test_help_has_no_cuda_or_download_side_effects(self) -> None:
        result = self.run_cli("--help")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("validate-config", result.stdout)
        self.assertNotIn("download", result.stderr.lower())
        self.assertNotIn("cuda", result.stderr.lower())

    def test_validate_config(self) -> None:
        result = self.run_cli("validate-config", "--config", "configs/wingman/default.yaml")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("valid: wingman:wingman_01", result.stdout)

    def test_runtime_and_simulation_cpu_smoke(self) -> None:
        cases = (
            ("wingman", "run", "--config", "configs/wingman/default.yaml"),
            ("intelligence", "run", "--config", "configs/intelligence/default.yaml"),
            ("simulate", "--scenario", "configs/simulation/two_node.yaml"),
            ("benchmark", "--suite", "smoke"),
            (
                "evaluate",
                "--dataset",
                "simulation",
                "--output",
                "simulation.json",
                "--wandb-mode",
                "disabled",
            ),
        )
        with tempfile.TemporaryDirectory() as directory:
            for arguments in cases:
                with self.subTest(arguments=arguments):
                    config_arguments = list(arguments)
                    for index, value in enumerate(config_arguments):
                        if value.startswith("configs/"):
                            config_arguments[index] = str(PROJECT_ROOT / value)
                    result = self.run_cli(*config_arguments, cwd=Path(directory))
                    self.assertEqual(result.returncode, 0, result.stderr)
                    payload = json.loads(result.stdout)
                    self.assertIn(payload["status"], ("ready", "passed"))
                    if arguments[0] == "benchmark":
                        self.assertEqual(payload["iterations"], 1_000)
                        self.assertGreater(payload["transform_round_trip_ns"], 0)
                        self.assertGreater(payload["peak_traced_bytes"], 0)
                    if arguments[0] == "evaluate":
                        self.assertEqual(payload["dataset"], "simulation")
                        self.assertTrue((Path(directory) / "simulation.json").is_file())

    def test_role_mismatch_fails_cleanly(self) -> None:
        result = self.run_cli("wingman", "run", "--config", "configs/intelligence/default.yaml")
        self.assertEqual(result.returncode, 2)
        self.assertIn("expected a 'wingman' config", result.stderr)


if __name__ == "__main__":
    unittest.main()
