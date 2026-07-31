import tempfile
import unittest
from pathlib import Path

from pydantic import ValidationError

from ariadne.config import load_config

PROJECT_ROOT = Path(__file__).resolve().parents[2]


class ConfigTest(unittest.TestCase):
    def test_all_example_configs_validate(self) -> None:
        cases = {
            "configs/wingman/default.yaml": "wingman",
            "configs/intelligence/default.yaml": "intelligence",
            "configs/simulation/two_node.yaml": "simulation",
        }
        for relative_path, role in cases.items():
            with self.subTest(path=relative_path):
                config = load_config(PROJECT_ROOT / relative_path)
                self.assertEqual(config.runtime.role, role)
                if role == "simulation":
                    self.assertEqual(config.simulation.partition_duration_seconds, 60.0)
                if role == "intelligence":
                    self.assertEqual(
                        config.intelligence.correction_scheduling.target_error_m,
                        0.1,
                    )
                    self.assertEqual(
                        config.intelligence.correction_scheduling.target_orientation_error_rad,
                        0.05,
                    )
                    self.assertEqual(
                        config.intelligence.correction.max_rotation_step_rad,
                        0.1,
                    )
                    self.assertEqual(
                        config.intelligence.correction.max_total_rotation_rad,
                        0.5,
                    )

    def test_unknown_fields_are_rejected(self) -> None:
        content = """
runtime:
  role: wingman
  node_id: wingman_01
  output_dir: outputs/test
  unexpected: true
wingman:
  local_frame: local_wingman_01
"""
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "invalid.yaml"
            path.write_text(content, encoding="utf-8")
            with self.assertRaises(ValidationError):
                load_config(path)

    def test_role_specific_section_is_required(self) -> None:
        content = """
runtime:
  role: intelligence
  node_id: intelligence_01
  output_dir: outputs/test
wingman:
  local_frame: local_wingman_01
"""
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "invalid.yaml"
            path.write_text(content, encoding="utf-8")
            with self.assertRaisesRegex(ValidationError, "configuration for role"):
                load_config(path)


if __name__ == "__main__":
    unittest.main()
