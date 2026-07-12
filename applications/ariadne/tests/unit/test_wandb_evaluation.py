import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from ariadne.datasets.base import DatasetEvaluation
from ariadne.evaluation import log_evaluation_to_wandb


class FakeSummary(dict):
    pass


class FakeArtifact:
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.files = []

    def add_file(self, path, name):
        self.files.append((path, name))


class FakeRun:
    id = "test-run"
    url = "https://wandb.invalid/test-run"

    def __init__(self):
        self.logged = []
        self.artifacts = []
        self.summary = FakeSummary()
        self.finished = False

    def log(self, payload):
        self.logged.append(payload)

    def log_artifact(self, artifact):
        self.artifacts.append(artifact)

    def finish(self):
        self.finished = True


class FakeWandb:
    Artifact = FakeArtifact

    def __init__(self):
        self.run = FakeRun()
        self.init_kwargs = None

    def init(self, **kwargs):
        self.init_kwargs = kwargs
        return self.run


class WandbEvaluationTest(unittest.TestCase):
    def test_disabled_mode_has_no_import(self) -> None:
        result = DatasetEvaluation("test", "passed", (), (), {})
        with patch("importlib.import_module") as importer:
            self.assertIsNone(
                log_evaluation_to_wandb(
                    result,
                    Path("report.json"),
                    mode="disabled",
                    project="test",
                    entity=None,
                    name=None,
                    group=None,
                    tags=[],
                )
            )
            importer.assert_not_called()

    def test_metrics_and_report_artifact_are_logged(self) -> None:
        result = DatasetEvaluation(
            "miluv", "passed", ("one", "two", "three"), ("imu", "vision"), {"frames": 12}
        )
        fake_wandb = FakeWandb()
        with tempfile.TemporaryDirectory() as directory:
            report = Path(directory) / "report.json"
            result.write_json(report)
            with patch("importlib.import_module", return_value=fake_wandb):
                url = log_evaluation_to_wandb(
                    result,
                    report,
                    mode="offline",
                    project="test",
                    entity=None,
                    name="miluv-test",
                    group="sequence",
                    tags=["cpu"],
                )
        self.assertEqual(url, fake_wandb.run.url)
        self.assertEqual(fake_wandb.run.logged[0]["evaluation/frames"], 12)
        self.assertEqual(fake_wandb.run.logged[0]["evaluation/passed"], 1)
        self.assertEqual(len(fake_wandb.run.artifacts), 1)
        self.assertTrue(fake_wandb.run.finished)


if __name__ == "__main__":
    unittest.main()
