import io
import json
import logging
import unittest
from contextlib import redirect_stderr

from ariadne.logging import configure_logging


class LoggingTest(unittest.TestCase):
    def test_json_logging_emits_structured_record(self) -> None:
        stream = io.StringIO()
        with redirect_stderr(stream):
            configure_logging("INFO", json_output=True)
            logging.getLogger("ariadne.test").info("sensor ready")
        payload = json.loads(stream.getvalue())
        self.assertEqual(payload["level"], "INFO")
        self.assertEqual(payload["logger"], "ariadne.test")
        self.assertEqual(payload["message"], "sensor ready")
        self.assertIn("timestamp", payload)


if __name__ == "__main__":
    unittest.main()
