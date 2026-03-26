import tempfile
import unittest
from pathlib import Path

from spam_filter import SpamFilterModel, load_model, save_model


class SpamFilterTests(unittest.TestCase):
    def test_basic_prediction(self):
        model = SpamFilterModel()
        model.update("spam", "win money now")
        model.update("spam", "free prize claim")
        model.update("ham", "meeting schedule for project")
        model.update("ham", "please review attached report")

        label, confidence = model.predict_with_score("claim your free money prize")
        self.assertEqual(label, "spam")
        self.assertGreater(confidence, 0.5)

    def test_save_and_load(self):
        model = SpamFilterModel()
        model.update("spam", "cheap pills buy now")
        model.update("ham", "family dinner this weekend")

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "spam_model.json"
            save_model(model, path)
            reloaded = load_model(path)
            label, _ = reloaded.predict_with_score("buy cheap now")
            self.assertEqual(label, "spam")


if __name__ == "__main__":
    unittest.main()
