"""Smoke tests for the Streamlit dashboard."""

from pathlib import Path
import unittest

from streamlit.testing.v1 import AppTest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DASHBOARD = PROJECT_ROOT / "app" / "dashboard.py"


class DashboardTest(unittest.TestCase):
    def test_dashboard_loads_and_classifies_baseball_text(self) -> None:
        app = AppTest.from_file(str(DASHBOARD)).run()

        self.assertEqual(app.exception, [])
        self.assertEqual(app.title[0].value, "📄 Intelligent Document Classification")
        self.assertIn("89.9%", app.metric[0].value)

        app.text_area[0].input(
            "The baseball team won after a dominant pitching performance."
        ).run()
        app.button[0].click().run()

        self.assertEqual(app.exception, [])
        self.assertIn("rec.sport.baseball", app.success[0].value)


if __name__ == "__main__":
    unittest.main()
