"""
tests/test_self_learner.py
──────────────────────────
Unit tests for the EOD automated self-learning engine.
"""

import unittest
from unittest.mock import patch, MagicMock

from utils.nifty_self_learner import run_daily_self_learning


class TestSelfLearner(unittest.TestCase):

    @patch("utils.nifty_self_learner.yf.Ticker")
    @patch("utils.nifty_self_learner.train_and_predict")
    @patch("utils.nifty_self_learner.send_telegram_alert")
    def test_run_daily_self_learning(self, mock_tg, mock_train, mock_yf):
        # Mock price history
        mock_hist = MagicMock()
        mock_hist.empty = False
        mock_hist.__len__.return_value = 2
        
        row1 = {"Open": 24400.0, "High": 24550.0, "Low": 24380.0, "Close": 24450.0}
        row2 = {"Open": 24460.0, "High": 24600.0, "Low": 24450.0, "Close": 24580.0}
        
        import pandas as pd
        df = pd.DataFrame([row1, row2])
        mock_yf.return_value.history.return_value = df

        mock_train.return_value = {
            "status": "success",
            "direction": "UP",
            "conviction_pct": 66.5,
            "regime": "TRENDING",
            "required_threshold": 60.0,
            "latest_close": 24580.0,
        }

        result = run_daily_self_learning("^NSEI")
        self.assertEqual(result["status"], "success")
        self.assertEqual(result["ticker"], "NIFTY 50")
        self.assertIn("Correct", result["direction_accuracy"])
        mock_tg.assert_called_once()


if __name__ == "__main__":
    unittest.main()
