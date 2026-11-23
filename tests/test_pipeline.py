"""
FinSight AI - System Integration Tests
Validates database connection, technical indicator calculations, and ML forecasting.
"""

import unittest
import pandas as pd
import numpy as np
from database import test_connection, create_tables
from src.analytics import compute_technical_indicators
from src.ml_predictor import train_and_forecast_stock


class TestFinSightPipeline(unittest.TestCase):

    def test_01_database_connectivity(self):
        """Verifies MySQL connection pool."""
        create_tables()
        version = test_connection()
        self.assertIsNotNone(version)

    def test_02_technical_indicators(self):
        """Verifies SMA and RSI calculations."""
        # Generate 65 days of synthetic price data
        dates = pd.date_range("2024-01-01", periods=65)
        prices = np.linspace(150, 180, 65) + np.random.normal(0, 1, 65)
        df = pd.DataFrame({"trade_date": dates, "close_price": prices})

        res = compute_technical_indicators(df)
        self.assertIn("SMA_20", res.columns)
        self.assertIn("SMA_50", res.columns)
        self.assertIn("RSI_14", res.columns)
        self.assertTrue(pd.notna(res["SMA_20"].iloc[-1]))
        self.assertTrue(0 <= res["RSI_14"].iloc[-1] <= 100)

    def test_03_ml_forecaster(self):
        """Verifies Random Forest trend forecasting on historical data."""
        dates = pd.date_range("2024-01-01", periods=75)
        prices = np.linspace(100, 130, 75)
        df = pd.DataFrame({"trade_date": dates, "close_price": prices})

        forecast = train_and_forecast_stock(df)
        self.assertIn("trend_direction", forecast)
        self.assertIn("predicted_price", forecast)
        self.assertIn("confidence_score", forecast)
        self.assertTrue(forecast["predicted_price"] > 0)


if __name__ == "__main__":
    unittest.main(verbosity=2)