"""
Tests for src/train.py
Covers prepare_features() and train_models() logic
"""

import sys
import unittest
import tempfile
import os
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.train import prepare_features


def _make_sample_df(n=60):
    """Helper: create a minimal DataFrame suitable for prepare_features()."""
    dates = pd.date_range("2020-01-01", periods=n, freq="D")
    rng = np.random.RandomState(42)
    return pd.DataFrame(
        {
            "fecha": dates,
            "estacion": "station_A",
            "lat": -15.5,
            "lon": -70.1,
            "tmin": rng.normal(-3, 4, n),
            "tmax": rng.normal(12, 3, n),
            "precip": rng.exponential(1, n),
        }
    )


class TestPrepareFeatures(unittest.TestCase):
    """Tests for the prepare_features function in src/train.py."""

    def test_adds_temporal_columns(self):
        """day_of_year, month, year columns are created."""
        df = prepare_features(_make_sample_df())
        for col in ("day_of_year", "month", "year"):
            self.assertIn(col, df.columns)

    def test_adds_lag_columns(self):
        """tmin_lag_1..3 columns are created."""
        df = prepare_features(_make_sample_df())
        for lag in (1, 2, 3):
            self.assertIn(f"tmin_lag_{lag}", df.columns)

    def test_no_nulls_after_prepare(self):
        """dropna() inside prepare_features removes any rows with NaNs."""
        df = prepare_features(_make_sample_df())
        self.assertFalse(df.isnull().any().any())

    def test_rows_dropped_for_lags(self):
        """At least 3 rows should be dropped (lag-3 produces 3 NaN rows per station)."""
        raw = _make_sample_df(30)
        df = prepare_features(raw)
        self.assertLess(len(df), len(raw))
        self.assertEqual(len(df), len(raw) - 3)

    def test_sorted_by_station_and_date(self):
        """Output is sorted by estacion then fecha."""
        raw = _make_sample_df(20)
        df = prepare_features(raw)
        dates = df["fecha"].values
        self.assertTrue((dates[:-1] <= dates[1:]).all())

    def test_day_of_year_range(self):
        """day_of_year values in [1, 366]."""
        df = prepare_features(_make_sample_df())
        self.assertTrue((df["day_of_year"] >= 1).all())
        self.assertTrue((df["day_of_year"] <= 366).all())

    def test_month_range(self):
        """month values in [1, 12]."""
        df = prepare_features(_make_sample_df())
        self.assertTrue((df["month"] >= 1).all())
        self.assertTrue((df["month"] <= 12).all())

    def test_multiple_stations(self):
        """Lags are computed per station independently."""
        df1 = _make_sample_df(20)
        df1["estacion"] = "A"
        df2 = _make_sample_df(20)
        df2["estacion"] = "B"
        df2["fecha"] = pd.date_range("2020-01-01", periods=20, freq="D")
        combined = pd.concat([df1, df2], ignore_index=True)
        result = prepare_features(combined)
        # Each station loses 3 rows => total 40 - 6 = 34
        self.assertEqual(len(result), 34)

    def test_original_not_modified(self):
        """prepare_features copies the input; original stays unchanged."""
        raw = _make_sample_df(10)
        original_cols = set(raw.columns)
        prepare_features(raw)
        self.assertEqual(set(raw.columns), original_cols)


class TestTrainModels(unittest.TestCase):
    """Tests for train_models() with mocked XGBoost."""

    def test_train_models_saves_files(self):
        """train_models() produces xgb_reg.pkl and xgb_clf.pkl."""
        raw = _make_sample_df(200)
        # Add a second year so train/test split works
        raw2 = _make_sample_df(200)
        raw2["fecha"] = pd.date_range("2021-01-01", periods=200, freq="D")
        combined = pd.concat([raw, raw2], ignore_index=True)

        with tempfile.TemporaryDirectory() as tmpdir:
            csv_path = os.path.join(tmpdir, "data.csv")
            combined.to_csv(csv_path, index=False)
            models_dir = os.path.join(tmpdir, "models")

            from src.train import train_models
            train_models(csv_path, models_dir=models_dir)

            self.assertTrue(os.path.exists(os.path.join(models_dir, "xgb_reg.pkl")))
            self.assertTrue(os.path.exists(os.path.join(models_dir, "xgb_clf.pkl")))


if __name__ == "__main__":
    unittest.main()
