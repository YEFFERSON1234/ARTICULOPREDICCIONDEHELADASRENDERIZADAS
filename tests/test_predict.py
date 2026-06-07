"""
Tests for src/predict.py
Covers prepare_features() and predict() logic
"""

import sys
import unittest
import tempfile
import os
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.predict import prepare_features


def _make_sample_df(n=60):
    """Helper: create a minimal DataFrame for prepare_features()."""
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


class TestPredictPrepareFeatures(unittest.TestCase):
    """Tests for prepare_features in src/predict.py (mirrors train.py)."""

    def test_adds_temporal_columns(self):
        df = prepare_features(_make_sample_df())
        for col in ("day_of_year", "month", "year"):
            self.assertIn(col, df.columns)

    def test_adds_lag_columns(self):
        df = prepare_features(_make_sample_df())
        for lag in (1, 2, 3):
            self.assertIn(f"tmin_lag_{lag}", df.columns)

    def test_no_nulls(self):
        df = prepare_features(_make_sample_df())
        self.assertFalse(df.isnull().any().any())

    def test_original_not_modified(self):
        raw = _make_sample_df(10)
        cols_before = set(raw.columns)
        prepare_features(raw)
        self.assertEqual(set(raw.columns), cols_before)


class TestPredict(unittest.TestCase):
    """Tests for predict() with mocked models."""

    @patch("src.predict.joblib.load")
    def test_predict_writes_csv(self, mock_load):
        """predict() loads models, generates predictions, writes CSV."""
        # Build multi-year data so test_mask selects rows
        raw = _make_sample_df(100)
        raw2 = _make_sample_df(100)
        raw2["fecha"] = pd.date_range("2021-01-01", periods=100, freq="D")
        combined = pd.concat([raw, raw2], ignore_index=True)

        with tempfile.TemporaryDirectory() as tmpdir:
            csv_path = os.path.join(tmpdir, "data.csv")
            combined.to_csv(csv_path, index=False)

            models_dir = os.path.join(tmpdir, "models")
            os.makedirs(models_dir)
            # Create dummy files so the directory exists
            open(os.path.join(models_dir, "xgb_reg.pkl"), "w").close()
            open(os.path.join(models_dir, "xgb_clf.pkl"), "w").close()

            # Mock models returned by joblib.load
            mock_reg = MagicMock()
            mock_clf = MagicMock()

            def _predict_side_effect(X):
                return np.zeros(len(X))

            def _proba_side_effect(X):
                return np.column_stack(
                    [np.ones(len(X)) * 0.5, np.ones(len(X)) * 0.5]
                )

            mock_reg.predict.side_effect = _predict_side_effect
            mock_clf.predict_proba.side_effect = _proba_side_effect
            mock_load.side_effect = [mock_reg, mock_clf]

            out_path = os.path.join(tmpdir, "predictions.csv")
            from src.predict import predict
            predict(csv_path, models_dir=models_dir, out_path=out_path)

            self.assertTrue(os.path.exists(out_path))
            result = pd.read_csv(out_path)
            self.assertIn("tmin_pred", result.columns)
            self.assertIn("probabilidad_helada", result.columns)

    def test_predict_missing_models_dir_raises(self):
        """predict() raises FileNotFoundError when models_dir does not exist."""
        from src.predict import predict

        with tempfile.TemporaryDirectory() as tmpdir:
            csv_path = os.path.join(tmpdir, "data.csv")
            _make_sample_df(10).to_csv(csv_path, index=False)
            with self.assertRaises(FileNotFoundError):
                predict(csv_path, models_dir="/nonexistent/models")


if __name__ == "__main__":
    unittest.main()
