"""
Tests for src/walk_forward_cv.py
Covers walk_forward_cross_validation() with synthetic data
"""

import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.walk_forward_cv import walk_forward_cross_validation


def _make_cv_dataset(n_years=8, rows_per_year=50):
    """Create a synthetic dataset suitable for walk-forward CV."""
    rng = np.random.RandomState(42)
    rows = []
    for year in range(2015, 2015 + n_years):
        for _ in range(rows_per_year):
            tmin = rng.normal(-3, 5)
            tmax = tmin + rng.uniform(5, 15)
            rows.append(
                {
                    "year": year,
                    "lat": -15.5,
                    "lon": -70.1,
                    "day_of_year": rng.randint(1, 366),
                    "month": rng.randint(1, 13),
                    "precip": rng.exponential(1),
                    "tmax": tmax,
                    "tmin_lag_1": tmin + rng.normal(0, 1),
                    "tmin_lag_2": tmin + rng.normal(0, 2),
                    "tmin_lag_3": tmin + rng.normal(0, 2),
                    "amp_termica": tmax - tmin,
                    "helada": int(tmin <= 0),
                }
            )
    return pd.DataFrame(rows)


class TestWalkForwardCV(unittest.TestCase):
    """Tests for walk_forward_cross_validation()."""

    def test_returns_list_of_dicts(self):
        """Result should be a list of dicts with expected keys."""
        df = _make_cv_dataset()
        results = walk_forward_cross_validation(df, n_folds=3, min_train_years=3)
        self.assertIsInstance(results, list)
        self.assertGreater(len(results), 0)
        for r in results:
            self.assertIn("fold", r)
            self.assertIn("f1_score", r)
            self.assertIn("auc_roc", r)
            self.assertIn("brier_score", r)

    def test_fold_count(self):
        """Number of folds should not exceed n_folds."""
        df = _make_cv_dataset(n_years=10)
        results = walk_forward_cross_validation(df, n_folds=4, min_train_years=3)
        self.assertLessEqual(len(results), 4)

    def test_metrics_in_valid_range(self):
        """All metrics should be in [0, 1]."""
        df = _make_cv_dataset()
        results = walk_forward_cross_validation(df, n_folds=3, min_train_years=3)
        for r in results:
            self.assertGreaterEqual(r["f1_score"], 0.0)
            self.assertLessEqual(r["f1_score"], 1.0)
            self.assertGreaterEqual(r["auc_roc"], 0.0)
            self.assertLessEqual(r["auc_roc"], 1.0)
            self.assertGreaterEqual(r["brier_score"], 0.0)
            self.assertLessEqual(r["brier_score"], 1.0)

    def test_train_test_no_overlap(self):
        """Train and test years should not overlap (walk-forward)."""
        df = _make_cv_dataset(n_years=8)
        results = walk_forward_cross_validation(df, n_folds=3, min_train_years=3)
        for r in results:
            train_end = int(r["train_years"].split("-")[1])
            test_start = int(r["test_years"].split("-")[0])
            self.assertGreaterEqual(test_start, train_end)

    def test_increasing_train_window(self):
        """Each subsequent fold should have a train window ending at or after the previous one."""
        df = _make_cv_dataset(n_years=10)
        results = walk_forward_cross_validation(df, n_folds=4, min_train_years=3)
        if len(results) > 1:
            for i in range(1, len(results)):
                prev_end = int(results[i - 1]["train_years"].split("-")[1])
                curr_end = int(results[i]["train_years"].split("-")[1])
                self.assertGreaterEqual(curr_end, prev_end)

    def test_result_includes_sample_counts(self):
        """Each fold result should include train and test sample counts."""
        df = _make_cv_dataset()
        results = walk_forward_cross_validation(df, n_folds=3, min_train_years=3)
        for r in results:
            self.assertIn("train_samples", r)
            self.assertIn("test_samples", r)
            self.assertGreater(r["train_samples"], 0)
            self.assertGreater(r["test_samples"], 0)


if __name__ == "__main__":
    unittest.main()
