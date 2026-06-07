"""
Tests for src/evaluate.py
Covers the evaluate() function with synthetic prediction data
"""

import sys
import unittest
import tempfile
import os
import json
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.evaluate import evaluate


def _make_predictions_csv(tmpdir, include_prob=False, n=100):
    """Create a synthetic predictions CSV and return its path."""
    rng = np.random.RandomState(42)
    tmin = rng.normal(-2, 5, n)
    tmin_pred = tmin + rng.normal(0, 1, n)  # close predictions
    data = {"tmin": tmin, "tmin_pred": tmin_pred}
    if include_prob:
        prob = 1.0 / (1.0 + np.exp(tmin_pred))  # sigmoid
        data["probabilidad_helada"] = np.clip(prob, 0, 1)
    df = pd.DataFrame(data)
    path = os.path.join(tmpdir, "predictions.csv")
    df.to_csv(path, index=False)
    return path


class TestEvaluate(unittest.TestCase):
    """Tests for the evaluate() function."""

    def test_evaluate_creates_summary_json(self):
        """evaluate() should write evaluation_summary.json."""
        with tempfile.TemporaryDirectory() as tmpdir:
            csv_path = _make_predictions_csv(tmpdir)
            out_dir = os.path.join(tmpdir, "results")
            evaluate(csv_path, out_dir=out_dir)

            summary_path = os.path.join(out_dir, "evaluation_summary.json")
            self.assertTrue(os.path.exists(summary_path))

    def test_summary_contains_rmse(self):
        """Summary JSON contains an RMSE value > 0."""
        with tempfile.TemporaryDirectory() as tmpdir:
            csv_path = _make_predictions_csv(tmpdir)
            out_dir = os.path.join(tmpdir, "results")
            evaluate(csv_path, out_dir=out_dir)

            with open(os.path.join(out_dir, "evaluation_summary.json")) as f:
                summary = json.load(f)
            self.assertIn("rmse", summary)
            self.assertGreater(summary["rmse"], 0)

    def test_summary_contains_confusion_matrix(self):
        """Summary should include a 2x2 confusion matrix."""
        with tempfile.TemporaryDirectory() as tmpdir:
            csv_path = _make_predictions_csv(tmpdir)
            out_dir = os.path.join(tmpdir, "results")
            evaluate(csv_path, out_dir=out_dir)

            with open(os.path.join(out_dir, "evaluation_summary.json")) as f:
                summary = json.load(f)
            self.assertIn("confusion_matrix", summary)
            self.assertEqual(len(summary["confusion_matrix"]), 2)
            self.assertEqual(len(summary["confusion_matrix"][0]), 2)

    def test_summary_contains_classification_report(self):
        """Summary should include classification_report dict."""
        with tempfile.TemporaryDirectory() as tmpdir:
            csv_path = _make_predictions_csv(tmpdir)
            out_dir = os.path.join(tmpdir, "results")
            evaluate(csv_path, out_dir=out_dir)

            with open(os.path.join(out_dir, "evaluation_summary.json")) as f:
                summary = json.load(f)
            self.assertIn("classification_report", summary)
            self.assertIsInstance(summary["classification_report"], dict)

    def test_roc_auc_with_probabilities(self):
        """When probabilidad_helada column exists, roc_auc is computed."""
        with tempfile.TemporaryDirectory() as tmpdir:
            csv_path = _make_predictions_csv(tmpdir, include_prob=True)
            out_dir = os.path.join(tmpdir, "results")
            evaluate(csv_path, out_dir=out_dir)

            with open(os.path.join(out_dir, "evaluation_summary.json")) as f:
                summary = json.load(f)
            self.assertIsNotNone(summary["roc_auc"])
            self.assertGreater(summary["roc_auc"], 0)

    def test_roc_auc_none_without_probabilities(self):
        """Without probabilidad_helada, roc_auc should be None."""
        with tempfile.TemporaryDirectory() as tmpdir:
            csv_path = _make_predictions_csv(tmpdir, include_prob=False)
            out_dir = os.path.join(tmpdir, "results")
            evaluate(csv_path, out_dir=out_dir)

            with open(os.path.join(out_dir, "evaluation_summary.json")) as f:
                summary = json.load(f)
            self.assertIsNone(summary["roc_auc"])

    def test_missing_columns_raises(self):
        """evaluate() raises ValueError when required columns are missing."""
        with tempfile.TemporaryDirectory() as tmpdir:
            bad_csv = os.path.join(tmpdir, "bad.csv")
            pd.DataFrame({"a": [1, 2]}).to_csv(bad_csv, index=False)
            with self.assertRaises(ValueError):
                evaluate(bad_csv)


if __name__ == "__main__":
    unittest.main()
