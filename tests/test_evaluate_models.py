"""
Tests for src/evaluate_models.py
Covers calculate_csi, calculate_brier_score, evaluate_model, print_metrics_report
"""

import sys
import unittest
from pathlib import Path
from io import StringIO

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.evaluate_models import (
    calculate_csi,
    calculate_brier_score,
    evaluate_model,
    print_metrics_report,
)


class TestCalculateCSI(unittest.TestCase):
    """Tests for the Critical Success Index calculation."""

    def test_perfect_predictions(self):
        """CSI should be 1.0 when predictions match perfectly."""
        y_true = np.array([1, 1, 0, 0, 1])
        y_pred = np.array([1.0, 1.0, 0.0, 0.0, 1.0])
        self.assertAlmostEqual(calculate_csi(y_true, y_pred), 1.0)

    def test_all_wrong(self):
        """CSI should be 0.0 when all positive predictions are wrong."""
        y_true = np.array([0, 0, 0, 0])
        y_pred = np.array([1.0, 1.0, 1.0, 1.0])
        self.assertAlmostEqual(calculate_csi(y_true, y_pred), 0.0)

    def test_no_positives_at_all(self):
        """When there are no positive labels at all, confusion_matrix is 1x1
        and ravel() does not produce 4 values. The current implementation
        raises ValueError in this edge case."""
        y_true = np.array([0, 0, 0])
        y_pred = np.array([0.0, 0.0, 0.0])
        with self.assertRaises(ValueError):
            calculate_csi(y_true, y_pred)

    def test_partial_predictions(self):
        """CSI between 0 and 1 for partial correctness."""
        y_true = np.array([1, 1, 0, 0])
        y_pred = np.array([0.8, 0.3, 0.0, 0.0])
        csi = calculate_csi(y_true, y_pred)
        self.assertGreater(csi, 0.0)
        self.assertLess(csi, 1.0)

    def test_custom_threshold(self):
        """CSI changes with different thresholds."""
        y_true = np.array([1, 1, 0])
        y_pred = np.array([0.6, 0.4, 0.3])
        csi_05 = calculate_csi(y_true, y_pred, threshold=0.5)
        csi_03 = calculate_csi(y_true, y_pred, threshold=0.3)
        # Lower threshold => more positives predicted
        self.assertNotEqual(csi_05, csi_03)


class TestCalculateBrierScore(unittest.TestCase):
    """Tests for the Brier Score calculation."""

    def test_perfect_predictions(self):
        """Brier score is 0 for perfect probabilistic predictions."""
        y_true = np.array([1, 0, 1, 0])
        y_pred = np.array([1.0, 0.0, 1.0, 0.0])
        self.assertAlmostEqual(calculate_brier_score(y_true, y_pred), 0.0)

    def test_worst_predictions(self):
        """Brier score is 1.0 for completely inverted predictions."""
        y_true = np.array([1, 0, 1, 0])
        y_pred = np.array([0.0, 1.0, 0.0, 1.0])
        self.assertAlmostEqual(calculate_brier_score(y_true, y_pred), 1.0)

    def test_uniform_predictions(self):
        """Brier score is 0.25 for uniform 0.5 predictions."""
        y_true = np.array([1, 0, 1, 0])
        y_pred = np.array([0.5, 0.5, 0.5, 0.5])
        self.assertAlmostEqual(calculate_brier_score(y_true, y_pred), 0.25)

    def test_score_range(self):
        """Brier score is always between 0 and 1."""
        rng = np.random.RandomState(42)
        y_true = rng.randint(0, 2, 100)
        y_pred = rng.uniform(0, 1, 100)
        score = calculate_brier_score(y_true, y_pred)
        self.assertGreaterEqual(score, 0.0)
        self.assertLessEqual(score, 1.0)


class TestEvaluateModel(unittest.TestCase):
    """Tests for the evaluate_model function."""

    def _make_data(self, n=200):
        rng = np.random.RandomState(42)
        y_true = rng.randint(0, 2, n)
        y_pred = np.clip(y_true + rng.normal(0, 0.3, n), 0, 1)
        return y_true, y_pred

    def test_returns_all_required_metrics(self):
        """evaluate_model should return all expected metric keys."""
        y_true, y_pred = self._make_data()
        metrics = evaluate_model(y_true, y_pred, "TestModel")
        expected_keys = [
            "f1_score", "auc_roc", "auc_pr", "brier_score", "csi",
            "true_positives", "true_negatives", "false_positives",
            "false_negatives", "accuracy", "precision", "recall", "specificity",
        ]
        for key in expected_keys:
            self.assertIn(key, metrics, f"Missing metric: {key}")

    def test_metrics_in_valid_range(self):
        """All rate metrics should be between 0 and 1."""
        y_true, y_pred = self._make_data()
        metrics = evaluate_model(y_true, y_pred, "TestModel")
        for key in ("f1_score", "auc_roc", "auc_pr", "brier_score",
                     "csi", "accuracy", "precision", "recall", "specificity"):
            self.assertGreaterEqual(metrics[key], 0.0, f"{key} < 0")
            self.assertLessEqual(metrics[key], 1.0, f"{key} > 1")

    def test_confusion_matrix_values_sum(self):
        """TP + TN + FP + FN should equal total samples."""
        n = 150
        y_true, y_pred = self._make_data(n)
        metrics = evaluate_model(y_true, y_pred)
        total = (
            metrics["true_positives"]
            + metrics["true_negatives"]
            + metrics["false_positives"]
            + metrics["false_negatives"]
        )
        self.assertEqual(total, n)

    def test_perfect_model(self):
        """Perfect predictions yield F1 = 1.0 and Brier = 0."""
        y_true = np.array([1, 0, 1, 0, 1, 1, 0, 0])
        y_pred = np.array([1.0, 0.0, 1.0, 0.0, 1.0, 1.0, 0.0, 0.0])
        metrics = evaluate_model(y_true, y_pred)
        self.assertAlmostEqual(metrics["f1_score"], 1.0)
        self.assertAlmostEqual(metrics["brier_score"], 0.0)
        self.assertAlmostEqual(metrics["accuracy"], 1.0)


class TestPrintMetricsReport(unittest.TestCase):
    """Tests for print_metrics_report (smoke test)."""

    def test_prints_without_error(self):
        """print_metrics_report should run without raising."""
        metrics = {
            "f1_score": 0.85,
            "auc_roc": 0.90,
            "auc_pr": 0.88,
            "brier_score": 0.12,
            "csi": 0.75,
            "accuracy": 0.88,
            "precision": 0.87,
            "recall": 0.83,
            "specificity": 0.91,
            "true_positives": 40,
            "true_negatives": 48,
            "false_positives": 6,
            "false_negatives": 8,
        }
        # Should not raise
        print_metrics_report(metrics, "TestModel")


if __name__ == "__main__":
    unittest.main()
