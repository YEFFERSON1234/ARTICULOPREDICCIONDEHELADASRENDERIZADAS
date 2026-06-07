"""
Tests for scripts/config.py and scripts/logging_config.py
Covers configuration constants and logging setup
"""

import sys
import unittest
import logging
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import scripts.config as config


class TestConfig(unittest.TestCase):
    """Tests for the project configuration module."""

    def test_project_root_exists(self):
        """PROJECT_ROOT should be a valid directory."""
        self.assertTrue(config.PROJECT_ROOT.exists())

    def test_puno_bounds_keys(self):
        """PUNO_BOUNDS should have lat_min, lat_max, lon_min, lon_max."""
        for key in ("lat_min", "lat_max", "lon_min", "lon_max"):
            self.assertIn(key, config.PUNO_BOUNDS)

    def test_puno_bounds_consistency(self):
        """lat_min < lat_max and lon_min < lon_max."""
        self.assertLess(config.PUNO_BOUNDS["lat_min"], config.PUNO_BOUNDS["lat_max"])
        self.assertLess(config.PUNO_BOUNDS["lon_min"], config.PUNO_BOUNDS["lon_max"])

    def test_puno_coords_inside_bounds(self):
        """PUNO_LAT and PUNO_LON should be within PUNO_BOUNDS."""
        self.assertGreaterEqual(config.PUNO_LAT, config.PUNO_BOUNDS["lat_min"])
        self.assertLessEqual(config.PUNO_LAT, config.PUNO_BOUNDS["lat_max"])
        self.assertGreaterEqual(config.PUNO_LON, config.PUNO_BOUNDS["lon_min"])
        self.assertLessEqual(config.PUNO_LON, config.PUNO_BOUNDS["lon_max"])

    def test_feature_cols_is_list(self):
        """FEATURE_COLS should be a non-empty list of strings."""
        self.assertIsInstance(config.FEATURE_COLS, list)
        self.assertGreater(len(config.FEATURE_COLS), 0)
        for col in config.FEATURE_COLS:
            self.assertIsInstance(col, str)

    def test_feature_cols_expected_values(self):
        """FEATURE_COLS should contain core features."""
        expected = {"lat", "lon", "day_of_year", "month"}
        actual = set(config.FEATURE_COLS)
        self.assertTrue(expected.issubset(actual))

    def test_train_params_keys(self):
        """TRAIN_PARAMS should have XGBoost hyperparameters."""
        for key in ("n_estimators", "max_depth", "learning_rate"):
            self.assertIn(key, config.TRAIN_PARAMS)

    def test_train_params_types(self):
        """Hyperparameter values should be correct types."""
        self.assertIsInstance(config.TRAIN_PARAMS["n_estimators"], int)
        self.assertIsInstance(config.TRAIN_PARAMS["max_depth"], int)
        self.assertIsInstance(config.TRAIN_PARAMS["learning_rate"], float)

    def test_train_params_reasonable_values(self):
        """Hyperparameters should be in reasonable ranges."""
        self.assertGreater(config.TRAIN_PARAMS["n_estimators"], 0)
        self.assertGreater(config.TRAIN_PARAMS["max_depth"], 0)
        self.assertGreater(config.TRAIN_PARAMS["learning_rate"], 0)
        self.assertLess(config.TRAIN_PARAMS["learning_rate"], 1)

    def test_log_level_valid(self):
        """LOG_LEVEL should be a valid Python logging level."""
        valid_levels = {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}
        self.assertIn(config.LOG_LEVEL, valid_levels)

    def test_directories_created(self):
        """MODELS_DIR, RESULTS_DIR, LOGS_DIR should exist (created at import)."""
        self.assertTrue(config.MODELS_DIR.exists())
        self.assertTrue(config.RESULTS_DIR.exists())
        self.assertTrue(config.LOGS_DIR.exists())


class TestLoggingConfig(unittest.TestCase):
    """Tests for the logging configuration module."""

    def test_setup_logging_returns_logger(self):
        """setup_logging should return a logging.Logger instance."""
        from scripts.logging_config import setup_logging

        logger = setup_logging("test_logger")
        self.assertIsInstance(logger, logging.Logger)

    def test_logger_has_handlers(self):
        """Logger should have at least one handler (or inherit from parent)."""
        from scripts.logging_config import setup_logging

        logger = setup_logging("test_logger_handlers_unique")
        # Logger either has own handlers or inherits via effective handlers
        has_own = len(logger.handlers) > 0
        has_effective = logger.parent is not None and len(logger.parent.handlers) > 0
        self.assertTrue(has_own or has_effective)

    def test_project_logger_exists(self):
        """The module-level project_logger should be available."""
        from scripts.logging_config import project_logger

        self.assertIsInstance(project_logger, logging.Logger)

    def test_setup_logging_no_duplicate_handlers(self):
        """Calling setup_logging twice with the same name should not duplicate handlers."""
        from scripts.logging_config import setup_logging

        name = "test_no_dup"
        logger1 = setup_logging(name)
        n_handlers = len(logger1.handlers)
        logger2 = setup_logging(name)
        self.assertEqual(len(logger2.handlers), n_handlers)
        self.assertIs(logger1, logger2)


if __name__ == "__main__":
    unittest.main()
