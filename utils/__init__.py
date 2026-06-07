"""
Utils package for data processing, downloading, and shared ML utilities.
"""

from utils.data_loading import (
    configure_encoding,
    load_senamhi_data,
    load_senamhi_csvs,
    add_temporal_features,
    add_lag_features,
    add_shifted_features,
    prepare_features,
    temporal_train_test_split,
    scale_features,
    get_project_paths,
    prepare_single_station,
    DEFAULT_FEATURE_COLS,
    LEAKAGE_FREE_FEATURE_COLS,
)
from utils.evaluation import (
    compute_regression_metrics,
    compute_classification_metrics,
    print_regression_results,
    print_classification_results,
    print_full_results,
    print_table_metrics,
    sigmoid_probability,
    save_predictions,
)
from utils.plotting import (
    plot_confusion_matrix,
    plot_roc_curve,
)

__all__ = [
    'configure_encoding',
    'load_senamhi_data',
    'load_senamhi_csvs',
    'add_temporal_features',
    'add_lag_features',
    'add_shifted_features',
    'prepare_features',
    'temporal_train_test_split',
    'scale_features',
    'get_project_paths',
    'prepare_single_station',
    'DEFAULT_FEATURE_COLS',
    'LEAKAGE_FREE_FEATURE_COLS',
    'compute_regression_metrics',
    'compute_classification_metrics',
    'print_regression_results',
    'print_classification_results',
    'print_full_results',
    'print_table_metrics',
    'sigmoid_probability',
    'save_predictions',
    'plot_confusion_matrix',
    'plot_roc_curve',
]
