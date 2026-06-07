"""
data_loading.py
Shared utilities for data loading, feature engineering, and train/test splitting.

Consolidates duplicated patterns from:
  - modelos/xgboost_model.py, random_forest.py, mlp_model.py, cnn_1d_model.py,
    lstm_pytorch.py, SVM_senamhi.py, holt_winters_model.py, prophet_model.py,
    sarima_ann_hybrid.py, sarimax_model.py
  - src/train.py, src/predict.py
  - modelos/{RandomForest,SVM,XGBoost}/entrenamiento_*.py
"""

import os
import sys
import glob
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler


def configure_encoding():
    """Fix stdout encoding on Windows. Safe no-op on other platforms."""
    if sys.platform == 'win32' and not hasattr(sys.stdout, 'buffer'):
        import io
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')


def load_senamhi_data(csv_path='data_process/datos_heladas_puno_REAL.csv',
                      sort_by=None, add_frost_col=True):
    """Load the consolidated SENAMHI CSV and perform common preprocessing.

    Parameters
    ----------
    csv_path : str
        Path to the CSV file.
    sort_by : list[str] or None
        Columns to sort by. Defaults to ``['estacion', 'fecha']``.
    add_frost_col : bool
        If ``True``, add a ``frost`` / ``helada`` column (``tmin <= 0``).

    Returns
    -------
    pd.DataFrame
        Preprocessed DataFrame with ``fecha`` parsed as datetime.
    """
    df = pd.read_csv(csv_path)
    df['fecha'] = pd.to_datetime(df['fecha'])

    if sort_by is None:
        sort_by = ['estacion', 'fecha']
    df = df.sort_values(sort_by).reset_index(drop=True)

    if add_frost_col:
        df['frost'] = (df['tmin'] <= 0).astype(int)
        if 'helada' not in df.columns:
            df['helada'] = df['frost']

    return df


def load_senamhi_csvs(folder_path):
    """Load and concatenate multiple SENAMHI station CSVs from a folder.

    Used by ``modelos/{RandomForest,SVM,XGBoost}/entrenamiento_*.py``.

    Parameters
    ----------
    folder_path : str
        Directory containing per-station ``*.csv`` files.

    Returns
    -------
    pd.DataFrame
        Concatenated DataFrame with an ``estacion`` column derived from filenames.

    Raises
    ------
    FileNotFoundError
        If no CSV files are found in *folder_path*.
    """
    archivos = glob.glob(os.path.join(folder_path, "*.csv"))
    if not archivos:
        raise FileNotFoundError(
            f"No CSV files found in {folder_path}"
        )

    frames = []
    for f in archivos:
        df_temp = pd.read_csv(f)
        df_temp['estacion'] = os.path.basename(f).replace('.csv', '')
        frames.append(df_temp)

    df = pd.concat(frames, ignore_index=True)
    df = df.dropna(subset=['temp_min', 'precipitacion'])
    df['helada'] = (df['temp_min'] <= 0).astype(int)
    return df


def add_temporal_features(df):
    """Add ``day_of_year``, ``month``, and ``year`` columns from ``fecha``.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain a ``fecha`` datetime column.

    Returns
    -------
    pd.DataFrame
        DataFrame with the three new columns added (modified in-place).
    """
    df['day_of_year'] = df['fecha'].dt.dayofyear
    df['month'] = df['fecha'].dt.month
    df['year'] = df['fecha'].dt.year
    return df


def add_lag_features(df, column='tmin', group_col='estacion', lags=None):
    """Create lagged features grouped by station.

    Parameters
    ----------
    df : pd.DataFrame
        Input data.
    column : str
        Column to lag.
    group_col : str
        Column to group by before shifting.
    lags : list[int] or None
        Lag steps; defaults to ``[1, 2, 3]``.

    Returns
    -------
    pd.DataFrame
        DataFrame with ``{column}_lag_{n}`` columns added (modified in-place).
    """
    if lags is None:
        lags = [1, 2, 3]
    for lag in lags:
        df[f'{column}_lag_{lag}'] = df.groupby(group_col)[column].shift(lag)
    return df


def add_shifted_features(df, columns=None, group_col='estacion', suffix='_ayer'):
    """Shift multiple columns by 1 step (yesterday's values) to avoid data leakage.

    Parameters
    ----------
    df : pd.DataFrame
        Input data.
    columns : list[str] or None
        Columns to shift. Defaults to ``['tmax', 'precip', 'amp_termica']``.
    group_col : str
        Column to group by before shifting.
    suffix : str
        Suffix appended to new column names.

    Returns
    -------
    pd.DataFrame
        DataFrame with shifted columns added (modified in-place).
    """
    if columns is None:
        columns = ['tmax', 'precip', 'amp_termica']
    for col in columns:
        if col in df.columns:
            df[f'{col}{suffix}'] = df.groupby(group_col)[col].shift(1)
    return df


def prepare_features(df, lags=None):
    """Full feature pipeline used by XGBoost, RF, and similar models.

    Adds temporal features and tmin lags, then drops NaN rows.
    Replaces the identical ``prepare_features`` functions in ``src/train.py``
    and ``src/predict.py``.

    Parameters
    ----------
    df : pd.DataFrame
        Raw SENAMHI data (must contain ``fecha``, ``estacion``, ``tmin``).
    lags : list[int] or None
        Lag steps for tmin; defaults to ``[1, 2, 3]``.

    Returns
    -------
    pd.DataFrame
        Feature-enriched DataFrame with NaN rows removed.
    """
    df = df.copy()
    df['fecha'] = pd.to_datetime(df['fecha'])
    df = df.sort_values(['estacion', 'fecha']).reset_index(drop=True)
    df = add_temporal_features(df)
    df = add_lag_features(df, column='tmin', lags=lags)
    df = df.dropna()
    return df


DEFAULT_FEATURE_COLS = [
    'lat', 'lon', 'day_of_year', 'month', 'precip', 'tmax',
    'tmin_lag_1', 'tmin_lag_2', 'tmin_lag_3',
]

LEAKAGE_FREE_FEATURE_COLS = [
    'lat', 'lon', 'day_of_year', 'month',
    'precip_ayer', 'tmax_ayer', 'amp_termica_ayer',
    'tmin_lag_1', 'tmin_lag_2', 'tmin_lag_3',
]


def temporal_train_test_split(df, test_year=None, year_col='year'):
    """Split data temporally: train on years before *test_year*, test on the rest.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain *year_col*.
    test_year : int or None
        First year of the test set. If ``None``, uses the last available year.
    year_col : str
        Name of the year column.

    Returns
    -------
    train_mask : pd.Series[bool]
    test_mask  : pd.Series[bool]
    """
    if test_year is None:
        test_year = df[year_col].max()
    train_mask = df[year_col] < test_year
    test_mask = df[year_col] >= test_year
    return train_mask, test_mask


def scale_features(X_train, X_test):
    """Fit a StandardScaler on training data and transform both splits.

    Parameters
    ----------
    X_train : array-like
    X_test : array-like

    Returns
    -------
    X_train_scaled : np.ndarray
    X_test_scaled  : np.ndarray
    scaler         : StandardScaler
    """
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    return X_train_scaled, X_test_scaled, scaler


def get_project_paths(script_file):
    """Derive standard project directory paths from a script's ``__file__``.

    Returns a dict with keys:
    ``script_dir``, ``project_root``, ``senamhi_dir``, ``era5_dir``,
    ``predictions_dir``, ``plots_dir``.

    Parameters
    ----------
    script_file : str
        Typically ``__file__`` from the calling module.
    """
    script_dir = os.path.dirname(os.path.abspath(script_file))
    project_root = os.path.abspath(os.path.join(script_dir, '..', '..'))

    paths = {
        'script_dir': script_dir,
        'project_root': project_root,
        'senamhi_dir': os.path.join(project_root, 'data', 'datos_senamhi_puno_csv'),
        'era5_dir': os.path.join(project_root, 'data', 'datos_era5_puno_csv'),
        'predictions_dir': os.path.join(project_root, 'Predicciones'),
        'plots_dir': os.path.join(script_dir, 'graficos_resultados'),
    }

    os.makedirs(paths['predictions_dir'], exist_ok=True)
    os.makedirs(paths['plots_dir'], exist_ok=True)
    return paths


def prepare_single_station(df, cols_interes=None):
    """Select the most frequent station and prepare a daily time series.

    Used by Holt-Winters, Prophet, SARIMAX, and SARIMA-ANN models.

    Parameters
    ----------
    df : pd.DataFrame
        Full SENAMHI dataset.
    cols_interes : list[str] or None
        Columns to keep. Defaults to
        ``['fecha', 'tmin', 'precip', 'tmax', 'amp_termica', 'lat', 'lon']``.

    Returns
    -------
    station_name : str
    df_est : pd.DataFrame
        Daily-resampled, forward-filled DataFrame indexed by date.
    """
    station_name = df['estacion'].value_counts().index[0]

    df_est = df[df['estacion'] == station_name].copy()

    if cols_interes is None:
        cols_interes = ['fecha', 'tmin', 'precip', 'tmax', 'amp_termica', 'lat', 'lon']
    df_est = df_est[cols_interes].set_index('fecha')

    df_est = df_est.resample('D').mean()
    df_est = df_est.ffill()
    df_est.index.freq = 'D'

    return station_name, df_est
