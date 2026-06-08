"""
svm_model.py
Modelo SVM (Support Vector Machine) optimizado para datos masivos de estaciones SENAMHI
Corregido: Sin Data Leakage, ejecucion veloz mediante aproximacion lineal y sincronizado.
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import classification_report, confusion_matrix
from utils.data_loading import (
    configure_encoding, load_senamhi_data, add_temporal_features,
    add_lag_features, add_shifted_features, temporal_train_test_split,
    scale_features, LEAKAGE_FREE_FEATURE_COLS,
)
from utils.evaluation import (
    compute_classification_metrics, print_classification_results,
    save_predictions,
)

configure_encoding()


def train_svm_senamhi():
    print("=" * 70)
    print("SVM OPTIMIZADO CON DATOS SENAMHI")
    print("=" * 70)

    # 1. Cargar datos
    print("\n[1/5] Cargando datos SENAMHI...")
    df = load_senamhi_data()

    # 2. Ingenieria de caracteristicas (SOLUCION AL DATA LEAKAGE)
    print("[2/5] Creando caracteristicas desfasadas (Ayer)...")
    df = add_temporal_features(df)
    df = add_shifted_features(df)
    df = add_lag_features(df)
    df = df.dropna().reset_index(drop=True)

    X = df[LEAKAGE_FREE_FEATURE_COLS]
    y = df['helada']

    # 3. Division temporal sincronizada (Prueba >= 2015)
    print("[3/5] Dividiendo datos temporalmente (Corte 2015)...")
    train_mask, test_mask = temporal_train_test_split(df, test_year=2015)

    X_train, X_test = X[train_mask], X[test_mask]
    y_train, y_test = y[train_mask], y[test_mask]

    print(f"  Entrenamiento (< 2015): {len(X_train)} muestras")
    print(f"  Prueba (>= 2015): {len(X_test)} muestras")

    # 4. Escalado
    print("[4/5] Escalando caracteristicas...")
    X_train_scaled, X_test_scaled, _ = scale_features(X_train, X_test)

    # 5. Entrenamiento SVM de Alta Velocidad
    print("[5/5] Entrenando Clasificador SVM Calibrado...")
    base_svm = LinearSVC(
        C=0.5,
        class_weight='balanced',
        random_state=42,
        dual=False,
        max_iter=2000,
    )
    svm_model = CalibratedClassifierCV(estimator=base_svm, method='sigmoid', cv=3)
    svm_model.fit(X_train_scaled, y_train)
    print("  Modelo entrenado exitosamente.")

    # 6. Evaluacion Real
    y_pred = svm_model.predict(X_test_scaled)
    y_prob = svm_model.predict_proba(X_test_scaled)[:, 1]

    metrics = compute_classification_metrics(y_test, y_pred, y_prob)
    print_classification_results("SVM", metrics)

    print(f"\nMatriz de Confusion Real:")
    print(confusion_matrix(y_test, y_pred))

    print(f"\nReporte de Clasificacion Corregido:")
    print(classification_report(y_test, y_pred))

    # 7. Guardar predicciones
    save_predictions(
        df[test_mask],
        'data_process/predictions_svm.csv',
        extra_cols={'prob_helada_svm': y_prob},
        normalize_keys=True,
    )


if __name__ == '__main__':
    train_svm_senamhi()
