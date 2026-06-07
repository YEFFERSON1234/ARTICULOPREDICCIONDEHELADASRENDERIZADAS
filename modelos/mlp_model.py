"""
mlp_model.py
Modelo MLP (Perceptron Multicapa) para prediccion de heladas
Corregido: Sin Data Leakage y sincronizado para el articulo cientifico
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report
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


def train_mlp():
    print("=" * 70)
    print("MODELO MLP (PERCEPTRON MULTICAPA) - CORREGIDO")
    print("=" * 70)

    # 1. Cargar datos
    print("\n[1/5] Cargando datos SENAMHI...")
    df = load_senamhi_data()
    df = add_temporal_features(df)

    # 2. Ingenieria de caracteristicas (SOLUCION AL DATA LEAKAGE)
    print("[2/5] Creando caracteristicas desfasadas (Lags)...")
    df = add_shifted_features(df)
    df = add_lag_features(df)
    df = df.dropna().reset_index(drop=True)

    X = df[LEAKAGE_FREE_FEATURE_COLS]
    y = df['helada']

    # 3. Division temporal SINCRONIZADA (Prueba >= 2015)
    print("[3/5] Dividiendo datos temporalmente (Sincronizado >= 2015)...")
    train_mask, test_mask = temporal_train_test_split(df, test_year=2015)

    X_train, X_test = X[train_mask], X[test_mask]
    y_train, y_test = y[train_mask], y[test_mask]

    print(f"  Entrenamiento: {len(X_train)} muestras")
    print(f"  Prueba: {len(X_test)} muestras")

    if len(X_test) == 0:
        print("[ERROR] No hay datos en el set de prueba. Revisa los anios de tu dataset.")
        return

    # 4. Escalado
    print("[4/5] Escalando caracteristicas...")
    X_train_scaled, X_test_scaled, _ = scale_features(X_train, X_test)

    # 5. Entrenamiento MLP Optimizado
    print("[5/5] Entrenando MLP...")
    mlp = MLPClassifier(
        hidden_layer_sizes=(64, 32, 16),
        activation='relu',
        solver='adam',
        alpha=0.001,
        batch_size=256,
        learning_rate='adaptive',
        learning_rate_init=0.001,
        max_iter=150,
        random_state=42,
        early_stopping=True,
        validation_fraction=0.1,
        n_iter_no_change=10,
    )

    mlp.fit(X_train_scaled, y_train)
    print("Modelo entrenado exitosamente sin filtracion de datos.")

    # 6. Evaluacion Real
    y_pred = mlp.predict(X_test_scaled)
    y_prob = mlp.predict_proba(X_test_scaled)[:, 1]

    metrics = compute_classification_metrics(y_test, y_pred, y_prob)
    print_classification_results("MLP", metrics)

    print(f"\nReporte de Clasificacion Corregido:")
    print(classification_report(y_test, y_pred))

    # 7. Guardar predicciones con estructura compatible para el ensamble
    save_predictions(
        df[test_mask],
        'data_process/predictions_mlp.csv',
        extra_cols={'prob_helada_mlp': y_prob},
        normalize_keys=True,
    )


if __name__ == '__main__':
    train_mlp()
