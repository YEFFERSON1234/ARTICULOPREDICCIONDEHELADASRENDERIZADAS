import os
import pandas as pd
import matplotlib.pyplot as plt


def plot_tmin_vs_pred(predictions_csv, out_png='graficos_resultados/pipeline_tmin_vs_pred.png'):
    df = pd.read_csv(predictions_csv)
    if 'tmin' not in df.columns or 'tmin_pred' not in df.columns:
        raise ValueError('El CSV de predicciones debe contener `tmin` y `tmin_pred`.')
    plt.figure(figsize=(8, 6))
    plt.scatter(df['tmin'], df['tmin_pred'], s=10, alpha=0.6)
    minv = min(df['tmin'].min(), df['tmin_pred'].min())
    maxv = max(df['tmin'].max(), df['tmin_pred'].max())
    plt.plot([minv, maxv], [minv, maxv], 'r--')
    plt.xlabel('tmin real (°C)')
    plt.ylabel('tmin predicho (°C)')
    plt.title('tmin real vs predicho')
    os.makedirs(os.path.dirname(out_png), exist_ok=True)
    plt.savefig(out_png, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"Gráfico guardado en {out_png}")


if __name__ == '__main__':
    pred_csv = os.path.join('data_process', 'predictions_pipeline.csv')
    if not os.path.exists(pred_csv):
        pred_csv = os.path.join('limpiezadedatos', 'predictions.csv')
    if not os.path.exists(pred_csv):
        raise FileNotFoundError('No se encontró archivo de predicciones para visualizar.')
    plot_tmin_vs_pred(pred_csv)
