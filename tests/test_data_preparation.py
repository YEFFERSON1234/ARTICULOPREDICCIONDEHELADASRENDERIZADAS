"""
Tests básicos para preparación de datos
Valida integridad de datos, manejo de faltantes, y feature engineering
"""

import sys
import unittest
from pathlib import Path
import pandas as pd
import numpy as np

# Agregar ruta al path
sys.path.insert(0, str(Path(__file__).parent.parent))

import config


class TestDataIntegrity(unittest.TestCase):
    """Tests para integridad de datos"""
    
    @classmethod
    def setUpClass(cls):
        """Se ejecuta una sola vez antes de todos los tests"""
        cls.senamhi_path = config.SENAMHI_CSV
        if Path(cls.senamhi_path).exists():
            cls.senamhi = pd.read_csv(cls.senamhi_path)
        else:
            cls.senamhi = None
    
    def test_senamhi_file_exists(self):
        """Verifica que el archivo SENAMHI existe"""
        self.assertTrue(
            Path(self.senamhi_path).exists(),
            f"Archivo SENAMHI no encontrado: {self.senamhi_path}"
        )
    
    def test_senamhi_not_empty(self):
        """Verifica que SENAMHI tiene datos"""
        self.assertIsNotNone(self.senamhi)
        self.assertGreater(len(self.senamhi), 0, "Dataset SENAMHI está vacío")
    
    def test_required_columns_exist(self):
        """Verifica que existan columnas requeridas"""
        if self.senamhi is None:
            self.skipTest("No se pudo cargar SENAMHI")
        
        required_cols = ['fecha', 'estacion', 'lat', 'lon', 'tmax', 'tmin', 'precip']
        for col in required_cols:
            self.assertIn(
                col,
                self.senamhi.columns,
                f"Columna faltante: {col}"
            )
    
    def test_no_all_nulls(self):
        """Verifica que no haya columnas completamente vacías"""
        if self.senamhi is None:
            self.skipTest("No se pudo cargar SENAMHI")
        
        null_fractions = self.senamhi.isnull().sum() / len(self.senamhi)
        for col, fraction in null_fractions.items():
            self.assertLess(
                fraction,
                1.0,
                f"Columna completamente vacía: {col}"
            )
    
    def test_temperature_ranges(self):
        """Verifica que temperaturas estén en rangos razonables"""
        if self.senamhi is None:
            self.skipTest("No se pudo cargar SENAMHI")
        
        # Temperaturas típicas en Puno: -20 a 20°C
        if 'tmin' in self.senamhi.columns:
            tmin = self.senamhi['tmin'].dropna()
            self.assertTrue(
                (tmin >= -30).all() and (tmin <= 25).all(),
                f"Valores anómalos en tmin: {tmin.min()} a {tmin.max()}"
            )
        
        if 'tmax' in self.senamhi.columns:
            tmax = self.senamhi['tmax'].dropna()
            self.assertTrue(
                (tmax >= -20).all() and (tmax <= 35).all(),
                f"Valores anómalos en tmax: {tmax.min()} a {tmax.max()}"
            )
    
    def test_coordinates_in_puno_bounds(self):
        """Verifica que coordenadas estén dentro de Puno"""
        if self.senamhi is None:
            self.skipTest("No se pudo cargar SENAMHI")
        
        lat = self.senamhi['lat'].dropna()
        lon = self.senamhi['lon'].dropna()
        
        self.assertTrue(
            (lat >= config.PUNO_BOUNDS['lat_min']).all() and
            (lat <= config.PUNO_BOUNDS['lat_max']).all(),
            f"Latitudes fuera de Puno: {lat.min()} a {lat.max()}"
        )
        
        self.assertTrue(
            (lon >= config.PUNO_BOUNDS['lon_min']).all() and
            (lon <= config.PUNO_BOUNDS['lon_max']).all(),
            f"Longitudes fuera de Puno: {lon.min()} a {lon.max()}"
        )
    
    def test_date_consistency(self):
        """Verifica que las fechas sean válidas y consistentes"""
        if self.senamhi is None:
            self.skipTest("No se pudo cargar SENAMHI")
        
        try:
            fechas = pd.to_datetime(self.senamhi['fecha'])
            self.assertEqual(len(fechas.dropna()), len(self.senamhi), "Fechas inválidas detectadas")
            
            # Verificar que las fechas estén ordenadas
            fechas_sorted = fechas.sort_values()
            self.assertGreater(
                (fechas_sorted.iloc[-1] - fechas_sorted.iloc[0]).days,
                365,
                "Rango de fechas muy corto"
            )
        except Exception as e:
            self.fail(f"Error procesando fechas: {e}")


class TestFeatureEngineering(unittest.TestCase):
    """Tests para feature engineering"""
    
    def create_sample_df(self):
        """Crea un dataframe de prueba"""
        dates = pd.date_range('2020-01-01', periods=100)
        df = pd.DataFrame({
            'fecha': dates,
            'estacion': 'test_station',
            'lat': -15.5,
            'lon': -70.1,
            'tmin': np.random.normal(-5, 5, 100),
            'tmax': np.random.normal(10, 5, 100),
            'precip': np.random.exponential(1, 100)
        })
        return df
    
    def test_lag_features_creation(self):
        """Verifica que se creen correctamente features con lags"""
        df = self.create_sample_df()
        
        # Crear lags manualmente
        for lag in [1, 2, 3]:
            df[f'tmin_lag_{lag}'] = df.groupby('estacion')['tmin'].shift(lag)
        
        # Verificar que se crearon
        self.assertIn('tmin_lag_1', df.columns)
        self.assertIn('tmin_lag_2', df.columns)
        self.assertIn('tmin_lag_3', df.columns)
        
        # Verificar que hay nulos al inicio
        self.assertTrue(df['tmin_lag_1'].isnull().any())
    
    def test_frost_binary_creation(self):
        """Verifica creación correcta de variable binaria de helada"""
        df = self.create_sample_df()
        
        df['helada'] = (df['tmin'] <= 0).astype(int)
        
        self.assertIn('helada', df.columns)
        self.assertTrue(set(df['helada'].unique()).issubset({0, 1}))
        
        # Verificar que haya casos positivos y negativos
        self.assertGreater(df['helada'].sum(), 0)
        self.assertLess(df['helada'].sum(), len(df))
    
    def test_temporal_features(self):
        """Verifica extracción de features temporales"""
        df = self.create_sample_df()
        
        df['fecha'] = pd.to_datetime(df['fecha'])
        df['day_of_year'] = df['fecha'].dt.dayofyear
        df['month'] = df['fecha'].dt.month
        df['year'] = df['fecha'].dt.year
        
        self.assertIn('day_of_year', df.columns)
        self.assertIn('month', df.columns)
        self.assertIn('year', df.columns)
        
        # Verificar rangos
        self.assertTrue((df['day_of_year'] >= 1).all() and (df['day_of_year'] <= 366).all())
        self.assertTrue((df['month'] >= 1).all() and (df['month'] <= 12).all())


class TestMissingDataHandling(unittest.TestCase):
    """Tests para manejo de datos faltantes"""
    
    def test_missing_data_strategy(self):
        """Verifica estrategia de imputación"""
        df = pd.DataFrame({
            'valor': [1, 2, np.nan, 4, 5, np.nan, 7],
            'grupo': ['A', 'A', 'A', 'B', 'B', 'B', 'B']
        })
        
        # Estrategia 1: Media global
        media = df['valor'].mean()
        df['valor_imputado_global'] = df['valor'].fillna(media)
        self.assertFalse(df['valor_imputado_global'].isnull().any())
        
        # Estrategia 2: Media por grupo
        df['valor_imputado_grupo'] = df.groupby('grupo')['valor'].transform(
            lambda x: x.fillna(x.mean())
        )
        self.assertFalse(df['valor_imputado_grupo'].isnull().any())
    
    def test_nan_percentage_tracking(self):
        """Verifica que se pueda rastrear porcentaje de nulos"""
        df = pd.DataFrame({
            'A': [1, 2, np.nan, np.nan, 5],
            'B': [1, np.nan, np.nan, np.nan, 5],
            'C': [1, 2, 3, 4, 5]
        })
        
        nan_percentages = (df.isnull().sum() / len(df) * 100)
        
        self.assertAlmostEqual(nan_percentages['A'], 40.0)
        self.assertAlmostEqual(nan_percentages['B'], 60.0)
        self.assertAlmostEqual(nan_percentages['C'], 0.0)


def run_tests_verbose():
    """Ejecuta tests con salida detallada"""
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # Cargar todos los tests
    suite.addTests(loader.loadTestsFromTestCase(TestDataIntegrity))
    suite.addTests(loader.loadTestsFromTestCase(TestFeatureEngineering))
    suite.addTests(loader.loadTestsFromTestCase(TestMissingDataHandling))
    
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    return result.wasSuccessful()


if __name__ == "__main__":
    success = run_tests_verbose()
    sys.exit(0 if success else 1)
