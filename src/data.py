"""
data.py — Carga y preparación del dataset.

Contiene funciones para:
  1. Cargar el CSV de datos históricos.
  2. Separar las variables objetivo (y), referencia (z) y features (X).

Uso desde el notebook:
    from src.data import cargar_datos, separar_variables
"""

import pandas as pd


# ---------------------------------------------------------------------------
# 1. Carga de datos
# ---------------------------------------------------------------------------
def cargar_datos(ruta_csv: str) -> pd.DataFrame:
    """
    Lee el CSV y devuelve un DataFrame.

    Parámetros
    ----------
    ruta_csv : str
        Ruta absoluta o relativa al archivo CSV.

    Retorna
    -------
    pd.DataFrame
        DataFrame con los datos cargados.
    """
    df = pd.read_csv(ruta_csv)
    print(f"Dataset cargado: {df.shape[0]:,} filas × {df.shape[1]} columnas")
    return df


# ---------------------------------------------------------------------------
# 2. Separación de variables
# ---------------------------------------------------------------------------
def separar_variables(df: pd.DataFrame):
    """
    Separa el DataFrame en:
    - y : variable objetivo (Cupos_Vendidos)
    - z : variable de referencia (Ticket_Integral_Especialidad)
    - X : features explicativas (sin Ticket_Integral_Especialidad ni Cupos_Vendidos)

    Retorna
    -------
    tuple(pd.DataFrame, pd.Series, pd.Series)
        X, y, z
    """
    if 'Cupos_Vendidos' not in df.columns and 'Cupos_libres' in df.columns:
        # Compatibilidad con datasets antiguos
        df = df.rename(columns={'Cupos_libres': 'Cupos_Vendidos'})

    y = df['Cupos_Vendidos']
    z = df['Ticket_Integral_Especialidad']
    X = df.drop(['Ticket_Integral_Especialidad', 'Cupos_Vendidos', 'Sobrecupos'], axis=1)

    # Clasificación de features
    numerical   = ['Oferta_programada', 'Bloqueo', 'Oferta_disponible',
                   'Citas_asignadas']
    categorical = ['Especialidad', 'Semana_Iso', 'Mes', 'Dia_Sem_Iso', 'Jornada_Horaria']

    print(f"Features numéricas:   {numerical}")
    print(f"Features categóricas: {categorical}")
    print(f"Dimensión de X: {X.shape}")

    return X, y, z, numerical, categorical
