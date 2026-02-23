"""
features.py — Feature Engineering y diagnóstico de colinealidad.

Contiene funciones para:
  1. Crear ratios que reemplazan variables colineales.
  2. Calcular el VIF (Variance Inflation Factor) de un set de features.
  3. Codificación cíclica de variables temporales (Mes, Dia_Sem_Iso).
  4. Cálculo de promedio móvil para Tasa_asignación.

Uso desde el notebook:
    from src.features import crear_ratios, calcular_vif, codificar_ciclicas
"""

import numpy as np
import pandas as pd
from statsmodels.stats.outliers_influence import variance_inflation_factor


# ---------------------------------------------------------------------------
# 1. Creación de ratios
# ---------------------------------------------------------------------------
def crear_ratios(X: pd.DataFrame) -> tuple:
    """
    Crea ratios para eliminar colinealidad entre variables de oferta:
      - Tasa_bloqueo      = Bloqueo / Oferta_programada
      - Tasa_asignación   = Citas_asignadas / Oferta_disponible

    Parámetros
    ----------
    X : pd.DataFrame
        DataFrame con las features originales (debe contener las columnas
        Bloqueo, Oferta_programada, Citas_asignadas, Oferta_disponible).

    Retorna
    -------
    tuple(pd.DataFrame, list)
        X modificado con las nuevas columnas, y la lista de features numéricas limpias.
    """
    X = X.copy()  # No modificar el DataFrame original

    X['Tasa_bloqueo']   = X['Bloqueo'] / X['Oferta_programada'].replace(0, np.nan)
    X['Tasa_asignación'] = X['Citas_asignadas'] / X['Oferta_disponible'].replace(0, np.nan)

    # Rellenar NaN (divisiones por 0) con 0
    cols_ratio = ['Tasa_bloqueo', 'Tasa_asignación']
    X[cols_ratio] = X[cols_ratio].fillna(0)

    # Set limpio de features numéricas: 1 absoluta (escala) + 2 ratios
    numerical_clean = ['Oferta_programada', 'Tasa_bloqueo', 'Tasa_asignación']

    return X, numerical_clean


# ---------------------------------------------------------------------------
# 2. Codificación cíclica de variables temporales
# ---------------------------------------------------------------------------
def codificar_ciclicas(X: pd.DataFrame) -> tuple:
    """
    Transforma variables temporales cíclicas en componentes seno/coseno.
    
    Variables transformadas:
      - Mes (1-12) → Mes_sin, Mes_cos
      - Dia_Sem_Iso (1-7) → Dia_sin, Dia_cos
    
    Esto preserva la naturaleza cíclica (e.g., diciembre está cerca de enero,
    domingo está cerca de lunes) sin crear discontinuidades artificiales.
    
    Parámetros
    ----------
    X : pd.DataFrame
        DataFrame con las columnas 'Mes' y 'Dia_Sem_Iso'.
    
    Retorna
    -------
    tuple(pd.DataFrame, list)
        X modificado con las nuevas columnas cíclicas, y la lista de 
        features cíclicas generadas.
    """
    X = X.copy()
    
    # Mes: período de 12
    if 'Mes' in X.columns:
        X['Mes_sin'] = np.sin(2 * np.pi * X['Mes'] / 12)
        X['Mes_cos'] = np.cos(2 * np.pi * X['Mes'] / 12)
    
    # Día de la semana ISO: período de 7
    if 'Dia_Sem_Iso' in X.columns:
        X['Dia_sin'] = np.sin(2 * np.pi * X['Dia_Sem_Iso'] / 7)
        X['Dia_cos'] = np.cos(2 * np.pi * X['Dia_Sem_Iso'] / 7)
    
    ciclicas = ['Mes_sin', 'Mes_cos', 'Dia_sin', 'Dia_cos']
    
    return X, ciclicas


# ---------------------------------------------------------------------------
# 3. Cálculo de VIF
# ---------------------------------------------------------------------------
def calcular_vif(X: pd.DataFrame, columnas: list) -> pd.DataFrame:
    """
    Calcula el Variance Inflation Factor para un conjunto de columnas numéricas.

    Parámetros
    ----------
    X : pd.DataFrame
        DataFrame que contiene las columnas a analizar.
    columnas : list[str]
        Lista de nombres de columnas numéricas.

    Retorna
    -------
    pd.DataFrame
        DataFrame con columnas ['Feature', 'VIF'], ordenado de mayor a menor VIF.
    """
    vif = pd.DataFrame()
    vif['Feature'] = columnas
    vif['VIF'] = [
        variance_inflation_factor(X[columnas].values, i)
        for i in range(len(columnas))
    ]
    vif = vif.sort_values('VIF', ascending=False).reset_index(drop=True)
    return vif


# ---------------------------------------------------------------------------
# 4. Limpieza de outliers — Reglas de dominio
# ---------------------------------------------------------------------------
def aplicar_reglas_dominio(X: pd.DataFrame, y: pd.Series, z: pd.Series) -> tuple:
    """
    Filtra registros con valores lógicamente imposibles:
      - Tasas deben estar en [0, 1] (o se permiten sobrecupos > 1 si es intencional).
      - Oferta_programada debe ser > 0.
      - Tasa_bloqueo = 1.0 implica 0 oferta disponible → registro no informativo.

    Retorna
    -------
    tuple(pd.DataFrame, pd.Series, pd.Series, pd.DataFrame)
        X, y, z filtrados y un DataFrame resumen de lo eliminado.
    """
    n_antes = len(X)

    reglas = {
        'Oferta_programada <= 0':  X['Oferta_programada'] <= 0,
        'Tasa_bloqueo < 0':       X['Tasa_bloqueo'] < 0,
        'Tasa_bloqueo == 1 (sin oferta disponible)': X['Tasa_bloqueo'] >= 1.0,
        'Tasa_asignación < 0':     X['Tasa_asignación'] < 0,
    }

    # Registrar cuántos registros afecta cada regla
    resumen = pd.DataFrame({
        'Regla': list(reglas.keys()),
        'Filas afectadas': [mask.sum() for mask in reglas.values()],
    })

    # Máscara combinada: eliminar cualquier fila que incumpla al menos una regla
    mask_eliminar = pd.concat(reglas.values(), axis=1).any(axis=1)
    mask_conservar = ~mask_eliminar

    X_clean = X.loc[mask_conservar].reset_index(drop=True)
    y_clean = y.loc[mask_conservar].reset_index(drop=True)
    z_clean = z.loc[mask_conservar].reset_index(drop=True)

    n_despues = len(X_clean)
    resumen.loc[len(resumen)] = ['TOTAL eliminadas (únicas)', mask_eliminar.sum()]
    resumen.loc[len(resumen)] = ['Filas restantes', n_despues]

    print(f"Reglas de dominio: {n_antes:,} → {n_despues:,} filas ({mask_eliminar.sum():,} eliminadas, {mask_eliminar.sum()/n_antes*100:.1f}%)")

    return X_clean, y_clean, z_clean, resumen


# ---------------------------------------------------------------------------
# 5. Limpieza de outliers — IQR por grupo
# ---------------------------------------------------------------------------
def eliminar_outliers_iqr(
    X: pd.DataFrame,
    y: pd.Series,
    z: pd.Series,
    columnas: list,
    grupo: str = None,
    factor: float = 1.5,
) -> tuple:
    """
    Elimina outliers usando el método IQR, opcionalmente dentro de cada grupo.

    Para cada columna en `columnas`:
      - Calcula Q1, Q3 e IQR (por grupo si se indica).
      - Marca como outlier si el valor cae fuera de [Q1 - factor*IQR, Q3 + factor*IQR].
      - Elimina la fila si es outlier en **cualquier** columna.

    Parámetros
    ----------
    X : pd.DataFrame
    y, z : pd.Series
    columnas : list[str]
        Columnas numéricas sobre las que aplicar IQR.
    grupo : str, opcional
        Columna categórica para calcular IQR dentro de cada categoría.
    factor : float
        Multiplicador del IQR (default 1.5 = estándar).

    Retorna
    -------
    tuple(pd.DataFrame, pd.Series, pd.Series, pd.DataFrame)
        X, y, z filtrados y un DataFrame resumen por columna.
    """
    n_antes = len(X)
    es_outlier = pd.Series(False, index=X.index)

    resumen_filas = []

    for col in columnas:
        if grupo:
            q1 = X.groupby(grupo)[col].transform('quantile', 0.25)
            q3 = X.groupby(grupo)[col].transform('quantile', 0.75)
        else:
            q1 = X[col].quantile(0.25)
            q3 = X[col].quantile(0.75)

        iqr = q3 - q1
        lim_inf = q1 - factor * iqr
        lim_sup = q3 + factor * iqr

        outliers_col = (X[col] < lim_inf) | (X[col] > lim_sup)
        n_outliers = outliers_col.sum()
        resumen_filas.append({
            'Variable': col,
            'Outliers detectados': n_outliers,
            '% del total': f"{n_outliers / n_antes * 100:.1f}%",
        })
        es_outlier = es_outlier | outliers_col

    mask_conservar = ~es_outlier
    X_clean = X.loc[mask_conservar].reset_index(drop=True)
    y_clean = y.loc[mask_conservar].reset_index(drop=True)
    z_clean = z.loc[mask_conservar].reset_index(drop=True)

    n_despues = len(X_clean)
    resumen = pd.DataFrame(resumen_filas)

    tipo = f"IQR por {grupo}" if grupo else "IQR global"
    print(f"{tipo} (factor={factor}): {n_antes:,} → {n_despues:,} filas ({es_outlier.sum():,} eliminadas, {es_outlier.sum()/n_antes*100:.1f}%)")

    return X_clean, y_clean, z_clean, resumen


# ---------------------------------------------------------------------------
# 6. Cálculo de promedio móvil para Tasa_asignación
# ---------------------------------------------------------------------------
def calcular_promedio_movil_tasa_asignacion(
    df_historico: pd.DataFrame,
    especialidad: str,
    jornada: str,
    semana_objetivo: int,
    ventana: int = 4,
) -> float:
    """
    Calcula el promedio móvil de Tasa_asignación para una especialidad/jornada
    usando las últimas N semanas previas a la semana objetivo.
    
    Parámetros
    ----------
    df_historico : pd.DataFrame
        DataFrame histórico con columnas: Especialidad, Jornada_Horaria, 
        Semana_Iso, Oferta_disponible, Citas_asignadas.
    especialidad : str
        Nombre de la especialidad.
    jornada : str
        Jornada horaria (Turno_A, Turno_B, Turno_C).
    semana_objetivo : int
        Semana ISO para la cual se quiere predecir (1-53).
    ventana : int
        Número de semanas previas a considerar (default: 4).
    
    Retorna
    -------
    float
        Promedio móvil de Tasa_asignación. Si no hay suficiente data,
        devuelve el promedio general de la especialidad.
    """
    # Filtrar por especialidad y jornada
    mask = (
        (df_historico['Especialidad'] == especialidad) &
        (df_historico['Jornada_Horaria'] == jornada) &
        (df_historico['Semana_Iso'] < semana_objetivo)
    )
    data = df_historico[mask].copy()
    
    if len(data) == 0:
        # Fallback 1: Promedio de la especialidad (sin filtrar por jornada)
        mask_esp = (
            (df_historico['Especialidad'] == especialidad) &
            (df_historico['Semana_Iso'] < semana_objetivo)
        )
        data = df_historico[mask_esp].copy()
    
    if len(data) == 0:
        # Fallback 2: Retornar 0.8 (típico de sistemas de salud bien gestionados)
        return 0.8
    
    # Calcular Tasa_asignación
    data['Tasa_asignacion'] = (
        data['Citas_asignadas'] / data['Oferta_disponible'].replace(0, np.nan)
    ).fillna(0)
    
    # Tomar las últimas N semanas
    data_sorted = data.sort_values('Semana_Iso', ascending=False).head(ventana)
    
    promedio = data_sorted['Tasa_asignacion'].mean()
    
    return promedio if not np.isnan(promedio) else 0.8


# ---------------------------------------------------------------------------
# 7. Preprocesamiento — One-Hot Encoding + Escalado + Split
# ---------------------------------------------------------------------------
def preprocesar(
    X: pd.DataFrame,
    y: pd.Series,
    numerical: list,
    categorical: list,
    test_size: float = 0.2,
    random_state: int = 42,
) -> dict:
    """
    Pipeline completo de preprocesamiento:
      1. Codificación cíclica de Mes y Dia_Sem_Iso (seno/coseno).
      2. One-Hot Encoding de variables categóricas restantes (drop_first=True).
      3. Escalado (StandardScaler) de variables numéricas — solo fit sobre train.
      4. División train/test.

    Parámetros
    ----------
    X : pd.DataFrame
    y : pd.Series
    numerical : list[str]
    categorical : list[str]
        Lista de categóricas. Mes y Dia_Sem_Iso se codifican cíclicas,
        las demás se codifican one-hot.
    test_size : float
    random_state : int

    Retorna
    -------
    dict con claves:
        X_train, X_test, y_train, y_test, scaler, feature_names, 
        encoder_columns, ciclicas
    """
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler

    X = X.copy()

    # --- Codificación cíclica de variables temporales ---
    X, ciclicas = codificar_ciclicas(X)

    # --- One-Hot Encoding de categóricas NO temporales ---
    # Filtrar Mes y Dia_Sem_Iso de la lista de categóricas
    categorical_to_encode = [c for c in categorical if c not in ['Mes', 'Dia_Sem_Iso']]
    X_encoded = pd.get_dummies(X[categorical_to_encode], drop_first=True, dtype=int)
    encoder_columns = X_encoded.columns.tolist()

    # --- Combinar todas las features ---
    # numerical + ciclicas + encoded
    X_final = pd.concat([X[numerical], X[ciclicas], X_encoded], axis=1)
    feature_names = X_final.columns.tolist()

    print(f"Features finales: {len(feature_names)} "
          f"({len(numerical)} numéricas + {len(ciclicas)} cíclicas + {len(encoder_columns)} dummies)")

    # --- Split train/test ---
    X_train, X_test, y_train, y_test = train_test_split(
        X_final, y, test_size=test_size, random_state=random_state
    )

    # --- Escalado de numéricas (solo fit en train para evitar data leakage) ---
    # Escalar numéricas + cíclicas (las dummies quedan en {0,1})
    scaler = StandardScaler()
    cols_to_scale = numerical + ciclicas
    X_train.loc[:, cols_to_scale] = scaler.fit_transform(X_train[cols_to_scale])
    X_test.loc[:, cols_to_scale] = scaler.transform(X_test[cols_to_scale])

    print(f"Train: {X_train.shape[0]:,} filas  |  Test: {X_test.shape[0]:,} filas")

    return {
        'X_train': X_train,
        'X_test': X_test,
        'y_train': y_train,
        'y_test': y_test,
        'scaler': scaler,
        'feature_names': feature_names,
        'encoder_columns': encoder_columns,
        'ciclicas': ciclicas,
    }

