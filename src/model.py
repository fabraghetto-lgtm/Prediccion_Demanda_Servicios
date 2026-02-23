"""
model.py — Entrenamiento, tuning y evaluación de modelos predictivos.

Contiene:
  1. Entrenamiento de múltiples modelos (Regresión Lineal, Random Forest, Gradient Boosting).
  2. Tuning de hiperparámetros con GridSearchCV.
  3. Ensamble (VotingRegressor) de los mejores modelos.
  4. Evaluación con métricas de regresión.

Uso desde el notebook:
    from src.model import entrenar_modelos, tuning_modelo, crear_ensamble, evaluar_modelo
"""

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import (
    RandomForestRegressor,
    VotingRegressor,
)
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    mean_absolute_percentage_error,
)


# ---------------------------------------------------------------------------
# 1. Entrenamiento de múltiples modelos base
# ---------------------------------------------------------------------------
def entrenar_modelos(X_train, y_train, X_test, y_test, random_state=42) -> dict:
    """
    Entrena 3 modelos base y devuelve resultados comparativos.

    Modelos:
      - Regresión Lineal (baseline)
      - Random Forest (ensemble bagging)
      - Gradient Boosting (ensemble boosting)

    Retorna
    -------
    dict con claves: 'modelos', 'resultados', 'predicciones'
    """
    modelos = {
        'Regresión Lineal': LinearRegression(),
        'Random Forest': RandomForestRegressor(
            n_estimators=200, max_depth=15, min_samples_leaf=5,
            random_state=random_state, n_jobs=-1
        ),
        'XGBoost (GPU)': xgb.XGBRegressor(
            n_estimators=200, max_depth=5, learning_rate=0.1,
            min_child_weight=5, device='cuda',
            tree_method='hist', random_state=random_state,
            verbosity=0,
        ),
    }

    resultados = []
    predicciones = {}
    modelos_entrenados = {}

    for nombre, modelo in modelos.items():
        # Entrenar
        modelo.fit(X_train, y_train)
        modelos_entrenados[nombre] = modelo

        # Predecir
        y_pred_train = modelo.predict(X_train)
        y_pred_test = modelo.predict(X_test)
        predicciones[nombre] = y_pred_test

        # Métricas
        resultados.append({
            'Modelo': nombre,
            'R² Train': r2_score(y_train, y_pred_train),
            'R² Test': r2_score(y_test, y_pred_test),
            'MAE': mean_absolute_error(y_test, y_pred_test),
            'RMSE': np.sqrt(mean_squared_error(y_test, y_pred_test)),
            'MAPE': mean_absolute_percentage_error(y_test, y_pred_test),
        })

        print(f"  ✓ {nombre:<22} R²={resultados[-1]['R² Test']:.4f}  MAE={resultados[-1]['MAE']:.2f}")

    df_resultados = pd.DataFrame(resultados).sort_values('R² Test', ascending=False)

    return {
        'modelos': modelos_entrenados,
        'resultados': df_resultados,
        'predicciones': predicciones,
    }


# ---------------------------------------------------------------------------
# 2. Tuning de hiperparámetros
# ---------------------------------------------------------------------------
def tuning_modelo(
    modelo_nombre: str,
    X_train, y_train,
    param_grid: dict = None,
    cv: int = 5,
    random_state: int = 42,
) -> tuple:
    """
    Aplica GridSearchCV para encontrar los mejores hiperparámetros.

    Parámetros
    ----------
    modelo_nombre : str
        'Random Forest' o 'Gradient Boosting'
    param_grid : dict, opcional
        Grilla de parámetros. Si es None, usa una grilla predeterminada.
    cv : int
        Número de folds para cross-validation.

    Retorna
    -------
    tuple(mejor_modelo, resultados_cv)
    """
    if modelo_nombre == 'Random Forest':
        base = RandomForestRegressor(random_state=random_state, n_jobs=-1)
        if param_grid is None:
            param_grid = {
                'n_estimators': [100, 200, 300],
                'max_depth': [10, 15, 20, None],
                'min_samples_leaf': [3, 5, 10],
            }
    elif modelo_nombre in ('XGBoost', 'XGBoost (GPU)'):
        base = xgb.XGBRegressor(
            device='cuda', tree_method='hist',
            random_state=random_state, verbosity=0,
        )
        if param_grid is None:
            param_grid = {
                'n_estimators': [100, 200, 300],
                'max_depth': [3, 5, 7],
                'learning_rate': [0.05, 0.1, 0.2],
                'min_child_weight': [3, 5, 10],
            }
    else:
        raise ValueError(f"Modelo '{modelo_nombre}' no soportado para tuning.")

    print(f"Tuning {modelo_nombre} con GridSearchCV ({cv} folds)...")
    print(f"Combinaciones a evaluar: {np.prod([len(v) for v in param_grid.values()])}")

    grid = GridSearchCV(
        base, param_grid, cv=cv,
        scoring='r2', n_jobs=-1, verbose=0,
        return_train_score=True,
    )
    grid.fit(X_train, y_train)

    resultados_cv = pd.DataFrame(grid.cv_results_)[
        ['params', 'mean_train_score', 'mean_test_score', 'std_test_score', 'rank_test_score']
    ].sort_values('rank_test_score').head(10)

    print(f"\nMejores parámetros: {grid.best_params_}")
    print(f"Mejor R² (CV): {grid.best_score_:.4f}")

    return grid.best_estimator_, resultados_cv


# ---------------------------------------------------------------------------
# 3. Ensamble (VotingRegressor)
# ---------------------------------------------------------------------------
def crear_ensamble(modelos: dict, X_train, y_train, X_test, y_test) -> tuple:
    """
    Crea un VotingRegressor combinando los modelos proporcionados.

    Parámetros
    ----------
    modelos : dict
        Diccionario {nombre: modelo_entrenado}.

    Retorna
    -------
    tuple(ensamble, y_pred, métricas)
    """
    estimators = [(nombre, modelo) for nombre, modelo in modelos.items()]

    ensamble = VotingRegressor(estimators=estimators)
    ensamble.fit(X_train, y_train)

    y_pred = ensamble.predict(X_test)

    metricas = {
        'Modelo': 'Ensamble (Voting)',
        'R² Train': r2_score(y_train, ensamble.predict(X_train)),
        'R² Test': r2_score(y_test, y_pred),
        'MAE': mean_absolute_error(y_test, y_pred),
        'RMSE': np.sqrt(mean_squared_error(y_test, y_pred)),
        'MAPE': mean_absolute_percentage_error(y_test, y_pred),
    }

    print(f"  ✓ Ensamble (Voting)     R²={metricas['R² Test']:.4f}  MAE={metricas['MAE']:.2f}")

    return ensamble, y_pred, metricas


# ---------------------------------------------------------------------------
# 4. Evaluación detallada
# ---------------------------------------------------------------------------
def evaluar_modelo(y_test, y_pred, nombre: str = "Modelo") -> pd.DataFrame:
    """
    Calcula métricas de evaluación completas para un modelo.

    Retorna
    -------
    pd.DataFrame con las métricas.
    """
    metricas = pd.DataFrame([{
        'Modelo': nombre,
        'R²': r2_score(y_test, y_pred),
        'MAE': mean_absolute_error(y_test, y_pred),
        'RMSE': np.sqrt(mean_squared_error(y_test, y_pred)),
        'MAPE (%)': mean_absolute_percentage_error(y_test, y_pred) * 100,
    }])
    return metricas
