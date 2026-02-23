"""
plots.py — Funciones de visualización para el análisis exploratorio.

Uso desde el notebook:
    from src.plots import plot_correlacion
"""

import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd


def plot_correlacion(X: pd.DataFrame, columnas: list) -> None:
    """
    Genera un pairplot con coeficientes de correlación de Pearson anotados.

    Colores:
      - 🟢 |r| < 0.5  — correlación baja
      - 🟠 0.5 ≤ |r| < 0.8 — correlación moderada
      - 🔴 |r| ≥ 0.8  — correlación alta

    Parámetros
    ----------
    X : pd.DataFrame
        DataFrame con los datos.
    columnas : list[str]
        Lista de columnas numéricas a graficar.
    """
    g = sns.pairplot(
        X[columnas],
        diag_kind="kde",
        plot_kws=dict(alpha=0.4, s=15, edgecolor="none"),
        diag_kws=dict(fill=True, alpha=0.6, linewidth=1.5),
        corner=True,
        height=2.2,
        aspect=1,
    )

    # Etiquetas abreviadas para mejor legibilidad
    alias = {
        'Oferta_programada': 'Oferta Prog.',
        'Bloqueo': 'Bloqueo',
        'Oferta_disponible': 'Oferta Disp.',
        'Citas_asignadas': 'Citas Asig.',
        'Sobrecupos': 'Sobrecupos',
        'Tasa_bloqueo': 'Tasa Bloq.',
        'Tasa_asignación': 'Tasa Asig.',
        'Tasa_sobrecupo': 'Tasa Sobrec.',
    }
    for ax in g.axes.flat:
        if ax is None:
            continue
        if ax.get_xlabel():
            ax.set_xlabel(alias.get(ax.get_xlabel(), ax.get_xlabel()), fontsize=10)
        if ax.get_ylabel():
            ax.set_ylabel(alias.get(ax.get_ylabel(), ax.get_ylabel()), fontsize=10)

    # Anotar coeficiente de correlación en cada scatter
    corr = X[columnas].corr()
    for i in range(len(columnas)):
        for j in range(i):
            r = corr.iloc[i, j]
            color = "green" if abs(r) < 0.5 else "orange" if abs(r) < 0.8 else "red"
            g.axes[i, j].annotate(
                f"r = {r:.2f}",
                xy=(0.05, 0.92),
                xycoords="axes fraction",
                fontsize=9,
                fontweight="bold",
                color=color,
                bbox=dict(boxstyle="round,pad=0.2", fc="white", alpha=0.8),
            )

    g.figure.suptitle(
        "Matriz de Correlación — Variables Numéricas",
        y=1.02, fontsize=14, fontweight="bold",
    )
    plt.tight_layout()
    plt.show()


def plot_distribucion(X: pd.DataFrame, columnas: list, grupo: str = None) -> None:
    """
    Genera boxplots horizontales para cada variable numérica, opcionalmente
    agrupados por una variable categórica (e.g. Especialidad).

    Cada subplot muestra:
      - Boxplot con bigotes al percentil 5-95.
      - Strip de puntos individuales para detectar outliers visualmente.

    Parámetros
    ----------
    X : pd.DataFrame
        DataFrame con los datos.
    columnas : list[str]
        Lista de columnas numéricas a graficar.
    grupo : str, opcional
        Columna categórica para agrupar (eje Y). Si es None, muestra
        la distribución general de cada variable.
    """
    import numpy as np

    alias = {
        'Oferta_programada': 'Oferta Prog.',
        'Bloqueo': 'Bloqueo',
        'Oferta_disponible': 'Oferta Disp.',
        'Citas_asignadas': 'Citas Asig.',
        'Sobrecupos': 'Sobrecupos',
        'Tasa_bloqueo': 'Tasa Bloq.',
        'Tasa_asignación': 'Tasa Asig.',
        'Tasa_sobrecupo': 'Tasa Sobrec.',
    }

    n = len(columnas)
    fig, axes = plt.subplots(n, 1, figsize=(10, 3.5 * n), constrained_layout=True)
    if n == 1:
        axes = [axes]

    palette = "vlag" if grupo else "Set2"

    for ax, col in zip(axes, columnas):
        if grupo:
            # Ordenar categorías por mediana descendente
            orden = (
                X.groupby(grupo)[col]
                .median()
                .sort_values(ascending=False)
                .index.tolist()
            )
            sns.boxplot(
                data=X, x=col, y=grupo, order=orden,
                hue=grupo, palette=palette, legend=False,
                whis=[5, 95], width=0.6,
                flierprops=dict(marker='o', markersize=3, alpha=0.4),
                ax=ax,
            )
            sns.stripplot(
                data=X, x=col, y=grupo, order=orden,
                size=2.5, color=".3", alpha=0.25, ax=ax,
            )
            ax.set_ylabel("")
        else:
            sns.boxplot(
                data=X, x=col,
                color="steelblue",
                whis=[5, 95], width=0.4,
                flierprops=dict(marker='o', markersize=3, alpha=0.4),
                ax=ax,
            )
            sns.stripplot(
                data=X, x=col,
                size=2.5, color=".3", alpha=0.15, ax=ax,
            )

        label = alias.get(col, col)
        ax.set_xlabel(label, fontsize=12, fontweight="bold")
        ax.xaxis.grid(True, alpha=0.3)
        sns.despine(ax=ax, left=True)

        # Anotar estadísticas clave
        med = X[col].median()
        q1, q3 = X[col].quantile(0.25), X[col].quantile(0.75)
        iqr = q3 - q1
        outliers_low = (X[col] < q1 - 1.5 * iqr).sum()
        outliers_high = (X[col] > q3 + 1.5 * iqr).sum()
        total_outliers = outliers_low + outliers_high
        pct = total_outliers / len(X) * 100

        stats_text = f"Mediana: {med:.2f}  |  IQR: {iqr:.2f}  |  Outliers: {total_outliers:,} ({pct:.1f}%)"
        ax.annotate(
            stats_text, xy=(0.98, 0.95), xycoords="axes fraction",
            fontsize=9, ha="right", va="top",
            bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.85, edgecolor="gray"),
        )

    titulo = "Distribución por Especialidad" if grupo else "Distribución y Outliers — Variables Numéricas"
    fig.suptitle(titulo, fontsize=14, fontweight="bold")
    plt.show()


def plot_distribucion_grid(X: pd.DataFrame, columnas: list, grupo: str) -> None:
    """
    Genera un grid 2×2 de boxplots horizontales, uno por variable numérica,
    con las categorías del grupo en el eje Y ordenadas por mediana.

    Más compacto que subplots apilados cuando hay muchas categorías.

    Parámetros
    ----------
    X : pd.DataFrame
        DataFrame con los datos.
    columnas : list[str]
        Lista de columnas numéricas (idealmente 4 para un grid 2×2).
    grupo : str
        Columna categórica para agrupar en el eje Y (e.g. 'Especialidad').
    """
    import math

    alias = {
        'Oferta_programada': 'Oferta Prog.',
        'Bloqueo': 'Bloqueo',
        'Oferta_disponible': 'Oferta Disp.',
        'Citas_asignadas': 'Citas Asig.',
        'Sobrecupos': 'Sobrecupos',
        'Tasa_bloqueo': 'Tasa Bloq.',
        'Tasa_asignación': 'Tasa Asig.',
        'Tasa_sobrecupo': 'Tasa Sobrec.',
    }

    n = len(columnas)
    ncols = 2
    nrows = math.ceil(n / ncols)
    n_categorias = X[grupo].nunique()
    alto_por_cat = max(0.35, 7 / n_categorias)

    fig, axes = plt.subplots(
        nrows, ncols,
        figsize=(14, alto_por_cat * n_categorias),
        constrained_layout=True,
        sharey=True,
    )
    axes = axes.flatten()

    # Paleta consistente para todas las categorías
    categorias_unicas = sorted(X[grupo].unique())
    palette = dict(zip(categorias_unicas, sns.color_palette("husl", len(categorias_unicas))))

    for idx, col in enumerate(columnas):
        ax = axes[idx]

        # Ordenar por mediana descendente
        orden = (
            X.groupby(grupo)[col]
            .median()
            .sort_values(ascending=True)
            .index.tolist()
        )

        sns.boxplot(
            data=X, x=col, y=grupo, order=orden,
            hue=grupo, palette=palette, legend=False,
            whis=[5, 95], width=0.65,
            flierprops=dict(marker='.', markersize=2, alpha=0.3),
            linewidth=0.8,
            ax=ax,
        )

        label = alias.get(col, col)
        ax.set_xlabel(label, fontsize=11, fontweight="bold")
        ax.set_ylabel("")
        ax.xaxis.grid(True, alpha=0.3)
        ax.tick_params(axis='y', labelsize=8)
        sns.despine(ax=ax, left=True)

    # Ocultar ejes sobrantes si n no es par
    for idx in range(n, len(axes)):
        axes[idx].set_visible(False)

    fig.suptitle(
        f"Distribución por {grupo} — Variables Numéricas",
        fontsize=14, fontweight="bold",
    )
    plt.show()


def plot_heatmap_medianas(X: pd.DataFrame, columnas: list, grupo: str) -> None:
    """
    Heatmap de medianas normalizadas (0-1) por categoría y variable.

    Permite comparar rápidamente qué categorías tienen valores altos o bajos
    en cada variable, independientemente de la escala.

    Parámetros
    ----------
    X : pd.DataFrame
        DataFrame con los datos.
    columnas : list[str]
        Lista de columnas numéricas.
    grupo : str
        Columna categórica para agrupar (filas del heatmap).
    """
    alias = {
        'Oferta_programada': 'Oferta Prog.',
        'Bloqueo': 'Bloqueo',
        'Oferta_disponible': 'Oferta Disp.',
        'Citas_asignadas': 'Citas Asig.',
        'Sobrecupos': 'Sobrecupos',
        'Tasa_bloqueo': 'Tasa Bloq.',
        'Tasa_asignación': 'Tasa Asig.',
        'Tasa_sobrecupo': 'Tasa Sobrec.',
    }

    # Calcular medianas por grupo
    medianas = X.groupby(grupo)[columnas].median()

    # Normalizar entre 0 y 1 por columna para comparación visual
    medianas_norm = (medianas - medianas.min()) / (medianas.max() - medianas.min())
    medianas_norm = medianas_norm.fillna(0)

    # Ordenar por la mediana promedio (especialidades con más actividad arriba)
    medianas_norm = medianas_norm.loc[medianas_norm.mean(axis=1).sort_values(ascending=False).index]
    medianas = medianas.loc[medianas_norm.index]

    # Renombrar columnas
    medianas.columns = [alias.get(c, c) for c in medianas.columns]
    medianas_norm.columns = [alias.get(c, c) for c in medianas_norm.columns]

    n_cats = len(medianas)
    fig, ax = plt.subplots(figsize=(8, max(4, n_cats * 0.4)))

    sns.heatmap(
        medianas_norm,
        annot=medianas.values,
        fmt=".2f",
        cmap="YlOrRd",
        linewidths=0.5,
        linecolor="white",
        cbar_kws={"label": "Intensidad relativa (0 = mín, 1 = máx)", "shrink": 0.8},
        ax=ax,
    )

    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.set_title(
        f"Medianas por {grupo} — Variables Numéricas",
        fontsize=14, fontweight="bold", pad=15,
    )
    ax.tick_params(axis='x', rotation=0)
    ax.tick_params(axis='y', rotation=0, labelsize=9)
    plt.tight_layout()
    plt.show()


# =========================================================================
# GRÁFICOS DE EVALUACIÓN DE MODELOS
# =========================================================================

def plot_comparacion_modelos(df_resultados: pd.DataFrame) -> None:
    """
    Gráfico de barras comparando R² Train vs Test para cada modelo.
    Destaca posible overfitting cuando la diferencia es grande.
    """
    import numpy as np

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # --- R² Comparativo ---
    ax = axes[0]
    modelos = df_resultados['Modelo'].tolist()
    x = np.arange(len(modelos))
    w = 0.35

    bars1 = ax.bar(x - w/2, df_resultados['R² Train'], w, label='R² Train',
                   color='#4C72B0', alpha=0.85, edgecolor='white')
    bars2 = ax.bar(x + w/2, df_resultados['R² Test'], w, label='R² Test',
                   color='#DD8452', alpha=0.85, edgecolor='white')

    for bar in bars1:
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f'{bar.get_height():.3f}', ha='center', va='bottom', fontsize=9)
    for bar in bars2:
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f'{bar.get_height():.3f}', ha='center', va='bottom', fontsize=9)

    ax.set_xticks(x)
    ax.set_xticklabels(modelos, fontsize=10)
    ax.set_ylabel('R²', fontsize=11)
    ax.set_title('R² — Train vs Test', fontsize=13, fontweight='bold')
    ax.legend(loc='lower right')
    ax.set_ylim(0, 1.15)
    ax.axhline(y=0.6, color='green', linestyle='--', alpha=0.5, label='Umbral 60%')
    sns.despine(ax=ax)

    # --- MAE + RMSE ---
    ax = axes[1]
    bars1 = ax.bar(x - w/2, df_resultados['MAE'], w, label='MAE',
                   color='#55A868', alpha=0.85, edgecolor='white')
    bars2 = ax.bar(x + w/2, df_resultados['RMSE'], w, label='RMSE',
                   color='#C44E52', alpha=0.85, edgecolor='white')

    for bar in bars1:
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                f'{bar.get_height():.2f}', ha='center', va='bottom', fontsize=9)
    for bar in bars2:
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                f'{bar.get_height():.2f}', ha='center', va='bottom', fontsize=9)

    ax.set_xticks(x)
    ax.set_xticklabels(modelos, fontsize=10)
    ax.set_ylabel('Error', fontsize=11)
    ax.set_title('Error — MAE vs RMSE', fontsize=13, fontweight='bold')
    ax.legend(loc='upper right')
    sns.despine(ax=ax)

    fig.suptitle('Comparación de Modelos', fontsize=15, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.show()


def plot_prediccion_vs_real(y_test, predicciones: dict) -> None:
    """
    Scatter plot de predicciones vs valores reales para cada modelo.
    La línea diagonal perfecta (y=x) sirve de referencia.
    """
    import numpy as np

    n = len(predicciones)
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 5), constrained_layout=True)
    if n == 1:
        axes = [axes]

    colores = ['#4C72B0', '#DD8452', '#55A868', '#C44E52']

    for ax, (nombre, y_pred), color in zip(axes, predicciones.items(), colores):
        ax.scatter(y_test, y_pred, alpha=0.3, s=12, color=color, edgecolor='none')

        # Línea perfecta
        lim_min = min(y_test.min(), y_pred.min())
        lim_max = max(y_test.max(), y_pred.max())
        ax.plot([lim_min, lim_max], [lim_min, lim_max],
                'k--', alpha=0.6, linewidth=1.5, label='Predicción perfecta')

        from sklearn.metrics import r2_score, mean_absolute_error
        r2 = r2_score(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)

        ax.annotate(
            f"R² = {r2:.4f}\nMAE = {mae:.2f}",
            xy=(0.05, 0.88), xycoords="axes fraction",
            fontsize=10, fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.9, edgecolor="gray"),
        )

        ax.set_xlabel('Valor Real', fontsize=11)
        ax.set_ylabel('Predicción', fontsize=11)
        ax.set_title(nombre, fontsize=12, fontweight='bold')
        ax.legend(loc='lower right', fontsize=9)
        sns.despine(ax=ax)

    fig.suptitle('Predicción vs. Valor Real', fontsize=14, fontweight='bold')
    plt.show()


def plot_residuos(y_test, predicciones: dict) -> None:
    """
    Gráfico de residuos (error) para cada modelo.
    Idealmente los residuos deben estar centrados en 0 y sin patrón.
    """
    import numpy as np

    n = len(predicciones)
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 4), constrained_layout=True)
    if n == 1:
        axes = [axes]

    colores = ['#4C72B0', '#DD8452', '#55A868', '#C44E52']

    for ax, (nombre, y_pred), color in zip(axes, predicciones.items(), colores):
        residuos = y_test.values - y_pred
        ax.scatter(y_pred, residuos, alpha=0.3, s=10, color=color, edgecolor='none')
        ax.axhline(y=0, color='black', linewidth=1, linestyle='--', alpha=0.6)

        ax.set_xlabel('Predicción', fontsize=11)
        ax.set_ylabel('Residuo (Real − Pred)', fontsize=11)
        ax.set_title(nombre, fontsize=12, fontweight='bold')
        sns.despine(ax=ax)

    fig.suptitle('Análisis de Residuos', fontsize=14, fontweight='bold')
    plt.show()


def plot_importancia_features(modelo, feature_names: list, top_n: int = 15) -> None:
    """
    Gráfico horizontal de importancia de features (para modelos con feature_importances_).
    """
    import numpy as np

    if not hasattr(modelo, 'feature_importances_'):
        print("Este modelo no tiene feature_importances_.")
        return

    importancias = pd.DataFrame({
        'Feature': feature_names,
        'Importancia': modelo.feature_importances_,
    }).sort_values('Importancia', ascending=True).tail(top_n)

    fig, ax = plt.subplots(figsize=(8, max(4, top_n * 0.35)))

    bars = ax.barh(importancias['Feature'], importancias['Importancia'],
                   color='#4C72B0', alpha=0.85, edgecolor='white')

    for bar in bars:
        ax.text(bar.get_width() + 0.002, bar.get_y() + bar.get_height()/2,
                f'{bar.get_width():.3f}', va='center', fontsize=9)

    ax.set_xlabel('Importancia', fontsize=11)
    ax.set_title(f'Top {top_n} Features más Importantes', fontsize=13, fontweight='bold')
    sns.despine(ax=ax, left=True)
    plt.tight_layout()
    plt.show()
