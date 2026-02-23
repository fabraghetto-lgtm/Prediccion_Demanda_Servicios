# Modelo Predictivo de Demanda de Servicios

Modelo de Machine Learning que predice la **demanda de servicios** (cupos vendidos) a partir de variables de oferta, bloqueo, citas y características categóricas, con una API REST pública para consultas en tiempo real.

## Resultados

| Modelo | R² Test | MAE | RMSE |
|--------|---------|-----|------|
| Regresión Lineal | 0.81 | — | — |
| Random Forest (tuned) | 0.96 | — | — |
| XGBoost (tuned) | 0.98 | — | — |
| **Ensamble (RF + XGB)** | **0.986** | **2.11** | — |

> El modelo final es un **VotingRegressor** que combina Random Forest y XGBoost tuneados, logrando un **R² de 0.986** sobre datos de prueba.

---

## Estructura del Proyecto

```
├── Modelo_Predictivo.ipynb       # Notebook principal (EDA → Modelado → Evaluación)
├── README.md
├── render.yaml                   # Config de deploy (Render.com)
├── .gitignore
├── data/                         # Datasets
│   ├── dataset_historico.csv     # Datos de entrenamiento
│   ├── dataset_validacion.csv    # Validación out-of-time
│   ├── semanas_a_predecir.csv    # Semanas futuras
│   ├── predicciones.csv          # Predicciones generadas
│   ├── resumen_predicciones.csv  # Resumen por categoría
│   └── resumen_completo.csv      # Resumen consolidado
├── src/                          # Módulos Python
│   ├── data.py                   # Carga y separación de variables
│   ├── features.py               # Feature engineering y preprocesamiento
│   ├── model.py                  # Entrenamiento, tuning y ensamble
│   └── plots.py                  # Visualizaciones personalizadas
└── api/                          # API REST (FastAPI)
    ├── app.py                    # Aplicación principal
    ├── requirements.txt          # Dependencias del API
    └── artifacts/                # Artefactos del modelo
        ├── modelo_ensamble.joblib
        ├── modelo_rf.joblib
        ├── modelo_xgb.joblib
        ├── scaler.joblib
        └── metadata.json
```

---

## Metodología

### 1. Análisis Exploratorio (EDA)
- Estadísticas descriptivas y análisis de completitud
- Matriz de correlación de Pearson → detección de multicolinealidad
- Distribuciones y boxplots por categoría
- Heatmap de medianas

### 2. Feature Engineering
- **Ratios** para eliminar colinealidad: `Tasa_bloqueo = Bloqueo / Oferta_programada`, `Tasa_asignación = Citas / Oferta_disponible`
- **Codificación cíclica** (seno/coseno) para variables temporales (`Mes`, `Día_semana`)
- **One-Hot Encoding** para categóricas (`Especialidad`, `Jornada_Horaria`, `Semana_Iso`)
- **StandardScaler** ajustado solo sobre datos de entrenamiento (sin data leakage)
- **VIF** (Variance Inflation Factor) para validar ausencia de multicolinealidad

### 3. Limpieza de Outliers
1. **Reglas de dominio:** Elimina registros con valores lógicamente imposibles
2. **IQR por categoría:** Rango intercuartílico (1.5 × IQR) por grupo

### 4. Modelado
- **Regresión Lineal:** Baseline (R² ≈ 0.81)
- **Random Forest:** Ensemble de bagging (R² ≈ 0.96 post-tuning)
- **XGBoost:** Gradient Boosting (R² ≈ 0.98 post-tuning)
- **VotingRegressor:** Ensamble final RF + XGB (R² ≈ 0.986)
- **GridSearchCV** con 5-fold cross-validation para tuning de hiperparámetros

### 5. Evaluación
- Métricas: R², MAE, RMSE, MAPE
- Gráficos: Comparación de modelos, Predicción vs. Real, Residuos, Importancia de Features
- **Validación out-of-time** con datos de un período futuro no visto durante el entrenamiento

---

## API REST

### Endpoints

| Método | Ruta | Descripción |
|--------|------|-------------|
| `GET` | `/salud` | Health check |
| `GET` | `/metadata` | Info del modelo y valores válidos |
| `POST` | `/predecir` | Predicción + intervalo de confianza |
| `GET` | `/docs` | Documentación interactiva (Swagger UI) |

### Ejemplo de uso

```bash
curl -X POST http://localhost:8000/predecir \
  -H "Content-Type: application/json" \
  -d '{
    "Especialidad": "Categoria_08",
    "Semana_Iso": 25,
    "Mes": 6,
    "Dia_Sem_Iso": 3,
    "Jornada_Horaria": "Turno_A",
    "Oferta_programada": 20,
    "Bloqueo": 2,
    "Oferta_disponible": 18,
    "Citas_asignadas": 15
  }'
```

**Respuesta:**
```json
{
  "prediccion": 5.94,
  "confianza_porcentaje": 79.7,
  "intervalo_inferior": 3.78,
  "intervalo_superior": 8.1,
  "detalle": {
    "random_forest": 4.84,
    "xgboost": 7.04,
    "ensamble": 5.94
  }
}
```

### Ejecución local

```bash
pip install -r api/requirements.txt
uvicorn api.app:app --host 127.0.0.1 --port 8000
```

---

## Instalación y Ejecución

### Requisitos
- Python 3.11+
- pip

### Setup

```bash
# 1. Clonar el repositorio
git clone https://github.com/<tu-usuario>/Prediccion_Demanda_Servicios.git
cd Prediccion_Demanda_Servicios

# 2. Crear entorno virtual
python -m venv venv
source venv/bin/activate   # Linux/Mac
# venv\Scripts\activate    # Windows

# 3. Instalar dependencias
pip install pandas numpy scikit-learn xgboost seaborn matplotlib joblib

# 4. Ejecutar el notebook
jupyter notebook Modelo_Predictivo.ipynb

# 5. (Opcional) Levantar la API
pip install -r api/requirements.txt
uvicorn api.app:app --port 8000
```

---

## Tecnologías

| Categoría | Herramientas |
|-----------|-------------|
| Lenguaje | Python 3.11 |
| ML | scikit-learn, XGBoost |
| Datos | pandas, NumPy |
| Visualización | Matplotlib, Seaborn |
| API | FastAPI, Uvicorn, Pydantic |
| Deploy | Render.com |

---

## Licencia

Este proyecto fue desarrollado con fines educativos.
