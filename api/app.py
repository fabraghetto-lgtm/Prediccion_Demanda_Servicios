"""
API REST — Predicción de Demanda de Servicios

Endpoints:
  POST /predecir  → Recibe datos de oferta/citas y devuelve predicción + confianza.
  GET  /salud     → Health check.
  GET  /metadata  → Información del modelo y valores válidos.
"""

import json
import os
from contextlib import asynccontextmanager
from typing import Optional

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# ---------------------------------------------------------------------------
# Configuración
# ---------------------------------------------------------------------------
ARTIFACTS_DIR = os.path.join(os.path.dirname(__file__), "artifacts")

models = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Carga modelos y artefactos al iniciar."""
    with open(os.path.join(ARTIFACTS_DIR, "metadata.json"), encoding="utf-8") as f:
        models["metadata"] = json.load(f)

    models["ensamble"] = joblib.load(os.path.join(ARTIFACTS_DIR, "modelo_ensamble.joblib"))
    models["rf"] = joblib.load(os.path.join(ARTIFACTS_DIR, "modelo_rf.joblib"))
    models["xgb"] = joblib.load(os.path.join(ARTIFACTS_DIR, "modelo_xgb.joblib"))
    models["scaler"] = joblib.load(os.path.join(ARTIFACTS_DIR, "scaler.joblib"))

    # Cargar CSV como fuente de datos históricos
    historico_path = os.path.join(os.path.dirname(ARTIFACTS_DIR), "..", "data", "dataset_historico.csv")
    if os.path.exists(historico_path):
        models["historico"] = pd.read_csv(historico_path)
        print(f"✓ CSV histórico cargado: {len(models['historico']):,} registros")
    else:
        models["historico"] = None
        print("⚠ CSV histórico no encontrado.")

    print(f"✓ Modelos cargados. Features: {len(models['metadata']['feature_names'])}")
    yield
    models.clear()


app = FastAPI(
    title="API Predicción de Demanda de Servicios",
    description="Predice la demanda (cupos vendidos) por categoría de servicio.",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Esquemas de entrada/salida
# ---------------------------------------------------------------------------
class DatosEntrada(BaseModel):
    """Datos necesarios para predecir la demanda."""

    Especialidad: str = Field(..., example="Categoria_08")
    Semana_Iso: int = Field(..., ge=1, le=53, example=25)
    Mes: int = Field(..., ge=1, le=12, example=6)
    Dia_Sem_Iso: int = Field(..., ge=1, le=7, example=3)
    Jornada_Horaria: str = Field(..., example="Turno_A")
    Oferta_programada: float = Field(..., gt=0, example=20)
    Bloqueo: float = Field(..., ge=0, example=2)
    Oferta_disponible: float = Field(..., ge=0, example=18)
    Citas_asignadas: Optional[float] = Field(None, ge=0, example=15,
        description="Opcional. Si no se proporciona, se estima con promedio móvil de 4 semanas.")

    model_config = {"json_schema_extra": {"examples": [{
        "Especialidad": "Categoria_08",
        "Semana_Iso": 25,
        "Mes": 6,
        "Dia_Sem_Iso": 3,
        "Jornada_Horaria": "Turno_A",
        "Oferta_programada": 20,
        "Bloqueo": 2,
        "Oferta_disponible": 18,
    }]}}


class Prediccion(BaseModel):
    """Resultado de la predicción."""

    prediccion: float = Field(..., description="Cupos vendidos predichos")
    confianza_porcentaje: float = Field(..., description="Porcentaje de confianza de la predicción (0-100)")
    intervalo_inferior: float = Field(..., description="Límite inferior del intervalo de confianza")
    intervalo_superior: float = Field(..., description="Límite superior del intervalo de confianza")
    detalle: dict = Field(..., description="Desglose de predicciones por sub-modelo")


# ---------------------------------------------------------------------------
# Lógica de preprocesamiento (replica el pipeline del notebook)
# ---------------------------------------------------------------------------
def calcular_promedio_movil_tasa_asignacion(
    especialidad: str,
    jornada: str,
    semana_objetivo: int,
    ventana: int = 4,
) -> float:
    """
    Calcula el promedio móvil de Tasa_asignación para una categoría/jornada
    usando las últimas N semanas previas a la semana objetivo.
    """
    if models.get("historico") is None:
        return 0.8

    df_historico = models["historico"]

    # Filtrar por categoría y jornada
    mask = (
        (df_historico['Especialidad'] == especialidad) &
        (df_historico['Jornada_Horaria'] == jornada) &
        (df_historico['Semana_Iso'] < semana_objetivo)
    )
    data = df_historico[mask].copy()

    if len(data) == 0:
        # Fallback: Promedio de la categoría (sin filtrar por jornada)
        mask_esp = (
            (df_historico['Especialidad'] == especialidad) &
            (df_historico['Semana_Iso'] < semana_objetivo)
        )
        data = df_historico[mask_esp].copy()

    if len(data) == 0:
        return 0.8  # Valor default típico

    # Calcular Tasa_asignación
    data['Tasa_asignacion'] = (
        data['Citas_asignadas'] / data['Oferta_disponible'].replace(0, np.nan)
    ).fillna(0)

    # Tomar las últimas N semanas
    data_sorted = data.sort_values('Semana_Iso', ascending=False).head(ventana)

    promedio = data_sorted['Tasa_asignacion'].mean()

    return promedio if not np.isnan(promedio) else 0.8


def preprocesar_input(datos: DatosEntrada) -> pd.DataFrame:
    """
    Transforma los datos de entrada al formato esperado por el modelo:
      1. Estima Tasa_asignación con promedio móvil si no se proporciona Citas_asignadas.
      2. Calcula ratios (Tasa_bloqueo, Tasa_asignación).
      3. Codificación cíclica de Mes y Dia_Sem_Iso (seno/coseno).
      4. One-hot encoding de Especialidad y Jornada_Horaria + Semana_Iso directo.
      5. Escalado con el StandardScaler guardado (numéricas + cíclicas).
    """
    meta = models["metadata"]
    scaler = models["scaler"]

    # --- Validar valores categóricos ---
    cat_values = meta["categorical_values"]
    if datos.Especialidad not in cat_values["Especialidad"]:
        raise HTTPException(
            status_code=422,
            detail=f"Especialidad '{datos.Especialidad}' no válida. "
                   f"Opciones: {cat_values['Especialidad']}",
        )
    if datos.Jornada_Horaria not in cat_values["Jornada_Horaria"]:
        raise HTTPException(
            status_code=422,
            detail=f"Jornada_Horaria '{datos.Jornada_Horaria}' no válida. "
                   f"Opciones: {cat_values['Jornada_Horaria']}",
        )

    # --- 1. Calcular Tasa_asignación ---
    oferta_disp = datos.Oferta_disponible if datos.Oferta_disponible > 0 else 1.0

    if datos.Citas_asignadas is not None:
        # Usar valor proporcionado
        tasa_asignacion = datos.Citas_asignadas / oferta_disp
    else:
        # Calcular con promedio móvil (CSV histórico)
        tasa_asignacion = calcular_promedio_movil_tasa_asignacion(
            datos.Especialidad,
            datos.Jornada_Horaria,
            datos.Semana_Iso,
            ventana=4
        )

    # --- 2. Calcular Tasa_bloqueo ---
    oferta_prog = datos.Oferta_programada
    tasa_bloqueo = datos.Bloqueo / oferta_prog if oferta_prog > 0 else 0

    # --- 3. Codificación cíclica de variables temporales ---
    mes_sin = np.sin(2 * np.pi * datos.Mes / 12)
    mes_cos = np.cos(2 * np.pi * datos.Mes / 12)
    dia_sin = np.sin(2 * np.pi * datos.Dia_Sem_Iso / 7)
    dia_cos = np.cos(2 * np.pi * datos.Dia_Sem_Iso / 7)

    # --- 4. Construir vector de features ---
    feature_names = meta["feature_names"]
    ciclicas = ["Mes_sin", "Mes_cos", "Dia_sin", "Dia_cos"]

    # Crear DataFrame con todas las features en cero
    X = pd.DataFrame(np.zeros((1, len(feature_names))), columns=feature_names)

    # Rellenar numéricas
    X["Oferta_programada"] = oferta_prog
    X["Tasa_bloqueo"] = tasa_bloqueo
    X["Tasa_asignación"] = tasa_asignacion

    # Rellenar cíclicas
    X["Mes_sin"] = mes_sin
    X["Mes_cos"] = mes_cos
    X["Dia_sin"] = dia_sin
    X["Dia_cos"] = dia_cos

    # Rellenar Semana_Iso como feature directa
    if "Semana_Iso" in X.columns:
        X["Semana_Iso"] = datos.Semana_Iso

    # Rellenar dummies (one-hot manual para Especialidad y Jornada_Horaria)
    categorical_input = {
        "Especialidad": datos.Especialidad,
        "Jornada_Horaria": datos.Jornada_Horaria,
    }

    for col_name in meta["encoder_columns"]:
        for cat_col in categorical_input:
            prefix = f"{cat_col}_"
            if col_name.startswith(prefix):
                valor_col = col_name[len(prefix):]
                valor_input = str(categorical_input[cat_col])
                if valor_col == valor_input:
                    X[col_name] = 1
                break

    # --- 5. Escalar numéricas + cíclicas ---
    cols_to_scale = meta["numerical"] + ciclicas
    cols_to_scale = [c for c in cols_to_scale if c in X.columns]
    X[cols_to_scale] = scaler.transform(X[cols_to_scale])

    return X


# ---------------------------------------------------------------------------
# Cálculo de confianza
# ---------------------------------------------------------------------------
def calcular_confianza(pred_rf: float, pred_xgb: float, pred_ensamble: float) -> dict:
    """
    Estima la confianza basándose en:
      1. Concordancia entre sub-modelos (menor dispersión → mayor confianza).
      2. R² del modelo como base de confianza.
    """
    meta = models["metadata"]
    r2_base = meta["r2_test"]

    # Dispersión relativa entre modelos
    preds = np.array([pred_rf, pred_xgb])
    std = np.std(preds)
    media = np.abs(pred_ensamble) if pred_ensamble != 0 else 1

    # Coeficiente de variación (CV)
    cv = std / max(media, 1)

    # Confianza = R² base ajustada por acuerdo entre modelos
    ajuste = max(0, 1 - cv)
    confianza = r2_base * ajuste * 100
    confianza = round(min(max(confianza, 0), 99.9), 1)

    # Intervalo de confianza simple basado en MAE del test
    mae = meta["mae_test"]
    margen = max(mae, std * 1.96)

    return {
        "confianza": confianza,
        "intervalo_inf": round(pred_ensamble - margen, 2),
        "intervalo_sup": round(pred_ensamble + margen, 2),
    }


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------
@app.get("/salud", tags=["Sistema"])
async def health_check():
    """Health check del servicio."""
    csv_status = "cargado" if models.get("historico") is not None else "no disponible"

    return {
        "estado": "activo",
        "modelo": "Ensamble (RF + XGBoost)",
        "features": len(models["metadata"]["feature_names"]),
        "fuentes_datos": {
            "csv_local": csv_status,
        }
    }


@app.get("/metadata", tags=["Sistema"])
async def get_metadata():
    """Devuelve información del modelo y valores válidos para los campos categóricos."""
    meta = models["metadata"]
    return {
        "modelo": "Ensamble (Random Forest + XGBoost)",
        "r2_test": meta["r2_test"],
        "mae_test": meta["mae_test"],
        "variables_categoricas": meta["categorical_values"],
        "variables_numericas": meta["numerical"],
        "total_features": len(meta["feature_names"]),
    }


@app.post("/predecir", response_model=Prediccion, tags=["Predicción"])
async def predecir(datos: DatosEntrada):
    """
    Predice los cupos vendidos para una categoría de servicio a partir de datos
    de oferta y citas.

    Devuelve la predicción, un porcentaje de confianza y el intervalo estimado.
    """
    # Preprocesar
    X = preprocesar_input(datos)

    # Predicciones individuales
    pred_rf = float(models["rf"].predict(X)[0])
    pred_xgb = float(models["xgb"].predict(X)[0])
    pred_ensamble = float(models["ensamble"].predict(X)[0])

    # Calcular confianza
    confianza = calcular_confianza(pred_rf, pred_xgb, pred_ensamble)

    return Prediccion(
        prediccion=round(pred_ensamble, 2),
        confianza_porcentaje=confianza["confianza"],
        intervalo_inferior=confianza["intervalo_inf"],
        intervalo_superior=confianza["intervalo_sup"],
        detalle={
            "random_forest": round(pred_rf, 2),
            "xgboost": round(pred_xgb, 2),
            "ensamble": round(pred_ensamble, 2),
        },
    )
