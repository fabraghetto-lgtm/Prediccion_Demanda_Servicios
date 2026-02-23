# Presentación: Modelo Predictivo de Demanda de Servicios

---

## Diapositiva 1 — Contexto y Problema

### ¿Cuál es el problema?

Cada semana se planifican cientos de bloques de atención para distintas categorías de servicio.
Sin embargo, **no se sabe con precisión cuántos de esos cupos serán realmente utilizados**.

Esto genera dos problemas:

- **Sobreoferta:** Se abren más cupos de los que se necesitan → recursos desperdiciados.
- **Suboferta:** Se abren pocos cupos → personas sin atención, listas de espera.

### Objetivo

> Construir un modelo de Machine Learning capaz de **predecir la demanda real** (cupos vendidos)
> a partir de la oferta programada, bloqueos, citas asignadas y variables como la categoría, el mes
> y el día de la semana.

---

## Diapositiva 2 — Los Datos

### ¿Con qué datos se trabajó?

| Aspecto | Detalle |
|---------|---------|
| **Registros** | ~23,900 filas históricas |
| **Período** | Datos semanales de varios meses |
| **Variables** | Oferta programada, bloqueo, oferta disponible, citas asignadas, categoría de servicio, jornada, semana, mes, día |
| **Variable a predecir** | **Cupos vendidos** (demanda) |

### Ejemplo simplificado

| Categoría | Mes | Oferta | Bloqueo | Disponible | Citas | **Cupos Vendidos** |
|-----------|-----|--------|---------|------------|-------|-------------------|
| Categoria_08 | 6 | 20 | 2 | 18 | 15 | **6** |
| Categoria_15 | 3 | 35 | 5 | 30 | 28 | **12** |

---

## Diapositiva 3 — Proceso de Solución

### Metodología paso a paso

```
1. Exploración de Datos (EDA)
   ↓
2. Limpieza y Transformación
   ↓
3. Entrenamiento de Modelos
   ↓
4. Ajuste Fino (Tuning)
   ↓
5. Combinación de Modelos (Ensamble)
   ↓
6. Validación con Datos Nuevos
   ↓
7. Despliegue como API REST
```

### ¿Qué significa cada paso?

1. **Explorar:** Entender los datos, detectar valores faltantes, identificar patrones.
2. **Limpiar:** Eliminar datos imposibles (ej: oferta negativa) y valores atípicos.
3. **Entrenar:** Probar 3 algoritmos diferentes para encontrar el mejor.
4. **Ajustar:** Optimizar la configuración de cada algoritmo para mejorar precisión.
5. **Combinar:** Unir los mejores modelos para mayor estabilidad.
6. **Validar:** Probar con datos que el modelo nunca vio.
7. **Desplegar:** Crear un servicio web que cualquier sistema puede consultar.

---

## Diapositiva 4 — Modelos Utilizados

### ¿Qué modelos se probaron?

| Modelo | Descripción en simple | Resultado (R²) |
|--------|----------------------|----------------|
| **Regresión Lineal** | Traza una línea recta que mejor se ajusta a los datos | 0.81 (81%) |
| **Random Forest** | Combina cientos de "árboles de decisión" independientes | 0.96 (96%) |
| **XGBoost** | Similar, pero cada árbol aprende de los errores del anterior | 0.98 (98%) |
| **Ensamble Final** | Promedia las predicciones de Random Forest + XGBoost | **0.986 (98.6%)** |

> **R² = 0.986** significa que el modelo explica el **98.6%** de la variación en la demanda.
> Solo un 1.4% queda sin explicar.

---

## Diapositiva 5 — Rendimiento del Modelo

### Gráfico 1: Comparación de Modelos

```
R² (mayor es mejor)              Error MAE (menor es mejor)

1.0 ┤                             15 ┤
    │  ████                          │  ████
0.9 ┤  ████  ████                 10 ┤  ████  
    │  ████  ████  ████              │  ████  ████
0.8 ┤  ████  ████  ████  ████     5 ┤  ████  ████  ████
    │  ████  ████  ████  ████        │  ████  ████  ████  ████
    └──────────────────────          └──────────────────────
     Linear  RF   XGB  Ensamble      Linear  RF   XGB  Ensamble
```

### Gráfico 2: Predicción vs. Realidad

El modelo predice valores muy cercanos a la realidad: los puntos se alinean sobre la diagonal perfecta.

### Métricas clave del Ensamble

| Métrica | Valor | Significado |
|---------|-------|-------------|
| **R²** | 0.986 | El modelo captura el 98.6% de la variabilidad |
| **MAE** | ~2.11 | En promedio, se equivoca por ≈2 cupos |
| **RMSE** | ~3.5 | Penaliza errores grandes. Muy cercano al MAE → errores consistentes |

---

## Diapositiva 6 — API REST (Demo)

### ¿Cómo se usa el modelo?

Se creó un **servicio web (API)** al que cualquier sistema puede enviar datos y recibir una predicción al instante.

### Flujo

```
Sistema externo                           API REST
     │                                       │
     │── POST /predecir ──────────────────→  │
     │   {categoría, mes, oferta, ...}       │
     │                                       │── Procesa datos
     │                                       │── Aplica modelo
     │                                       │── Calcula confianza
     │  ←────────────────── Respuesta ──────│
     │   {predicción: 6, confianza: 80%}     │
```

### Respuesta del modelo

El API no solo predice un número, sino que también indica:
- **Predicción:** Cuántos cupos se venderán.
- **Confianza:** Qué tan seguro está el modelo (%).
- **Intervalo:** Rango probable (ej: entre 4 y 8 cupos).
- **Detalle:** Predicción individual de cada sub-modelo.

---

## Diapositiva 7 — Validación con Datos Nuevos

### ¿El modelo funciona con datos que nunca vio?

Se evaluó el modelo con **datos de un período futuro** que no se usaron para entrenar:

| Evaluación | R² | MAE |
|------------|-----|-----|
| Test original (80/20) | 0.986 | 2.11 |
| Datos nuevos (out-of-time) | ~0.95+ | ~3 |

> El modelo mantiene un rendimiento alto incluso con datos completamente nuevos,
> confirmando que **no está sobreajustado** y **generaliza correctamente**.

---

## Diapositiva 8 — Aprendizajes

### ¿Qué se aprendió?

1. **Feature Engineering marca la diferencia:** Transformar variables "brutas" en ratios (tasas de bloqueo, tasas de asignación) redujo la multicolinealidad y mejoró todos los modelos.

2. **No basta un solo modelo:** Combinar modelos (Ensamble) produjo resultados más estables y precisos que cualquier modelo individual.

3. **La validación out-of-time es clave:** El test 80/20 no es suficiente. Probar con datos de un período futuro real da mayor certeza de que el modelo funcionará en producción.

4. **La limpieza de datos es el 80% del trabajo:** La mayor parte del esfuerzo estuvo en entender, limpiar y transformar los datos, no en el modelo en sí.

5. **Una API convierte un notebook en una herramienta real:** Sin el API, el modelo queda en un archivo. Con el API, cualquier sistema puede consultar predicciones en tiempo real.

### Tecnologías principales

Python · scikit-learn · XGBoost · FastAPI · pandas · Seaborn

---

## Diapositiva 9 — Resumen Ejecutivo

| Aspecto | Detalle |
|---------|---------|
| **Problema** | Predecir demanda de servicios para optimizar la oferta |
| **Solución** | Modelo de ML con ensamble de Random Forest + XGBoost |
| **Precisión** | R² = 0.986 (98.6% de la variabilidad explicada) |
| **Error promedio** | ≈ 2 cupos de diferencia |
| **Despliegue** | API REST pública con FastAPI en Render.com |
| **Valor** | Permite planificación informada: reducir sobreoferta y suboferta |

---

*Presentación generada para el Proyecto Final — Módulo 7*
