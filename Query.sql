-- ============================================================
-- Query de extracción de datos para el modelo predictivo
-- ============================================================
-- Fuente: Data Warehouse (BigQuery)
-- Descripción: Agrega oferta, bloqueo, citas y demanda por
--              categoría de servicio, turno, semana y día.
-- ============================================================

With Base_Agenda AS (
  SELECT 
    Categorias.categoria_interna as Especialidad,
    CASE 
      WHEN CAST(SUBSTR(CAST(Oferta.Interval_From_Hrs AS STRING), 1, 2) AS INT64) < 13 THEN 'Turno_A'
      WHEN CAST(SUBSTR(CAST(Oferta.Interval_From_Hrs AS STRING), 1, 2) AS INT64) < 18 THEN 'Turno_B'
      ELSE 'Turno_C'
    END as Jornada_Horaria,
    EXTRACT(MONTH FROM Oferta.fecha) as Mes,
    
    -- Semana ISO (1 a 53)
    EXTRACT(ISOWEEK FROM Oferta.fecha) as Semana_Iso,
    
    -- Día de la semana ISO (Lunes=1 ... Domingo=7)
    CAST(FORMAT_DATE('%u', Oferta.fecha) AS INT64) as Dia_Sem_Iso,

    ROUND(SUM(Oferta.Oferta_programada)) as Oferta_programada,
    ROUND(SUM(Oferta.bloqueo)) as Bloqueo,
    ROUND(SUM(Oferta.Oferta_disponible)) as Oferta_disponible,
    ROUND(SUM(Oferta.citas_asignadas)) as Citas_asignadas,
    ROUND(SUM(Oferta.citas_sobrecupo)) as Sobrecupos,
    ROUND(SUM(Oferta.cupos_vendidos)) as Cupos_Vendidos
    
  FROM `proyecto.dataset.indicadores_oferta` as Oferta
      
  LEFT JOIN `proyecto.dataset.maestro_categorias` as Categorias
    ON Oferta.id = Categorias.id
  WHERE LOWER(centro_diccionario_agenda) LIKE '%centro%'
    AND fecha BETWEEN '2025-01-01' AND '2025-12-31'

  GROUP BY ALL
)

SELECT 
  A.*, 
  CAST(REPLACE(string_field_1,".","") AS INT) as Ticket_Integral_Especialidad
FROM Base_Agenda as A
LEFT JOIN `proyecto.dataset.ticket_integral` as B
  ON A.Especialidad = B.string_field_0
