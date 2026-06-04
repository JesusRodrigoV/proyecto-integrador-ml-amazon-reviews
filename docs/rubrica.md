# DIRECTRICES DE EVALUACIÓN Y RÚBRICA - PROYECTO "AMAZON REVIEWS" (PARA IA AGENTE)

## 1. PARÁMETRO GENERAL DE OPERACIÓN
Como agente de IA asistente, tu objetivo es maximizar la puntuación del equipo asegurando que cada bloque de código, experimento y documento cumpla estrictamente con la rúbrica oficial. No se admiten decisiones heurísticas sin respaldo empírico. Todo resultado debe ser explicable, trazable y reproducible.

## 2. DESGLOSE DE EVALUACIÓN POR FASE (100 PUNTOS POR FASE)

### Fase 1: Propuesta (Fundamentación y Diseño)
* **Claridad del problema (15 pts):** El problema debe estar enfocado en la asimetría de información y la insuficiencia de la calificación de 5 estrellas.
* **Pertinencia del dataset y fuente (15 pts):** Se debe justificar el submuestreo desde la fuente oficial del McAuley Lab.
* **Diseño metodológico CRISP-DM (20 pts):** Integración estricta de las fases de entendimiento comercial, datos, preparación, modelado, evaluación y despliegue.
* **Coherencia de los 3 mini proyectos (20 pts):** La transición de ML clásico a DL y luego a MLOps/RAG debe ser lógica, no fragmentada.
* **Factibilidad técnica (15 pts):** Justificación del uso de CPU para ML y GPU (NVIDIA T4) para DL, controlando el OOM (Out Of Memory).
* **Presentación y defensa (15 pts).**

### Fase 2: Machine Learning Clásico
* **EDA y preparación de datos (20 pts):** Es crítico documentar el filtro por `word_count` y el tratamiento del desbalance de clases.
* **Modelos clásicos, AutoML y comparación (25 pts):** Obligatorio el uso de Regresión Logística (baseline), ensamblajes (Random Forest/LightGBM) y la ejecución de `LazyPredict` como benchmark automatizado sobre la misma partición.
* **Validación, métricas y análisis de error (25 pts):** Puntuación penalizada si se prioriza el Accuracy. Obligatorio maximizar el F1-Score macro.
* **Componente no supervisado (15 pts):** Uso de K-Means o LDA con interpretación sustantiva de los clústeres de quejas, no solo gráficos decorativos.
* **Documentación y reproducibilidad (15 pts).**

### Fase 3: Deep Learning (Transformers)
* **Diseño e implementación del modelo (25 pts):** Uso de Transformers preentrenados destilados (ej. DistilBERT) limitados a inferencia o ajuste liviano. Cero entrenamientos desde cero.
* **Entrenamiento y evaluación (20 pts):** Control estricto de hiperparámetros.
* **Embeddings o representación latente (20 pts):** Extracción matemática de la intención del texto a un espacio vectorial para superar el filtrado léxico.
* **Comparación con baseline clásico (20 pts):** Contraste objetivo y cuantitativo contra la Regresión Logística/AutoML de la Fase 2.
* **Claridad técnica y documentación (15 pts).**

### Fase 4: MLOps + RAG (Recuperación Semántica)
* **Pipeline reproducible y MLOps mínimo (20 pts):** Automatización del flujo desde la ingesta hasta la inferencia.
* **Tracking y versionado (20 pts):** Uso estricto de `Git` (código), `DVC` (datos/modelos) y `MLflow` (telemetría de experimentos).
* **Recuperación semántica o visual (25 pts):** Implementación de motor vectorial (FAISS/ChromaDB). Penalización severa si la búsqueda es léxica disfrazada de semántica.
* **Demo funcional (20 pts):** Prototipo de inferencia interactiva.
* **Monitoreo y reentrenamiento (15 pts):** Estrategia definida para actualización de índices ante nuevas reseñas.

### Fase 5: Integración Final
* **Integración de las tres partes (25 pts):** Continuidad arquitectónica.
* **Coherencia con CRISP-DM (20 pts).**
* **Calidad del informe final (20 pts).**
* **Presentación y defensa (20 pts).**
* **Reflexión crítica, ética y limitaciones (15 pts):** Exposición honesta de los sesgos del modelo, errores persistentes y limitaciones de hardware.

## 3. REGLAS ÉTICAS Y DE INTEGRIDAD (RESTRICCIONES DURAS)
1. **Evidencia Empírica:** Prohibido redactar conclusiones basadas en intuición. Si el F1-Score es deficiente, se documenta como tal con su respectivo análisis de error.
2. **Honestidad Académica:** Todo uso de librerías, código base o tutoriales debe ser explícitamente citado en los reportes generados.
3. **Trazabilidad de Transformaciones:** Cualquier criterio de exclusión (ej. eliminar reseñas con menos de N palabras) debe quedar registrado en código y en texto.