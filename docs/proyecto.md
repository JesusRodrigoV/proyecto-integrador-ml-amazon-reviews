# CONTEXTO GENERAL DEL PROYECTO INTEGRADOR

**Asignatura:** Machine Learning (Ciclo 2026-I)

**Metodología Marco:** CRISP-DM

**Grupo de Trabajo:** Grupo 2

**Dominio de Aplicación:** Reseñas de Productos de Oficina (*Amazon Reviews 2023 – Office Products*)

---

## 1. Propósito y Enfoque del Negocio (Business Understanding)

El objetivo primordial de este proyecto es transformar datos no estructurados (reseñas textuales y metadatos de productos de oficina) en un **sistema inteligente de apoyo a la toma de decisiones y consulta**. El sistema busca identificar con precisión los atributos críticamente valorados o rechazados por los usuarios, permitiendo la segmentación de productos y la optimización de la experiencia del cliente mediante el análisis automatizado de feedback.

---

## 2. Alcance Técnico y Arquitectura Progresiva

El proyecto no se concibe como una serie de ejercicios aislados, sino como una solución de ingeniería de datos articulada en tres grandes hitos evolutivos:

* **Fase de Machine Learning Clásico:** Establecimiento de la línea base (*baseline*). Modelado supervisado de una variable objetivo clave (Satisfacción, polaridad o *helpfulness*) mediante al menos tres algoritmos tradicionales, contrastados obligatoriamente contra un *benchmark* automatizado mediante **AutoML**. Incluye una capa no supervisada para la segmentación de productos o tipologías de reseñas.
* **Fase de Deep Learning:** Evolución del modelado mediante el uso de **Transformers preentrenados** (bajo política estricta de inferencia o *fine-tuning* ultra-liviano para viabilidad de hardware). Extracción de representaciones latentes y mapeo de embeddings para el análisis de similitud y clustering avanzado, comparando el rendimiento vs. los modelos de la fase clásica.
* **Fase de MLOps + Recuperación Semántica (RAG):** Construcción de un motor de búsqueda vectorial que permita realizar consultas de lenguaje natural sobre las opiniones. Se incluye un pipeline de producción mínimo viable (MLOps) enfocado en la **reproducibilidad**, el *tracking* de experimentos, el versionado del índice vectorial y del modelo, y un esquema de monitoreo/actualización ante la llegada de nuevas reseñas.

---

## 3. Especificaciones del Dataset Principal

* **Fuente Oficial:** McAuley Lab (*Amazon Reviews 2023*).
* **Categoría Asignada:** *Office Products*.
* **Restricciones de Datos:** Debido al volumen masivo de la fuente original, se aplica un criterio riguroso de submuestreo documentado en la carpeta `/data`, asegurando la representatividad, el balance de clases y la eliminación de *data leakage* o sesgos evidentes en el EDA.

---

## 4. Reglas de Operación y Criterios de Calidad

1. **Gobernanza de Modelos:** Todo experimento, ya sea manual o vía AutoML, debe registrarse en el componente de *tracking*. No se aceptan modelos en producción sin trazabilidad de hiperparámetros.
2. **Principio de Reproducibilidad:** El entorno completo debe ser replicable de extremo a extremo utilizando la estructura de carpetas estándar (`/src`, `/data`, `/models`, `/config`) y el archivo de dependencias correspondiente.
3. **Evaluación Justa:** La comparación de métricas entre modelos clásicos, AutoML y Deep Learning se realiza estrictamente bajo el mismo protocolo de evaluación y la misma partición de datos (*train / validation / test*).
4. **Criterio de Veracidad:** El sistema de búsqueda vectorial debe demostrar pertinencia semántica real; no se admiten respuestas simuladas o heurísticas simples sin respaldo del espacio de embeddings.