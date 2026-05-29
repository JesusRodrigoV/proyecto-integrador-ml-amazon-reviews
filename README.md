# Proyecto Integrador — ML sobre Amazon Reviews (Office Products)

**Grupo 2** · Auditoría de satisfacción + Sistema de recuperación semántica (RAG)

Dataset: [Amazon Reviews 2023](https://cseweb.ucsd.edu/~jmcauley/datasets.html) — categoría **Office Products**.

## Metodología

CRISP-DM — 5 fases progresivas:

| Fase | Contenido |
|------|-----------|
| **Fase 1** | Propuesta del proyecto (definición, justificación, métricas, cronograma) |
| **Fase 2** | ML Clásico: 3 modelos (baseline + RF + LightGBM) + AutoML (LazyPredict) + segmentación no supervisada |
| **Fase 3** | Deep Learning: DistilBERT en modo inferencia, embeddings + clasificador |
| **Fase 4** | MLOps (MLflow + DVC) + RAG (FAISS/ChromaDB) + demo funcional |
| **Fase 5** | Integración final, informe CRISP-DM, análisis ético y defensa |

## Estructura del repositorio

```
├── data/          # Dataset original y subsets muestreados
├── notebooks/     # EDA, experimentos, prototipos
├── src/           # Scripts modulares (entrenamiento, evaluación, inferencia)
├── models/        # Artefactos, índices vectoriales, logs de tracking
├── reports/       # Informe final, tablas, gráficos
├── app/           # Demo funcional del sistema de consulta
├── config/        # Configuración de experimentos y AutoML
├── README.md
└── REQUIREMENTS.txt
```

## Setup

```bash
git clone git@github.com:JesusRodrigoV/proyecto-integrador-ml-amazon-reviews.git
cd proyecto-integrador-ml-amazon-reviews
python -m venv .venv && source .venv/bin/activate
pip install -r REQUIREMENTS.txt
```

## Equipo

Grupo 2 — Proyecto Integrador de Machine Learning.
