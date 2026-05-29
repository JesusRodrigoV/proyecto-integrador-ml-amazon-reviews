# Proyecto Integrador — ML sobre Amazon Reviews (Office Products)

**Grupo 2** · Auditoría de satisfacción + Sistema de recuperación semántica (RAG)

Dataset: [Amazon Reviews 2023](https://cseweb.ucsd.edu/~jmcauley/datasets.html) — categoría **Office Products**.

## Metodología

CRISP-DM — 5 fases progresivas:

| Fase | Contenido |
|------|-----------|
| **Fase 1** | Propuesta del proyecto (definición, justificación, métricas, cronograma) ✅ |
| **Fase 2** | ML Clásico: 3 modelos + AutoML (LazyPredict) + segmentación no supervisada |
| **Fase 3** | Deep Learning: DistilBERT en modo inferencia, embeddings + clasificador |
| **Fase 4** | MLOps (MLflow + DVC) + RAG (FAISS/ChromaDB) + demo funcional |
| **Fase 5** | Integración final, informe CRISP-DM, análisis ético y defensa |

## Estructura del repositorio

```
├── data/              # Dataset original y subsets muestreados
├── notebooks/         # Notebooks por fase
│   ├── f1_extraccion_muestreo.ipynb   # F1: Extracción + muestreo Bernoulli
│   ├── f1_eda_balanceo_clustering.ipynb # F1: EDA, balanceo, clustering
│   ├── f2_modelado_clasico.ipynb       # F2: Baseline + SVC + XGBoost
│   └── f2_automl_lazypredict.ipynb     # F2: AutoML benchmark (LazyPredict)
├── src/               # Scripts modulares (entrenamiento, evaluación, inferencia)
├── models/            # Artefactos, índices vectoriales, logs de tracking
├── reports/           # Informe final, tablas comparativas, gráficos interpretados
├── app/               # Demo funcional del sistema de consulta
├── config/            # Configuración de experimentos y AutoML
├── docs/              # Documentos de propuesta e informes (gitignored)
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

## Estado de Fase 2

Los notebooks de modelado clásico (`f2_modelado_clasico.ipynb`) y AutoML (`f2_automl_lazypredict.ipynb`) están implementados. Pendiente para próxima iteración:
- [ ] Tabla comparativa unificada con métricas F1-score macro
- [ ] Evaluación sobre la misma partición 80/20 estratificada
- [ ] Análisis de error por clase
- [ ] Integración con MLflow para tracking de experimentos

## Equipo

Grupo 2 — Proyecto Integrador de Machine Learning.
