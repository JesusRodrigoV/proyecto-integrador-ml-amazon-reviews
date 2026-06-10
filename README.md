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
├── notebooks/         # Notebooks por fase (estandarizados)
│   ├── f1_eda_definitivo.ipynb       # F1: EDA + Canonical Dataset generation
│   ├── f2a_modelado_clasico.ipynb    # F2: ML Clásico (LogReg, RF, LightGBM, XGB)
│   ├── f2b_automl_lazypredict.ipynb  # F2: AutoML Benchmark (LazyPredict)
│   ├── f3a_extraer_embeddings.ipynb  # F3: Embeddings generation (DistilBERT)
│   ├── f3b_distilbert.ipynb          # F3: LogisticRegression sobre embeddings
│   ├── f3c_clasicos.ipynb            # F3: RF + XGBoost sobre embeddings
│   └── f3d_lora_ensemble.ipynb      # F3: Fine-tuning LoRA + Stacking Ensemble
├── src/               # Scripts modulares (entrenamiento, evaluación, inferencia)
├── models/            # Artefactos, índices vectoriales, logs de tracking
├── reports/           # Métricas, gráficos y reportes unificados
├── app/               # Demo funcional del sistema de consulta
├── config/            # Configuración de experimentos y AutoML
├── docs/              # Documentos de propuesta e informes (gitignored)
├── README.md
└── REQUIREMENTS.txt
```

## Setup (Local)

```bash
git clone git@github.com:JesusRodrigoV/proyecto-integrador-ml-amazon-reviews.git
cd proyecto-integrador-ml-amazon-reviews
python -m venv .venv && source .venv/bin/activate
pip install -r REQUIREMENTS.txt
```

## Estado del Proyecto

- **Fase 1 (EDA & Prep)**: Finalizada ✅. Dataset canónico de 2.5M de filas generado en `data/office_products_balanced.parquet`.
- **Fase 2 (ML Clásico)**: Finalizada ✅. Baseline establecido con `RandomForest` y `LightGBM`.
- **Fase 3 (NLP & DL)**: En progreso ⏳. Implementación de `DistilBERT` (frozen & fine-tuned via LoRA) y Stacking Ensemble.
- **Fase 4 (MLOps & RAG)**: Pendiente 🔜.

## Lineamientos Críticos (Rubrica)

- **Variable Objetivo**: 3 clases (1-2 Negativo, 3 Neutro, 4-5 Positivo).
- **Tratamiento Imbalance**: `class_weights='balanced'` obligatorio en todos los modelos.
- **Limpieza**: Filtro de `word_count >= 5` para evitar ruido semántico.
- **Escalabilidad**: Uso de `Parquet` + `Polars` para manejo eficiente de 2.5M de registros.

## Equipo

Grupo 2 — Proyecto Integrador de Machine Learning.
