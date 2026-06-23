# Proyecto Integrador — ML sobre Amazon Reviews (Office Products)

**Grupo 2** · Sentiment Analysis + Semantic Search (RAG)

Clasificación de sentimiento y búsqueda semántica sobre reseñas de Office Products del dataset [Amazon Reviews 2023](https://cseweb.ucsd.edu/~jmcauley/datasets.html) (McAuley Lab).

Materia: Machine Learning (2026-I) — Prof. Ovidio Paton.

## Tabla de Contenidos

- [Quick Start](#quick-start)
- [Prerequisitos](#prerequisitos)
- [Estructura del Proyecto](#estructura-del-proyecto)
- [Setup](#setup)
- [Flujo de Trabajo (orden de ejecución)](#flujo-de-trabajo)
- [Assets](#assets)
- [Resultados](#resultados)
- [Lineamientos de Rúbrica](#lineamientos-de-rúbrica)
- [Equipo](#equipo)

---

## Quick Start

Para tener la app funcionando en 2 minutos con Docker:

```bash
git clone git@github.com:JesusRodrigoV/proyecto-integrador-ml-amazon-reviews.git
cd proyecto-integrador-ml-amazon-reviews
docker compose up --build
```

Esto levanta:

- **MLflow** → `http://localhost:5000`
- **Streamlit** → `http://localhost:8501` (búsqueda semántica + clasificador + dashboard)

> Nota: la app requiere los assets (embeddings, clasificador, índice FAISS). Ver sección [Assets](#assets).

---

## Prerequisitos

- **Python 3.10+** (el proyecto usa 3.14, pero corre en 3.10+)
- **Git**
- **Docker + Docker Compose** (opcional, para el quick start)
- **Google Drive** (para los assets pesados que se generan en Colab)
- **Cuenta de Google** (para Google Colab, necesario solo si se regeneran los notebooks)

No se requiere GPU para los scripts Python del proyecto. Para regenerar los notebooks desde cero se necesita una GPU (gratis en Colab).

---

## Estructura del Proyecto

```
proyecto-integrador/
├── app/                  # Streamlit: search + predict + dashboard
├── config/               # params.yaml (rutas de assets, configuración)
├── data/                 # Dataset canónico en Parquet
├── docs/                 # Documentación y rúbrica
├── models/               # Assets: embeddings, clasificador, índice FAISS
├── notebooks/            # Notebooks ordenados por fase (F1 → F3)
│   ├── f1_eda_definitivo.ipynb
│   ├── f2a_modelado_clasico.ipynb
│   ├── f2b_automl_flaml.ipynb
│   ├── f3a_extraer_embeddings.ipynb
│   ├── f3b_distilbert.ipynb
│   ├── f3c_clasicos.ipynb
│   └── f3d_lora_ensemble.ipynb
├── reports/              # Métricas en JSON + gráficos generados
│   ├── metrics_fase2.json
│   ├── metrics_distilbert.json
│   └── figures/fase3/
├── scripts/              # Utilidades (generación de notebooks)
├── src/                  # Scripts reutilizables
│   ├── data_loader.py
│   ├── build_index.py
│   ├── pipeline.py
│   ├── predict.py
│   ├── search.py
│   ├── start_mlflow.py
│   ├── visualize.py
│   └── f3_generar_graficos.py
├── Dockerfile
├── docker-compose.yml
└── requirements.txt
```

---

## Setup

### Local

```bash
git clone git@github.com:JesusRodrigoV/proyecto-integrador-ml-amazon-reviews.git
cd proyecto-integrador-ml-amazon-reviews
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### Docker

```bash
docker compose up --build
```

Dos servicios:

| Servicio      | Puerto | URL                     |
| ------------- | ------ | ----------------------- |
| MLflow        | 5000   | <http://localhost:5000> |
| Streamlit App | 8501   | <http://localhost:8501> |

---

## Flujo de Trabajo

El proyecto sigue CRISP-DM. Cada fase tiene sus notebooks y scripts. Este es el orden de ejecución:

### Fase 1 — EDA y Dataset

Correr en Colab (o local):

1. **`notebooks/f1_eda_definitivo.ipynb`**
   - Carga el dataset original de Amazon Reviews (Office Products)
   - Hace EDA completo
   - Genera el dataset canónico balanceado (2.5M filas, Parquet)
   - Guarda `data/office_products_balanced.parquet`

### Fase 2 — ML Clásico

1. **`notebooks/f2a_modelado_clasico.ipynb`**
   - Entrena Logistic Regression, Random Forest, LightGBM, XGBoost sobre TF-IDF
   - Guarda métricas en `reports/metrics_fase2.json`

2. **`notebooks/f2b_automl_flaml.ipynb`**
   - AutoML con FLAML para comparar contra los modelos clásicos

### Fase 3 — Deep Learning (DistilBERT)

Estos notebooks se corren en Colab con GPU. Generan los embeddings y modelos:

1. **`notebooks/f3a_extraer_embeddings.ipynb`**
   - Carga el dataset balanceado
   - Extrae embeddings de DistilBERT (capa frozen, sin fine-tune)
   - Genera: `*_embeddings.npy`, `*_labels.npy`, `*_texts.pkl`, `*_eng_features.npy`, `scaler.pkl`
   - Exporta todo a Drive

2. **`notebooks/f3b_distilbert.ipynb`**
   - Carga los embeddings de F3-A
   - Entrena LogisticRegression como baseline
   - Genera `classifier.pkl`

3. **`notebooks/f3c_clasicos.ipynb`**
   - Random Forest y XGBoost sobre los embeddings de DistilBERT

4. **`notebooks/f3d_lora_ensemble.ipynb`**
   - Fine-tuning con LoRA + Stacking Ensemble

### Fase 4 — MLOps y RAG

Se corre local (no necesita GPU):

1. **`python src/pipeline.py`**
   - Carga embeddings → construye índice FAISS → registra en MLflow
   - Genera `models/faiss_index/index.faiss` y `id_map.pkl`

2. **`streamlit run app/app.py`**
   - Demo con 3 pestañas: búsqueda semántica, clasificador, dashboard

---

## Assets

Los archivos grandes (embeddings, modelos entrenados, índice FAISS) están en Google Drive en la carpeta `ML/proyecto_integrador/models/`.

**Lo más rápido es descargarlos de Drive** y copiarlos a `models/` en el repo local:

🔗 [Google Drive — ML/proyecto_integrador/models](https://drive.google.com/drive/folders/1ZhQlWwJLRLDzr7uZqDH-wXGJ2ONMLYF0?usp=sharing)

La estructura de `models/` debería quedar así:

```
models/
├── embeddings/
│   ├── train_embeddings.npy
│   ├── train_labels.npy
│   ├── train_texts.pkl
│   ├── val_embeddings.npy
│   ├── val_labels.npy
│   ├── val_texts.pkl
│   ├── test_embeddings.npy
│   ├── test_labels.npy
│   └── test_texts.pkl
├── faiss_index/
│   ├── index.faiss
│   └── id_map.pkl
└── classifier.pkl
```

Para generarlos desde cero se deben correr los notebooks en orden (F3-A primero, que genera los embeddings, y después F3-B para el clasificador). Luego ejecutar `src/pipeline.py` para construir el índice FAISS.

---

## Scripts Disponibles

| Comando                      | Qué hace                                   |
| ---------------------------- | ------------------------------------------ |
| `streamlit run app/app.py`   | Abre la demo web                           |
| `python src/pipeline.py`     | Pipeline F4: carga → FAISS → MLflow        |
| `python src/predict.py`      | Clasificador de sentimiento por terminal   |
| `python src/search.py`       | Búsqueda semántica por terminal            |
| `python src/visualize.py`    | Genera gráficos desde los JSON de métricas |
| `python src/start_mlflow.py` | Levanta MLflow server con túnel ngrok      |

---

## Resultados

### Fase 2 — Clásicos

| Modelo              | F1 Macro   | Accuracy |
| ------------------- | ---------- | -------- |
| Logistic Regression | **0.6575** | 0.6894   |
| LightGBM            | 0.6200     | 0.6519   |
| Random Forest       | 0.5492     | 0.6855   |
| XGBoost             | 0.3015     | 0.3269   |

### Fase 3 — DistilBERT

| Modelo                           | F1 Macro |
| -------------------------------- | -------- |
| Logistic Regression (baseline)   | 0.6705   |
| Random Forest                    | 0.6295   |
| XGBoost                          | 0.6592   |
| **DistilBERT + LoRA**            | **0.7380** |
| Stacking + Threshold Tuning      | **0.7391** |

---

## Lineamientos de Rúbrica

Estas reglas vienen de la rúbrica de la materia y hay que cumplirlas en todo el proyecto:

- **Variable objetivo**: 3 clases (1-2 → Negativo, 3 → Neutro, 4-5 → Positivo)
- **Manejo de imbalance**: usar `class_weights='balanced'` en todos los modelos
- **Limpieza**: filtrar reseñas con menos de 5 palabras (`word_count >= 5`)
- **Formato de datos**: Parquet + Polars (para manejar 2.5M filas eficientemente)
- **Dataset balanceado**: 500K por rating (1M Neg / 500K Neu / 1M Pos)

---

## Equipo

**Grupo 2** — ML 2026-I

| Nombre                |     |
| --------------------- | --- |
| Jesus Rodrigo Velasco |     |
| Samuel Villca         |     |
| Ana Carolina Zeballos |     |
| Manuel Franco Jimenez |     |
