#!/usr/bin/env python3
"""Generate the 4 F3 notebooks for the redesigned pipeline."""

import nbformat as nbf
import os

NB_DIR = "/home/jr/work/ml/proyecto-integrador/notebooks"

MARKDOWN_TEMPLATES = True

def md(source):
    flat = "".join(source) if isinstance(source, list) else source
    return nbf.v4.new_markdown_cell(flat)

def code(source, hidden=False):
    flat = "".join(source) if isinstance(source, list) else source
    cell = nbf.v4.new_code_cell(flat)
    if hidden:
        cell.metadata["jupyter"] = {"source_hidden": True}
    return cell

# ──────────────────────────────────────────────
# F3-A: extract_embeddings
# ──────────────────────────────────────────────
def build_f3_a():
    cells = []

    cells.append(md([
        "# F3-A — Extracción de Embeddings (DistilBERT frozen)\n",
        "\n",
        "**Objetivo**: Cargar el dataset balanceado, samplear 200k, extraer embeddings frozen de DistilBERT ",
        "y engineered features. Este notebook es el **único** que extrae embeddings — los notebooks F3-B, F3-C y F3-D ",
        "cargan los `.npy` generados aquí.\n",
        "\n",
        "**Salidas en Drive**: `embeddings/train/val/test_embeddings.npy`, `embeddings/train/val/test_eng_features.npy`, ",
        "`embeddings/train/val/test_labels.npy`, `embeddings/train/val/test_texts.pkl`, scaler.pkl\n",
        "\n",
        "**Tiempo estimado**: ~40 min (GPU T4)\n"
    ]))

    cells.append(md([
        "## 1. Instalar dependencias\n"
    ]))

    cells.append(code([
        "!pip install -q polars mlflow transformers umap-learn -U\n"
    ]))

    cells.append(code([
        "import polars as pl\n",
        "import numpy as np\n",
        "import torch\n",
        "import gc\n",
        "import os\n",
        "import json\n",
        "import pickle\n",
        "import time\n",
        "from google.colab import drive\n",
        "from sklearn.model_selection import train_test_split\n",
        "from sklearn.preprocessing import StandardScaler\n",
        "from transformers import AutoTokenizer, AutoModel\n",
        "from tqdm.notebook import tqdm\n"
    ]))

    cells.append(md([
        "## 2. Montar Google Drive\n",
        "\n",
        "Montamos Drive para leer el parquet del EDA y persistir los embeddings.\n"
    ]))

    cells.append(code([
        "drive.mount('/content/drive')\n",
        "\n",
        "DRIVE_BASE = \"/content/drive/MyDrive/ML/proyecto_integrador\"\n",
        "DATA_DIR = f\"{DRIVE_BASE}/data\"\n",
        "PARQUET_PATH = f\"{DATA_DIR}/office_products_balanced.parquet\"\n",
        "EMB_DIR = f\"{DRIVE_BASE}/embeddings\"\n",
        "REPORTS_DIR = f\"{DRIVE_BASE}/reports\"\n",
        "\n",
        "print(f\"Parquet: {PARQUET_PATH}\")\n",
        "print(f\"Embs:    {EMB_DIR}\")\n",
        "print(f\"Reports: {REPORTS_DIR}\")\n",
        "\n",
        "for d in [DATA_DIR, EMB_DIR, REPORTS_DIR]:\n",
        "    os.makedirs(d, exist_ok=True)\n"
    ]))

    cells.append(md([
        "## 3. Cargar datos y muestreo estratificado\n",
        "\n",
        "Cargamos el dataset balanceado (2.5M filas, 3 clases). Sampleamos 200k estratificado ",
        "manteniendo proporción entre clases. Semilla fija 42 para reproducibilidad.\n"
    ]))

    cells.append(code([
        "SAMPLE_SIZE = 200_000\n",
        "BATCH_SIZE = 256\n",
        "MAX_LENGTH = 128\n",
        "RANDOM_STATE = 42\n",
        "\n",
        "ENG_FEATURES = [\n",
        "    'mayusculas_count', 'char_total', 'exclamacion_count',\n",
        "    'interrogacion_count', 'porcentaje_mayusculas',\n",
        "    'puntuacion_emocional', 'total_tokens', 'unique_types', 'ttr'\n",
        "]\n",
        "\n",
        "df = pl.read_parquet(PARQUET_PATH)\n",
        "dfs = []\n",
        "for s in [0, 1, 2]:\n",
        "    sub = df.filter(pl.col('sentiment') == s)\n",
        "    n = min(SAMPLE_SIZE // 3, sub.shape[0])\n",
        "    dfs.append(sub.sample(n=n, seed=RANDOM_STATE))\n",
        "\n",
        "df_sample = pl.concat(dfs).sample(fraction=1.0, seed=RANDOM_STATE)\n",
        "print(f\"Sample: {df_sample.shape}\")\n",
        "print(df_sample['sentiment'].value_counts().sort('sentiment'))\n"
    ]))

    cells.append(md([
        "## 4. Train/Val/Test split\n",
        "\n",
        "70% entrenamiento, 15% validacion, 15% prueba. Split estratificado.\n"
    ]))

    cells.append(code([
        "texts = df_sample['text'].to_list()\n",
        "labels = df_sample['sentiment'].to_numpy()\n",
        "\n",
        "X_temp, X_test, y_temp, y_test = train_test_split(\n",
        "    texts, labels, test_size=0.15, random_state=RANDOM_STATE, stratify=labels\n",
        ")\n",
        "X_train, X_val, y_train, y_val = train_test_split(\n",
        "    X_temp, y_temp, test_size=round(0.15/0.85, 3),\n",
        "    random_state=RANDOM_STATE, stratify=y_temp\n",
        ")\n",
        "\n",
        "print(f\"Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}\")\n"
    ]))

    cells.append(md([
        "## 5. Extraer embeddings frozen con DistilBERT\n",
        "\n",
        "DistilBERT en modo inference (congelado). Extraemos embedding [CLS] de 768d por reseña. ",
        "Si ya existen en Drive, los carga automáticamente.\n"
    ]))

    cells.append(code([
        "MODEL_NAME = \"distilbert-base-uncased\"\n",
        "tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)\n",
        "model = AutoModel.from_pretrained(MODEL_NAME)\n",
        "device = torch.device(\"cuda\" if torch.cuda.is_available() else \"cpu\")\n",
        "model = model.to(device)\n",
        "model.eval()\n",
        "print(f\"Device: {device}\")\n",
        "\n",
        "def extract_embeddings(texts, model, tokenizer, batch_size=BATCH_SIZE, max_length=MAX_LENGTH):\n",
        "    all_embeddings = []\n",
        "    for i in tqdm(range(0, len(texts), batch_size)):\n",
        "        batch_texts = texts[i:i+batch_size]\n",
        "        encoded = tokenizer(\n",
        "            batch_texts, padding=True, truncation=True,\n",
        "            max_length=max_length, return_tensors='pt'\n",
        "        ).to(device)\n",
        "        with torch.no_grad():\n",
        "            outputs = model(**encoded)\n",
        "        embeddings = outputs.last_hidden_state[:, 0, :].cpu().numpy()\n",
        "        all_embeddings.append(embeddings)\n",
        "        del encoded, outputs\n",
        "        if i % (batch_size * 10) == 0:\n",
        "            gc.collect()\n",
        "    return np.vstack(all_embeddings)\n"
    ]))

    cells.append(code([
        "emb_paths = {\n",
        "    'train': f\"{EMB_DIR}/train_embeddings.npy\",\n",
        "    'val':   f\"{EMB_DIR}/val_embeddings.npy\",\n",
        "    'test':  f\"{EMB_DIR}/test_embeddings.npy\",\n",
        "}\n",
        "\n",
        "if all(os.path.exists(p) for p in emb_paths.values()):\n",
        "    print(\"Cargando embeddings existentes...\")\n",
        "    X_train_emb = np.load(emb_paths['train'])\n",
        "    X_val_emb   = np.load(emb_paths['val'])\n",
        "    X_test_emb  = np.load(emb_paths['test'])\n",
        "    print(f\"Train: {X_train_emb.shape}, Val: {X_val_emb.shape}, Test: {X_test_emb.shape}\")\n",
        "else:\n",
        "    print(\"Extrayendo embeddings TRAIN...\")\n",
        "    X_train_emb = extract_embeddings(X_train, model, tokenizer)\n",
        "    print(f\"Train: {X_train_emb.shape}\")\n",
        "    print(\"Extrayendo embeddings VAL...\")\n",
        "    X_val_emb = extract_embeddings(X_val, model, tokenizer)\n",
        "    print(f\"Val: {X_val_emb.shape}\")\n",
        "    print(\"Extrayendo embeddings TEST...\")\n",
        "    X_test_emb = extract_embeddings(X_test, model, tokenizer)\n",
        "    print(f\"Test: {X_test_emb.shape}\")\n",
        "    for k in emb_paths:\n",
        "        np.save(emb_paths[k], locals()[f'X_{k}_emb'])\n",
        "    print(\"Embeddings guardados en Drive\")\n"
    ]))

    cells.append(md([
        "## 6. Feature engineering + scaling\n",
        "\n",
        "Extraemos las 9 features lingüísticas del EDA y las normalizamos con Z-score. ",
        "Se concatenarán con los embeddings en los notebooks de modelado.\n"
    ]))

    cells.append(code([
        "eng_paths = {\n",
        "    'train': f\"{EMB_DIR}/train_eng_features.npy\",\n",
        "    'val':   f\"{EMB_DIR}/val_eng_features.npy\",\n",
        "    'test':  f\"{EMB_DIR}/test_eng_features.npy\",\n",
        "}\n",
        "\n",
        "if all(os.path.exists(p) for p in eng_paths.values()):\n",
        "    print(\"Feature engineering ya completado, cargando...\")\n",
        "    eng_train = np.load(eng_paths['train'])\n",
        "    eng_val   = np.load(eng_paths['val'])\n",
        "    eng_test  = np.load(eng_paths['test'])\n",
        "else:\n",
        "    print(\"Computando engineered features...\")\n",
        "    eng_train = df_sample.filter(pl.col('text').is_in(X_train)).select(ENG_FEATURES).to_numpy()\n",
        "    eng_val   = df_sample.filter(pl.col('text').is_in(X_val)).select(ENG_FEATURES).to_numpy()\n",
        "    eng_test  = df_sample.filter(pl.col('text').is_in(X_test)).select(ENG_FEATURES).to_numpy()\n",
        "\n",
        "    scaler = StandardScaler()\n",
        "    eng_train = scaler.fit_transform(eng_train)\n",
        "    eng_val   = scaler.transform(eng_val)\n",
        "    eng_test  = scaler.transform(eng_test)\n",
        "\n",
        "    for k in eng_paths:\n",
        "        np.save(eng_paths[k], locals()[f'eng_{k}'])\n",
        "    # Also save scaler\n",
        "    import joblib\n",
        "    joblib.dump(scaler, f\"{EMB_DIR}/scaler.pkl\")\n",
        "    print(\"Engineered features guardadas en Drive\")\n",
        "\n",
        "print(f\"eng_train: {eng_train.shape}, eng_val: {eng_val.shape}, eng_test: {eng_test.shape}\")\n"
    ]))

    cells.append(md([
        "## 7. Guardar metadatos (textos, labels) para notebooks downstream\n",
        "\n",
        "Guardamos los textos crudos (necesarios para LoRA) y las etiquetas en .pkl/.npy\n"
    ]))

    cells.append(code([
        "label_paths = {\n",
        "    'train': f\"{EMB_DIR}/train_labels.npy\",\n",
        "    'val':   f\"{EMB_DIR}/val_labels.npy\",\n",
        "    'test':  f\"{EMB_DIR}/test_labels.npy\",\n",
        "}\n",
        "for k in label_paths:\n",
        "    np.save(label_paths[k], locals()[f'y_{k}'])\n",
        "\n",
        "# Guardar textos crudos para LoRA (necesitan el texto original)\n",
        "text_paths = {\n",
        "    'train': f\"{EMB_DIR}/train_texts.pkl\",\n",
        "    'val':   f\"{EMB_DIR}/val_texts.pkl\",\n",
        "    'test':  f\"{EMB_DIR}/test_texts.pkl\",\n",
        "}\n",
        "for k in text_paths:\n",
        "    with open(text_paths[k], 'wb') as f:\n",
        "        pickle.dump(locals()[f'X_{k}'], f)\n",
        "\n",
        "print(\"Labels y textos guardados en Drive\")\n"
    ]))

    cells.append(md([
        "## 8. Guardar embeddings para F4 (RAG)\n",
        "\n",
        "Sample estratificado de 10k embeddings para F4.\n"
    ]))

    cells.append(code([
        "EMBEDDINGS_PATH = f\"{EMB_DIR}/distilbert_embeddings_sample.npy\"\n",
        "LABELS_PATH = f\"{EMB_DIR}/distilbert_labels_sample.npy\"\n",
        "\n",
        "emb_sample_size = 10_000\n",
        "n_per_class = emb_sample_size // 3\n",
        "rng = np.random.RandomState(RANDOM_STATE)\n",
        "emb_chunks, label_chunks = [], []\n",
        "for s in [0, 1, 2]:\n",
        "    idx = np.where(y_test == s)[0]\n",
        "    n = min(n_per_class, len(idx))\n",
        "    chosen = rng.choice(idx, n, replace=False)\n",
        "    emb_chunks.append(X_test_emb[chosen])\n",
        "    label_chunks.append(y_test[chosen])\n",
        "\n",
        "emb_sample = np.concatenate(emb_chunks)\n",
        "label_sample = np.concatenate(label_chunks)\n",
        "\n",
        "np.save(EMBEDDINGS_PATH, emb_sample)\n",
        "np.save(LABELS_PATH, label_sample)\n",
        "print(f\"Saved {len(emb_sample)} embeddings para F4, shape: {emb_sample.shape}\")\n"
    ]))

    cells.append(code([
        "# Liberar memoria\n",
        "del df, df_sample, model, tokenizer\n",
        "gc.collect()\n",
        "if torch.cuda.is_available():\n",
        "    torch.cuda.empty_cache()\n",
        "print(\"\\nF3-A completado. Ahora ejecute F3-B (clásicos), F3-C (baseline) y F3-D (LoRA+ensemble).\")\n"
    ]))

    nb = nbf.v4.new_notebook(
        metadata={
            "kernelspec": {"name": "python3", "display_name": "Python 3"},
            "language_info": {"name": "python"},
            "accelerator": "GPU",
            "colab": {"provenance": [], "gpuType": "T4"}
        },
        cells=cells
    )
    path = os.path.join(NB_DIR, "f3_extraer_embeddings.ipynb")
    with open(path, "w") as f:
        nbf.write(nb, f)
    print(f"  ✓ {path}")


# ──────────────────────────────────────────────
# F3-B: clasicos (RF + XGBoost)
# ──────────────────────────────────────────────
def build_f3_b():
    cells = []

    cells.append(md([
        "# F3-B — Modelos Clásicos: Random Forest + XGBoost\n",
        "\n",
        "**Objetivo**: Entrenar Random Forest y XGBoost sobre embeddings + engineered features ",
        "extraídos en F3-A. Notebook puramente CPU.\n",
        "\n",
        "**Tiempo estimado**: ~60 min (CPU)\n"
    ]))

    cells.append(md(["## 1. Instalar dependencias\n"]))
    cells.append(code([
        "!pip install -q polars mlflow xgboost -U\n"
    ]))

    cells.append(code([
        "import polars as pl\n",
        "import numpy as np\n",
        "import gc\n",
        "import os\n",
        "import json\n",
        "import time\n",
        "import mlflow\n",
        "import matplotlib.pyplot as plt\n",
        "import seaborn as sns\n",
        "from google.colab import drive\n",
        "from sklearn.model_selection import ParameterGrid\n",
        "from sklearn.ensemble import RandomForestClassifier\n",
        "from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score\n",
        "from xgboost import XGBClassifier\n"
    ]))

    cells.append(md(["## 2. Montar Google Drive y cargar embeddings\n"]))
    cells.append(code([
        "drive.mount('/content/drive')\n",
        "\n",
        "DRIVE_BASE = \"/content/drive/MyDrive/ML/proyecto_integrador\"\n",
        "EMB_DIR = f\"{DRIVE_BASE}/embeddings\"\n",
        "REPORTS_DIR = f\"{DRIVE_BASE}/reports\"\n",
        "RANDOM_STATE = 42\n",
        "\n",
        "for d in [REPORTS_DIR]:\n",
        "    os.makedirs(d, exist_ok=True)\n",
        "\n",
        "print(\"Cargando embeddings y features...\")\n",
        "X_train_emb = np.load(f\"{EMB_DIR}/train_embeddings.npy\")\n",
        "X_val_emb   = np.load(f\"{EMB_DIR}/val_embeddings.npy\")\n",
        "X_test_emb  = np.load(f\"{EMB_DIR}/test_embeddings.npy\")\n",
        "\n",
        "eng_train = np.load(f\"{EMB_DIR}/train_eng_features.npy\")\n",
        "eng_val   = np.load(f\"{EMB_DIR}/val_eng_features.npy\")\n",
        "eng_test  = np.load(f\"{EMB_DIR}/test_eng_features.npy\")\n",
        "\n",
        "y_train = np.load(f\"{EMB_DIR}/train_labels.npy\")\n",
        "y_val   = np.load(f\"{EMB_DIR}/val_labels.npy\")\n",
        "y_test  = np.load(f\"{EMB_DIR}/test_labels.npy\")\n",
        "\n",
        "# Concatenar embeddings + engineered features\n",
        "X_train = np.concatenate([X_train_emb, eng_train], axis=1)\n",
        "X_val   = np.concatenate([X_val_emb, eng_val], axis=1)\n",
        "X_test  = np.concatenate([X_test_emb, eng_test], axis=1)\n",
        "\n",
        "ENG_FEATURE_NAMES = [\n",
        "    'mayusculas_count', 'char_total', 'exclamacion_count',\n",
        "    'interrogacion_count', 'porcentaje_mayusculas',\n",
        "    'puntuacion_emocional', 'total_tokens', 'unique_types', 'ttr'\n",
        "]\n",
        "\n",
        "print(f\"Train: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape}\")\n"
    ]))

    cells.append(md([
        "## 3. Helper: evaluación y registro\n",
        "\n",
        "Función que calcula métricas por clase y globales, y las registra en MLflow.\n"
    ]))

    cells.append(code([
        "results = []\n",
        "\n",
        "def eval_and_record(name, y_true, y_pred, training_time):\n",
        "    from sklearn.metrics import precision_recall_fscore_support, confusion_matrix\n",
        "    p, r, f, _ = precision_recall_fscore_support(y_true, y_pred, labels=[0, 1, 2])\n",
        "    per_class = {\n",
        "        label: {'precision': round(p[i], 4), 'recall': round(r[i], 4), 'f1': round(f[i], 4)}\n",
        "        for i, label in enumerate(['Negativo', 'Neutro', 'Positivo'])\n",
        "    }\n",
        "    metrics = {\n",
        "        'model_name': name,\n",
        "        'training_time_seconds': round(training_time, 2),\n",
        "        'f1_macro': round(f1_score(y_true, y_pred, average='macro'), 4),\n",
        "        'precision_macro': round(precision_score(y_true, y_pred, average='macro'), 4),\n",
        "        'recall_macro': round(recall_score(y_true, y_pred, average='macro'), 4),\n",
        "        'accuracy': round(accuracy_score(y_true, y_pred), 4),\n",
        "        'per_class': per_class,\n",
        "        'confusion_matrix': confusion_matrix(y_true, y_pred).tolist(),\n",
        "    }\n",
        "    results.append(metrics)\n",
        "    return metrics, y_pred\n"
    ]))

    cells.append(md([
        "## 4. Random Forest\n",
        "\n",
        "Grid search simple sobre `min_samples_leaf` con validación.\n"
    ]))

    cells.append(code([
        "print(\"=== Random Forest ===\")\n",
        "rf_params = {'n_estimators': [200], 'max_depth': [None], 'min_samples_leaf': [1, 4]}\n",
        "best_f1 = -1\n",
        "best_rf = None\n",
        "best_elapsed = None\n",
        "\n",
        "for params in ParameterGrid(rf_params):\n",
        "    start = time.time()\n",
        "    rf = RandomForestClassifier(**params, random_state=RANDOM_STATE, n_jobs=-1)\n",
        "    rf.fit(X_train, y_train)\n",
        "    elapsed = time.time() - start\n",
        "    val_pred = rf.predict(X_val)\n",
        "    f1 = f1_score(y_val, val_pred, average='macro')\n",
        "    print(f\"  params={params} -> val_f1={f1:.4f} ({elapsed:.0f}s)\")\n",
        "    if f1 > best_f1:\n",
        "        best_f1 = f1\n",
        "        best_rf = rf\n",
        "        best_elapsed = elapsed\n",
        "\n",
        "rf = best_rf\n",
        "y_pred_rf = rf.predict(X_test)\n",
        "rf_metrics, _ = eval_and_record('Random Forest', y_test, y_pred_rf, best_elapsed)\n",
        "print(f\"RF test F1-macro: {rf_metrics['f1_macro']}\")\n"
    ]))

    cells.append(md([
        "## 5. Feature importance — Random Forest\n"
    ]))

    cells.append(code([
        "rf_feat_imp = rf.feature_importances_\n",
        "emb_importance = rf_feat_imp[:768].sum()\n",
        "eng_importance = rf_feat_imp[768:]\n",
        "\n",
        "labels_bar = ['Embeddings (768d)'] + ENG_FEATURE_NAMES\n",
        "values = [emb_importance] + list(eng_importance)\n",
        "\n",
        "plt.figure(figsize=(10, 6))\n",
        "colors = ['#3498db'] + ['#2ecc71'] * len(ENG_FEATURE_NAMES)\n",
        "plt.barh(range(len(values)), values, color=colors, edgecolor='white')\n",
        "plt.yticks(range(len(values)), labels_bar)\n",
        "plt.xlabel('Importancia (RF)')\n",
        "plt.title('Feature Importance: Embeddings vs Engineered (Random Forest)')\n",
        "plt.tight_layout()\n",
        "plt.show()\n"
    ]))

    cells.append(md(["## 6. XGBoost\n"]))
    cells.append(code([
        "print(\"=== XGBoost ===\")\n",
        "xgb_params = {\n",
        "    'n_estimators': 300,\n",
        "    'max_depth': 6,\n",
        "    'learning_rate': 0.1,\n",
        "    'subsample': 0.8,\n",
        "    'colsample_bytree': 0.8,\n",
        "    'eval_metric': 'mlogloss',\n",
        "    'random_state': RANDOM_STATE,\n",
        "    'early_stopping_rounds': 10,\n",
        "}\n",
        "\n",
        "start = time.time()\n",
        "xgb = XGBClassifier(**xgb_params)\n",
        "xgb.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)\n",
        "elapsed = time.time() - start\n",
        "y_pred_xgb = xgb.predict(X_test)\n",
        "xgb_metrics, _ = eval_and_record('XGBoost', y_test, y_pred_xgb, elapsed)\n",
        "print(f\"XGBoost test F1-macro: {xgb_metrics['f1_macro']} ({elapsed:.0f}s)\")\n"
    ]))

    cells.append(md(["## 7. Feature importance — XGBoost\n"]))
    cells.append(code([
        "xgb_feat_imp = xgb.feature_importances_\n",
        "emb_importance = xgb_feat_imp[:768].sum()\n",
        "eng_importance = xgb_feat_imp[768:]\n",
        "\n",
        "labels_bar = ['Embeddings (768d)'] + ENG_FEATURE_NAMES\n",
        "values = [emb_importance] + list(eng_importance)\n",
        "\n",
        "plt.figure(figsize=(10, 6))\n",
        "colors = ['#e74c3c'] + ['#f39c12'] * len(ENG_FEATURE_NAMES)\n",
        "plt.barh(range(len(values)), values, color=colors, edgecolor='white')\n",
        "plt.yticks(range(len(values)), labels_bar)\n",
        "plt.xlabel('Importancia (XGBoost)')\n",
        "plt.title('Feature Importance: Embeddings vs Engineered (XGBoost)')\n",
        "plt.tight_layout()\n",
        "plt.show()\n"
    ]))

    cells.append(md(["## 8. MLflow Tracking\n"]))
    cells.append(code([
        "MLFLOW_TRACKING_URI = os.getenv(\"MLFLOW_TRACKING_URI\", \"https://humorous-trusting-domelike.ngrok-free.dev\")\n",
        "import requests\n",
        "try:\n",
        "    r = requests.get(f\"{MLFLOW_TRACKING_URI}/api/2.0/mlflow/experiments/list\", timeout=5)\n",
        "    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)\n",
        "    print(f\"MLflow OK via {MLFLOW_TRACKING_URI}\")\n",
        "except Exception as e:\n",
        "    print(f\"MLflow no disponible: {e}, fallback a SQLite\")\n",
        "    mlflow.set_tracking_uri(f\"sqlite:///{DRIVE_BASE}/mlflow_fallback.db\")\n",
        "\n",
        "mlflow.set_experiment(\"distilbert_clasicos\")\n",
        "\n",
        "for r in results:\n",
        "    with mlflow.start_run(run_name=r['model_name']):\n",
        "        mlflow.log_params({'model_name': r['model_name']})\n",
        "        mlflow.log_metrics({\n",
        "            'f1_macro': r['f1_macro'],\n",
        "            'accuracy': r['accuracy'],\n",
        "            'training_time_seconds': r['training_time_seconds'],\n",
        "        })\n",
        "        mlflow.log_dict(r['confusion_matrix'], f\"{r['model_name']}_confusion_matrix.json\")\n",
        "\n",
        "print(\"MLflow tracking completado\")\n"
    ]))

    cells.append(md(["## 9. Exportar métricas a JSON\n"]))
    cells.append(code([
        "report_path = f\"{REPORTS_DIR}/metrics_clasicos.json\"\n",
        "with open(report_path, 'w') as f:\n",
        "    json.dump(results, f, indent=2)\n",
        "print(f\"Exportado: {report_path}\")\n"
    ]))

    cells.append(md(["## 10. Guardar predicciones para Ensemble\n"]))
    cells.append(code([
        "PREDS_DIR = f\"{DRIVE_BASE}/preds\"\n",
        "os.makedirs(PREDS_DIR, exist_ok=True)\n",
        "np.save(f\"{PREDS_DIR}/y_pred_rf.npy\", y_pred_rf)\n",
        "np.save(f\"{PREDS_DIR}/y_pred_xgb.npy\", y_pred_xgb)\n",
        "\n",
        "part1_results = [r for r in results if r['model_name'] in ['Random Forest', 'XGBoost']]\n",
        "with open(f\"{PREDS_DIR}/part1_results.json\", 'w') as f:\n",
        "    json.dump(part1_results, f, indent=2)\n",
        "print(\"Predicciones guardadas para F3-D\")\n"
    ]))

    cells.append(code([
        "# Liberar memoria\n",
        "del X_train, X_val, X_test, X_train_emb, X_val_emb, X_test_emb\n",
        "del eng_train, eng_val, eng_test\n",
        "gc.collect()\n",
        "print(\"\\nF3-B completado. Puede ejecutar F3-D para el ensemble.\")\n"
    ]))

    nb = nbf.v4.new_notebook(
        metadata={
            "kernelspec": {"name": "python3", "display_name": "Python 3"},
            "language_info": {"name": "python"},
        },
        cells=cells
    )
    path = os.path.join(NB_DIR, "f3_clasicos.ipynb")
    with open(path, "w") as f:
        nbf.write(nb, f)
    print(f"  ✓ {path}")


# ──────────────────────────────────────────────
# F3-C: distilbert (modified) — LogReg + viz
# ──────────────────────────────────────────────
def build_f3_c():
    cells = []

    cells.append(md([
        "# F3-C — Baseline: LogisticRegression sobre embeddings DistilBERT\n",
        "\n",
        "**Objetivo**: Cargar embeddings frozen de F3-A y entrenar una regresión logística como baseline. ",
        "Incluye visualizaciones (t-SNE, UMAP, K-Means, similitud coseno).\n",
        "\n",
        "**Tiempo estimado**: ~5 min (CPU)\n"
    ]))

    cells.append(md(["## 1. Instalar dependencias\n"]))
    cells.append(code([
        "!pip install -q mlflow umap-learn\n"
    ]))

    cells.append(code([
        "import numpy as np\n",
        "import os\n",
        "import json\n",
        "import time\n",
        "import mlflow\n",
        "import matplotlib.pyplot as plt\n",
        "import seaborn as sns\n",
        "from google.colab import drive\n",
        "from sklearn.linear_model import LogisticRegression\n",
        "from sklearn.metrics import (f1_score, precision_score, recall_score,\n",
        "                             accuracy_score, confusion_matrix,\n",
        "                             classification_report)\n",
        "from sklearn.manifold import TSNE\n",
        "from sklearn.cluster import KMeans\n",
        "from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score\n",
        "from sklearn.metrics.pairwise import cosine_similarity\n",
        "import umap\n",
        "import joblib\n",
        "import gc\n"
    ]))

    cells.append(md(["## 2. Montar Google Drive y cargar embeddings\n"]))
    cells.append(code([
        "drive.mount('/content/drive')\n",
        "\n",
        "DRIVE_BASE = \"/content/drive/MyDrive/ML/proyecto_integrador\"\n",
        "EMB_DIR = f\"{DRIVE_BASE}/embeddings\"\n",
        "REPORTS_DIR = f\"{DRIVE_BASE}/reports\"\n",
        "RANDOM_STATE = 42\n",
        "\n",
        "for d in [REPORTS_DIR]:\n",
        "    os.makedirs(d, exist_ok=True)\n",
        "\n",
        "print(\"Cargando embeddings desde F3-A...\")\n",
        "X_train_emb = np.load(f\"{EMB_DIR}/train_embeddings.npy\")\n",
        "X_val_emb   = np.load(f\"{EMB_DIR}/val_embeddings.npy\")\n",
        "X_test_emb  = np.load(f\"{EMB_DIR}/test_embeddings.npy\")\n",
        "y_train = np.load(f\"{EMB_DIR}/train_labels.npy\")\n",
        "y_val   = np.load(f\"{EMB_DIR}/val_labels.npy\")\n",
        "y_test  = np.load(f\"{EMB_DIR}/test_labels.npy\")\n",
        "\n",
        "print(f\"Train embeddings: {X_train_emb.shape}\")\n",
        "print(f\"Val embeddings:   {X_val_emb.shape}\")\n",
        "print(f\"Test embeddings:  {X_test_emb.shape}\")\n"
    ]))

    cells.append(md([
        "## 3. Logistic Regression (baseline)\n",
        "\n",
        "Regresión logística sobre embeddings de 768d. Baseline rápido.\n"
    ]))

    cells.append(code([
        "clf = LogisticRegression(max_iter=1000, random_state=RANDOM_STATE, n_jobs=-1)\n",
        "\n",
        "start = time.time()\n",
        "clf.fit(X_train_emb, y_train)\n",
        "training_time = time.time() - start\n",
        "print(f\"Entrenado en {training_time:.1f}s\")\n",
        "\n",
        "y_pred = clf.predict(X_test_emb)\n",
        "\n",
        "report = classification_report(y_test, y_pred,\n",
        "                               target_names=['Negativo', 'Neutro', 'Positivo'],\n",
        "                               output_dict=True)\n",
        "per_class = {cl: report[cl] for cl in ['Negativo', 'Neutro', 'Positivo']}\n",
        "\n",
        "metrics = {\n",
        "    'model_name': 'DistilBERT + LogisticRegression',\n",
        "    'model_type': 'distilbert_base_uncased + logreg',\n",
        "    'sample_size': 200_000,\n",
        "    'embedding_dim': 768,\n",
        "    'training_time_seconds': round(training_time, 2),\n",
        "    'f1_macro': round(report['macro avg']['f1-score'], 4),\n",
        "    'precision_macro': round(report['macro avg']['precision'], 4),\n",
        "    'recall_macro': round(report['macro avg']['recall'], 4),\n",
        "    'accuracy': round(accuracy_score(y_test, y_pred), 4),\n",
        "    'class_labels': ['Negativo', 'Neutro', 'Positivo'],\n",
        "    'confusion_matrix': confusion_matrix(y_test, y_pred).tolist(),\n",
        "    'per_class': per_class,\n",
        "}\n",
        "\n",
        "for k, v in metrics.items():\n",
        "    if k not in ('confusion_matrix', 'per_class'):\n",
        "        print(f\"  {k}: {v}\")\n"
    ]))

    cells.append(md(["## 4. MLflow Tracking\n"]))
    cells.append(code([
        "MLFLOW_TRACKING_URI = os.getenv(\"MLFLOW_TRACKING_URI\", \"https://humorous-trusting-domelike.ngrok-free.dev\")\n",
        "import requests\n",
        "try:\n",
        "    r = requests.get(f\"{MLFLOW_TRACKING_URI}/api/2.0/mlflow/experiments/list\", timeout=5)\n",
        "    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)\n",
        "    print(f\"MLflow OK via {MLFLOW_TRACKING_URI}\")\n",
        "except Exception as e:\n",
        "    print(f\"MLflow no disponible: {e}, fallback a SQLite\")\n",
        "    mlflow.set_tracking_uri(f\"sqlite:///{DRIVE_BASE}/mlflow_fallback.db\")\n",
        "\n",
        "mlflow.set_experiment(\"distilbert_baseline\")\n",
        "\n",
        "with mlflow.start_run():\n",
        "    mlflow.log_params({\n",
        "        \"model_name\": \"DistilBERT + LogisticRegression\",\n",
        "        \"sample_size\": 200_000,\n",
        "        \"embedding_dim\": 768,\n",
        "    })\n",
        "    for label, scores in metrics['per_class'].items():\n",
        "        mlflow.log_metric(f'f1_{label.lower()}', scores['f1-score'])\n",
        "    mlflow.log_metrics({\n",
        "        'f1_macro': metrics['f1_macro'],\n",
        "        'accuracy': metrics['accuracy'],\n",
        "        'training_time_seconds': metrics['training_time_seconds'],\n",
        "    })\n",
        "    MODEL_PATH = f\"{EMB_DIR}/classifier.pkl\"\n",
        "    joblib.dump(clf, MODEL_PATH)\n",
        "    mlflow.log_artifact(MODEL_PATH, artifact_path=\"models\")\n",
        "    print(f\"MLflow run ID: {mlflow.active_run().info.run_id}\")\n"
    ]))

    cells.append(md(["## 5. Exportar métricas a JSON\n"]))
    cells.append(code([
        "report_path = f\"{REPORTS_DIR}/metrics_distilbert.json\"\n",
        "with open(report_path, 'w') as f:\n",
        "    json.dump(metrics, f, indent=2)\n",
        "print(f\"Exportado: {report_path}\")\n"
    ]))

    cells.append(md(["## 6. Guardar embeddings para F4 (RAG)\n"]))
    cells.append(code([
        "EMBEDDINGS_PATH = f\"{EMB_DIR}/distilbert_embeddings_sample.npy\"\n",
        "LABELS_PATH = f\"{EMB_DIR}/distilbert_labels_sample.npy\"\n",
        "\n",
        "# Si F3-A ya los guardó, verificar\n",
        "if os.path.exists(EMBEDDINGS_PATH):\n",
        "    print(\"Embeddings para F4 ya existen (de F3-A), omitiendo...\")\n",
        "else:\n",
        "    emb_sample_size = 10_000\n",
        "    n_per_class = emb_sample_size // 3\n",
        "    rng = np.random.RandomState(RANDOM_STATE)\n",
        "    emb_chunks, label_chunks = [], []\n",
        "    for s in [0, 1, 2]:\n",
        "        idx = np.where(y_test == s)[0]\n",
        "        n = min(n_per_class, len(idx))\n",
        "        chosen = rng.choice(idx, n, replace=False)\n",
        "        emb_chunks.append(X_test_emb[chosen])\n",
        "        label_chunks.append(y_test[chosen])\n",
        "    emb_sample = np.concatenate(emb_chunks)\n",
        "    label_sample = np.concatenate(label_chunks)\n",
        "    np.save(EMBEDDINGS_PATH, emb_sample)\n",
        "    np.save(LABELS_PATH, label_sample)\n",
        "    print(f\"Saved {len(emb_sample)} embeddings para F4\")\n"
    ]))

    cells.append(md([
        "## 7. Visualizaciones\n",
        "\n",
        "Matriz de confusion, t-SNE, UMAP, K-Means clustering, similitud coseno.\n"
    ]))

    cells.append(md(["### 7a. Matriz de confusion\n"]))
    cells.append(code([
        "cm = np.array(metrics['confusion_matrix'])\n",
        "labels_names = metrics['class_labels']\n",
        "\n",
        "plt.figure(figsize=(6, 5))\n",
        "sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',\n",
        "            xticklabels=labels_names, yticklabels=labels_names)\n",
        "plt.xlabel('Predicho')\n",
        "plt.ylabel('Real')\n",
        "plt.title('Matriz de Confusión - DistilBERT')\n",
        "plt.tight_layout()\n",
        "plt.show()\n"
    ]))

    cells.append(md(["### 7b. t-SNE (5k muestras)\n"]))
    cells.append(code([
        "tsne = TSNE(n_components=2, random_state=RANDOM_STATE, perplexity=30)\n",
        "emb_tsne = tsne.fit_transform(X_test_emb[:5000])\n",
        "\n",
        "plt.figure(figsize=(10, 8))\n",
        "palette = ['#e74c3c', '#f39c12', '#2ecc71']\n",
        "for label in [0, 1, 2]:\n",
        "    mask = y_test[:5000] == label\n",
        "    plt.scatter(emb_tsne[mask, 0], emb_tsne[mask, 1],\n",
        "                c=palette[label], label=labels_names[label], alpha=0.6, s=10)\n",
        "plt.title('t-SNE de Embeddings DistilBERT (5k muestras)')\n",
        "plt.legend()\n",
        "plt.tight_layout()\n",
        "plt.show()\n"
    ]))

    cells.append(md(["### 7c. UMAP\n"]))
    cells.append(code([
        "reducer = umap.UMAP(n_neighbors=50, min_dist=0.1, n_components=2,\n",
        "                   random_state=RANDOM_STATE, init='random')\n",
        "emb_umap = reducer.fit_transform(X_test_emb)\n",
        "\n",
        "plt.figure(figsize=(8, 6))\n",
        "for s in [0, 1, 2]:\n",
        "    mask = y_test == s\n",
        "    plt.scatter(emb_umap[mask, 0], emb_umap[mask, 1],\n",
        "                c=palette[s], label=labels_names[s], alpha=0.5, s=8)\n",
        "plt.title('UMAP: Embeddings DistilBERT')\n",
        "plt.legend()\n",
        "plt.tight_layout()\n",
        "plt.show()\n"
    ]))

    cells.append(md(["### 7d. K-Means clustering\n"]))
    cells.append(code([
        "K_range = range(2, 9)\n",
        "inertias, sil_scores, db_scores, ch_scores = [], [], [], []\n",
        "\n",
        "rng = np.random.RandomState(RANDOM_STATE)\n",
        "idx_sub = rng.choice(len(X_test_emb), 5_000, replace=False)\n",
        "emb_sub = X_test_emb[idx_sub]\n",
        "\n",
        "for k in K_range:\n",
        "    km = KMeans(n_clusters=k, random_state=RANDOM_STATE, n_init=10)\n",
        "    labels = km.fit_predict(emb_sub)\n",
        "    inertias.append(km.inertia_)\n",
        "    sil_scores.append(silhouette_score(emb_sub, labels))\n",
        "    db_scores.append(davies_bouldin_score(emb_sub, labels))\n",
        "    ch_scores.append(calinski_harabasz_score(emb_sub, labels))\n",
        "\n",
        "fig, axes = plt.subplots(2, 2, figsize=(12, 10))\n",
        "axes[0, 0].plot(list(K_range), inertias, marker='o')\n",
        "axes[0, 0].set_title('Inercia (codo)')\n",
        "axes[0, 1].plot(list(K_range), sil_scores, marker='o', color='green')\n",
        "axes[0, 1].set_title('Silhouette Score (mayor = mejor)')\n",
        "axes[1, 0].plot(list(K_range), db_scores, marker='o', color='red')\n",
        "axes[1, 0].set_title('Davies-Bouldin (menor = mejor)')\n",
        "axes[1, 1].plot(list(K_range), ch_scores, marker='o', color='purple')\n",
        "axes[1, 1].set_title('Calinski-Harabasz (mayor = mejor)')\n",
        "for ax in axes.flat:\n",
        "    ax.set_xlabel('K')\n",
        "plt.tight_layout()\n",
        "plt.show()\n"
    ]))

    cells.append(md(["### 7e. Similitud coseno\n"]))
    cells.append(code([
        "n_per_class = 100\n",
        "emb_list, txt_list, lbl_list = [], [], []\n",
        "for s in [0, 1, 2]:\n",
        "    idx_s = np.where(y_test == s)[0]\n",
        "    chosen = np.random.RandomState(RANDOM_STATE).choice(idx_s, n_per_class, replace=False)\n",
        "    emb_list.append(X_test_emb[chosen])\n",
        "    lbl_list.append(y_test[chosen])\n",
        "\n",
        "emb_cos = np.concatenate(emb_list)\n",
        "sim_matrix = cosine_similarity(emb_cos)\n",
        "\n",
        "plt.figure(figsize=(8, 6))\n",
        "sns.heatmap(sim_matrix, cmap='coolwarm', vmin=0, vmax=1,\n",
        "            xticklabels=False, yticklabels=False)\n",
        "plt.title('Similitud Coseno entre Embeddings')\n",
        "plt.tight_layout()\n",
        "plt.show()\n"
    ]))

    cells.append(code([
        "# Liberar memoria\n",
        "del X_train_emb, X_val_emb, X_test_emb, y_train, y_val, y_test\n",
        "gc.collect()\n",
        "print(\"\\nF3-C completado.\")\n"
    ]))

    nb = nbf.v4.new_notebook(
        metadata={
            "kernelspec": {"name": "python3", "display_name": "Python 3"},
            "language_info": {"name": "python"},
            "accelerator": "GPU",
            "colab": {"provenance": [], "gpuType": "T4"}
        },
        cells=cells
    )
    path = os.path.join(NB_DIR, "f3_distilbert.ipynb")
    with open(path, "w") as f:
        nbf.write(nb, f)
    print(f"  ✓ {path} (modificado)")


# ──────────────────────────────────────────────
# F3-D: lora_ensemble
# ──────────────────────────────────────────────
def build_f3_d():
    cells = []

    cells.append(md([
        "# F3-D — LoRA Fine-Tuning + Ensemble Ponderado\n",
        "\n",
        "**Objetivo**: Fine-tuning con LoRA de DistilBERT + Ensemble ponderado ",
        "(RF + XGBoost + LoRA). Notebook GPU.\n",
        "\n",
        "**Tiempo estimado**: ~2h (GPU T4)\n"
    ]))

    cells.append(md(["## 1. Instalar dependencias\n"]))
    cells.append(code([
        "!pip install -q polars mlflow xgboost transformers -U\n",
        "!pip install -q peft -U\n"
    ]))

    cells.append(code([
        "import numpy as np\n",
        "import torch\n",
        "import gc\n",
        "import os\n",
        "import json\n",
        "import time\n",
        "import mlflow\n",
        "import matplotlib.pyplot as plt\n",
        "import pickle\n",
        "from google.colab import drive\n",
        "from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score\n",
        "from sklearn.metrics import precision_recall_fscore_support, confusion_matrix\n",
        "from transformers import (AutoTokenizer, AutoModelForSequenceClassification,\n",
        "                           TrainingArguments, Trainer, DataCollatorWithPadding)\n",
        "from peft import get_peft_model, LoraConfig, TaskType\n",
        "from datasets import Dataset\n",
        "import warnings\n",
        "warnings.filterwarnings('ignore')\n"
    ]))

    cells.append(md(["## 2. Montar Google Drive y cargar datos\n"]))
    cells.append(code([
        "drive.mount('/content/drive')\n",
        "\n",
        "DRIVE_BASE = \"/content/drive/MyDrive/ML/proyecto_integrador\"\n",
        "EMB_DIR = f\"{DRIVE_BASE}/embeddings\"\n",
        "REPORTS_DIR = f\"{DRIVE_BASE}/reports\"\n",
        "PREDS_DIR = f\"{DRIVE_BASE}/preds\"\n",
        "RANDOM_STATE = 42\n",
        "BATCH_SIZE = 256\n",
        "MAX_LENGTH = 128\n",
        "\n",
        "for d in [REPORTS_DIR, PREDS_DIR]:\n",
        "    os.makedirs(d, exist_ok=True)\n",
        "\n",
        "print(\"Cargando embeddings, features y textos desde F3-A...\")\n",
        "# Embeddings\n",
        "X_train_emb = np.load(f\"{EMB_DIR}/train_embeddings.npy\")\n",
        "X_val_emb   = np.load(f\"{EMB_DIR}/val_embeddings.npy\")\n",
        "X_test_emb  = np.load(f\"{EMB_DIR}/test_embeddings.npy\")\n",
        "# Engineered features\n",
        "eng_train = np.load(f\"{EMB_DIR}/train_eng_features.npy\")\n",
        "eng_val   = np.load(f\"{EMB_DIR}/val_eng_features.npy\")\n",
        "eng_test  = np.load(f\"{EMB_DIR}/test_eng_features.npy\")\n",
        "# Labels\n",
        "y_train = np.load(f\"{EMB_DIR}/train_labels.npy\")\n",
        "y_val   = np.load(f\"{EMB_DIR}/val_labels.npy\")\n",
        "y_test  = np.load(f\"{EMB_DIR}/test_labels.npy\")\n",
        "# Textos crudos (para LoRA)\n",
        "with open(f\"{EMB_DIR}/train_texts.pkl\", 'rb') as f:\n",
        "    X_train_texts = pickle.load(f)\n",
        "with open(f\"{EMB_DIR}/val_texts.pkl\", 'rb') as f:\n",
        "    X_val_texts = pickle.load(f)\n",
        "with open(f\"{EMB_DIR}/test_texts.pkl\", 'rb') as f:\n",
        "    X_test_texts = pickle.load(f)\n",
        "\n",
        "# Concatenar para clasicos\n",
        "X_train = np.concatenate([X_train_emb, eng_train], axis=1)\n",
        "X_val   = np.concatenate([X_val_emb, eng_val], axis=1)\n",
        "X_test  = np.concatenate([X_test_emb, eng_test], axis=1)\n",
        "\n",
        "print(f\"Datos cargados: train {len(X_train_texts)}, val {len(X_val_texts)}, test {len(X_test_texts)}\")\n"
    ]))

    cells.append(md([
        "## 3. Cargar predicciones de RF y XGBoost (desde F3-B)\n",
        "\n",
        "Si no existen (porque F3-B no se ejecutó), se cargan modelos pre-entrenados.\n"
    ]))

    cells.append(code([
        "PREDS_DIR = f\"{DRIVE_BASE}/preds\"\n",
        "\n",
        "if os.path.exists(f\"{PREDS_DIR}/y_pred_rf.npy\"):\n",
        "    print(\"Cargando predicciones de F3-B...\")\n",
        "    y_pred_rf   = np.load(f\"{PREDS_DIR}/y_pred_rf.npy\")\n",
        "    y_pred_xgb  = np.load(f\"{PREDS_DIR}/y_pred_xgb.npy\")\n",
        "    with open(f\"{PREDS_DIR}/part1_results.json\") as f:\n",
        "        part1_results = json.load(f)\n",
        "    rf_metrics  = [r for r in part1_results if r['model_name'] == 'Random Forest'][0]\n",
        "    xgb_metrics = [r for r in part1_results if r['model_name'] == 'XGBoost'][0]\n",
        "    print(f\"RF F1: {rf_metrics['f1_macro']}, XGB F1: {xgb_metrics['f1_macro']}\")\n",
        "else:\n",
        "    print(\"Predicciones de F3-B no encontradas. Entrenando modelos aquí...\")\n",
        "    from sklearn.ensemble import RandomForestClassifier\n",
        "    from xgboost import XGBClassifier\n",
        "    rf = RandomForestClassifier(n_estimators=200, random_state=RANDOM_STATE, n_jobs=-1)\n",
        "    rf.fit(X_train, y_train)\n",
        "    y_pred_rf = rf.predict(X_test)\n",
        "    xgb = XGBClassifier(n_estimators=300, max_depth=6, learning_rate=0.1,\n",
        "                        random_state=RANDOM_STATE, eval_metric='mlogloss')\n",
        "    xgb.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)\n",
        "    y_pred_xgb = xgb.predict(X_test)\n",
        "    print(\"Modelos clásicos entrenados localmente\")\n"
    ]))

    cells.append(md(["## 4. LoRA Fine-Tuning\n"]))

    cells.append(code([
        "MODEL_NAME = \"distilbert-base-uncased\"\n",
        "tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)\n",
        "device = torch.device(\"cuda\" if torch.cuda.is_available() else \"cpu\")\n",
        "\n",
        "def tokenize_fn(batch):\n",
        "    return tokenizer(batch['text'], truncation=True, max_length=MAX_LENGTH)\n",
        "\n",
        "train_ds = Dataset.from_dict({'text': X_train_texts, 'label': y_train})\n",
        "val_ds   = Dataset.from_dict({'text': X_val_texts, 'label': y_val})\n",
        "test_ds  = Dataset.from_dict({'text': X_test_texts, 'label': y_test})\n",
        "\n",
        "train_ds = train_ds.map(tokenize_fn, batched=True)\n",
        "val_ds   = val_ds.map(tokenize_fn, batched=True)\n",
        "test_ds  = test_ds.map(tokenize_fn, batched=True)\n",
        "\n",
        "data_collator = DataCollatorWithPadding(tokenizer=tokenizer)\n",
        "\n",
        "def compute_metrics(eval_pred):\n",
        "    logits, labels = eval_pred\n",
        "    predictions = np.argmax(logits, axis=-1)\n",
        "    return {'f1_macro': f1_score(labels, predictions, average='macro'),\n",
        "            'accuracy': accuracy_score(labels, predictions)}\n"
    ]))

    cells.append(code([
        "model_cls = AutoModelForSequenceClassification.from_pretrained(\n",
        "    MODEL_NAME, num_labels=3\n",
        ").to(device)\n",
        "\n",
        "lora_config = LoraConfig(\n",
        "    task_type=TaskType.SEQ_CLS,\n",
        "    r=16,\n",
        "    lora_alpha=32,\n",
        "    lora_dropout=0.1,\n",
        "    target_modules=['q_lin', 'k_lin', 'v_lin', 'out_lin']\n",
        ")\n",
        "model_lora = get_peft_model(model_cls, lora_config)\n",
        "model_lora.print_trainable_parameters()\n"
    ]))

    cells.append(code([
        "lora_args = TrainingArguments(\n",
        "    output_dir='/content/lora_checkpoints',\n",
        "    eval_strategy='epoch',\n",
        "    save_strategy='epoch',\n",
        "    per_device_train_batch_size=128,\n",
        "    per_device_eval_batch_size=256,\n",
        "    num_train_epochs=4,\n",
        "    learning_rate=2e-4,\n",
        "    weight_decay=0.01,\n",
        "    logging_steps=50,\n",
        "    load_best_model_at_end=True,\n",
        "    metric_for_best_model='f1_macro',\n",
        "    report_to='none',\n",
        ")\n",
        "\n",
        "trainer_lora = Trainer(\n",
        "    model=model_lora,\n",
        "    args=lora_args,\n",
        "    train_dataset=train_ds,\n",
        "    eval_dataset=val_ds,\n",
        "    tokenizer=tokenizer,\n",
        "    data_collator=data_collator,\n",
        "    compute_metrics=compute_metrics,\n",
        ")\n",
        "\n",
        "print(\"Iniciando LoRA fine-tuning...\")\n",
        "start = time.time()\n",
        "trainer_lora.train()\n",
        "lora_time = time.time() - start\n",
        "print(f\"LoRA completado en {lora_time:.0f}s\")\n"
    ]))

    cells.append(md(["## 5. Evaluar LoRA\n"]))
    cells.append(code([
        "y_pred_lora = trainer_lora.predict(test_ds).predictions.argmax(-1)\n",
        "lora_f1 = f1_score(y_test, y_pred_lora, average='macro')\n",
        "print(f\"LoRA test F1-macro: {lora_f1:.4f}\")\n",
        "\n",
        "lora_metrics, _ = None, None\n",
        "def eval_and_record(name, y_true, y_pred, training_time):\n",
        "    p, r, f, _ = precision_recall_fscore_support(y_true, y_pred, labels=[0, 1, 2])\n",
        "    per_class = {\n",
        "        label: {'precision': round(p[i], 4), 'recall': round(r[i], 4), 'f1': round(f[i], 4)}\n",
        "        for i, label in enumerate(['Negativo', 'Neutro', 'Positivo'])\n",
        "    }\n",
        "    return {\n",
        "        'model_name': name,\n",
        "        'training_time_seconds': round(training_time, 2),\n",
        "        'f1_macro': round(f1_score(y_true, y_pred, average='macro'), 4),\n",
        "        'precision_macro': round(precision_score(y_true, y_pred, average='macro'), 4),\n",
        "        'recall_macro': round(recall_score(y_true, y_pred, average='macro'), 4),\n",
        "        'accuracy': round(accuracy_score(y_true, y_pred), 4),\n",
        "        'per_class': per_class,\n",
        "        'confusion_matrix': confusion_matrix(y_true, y_pred).tolist(),\n",
        "    }\n",
        "\n",
        "lora_metrics = eval_and_record('DistilBERT + LoRA', y_test, y_pred_lora, lora_time)\n",
        "results = []\n",
        "if 'rf_metrics' in dir():\n",
        "    results.append(rf_metrics)\n",
        "    results.append(xgb_metrics)\n",
        "results.append(lora_metrics)\n"
    ]))

    cells.append(md(["## 6. Learning Curves\n"]))
    cells.append(code([
        "def _plot_learning_curve(log_history, title):\n",
        "    train_loss = [x['loss'] for x in log_history if 'loss' in x and 'eval_loss' not in x]\n",
        "    eval_loss = [x['eval_loss'] for x in log_history if 'eval_loss' in x]\n",
        "    eval_f1 = [x.get('eval_f1_macro', None) for x in log_history if 'eval_loss' in x]\n",
        "    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))\n",
        "    ax1.plot(range(1, len(train_loss) + 1), train_loss, label='Train loss', color='#3498db', linewidth=2)\n",
        "    epochs = list(range(1, len(eval_loss) + 1))\n",
        "    ax1.plot(epochs, eval_loss, label='Val loss', color='#e74c3c', linewidth=2, marker='o')\n",
        "    ax1.set_xlabel('Epoch'); ax1.set_ylabel('Loss'); ax1.legend(); ax1.set_title(f'{title} - Loss')\n",
        "    ax2.plot(epochs, eval_f1, label='Val F1', color='#2ecc71', linewidth=2, marker='o')\n",
        "    ax2.set_xlabel('Epoch'); ax2.set_ylabel('F1-macro'); ax2.legend(); ax2.set_title(f'{title} - Val F1')\n",
        "    plt.tight_layout(); plt.show()\n",
        "\n",
        "_plot_learning_curve(trainer_lora.state.log_history, 'LoRA')\n"
    ]))

    cells.append(md(["## 7. Ensemble Ponderado\n"]))
    cells.append(code([
        "print(\"\\n\" + \"=\"*60)\n",
        "print(\"Ensemble ponderado\")\n",
        "print(\"=\"*60)\n",
        "\n",
        "model_preds = {\n",
        "    'Random Forest': y_pred_rf,\n",
        "    'XGBoost': y_pred_xgb,\n",
        "    'DistilBERT + LoRA': y_pred_lora,\n",
        "}\n",
        "\n",
        "weights_list = []\n",
        "preds_list = []\n",
        "for r in results:\n",
        "    if r['model_name'] in model_preds:\n",
        "        weights_list.append(r['f1_macro'])\n",
        "        preds_list.append(model_preds[r['model_name']])\n",
        "\n",
        "weights_list = np.array(weights_list)\n",
        "preds_list = np.array(preds_list)\n",
        "\n",
        "# Weighted majority vote: sumar votos ponderados por clase\n",
        "n_classes = 3\n",
        "weighted_votes = np.zeros((len(y_test), n_classes))\n",
        "for w, pred in zip(weights_list, preds_list):\n",
        "    for c in range(n_classes):\n",
        "        weighted_votes[:, c] += w * (pred == c)\n",
        "\n",
        "y_pred_ensemble = np.argmax(weighted_votes, axis=1)\n",
        "\n",
        "weights_str = ', '.join([f\"{r['model_name']}: {w:.4f}\" for r, w in zip(results, weights_list)])\n",
        "print(f\"Pesos del ensemble: {weights_str}\")\n",
        "\n",
        "ens_metrics = eval_and_record('Ensemble (ponderado)', y_test, y_pred_ensemble, 0)\n",
        "results.append(ens_metrics)\n",
        "print(f\"Ensemble test F1-macro: {ens_metrics['f1_macro']}\")\n"
    ]))

    cells.append(md(["## 8. Guardar predicciones\n"]))
    cells.append(code([
        "os.makedirs(PREDS_DIR, exist_ok=True)\n",
        "np.save(f\"{PREDS_DIR}/y_pred_lora.npy\", y_pred_lora)\n",
        "np.save(f\"{PREDS_DIR}/y_pred_ensemble.npy\", y_pred_ensemble)\n",
        "print(\"Predicciones guardadas en Drive\")\n"
    ]))

    cells.append(md(["## 9. Resultados comparativos\n"]))
    cells.append(code([
        "model_names = [r['model_name'] for r in results]\n",
        "f1_scores = [r['f1_macro'] for r in results]\n",
        "class_labels = ['Negativo', 'Neutro', 'Positivo']\n",
        "\n",
        "plt.figure(figsize=(12, 5))\n",
        "plt.subplot(1, 2, 1)\n",
        "colors = ['#3498db', '#2ecc71', '#e74c3c', '#9b59b6']\n",
        "bars = plt.barh(range(len(results)), f1_scores, color=colors[:len(results)])\n",
        "plt.yticks(range(len(results)), model_names)\n",
        "plt.xlabel('F1-macro')\n",
        "plt.title('Comparacion de Modelos - F1-macro')\n",
        "plt.xlim(0, 1)\n",
        "for bar, val in zip(bars, f1_scores):\n",
        "    plt.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height()/2,\n",
        "             f'{val:.4f}', va='center', fontsize=10)\n",
        "\n",
        "plt.subplot(1, 2, 2)\n",
        "x = np.arange(len(class_labels))\n",
        "width = 0.2\n",
        "for i, r in enumerate(results):\n",
        "    f1_per = [r['per_class'][c]['f1'] for c in class_labels]\n",
        "    plt.bar(x + i*width, f1_per, width, label=r['model_name'], color=colors[i])\n",
        "plt.xticks(x + width * 1.5, class_labels)\n",
        "plt.ylabel('F1-score')\n",
        "plt.title('F1 por clase')\n",
        "plt.legend(loc='best', fontsize=8)\n",
        "plt.tight_layout()\n",
        "plt.show()\n",
        "\n",
        "print(\"\\n\" + \"=\"*60)\n",
        "print(\"Resumen de metricas\")\n",
        "print(\"=\"*60)\n",
        "for r in results:\n",
        "    print(f\"{r['model_name']:30s} F1={r['f1_macro']:.4f}  Acc={r['accuracy']:.4f}  T={r['training_time_seconds']:.0f}s\")\n"
    ]))

    cells.append(md(["## 10. MLflow Tracking\n"]))
    cells.append(code([
        "MLFLOW_TRACKING_URI = os.getenv(\"MLFLOW_TRACKING_URI\", \"https://humorous-trusting-domelike.ngrok-free.dev\")\n",
        "import requests\n",
        "try:\n",
        "    r = requests.get(f\"{MLFLOW_TRACKING_URI}/api/2.0/mlflow/experiments/list\", timeout=5)\n",
        "    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)\n",
        "except Exception:\n",
        "    mlflow.set_tracking_uri(f\"sqlite:///{DRIVE_BASE}/mlflow_fallback.db\")\n",
        "\n",
        "mlflow.set_experiment(\"distilbert_improved\")\n",
        "\n",
        "for r in results:\n",
        "    with mlflow.start_run(run_name=r['model_name']):\n",
        "        mlflow.log_params({'model_name': r['model_name']})\n",
        "        mlflow.log_metrics({\n",
        "            'f1_macro': r['f1_macro'],\n",
        "            'accuracy': r['accuracy'],\n",
        "            'training_time_seconds': r['training_time_seconds'],\n",
        "        })\n",
        "        mlflow.log_dict(r['confusion_matrix'], f\"{r['model_name']}_confusion_matrix.json\")\n",
        "\n",
        "print(\"MLflow tracking completado\")\n"
    ]))

    cells.append(md(["## 11. Exportar métricas a JSON\n"]))
    cells.append(code([
        "report_path = f\"{REPORTS_DIR}/metrics_distilbert_improved.json\"\n",
        "with open(report_path, 'w') as f:\n",
        "    json.dump(results, f, indent=2)\n",
        "print(f\"Exportado: {report_path}\")\n"
    ]))

    cells.append(code([
        "# Liberar memoria\n",
        "del model_lora, trainer_lora, model_cls, tokenizer\n",
        "del X_train_emb, X_val_emb, X_test_emb, eng_train, eng_val, eng_test\n",
        "del X_train, X_val, X_test, y_train, y_val, y_test\n",
        "gc.collect()\n",
        "if torch.cuda.is_available():\n",
        "    torch.cuda.empty_cache()\n",
        "print(\"\\nF3-D completado. Todos los modelos entrenados.\")\n"
    ]))

    nb = nbf.v4.new_notebook(
        metadata={
            "kernelspec": {"name": "python3", "display_name": "Python 3"},
            "language_info": {"name": "python"},
            "accelerator": "GPU",
            "colab": {"provenance": [], "gpuType": "T4"}
        },
        cells=cells
    )
    path = os.path.join(NB_DIR, "f3_lora_ensemble.ipynb")
    with open(path, "w") as f:
        nbf.write(nb, f)
    print(f"  ✓ {path}")


# ──────────────────────────────────────────────
# MAIN
# ──────────────────────────────────────────────
if __name__ == "__main__":
    print("Generando notebooks F3...\n")
    build_f3_a()
    build_f3_b()
    build_f3_c()
    build_f3_d()
    print("\nTodos los notebooks generados exitosamente.")
