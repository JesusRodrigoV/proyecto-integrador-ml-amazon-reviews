#!/usr/bin/env python3
"""Genera gráficas comparativas a partir de reports/*.json
   Uso: python src/visualize.py
   Requiere: matplotlib, seaborn, pandas, numpy
   Corre en local (PC del equipo), 0 GPU necesaria.
"""

import json
import os
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

REPORTS_DIR = os.path.join(os.path.dirname(__file__), "..", "reports")
FIGURES_DIR = os.path.join(REPORTS_DIR, "figures")

sns.set_theme(style="whitegrid", palette="muted")
os.makedirs(FIGURES_DIR, exist_ok=True)

def load_json(filename):
    path = os.path.join(REPORTS_DIR, filename)
    if not os.path.exists(path):
        print(f"[SKIP] No encontrado: {path}")
        return None
    with open(path) as f:
        return json.load(f)

def plot_f1_comparison(metrics_files):
    models_data = []
    for fname in metrics_files:
        data = load_json(fname)
        if data is None:
            continue
        model_name = data.get("model_name", fname.replace("metrics_", "").replace(".json", ""))
        models_data.append({
            "model": model_name,
            "f1_macro": data.get("f1_macro"),
            "precision_macro": data.get("precision_macro"),
            "recall_macro": data.get("recall_macro"),
            "accuracy": data.get("accuracy"),
        })

    if not models_data:
        print("[SKIP] No hay datos de métricas para graficar.")
        return

    df = pd.DataFrame(models_data)
    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(len(df))
    width = 0.2
    metrics = ["f1_macro", "precision_macro", "recall_macro", "accuracy"]
    colors = ["#e74c3c", "#f39c12", "#2ecc71", "#3498db"]

    for i, (metric, color) in enumerate(zip(metrics, colors)):
        ax.bar(x + i * width, df[metric], width, label=metric.replace("_", " ").title(), color=color)

    ax.set_xticks(x + width * 1.5)
    ax.set_xticklabels(df["model"], rotation=15, ha="right")
    ax.set_ylabel("Score")
    ax.set_title("Comparación de Métricas por Modelo")
    ax.legend(loc="lower right")
    ax.set_ylim(0, 1)
    plt.tight_layout()
    path = os.path.join(FIGURES_DIR, "f1_comparison.png")
    fig.savefig(path, dpi=150)
    print(f"[OK] Guardado: {path}")
    plt.close(fig)

def plot_f1_vs_training_time(metrics_files):
    models_data = []
    for fname in metrics_files:
        data = load_json(fname)
        if data is None:
            continue
        training_time = data.get("training_time_seconds")
        f1 = data.get("f1_macro")
        if training_time is None or f1 is None:
            continue
        models_data.append({
            "model": data.get("model_name", fname.replace("metrics_", "").replace(".json", "")),
            "training_time_seconds": training_time,
            "f1_macro": f1,
        })

    if not models_data:
        print("[SKIP] No hay suficientes datos para F1 vs Tiempo.")
        return

    df = pd.DataFrame(models_data)
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.scatter(df["training_time_seconds"], df["f1_macro"], s=100, c="#3498db", zorder=3)
    for _, row in df.iterrows():
        ax.annotate(row["model"], (row["training_time_seconds"], row["f1_macro"]),
                    textcoords="offset points", xytext=(8, -8), fontsize=9)
    ax.set_xlabel("Tiempo de Entrenamiento (segundos)")
    ax.set_ylabel("F1 Macro")
    ax.set_title("Tradeoff: F1 Macro vs Tiempo de Entrenamiento")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    path = os.path.join(FIGURES_DIR, "f1_vs_time.png")
    fig.savefig(path, dpi=150)
    print(f"[OK] Guardado: {path}")
    plt.close(fig)

def plot_confusion_matrices(metrics_files):
    for fname in metrics_files:
        data = load_json(fname)
        if data is None:
            continue
        cm = data.get("confusion_matrix")
        if cm is None:
            continue
        model_name = data.get("model_name", fname.replace("metrics_", "").replace(".json", ""))
        cm_array = np.array(cm)
        labels = data.get("class_labels", ["Neg", "Neu", "Pos"])

        fig, ax = plt.subplots(figsize=(6, 5))
        sns.heatmap(cm_array, annot=True, fmt="d", cmap="Blues",
                    xticklabels=labels, yticklabels=labels, ax=ax)
        ax.set_xlabel("Predicho")
        ax.set_ylabel("Real")
        ax.set_title(f"Matriz de Confusión - {model_name}")
        plt.tight_layout()
        path = os.path.join(FIGURES_DIR, f"confusion_{model_name.replace(' ', '_').lower()}.png")
        fig.savefig(path, dpi=150)
        print(f"[OK] Guardado: {path}")
        plt.close(fig)

def plot_eda_summary():
    data = load_json("eda_summary.json")
    if data is None:
        print("[SKIP] No hay EDA summary para graficar.")
        return

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    # Distribución rating original
    rating_dist = data.get("rating_distribution", {})
    if rating_dist:
        df_r = pd.DataFrame(list(rating_dist.items()), columns=["rating", "count"])
        df_r["rating"] = df_r["rating"].astype(int)
        df_r = df_r.sort_values("rating")
        axes[0].bar(df_r["rating"].astype(str), df_r["count"], color="#3498db", edgecolor="black")
        axes[0].set_title("Distribución Rating Original (balanceado)")
        axes[0].set_xlabel("Rating")
        axes[0].set_ylabel("Cantidad")

    # Distribución sentiment (3 clases)
    sentiment_dist = data.get("sentiment_distribution", {})
    if sentiment_dist:
        labels_map = {"0": "Negativo", "1": "Neutro", "2": "Positivo"}
        df_s = pd.DataFrame([
            (labels_map.get(k, k), v) for k, v in sentiment_dist.items()
        ], columns=["sentiment", "count"])
        colors = ["#e74c3c", "#f39c12", "#2ecc71"]
        axes[1].bar(df_s["sentiment"], df_s["count"], color=colors, edgecolor="black")
        axes[1].set_title("Distribución Sentiment (3 clases)")
        axes[1].set_xlabel("Clase")
        axes[1].set_ylabel("Cantidad")

    plt.tight_layout()
    path = os.path.join(FIGURES_DIR, "eda_summary.png")
    fig.savefig(path, dpi=150)
    print(f"[OK] Guardado: {path}")
    plt.close(fig)

def main():
    print("=" * 50)
    print("  src/visualize.py — Generación de gráficas")
    print("=" * 50)
    print(f"  Reports:  {REPORTS_DIR}")
    print(f"  Output:   {FIGURES_DIR}")
    print("=" * 50)

    metrics_files = sorted([
        f for f in os.listdir(REPORTS_DIR)
        if f.startswith("metrics_") and f.endswith(".json")
    ])
    print(f"  Archivos de métricas encontrados: {metrics_files or '(ninguno)'}")

    plot_eda_summary()
    plot_f1_comparison(metrics_files)
    plot_f1_vs_training_time(metrics_files)
    plot_confusion_matrices(metrics_files)

    print("=" * 50)
    print(f"  Listo. Gráficas en: {FIGURES_DIR}")
    print("=" * 50)

if __name__ == "__main__":
    main()
