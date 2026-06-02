#!/usr/bin/env python3
"""
f3_generar_graficos.py — Genera graficas para el informe de Fase 3.

Uso:
    python src/f3_generar_graficos.py

Requiere: matplotlib, seaborn, numpy
Corre en local (PC del equipo), 0 GPU necesaria.

Salida:
    reports/figures/fase3/01_comparativa_f1_macro.png
    reports/figures/fase3/02_matrices_confusion.png
    reports/figures/fase3/03_f1_por_clase.png
    reports/figures/fase3/04_tiempo_vs_f1.png
    reports/figures/fase3/05_radar_multimetrica.png
    reports/figures/fase3/06_heatmap_classification.png
    reports/figures/fase3/07_ranking_metricas.png
    reports/figures/fase3/08_scatter_3d.png
"""

import json
import os
import sys
from pathlib import Path

import numpy as np

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import seaborn as sns


# ============================================================
# CONFIG
# ============================================================
REPORTS_DIR = Path(__file__).resolve().parent.parent / 'reports'
FIGURES_DIR = REPORTS_DIR / 'figures' / 'fase3'
BASELINE_JSON = REPORTS_DIR / 'metrics_distilbert.json'
IMPROVED_JSON = REPORTS_DIR / 'metrics_distilbert_improved.json'

CLASS_LABELS = ['Negativo', 'Neutro', 'Positivo']
CLASS_COLORS = ['#e74c3c', '#f39c12', '#2ecc71']

sns.set_theme(style='whitegrid', palette='muted')
plt.rcParams.update({
    'figure.dpi': 150,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'font.size': 11,
    'axes.titlesize': 14,
    'axes.labelsize': 12,
})

os.makedirs(FIGURES_DIR, exist_ok=True)


# ============================================================
# HELPERS
# ============================================================

def _cargar_json(path):
    try:
        with open(path) as f:
            return json.load(f)
    except FileNotFoundError:
        print(f'  [SALT] No encontrado: {path}')
        return None
    except json.JSONDecodeError as e:
        print(f'  [ERR] JSON invalido en {path}: {e}')
        return None


def _unificar_modelos():
    """Carga ambos JSONs y retorna una lista de modelos normalizada (dicts)."""
    baseline = _cargar_json(BASELINE_JSON)
    improved = _cargar_json(IMPROVED_JSON)
    modelos = []

    if baseline:
        modelos.append({
            'model_name': baseline.get('model_name', 'Baseline'),
            'model_type': baseline.get('model_type', ''),
            'f1_macro': baseline.get('f1_macro'),
            'precision_macro': baseline.get('precision_macro'),
            'recall_macro': baseline.get('recall_macro'),
            'accuracy': baseline.get('accuracy'),
            'training_time_seconds': baseline.get('training_time_seconds'),
            'confusion_matrix': baseline.get('confusion_matrix'),
            'per_class': baseline.get('per_class'),
        })

    if improved:
        vistos = {}
        for m in improved.get('improved_results', []):
            nombre = m.get('model_name', '')
            # dedup: mismo nombre -> quedarse con el de mayor f1_macro
            existente = vistos.get(nombre)
            if existente and existente.get('f1_macro', 0) >= m.get('f1_macro', 0):
                continue
            m = dict(m)
            if 'confusion_matrix' not in m or not m['confusion_matrix']:
                m['confusion_matrix'] = None
            if 'per_class' not in m or not m['per_class']:
                m['per_class'] = None
            vistos[nombre] = m
        modelos.extend(vistos.values())

    return modelos


def _guardar_figura(fig, nombre):
    ruta = FIGURES_DIR / nombre
    fig.savefig(ruta, dpi=300)
    print(f'  [OK]  {nombre}')
    plt.close(fig)


def _tiene_por_clase(m):
    return bool(m.get('per_class'))


def _tiene_cm(m):
    return bool(m.get('confusion_matrix'))


def _extraer_por_clase(m):
    """Retorna matrices (3,) de precision, recall, f1 por clase."""
    pc = m['per_class']
    prec = [pc[cl]['precision'] for cl in CLASS_LABELS]
    rec = [pc[cl]['recall'] for cl in CLASS_LABELS]
    f1s = [pc[cl]['f1'] for cl in CLASS_LABELS]
    return prec, rec, f1s


# ============================================================
# GRAFICAS
# ============================================================

def fig_01_comparativa_f1_macro(modelos):
    """Barras horizontales: F1-macro de todos los modelos."""
    nombres = [m['model_name'] for m in modelos]
    valores = [m['f1_macro'] for m in modelos]
    colores = sns.color_palette('husl', len(modelos))

    fig, ax = plt.subplots(figsize=(10, 5))
    bars = ax.barh(nombres, valores, color=colores, edgecolor='white', height=0.6)
    for bar, v in zip(bars, valores):
        ax.text(bar.get_width() + 0.005, bar.get_y() + bar.get_height() / 2,
                f'{v:.4f}', va='center', fontsize=10, fontweight='bold')

    ax.set_xlabel('F1-macro')
    ax.set_title('Fase 3 — Comparativa F1-macro por Modelo')
    ax.set_xlim(0, 1)
    ax.invert_yaxis()
    _guardar_figura(fig, '01_comparativa_f1_macro.png')


def fig_02_matrices_confusion(modelos):
    """Grid de matrices de confusion para los modelos que tengan datos."""
    elegibles = [m for m in modelos if _tiene_cm(m)]
    if not elegibles:
        print('  [SALT] 02_matrices_confusion: ningun modelo tiene confusion_matrix')
        return

    n = len(elegibles)
    cols = min(3, n)
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 4 * rows))
    axes = np.atleast_1d(axes).ravel()

    for ax, m in zip(axes, elegibles):
        cm = np.array(m['confusion_matrix'])
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=CLASS_LABELS, yticklabels=CLASS_LABELS,
                    ax=ax, cbar=False)
        ax.set_xlabel('Predicho')
        ax.set_ylabel('Real')
        ax.set_title(m['model_name'])

    for ax in axes[len(elegibles):]:
        ax.set_visible(False)

    fig.suptitle('Fase 3 — Matrices de Confusion', fontsize=15, y=1.02)
    fig.tight_layout()
    _guardar_figura(fig, '02_matrices_confusion.png')


def fig_03_f1_por_clase(modelos):
    """Barras agrupadas: F1 por clase (Neg, Neu, Pos) para cada modelo."""
    elegibles = [m for m in modelos if _tiene_por_clase(m)]
    if not elegibles:
        print('  [SALT] 03_f1_por_clase: ningun modelo tiene per_class')
        return

    x = np.arange(len(CLASS_LABELS))
    n = len(elegibles)
    ancho = 0.8 / n

    fig, ax = plt.subplots(figsize=(10, 6))
    for i, m in enumerate(elegibles):
        _, _, f1s = _extraer_por_clase(m)
        offset = (i - n / 2 + 0.5) * ancho
        bars = ax.bar(x + offset, f1s, ancho * 0.9, label=m['model_name'])
        for bar, v in zip(bars, f1s):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                    f'{v:.3f}', ha='center', fontsize=7)

    ax.set_xticks(x)
    ax.set_xticklabels(CLASS_LABELS)
    ax.set_ylabel('F1-score')
    ax.set_title('Fase 3 — F1 por Clase')
    ax.set_ylim(0, 1)
    ax.legend(loc='lower right', fontsize=9)
    ax.grid(axis='x', visible=False)
    _guardar_figura(fig, '03_f1_por_clase.png')


def fig_04_tiempo_vs_f1(modelos):
    """Scatter 2D con burbujas: tiempo de entrenamiento vs F1-macro,
    tamano de burbuja = accuracy."""
    tiempos = [m['training_time_seconds'] for m in modelos]
    f1s = [m['f1_macro'] for m in modelos]
    accs = [m['accuracy'] for m in modelos]
    nombres = [m['model_name'] for m in modelos]
    colores = sns.color_palette('husl', len(modelos))

    fig, ax = plt.subplots(figsize=(10, 6))
    sizes = [a * 800 + 50 for a in accs]

    scatter = ax.scatter(tiempos, f1s, s=sizes, c=colores, alpha=0.8,
                         edgecolors='black', linewidth=0.5, zorder=3)
    for nombre, t, f in zip(nombres, tiempos, f1s):
        ax.annotate(nombre, (t, f), textcoords='offset points',
                    xytext=(8, -8), fontsize=9,
                    bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.7))

    ax.set_xlabel('Tiempo de Entrenamiento (segundos)')
    ax.set_ylabel('F1-macro')
    ax.set_title('Fase 3 — Tradeoff: Rendimiento vs Tiempo de Entrenamiento')
    ax.grid(True, alpha=0.3)

    legend_elements = [Patch(facecolor='none', edgecolor='black',
                             label=f'Tamano = Accuracy ({a:.3f})')
                       for a in sorted(set(round(a, 3) for a in accs))]
    _guardar_figura(fig, '04_tiempo_vs_f1.png')


def fig_05_radar_multimetrica(modelos):
    """Spider/radar plot: F1-macro + F1 por clase para cada modelo."""
    elegibles = [m for m in modelos if _tiene_por_clase(m)]
    if not elegibles:
        print('  [SALT] 05_radar_multimetrica: ningun modelo tiene per_class')
        return

    categorias = ['F1-Macro', 'F1-Negativo', 'F1-Neutro', 'F1-Positivo']
    n_cat = len(categorias)
    angulos = np.linspace(0, 2 * np.pi, n_cat, endpoint=False).tolist()
    angulos += angulos[:1]

    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
    colores = sns.color_palette('husl', len(elegibles))

    for idx, m in enumerate(elegibles):
        _, _, f1s = _extraer_por_clase(m)
        valores = [m['f1_macro']] + f1s
        valores += valores[:1]
        ax.plot(angulos, valores, 'o-', linewidth=2, label=m['model_name'],
                color=colores[idx])
        ax.fill(angulos, valores, alpha=0.05, color=colores[idx])

    ax.set_xticks(angulos[:-1])
    ax.set_xticklabels(categorias, fontsize=11)
    ax.set_ylim(0, 1)
    ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_yticklabels(['0.2', '0.4', '0.6', '0.8', '1.0'], fontsize=8)
    ax.set_title('Fase 3 — Radar Multimetrica', pad=25, fontsize=14)
    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1), fontsize=9)
    _guardar_figura(fig, '05_radar_multimetrica.png')


def fig_06_heatmap_classification(modelos):
    """Heatmap: precision, recall, f1 por clase y por modelo."""
    elegibles = [m for m in modelos if _tiene_por_clase(m)]
    if not elegibles:
        print('  [SALT] 06_heatmap_classification: ningun modelo tiene per_class')
        return

    filas = []
    for m in elegibles:
        prec, rec, f1s = _extraer_por_clase(m)
        etiquetas_clase = [f'{m["model_name"]} — {cl}' for cl in CLASS_LABELS]
        for i, cl in enumerate(etiquetas_clase):
            filas.append({'Modelo / Clase': cl,
                          'Precision': prec[i],
                          'Recall': rec[i],
                          'F1-score': f1s[i]})

    import pandas as pd
    df = pd.DataFrame(filas).set_index('Modelo / Clase')

    fig, ax = plt.subplots(figsize=(8, max(4, len(df) * 0.4)))
    sns.heatmap(df, annot=True, fmt='.4f', cmap='YlOrRd',
                vmin=0.5, vmax=1.0, ax=ax, cbar_kws={'label': 'Score'})
    ax.set_title('Fase 3 — Metricas por Clase (Precision / Recall / F1)')
    ax.set_ylabel('')
    ax.set_xticklabels(ax.get_xticklabels(), rotation=0)
    fig.tight_layout()
    _guardar_figura(fig, '06_heatmap_classification.png')


def fig_07_ranking_metricas(modelos):
    """Dot plot: cada metrica como columna, modelos ordenados por F1-macro."""
    metricas = ['f1_macro', 'precision_macro', 'recall_macro', 'accuracy']
    etiquetas = ['F1-macro', 'Precision', 'Recall', 'Accuracy']

    ordenados = sorted(modelos, key=lambda m: m['f1_macro'], reverse=True)
    nombres = [m['model_name'] for m in ordenados]

    fig, ax = plt.subplots(figsize=(10, 5))
    cmap = plt.cm.Blues

    for i, met in enumerate(metricas):
        valores = [m[met] for m in ordenados]
        y = np.full(len(ordenados), i)
        colores_punto = [cmap(v * 0.8 + 0.2) for v in valores]
        ax.scatter(valores, y, s=200, c=colores_punto, edgecolors='black',
                   linewidth=0.5, zorder=3)
        for j, (v, nom) in enumerate(zip(valores, nombres)):
            ax.text(v + 0.008, i - 0.2 + j * 0.02, f'{v:.4f}',
                    fontsize=7, va='center')

    ax.set_yticks(range(len(metricas)))
    ax.set_yticklabels(etiquetas)
    ax.set_xlim(0.55, 0.85)
    ax.set_xlabel('Score')
    ax.set_title('Fase 3 — Ranking de Metricas por Modelo')
    ax.grid(axis='y', visible=False)

    legend_patches = [Patch(color=cmap(0.3), label=nom) for nom in nombres]
    ax.legend(handles=legend_patches, loc='lower left', fontsize=8,
              title='Modelos (ordenados por F1)', title_fontsize=9)
    fig.tight_layout()
    _guardar_figura(fig, '07_ranking_metricas.png')


def fig_08_scatter_3d(modelos):
    """Scatter 3D: F1-macro, training_time, accuracy con proyecciones."""
    tiempos = np.array([m['training_time_seconds'] for m in modelos])
    f1s = np.array([m['f1_macro'] for m in modelos])
    accs = np.array([m['accuracy'] for m in modelos])
    nombres = [m['model_name'] for m in modelos]
    colores = sns.color_palette('husl', len(modelos))

    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')

    ax.scatter(tiempos, f1s, accs, c=colores, s=150,
               edgecolors='black', linewidth=0.5, alpha=0.9, zorder=5)

    for nombre, t, f, a in zip(nombres, tiempos, f1s, accs):
        ax.text(t, f, a, f' {nombre}', fontsize=9,
                bbox=dict(boxstyle='round,pad=0.15', facecolor='white', alpha=0.7))

    # proyecciones en los planos
    for t, f, a in zip(tiempos, f1s, accs):
        ax.plot([t, t], [f, f], [0, a], color='gray', linewidth=0.5,
                linestyle='--', alpha=0.3)
        ax.plot([t, t], [0, f], [a, a], color='gray', linewidth=0.5,
                linestyle='--', alpha=0.3)
        ax.plot([0, t], [f, f], [a, a], color='gray', linewidth=0.5,
                linestyle='--', alpha=0.3)

    ax.set_xlabel('Tiempo (s)')
    ax.set_ylabel('F1-macro')
    ax.set_zlabel('Accuracy')
    ax.set_title('Fase 3 — Scatter 3D: Tiempo vs F1-macro vs Accuracy',
                 fontsize=14, pad=15)
    ax.view_init(elev=25, azim=-60)
    fig.tight_layout()
    _guardar_figura(fig, '08_scatter_3d.png')


# ============================================================
# MAIN
# ============================================================

def main():
    print('=' * 55)
    print('  f3_generar_graficos.py — Graficas Fase 3')
    print('=' * 55)
    print(f'  Reports: {REPORTS_DIR}')
    print(f'  Salida:  {FIGURES_DIR}')
    print('=' * 55)

    modelos = _unificar_modelos()
    if not modelos:
        print('  [ERR] No se pudo cargar ningun modelo. Saliendo.')
        sys.exit(1)

    print(f'  Modelos cargados: {len(modelos)}')
    for m in modelos:
        info = f'    - {m["model_name"]}'
        if _tiene_cm(m):
            info += ' [CM]'
        if _tiene_por_clase(m):
            info += ' [per_class]'
        print(info)
    print()

    graficas = [
        ('01_comparativa_f1_macro', fig_01_comparativa_f1_macro),
        ('02_matrices_confusion', fig_02_matrices_confusion),
        ('03_f1_por_clase', fig_03_f1_por_clase),
        ('04_tiempo_vs_f1', fig_04_tiempo_vs_f1),
        ('05_radar_multimetrica', fig_05_radar_multimetrica),
        ('06_heatmap_classification', fig_06_heatmap_classification),
        ('07_ranking_metricas', fig_07_ranking_metricas),
        ('08_scatter_3d', fig_08_scatter_3d),
    ]

    for nombre, fn in graficas:
        print(f'  --- {nombre} ---')
        try:
            fn(modelos)
        except Exception as e:
            print(f'  [ERR] {nombre} fallo: {e}')

    print('=' * 55)
    print(f'  Listo. {len(os.listdir(FIGURES_DIR))} graficas en:')
    print(f'    {FIGURES_DIR}')
    print('=' * 55)


if __name__ == '__main__':
    main()
