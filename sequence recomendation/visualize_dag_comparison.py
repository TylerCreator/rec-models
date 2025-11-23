#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Визуализация сравнения Directed DAG моделей
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Создаёт графики сравнения всех моделей из directed_dag_models.py,
включая новую DA-GCN модель.
"""

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from pathlib import Path

# Устанавливаем стиль
sns.set_style("whitegrid")
plt.rcParams['font.size'] = 10
plt.rcParams['figure.figsize'] = (14, 10)

# Результаты (150 эпох, hidden=64, dropout=0.4)
models = {
    'Popularity': {
        'accuracy': 0.4787,
        'ndcg': 0.6106,
        'f1': 0.0540,
        'precision': 0.0399,
        'recall': 0.0833,
        'type': 'Baseline',
        'year': '-'
    },
    'DirectedDAGNN': {
        'accuracy': 0.5213,
        'ndcg': 0.7566,
        'f1': 0.1805,
        'precision': 0.1675,
        'recall': 0.2250,
        'type': 'GNN',
        'year': '2020'
    },
    'DA-GCN': {
        'accuracy': 0.5115,
        'ndcg': 0.7530,
        'f1': 0.1138,
        'precision': 0.0864,
        'recall': 0.1667,
        'type': 'GNN',
        'year': '2024'
    },
    'DeepDAG2022': {
        'accuracy': 0.5213,
        'ndcg': 0.7571,
        'f1': 0.1805,
        'precision': 0.1675,
        'recall': 0.2250,
        'type': 'GNN',
        'year': '2022'
    },
    'DAG-GNN': {
        'accuracy': 0.5213,
        'ndcg': 0.7571,
        'f1': 0.1805,
        'precision': 0.1675,
        'recall': 0.2250,
        'type': 'GNN',
        'year': '2019'
    },
    'DAGNN2021': {
        'accuracy': 0.5213,
        'ndcg': 0.7571,
        'f1': 0.1805,
        'precision': 0.1675,
        'recall': 0.2250,
        'type': 'GNN',
        'year': '2021'
    },
    'GRU4Rec': {
        'accuracy': 0.5344,
        'ndcg': 0.7674,
        'f1': 0.2241,
        'precision': 0.3219,
        'recall': 0.2625,
        'type': 'RNN',
        'year': '2016'
    }
}

# Цветовая схема
colors = {
    'Popularity': '#95a5a6',
    'DirectedDAGNN': '#3498db',
    'DA-GCN': '#e74c3c',  # Красный для выделения новой модели
    'DeepDAG2022': '#2ecc71',
    'DAG-GNN': '#9b59b6',
    'DAGNN2021': '#f39c12',
    'GRU4Rec': '#34495e'
}

def create_comparison_plots():
    """Создаёт комплексное сравнение моделей."""
    fig = plt.figure(figsize=(16, 12))
    
    # 1. Основные метрики (столбчатая диаграмма)
    ax1 = plt.subplot(2, 3, 1)
    metrics = ['accuracy', 'ndcg', 'f1']
    metric_names = ['Accuracy', 'NDCG@10', 'F1-Score']
    x = np.arange(len(models))
    width = 0.25
    
    for i, (metric, name) in enumerate(zip(metrics, metric_names)):
        values = [models[m][metric] for m in models.keys()]
        ax1.bar(x + i * width, values, width, label=name, alpha=0.8)
    
    ax1.set_xlabel('Модели')
    ax1.set_ylabel('Значение метрики')
    ax1.set_title('Основные метрики (150 эпох)')
    ax1.set_xticks(x + width)
    ax1.set_xticklabels(models.keys(), rotation=45, ha='right')
    ax1.legend()
    ax1.grid(axis='y', alpha=0.3)
    
    # 2. Accuracy vs NDCG (scatter plot)
    ax2 = plt.subplot(2, 3, 2)
    for name, data in models.items():
        ax2.scatter(data['accuracy'], data['ndcg'], 
                   s=200, c=colors[name], alpha=0.7, 
                   edgecolors='black', linewidth=1.5,
                   label=name)
        # Аннотация для DA-GCN
        if name == 'DA-GCN':
            ax2.annotate('NEW!', xy=(data['accuracy'], data['ndcg']),
                        xytext=(data['accuracy'] - 0.01, data['ndcg'] + 0.015),
                        fontsize=9, fontweight='bold', color='red',
                        bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.7))
    
    ax2.set_xlabel('Accuracy')
    ax2.set_ylabel('NDCG@10')
    ax2.set_title('Accuracy vs NDCG (больше = лучше)')
    ax2.legend(loc='lower right', fontsize=8)
    ax2.grid(alpha=0.3)
    
    # 3. Precision-Recall
    ax3 = plt.subplot(2, 3, 3)
    for name, data in models.items():
        marker = 'D' if name == 'DA-GCN' else 'o'
        markersize = 12 if name == 'DA-GCN' else 10
        ax3.scatter(data['recall'], data['precision'],
                   s=200, c=colors[name], alpha=0.7,
                   marker=marker, edgecolors='black', linewidth=1.5,
                   label=name)
    
    ax3.set_xlabel('Recall')
    ax3.set_ylabel('Precision')
    ax3.set_title('Precision-Recall Trade-off')
    ax3.legend(loc='best', fontsize=8)
    ax3.grid(alpha=0.3)
    
    # 4. Radar chart (нормализованные метрики)
    ax4 = plt.subplot(2, 3, 4, projection='polar')
    
    categories = ['Accuracy', 'NDCG', 'F1', 'Precision', 'Recall']
    metrics_keys = ['accuracy', 'ndcg', 'f1', 'precision', 'recall']
    N = len(categories)
    angles = [n / float(N) * 2 * np.pi for n in range(N)]
    angles += angles[:1]
    
    # Показываем только лучшие модели для читаемости
    selected_models = ['DA-GCN', 'GRU4Rec', 'DirectedDAGNN', 'DeepDAG2022']
    
    for name in selected_models:
        values = [models[name][k] for k in metrics_keys]
        values += values[:1]
        ax4.plot(angles, values, 'o-', linewidth=2, label=name, color=colors[name])
        ax4.fill(angles, values, alpha=0.15, color=colors[name])
    
    ax4.set_xticks(angles[:-1])
    ax4.set_xticklabels(categories, fontsize=9)
    ax4.set_ylim(0, 0.8)
    ax4.set_title('Radar Chart: Топ-4 модели', y=1.08, fontsize=11, fontweight='bold')
    ax4.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0), fontsize=9)
    ax4.grid(True)
    
    # 5. Heatmap метрик
    ax5 = plt.subplot(2, 3, 5)
    metric_matrix = []
    for name in models.keys():
        metric_matrix.append([
            models[name]['accuracy'],
            models[name]['ndcg'],
            models[name]['f1'],
            models[name]['precision'],
            models[name]['recall']
        ])
    
    im = ax5.imshow(metric_matrix, cmap='YlOrRd', aspect='auto')
    ax5.set_xticks(np.arange(5))
    ax5.set_yticks(np.arange(len(models)))
    ax5.set_xticklabels(['Acc', 'NDCG', 'F1', 'Prec', 'Rec'])
    ax5.set_yticklabels(models.keys())
    ax5.set_title('Heatmap всех метрик')
    
    # Добавляем значения в ячейки
    for i in range(len(models)):
        for j in range(5):
            text = ax5.text(j, i, f'{metric_matrix[i][j]:.3f}',
                          ha="center", va="center", color="black", fontsize=8)
    
    plt.colorbar(im, ax=ax5)
    
    # 6. Сравнение по годам
    ax6 = plt.subplot(2, 3, 6)
    
    years = []
    ndcgs = []
    names = []
    for name, data in models.items():
        if data['year'] != '-':
            years.append(int(data['year']))
            ndcgs.append(data['ndcg'])
            names.append(name)
    
    # Сортируем по годам
    sorted_indices = np.argsort(years)
    years = [years[i] for i in sorted_indices]
    ndcgs = [ndcgs[i] for i in sorted_indices]
    names = [names[i] for i in sorted_indices]
    
    colors_timeline = [colors[n] for n in names]
    
    ax6.plot(years, ndcgs, '--', linewidth=1.5, color='gray', alpha=0.5, zorder=1)
    for i, (year, ndcg, name) in enumerate(zip(years, ndcgs, names)):
        marker = 'D' if name == 'DA-GCN' else 'o'
        size = 250 if name == 'DA-GCN' else 150
        ax6.scatter(year, ndcg, s=size, c=colors[name], alpha=0.8, 
                   marker=marker, edgecolors='black', linewidth=2, zorder=10)
        # Упрощённая аннотация без проблемных смещений
        if name == 'DA-GCN':
            ax6.text(year, ndcg + 0.02, name, ha='center', fontsize=9,
                    fontweight='bold', color='red',
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.7))
        elif i % 2 == 0:
            ax6.text(year, ndcg - 0.025, name, ha='center', fontsize=7, rotation=15)
        else:
            ax6.text(year, ndcg + 0.015, name, ha='center', fontsize=7, rotation=15)
    
    ax6.set_xlabel('Год публикации')
    ax6.set_ylabel('NDCG@10')
    ax6.set_title('Эволюция моделей по годам')
    ax6.grid(alpha=0.3)
    ax6.set_ylim(0.6, 0.8)
    
    plt.tight_layout(pad=1.5)
    
    # Сохраняем
    output_path = Path(__file__).parent / 'images'
    output_path.mkdir(exist_ok=True)
    plt.savefig(output_path / 'dag_models_comparison_with_dagcn.png', dpi=300, bbox_inches='tight')
    print(f"✅ График сохранён: {output_path / 'dag_models_comparison_with_dagcn.png'}")
    
    plt.show()


def create_summary_table():
    """Выводит итоговую таблицу в консоль."""
    print("\n" + "="*80)
    print("ИТОГОВОЕ СРАВНЕНИЕ DIRECTED DAG МОДЕЛЕЙ (150 эпох)")
    print("="*80)
    print(f"{'Модель':<20} {'Type':<10} {'Year':<6} {'Acc':<8} {'NDCG':<8} {'F1':<8}")
    print("-"*80)
    
    # Сортируем по NDCG
    sorted_models = sorted(models.items(), key=lambda x: x[1]['ndcg'], reverse=True)
    
    for i, (name, data) in enumerate(sorted_models):
        marker = '⭐ ' if name == 'DA-GCN' else f"{i+1}. "
        print(f"{marker}{name:<18} {data['type']:<10} {data['year']:<6} "
              f"{data['accuracy']:.4f}   {data['ndcg']:.4f}   {data['f1']:.4f}")
    
    print("="*80)
    print("\n🔍 Ключевые наблюдения:")
    print("  • GRU4Rec показывает лучшие результаты по всем метрикам")
    print("  • DA-GCN (2024) показывает конкурентоспособный NDCG@10 = 0.7530")
    print("  • DirectedDAGNN, DeepDAG2022, DAG-GNN и DAGNN2021 имеют схожие результаты")
    print("  • Направленное кодирование рёбер (DA-GCN) эффективно для DAG структур")
    print()


if __name__ == '__main__':
    print("🎨 Создание визуализации сравнения моделей...")
    create_summary_table()
    create_comparison_plots()

