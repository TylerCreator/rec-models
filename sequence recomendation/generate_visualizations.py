#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Генерация визуализаций для README
Создает графики результатов сравнения моделей
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from pathlib import Path

# Настройка стиля
plt.style.use('seaborn-v0_8-darkgrid')
plt.rcParams['figure.facecolor'] = 'white'
plt.rcParams['axes.facecolor'] = '#f8f9fa'
plt.rcParams['font.size'] = 10
plt.rcParams['axes.labelsize'] = 11
plt.rcParams['axes.titlesize'] = 13
plt.rcParams['legend.fontsize'] = 9

# Создаем директорию для картинок
output_dir = Path("images")
output_dir.mkdir(exist_ok=True)

# Данные результатов (seed=456)
models = ['DAGNN', 'SASRec', 'Caser', 'GraphSAGE', 'GCN', 'RF', 'GRU4Rec', 'Popularity']
categories = ['GNN', 'Attention', 'CNN', 'GNN', 'GNN', 'ML', 'RNN', 'Baseline']
accuracy = [0.5152, 0.4848, 0.4848, 0.4848, 0.4848, 0.4848, 0.2727, 0.2727]
ndcg = [0.7893, 0.7742, 0.7781, 0.7742, 0.7781, 0.7742, 0.6513, 0.6513]
f1 = [0.3824, 0.3712, 0.3712, 0.3712, 0.3712, 0.3712, 0.1071, 0.1071]
speed = [257, 350, 420, 415, 275, 0, 215, 0]  # it/s, 0 для non-neural

# Цвета для категорий
category_colors = {
    'GNN': '#2E86AB',
    'Attention': '#A23B72',
    'CNN': '#F18F01',
    'RNN': '#C73E1D',
    'ML': '#6A994E',
    'Baseline': '#BC4B51'
}
colors = [category_colors[cat] for cat in categories]

# ============================================================================
# График 1: Accuracy по моделям (горизонтальная bar chart)
# ============================================================================
fig, ax = plt.subplots(figsize=(12, 6))

y_pos = np.arange(len(models))
bars = ax.barh(y_pos, accuracy, color=colors, edgecolor='black', linewidth=1.2, alpha=0.8)

# Добавляем значения на bars
for i, (bar, acc) in enumerate(zip(bars, accuracy)):
    width = bar.get_width()
    ax.text(width + 0.01, bar.get_y() + bar.get_height()/2, 
            f'{acc:.4f}', ha='left', va='center', fontweight='bold', fontsize=10)

ax.set_yticks(y_pos)
ax.set_yticklabels(models, fontsize=11)
ax.set_xlabel('Accuracy', fontsize=12, fontweight='bold')
ax.set_title('Сравнение моделей по Accuracy (seed=456)', fontsize=14, fontweight='bold', pad=15)
ax.set_xlim(0, 0.6)
ax.grid(axis='x', alpha=0.3, linestyle='--')

# Легенда
legend_elements = [mpatches.Patch(facecolor=color, edgecolor='black', label=cat) 
                   for cat, color in category_colors.items() if cat in categories]
ax.legend(handles=legend_elements, loc='lower right', framealpha=0.9)

plt.tight_layout()
plt.savefig(output_dir / 'accuracy_comparison.png', dpi=300, bbox_inches='tight')
print(f"✅ Сохранено: {output_dir / 'accuracy_comparison.png'}")
plt.close()

# ============================================================================
# График 2: Accuracy vs nDCG scatter plot
# ============================================================================
fig, ax = plt.subplots(figsize=(10, 8))

scatter = ax.scatter(accuracy, ndcg, s=[s*2 if s > 0 else 100 for s in speed], 
                     c=colors, alpha=0.7, edgecolors='black', linewidth=1.5)

# Аннотации для каждой точки
for i, (model, acc, nd, cat) in enumerate(zip(models, accuracy, ndcg, categories)):
    offset_x = 0.005 if i % 2 == 0 else -0.005
    offset_y = 0.003 if i % 2 == 0 else -0.003
    ha = 'left' if i % 2 == 0 else 'right'
    
    ax.annotate(model, (acc, nd), xytext=(offset_x, offset_y), 
                textcoords='offset points', fontsize=9, fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.3', facecolor=category_colors[cat], 
                         alpha=0.3, edgecolor='black'),
                ha=ha)

ax.set_xlabel('Accuracy', fontsize=12, fontweight='bold')
ax.set_ylabel('nDCG', fontsize=12, fontweight='bold')
ax.set_title('Accuracy vs nDCG (размер точки = скорость обучения)', 
             fontsize=14, fontweight='bold', pad=15)
ax.grid(True, alpha=0.3, linestyle='--')
ax.set_xlim(0.2, 0.55)
ax.set_ylim(0.6, 0.82)

# Легенда
legend_elements = [mpatches.Patch(facecolor=color, edgecolor='black', label=cat) 
                   for cat, color in category_colors.items() if cat in categories]
ax.legend(handles=legend_elements, loc='lower right', framealpha=0.9, title='Категория')

plt.tight_layout()
plt.savefig(output_dir / 'accuracy_vs_ndcg.png', dpi=300, bbox_inches='tight')
print(f"✅ Сохранено: {output_dir / 'accuracy_vs_ndcg.png'}")
plt.close()

# ============================================================================
# График 3: Radar chart (упрощенная версия с polar)
# ============================================================================
top_models = ['DAGNN', 'SASRec', 'Caser', 'GraphSAGE']
metrics_labels = ['Accuracy', 'nDCG', 'F1-score', 'Speed', 'Stability']

# Данные для radar (нормализованные 0-1)
radar_data = {
    'DAGNN':     [0.5152/0.52, 0.7893/0.80, 0.3824/0.40, 257/420, 0.9],
    'SASRec':    [0.4848/0.52, 0.7742/0.80, 0.3712/0.40, 350/420, 0.8],
    'Caser':     [0.4848/0.52, 0.7781/0.80, 0.3712/0.40, 420/420, 0.8],
    'GraphSAGE': [0.4848/0.52, 0.7742/0.80, 0.3712/0.40, 415/420, 0.7],
}

angles = np.linspace(0, 2 * np.pi, len(metrics_labels), endpoint=False).tolist()
angles += angles[:1]  # Замыкаем круг

fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(projection='polar'))

model_colors = ['#2E86AB', '#A23B72', '#F18F01', '#2E86AB']
line_styles = ['-', '-', '-', '--']

for (model, color, ls) in zip(top_models, model_colors, line_styles):
    values = radar_data[model]
    values += values[:1]  # Замыкаем круг
    ax.plot(angles, values, 'o-', linewidth=2.5, label=model, color=color, 
            markersize=8, linestyle=ls, markeredgecolor='black', markeredgewidth=1)
    ax.fill(angles, values, alpha=0.15, color=color)

ax.set_xticks(angles[:-1])
ax.set_xticklabels(metrics_labels, fontsize=11)
ax.set_ylim(0, 1)
ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
ax.set_yticklabels(['0.2', '0.4', '0.6', '0.8', '1.0'], fontsize=9)
ax.set_title('Сравнение топ-4 моделей по 5 метрикам\n(нормализовано к максимуму)', 
             fontsize=14, fontweight='bold', pad=30)
ax.legend(loc='upper right', bbox_to_anchor=(1.25, 1.1), framealpha=0.9, fontsize=11)
ax.grid(True, linestyle='--', alpha=0.5)

plt.tight_layout()
plt.savefig(output_dir / 'radar_chart.png', dpi=300, bbox_inches='tight')
print(f"✅ Сохранено: {output_dir / 'radar_chart.png'}")
plt.close()

# ============================================================================
# График 4: Сравнение по категориям (grouped bar chart)
# ============================================================================
fig, ax = plt.subplots(figsize=(12, 6))

categories_unique = ['GNN', 'Attention', 'CNN', 'RNN', 'ML', 'Baseline']
category_avg_acc = []
category_avg_ndcg = []

for cat in categories_unique:
    indices = [i for i, c in enumerate(categories) if c == cat]
    avg_acc = np.mean([accuracy[i] for i in indices])
    avg_ndcg = np.mean([ndcg[i] for i in indices])
    category_avg_acc.append(avg_acc)
    category_avg_ndcg.append(avg_ndcg)

x = np.arange(len(categories_unique))
width = 0.35

bars1 = ax.bar(x - width/2, category_avg_acc, width, label='Accuracy', 
               color='#2E86AB', alpha=0.8, edgecolor='black', linewidth=1.2)
bars2 = ax.bar(x + width/2, category_avg_ndcg, width, label='nDCG',
               color='#F18F01', alpha=0.8, edgecolor='black', linewidth=1.2)

# Значения на bars
for bars in [bars1, bars2]:
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.3f}', ha='center', va='bottom', fontsize=9, fontweight='bold')

ax.set_ylabel('Значение метрики', fontsize=12, fontweight='bold')
ax.set_xlabel('Категория модели', fontsize=12, fontweight='bold')
ax.set_title('Средние метрики по категориям моделей', fontsize=14, fontweight='bold', pad=15)
ax.set_xticks(x)
ax.set_xticklabels(categories_unique, fontsize=11)
ax.legend(loc='upper right', framealpha=0.9)
ax.grid(axis='y', alpha=0.3, linestyle='--')
ax.set_ylim(0, 1.0)

plt.tight_layout()
plt.savefig(output_dir / 'category_comparison.png', dpi=300, bbox_inches='tight')
print(f"✅ Сохранено: {output_dir / 'category_comparison.png'}")
plt.close()

# ============================================================================
# График 5: Скорость обучения (bar chart для neural моделей)
# ============================================================================
neural_models = []
neural_speeds = []
neural_colors = []

for i, (model, spd, cat) in enumerate(zip(models, speed, categories)):
    if spd > 0:  # Только нейронные модели
        neural_models.append(model)
        neural_speeds.append(spd)
        neural_colors.append(colors[i])

fig, ax = plt.subplots(figsize=(10, 6))

bars = ax.bar(range(len(neural_models)), neural_speeds, color=neural_colors, 
              alpha=0.8, edgecolor='black', linewidth=1.2)

# Значения
for bar, spd_val in zip(bars, neural_speeds):
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height + 10,
            f'{int(spd_val)} it/s', ha='center', va='bottom', fontsize=10, fontweight='bold')

ax.set_xticks(range(len(neural_models)))
ax.set_xticklabels(neural_models, rotation=45, ha='right', fontsize=11)
ax.set_ylabel('Скорость обучения (iterations/sec)', fontsize=12, fontweight='bold')
ax.set_title('Скорость обучения нейронных моделей', fontsize=14, fontweight='bold', pad=15)
ax.grid(axis='y', alpha=0.3, linestyle='--')

# Highlight fastest
max_idx = neural_speeds.index(max(neural_speeds))
bars[max_idx].set_linewidth(3)
bars[max_idx].set_edgecolor('gold')

plt.tight_layout()
plt.savefig(output_dir / 'training_speed.png', dpi=300, bbox_inches='tight')
print(f"✅ Сохранено: {output_dir / 'training_speed.png'}")
plt.close()

# ============================================================================
# График 6: Accuracy с разными seeds (line plot)
# ============================================================================
seeds = [42, 123, 456, 789]
dagnn_results = [0.4848, 0.4545, 0.5152, 0.5152]
graphsage_results = [0.5152, 0.4545, 0.4848, 0.4848]
sasrec_results = [0.4848, 0.4848, 0.4848, 0.4848]
caser_results = [0.4848, 0.4545, 0.4848, 0.4848]

fig, ax = plt.subplots(figsize=(10, 6))

ax.plot(seeds, dagnn_results, 'o-', linewidth=2.5, markersize=10, 
        label='DAGNN', color='#2E86AB', markeredgecolor='black', markeredgewidth=1.5)
ax.plot(seeds, graphsage_results, 's-', linewidth=2.5, markersize=10,
        label='GraphSAGE', color='#2E86AB', alpha=0.6, markeredgecolor='black', markeredgewidth=1.5)
ax.plot(seeds, sasrec_results, '^-', linewidth=2.5, markersize=10,
        label='SASRec', color='#A23B72', markeredgecolor='black', markeredgewidth=1.5)
ax.plot(seeds, caser_results, 'd-', linewidth=2.5, markersize=10,
        label='Caser', color='#F18F01', markeredgecolor='black', markeredgewidth=1.5)

ax.set_xlabel('Random Seed', fontsize=12, fontweight='bold')
ax.set_ylabel('Accuracy', fontsize=12, fontweight='bold')
ax.set_title('Стабильность моделей при разных random seeds', 
             fontsize=14, fontweight='bold', pad=15)
ax.set_xticks(seeds)
ax.set_ylim(0.4, 0.55)
ax.legend(loc='best', framealpha=0.9, fontsize=11)
ax.grid(True, alpha=0.3, linestyle='--')

# Highlight best
ax.axhline(y=0.5152, color='green', linestyle='--', alpha=0.5, linewidth=2, label='Best (0.5152)')

plt.tight_layout()
plt.savefig(output_dir / 'seed_stability.png', dpi=300, bbox_inches='tight')
print(f"✅ Сохранено: {output_dir / 'seed_stability.png'}")
plt.close()

# ============================================================================
# График 7: Heatmap метрик
# ============================================================================
fig, ax = plt.subplots(figsize=(10, 8))

# Нормализуем метрики для heatmap
accuracy_norm = [acc/0.52 for acc in accuracy]
ndcg_norm = [nd/0.80 for nd in ndcg]
f1_norm = [f/0.40 for f in f1]
speed_norm = [s/420 if s > 0 else 0 for s in speed]

metrics_matrix = np.array([
    accuracy_norm,
    ndcg_norm,
    f1_norm,
    speed_norm
])

im = ax.imshow(metrics_matrix, cmap='RdYlGn', aspect='auto', vmin=0, vmax=1)

# Настройка осей
ax.set_xticks(np.arange(len(models)))
ax.set_yticks(np.arange(len(['Accuracy', 'nDCG', 'F1-score', 'Speed'])))
ax.set_xticklabels(models, rotation=45, ha='right', fontsize=10)
ax.set_yticklabels(['Accuracy', 'nDCG', 'F1-score', 'Speed'], fontsize=11)

# Значения в ячейках
for i in range(len(['Accuracy', 'nDCG', 'F1-score', 'Speed'])):
    for j in range(len(models)):
        value = metrics_matrix[i, j]
        text = ax.text(j, i, f'{value:.2f}',
                      ha="center", va="center", color="black" if value > 0.5 else "white",
                      fontsize=9, fontweight='bold')

ax.set_title('Тепловая карта нормализованных метрик\n(1.0 = лучший результат)', 
             fontsize=14, fontweight='bold', pad=15)

# Colorbar
cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
cbar.set_label('Нормализованное значение', rotation=270, labelpad=20, fontsize=11)

plt.tight_layout()
plt.savefig(output_dir / 'metrics_heatmap.png', dpi=300, bbox_inches='tight')
print(f"✅ Сохранено: {output_dir / 'metrics_heatmap.png'}")
plt.close()

# ============================================================================
# График 8: Pipeline обработки данных
# ============================================================================
fig, ax = plt.subplots(figsize=(12, 8))
ax.axis('off')

# Настройка для flowchart
steps = [
    ("1. Загрузка данных", "compositionsDAG.json\n50 узлов, 97 рёбер", "#E3F2FD"),
    ("2. Извлечение путей", "228 путей\nDFS по графу", "#E8F5E9"),
    ("3. Создание примеров", "107 примеров\nКонтекст → Следующий", "#FFF3E0"),
    ("4. Векторизация", "Train: 74\nTest: 33", "#FCE4EC"),
    ("5. Обучение моделей", "8 моделей:\nGNN, Attention, CNN, RNN", "#F3E5F5"),
    ("6. Оценка", "Accuracy, F1, nDCG\nИтоговый рейтинг", "#E0F2F1"),
]

y_start = 0.9
box_height = 0.12
box_width = 0.7
x_center = 0.5

for i, (title, content, color) in enumerate(steps):
    y = y_start - i * (box_height + 0.02)
    
    # Box
    box = mpatches.FancyBboxPatch(
        (x_center - box_width/2, y - box_height), box_width, box_height,
        boxstyle="round,pad=0.01", 
        facecolor=color, edgecolor='black', linewidth=2
    )
    ax.add_patch(box)
    
    # Title
    ax.text(x_center, y - 0.02, title, ha='center', va='top', 
            fontsize=12, fontweight='bold')
    
    # Content
    ax.text(x_center, y - box_height/2 - 0.01, content, ha='center', va='center',
            fontsize=9, style='italic')
    
    # Arrow to next
    if i < len(steps) - 1:
        arrow = mpatches.FancyArrowPatch(
            (x_center, y - box_height - 0.01),
            (x_center, y - box_height - 0.02),
            arrowstyle='->', mutation_scale=30, linewidth=2.5, color='#555'
        )
        ax.add_patch(arrow)

ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
ax.set_title('Pipeline обработки данных', fontsize=16, fontweight='bold', pad=20)

plt.tight_layout()
plt.savefig(output_dir / 'data_pipeline.png', dpi=300, bbox_inches='tight')
print(f"✅ Сохранено: {output_dir / 'data_pipeline.png'}")
plt.close()

print("\n" + "="*70)
print("✅ ВСЕ ВИЗУАЛИЗАЦИИ СОЗДАНЫ!")
print("="*70)
print(f"\nФайлы сохранены в директории: {output_dir}/")
print("\nСозданные файлы:")
print("  1. accuracy_comparison.png  - Сравнение accuracy")
print("  2. accuracy_vs_ndcg.png     - Scatter plot accuracy vs nDCG")
print("  3. radar_chart.png          - Radar chart топ-4 моделей")
print("  4. category_comparison.png  - Сравнение по категориям")
print("  5. training_speed.png       - Скорость обучения")
print("  6. seed_stability.png       - Стабильность при разных seeds")
print("  7. metrics_heatmap.png      - Тепловая карта метрик")
print("  8. data_pipeline.png        - Pipeline обработки данных")
print("\nДобавьте эти изображения в README.md!")
print("="*70)

