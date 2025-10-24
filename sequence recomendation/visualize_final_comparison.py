#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Визуализация финального сравнения всех версий на реальных путях
"""

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from pathlib import Path

# Настройка стиля
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (14, 10)
plt.rcParams['font.size'] = 11

# Данные результатов
results = {
    'v3.0 DAGNN': {'accuracy': 1.0000, 'ndcg': 1.0000, 'f1': 1.0000, 'precision': 1.0000, 'recall': 1.0000},
    'v3.0 Random Forest': {'accuracy': 0.8333, 'ndcg': 0.9167, 'f1': 0.4545, 'precision': 0.4167, 'recall': 0.5000},
    'v3.0 Popularity': {'accuracy': 0.8333, 'ndcg': 0.9201, 'f1': 0.4545, 'precision': 0.4167, 'recall': 0.5000},
    'v3.0 GraphSAGE': {'accuracy': 0.8333, 'ndcg': 0.9385, 'f1': 0.3333, 'precision': 0.3333, 'recall': 0.3333},
    'v3.1 All Models': {'accuracy': 0.8333, 'ndcg': 0.9276, 'f1': 0.3788, 'precision': 0.3712, 'recall': 0.4091},
    'v4.1 Personalized': {'accuracy': 0.8333, 'ndcg': 0.9385, 'f1': 0.3333, 'precision': 0.3333, 'recall': 0.3333},
}

# Создаем фигуру с 4 графиками
fig, axes = plt.subplots(2, 2, figsize=(16, 12))
fig.suptitle('Финальное сравнение всех версий на РЕАЛЬНЫХ путях из композиций', 
             fontsize=16, fontweight='bold', y=0.995)

# 1. Accuracy comparison
ax1 = axes[0, 0]
models = list(results.keys())
accuracies = [results[m]['accuracy'] for m in models]
colors = ['#2ecc71' if acc == 1.0 else '#3498db' if acc > 0.83 else '#95a5a6' for acc in accuracies]

bars1 = ax1.barh(models, accuracies, color=colors, alpha=0.8, edgecolor='black')
ax1.set_xlabel('Accuracy', fontweight='bold')
ax1.set_title('Accuracy по версиям', fontweight='bold', fontsize=13)
ax1.set_xlim([0, 1.05])
ax1.axvline(x=1.0, color='red', linestyle='--', alpha=0.5, label='Perfect Score')

# Добавляем значения на бары
for i, (bar, acc) in enumerate(zip(bars1, accuracies)):
    width = bar.get_width()
    label = f'{acc:.1%}'
    if acc == 1.0:
        label += ' 🏆'
    ax1.text(width + 0.01, bar.get_y() + bar.get_height()/2, label,
             ha='left', va='center', fontweight='bold' if acc == 1.0 else 'normal')

ax1.grid(axis='x', alpha=0.3)
ax1.legend()

# 2. nDCG comparison  
ax2 = axes[0, 1]
ndcgs = [results[m]['ndcg'] for m in models]
colors2 = ['#2ecc71' if n == 1.0 else '#3498db' if n > 0.93 else '#95a5a6' for n in ndcgs]

bars2 = ax2.barh(models, ndcgs, color=colors2, alpha=0.8, edgecolor='black')
ax2.set_xlabel('nDCG', fontweight='bold')
ax2.set_title('nDCG по версиям', fontweight='bold', fontsize=13)
ax2.set_xlim([0, 1.05])
ax2.axvline(x=1.0, color='red', linestyle='--', alpha=0.5, label='Perfect Score')

for i, (bar, ndcg) in enumerate(zip(bars2, ndcgs)):
    width = bar.get_width()
    label = f'{ndcg:.1%}'
    if ndcg == 1.0:
        label += ' 🏆'
    ax2.text(width + 0.01, bar.get_y() + bar.get_height()/2, label,
             ha='left', va='center', fontweight='bold' if ndcg == 1.0 else 'normal')

ax2.grid(axis='x', alpha=0.3)
ax2.legend()

# 3. Radar chart для v3.0 DAGNN
ax3 = axes[1, 0]
metrics = ['Accuracy', 'nDCG', 'F1', 'Precision', 'Recall']
dagnn_values = [
    results['v3.0 DAGNN']['accuracy'],
    results['v3.0 DAGNN']['ndcg'],
    results['v3.0 DAGNN']['f1'],
    results['v3.0 DAGNN']['precision'],
    results['v3.0 DAGNN']['recall']
]

angles = np.linspace(0, 2 * np.pi, len(metrics), endpoint=False).tolist()
dagnn_values += dagnn_values[:1]
angles += angles[:1]

ax3 = plt.subplot(2, 2, 3, projection='polar')
ax3.plot(angles, dagnn_values, 'o-', linewidth=2, color='#2ecc71', label='v3.0 DAGNN')
ax3.fill(angles, dagnn_values, alpha=0.25, color='#2ecc71')
ax3.set_xticks(angles[:-1])
ax3.set_xticklabels(metrics, fontsize=10)
ax3.set_ylim(0, 1.0)
ax3.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
ax3.set_title('v3.0 DAGNN - Все метрики = 100% 🏆', fontweight='bold', fontsize=13, pad=20)
ax3.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))
ax3.grid(True)

# 4. Сравнение версий
ax4 = axes[1, 1]
versions = ['v3.0\nDAGNN', 'v3.0\nОстальные', 'v3.1\nВсе', 'v4.1\nPers.']
version_acc = [1.0000, 0.8333, 0.8333, 0.8333]
version_ndcg = [1.0000, 0.9250, 0.9276, 0.9385]

x = np.arange(len(versions))
width = 0.35

bars_acc = ax4.bar(x - width/2, version_acc, width, label='Accuracy', 
                   color='#3498db', alpha=0.8, edgecolor='black')
bars_ndcg = ax4.bar(x + width/2, version_ndcg, width, label='nDCG',
                    color='#e74c3c', alpha=0.8, edgecolor='black')

ax4.set_ylabel('Score', fontweight='bold')
ax4.set_title('Сравнение версий: Accuracy vs nDCG', fontweight='bold', fontsize=13)
ax4.set_xticks(x)
ax4.set_xticklabels(versions)
ax4.set_ylim([0, 1.1])
ax4.legend()
ax4.grid(axis='y', alpha=0.3)

# Добавляем значения на бары
for bar in bars_acc:
    height = bar.get_height()
    label = f'{height:.1%}'
    if height == 1.0:
        label += '\n🏆'
    ax4.text(bar.get_x() + bar.get_width()/2, height + 0.02, label,
             ha='center', va='bottom', fontsize=9, fontweight='bold' if height == 1.0 else 'normal')

for bar in bars_ndcg:
    height = bar.get_height()
    label = f'{height:.1%}'
    if height == 1.0:
        label += '\n🏆'
    ax4.text(bar.get_x() + bar.get_width()/2, height + 0.02, label,
             ha='center', va='bottom', fontsize=9, fontweight='bold' if height == 1.0 else 'normal')

plt.tight_layout()

# Сохраняем
output_dir = Path('images')
output_dir.mkdir(exist_ok=True)
output_path = output_dir / 'final_comparison_real_paths.png'
plt.savefig(output_path, dpi=300, bbox_inches='tight')
print(f"✅ Saved: {output_path}")

plt.show()

print("\n" + "="*60)
print("🏆 ИТОГОВЫЕ РЕЗУЛЬТАТЫ НА РЕАЛЬНЫХ ПУТЯХ")
print("="*60)
print(f"v3.0 DAGNN: Accuracy={results['v3.0 DAGNN']['accuracy']:.1%}, nDCG={results['v3.0 DAGNN']['ndcg']:.1%} 🏆")
print(f"v3.1 Average: Accuracy=83.33%, nDCG=~93%")
print(f"v4.1 Personalized: Accuracy={results['v4.1 Personalized']['accuracy']:.1%}, nDCG={results['v4.1 Personalized']['ndcg']:.1%}")
print("="*60)

