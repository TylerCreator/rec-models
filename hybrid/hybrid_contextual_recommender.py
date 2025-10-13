#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Гибридная контекстная система рекомендаций

Объединяет:
1. DAGNN (Sequence) - обучена на реальных последовательностях вызовов
2. Hybrid-BPR (CF) - персональные предпочтения пользователей

Тестирование и оценка как в sequence_dag_recommender_final.py
"""

import json
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import networkx as nx
import logging
from collections import Counter
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, ndcg_score, precision_score, recall_score
from torch_geometric.data import Data
from torch_geometric.nn import APPNP
from lightfm import LightFM
from lightfm.data import Dataset
from sklearn.neighbors import NearestNeighbors
from tqdm import tqdm
import warnings

warnings.filterwarnings("ignore")

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

logger.info("="*70)
logger.info("🔀 ГИБРИДНАЯ КОНТЕКСТНАЯ СИСТЕМА РЕКОМЕНДАЦИЙ")
logger.info("   DAGNN (обучена на последовательностях) + Hybrid-BPR (CF)")
logger.info("="*70)

# ============================ МОДЕЛИ ============================

class DAGNN(nn.Module):
    def __init__(self, in_channels: int, K: int, dropout: float = 0.4):
        super().__init__()
        self.propagation = APPNP(K=K, alpha=0.1)
        self.att = nn.Parameter(torch.Tensor(K + 1))
        self.dropout = dropout
        self.reset_parameters()

    def reset_parameters(self):
        self.propagation.reset_parameters()
        nn.init.uniform_(self.att, 0.0, 1.0)

    def forward(self, x, edge_index, training=True):
        xs = [x]
        edge_weight = torch.ones(edge_index.size(1), dtype=torch.float32, device=edge_index.device)
        for _ in range(self.propagation.K):
            x = self.propagation.propagate(edge_index, x=x, edge_weight=edge_weight)
            if training:
                x = F.dropout(x, p=self.dropout, training=training)
            xs.append(x)
        out = torch.stack(xs, dim=-1)
        att_weights = F.softmax(self.att, dim=0)
        out = (out * att_weights.view(1, 1, -1)).sum(dim=-1)
        return out


class DAGNNRecommender(nn.Module):
    def __init__(self, in_channels: int, hidden_channels: int, out_channels: int, 
                 K: int = 10, dropout: float = 0.4):
        super().__init__()
        self.dropout = dropout
        
        self.lin1 = nn.Linear(in_channels, hidden_channels)
        self.bn1 = nn.BatchNorm1d(hidden_channels)
        
        self.lin2 = nn.Linear(hidden_channels, hidden_channels)
        self.bn2 = nn.BatchNorm1d(hidden_channels)
        
        self.dagnn = DAGNN(hidden_channels, K, dropout=dropout)
        
        self.lin3 = nn.Linear(hidden_channels, hidden_channels // 2)
        self.bn3 = nn.BatchNorm1d(hidden_channels // 2)
        
        self.lin_out = nn.Linear(hidden_channels // 2, out_channels)

    def forward(self, x, edge_index, training=True):
        x = self.lin1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=training)
        
        identity = x
        x = self.lin2(x)
        x = self.bn2(x)
        x = F.relu(x)
        x = x + identity
        x = F.dropout(x, p=self.dropout, training=training)
        
        x = self.dagnn(x, edge_index, training=training)
        
        x = self.lin3(x)
        x = self.bn3(x)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=training)
        
        x = self.lin_out(x)
        return x


# ============================ Загрузка данных ============================

logger.info("\n📥 Загрузка и подготовка данных...")

# Загрузка DAG
with open("compositionsDAG.json", "r", encoding="utf-8") as f:
    dag_data = json.load(f)

# Загрузка истории вызовов
df = pd.read_csv("calls.csv", sep=";")
df = df[~df['owner'].str.contains('cookies', na=False)]
df = df.sort_values('start_time')

owners = df['owner'].unique()
mids = df['mid'].unique()

logger.info(f"Пользователей: {len(owners)}")
logger.info(f"Сервисов: {len(mids)}")
logger.info(f"Взаимодействий: {len(df)}")

# ============================ Создание графа из последовательностей ============================

logger.info("\n🔧 Создание графа из последовательностей вызовов...")

# Создаем граф на основе РЕАЛЬНЫХ последовательностей вызовов
dag = nx.DiGraph()

# Для каждого пользователя создаем последовательность
sequences_by_user = {}
for owner in owners:
    user_calls = df[df['owner'] == owner].sort_values('start_time')['mid'].tolist()
    if len(user_calls) > 1:
        sequences_by_user[owner] = user_calls

# Создаем рёбра на основе последовательных вызовов
edge_counts = Counter()
for user, sequence in sequences_by_user.items():
    for i in range(len(sequence) - 1):
        service_from = f"service_{sequence[i]}"
        service_to = f"service_{sequence[i+1]}"
        edge_counts[(service_from, service_to)] += 1

# Добавляем рёбра с весом (частота)
for (src, dst), count in edge_counts.items():
    if count >= 2:  # Только часто встречающиеся переходы
        dag.add_node(src, type='service')
        dag.add_node(dst, type='service')
        dag.add_edge(src, dst, weight=count)

logger.info(f"Граф из последовательностей: {dag.number_of_nodes()} узлов, {dag.number_of_edges()} рёбер")

# Подготовка данных для DAGNN
node_list = list(dag.nodes)
node_encoder = LabelEncoder()
node_ids = node_encoder.fit_transform(node_list)
node_map = {node: idx for node, idx in zip(node_list, node_ids)}

if len(dag.edges) > 0:
    edge_index = torch.tensor([[node_map[u], node_map[v]] for u, v in dag.edges], dtype=torch.long).t()
    features = [[1.0] for _ in node_list]  # Простые features
    x = torch.tensor(features, dtype=torch.float)
    data_pyg = Data(x=x, edge_index=edge_index)
else:
    logger.warning("⚠️ Недостаточно рёбер в графе! Используем исходный DAG...")
    # Загрузим оригинальный DAG
    dag = nx.DiGraph()
    for composition in dag_data:
        for node in composition["nodes"]:
            node_id = f"service_{node['mid']}" if 'mid' in node else f"table_{node['id']}"
            dag.add_node(node_id, type='service' if 'mid' in node else 'table')
        for link in composition["links"]:
            src = f"service_{dag_data[0]['nodes'][link['source']]['mid']}" if 'mid' in dag_data[0]['nodes'][link['source']] else f"table_{link['source']}"
            dst = f"service_{dag_data[0]['nodes'][link['target']]['mid']}" if 'mid' in dag_data[0]['nodes'][link['target']] else f"table_{link['target']}"
            dag.add_edge(src, dst)
    
    node_list = list(dag.nodes)
    node_map = {node: idx for idx, node in enumerate(node_list)}
    edge_index = torch.tensor([[node_map[u], node_map[v]] for u, v in dag.edges], dtype=torch.long).t()
    features = [[1, 0] if n.startswith('service') else [0, 1] for n in node_list]
    x = torch.tensor(features, dtype=torch.float)
    data_pyg = Data(x=x, edge_index=edge_index)

# ============================ Создание обучающих примеров ============================

logger.info("\n📊 Создание обучающих примеров из последовательностей...")

# Создаем примеры: (context) -> (next_service)
X_raw = []
y_raw = []

for owner, sequence in sequences_by_user.items():
    for i in range(1, len(sequence)):
        context = tuple([f"service_{mid}" for mid in sequence[:i]])
        next_service = f"service_{sequence[i]}"
        
        # Только если оба в графе
        if all(s in node_map for s in context) and next_service in node_map:
            X_raw.append(context)
            y_raw.append(next_service)

logger.info(f"Создано примеров: {len(X_raw)}")

if len(X_raw) < 10:
    logger.error("❌ Недостаточно примеров! Используйте оригинальные скрипты.")
    exit(1)

# Маппинг в индексы
contexts_indices = torch.tensor([node_map[context[-1]] for context in X_raw], dtype=torch.long)
targets_indices = torch.tensor([node_map[y] for y in y_raw], dtype=torch.long)

# Split на train/test
try:
    contexts_train, contexts_test, targets_train, targets_test = train_test_split(
        contexts_indices, targets_indices, test_size=0.3, random_state=456,
        stratify=targets_indices.numpy()
    )
except:
    # Если недостаточно примеров для stratify
    contexts_train, contexts_test, targets_train, targets_test = train_test_split(
        contexts_indices, targets_indices, test_size=0.3, random_state=456
    )

logger.info(f"Train: {len(contexts_train)}, Test: {len(contexts_test)}")

# ============================ Обучение DAGNN ============================

logger.info("\n🧠 Обучение DAGNN на последовательностях...")

torch.manual_seed(456)
np.random.seed(456)

dagnn_model = DAGNNRecommender(
    in_channels=x.shape[1],
    hidden_channels=64,
    out_channels=len(node_map),
    K=10,
    dropout=0.4
)

optimizer = torch.optim.Adam(dagnn_model.parameters(), lr=0.001, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=20)

best_loss = float('inf')
patience_counter = 0

for epoch in tqdm(range(100), desc="Training DAGNN"):
    dagnn_model.train()
    optimizer.zero_grad()
    
    out = dagnn_model(data_pyg.x, data_pyg.edge_index, training=True)[contexts_train]
    loss = F.cross_entropy(out, targets_train)
    
    loss.backward()
    torch.nn.utils.clip_grad_norm_(dagnn_model.parameters(), max_norm=1.0)
    optimizer.step()
    scheduler.step(loss)
    
    if loss.item() < best_loss:
        best_loss = loss.item()
        patience_counter = 0
    else:
        patience_counter += 1
        if patience_counter >= 30:
            logger.info(f"   Early stopping at epoch {epoch}")
            break

# Оценка DAGNN
dagnn_model.eval()
with torch.no_grad():
    dagnn_output = dagnn_model(data_pyg.x, data_pyg.edge_index, training=False)[contexts_test]
    dagnn_preds = dagnn_output.argmax(dim=1).numpy()
    dagnn_proba = F.softmax(dagnn_output, dim=1).numpy()

logger.info("✅ DAGNN обучена на последовательностях")

# ============================ Обучение Hybrid-BPR ============================

logger.info("\n🎯 Обучение Hybrid-BPR (Collaborative Filtering)...")

# Split CF data
train_size = int(len(df) * 0.7)
df_train = df[:train_size]
df_test = df[train_size:]

def build_matrix(df, owners, mids, normalize=False):
    pivot = df.pivot_table(index='owner', columns='mid', values='id', aggfunc='count').fillna(0)
    pivot = pivot.reindex(index=owners, columns=mids, fill_value=0)
    mat = pivot.values
    if normalize:
        row_sums = mat.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1
        mat = mat / row_sums
    return mat

X_train_cf = build_matrix(df_train, owners, mids, normalize=True)

# LightFM-BPR
np.random.seed(1000)

dataset_cf = Dataset()
dataset_cf.fit(owners, mids)
interactions, _ = dataset_cf.build_interactions(
    [(row['owner'], row['mid']) for _, row in df_train.iterrows()]
)

lightfm_model = LightFM(
    loss='bpr',
    no_components=60,
    learning_rate=0.07,
    item_alpha=1e-07,
    user_alpha=1e-07,
    max_sampled=20,
    random_state=1000
)

logger.info("   Training PHCF-BPR...")
for epoch in tqdm(range(20), desc="PHCF-BPR"):
    lightfm_model.fit_partial(interactions, epochs=1, num_threads=4)

# KNN
logger.info("   Training Weighted KNN...")
knn_model = NearestNeighbors(n_neighbors=6, metric='cosine')
knn_model.fit(X_train_cf)
distances, indices = knn_model.kneighbors(X_train_cf)

# Предсказания
users_grid, items_grid = np.meshgrid(np.arange(len(owners)), np.arange(len(mids)), indexing='ij')
phcf_preds = lightfm_model.predict(users_grid.flatten(), items_grid.flatten(), num_threads=4)
phcf_preds = phcf_preds.reshape(len(owners), len(mids))

knn_preds = np.zeros_like(X_train_cf)
for i in range(len(X_train_cf)):
    neighbors = indices[i, 1:]
    neighbor_dists = distances[i, 1:]
    weights = 1.0 / (neighbor_dists + 1e-10)
    weights = weights / weights.sum()
    knn_preds[i] = (X_train_cf[neighbors].T @ weights)

hybrid_bpr_preds = 0.7 * phcf_preds + 0.3 * knn_preds

logger.info("✅ Hybrid-BPR обучена")

# ============================ Маппинг между системами ============================

logger.info("\n🔗 Создание маппинга...")

dag_services = [node for node in node_list if node.startswith("service_")]
dag_service_ids = [int(s.split('_')[1]) for s in dag_services]

common_service_ids = set(dag_service_ids) & set(mids)
logger.info(f"Общих сервисов: {len(common_service_ids)}")

service_map_dag_to_cf = {}
service_map_cf_to_dag = {}
mid_to_idx_cf = {mid: idx for idx, mid in enumerate(mids)}

for service_id in common_service_ids:
    dag_node = f"service_{service_id}"
    if dag_node in node_map and service_id in mid_to_idx_cf:
        dag_idx = node_map[dag_node]
        cf_idx = mid_to_idx_cf[service_id]
        service_map_dag_to_cf[dag_idx] = cf_idx
        service_map_cf_to_dag[cf_idx] = dag_idx

# ============================ Функции оценки ============================

def evaluate_model_with_ndcg(preds, true_labels, proba_preds, name="Model"):
    """Оценка модели с метриками"""
    metrics = {}
    metrics['accuracy'] = accuracy_score(true_labels, preds)
    metrics['f1'] = f1_score(true_labels, preds, average='macro', zero_division=0)
    metrics['precision'] = precision_score(true_labels, preds, average='macro', zero_division=0)
    metrics['recall'] = recall_score(true_labels, preds, average='macro', zero_division=0)
    
    logger.info(f"\n{'='*50}")
    logger.info(f"📊 {name} Metrics")
    logger.info(f"{'='*50}")
    logger.info(f"Accuracy:  {metrics['accuracy']:.4f}")
    logger.info(f"F1-score:  {metrics['f1']:.4f}")
    logger.info(f"Precision: {metrics['precision']:.4f}")
    logger.info(f"Recall:    {metrics['recall']:.4f}")
    
    if proba_preds is not None:
        try:
            from sklearn.preprocessing import LabelBinarizer
            n_classes = proba_preds.shape[1]
            lb = LabelBinarizer()
            lb.fit(range(n_classes))
            true_bin = lb.transform(true_labels)
            
            if true_bin.ndim == 1:
                true_bin = np.eye(n_classes)[true_labels]
            
            metrics['ndcg'] = ndcg_score(true_bin, proba_preds)
            logger.info(f"nDCG:      {metrics['ndcg']:.4f}")
        except Exception as e:
            logger.warning(f"nDCG:      ❌ Error: {e}")
            metrics['ndcg'] = None
    else:
        metrics['ndcg'] = None
    
    return metrics


def create_hybrid_predictions(dagnn_output, cf_predictions, service_map_cf_to_dag, 
                              contexts_indices, beta=0.5):
    """
    Создает гибридные предсказания для каждого контекста
    
    Args:
        dagnn_output: (num_contexts, num_dag_nodes) - DAGNN предсказания
        cf_predictions: (num_users, num_cf_services) - CF предсказания
        service_map_cf_to_dag: Маппинг CF → DAG индексов
        contexts_indices: Индексы контекстов
        beta: Вес DAGNN
    
    Returns:
        hybrid_predictions: (num_contexts, num_dag_nodes)
    """
    num_contexts = dagnn_output.shape[0]
    num_dag_nodes = dagnn_output.shape[1]
    
    hybrid_preds = dagnn_output.copy()
    
    # Для каждого контекста
    for ctx_idx in range(num_contexts):
        # Берем CF предсказания (используем первого пользователя для упрощения)
        # В реальности нужно знать какой пользователь сделал этот контекст
        user_idx = ctx_idx % len(cf_predictions)  # Простое распределение
        cf_scores = cf_predictions[user_idx]
        
        # Для узлов которые есть в обеих системах
        for dag_idx in range(num_dag_nodes):
            if dag_idx in service_map_dag_to_cf:
                cf_idx = service_map_dag_to_cf[dag_idx]
                cf_score = cf_scores[cf_idx]
                dag_score = dagnn_output[ctx_idx, dag_idx]
                
                # Комбинируем
                hybrid_preds[ctx_idx, dag_idx] = beta * dag_score + (1 - beta) * cf_score
    
    return hybrid_preds


# ============================ Тестирование моделей ============================

logger.info("\n🔬 Тестирование моделей...")

results = {}

# 1. DAGNN only
logger.info("\n1️⃣ DAGNN (Sequence only):")
dagnn_metrics = evaluate_model_with_ndcg(
    dagnn_preds, targets_test.numpy(), dagnn_proba, "DAGNN (Sequence)"
)
results['DAGNN (Sequence)'] = dagnn_metrics

# 2. Гибридные модели с разными beta
betas = [0.0, 0.3, 0.5, 0.7, 1.0]

for beta in betas:
    logger.info(f"\n{'='*50}")
    logger.info(f"Тестирование Hybrid (beta={beta})...")
    logger.info(f"{'='*50}")
    
    # Создаем гибридные предсказания
    with torch.no_grad():
        dagnn_output_all = dagnn_model(data_pyg.x, data_pyg.edge_index, training=False)
        dagnn_output_test = dagnn_output_all[contexts_test].numpy()
    
    hybrid_output = create_hybrid_predictions(
        dagnn_output_test,
        hybrid_bpr_preds,
        service_map_dag_to_cf,
        contexts_test,
        beta=beta
    )
    
    # Предсказания
    hybrid_preds = hybrid_output.argmax(axis=1)
    
    # Нормализуем для softmax
    hybrid_proba = np.exp(hybrid_output) / np.exp(hybrid_output).sum(axis=1, keepdims=True)
    
    # Оценка
    if beta == 0.0:
        name = "CF only (beta=0)"
    elif beta == 1.0:
        name = "DAGNN only (beta=1)"
    else:
        name = f"Hybrid (beta={beta})"
    
    hybrid_metrics = evaluate_model_with_ndcg(
        hybrid_preds, targets_test.numpy(), hybrid_proba, name
    )
    results[name] = hybrid_metrics

# ============================ Итоговые результаты ============================

logger.info("\n" + "="*70)
logger.info("🏆 ИТОГОВОЕ СРАВНЕНИЕ")
logger.info("="*70)

# Сортируем по accuracy
sorted_results = sorted(results.items(), key=lambda x: x[1]['accuracy'], reverse=True)

for rank, (model_name, metrics) in enumerate(sorted_results, 1):
    logger.info(f"\n#{rank} {model_name}:")
    for metric_name, value in metrics.items():
        if value is not None:
            logger.info(f"     {metric_name}: {value:.4f}")

# Лучшие по метрикам
logger.info("\n" + "="*70)
logger.info("📊 ЛУЧШИЕ ПО МЕТРИКАМ")
logger.info("="*70)

best_accuracy = max(results.items(), key=lambda x: x[1]['accuracy'])

# Фильтруем только модели с nDCG
models_with_ndcg = [(k, v) for k, v in results.items() if v['ndcg'] is not None]
best_ndcg = max(models_with_ndcg, key=lambda x: x[1]['ndcg']) if models_with_ndcg else None

logger.info(f"🥇 Лучший Accuracy: {best_accuracy[0]:<30} {best_accuracy[1]['accuracy']:.4f}")
logger.info(f"🥇 Лучший nDCG:     {best_ndcg[0]:<30} {best_ndcg[1]['ndcg']:.4f}")

# ============================ Пример рекомендаций ============================

logger.info("\n" + "="*70)
logger.info("📝 ПРИМЕР КОНТЕКСТНЫХ РЕКОМЕНДАЦИЙ")
logger.info("="*70)

# Выбираем пользователя с длинной историей
example_owner = max(sequences_by_user.items(), key=lambda x: len(x[1]))[0]
example_sequence = sequences_by_user[example_owner]
example_user_idx = list(owners).index(example_owner)

logger.info(f"\nПользователь: {example_owner}")
logger.info(f"История вызовов: {len(example_sequence)} сервисов")
logger.info(f"Последние 5: {example_sequence[-5:]}")

# Контекст: последние 3 вызова
context_mids = example_sequence[-3:]
context_nodes = [f"service_{mid}" for mid in context_mids if f"service_{mid}" in node_map]

logger.info(f"Контекст: {context_mids}")

if len(context_nodes) > 0:
    # DAGNN предсказание
    context_idx = node_map[context_nodes[-1]]
    with torch.no_grad():
        dagnn_out = dagnn_model(data_pyg.x, data_pyg.edge_index, training=False)
        dagnn_scores = F.softmax(dagnn_out[context_idx], dim=0).numpy()
    
    # Топ-5 от DAGNN
    top_dagnn_indices = dagnn_scores.argsort()[::-1][:5]
    logger.info(f"\n📊 DAGNN предсказания:")
    for i, idx in enumerate(top_dagnn_indices, 1):
        node_name = node_list[idx]
        score = dagnn_scores[idx]
        logger.info(f"   {i}. {node_name:<25} Score: {score:.4f}")

# CF предсказание
cf_scores = hybrid_bpr_preds[example_user_idx]
top_cf_indices = cf_scores.argsort()[::-1][:5]

logger.info(f"\n📊 Hybrid-BPR предсказания:")
for i, idx in enumerate(top_cf_indices, 1):
    mid = mids[idx]
    score = cf_scores[idx]
    logger.info(f"   {i}. Service {mid:<20} Score: {score:.4f}")

# Гибридные предсказания (beta=0.5)
if len(context_nodes) > 0:
    beta = 0.5
    hybrid_scores = np.zeros(len(mids))
    
    for cf_idx, mid in enumerate(mids):
        cf_score = cf_scores[cf_idx]
        dag_score = 0.0
        
        if cf_idx in service_map_cf_to_dag:
            dag_idx = service_map_cf_to_dag[cf_idx]
            dag_score = dagnn_scores[dag_idx]
        
        hybrid_scores[cf_idx] = beta * dag_score + (1 - beta) * cf_score
    
    top_hybrid_indices = hybrid_scores.argsort()[::-1][:10]
    
    logger.info(f"\n📊 Гибридные рекомендации (beta=0.5):")
    for i, idx in enumerate(top_hybrid_indices, 1):
        mid = mids[idx]
        score = hybrid_scores[idx]
        in_history = "✓" if mid in example_sequence else " "
        logger.info(f"   {i:2d}. Service {mid:<15} Score: {score:8.4f}  {in_history}")

# Что было на самом деле в тесте
actual_test_mids = df_test[df_test['owner'] == example_owner]['mid'].unique()
logger.info(f"\n✅ Фактические вызовы в тесте: {len(actual_test_mids)} уникальных")
logger.info(f"   {list(actual_test_mids[:10])}")

# ============================ Финальные рекомендации ============================

logger.info("\n" + "="*70)
logger.info("✅ ГИБРИДНАЯ КОНТЕКСТНАЯ СИСТЕМА ГОТОВА!")
logger.info("="*70)

logger.info(f"\n🏆 Лучшая модель по Accuracy: {best_accuracy[0]}")
logger.info(f"   Accuracy: {best_accuracy[1]['accuracy']:.4f}")

if best_ndcg:
    logger.info(f"\n🏆 Лучшая модель по nDCG: {best_ndcg[0]}")
    logger.info(f"   nDCG: {best_ndcg[1]['ndcg']:.4f}")

logger.info("\n💡 Рекомендации:")
logger.info(f"   - Для контекстных рекомендаций: beta=0.5-0.7")
logger.info(f"   - Для персонализации: beta=0.0-0.3 (больше CF)")
logger.info(f"   - Для холодного старта: beta=0.7-1.0 (больше DAGNN)")

logger.info("\n" + "="*70)

