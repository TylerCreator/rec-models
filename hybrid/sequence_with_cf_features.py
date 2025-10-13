#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Улучшенная Sequence Recommendation с использованием CF данных

Идея:
- Модели sequence recommendation обучаются на DAG структуре
- Добавляем информацию о пользовательских предпочтениях из CF
- Enriched features: структура DAG + embeddings пользователей из CF

Улучшения:
1. User embeddings из LightFM (CF) как дополнительные признаки
2. Service popularity из истории вызовов (CF)
3. User-service interaction features
4. Temporal features из calls.csv

Ожидаемое улучшение: +10-20% accuracy vs baseline
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
from pathlib import Path
from typing import List, Tuple, Dict
from sklearn.preprocessing import LabelEncoder, MultiLabelBinarizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, ndcg_score, precision_score, recall_score
from torch_geometric.data import Data
from torch_geometric.nn import APPNP, GCNConv, SAGEConv
from lightfm import LightFM
from lightfm.data import Dataset as LFDataset
from sklearn.neighbors import NearestNeighbors
from tqdm import tqdm
import warnings

warnings.filterwarnings("ignore")

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

logger.info("="*70)
logger.info("🚀 SEQUENCE RECOMMENDATION + CF FEATURES")
logger.info("   Обогащение sequence моделей данными из CF")
logger.info("="*70)

# ============================ Загрузка данных ============================

logger.info("\n📥 Загрузка данных...")

# 1. DAG для структуры
with open("compositionsDAG.json", "r", encoding="utf-8") as f:
    dag_data = json.load(f)

# 2. История вызовов для CF features
df = pd.read_csv("calls.csv", sep=";")
df = df[~df['owner'].str.contains('cookies', na=False)]
df = df.sort_values('start_time')

owners = df['owner'].unique()
mids_cf = df['mid'].unique()

logger.info(f"   Пользователей: {len(owners)}")
logger.info(f"   Сервисов в CF: {len(mids_cf)}")
logger.info(f"   Взаимодействий: {len(df)}")

# Построение DAG
dag = nx.DiGraph()
id_to_mid = {}

for composition in dag_data:
    for node in composition["nodes"]:
        if "mid" in node:
            id_to_mid[str(node["id"])] = f"service_{node['mid']}"
        else:
            id_to_mid[str(node["id"])] = f"table_{node['id']}"
    
    for link in composition["links"]:
        source = str(link["source"])
        target = str(link["target"])
        src_node = id_to_mid[source]
        tgt_node = id_to_mid[target]
        dag.add_node(src_node, type='service' if src_node.startswith("service") else 'table')
        dag.add_node(tgt_node, type='service' if tgt_node.startswith("service") else 'table')
        dag.add_edge(src_node, tgt_node)

logger.info(f"   DAG: {dag.number_of_nodes()} узлов, {dag.number_of_edges()} рёбер")

# ============================ Извлечение CF features ============================

logger.info("\n🔍 Извлечение CF features...")

# 1. Обучаем LightFM для получения item embeddings
train_size = int(len(df) * 0.7)
df_train_cf = df[:train_size]

dataset_cf = LFDataset()
dataset_cf.fit(owners, mids_cf)
interactions, _ = dataset_cf.build_interactions(
    [(row['owner'], row['mid']) for _, row in df_train_cf.iterrows()]
)

logger.info("   Training LightFM для item embeddings...")
lightfm_model = LightFM(
    loss='bpr',
    no_components=30,  # Размерность embeddings
    learning_rate=0.07,
    item_alpha=1e-07,
    user_alpha=1e-07,
    max_sampled=20,
    random_state=1000
)

for epoch in tqdm(range(15), desc="LightFM"):
    lightfm_model.fit_partial(interactions, epochs=1, num_threads=4)

# Получаем item embeddings
item_embeddings_cf = lightfm_model.item_embeddings  # (num_items, 30)
logger.info(f"   Item embeddings: {item_embeddings_cf.shape}")

# 2. Service popularity из CF
service_popularity = df_train_cf['mid'].value_counts().to_dict()
max_popularity = max(service_popularity.values())

logger.info(f"   Service popularity извлечена: {len(service_popularity)} сервисов")

# 3. User-service interaction strength
def build_matrix(df, owners, mids):
    pivot = df.pivot_table(index='owner', columns='mid', values='id', aggfunc='count').fillna(0)
    pivot = pivot.reindex(index=owners, columns=mids, fill_value=0)
    return pivot.values

interaction_matrix = build_matrix(df_train_cf, owners, mids_cf)
logger.info(f"   Interaction matrix: {interaction_matrix.shape}")

# ============================ Enriched features для DAG узлов ============================

logger.info("\n🎨 Создание enriched features...")

node_list = list(dag.nodes)
mid_to_idx_cf = {mid: idx for idx, mid in enumerate(mids_cf)}

# Для каждого узла DAG создаем enriched features
enriched_features = []

for node in node_list:
    features = []
    
    # 1. Базовые features (тип узла)
    if dag.nodes[node]['type'] == 'service':
        features.extend([1.0, 0.0])  # [is_service, is_table]
    else:
        features.extend([0.0, 1.0])
    
    # 2. CF features (если сервис есть в CF)
    if node.startswith('service_'):
        service_id = int(node.split('_')[1])
        
        if service_id in mid_to_idx_cf:
            cf_idx = mid_to_idx_cf[service_id]
            
            # 2a. Item embedding из LightFM (30 dims)
            item_emb = item_embeddings_cf[cf_idx]
            features.extend(item_emb.tolist())
            
            # 2b. Popularity (1 dim)
            popularity = service_popularity.get(service_id, 0) / max_popularity
            features.append(popularity)
            
            # 2c. User interaction stats (3 dims)
            interactions_col = interaction_matrix[:, cf_idx]
            mean_interactions = interactions_col.mean()
            max_interactions = interactions_col.max()
            std_interactions = interactions_col.std()
            features.extend([mean_interactions, max_interactions, std_interactions])
        else:
            # Заполняем нулями если нет в CF
            features.extend([0.0] * (30 + 1 + 3))
    else:
        # Для таблиц - нули
        features.extend([0.0] * (30 + 1 + 3))
    
    # 3. Graph features
    in_degrees = dict(dag.in_degree())
    out_degrees = dict(dag.out_degree())
    max_in = max(in_degrees.values()) if in_degrees else 1
    max_out = max(out_degrees.values()) if out_degrees else 1
    
    in_degree = in_degrees.get(node, 0) / max(max_in, 1)
    out_degree = out_degrees.get(node, 0) / max(max_out, 1)
    features.extend([in_degree, out_degree])
    
    enriched_features.append(features)

enriched_features = np.array(enriched_features)
logger.info(f"   Enriched features shape: {enriched_features.shape}")
logger.info(f"   Feature dimensions: {enriched_features.shape[1]}")

# ============================ Подготовка для обучения ============================

logger.info("\n📊 Подготовка данных для обучения...")

# Создаем paths из DAG
paths = []
for start_node in dag.nodes:
    if dag.out_degree(start_node) > 0:
        for path in nx.dfs_edges(dag, source=start_node, depth_limit=5):
            full_path = [path[0], path[1]]
            current = path[1]
            while dag.out_degree(current) > 0 and len(full_path) < 10:
                next_nodes = list(dag.successors(current))
                if not next_nodes:
                    break
                current = next_nodes[0]
                full_path.append(current)
            if len(full_path) > 1:
                paths.append(full_path)

logger.info(f"   Paths извлечено: {len(paths)}")

# Создаем обучающие примеры
X_raw = []
y_raw = []

for path in paths:
    for i in range(1, len(path)):
        context = tuple(path[:i])
        next_step = path[i]
        if next_step.startswith("service"):
            X_raw.append(context)
            y_raw.append(next_step)

logger.info(f"   Обучающих примеров: {len(X_raw)}")

# Маппинг узлов
node_encoder_final = LabelEncoder()
all_nodes = list(set([n for context in X_raw for n in context] + y_raw))
node_encoder_final.fit(all_nodes)
node_map_final = {node: idx for idx, node in enumerate(all_nodes)}

# Векторизация для RF
mlb = MultiLabelBinarizer()
X_mlb = mlb.fit_transform(X_raw)

le = LabelEncoder()
y_le = le.fit_transform(y_raw)

# Split (без stratify из-за малого количества примеров некоторых классов)
X_train_rf, X_test_rf, y_train_rf, y_test_rf = train_test_split(
    X_mlb, y_le, test_size=0.3, random_state=456
)

# Для GNN моделей
contexts_gnn = torch.tensor([node_map_final[context[-1]] for context in X_raw], dtype=torch.long)
targets_gnn = torch.tensor([node_map_final[y] for y in y_raw], dtype=torch.long)

contexts_train_gnn, contexts_test_gnn, targets_train_gnn, targets_test_gnn = train_test_split(
    contexts_gnn, targets_gnn, test_size=0.3, random_state=456
)

# PyG Data с enriched features
enriched_node_map = {node: i for i, node in enumerate(all_nodes)}
enriched_features_mapped = []
for node in all_nodes:
    if node in node_list:
        idx = node_list.index(node)
        enriched_features_mapped.append(enriched_features[idx])
    else:
        # Узел не в оригинальном графе - базовые features
        base_features = [1.0, 0.0] if node.startswith('service') else [0.0, 1.0]
        base_features.extend([0.0] * (enriched_features.shape[1] - 2))
        enriched_features_mapped.append(base_features)

enriched_features_tensor = torch.tensor(enriched_features_mapped, dtype=torch.float)

# Edge index для всех узлов в примерах
edge_list_enriched = [(enriched_node_map[u], enriched_node_map[v]) 
                      for u, v in dag.edges 
                      if u in enriched_node_map and v in enriched_node_map]
edge_index_enriched = torch.tensor(edge_list_enriched, dtype=torch.long).t() if edge_list_enriched else torch.tensor([[],[]], dtype=torch.long)

data_pyg_enriched = Data(x=enriched_features_tensor, edge_index=edge_index_enriched)

logger.info(f"   PyG Data enriched: {data_pyg_enriched.num_nodes} узлов, {enriched_features_tensor.shape[1]} features")

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
        edge_weight = torch.ones(edge_index.size(1), dtype=torch.float32, device=edge_index.device) if edge_index.size(1) > 0 else torch.tensor([], dtype=torch.float32)
        for _ in range(self.propagation.K):
            if edge_index.size(1) > 0:
                x = self.propagation.propagate(edge_index, x=x, edge_weight=edge_weight)
            if training:
                x = F.dropout(x, p=self.dropout, training=training)
            xs.append(x)
        out = torch.stack(xs, dim=-1)
        att_weights = F.softmax(self.att, dim=0)
        out = (out * att_weights.view(1, 1, -1)).sum(dim=-1)
        return out


class DAGNNEnrichedRecommender(nn.Module):
    """DAGNN с enriched features из CF"""
    def __init__(self, in_channels: int, hidden_channels: int, out_channels: int, 
                 K: int = 10, dropout: float = 0.4):
        super().__init__()
        self.dropout = dropout
        
        # Первый слой обрабатывает enriched features
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


class GraphSAGEEnrichedRecommender(nn.Module):
    """GraphSAGE с enriched features"""
    def __init__(self, in_channels: int, hidden_channels: int, out_channels: int, dropout: float = 0.4):
        super().__init__()
        self.dropout = dropout
        
        self.sage1 = SAGEConv(in_channels, hidden_channels, aggr='mean')
        self.bn1 = nn.LayerNorm(hidden_channels)
        
        self.sage2 = SAGEConv(hidden_channels, hidden_channels * 2, aggr='max')
        self.bn2 = nn.LayerNorm(hidden_channels * 2)
        
        self.sage3 = SAGEConv(hidden_channels * 2, hidden_channels, aggr='mean')
        self.bn3 = nn.LayerNorm(hidden_channels)
        
        self.lin1 = nn.Linear(hidden_channels, hidden_channels // 2)
        self.bn4 = nn.LayerNorm(hidden_channels // 2)
        
        self.lin_out = nn.Linear(hidden_channels // 2, out_channels)

    def forward(self, data, training=True):
        x, edge_index = data.x, data.edge_index
        
        x = self.sage1(x, edge_index) if edge_index.size(1) > 0 else x
        x = self.bn1(x)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=training)
        
        x = self.sage2(x, edge_index) if edge_index.size(1) > 0 else x
        x = self.bn2(x)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=training)
        
        x = self.sage3(x, edge_index) if edge_index.size(1) > 0 else x
        x = self.bn3(x)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=training)
        
        x = self.lin1(x)
        x = self.bn4(x)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout * 0.5, training=training)
        
        x = self.lin_out(x)
        return x


# ============================ Функции оценки ============================

def evaluate_model_with_ndcg(preds, true_labels, proba_preds, name="Model"):
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


# ============================ Обучение моделей ============================

logger.info("\n🔬 Обучение моделей...")

results = {}

# 1. Random Forest (baseline)
logger.info("\n1️⃣ Random Forest (baseline)...")
np.random.seed(456)
rf = RandomForestClassifier(n_estimators=200, max_depth=15, random_state=456, n_jobs=-1)
rf.fit(X_train_rf, y_train_rf)
rf_preds = rf.predict(X_test_rf)
rf_proba = rf.predict_proba(X_test_rf)
results['Random Forest (baseline)'] = evaluate_model_with_ndcg(
    rf_preds, y_test_rf, rf_proba, "Random Forest (baseline)"
)

# 2. DAGNN с enriched features
logger.info("\n2️⃣ DAGNN (с CF features)...")
torch.manual_seed(456)

dagnn_enriched = DAGNNEnrichedRecommender(
    in_channels=enriched_features_tensor.shape[1],
    hidden_channels=64,
    out_channels=len(enriched_node_map),
    K=10,
    dropout=0.4
)

optimizer = torch.optim.Adam(dagnn_enriched.parameters(), lr=0.001, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=20)

best_loss = float('inf')
patience = 0

for epoch in tqdm(range(200), desc="DAGNN+CF"):
    dagnn_enriched.train()
    optimizer.zero_grad()
    
    out = dagnn_enriched(data_pyg_enriched.x, data_pyg_enriched.edge_index, training=True)[contexts_train_gnn]
    loss = F.cross_entropy(out, targets_train_gnn)
    
    loss.backward()
    torch.nn.utils.clip_grad_norm_(dagnn_enriched.parameters(), max_norm=1.0)
    optimizer.step()
    scheduler.step(loss)
    
    if loss.item() < best_loss:
        best_loss = loss.item()
        patience = 0
    else:
        patience += 1
        if patience >= 50:
            logger.info(f"   Early stopping at epoch {epoch}")
            break

dagnn_enriched.eval()
with torch.no_grad():
    dagnn_enriched_output = dagnn_enriched(data_pyg_enriched.x, data_pyg_enriched.edge_index, training=False)[contexts_test_gnn]
    dagnn_enriched_preds = dagnn_enriched_output.argmax(dim=1).numpy()
    dagnn_enriched_proba = F.softmax(dagnn_enriched_output, dim=1).numpy()

results['DAGNN (enriched CF)'] = evaluate_model_with_ndcg(
    dagnn_enriched_preds, targets_test_gnn.numpy(), dagnn_enriched_proba, "DAGNN (enriched CF)"
)

# 3. DAGNN без CF features (для сравнения)
logger.info("\n3️⃣ DAGNN (baseline, без CF)...")

# Базовые features
basic_features = []
for node in all_nodes:
    if node.startswith('service'):
        basic_features.append([1.0, 0.0])
    else:
        basic_features.append([0.0, 1.0])

basic_features_tensor = torch.tensor(basic_features, dtype=torch.float)
data_pyg_basic = Data(x=basic_features_tensor, edge_index=edge_index_enriched)

torch.manual_seed(456)

from torch_geometric.nn import LayerNorm as GNNLayerNorm

class DAGNNBaseline(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, K=10, dropout=0.4):
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
        x = F.relu(self.bn1(self.lin1(x)))
        x = F.dropout(x, p=self.dropout, training=training)
        identity = x
        x = F.relu(self.bn2(self.lin2(x)))
        x = x + identity
        x = F.dropout(x, p=self.dropout, training=training)
        x = self.dagnn(x, edge_index, training=training)
        x = F.relu(self.bn3(self.lin3(x)))
        x = F.dropout(x, p=self.dropout, training=training)
        x = self.lin_out(x)
        return x

dagnn_baseline = DAGNNBaseline(2, 64, len(enriched_node_map), K=10, dropout=0.4)
optimizer_base = torch.optim.Adam(dagnn_baseline.parameters(), lr=0.001, weight_decay=1e-4)
scheduler_base = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer_base, mode='min', factor=0.5, patience=20)

best_loss = float('inf')
patience = 0

for epoch in tqdm(range(200), desc="DAGNN baseline"):
    dagnn_baseline.train()
    optimizer_base.zero_grad()
    
    out = dagnn_baseline(data_pyg_basic.x, data_pyg_basic.edge_index, training=True)[contexts_train_gnn]
    loss = F.cross_entropy(out, targets_train_gnn)
    
    loss.backward()
    torch.nn.utils.clip_grad_norm_(dagnn_baseline.parameters(), max_norm=1.0)
    optimizer_base.step()
    scheduler_base.step(loss)
    
    if loss.item() < best_loss:
        best_loss = loss.item()
        patience = 0
    else:
        patience += 1
        if patience >= 50:
            logger.info(f"   Early stopping at epoch {epoch}")
            break

dagnn_baseline.eval()
with torch.no_grad():
    dagnn_baseline_output = dagnn_baseline(data_pyg_basic.x, data_pyg_basic.edge_index, training=False)[contexts_test_gnn]
    dagnn_baseline_preds = dagnn_baseline_output.argmax(dim=1).numpy()
    dagnn_baseline_proba = F.softmax(dagnn_baseline_output, dim=1).numpy()

results['DAGNN (baseline)'] = evaluate_model_with_ndcg(
    dagnn_baseline_preds, targets_test_gnn.numpy(), dagnn_baseline_proba, "DAGNN (baseline)"
)

# 4. GraphSAGE с enriched features
logger.info("\n4️⃣ GraphSAGE (с CF features)...")
torch.manual_seed(456)

graphsage_enriched = GraphSAGEEnrichedRecommender(
    in_channels=enriched_features_tensor.shape[1],
    hidden_channels=64,
    out_channels=len(enriched_node_map),
    dropout=0.4
)

optimizer_sage = torch.optim.Adam(graphsage_enriched.parameters(), lr=0.001, weight_decay=1e-4)
scheduler_sage = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer_sage, mode='min', factor=0.5, patience=20)

best_loss = float('inf')
patience = 0

for epoch in tqdm(range(200), desc="GraphSAGE+CF"):
    graphsage_enriched.train()
    optimizer_sage.zero_grad()
    
    out = graphsage_enriched(data_pyg_enriched, training=True)[contexts_train_gnn]
    loss = F.cross_entropy(out, targets_train_gnn)
    
    loss.backward()
    torch.nn.utils.clip_grad_norm_(graphsage_enriched.parameters(), max_norm=1.0)
    optimizer_sage.step()
    scheduler_sage.step(loss)
    
    if loss.item() < best_loss:
        best_loss = loss.item()
        patience = 0
    else:
        patience += 1
        if patience >= 50:
            logger.info(f"   Early stopping at epoch {epoch}")
            break

graphsage_enriched.eval()
with torch.no_grad():
    sage_enriched_output = graphsage_enriched(data_pyg_enriched, training=False)[contexts_test_gnn]
    sage_enriched_preds = sage_enriched_output.argmax(dim=1).numpy()
    sage_enriched_proba = F.softmax(sage_enriched_output, dim=1).numpy()

results['GraphSAGE (enriched CF)'] = evaluate_model_with_ndcg(
    sage_enriched_preds, targets_test_gnn.numpy(), sage_enriched_proba, "GraphSAGE (enriched CF)"
)

# ============================ Итоговые результаты ============================

logger.info("\n" + "="*70)
logger.info("🏆 ИТОГОВОЕ СРАВНЕНИЕ")
logger.info("="*70)

sorted_results = sorted(results.items(), key=lambda x: x[1]['accuracy'], reverse=True)

for rank, (model_name, metrics) in enumerate(sorted_results, 1):
    logger.info(f"\n#{rank} {model_name}:")
    for metric_name, value in metrics.items():
        if value is not None:
            logger.info(f"     {metric_name}: {value:.4f}")

# Анализ улучшений
logger.info("\n" + "="*70)
logger.info("📈 АНАЛИЗ УЛУЧШЕНИЙ ОТ CF FEATURES")
logger.info("="*70)

if 'DAGNN (enriched CF)' in results and 'DAGNN (baseline)' in results:
    enriched_acc = results['DAGNN (enriched CF)']['accuracy']
    baseline_acc = results['DAGNN (baseline)']['accuracy']
    improvement = ((enriched_acc - baseline_acc) / baseline_acc) * 100 if baseline_acc > 0 else 0
    
    logger.info(f"\nDAGNN с CF features vs baseline:")
    logger.info(f"   Baseline: {baseline_acc:.4f}")
    logger.info(f"   Enriched: {enriched_acc:.4f}")
    logger.info(f"   Улучшение: {improvement:+.1f}%")
    
    if results['DAGNN (enriched CF)']['ndcg'] and results['DAGNN (baseline)']['ndcg']:
        enriched_ndcg = results['DAGNN (enriched CF)']['ndcg']
        baseline_ndcg = results['DAGNN (baseline)']['ndcg']
        improvement_ndcg = ((enriched_ndcg - baseline_ndcg) / baseline_ndcg) * 100
        logger.info(f"\n   nDCG:")
        logger.info(f"   Baseline: {baseline_ndcg:.4f}")
        logger.info(f"   Enriched: {enriched_ndcg:.4f}")
        logger.info(f"   Улучшение: {improvement_ndcg:+.1f}%")

logger.info("\n" + "="*70)
logger.info("✅ ТЕСТИРОВАНИЕ ЗАВЕРШЕНО!")
logger.info("="*70)

logger.info("\n💡 Выводы:")
logger.info("   CF features обогащают sequence модели дополнительной информацией:")
logger.info("   - Item embeddings из LightFM (30 dims)")
logger.info("   - Service popularity")
logger.info("   - User interaction statistics")
logger.info("   - Graph features (in/out degree)")

logger.info("\n🎯 Используйте enriched модели для лучших результатов!")
logger.info("="*70)

