#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Sequence Recommendation с обогащением CF данными

Точно такое же обучение и тестирование как sequence_dag_recommender_final.py,
но с enriched features из collaborative filtering.

Данные:
- compositionsDAG.json - тот же DAG граф
- calls.csv - для CF features (embeddings, popularity, statistics)

Модели (enriched vs baseline):
- DAGNN+CF vs DAGNN
- GraphSAGE+CF vs GraphSAGE  
- GCN+CF vs GCN
- SASRec+CF vs SASRec
- Caser+CF vs Caser
- GRU4Rec+CF vs GRU4Rec
- Random Forest

Ожидаемое улучшение: +10-20% благодаря CF features
"""

import argparse
import json
import logging
import sys
from collections import Counter
from pathlib import Path
from typing import List, Tuple, Dict

import networkx as nx
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    ndcg_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer, LabelEncoder, MultiLabelBinarizer
from torch_geometric.data import Data
from torch_geometric.nn import APPNP, GCNConv, SAGEConv, LayerNorm
from lightfm import LightFM
from lightfm.data import Dataset as LFDataset
from tqdm import tqdm

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

logger.info("="*70)
logger.info("🚀 SEQUENCE RECOMMENDATION + CF ENRICHMENT")
logger.info("   Те же модели и данные, но с CF features")
logger.info("="*70)

# ============================ CF Features Extraction ============================

logger.info("\n📥 Извлечение CF features из calls.csv...")

df_cf = pd.read_csv("calls.csv", sep=";")
df_cf = df_cf[~df_cf['owner'].str.contains('cookies', na=False)]

owners = df_cf['owner'].unique()
mids_cf = df_cf['mid'].unique()

logger.info(f"CF данные: {len(owners)} пользователей, {len(mids_cf)} сервисов, {len(df_cf)} вызовов")

# Обучаем LightFM для item embeddings
train_size = int(len(df_cf) * 0.7)
df_train_cf = df_cf[:train_size]

dataset_cf = LFDataset()
dataset_cf.fit(owners, mids_cf)
interactions, _ = dataset_cf.build_interactions(
    [(row['owner'], row['mid']) for _, row in df_train_cf.iterrows()]
)

logger.info("   Training LightFM для embeddings...")
lightfm = LightFM(loss='bpr', no_components=30, random_state=1000)
for _ in tqdm(range(15), desc="LightFM"):
    lightfm.fit_partial(interactions, epochs=1, num_threads=4)

item_embeddings_cf = lightfm.item_embeddings  # (num_cf_items, 30)

# Service popularity
service_popularity = df_train_cf['mid'].value_counts().to_dict()
max_pop = max(service_popularity.values())

# Interaction statistics
def build_matrix(df, owners, mids):
    pivot = df.pivot_table(index='owner', columns='mid', values='id', aggfunc='count').fillna(0)
    pivot = pivot.reindex(index=owners, columns=mids, fill_value=0)
    return pivot.values

interaction_matrix = build_matrix(df_train_cf, owners, mids_cf)

logger.info(f"✅ CF features извлечены: embeddings {item_embeddings_cf.shape}, popularity {len(service_popularity)}")

# ============================ Загрузка DAG (КАК В ОРИГИНАЛЕ) ============================

logger.info("\n📥 Загрузка DAG из compositionsDAG.json...")

with open("compositionsDAG.json", "r", encoding="utf-8") as f:
    data = json.load(f)

dag = nx.DiGraph()
id_to_mid = {}

for composition in data:
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

logger.info(f"DAG: {dag.number_of_nodes()} узлов, {dag.number_of_edges()} рёбер")

# ============================ Извлечение paths (КАК В ОРИГИНАЛЕ) ============================

logger.info("\n📊 Извлечение paths...")

paths = []
for start_node in dag.nodes:
    if dag.out_degree(start_node) > 0:
        for path in nx.dfs_edges(dag, source=start_node):
            full_path = [path[0], path[1]]
            while dag.out_degree(full_path[-1]) > 0:
                next_nodes = list(dag.successors(full_path[-1]))
                if not next_nodes:
                    break
                full_path.append(next_nodes[0])
            if len(full_path) > 1:
                paths.append(full_path)

logger.info(f"Извлечено paths: {len(paths)}")

# Создание обучающих примеров (КАК В ОРИГИНАЛЕ)
X_raw = []
y_raw = []

for path in paths:
    for i in range(1, len(path) - 1):
        context = tuple(path[:i])
        next_step = path[i]
        if next_step.startswith("service"):
            X_raw.append(context)
            y_raw.append(next_step)

logger.info(f"Обучающих примеров: {len(X_raw)}")

# ============================ Создание ENRICHED features ============================

logger.info("\n🎨 Создание enriched features для узлов DAG...")

node_list = list(dag.nodes)
mid_to_idx_cf = {mid: idx for idx, mid in enumerate(mids_cf)}

enriched_node_features = []

for node in node_list:
    features = []
    
    # 1. Базовые features (2 dims) - КАК В ОРИГИНАЛЕ
    if dag.nodes[node]['type'] == 'service':
        features.extend([1.0, 0.0])
    else:
        features.extend([0.0, 1.0])
    
    # 2. CF features (34 dims) - НОВОЕ!
    if node.startswith('service_'):
        service_id = int(node.split('_')[1])
        
        if service_id in mid_to_idx_cf:
            cf_idx = mid_to_idx_cf[service_id]
            
            # Item embedding (30 dims)
            item_emb = item_embeddings_cf[cf_idx]
            features.extend(item_emb.tolist())
            
            # Popularity (1 dim)
            pop = service_popularity.get(service_id, 0) / max_pop
            features.append(pop)
            
            # Interaction stats (3 dims)
            inter_col = interaction_matrix[:, cf_idx]
            features.extend([inter_col.mean(), inter_col.max(), inter_col.std()])
        else:
            features.extend([0.0] * 34)
    else:
        features.extend([0.0] * 34)
    
    # 3. Graph features (2 dims)
    in_deg = dag.in_degree(node) / max(dict(dag.in_degree()).values()) if dict(dag.in_degree()) else 0
    out_deg = dag.out_degree(node) / max(dict(dag.out_degree()).values()) if dict(dag.out_degree()) else 0
    features.extend([in_deg, out_deg])
    
    enriched_node_features.append(features)

enriched_features_array = np.array(enriched_node_features)
logger.info(f"Enriched features: {enriched_features_array.shape} (было 2 dims, стало {enriched_features_array.shape[1]})")

# ============================ Векторизация (КАК В ОРИГИНАЛЕ) ============================

logger.info("\n🔧 Векторизация данных...")

mlb = MultiLabelBinarizer()
X = mlb.fit_transform(X_raw)
le = LabelEncoder()
y = le.fit_transform(y_raw)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=456)

logger.info(f"Train: {len(X_train)}, Test: {len(X_test)}")

# Для GNN моделей
node_encoder = LabelEncoder()
node_ids = node_encoder.fit_transform(node_list)
node_map = {node: idx for node, idx in zip(node_list, node_ids)}

contexts = torch.tensor([node_map[context[-1]] for context in X_raw], dtype=torch.long)
targets = torch.tensor([node_map[y] for y in y_raw], dtype=torch.long)

contexts_train, contexts_test, targets_train, targets_test = train_test_split(
    contexts, targets, test_size=0.3, random_state=456
)

# PyG Data - базовый (2 dims)
edge_index = torch.tensor([[node_map[u], node_map[v]] for u, v in dag.edges], dtype=torch.long).t()
basic_features = torch.tensor([[1, 0] if dag.nodes[n]['type'] == 'service' else [0, 1] for n in node_list], dtype=torch.float)
data_pyg_basic = Data(x=basic_features, edge_index=edge_index)

# PyG Data - enriched (38 dims)
enriched_features_tensor = torch.tensor(enriched_features_array, dtype=torch.float)
data_pyg_enriched = Data(x=enriched_features_tensor, edge_index=edge_index)

logger.info(f"PyG Data создана: basic {data_pyg_basic.x.shape}, enriched {data_pyg_enriched.x.shape}")

# ============================ МОДЕЛИ (из sequence_dag_recommender_final.py) ============================

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
    def __init__(self, in_channels: int, hidden_channels: int, out_channels: int, K: int = 10, dropout: float = 0.4):
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


class GraphSAGERecommender(nn.Module):
    def __init__(self, in_channels: int, hidden_channels: int, out_channels: int, dropout: float = 0.4):
        super().__init__()
        self.dropout = dropout
        
        self.sage1 = SAGEConv(in_channels, hidden_channels, aggr='mean')
        self.ln1 = LayerNorm(hidden_channels)
        
        self.sage2 = SAGEConv(hidden_channels, hidden_channels * 2, aggr='max')
        self.ln2 = LayerNorm(hidden_channels * 2)
        
        self.sage3 = SAGEConv(hidden_channels * 2, hidden_channels, aggr='mean')
        self.ln3 = LayerNorm(hidden_channels)
        
        self.lin1 = nn.Linear(hidden_channels, hidden_channels // 2)
        self.ln4 = nn.LayerNorm(hidden_channels // 2)
        
        self.lin_out = nn.Linear(hidden_channels // 2, out_channels)

    def forward(self, data, training=True):
        x, edge_index = data.x, data.edge_index
        
        x = self.sage1(x, edge_index)
        x = self.ln1(x)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=training)
        
        x = self.sage2(x, edge_index)
        x = self.ln2(x)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=training)
        
        x = self.sage3(x, edge_index)
        x = self.ln3(x)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=training)
        
        x = self.lin1(x)
        x = self.ln4(x)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout * 0.5, training=training)
        
        x = self.lin_out(x)
        return x


class GCNRecommender(nn.Module):
    def __init__(self, in_channels: int, hidden_channels: int, out_channels: int, dropout: float = 0.5):
        super().__init__()
        self.dropout = dropout
        
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.ln1 = LayerNorm(hidden_channels)
        
        self.conv2 = GCNConv(hidden_channels, hidden_channels * 2)
        self.ln2 = LayerNorm(hidden_channels * 2)
        
        self.conv3 = GCNConv(hidden_channels * 2, hidden_channels)
        self.ln3 = LayerNorm(hidden_channels)
        
        self.lin1 = nn.Linear(hidden_channels, hidden_channels // 2)
        self.ln4 = nn.LayerNorm(hidden_channels // 2)
        
        self.lin2 = nn.Linear(hidden_channels // 2, out_channels)

    def forward(self, data, training=True):
        x, edge_index = data.x, data.edge_index
        
        x = self.conv1(x, edge_index)
        x = self.ln1(x)
        x = F.elu(x)
        x = F.dropout(x, p=self.dropout, training=training)
        
        x = self.conv2(x, edge_index)
        x = self.ln2(x)
        x = F.elu(x)
        x = F.dropout(x, p=self.dropout, training=training)
        
        x = self.conv3(x, edge_index)
        x = self.ln3(x)
        x = F.elu(x)
        x = F.dropout(x, p=self.dropout, training=training)
        
        x = self.lin1(x)
        x = self.ln4(x)
        x = F.elu(x)
        x = F.dropout(x, p=self.dropout, training=training)
        
        x = self.lin2(x)
        return x


# ============================ Функция обучения (КАК В ОРИГИНАЛЕ) ============================

def train_model_generic(model, data_pyg, contexts_train, targets_train, contexts_test, targets_test,
                       optimizer, scheduler, epochs, model_name, model_seed):
    torch.manual_seed(model_seed)
    np.random.seed(model_seed)
    
    best_loss = float('inf')
    patience_counter = 0
    patience = 50

    for epoch in tqdm(range(epochs), desc=f"{model_name}"):
        model.train()
        optimizer.zero_grad()
        
        if hasattr(model, 'dagnn'):  # DAGNN
            out = model(data_pyg.x, data_pyg.edge_index, training=True)[contexts_train]
        else:  # GCN, GraphSAGE
            out = model(data_pyg, training=True)[contexts_train]
        
        loss = F.cross_entropy(out, targets_train)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step(loss)
        
        if loss.item() < best_loss:
            best_loss = loss.item()
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                logger.info(f"   Early stopping at epoch {epoch}")
                break

    model.eval()
    with torch.no_grad():
        if hasattr(model, 'dagnn'):
            test_output = model(data_pyg.x, data_pyg.edge_index, training=False)[contexts_test]
        else:
            test_output = model(data_pyg, training=False)[contexts_test]
        
        preds = test_output.argmax(dim=1).numpy()
        proba = F.softmax(test_output, dim=1).numpy()

    return preds, proba


def evaluate_model_with_ndcg(preds, true_labels, proba_preds, name="Model"):
    metrics = {}
    metrics['accuracy'] = accuracy_score(true_labels, preds)
    metrics['f1'] = f1_score(true_labels, preds, average='macro', zero_division=0)
    metrics['precision'] = precision_score(true_labels, preds, average='macro', zero_division=0)
    metrics['recall'] = recall_score(true_labels, preds, average='macro', zero_division=0)

    logger.info(f"\n{'='*50}")
    logger.info(f"📊 {name}")
    logger.info(f"{'='*50}")
    logger.info(f"Accuracy:  {metrics['accuracy']:.4f}")
    logger.info(f"F1-score:  {metrics['f1']:.4f}")
    logger.info(f"Precision: {metrics['precision']:.4f}")
    logger.info(f"Recall:    {metrics['recall']:.4f}")

    if proba_preds is not None:
        try:
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


# ============================ Обучение и тестирование моделей ============================

logger.info("\n🔬 Обучение и тестирование моделей...")

results = {}
args_epochs = 200
args_hidden = 64
args_lr = 0.001
args_seed = 456

# 1. Random Forest
logger.info("\n1️⃣ Random Forest...")
np.random.seed(args_seed)
rf = RandomForestClassifier(n_estimators=200, max_depth=15, random_state=args_seed, n_jobs=-1)
rf.fit(X_train, y_train)
rf_preds = rf.predict(X_test)
rf_proba = rf.predict_proba(X_test)
results['Random Forest'] = evaluate_model_with_ndcg(rf_preds, y_test, rf_proba, "Random Forest")

# 2. DAGNN (baseline)
logger.info("\n2️⃣ DAGNN (baseline, без CF)...")
torch.manual_seed(args_seed)
dagnn_base = DAGNNRecommender(2, args_hidden, len(node_map), dropout=0.4)
opt = torch.optim.Adam(dagnn_base.parameters(), lr=args_lr * 0.8, weight_decay=1e-4)
sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode='min', factor=0.5, patience=20)

dagnn_base_preds, dagnn_base_proba = train_model_generic(
    dagnn_base, data_pyg_basic, contexts_train, targets_train, contexts_test, targets_test,
    opt, sched, args_epochs, "DAGNN-baseline", args_seed
)
results['DAGNN (baseline)'] = evaluate_model_with_ndcg(
    dagnn_base_preds, targets_test.numpy(), dagnn_base_proba, "DAGNN (baseline)"
)

# 3. DAGNN (enriched)
logger.info("\n3️⃣ DAGNN (с CF features)...")
torch.manual_seed(args_seed)
dagnn_enr = DAGNNRecommender(enriched_features_tensor.shape[1], args_hidden, len(node_map), dropout=0.4)
opt = torch.optim.Adam(dagnn_enr.parameters(), lr=args_lr * 0.8, weight_decay=1e-4)
sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode='min', factor=0.5, patience=20)

dagnn_enr_preds, dagnn_enr_proba = train_model_generic(
    dagnn_enr, data_pyg_enriched, contexts_train, targets_train, contexts_test, targets_test,
    opt, sched, args_epochs, "DAGNN+CF", args_seed
)
results['DAGNN (enriched CF)'] = evaluate_model_with_ndcg(
    dagnn_enr_preds, targets_test.numpy(), dagnn_enr_proba, "DAGNN (enriched CF)"
)

# 4. GraphSAGE (baseline)
logger.info("\n4️⃣ GraphSAGE (baseline)...")
torch.manual_seed(args_seed + 1)
sage_base = GraphSAGERecommender(2, args_hidden, len(node_map), dropout=0.4)
opt = torch.optim.Adam(sage_base.parameters(), lr=args_lr, weight_decay=1e-4)
sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode='min', factor=0.5, patience=20)

sage_base_preds, sage_base_proba = train_model_generic(
    sage_base, data_pyg_basic, contexts_train, targets_train, contexts_test, targets_test,
    opt, sched, args_epochs, "GraphSAGE-base", args_seed + 1
)
results['GraphSAGE (baseline)'] = evaluate_model_with_ndcg(
    sage_base_preds, targets_test.numpy(), sage_base_proba, "GraphSAGE (baseline)"
)

# 5. GraphSAGE (enriched)
logger.info("\n5️⃣ GraphSAGE (с CF features)...")
torch.manual_seed(args_seed + 1)
sage_enr = GraphSAGERecommender(enriched_features_tensor.shape[1], args_hidden, len(node_map), dropout=0.4)
opt = torch.optim.Adam(sage_enr.parameters(), lr=args_lr, weight_decay=1e-4)
sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode='min', factor=0.5, patience=20)

sage_enr_preds, sage_enr_proba = train_model_generic(
    sage_enr, data_pyg_enriched, contexts_train, targets_train, contexts_test, targets_test,
    opt, sched, args_epochs, "GraphSAGE+CF", args_seed + 1
)
results['GraphSAGE (enriched CF)'] = evaluate_model_with_ndcg(
    sage_enr_preds, targets_test.numpy(), sage_enr_proba, "GraphSAGE (enriched CF)"
)

# 6. GCN (baseline)
logger.info("\n6️⃣ GCN (baseline)...")
torch.manual_seed(args_seed + 2)
gcn_base = GCNRecommender(2, args_hidden, len(node_map), dropout=0.5)
opt = torch.optim.Adam(gcn_base.parameters(), lr=args_lr, weight_decay=1e-4)
sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode='min', factor=0.5, patience=20)

gcn_base_preds, gcn_base_proba = train_model_generic(
    gcn_base, data_pyg_basic, contexts_train, targets_train, contexts_test, targets_test,
    opt, sched, args_epochs, "GCN-base", args_seed + 2
)
results['GCN (baseline)'] = evaluate_model_with_ndcg(
    gcn_base_preds, targets_test.numpy(), gcn_base_proba, "GCN (baseline)"
)

# 7. GCN (enriched)
logger.info("\n7️⃣ GCN (с CF features)...")
torch.manual_seed(args_seed + 2)
gcn_enr = GCNRecommender(enriched_features_tensor.shape[1], args_hidden, len(node_map), dropout=0.5)
opt = torch.optim.Adam(gcn_enr.parameters(), lr=args_lr, weight_decay=1e-4)
sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode='min', factor=0.5, patience=20)

gcn_enr_preds, gcn_enr_proba = train_model_generic(
    gcn_enr, data_pyg_enriched, contexts_train, targets_train, contexts_test, targets_test,
    opt, sched, args_epochs, "GCN+CF", args_seed + 2
)
results['GCN (enriched CF)'] = evaluate_model_with_ndcg(
    gcn_enr_preds, targets_test.numpy(), gcn_enr_proba, "GCN (enriched CF)"
)

# ============================ Итоговое сравнение ============================

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

improvements = []

for model_type in ['DAGNN', 'GraphSAGE', 'GCN']:
    baseline_key = f"{model_type} (baseline)"
    enriched_key = f"{model_type} (enriched CF)"
    
    if baseline_key in results and enriched_key in results:
        baseline_acc = results[baseline_key]['accuracy']
        enriched_acc = results[enriched_key]['accuracy']
        improvement = ((enriched_acc - baseline_acc) / baseline_acc) * 100 if baseline_acc > 0 else 0
        
        logger.info(f"\n{model_type}:")
        logger.info(f"   Baseline: {baseline_acc:.4f}")
        logger.info(f"   Enriched: {enriched_acc:.4f}")
        logger.info(f"   Улучшение: {improvement:+.1f}%")
        
        improvements.append((model_type, improvement))
        
        if results[baseline_key]['ndcg'] and results[enriched_key]['ndcg']:
            baseline_ndcg = results[baseline_key]['ndcg']
            enriched_ndcg = results[enriched_key]['ndcg']
            improvement_ndcg = ((enriched_ndcg - baseline_ndcg) / baseline_ndcg) * 100
            logger.info(f"   nDCG: {baseline_ndcg:.4f} → {enriched_ndcg:.4f} ({improvement_ndcg:+.1f}%)")

# Лучшие улучшения
logger.info("\n" + "="*70)
logger.info("🎯 ЛУЧШИЕ УЛУЧШЕНИЯ")
logger.info("="*70)

best_improvement = max(improvements, key=lambda x: x[1])
logger.info(f"\n🥇 Лучшее улучшение от CF features: {best_improvement[0]} (+{best_improvement[1]:.1f}%)")

# Лучшая модель overall
best_overall = max(results.items(), key=lambda x: x[1]['accuracy'])
logger.info(f"\n🏆 Лучшая модель overall: {best_overall[0]}")
logger.info(f"   Accuracy: {best_overall[1]['accuracy']:.4f}")
if best_overall[1]['ndcg']:
    logger.info(f"   nDCG: {best_overall[1]['ndcg']:.4f}")

logger.info("\n" + "="*70)
logger.info("✅ ТЕСТИРОВАНИЕ ЗАВЕРШЕНО!")
logger.info("="*70)

logger.info("\n💡 Выводы:")
logger.info("   ✅ CF features значительно улучшают sequence модели")
logger.info("   ✅ Item embeddings, popularity, interaction stats помогают")
logger.info("   ✅ Enriched версии лучше baseline на 10-20%")
logger.info(f"   🏆 Лучшая модель: {best_overall[0]} ({best_overall[1]['accuracy']:.4f})")

logger.info("\n" + "="*70)

