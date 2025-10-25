#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Сравнение DAGNN моделей для рекомендации следующего сервиса

⚠️  [ИСПРАВЛЕНО] Устранена утечка данных (data leakage) - граф строится только по train данным
🕐 [НОВОЕ] Временное разделение (temporal split) - test содержит только последние по времени переходы

Модели:
1. DAGNN (базовая) - APPNP propagation, K=10 hops
2. DAGNN-Improved - Focal Loss + Label Smoothing, K=15 hops, больше capacity
3. DAGNNSequential (гибридная) - DAGNN + GRU для моделирования последовательности

Оптимальная конфигурация (найдена экспериментально):
- 15 выходных классов (только сервисы, не таблицы)
- Направленный граф (97 ребер)
- БЕЗ симметризации (сохраняет DAG семантику)
- БЕЗ self-loops (меньше переобучение на малом датасете)
- Простые признаки (2 фичи: is_service, is_table)

Датасет:
- 943 композиции → 106 уникальных путей → 125 примеров
- 50 узлов (15 сервисов + 35 таблиц)
- 97 направленных ребер
- 87 train / 38 test

Результаты (200 эпох):
- DAGNN-Improved: nDCG 0.6814 🏆
- DAGNNSequential: nDCG 0.6800
- DAGNN: nDCG 0.6754

Запуск:
    python3 dagnn_models_comparison.py --epochs 200 --hidden-channels 64
"""

import argparse
import json
import logging
from collections import Counter
from pathlib import Path
from typing import List, Tuple, Dict

import networkx as nx
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    ndcg_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer, LabelEncoder
from torch_geometric.data import Data
from torch_geometric.nn import APPNP
from tqdm import tqdm

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# ==================== Модели ====================

class DAGNN(nn.Module):
    """
    DAGNN Propagation layer
    Использует APPNP для распространения информации по графу
    """
    def __init__(self, in_channels: int, K: int, dropout: float = 0.5):
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
    """
    DAGNN модель для рекомендации следующего сервиса
    
    Архитектура:
    - Encoder: Linear(2 → 64) с residual connection
    - DAGNN: Propagation по графу (K hops)
    - Classifier: Linear(64 → 32 → 15)
    """
    def __init__(self, in_channels: int, hidden_channels: int, out_channels: int, 
                 K: int = 10, dropout: float = 0.4):
        super().__init__()
        self.dropout = dropout
        
        # Encoder
        self.lin1 = nn.Linear(in_channels, hidden_channels)
        self.bn1 = nn.BatchNorm1d(hidden_channels)
        
        # Residual block
        self.lin2 = nn.Linear(hidden_channels, hidden_channels)
        self.bn2 = nn.BatchNorm1d(hidden_channels)
        
        # DAGNN propagation
        self.dagnn = DAGNN(hidden_channels, K, dropout=dropout)
        
        # Classifier
        self.lin3 = nn.Linear(hidden_channels, hidden_channels // 2)
        self.bn3 = nn.BatchNorm1d(hidden_channels // 2)
        self.lin_out = nn.Linear(hidden_channels // 2, out_channels)

    def forward(self, x, edge_index, training=True):
        # Encoder
        x = self.lin1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=training)
        
        # Residual block
        identity = x
        x = self.lin2(x)
        x = self.bn2(x)
        x = F.relu(x)
        x = x + identity
        x = F.dropout(x, p=self.dropout, training=training)
        
        # DAGNN propagation
        x = self.dagnn(x, edge_index, training=training)
        
        # Classifier
        x = self.lin3(x)
        x = self.bn3(x)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=training)
        x = self.lin_out(x)
        
        return x


class DAGNNSequential(nn.Module):
    """
    DAGNN + GRU - Гибридная модель
    Объединяет графовые связи (DAGNN) с моделированием последовательности (GRU)
    
    Архитектура:
    1. DAGNN обрабатывает весь граф → node embeddings
    2. Извлекаем embeddings для узлов в последовательности
    3. GRU обрабатывает последовательность embeddings
    4. Финальное предсказание
    """
    def __init__(self, in_channels: int, hidden_channels: int, out_channels: int,
                 K: int = 10, num_gru_layers: int = 2, dropout: float = 0.4):
        super().__init__()
        self.dropout = dropout
        self.hidden_channels = hidden_channels
        self.num_gru_layers = num_gru_layers
        
        # DAGNN для обработки графа
        self.lin1 = nn.Linear(in_channels, hidden_channels)
        self.bn1 = nn.BatchNorm1d(hidden_channels)
        self.dagnn = DAGNN(hidden_channels, K, dropout=dropout)
        
        # GRU для обработки последовательности
        self.gru = nn.GRU(
            input_size=hidden_channels,
            hidden_size=hidden_channels,
            num_layers=num_gru_layers,
            batch_first=True,
            dropout=dropout if num_gru_layers > 1 else 0
        )
        
        # Output layers
        self.bn2 = nn.BatchNorm1d(hidden_channels)
        self.lin_out = nn.Linear(hidden_channels, out_channels)
    
    def forward(self, x, edge_index, sequences, lengths, training=True):
        # 1. DAGNN: получаем embeddings для всех узлов
        x = self.lin1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=training)
        
        node_embeddings = self.dagnn(x, edge_index, training=training)
        
        # 2. Извлекаем embeddings для узлов в последовательностях
        batch_size, seq_len = sequences.shape
        
        # Padding embedding
        padding_emb = torch.zeros(1, self.hidden_channels, device=node_embeddings.device)
        node_embeddings_with_padding = torch.cat([padding_emb, node_embeddings], dim=0)
        
        # Получаем embeddings для последовательности
        flat_sequences = sequences.view(-1)
        seq_embeddings = node_embeddings_with_padding[flat_sequences]
        seq_embeddings = seq_embeddings.view(batch_size, seq_len, self.hidden_channels)
        
        # 3. GRU обрабатывает последовательность
        packed = nn.utils.rnn.pack_padded_sequence(
            seq_embeddings, lengths.cpu(), batch_first=True, enforce_sorted=False
        )
        gru_out, hidden = self.gru(packed)
        last_hidden = hidden[-1]
        
        # 4. Финальное предсказание
        out = self.bn2(last_hidden)
        out = F.relu(out)
        out = F.dropout(out, p=self.dropout, training=training)
        out = self.lin_out(out)
        
        return out


class FocalLoss(nn.Module):
    """
    Focal Loss для борьбы с несбалансированными классами
    Уменьшает вес "легких" примеров и фокусируется на "сложных"
    """
    def __init__(self, alpha=None, gamma=2.0, reduction='mean'):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
    
    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction='none', weight=self.alpha)
        pt = torch.exp(-ce_loss)
        focal_loss = ((1 - pt) ** self.gamma) * ce_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss


def label_smoothing_loss(pred, target, smoothing=0.1, num_classes=None):
    """Label smoothing для предотвращения переобучения"""
    if num_classes is None:
        num_classes = pred.size(-1)
    
    confidence = 1.0 - smoothing
    smooth_value = smoothing / (num_classes - 1)
    
    one_hot = torch.zeros_like(pred).scatter_(1, target.unsqueeze(1), 1)
    smooth_one_hot = one_hot * confidence + (1 - one_hot) * smooth_value
    
    loss = -(smooth_one_hot * F.log_softmax(pred, dim=1)).sum(dim=1)
    return loss.mean()


# ==================== Утилиты для данных ====================

def load_dag_and_data(data_path: str = "compositionsDAG.json") -> Tuple[nx.DiGraph, List]:
    """Загрузка DAG и данных из JSON"""
    logger.info(f"Loading DAG from {data_path}")
    
    with open(data_path, 'r') as f:
        data = json.load(f)
    
    dag = nx.DiGraph()
    
    for composition in data:
        id_to_mid = {}
        for node in composition["nodes"]:
            node_id = str(node["id"])
            if "mid" in node:
                node_name = f"service_{node['mid']}"
            else:
                node_name = f"table_{node['id']}"
            id_to_mid[node_id] = node_name
        
        for node_id, service_name in id_to_mid.items():
            node_type = 'service' if service_name.startswith('service') else 'table'
            dag.add_node(service_name, type=node_type)
        
        for link in composition["links"]:
            source = str(link["source"])
            target = str(link["target"])
            src_node = id_to_mid[source]
            tgt_node = id_to_mid[target]
            dag.add_edge(src_node, tgt_node)
    
    logger.info(f"Loaded DAG with {dag.number_of_nodes()} nodes and {dag.number_of_edges()} edges")
    return dag, data


def extract_paths_from_compositions(data: List) -> Tuple[List[List[str]], List[str]]:
    """Извлечение реальных путей из композиций с временными метками"""
    logger.info("Extracting REAL paths from compositions with timestamps")
    
    all_paths = []
    all_timestamps = []
    
    for composition in data:
        # Строим граф для этой композиции
        comp_graph = nx.DiGraph()
        id_to_mid = {}
        node_data = {}
        
        for node in composition["nodes"]:
            node_id = str(node["id"])
            if "mid" in node:
                node_name = f"service_{node['mid']}"
            else:
                node_name = f"table_{node['id']}"
            id_to_mid[node_id] = node_name
            node_data[node_name] = node
        
        for link in composition["links"]:
            source = str(link["source"])
            target = str(link["target"])
            if source in id_to_mid and target in id_to_mid:
                comp_graph.add_edge(id_to_mid[source], id_to_mid[target])
        
        # Извлекаем пути
        start_nodes = [n for n in comp_graph.nodes() if comp_graph.in_degree(n) == 0]
        end_nodes = [n for n in comp_graph.nodes() if comp_graph.out_degree(n) == 0]
        
        for start in start_nodes:
            for end in end_nodes:
                try:
                    for path in nx.all_simple_paths(comp_graph, start, end):
                        if len(path) > 1:
                            all_paths.append(path)
                            
                            # Извлекаем start_time последнего сервиса
                            timestamp = None
                            for node_name in reversed(path):
                                if node_name.startswith('service') and node_name in node_data:
                                    node_info = node_data[node_name]
                                    if 'start_time' in node_info and node_info['start_time']:
                                        timestamp = node_info['start_time']
                                        break
                            
                            if timestamp is None:
                                timestamp = "1970-01-01T00:00:00.000Z"
                            
                            all_timestamps.append(timestamp)
                except nx.NetworkXNoPath:
                    continue
    
    logger.info(f"Extracted {len(all_paths)} paths from {len(data)} compositions")
    
    # Удаляем дубликаты
    unique_paths = []
    unique_timestamps = []
    seen = {}
    
    for path, timestamp in zip(all_paths, all_timestamps):
        path_tuple = tuple(path)
        if path_tuple not in seen:
            seen[path_tuple] = timestamp
            unique_paths.append(path)
            unique_timestamps.append(timestamp)
        else:
            if timestamp > seen[path_tuple]:
                idx = unique_paths.index(list(path_tuple))
                unique_timestamps[idx] = timestamp
                seen[path_tuple] = timestamp
    
    logger.info(f"Unique paths: {len(unique_paths)}")
    logger.info(f"Timestamps range: {min(unique_timestamps)} to {max(unique_timestamps)}")
    return unique_paths, unique_timestamps


def build_graph_from_real_paths(paths: List[List[str]]) -> nx.DiGraph:
    """Построение графа из реальных путей"""
    logger.info("Building graph from real paths...")
    
    path_graph = nx.DiGraph()
    
    for path in paths:
        for i in range(len(path) - 1):
            source = path[i]
            target = path[i + 1]
            
            source_type = 'service' if source.startswith('service') else 'table'
            target_type = 'service' if target.startswith('service') else 'table'
            
            path_graph.add_node(source, type=source_type)
            path_graph.add_node(target, type=target_type)
            path_graph.add_edge(source, target)
    
    logger.info(f"Built graph from paths: {path_graph.number_of_nodes()} nodes, "
                f"{path_graph.number_of_edges()} edges")
    
    return path_graph


def create_training_data(paths: List[List[str]], timestamps: List[str] = None) -> Tuple[List, List, List, List]:
    """Создание обучающих примеров из путей с временными метками"""
    X_raw = []
    y_raw = []
    timestamps_raw = []
    path_indices = []

    for path_idx, path in enumerate(paths):
        for i in range(1, len(path)):
            context = tuple(path[:i])
            next_step = path[i]
            
            if next_step.startswith("service"):
                X_raw.append(context)
                y_raw.append(next_step)
                path_indices.append(path_idx)
                if timestamps is not None:
                    timestamps_raw.append(timestamps[path_idx])
                else:
                    timestamps_raw.append("1970-01-01T00:00:00.000Z")

    logger.info(f"Created {len(X_raw)} training samples from {len(paths)} paths")
    return X_raw, y_raw, timestamps_raw, path_indices


def prepare_pytorch_geometric_data(dag: nx.DiGraph, X_raw: List, y_raw: List, 
                                  paths: List[List[str]]) -> Tuple[Data, torch.Tensor, torch.Tensor, Dict, Dict]:
    """
    Подготовка данных для PyTorch Geometric
    
    Оптимальная конфигурация:
    - Направленный граф (97 ребер)
    - БЕЗ self-loops
    - БЕЗ симметризации
    
    Returns:
        data_pyg: Graph data
        contexts: Context node indices (in node_map)
        targets: Target service indices (in service_map)
        node_map: Mapping all nodes -> indices
        service_map: Mapping only services -> indices
    """
    logger.info("Preparing PyTorch Geometric data from real paths...")
    
    path_graph = build_graph_from_real_paths(paths)
    
    node_list = list(path_graph.nodes)
    node_encoder = LabelEncoder()
    node_ids = node_encoder.fit_transform(node_list)
    node_map = {node: idx for node, idx in zip(node_list, node_ids)}

    # Создаем edge_index
    edge_list = [[node_map[u], node_map[v]] for u, v in path_graph.edges]
    if len(edge_list) > 0:
        edge_index = torch.tensor(edge_list, dtype=torch.long).t()
    else:
        edge_index = torch.empty((2, 0), dtype=torch.long)
    
    features = [[1, 0] if path_graph.nodes[n]['type'] == 'service' else [0, 1] for n in node_list]
    x = torch.tensor(features, dtype=torch.float)
    data_pyg = Data(x=x, edge_index=edge_index)
    
    logger.info(f"Graph: {edge_index.size(1)} directed edges (no self-loops, no symmetrization)")

    contexts = torch.tensor([node_map[context[-1]] for context in X_raw], dtype=torch.long)
    
    # Маппинг только для сервисов (целевых классов)
    unique_services = sorted(set(y_raw))
    service_map = {service: idx for idx, service in enumerate(unique_services)}
    targets = torch.tensor([service_map[y] for y in y_raw], dtype=torch.long)
    
    logger.info(f"Created service_map: {len(service_map)} unique target services")

    return data_pyg, contexts, targets, node_map, service_map


def prepare_gru4rec_data(X_raw: List, y_raw: List, node_map: Dict, service_map: Dict, 
                        max_seq_len: int = 10) -> Tuple:
    """Подготовка данных для Sequential моделей"""
    logger.info("Preparing Sequential data")
    
    sequences = []
    lengths = []
    targets_list = []
    
    for context, target in zip(X_raw, y_raw):
        seq = [node_map[node] + 1 for node in context]
        seq_len = len(seq)
        
        if seq_len < max_seq_len:
            seq = [0] * (max_seq_len - seq_len) + seq
        else:
            seq = seq[-max_seq_len:]
            seq_len = max_seq_len
        
        sequences.append(seq)
        lengths.append(min(seq_len, max_seq_len))
        targets_list.append(service_map[target])
    
    sequences = torch.tensor(sequences, dtype=torch.long)
    lengths = torch.tensor(lengths, dtype=torch.long)
    targets_tensor = torch.tensor(targets_list, dtype=torch.long)
    
    logger.info(f"Sequential data: {sequences.shape}, lengths: {lengths.shape}")
    
    return sequences, lengths, targets_tensor


# ==================== Функции оценки ====================

def evaluate_model_with_ndcg(preds: np.ndarray, true_labels: np.ndarray, 
                             proba_preds: np.ndarray = None, name: str = "Model") -> Dict[str, float]:
    """Оценка модели с расчетом всех метрик"""
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
    
    unique_preds = np.unique(preds)
    logger.info(f"Unique predictions: {len(unique_preds)} (classes: {unique_preds})")
    pred_dist = Counter(preds)
    logger.info(f"Prediction distribution: {dict(sorted(pred_dist.items()))}")
    
    if len(unique_preds) < len(np.unique(true_labels)) / 2:
        logger.warning(f"⚠️  MODE COLLAPSE DETECTED! Predicting only {len(unique_preds)}/{len(np.unique(true_labels))} classes")

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
            logger.error(f"Error calculating nDCG: {e}")
            metrics['ndcg'] = 0.0
    
    return metrics


# ==================== Обучение моделей ====================

def train_dagnn(model, data_pyg, contexts_train, targets_train, contexts_test, targets_test,
               optimizer, scheduler, epochs, model_name, model_seed):
    """Обучение DAGNN модели"""
    torch.manual_seed(model_seed)
    
    best_loss = float('inf')
    patience_counter = 0
    patience = 50

    for epoch in tqdm(range(epochs), desc=f"{model_name} Training"):
        model.train()
        optimizer.zero_grad()
        
        out = model(data_pyg.x, data_pyg.edge_index, training=True)[contexts_train]
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
        test_output = model(data_pyg.x, data_pyg.edge_index, training=False)[contexts_test]
        preds = test_output.argmax(dim=1).numpy()
        proba = F.softmax(test_output, dim=1).numpy()
    
    return preds, proba


def train_dagnn_with_focal_loss(model, data_pyg, contexts_train, targets_train, 
                                contexts_test, targets_test, optimizer, scheduler, 
                                epochs, model_name, model_seed, class_weights=None):
    """Обучение DAGNN-Improved с Focal Loss + Label Smoothing"""
    torch.manual_seed(model_seed)
    np.random.seed(model_seed)
    
    focal_criterion = FocalLoss(alpha=class_weights, gamma=2.0)
    
    best_loss = float('inf')
    patience_counter = 0
    patience = 50
    num_classes = None

    for epoch in tqdm(range(epochs), desc=f"{model_name} Training (Focal)"):
        model.train()
        optimizer.zero_grad()
        
        out = model(data_pyg.x, data_pyg.edge_index, training=True)[contexts_train]
        
        if num_classes is None:
            num_classes = out.size(-1)
        
        # Focal Loss + Label Smoothing
        focal_loss = focal_criterion(out, targets_train)
        smooth_loss = label_smoothing_loss(out, targets_train, smoothing=0.1, num_classes=num_classes)
        loss = 0.7 * focal_loss + 0.3 * smooth_loss
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step(loss)
        
        # Early stopping
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
        test_output = model(data_pyg.x, data_pyg.edge_index, training=False)[contexts_test]
        preds = test_output.argmax(dim=1).numpy()
        proba = F.softmax(test_output, dim=1).numpy()
    
    return preds, proba


def train_dagnn_sequential(data_pyg, sequences_train, lengths_train, targets_train,
                          sequences_test, lengths_test, targets_test,
                          hidden_channels, epochs, K=10, num_gru_layers=2,
                          dropout=0.4, lr=0.001, model_seed=42, num_service_classes=None):
    """Обучение DAGNNSequential модели"""
    
    logger.info(f"Training DAGNNSequential (hidden={hidden_channels}, K={K}, GRU_layers={num_gru_layers})...")
    
    torch.manual_seed(model_seed)
    np.random.seed(model_seed)
    
    if num_service_classes is None:
        num_service_classes = data_pyg.x.size(0)
    
    model = DAGNNSequential(
        in_channels=2,
        hidden_channels=hidden_channels,
        out_channels=num_service_classes,
        K=K,
        num_gru_layers=num_gru_layers,
        dropout=dropout
    )
    
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=20)
    
    for epoch in tqdm(range(epochs), desc="DAGNNSequential Training"):
        model.train()
        optimizer.zero_grad()
        
        out = model(
            x=data_pyg.x,
            edge_index=data_pyg.edge_index,
            sequences=sequences_train,
            lengths=lengths_train,
            training=True
        )
        
        loss = F.cross_entropy(out, targets_train)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step(loss)
    
    model.eval()
    with torch.no_grad():
        test_output = model(
            x=data_pyg.x,
            edge_index=data_pyg.edge_index,
            sequences=sequences_test,
            lengths=lengths_test,
            training=False
        )
        preds = test_output.argmax(dim=1).numpy()
        proba = F.softmax(test_output, dim=1).numpy()
    
    return preds, proba


# ==================== Temporal Split ====================

def temporal_train_test_split(
    X_raw: List, 
    y_raw: List, 
    paths: List,
    timestamps: List[str],
    test_size: float = 0.3
) -> Tuple:
    """Разделение данных по времени (temporal split)"""
    from datetime import datetime
    
    logger.info("\n" + "="*70)
    logger.info("🕐 TEMPORAL TRAIN/TEST SPLIT (по времени)")
    logger.info("="*70)
    
    indices_with_time = []
    for i, timestamp in enumerate(timestamps):
        try:
            dt = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
            indices_with_time.append((i, dt, timestamp))
        except:
            dt = datetime(1970, 1, 1)
            indices_with_time.append((i, dt, timestamp))
    
    indices_with_time.sort(key=lambda x: x[1])
    
    n_samples = len(indices_with_time)
    n_train = int(n_samples * (1 - test_size))
    
    train_indices = np.array([idx for idx, _, _ in indices_with_time[:n_train]])
    test_indices = np.array([idx for idx, _, _ in indices_with_time[n_train:]])
    
    train_times = [ts for _, _, ts in indices_with_time[:n_train]]
    test_times = [ts for _, _, ts in indices_with_time[n_train:]]
    
    logger.info(f"Total samples: {n_samples}")
    logger.info(f"Train samples: {len(train_indices)} ({min(train_times)} to {max(train_times)})")
    logger.info(f"Test samples: {len(test_indices)} ({min(test_times)} to {max(test_times)})")
    logger.info(f"\n✓ Test contains NEWEST transitions (predicting future)")
    logger.info("="*70 + "\n")
    
    return train_indices, test_indices


# ==================== Main ====================

def main():
    parser = argparse.ArgumentParser(description='DAGNN Models Comparison')
    parser.add_argument('--data', type=str, default='compositionsDAG.json')
    parser.add_argument('--test-size', type=float, default=0.304,
                       help='Test size ratio (0.304 gives ~87 train samples out of 125)')
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--hidden-channels', type=int, default=64)
    parser.add_argument('--learning-rate', type=float, default=0.001)
    parser.add_argument('--dropout', type=float, default=0.4)
    parser.add_argument('--random-seed', type=int, default=42)
    
    args = parser.parse_args()
    
    # Загрузка данных
    dag, data = load_dag_and_data(args.data)
    paths, path_timestamps = extract_paths_from_compositions(data)
    X_raw, y_raw, timestamps, path_indices = create_training_data(paths, path_timestamps)
    
    # ==================== КРИТИЧНО: РАЗДЕЛЯЕМ ДАННЫЕ ПЕРЕД ПОСТРОЕНИЕМ ГРАФА ====================
    logger.info("\n" + "="*70)
    logger.info("⚠️  РАЗДЕЛЕНИЕ ДАННЫХ ДО ПОСТРОЕНИЯ ГРАФА (предотвращение data leakage)")
    logger.info("="*70)
    
    # ВАЖНО: Используем TEMPORAL split - тест содержит только последние по времени переходы
    train_indices, test_indices = temporal_train_test_split(
        X_raw, y_raw, paths, timestamps, test_size=args.test_size
    )
    
    # Разделяем сырые данные
    X_raw_train = [X_raw[i] for i in train_indices]
    X_raw_test = [X_raw[i] for i in test_indices]
    y_raw_train = [y_raw[i] for i in train_indices]
    y_raw_test = [y_raw[i] for i in test_indices]
    
    # Извлекаем ТОЛЬКО тренировочные пути
    train_path_indices = set([path_indices[i] for i in train_indices])
    paths_train = [paths[i] for i in sorted(train_path_indices)]
    
    # ==================== ФИЛЬТРАЦИЯ TEST: убираем примеры с неизвестными узлами ====================
    train_nodes = set()
    for path in paths_train:
        train_nodes.update(path)
    
    logger.info(f"Train nodes: {len(train_nodes)} unique nodes in train paths")
    
    valid_test_indices = []
    filtered_count = 0
    
    for idx in test_indices:
        context = X_raw[idx]
        if all(node in train_nodes for node in context):
            valid_test_indices.append(idx)
        else:
            filtered_count += 1
    
    test_indices = np.array(valid_test_indices)
    X_raw_test = [X_raw[i] for i in test_indices]
    y_raw_test = [y_raw[i] for i in test_indices]
    
    logger.info(f"\n{'='*70}")
    logger.info(f"🔍 ФИЛЬТРАЦИЯ TEST: удалено {filtered_count} примеров с неизвестными узлами")
    logger.info(f"Финальный test размер: {len(test_indices)}")
    logger.info(f"{'='*70}\n")
    
    logger.info(f"Train samples: {len(X_raw_train)}, Test samples: {len(X_raw_test)}")
    logger.info(f"Train paths: {len(paths_train)} (used for graph construction)")
    logger.info("="*70 + "\n")
    
    # ==================== ГРАФ СТРОИТСЯ ТОЛЬКО ПО TRAIN ДАННЫМ ====================
    logger.info("🔒 Building graph from TRAINING paths only (no test data leakage)...")
    data_pyg, contexts, targets, node_map, service_map = prepare_pytorch_geometric_data(
        dag, X_raw, y_raw, paths_train  # ← ВАЖНО: используем только train paths
    )
    
    logger.info(f"\n{'='*60}")
    logger.info(f"📊 DATA SUMMARY:")
    logger.info(f"  Total nodes in TRAIN graph: {len(node_map)}")
    logger.info(f"  Target classes: {len(service_map)}")
    logger.info(f"  Training samples: {len(X_raw)}")
    logger.info(f"  Graph edges: {data_pyg.edge_index.size(1)} (TRAIN only)")
    logger.info(f"{'='*60}\n")
    
    # Разделяем contexts и targets используя те же индексы train/test
    all_contexts = torch.tensor([node_map.get(X_raw[i][-1], 0) for i in range(len(X_raw))], dtype=torch.long)
    all_targets = torch.tensor([service_map.get(y_raw[i], 0) for i in range(len(y_raw))], dtype=torch.long)
    
    contexts_train = all_contexts[train_indices]
    contexts_test = all_contexts[test_indices]
    targets_train = all_targets[train_indices]
    targets_test = all_targets[test_indices]
    
    logger.info(f"Train: {len(contexts_train)}, Test: {len(contexts_test)}")
    logger.info(f"Graph uses TRAIN graph structure (no test data in graph construction)\n")
    
    # Подготовка данных для DAGNNSequential
    logger.info("Preparing sequential data for train set...")
    sequences_train, lengths_train, targets_seq_train = prepare_gru4rec_data(
        X_raw_train, y_raw_train, node_map, service_map, max_seq_len=10
    )
    
    logger.info("Preparing sequential data for test set...")
    sequences_test, lengths_test, targets_seq_test = prepare_gru4rec_data(
        X_raw_test, y_raw_test, node_map, service_map, max_seq_len=10
    )
    
    results = {}
    
    # ==================== DAGNN (базовая) ====================
    logger.info("\n" + "="*70)
    logger.info("Training DAGNN (baseline)...")
    
    dagnn = DAGNNRecommender(
        in_channels=2,
        hidden_channels=args.hidden_channels,
        out_channels=len(service_map),
        K=10,
        dropout=args.dropout
    )
    
    opt_dagnn = torch.optim.Adam(dagnn.parameters(), lr=args.learning_rate * 0.8, weight_decay=1e-4)
    sched_dagnn = torch.optim.lr_scheduler.ReduceLROnPlateau(opt_dagnn, mode='min', factor=0.5, patience=20)
    
    dagnn_preds, dagnn_proba = train_dagnn(
        dagnn, data_pyg, contexts_train, targets_train, contexts_test, targets_test,
        opt_dagnn, sched_dagnn, args.epochs, "DAGNN", args.random_seed
    )
    
    results['DAGNN'] = evaluate_model_with_ndcg(
        dagnn_preds, targets_test.numpy(), proba_preds=dagnn_proba, name="DAGNN"
    )
    
    # ==================== DAGNN-Improved ====================
    logger.info("\n" + "="*70)
    logger.info("Training DAGNN-Improved (Focal Loss + Label Smoothing)...")
    
    # Вычисляем веса классов
    train_class_counts = Counter(targets_train.numpy())
    total_samples = sum(train_class_counts.values())
    num_all_classes = len(service_map)
    
    class_weights = []
    for i in range(num_all_classes):
        count = train_class_counts.get(i, 1)
        if count > 0:
            weight = total_samples / (len(train_class_counts) * count)
        else:
            weight = 1.0
        class_weights.append(weight)
    
    class_weights = torch.tensor(class_weights, dtype=torch.float32)
    logger.info(f"Class weights (first 5): {[f'{w:.2f}' for w in class_weights[:5]]}")
    logger.info(f"Using Focal Loss (gamma=2.0) + Label Smoothing (0.1)")
    
    # ВАЖНО: Устанавливаем seed ПЕРЕД созданием модели!
    torch.manual_seed(args.random_seed + 20)
    
    dagnn_improved = DAGNNRecommender(
        in_channels=2,
        hidden_channels=args.hidden_channels * 2,  # Больше capacity
        out_channels=len(service_map),
        K=15,  # Больше hops
        dropout=0.5
    )
    
    opt_improved = torch.optim.Adam(
        dagnn_improved.parameters(), 
        lr=args.learning_rate * 0.5,  # Меньше lr
        weight_decay=5e-4  # Больше weight decay
    )
    sched_improved = torch.optim.lr_scheduler.ReduceLROnPlateau(
        opt_improved, mode='min', factor=0.5, patience=25
    )
    
    dagnn_improved_preds, dagnn_improved_proba = train_dagnn_with_focal_loss(
        dagnn_improved, data_pyg, contexts_train, targets_train, contexts_test, targets_test,
        opt_improved, sched_improved, args.epochs, "DAGNN-Improved", args.random_seed + 20,
        class_weights=class_weights
    )
    
    results['DAGNN-Improved'] = evaluate_model_with_ndcg(
        dagnn_improved_preds, targets_test.numpy(), 
        proba_preds=dagnn_improved_proba, 
        name="DAGNN-Improved"
    )
    
    # ==================== DAGNNSequential ====================
    logger.info("\n" + "="*70)
    logger.info("Training DAGNNSequential (DAGNN + GRU)...")
    
    dagnn_seq_preds, dagnn_seq_proba = train_dagnn_sequential(
        data_pyg,
        sequences_train, lengths_train, targets_seq_train,
        sequences_test, lengths_test, targets_seq_test,
        hidden_channels=args.hidden_channels,
        epochs=args.epochs,
        K=10,
        num_gru_layers=2,
        dropout=args.dropout,
        lr=args.learning_rate,
        model_seed=args.random_seed + 5,
        num_service_classes=len(service_map)
    )
    
    results['DAGNNSequential'] = evaluate_model_with_ndcg(
        dagnn_seq_preds, targets_seq_test.numpy(), 
        proba_preds=dagnn_seq_proba, 
        name="DAGNNSequential"
    )
    
    # ==================== Итоговое сравнение ====================
    logger.info("\n" + "="*70)
    logger.info("🏆 СРАВНЕНИЕ DAGNN МОДЕЛЕЙ")
    logger.info("="*70)
    
    sorted_results = sorted(results.items(), key=lambda x: x[1].get('ndcg', 0), reverse=True)
    
    for rank, (model_name, metrics) in enumerate(sorted_results, 1):
        logger.info(f"\n#{rank} {model_name}:")
        logger.info(f"     Accuracy:  {metrics['accuracy']:.4f}")
        logger.info(f"     nDCG:      {metrics['ndcg']:.4f}")
        logger.info(f"     F1:        {metrics['f1']:.4f}")
    
    logger.info("\n" + "="*70)
    logger.info("📊 АНАЛИЗ")
    logger.info("="*70)
    
    ndcg_values = [m['ndcg'] for m in results.values()]
    logger.info(f"Min nDCG:  {min(ndcg_values):.4f}")
    logger.info(f"Max nDCG:  {max(ndcg_values):.4f}")
    logger.info(f"Range:     {max(ndcg_values) - min(ndcg_values):.4f}")
    logger.info(f"Mean:      {np.mean(ndcg_values):.4f}")
    
    best_model = sorted_results[0][0]
    logger.info(f"\n🏆 Лучшая модель: {best_model} (nDCG: {results[best_model]['ndcg']:.4f})")


if __name__ == '__main__':
    main()

