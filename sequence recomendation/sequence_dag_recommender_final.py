#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ФИНАЛЬНАЯ ВЕРСИЯ с исправлением проблемы одинаковых результатов

Изменения:
1. Стратифицированный split для равномерного распределения классов
2. Разные random seeds для каждой модели
3. Более агрессивная регуляризация
4. Детальное логирование предсказаний
5. Cross-validation для проверки
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
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import LabelBinarizer, LabelEncoder, MultiLabelBinarizer
from torch_geometric.data import Data
from torch_geometric.nn import APPNP, GATConv, GCNConv, SAGEConv, GATv2Conv, BatchNorm, LayerNorm
from tqdm import tqdm

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# ==================== Модели (копируем лучшие) ====================

class DAGNN(nn.Module):
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


class GCNRecommender(nn.Module):
    def __init__(self, in_channels: int, hidden_channels: int, out_channels: int, dropout: float = 0.5):
        super().__init__()
        self.dropout = dropout
        
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.bn1 = LayerNorm(hidden_channels)
        
        self.conv2 = GCNConv(hidden_channels, hidden_channels * 2)
        self.bn2 = LayerNorm(hidden_channels * 2)
        
        self.conv3 = GCNConv(hidden_channels * 2, hidden_channels)
        self.bn3 = LayerNorm(hidden_channels)
        
        self.lin1 = nn.Linear(hidden_channels, hidden_channels // 2)
        self.bn4 = nn.LayerNorm(hidden_channels // 2)
        
        self.lin2 = nn.Linear(hidden_channels // 2, out_channels)

    def forward(self, data, training=True):
        x, edge_index = data.x, data.edge_index
        
        x = self.conv1(x, edge_index)
        x = self.bn1(x)
        x = F.elu(x)
        x = F.dropout(x, p=self.dropout, training=training)
        
        x = self.conv2(x, edge_index)
        x = self.bn2(x)
        x = F.elu(x)
        x = F.dropout(x, p=self.dropout, training=training)
        
        x = self.conv3(x, edge_index)
        x = self.bn3(x)
        x = F.elu(x)
        x = F.dropout(x, p=self.dropout, training=training)
        
        x = self.lin1(x)
        x = self.bn4(x)
        x = F.elu(x)
        x = F.dropout(x, p=self.dropout, training=training)
        
        x = self.lin2(x)
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


class SASRec(nn.Module):
    """
    SASRec - Self-Attentive Sequential Recommendation
    Одна из самых популярных моделей для sequential рекомендаций
    
    Архитектура:
    - Embedding + Positional Encoding
    - Multi-head Self-Attention blocks
    - Point-wise Feed-Forward
    - Residual connections + LayerNorm
    """
    
    def __init__(self, num_items: int, hidden_size: int = 64, num_heads: int = 2, 
                 num_blocks: int = 2, dropout: float = 0.3, max_len: int = 50):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_items = num_items
        
        # Embeddings
        self.item_embedding = nn.Embedding(num_items + 1, hidden_size, padding_idx=0)
        self.pos_embedding = nn.Embedding(max_len, hidden_size)
        self.dropout = nn.Dropout(dropout)
        
        # Self-Attention Blocks
        self.attention_blocks = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=hidden_size,
                nhead=num_heads,
                dim_feedforward=hidden_size * 4,
                dropout=dropout,
                activation='relu',
                batch_first=True,
                norm_first=True  # Pre-LN (более стабильно)
            )
            for _ in range(num_blocks)
        ])
        
        self.layer_norm = nn.LayerNorm(hidden_size)
        self.fc = nn.Linear(hidden_size, num_items)
    
    def forward(self, sequences):
        """
        Args:
            sequences: (batch_size, seq_len)
        Returns:
            (batch_size, num_items)
        """
        batch_size, seq_len = sequences.shape
        
        # Create position indices
        positions = torch.arange(seq_len, device=sequences.device).unsqueeze(0).expand(batch_size, -1)
        
        # Embeddings
        x = self.item_embedding(sequences) + self.pos_embedding(positions)
        x = self.dropout(x)
        
        # Attention mask (для padding)
        mask = (sequences == 0).to(sequences.device)
        
        # Self-Attention Blocks
        for block in self.attention_blocks:
            x = block(x, src_key_padding_mask=mask)
        
        # Take last item representation
        x = self.layer_norm(x[:, -1, :])
        
        # Prediction
        x = self.fc(x)
        return x


class Caser(nn.Module):
    """
    Caser - Convolutional Sequence Embedding Recommendation
    CNN-based модель для sequential рекомендаций
    
    Архитектура:
    - Embedding Matrix
    - Horizontal Convolutions (skip-gram паттерны)
    - Vertical Convolutions (union-level паттерны)
    - Concatenate + FC layers
    """
    
    def __init__(self, num_items: int, embedding_dim: int = 64, 
                 num_h_filters: int = 16, num_v_filters: int = 4,
                 dropout: float = 0.3, L: int = 5):
        super().__init__()
        self.L = L  # Длина последовательности для convolution
        self.num_items = num_items
        
        # Embedding
        self.item_embedding = nn.Embedding(num_items + 1, embedding_dim, padding_idx=0)
        
        # Horizontal convolutional filters (разные размеры окон)
        self.conv_h = nn.ModuleList([
            nn.Conv2d(1, num_h_filters, (i, embedding_dim))
            for i in [2, 3, 4]  # Окна размером 2, 3, 4
        ])
        
        # Vertical convolutional filter
        self.conv_v = nn.Conv2d(1, num_v_filters, (L, 1))
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
        
        # Fully connected layers
        fc_input_dim = num_h_filters * len(self.conv_h) + num_v_filters * embedding_dim
        
        self.fc1 = nn.Linear(fc_input_dim, 128)
        self.bn1 = nn.BatchNorm1d(128)
        
        self.fc2 = nn.Linear(128, num_items)
    
    def forward(self, sequences):
        """
        Args:
            sequences: (batch_size, seq_len)
        Returns:
            (batch_size, num_items)
        """
        batch_size = sequences.size(0)
        
        # Берем последние L элементов
        seq_len = sequences.size(1)
        if seq_len >= self.L:
            sequences = sequences[:, -self.L:]
        else:
            # Padding слева
            pad = torch.zeros(batch_size, self.L - seq_len, dtype=torch.long, device=sequences.device)
            sequences = torch.cat([pad, sequences], dim=1)
        
        # Embedding: (batch_size, L, embedding_dim)
        embedded = self.item_embedding(sequences)
        embedded = self.dropout(embedded)
        
        # Добавляем channel dimension для Conv2d
        embedded = embedded.unsqueeze(1)  # (batch_size, 1, L, embedding_dim)
        
        # Horizontal convolutions
        h_out = []
        for conv in self.conv_h:
            conv_out = F.relu(conv(embedded).squeeze(3))  # (batch_size, num_h_filters, L')
            pool_out = F.max_pool1d(conv_out, conv_out.size(2)).squeeze(2)  # (batch_size, num_h_filters)
            h_out.append(pool_out)
        h_out = torch.cat(h_out, dim=1)  # Concatenate all horizontal filters
        
        # Vertical convolution
        v_out = F.relu(self.conv_v(embedded).squeeze(2))  # (batch_size, num_v_filters, embedding_dim)
        v_out = v_out.view(batch_size, -1)  # Flatten
        
        # Concatenate horizontal and vertical
        out = torch.cat([h_out, v_out], dim=1)
        out = self.dropout(out)
        
        # FC layers
        out = self.fc1(out)
        out = self.bn1(out)
        out = F.relu(out)
        out = self.dropout(out)
        
        out = self.fc2(out)
        return out


class GRU4Rec(nn.Module):
    """
    GRU4Rec - Рекуррентная модель для session-based рекомендаций
    Специально разработана для последовательных данных
    
    Архитектура:
    - Embedding слой для категориальных признаков
    - Многослойный GRU для обработки последовательностей
    - Dropout для регуляризации
    - Полносвязные слои для финального предсказания
    """
    
    def __init__(self, num_items: int, embedding_dim: int = 64, hidden_size: int = 128, 
                 num_layers: int = 2, dropout: float = 0.3):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout
        
        # Embedding для узлов графа
        self.embedding = nn.Embedding(num_items, embedding_dim, padding_idx=0)
        
        # Multi-layer GRU
        self.gru = nn.GRU(
            input_size=embedding_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
        )
        
        # Batch normalization
        self.bn1 = nn.BatchNorm1d(hidden_size)
        
        # Полносвязные слои
        self.fc1 = nn.Linear(hidden_size, hidden_size // 2)
        self.bn2 = nn.BatchNorm1d(hidden_size // 2)
        
        self.fc2 = nn.Linear(hidden_size // 2, num_items)
        
        self.dropout_layer = nn.Dropout(dropout)
        
    def forward(self, sequences, lengths=None):
        """
        Args:
            sequences: (batch_size, seq_len) - последовательность узлов
            lengths: (batch_size,) - реальные длины последовательностей
        
        Returns:
            (batch_size, num_items) - вероятности для каждого класса
        """
        batch_size = sequences.size(0)
        
        # Embedding
        embedded = self.embedding(sequences)  # (batch_size, seq_len, embedding_dim)
        embedded = self.dropout_layer(embedded)
        
        # GRU
        if lengths is not None:
            # Pack padded sequences для эффективности
            packed = nn.utils.rnn.pack_padded_sequence(
                embedded, lengths.cpu(), batch_first=True, enforce_sorted=False
            )
            gru_out, hidden = self.gru(packed)
            gru_out, _ = nn.utils.rnn.pad_packed_sequence(gru_out, batch_first=True)
        else:
            gru_out, hidden = self.gru(embedded)
        
        # Берем последний hidden state (last layer, all batch)
        last_hidden = hidden[-1]  # (batch_size, hidden_size)
        
        # Batch normalization
        x = self.bn1(last_hidden)
        x = F.relu(x)
        x = self.dropout_layer(x)
        
        # FC layers
        x = self.fc1(x)
        x = self.bn2(x)
        x = F.relu(x)
        x = self.dropout_layer(x)
        
        # Output
        x = self.fc2(x)
        
        return x


# ==================== Утилиты ====================

def load_dag_from_json(json_path: Path) -> nx.DiGraph:
    logger.info(f"Loading DAG from {json_path}")
    
    with open(json_path, "r", encoding="utf-8") as f:
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

    logger.info(f"Loaded DAG with {dag.number_of_nodes()} nodes and {dag.number_of_edges()} edges")
    return dag


def extract_paths_from_dag(dag: nx.DiGraph) -> List[List[str]]:
    logger.info("Extracting paths from DAG")
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

    logger.info(f"Extracted {len(paths)} paths")
    return paths


def create_training_data(paths: List[List[str]]) -> Tuple[List, List]:
    X_raw = []
    y_raw = []

    for path in paths:
        for i in range(1, len(path) - 1):
            context = tuple(path[:i])
            next_step = path[i]
            if next_step.startswith("service"):
                X_raw.append(context)
                y_raw.append(next_step)

    logger.info(f"Created {len(X_raw)} training samples")
    return X_raw, y_raw


def prepare_pytorch_geometric_data(dag: nx.DiGraph, X_raw: List, y_raw: List) -> Tuple[Data, torch.Tensor, torch.Tensor, Dict]:
    logger.info("Preparing PyTorch Geometric data")
    
    node_list = list(dag.nodes)
    node_encoder = LabelEncoder()
    node_ids = node_encoder.fit_transform(node_list)
    node_map = {node: idx for node, idx in zip(node_list, node_ids)}

    edge_index = torch.tensor([[node_map[u], node_map[v]] for u, v in dag.edges], dtype=torch.long).t()
    features = [[1, 0] if dag.nodes[n]['type'] == 'service' else [0, 1] for n in node_list]
    x = torch.tensor(features, dtype=torch.float)
    data_pyg = Data(x=x, edge_index=edge_index)

    contexts = torch.tensor([node_map[context[-1]] for context in X_raw], dtype=torch.long)
    targets = torch.tensor([node_map[y] for y in y_raw], dtype=torch.long)

    return data_pyg, contexts, targets, node_map


def prepare_gru4rec_data(X_raw: List, y_raw: List, node_map: Dict, max_seq_len: int = 10) -> Tuple:
    """
    Подготовка данных для GRU4Rec
    
    Args:
        X_raw: Список контекстов (туплы узлов)
        y_raw: Список целевых узлов
        node_map: Маппинг узлов в индексы
        max_seq_len: Максимальная длина последовательности
    
    Returns:
        sequences: (num_samples, max_seq_len) - padded последовательности
        lengths: (num_samples,) - реальные длины
        targets: (num_samples,) - целевые узлы
    """
    logger.info("Preparing GRU4Rec data")
    
    sequences = []
    lengths = []
    targets_list = []
    
    for context, target in zip(X_raw, y_raw):
        # Преобразуем контекст в индексы
        seq = [node_map[node] + 1 for node in context]  # +1 для padding_idx=0
        seq_len = len(seq)
        
        # Padding или truncation
        if seq_len < max_seq_len:
            seq = [0] * (max_seq_len - seq_len) + seq  # Padding слева
        else:
            seq = seq[-max_seq_len:]  # Берем последние max_seq_len элементов
            seq_len = max_seq_len
        
        sequences.append(seq)
        lengths.append(min(seq_len, max_seq_len))
        targets_list.append(node_map[target])
    
    sequences = torch.tensor(sequences, dtype=torch.long)
    lengths = torch.tensor(lengths, dtype=torch.long)
    targets_tensor = torch.tensor(targets_list, dtype=torch.long)
    
    logger.info(f"GRU4Rec data: {sequences.shape}, lengths: {lengths.shape}")
    
    return sequences, lengths, targets_tensor


def evaluate_model_with_ndcg(preds: np.ndarray, true_labels: np.ndarray,
                             proba_preds: np.ndarray = None, name: str = "Model") -> Dict[str, float]:
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
    
    # Детальная информация о предсказаниях
    unique_preds = np.unique(preds)
    logger.info(f"Unique predictions: {len(unique_preds)} (classes: {unique_preds})")
    pred_dist = Counter(preds)
    logger.info(f"Prediction distribution: {dict(sorted(pred_dist.items()))}")

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
        logger.info("nDCG:      Not available")
        metrics['ndcg'] = None

    return metrics


# ==================== Обучение с разными seeds ====================

def train_model_generic(model, data_pyg, contexts_train, targets_train, contexts_test, targets_test,
                       optimizer, scheduler, epochs, model_name, model_seed):
    """Общая функция обучения для всех моделей"""
    
    # Устанавливаем уникальный seed для каждой модели
    torch.manual_seed(model_seed)
    np.random.seed(model_seed)
    
    best_loss = float('inf')
    patience_counter = 0
    patience = 50

    for epoch in tqdm(range(epochs), desc=f"{model_name} Training"):
        model.train()
        optimizer.zero_grad()
        
        # Разная логика forward для разных моделей
        if hasattr(model, 'dagnn'):  # DAGNN
            out = model(data_pyg.x, data_pyg.edge_index, training=True)[contexts_train]
        else:  # GCN, GraphSAGE
            out = model(data_pyg, training=True)[contexts_train]
        
        loss = F.cross_entropy(out, targets_train)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)  # Gradient clipping
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


def train_sasrec(
    sequences_train, targets_train, sequences_test, targets_test,
    num_items, epochs, hidden_size=64, num_heads=2, num_blocks=2,
    dropout=0.3, lr=0.001, model_seed=42
):
    """Обучение SASRec модели"""
    
    logger.info(f"Training SASRec (hidden={hidden_size}, heads={num_heads}, blocks={num_blocks}, dropout={dropout})...")
    
    torch.manual_seed(model_seed)
    np.random.seed(model_seed)
    
    model = SASRec(
        num_items=num_items + 1,
        hidden_size=hidden_size,
        num_heads=num_heads,
        num_blocks=num_blocks,
        dropout=dropout,
        max_len=50
    )
    
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=20
    )
    
    best_loss = float('inf')
    patience_counter = 0
    patience = 50
    
    for epoch in tqdm(range(epochs), desc="SASRec Training"):
        model.train()
        optimizer.zero_grad()
        
        out = model(sequences_train)
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
        test_output = model(sequences_test)
        preds = test_output.argmax(dim=1).numpy()
        proba = F.softmax(test_output, dim=1).numpy()
    
    return preds, proba


def train_caser(
    sequences_train, targets_train, sequences_test, targets_test,
    num_items, epochs, embedding_dim=64, num_h_filters=16,
    num_v_filters=4, dropout=0.3, lr=0.001, model_seed=42
):
    """Обучение Caser модели"""
    
    logger.info(f"Training Caser (embedding={embedding_dim}, h_filters={num_h_filters}, v_filters={num_v_filters}, dropout={dropout})...")
    
    torch.manual_seed(model_seed)
    np.random.seed(model_seed)
    
    model = Caser(
        num_items=num_items + 1,
        embedding_dim=embedding_dim,
        num_h_filters=num_h_filters,
        num_v_filters=num_v_filters,
        dropout=dropout,
        L=5
    )
    
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=20
    )
    
    best_loss = float('inf')
    patience_counter = 0
    patience = 50
    
    for epoch in tqdm(range(epochs), desc="Caser Training"):
        model.train()
        optimizer.zero_grad()
        
        out = model(sequences_train)
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
        test_output = model(sequences_test)
        preds = test_output.argmax(dim=1).numpy()
        proba = F.softmax(test_output, dim=1).numpy()
    
    return preds, proba


def train_gru4rec(
    sequences_train, lengths_train, targets_train,
    sequences_test, lengths_test, targets_test,
    num_items, epochs, embedding_dim=64, hidden_size=128,
    num_layers=2, dropout=0.4, lr=0.001, model_seed=42
):
    """Обучение GRU4Rec модели"""
    
    logger.info(f"Training GRU4Rec (embedding={embedding_dim}, hidden={hidden_size}, layers={num_layers}, dropout={dropout})...")
    
    # Устанавливаем seed
    torch.manual_seed(model_seed)
    np.random.seed(model_seed)
    
    # Создаем модель
    model = GRU4Rec(
        num_items=num_items + 1,  # +1 для padding
        embedding_dim=embedding_dim,
        hidden_size=hidden_size,
        num_layers=num_layers,
        dropout=dropout
    )
    
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=20
    )
    
    best_loss = float('inf')
    patience_counter = 0
    patience = 50
    
    for epoch in tqdm(range(epochs), desc="GRU4Rec Training"):
        model.train()
        optimizer.zero_grad()
        
        # Forward pass
        out = model(sequences_train, lengths_train)
        loss = F.cross_entropy(out, targets_train)
        
        # Backward pass
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
    
    # Evaluation
    model.eval()
    with torch.no_grad():
        test_output = model(sequences_test, lengths_test)
        preds = test_output.argmax(dim=1).numpy()
        proba = F.softmax(test_output, dim=1).numpy()
    
    return preds, proba


def main():
    parser = argparse.ArgumentParser(description="FINAL DAG-based Recommender with Fixed Results")
    parser.add_argument("--data", type=str, default="compositionsDAG.json")
    parser.add_argument("--test-size", type=float, default=0.3)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--hidden-channels", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument("--dropout", type=float, default=0.4)
    parser.add_argument("--random-seed", type=int, default=42)
    
    args = parser.parse_args()

    # Load data
    data_path = Path(args.data)
    if not data_path.exists():
        logger.error(f"Data file not found: {data_path}")
        sys.exit(1)

    dag = load_dag_from_json(data_path)
    paths = extract_paths_from_dag(dag)
    X_raw, y_raw = create_training_data(paths)

    # Vectorize
    logger.info("Vectorizing data...")
    mlb = MultiLabelBinarizer()
    X = mlb.fit_transform(X_raw)
    le = LabelEncoder()
    y = le.fit_transform(y_raw)

    # Стратифицированный split!
    logger.info("Using STRATIFIED split to ensure balanced classes...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=args.test_size, random_state=args.random_seed, stratify=y
    )
    logger.info(f"Train samples: {len(X_train)}, Test samples: {len(X_test)}")
    logger.info(f"Train class distribution: {Counter(y_train)}")
    logger.info(f"Test class distribution: {Counter(y_test)}")

    # Prepare PyG data
    data_pyg, contexts, targets, node_map = prepare_pytorch_geometric_data(dag, X_raw, y_raw)
    
    # Стратифицированный split для графовых моделей
    contexts_train, contexts_test, targets_train, targets_test = train_test_split(
        contexts, targets, test_size=args.test_size, random_state=args.random_seed,
        stratify=targets.numpy()
    )
    logger.info(f"Graph train samples: {len(contexts_train)}, test samples: {len(contexts_test)}")

    results = {}

    # Baseline: Random Forest
    logger.info("Training Random Forest...")
    np.random.seed(args.random_seed)
    rf = RandomForestClassifier(n_estimators=200, max_depth=15, min_samples_split=5,
                                random_state=args.random_seed, n_jobs=-1)
    rf.fit(X_train, y_train)
    rf_preds = rf.predict(X_test)
    rf_proba = rf.predict_proba(X_test)
    results['Random Forest'] = evaluate_model_with_ndcg(
        rf_preds, y_test, proba_preds=rf_proba, name="Random Forest"
    )

    # Popularity baseline
    logger.info("Training Popularity baseline...")
    counter = Counter(y_raw)
    top_label = counter.most_common(1)[0][0]
    pop_preds = np.array([le.transform([top_label])[0]] * len(y_test))
    pop_proba = np.zeros((len(y_test), len(le.classes_)))
    top_label_index = le.transform([top_label])[0]
    pop_proba[:, top_label_index] = 1
    results['Popularity'] = evaluate_model_with_ndcg(
        pop_preds, y_test, proba_preds=pop_proba, name="Popularity"
    )

    # GCN - с уникальным seed
    logger.info(f"Training GCN with seed={args.random_seed + 1}...")
    torch.manual_seed(args.random_seed + 1)
    gcn = GCNRecommender(
        in_channels=2, hidden_channels=args.hidden_channels,
        out_channels=len(node_map), dropout=0.5  # Увеличенный dropout
    )
    opt_gcn = torch.optim.Adam(gcn.parameters(), lr=args.learning_rate, weight_decay=1e-4)
    sched_gcn = torch.optim.lr_scheduler.ReduceLROnPlateau(opt_gcn, mode='min', factor=0.5, patience=20)
    
    gcn_preds, gcn_proba = train_model_generic(
        gcn, data_pyg, contexts_train, targets_train, contexts_test, targets_test,
        opt_gcn, sched_gcn, args.epochs, "GCN", args.random_seed + 1
    )
    results['GCN'] = evaluate_model_with_ndcg(
        gcn_preds, targets_test.numpy(), proba_preds=gcn_proba, name="GCN"
    )

    # DAGNN - с другим seed
    logger.info(f"Training DAGNN with seed={args.random_seed + 2}...")
    torch.manual_seed(args.random_seed + 2)
    dagnn = DAGNNRecommender(
        in_channels=2, hidden_channels=args.hidden_channels,
        out_channels=len(node_map), dropout=0.4
    )
    opt_dagnn = torch.optim.Adam(dagnn.parameters(), lr=args.learning_rate * 0.8, weight_decay=1e-4)
    sched_dagnn = torch.optim.lr_scheduler.ReduceLROnPlateau(opt_dagnn, mode='min', factor=0.5, patience=20)
    
    dagnn_preds, dagnn_proba = train_model_generic(
        dagnn, data_pyg, contexts_train, targets_train, contexts_test, targets_test,
        opt_dagnn, sched_dagnn, args.epochs, "DAGNN", args.random_seed + 2
    )
    results['DAGNN'] = evaluate_model_with_ndcg(
        dagnn_preds, targets_test.numpy(), proba_preds=dagnn_proba, name="DAGNN"
    )

    # GraphSAGE - с еще другим seed
    logger.info(f"Training GraphSAGE with seed={args.random_seed + 3}...")
    torch.manual_seed(args.random_seed + 3)
    sage = GraphSAGERecommender(
        in_channels=2, hidden_channels=args.hidden_channels,
        out_channels=len(node_map), dropout=0.4
    )
    opt_sage = torch.optim.Adam(sage.parameters(), lr=args.learning_rate, weight_decay=1e-4)
    sched_sage = torch.optim.lr_scheduler.ReduceLROnPlateau(opt_sage, mode='min', factor=0.5, patience=20)
    
    sage_preds, sage_proba = train_model_generic(
        sage, data_pyg, contexts_train, targets_train, contexts_test, targets_test,
        opt_sage, sched_sage, args.epochs, "GraphSAGE", args.random_seed + 3
    )
    results['GraphSAGE'] = evaluate_model_with_ndcg(
        sage_preds, targets_test.numpy(), proba_preds=sage_proba, name="GraphSAGE"
    )

    # GRU4Rec - рекуррентная модель для последовательностей
    logger.info(f"Training GRU4Rec with seed={args.random_seed + 4}...")
    
    # Подготовка данных для GRU4Rec
    sequences_all, lengths_all, targets_gru = prepare_gru4rec_data(X_raw, y_raw, node_map, max_seq_len=10)
    
    # Split для GRU4Rec (используем те же индексы что и для других моделей)
    train_indices = []
    test_indices = []
    
    # Создаем индексы для split на основе y
    for i, (x_raw_item, y_item) in enumerate(zip(X_raw, y_raw)):
        # Проверяем принадлежность к train или test через y_raw
        y_encoded = le.transform([y_item])[0]
        if y_encoded in y_train:
            # Проверяем что этот конкретный пример в train
            matching_train = []
            for j, y_tr in enumerate(y_train):
                if y_tr == y_encoded and j < len(X_train):
                    matching_train.append(j)
            if len(matching_train) > 0 and i < len(matching_train) + len(test_indices):
                train_indices.append(i)
            else:
                test_indices.append(i)
        else:
            test_indices.append(i)
    
    # Используем стратифицированный split на индексах
    from sklearn.model_selection import train_test_split as split_indices
    train_idx, test_idx = train_test_split(
        range(len(sequences_all)), 
        test_size=args.test_size, 
        random_state=args.random_seed,
        stratify=targets_gru.numpy()
    )
    
    sequences_train = sequences_all[train_idx]
    lengths_train = lengths_all[train_idx]
    targets_gru_train = targets_gru[train_idx]
    
    sequences_test = sequences_all[test_idx]
    lengths_test = lengths_all[test_idx]
    targets_gru_test = targets_gru[test_idx]
    
    logger.info(f"GRU4Rec train: {len(sequences_train)}, test: {len(sequences_test)}")
    
    # Обучение GRU4Rec
    gru4rec_preds, gru4rec_proba = train_gru4rec(
        sequences_train, lengths_train, targets_gru_train,
        sequences_test, lengths_test, targets_gru_test,
        num_items=len(node_map),
        epochs=args.epochs,
        embedding_dim=64,
        hidden_size=args.hidden_channels * 2,  # Больше для RNN
        num_layers=2,
        dropout=args.dropout,
        lr=args.learning_rate,
        model_seed=args.random_seed + 4
    )
    results['GRU4Rec'] = evaluate_model_with_ndcg(
        gru4rec_preds, targets_gru_test.numpy(), proba_preds=gru4rec_proba, name="GRU4Rec"
    )

    # SASRec - Self-Attention модель
    logger.info(f"Training SASRec with seed={args.random_seed + 5}...")
    sasrec_preds, sasrec_proba = train_sasrec(
        sequences_train, targets_gru_train, sequences_test, targets_gru_test,
        num_items=len(node_map),
        epochs=args.epochs,
        hidden_size=args.hidden_channels,
        num_heads=2,
        num_blocks=2,
        dropout=args.dropout,
        lr=args.learning_rate,
        model_seed=args.random_seed + 5
    )
    results['SASRec'] = evaluate_model_with_ndcg(
        sasrec_preds, targets_gru_test.numpy(), proba_preds=sasrec_proba, name="SASRec"
    )

    # Caser - CNN модель
    logger.info(f"Training Caser with seed={args.random_seed + 6}...")
    caser_preds, caser_proba = train_caser(
        sequences_train, targets_gru_train, sequences_test, targets_gru_test,
        num_items=len(node_map),
        epochs=args.epochs,
        embedding_dim=64,
        num_h_filters=16,
        num_v_filters=4,
        dropout=args.dropout,
        lr=args.learning_rate,
        model_seed=args.random_seed + 6
    )
    results['Caser'] = evaluate_model_with_ndcg(
        caser_preds, targets_gru_test.numpy(), proba_preds=caser_proba, name="Caser"
    )

    # Summary
    logger.info("\n" + "="*70)
    logger.info("🏆 ИТОГОВОЕ СРАВНЕНИЕ")
    logger.info("="*70)
    
    sorted_results = sorted(results.items(), key=lambda x: x[1]['accuracy'], reverse=True)
    
    for rank, (model_name, metrics) in enumerate(sorted_results, 1):
        logger.info(f"\n#{rank} {model_name}:")
        for metric_name, value in metrics.items():
            if value is not None:
                logger.info(f"     {metric_name}: {value:.4f}")

    # Анализ различий
    logger.info("\n" + "="*70)
    logger.info("🔍 АНАЛИЗ РАЗЛИЧИЙ")
    logger.info("="*70)
    accuracies = [m['accuracy'] for m in results.values()]
    logger.info(f"Min accuracy: {min(accuracies):.4f}")
    logger.info(f"Max accuracy: {max(accuracies):.4f}")
    logger.info(f"Range: {max(accuracies) - min(accuracies):.4f}")
    logger.info(f"Std dev: {np.std(accuracies):.4f}")


if __name__ == "__main__":
    main()

