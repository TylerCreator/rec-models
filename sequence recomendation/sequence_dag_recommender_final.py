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

КРОСС-ВАЛИДАЦИЯ:
Файл поддерживает два режима работы:

1. СТАНДАРТНЫЙ РЕЖИМ (train/test split):
   python sequence_dag_recommender_final.py --data compositionsDAG.json

2. РЕЖИМ КРОСС-ВАЛИДАЦИИ (k-fold CV):
   python sequence_dag_recommender_final.py --data compositionsDAG.json --use-cv --cv-folds 5
   
   Параметры кросс-валидации:
   --use-cv         : Включить режим кросс-валидации
   --cv-folds N     : Количество фолдов (по умолчанию: 5)
   
   В режиме кросс-валидации все модели тренируются на каждом фолде,
   и в конце выводятся агрегированные метрики (среднее ± стандартное отклонение).

Модели для сравнения:
- Popularity (baseline)
- Markov Chain (n-th order, n = max path length in training data)
- GCN
- DAGNN
- DAGNN-Improved (с Focal Loss + Label Smoothing)
- GraphSAGE
- GRU4Rec
- SASRec
- Caser
- DAGNNSequential (DAGNN + GRU)
- DAGNNAttention (DAGNN + Attention)
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
from torch_geometric.nn import (
    APPNP,
    GATConv,
    GCNConv,
    SAGEConv,
    GATv2Conv,
    BatchNorm,
    LayerNorm,
)
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
            if training and self.dropout > 0:
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
                 num_blocks: int = 2, dropout: float = 0.3, max_len: int = 50, num_classes: int = None):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_items = num_items
        
        # Если num_classes не задан, используем num_items
        if num_classes is None:
            num_classes = num_items
        
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
        # Выходной слой предсказывает только целевые классы (сервисы)
        self.fc = nn.Linear(hidden_size, num_classes)
    
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
                 dropout: float = 0.3, L: int = 5, num_classes: int = None):
        super().__init__()
        self.L = L  # Длина последовательности для convolution
        self.num_items = num_items
        
        # Если num_classes не задан, используем num_items
        if num_classes is None:
            num_classes = num_items
        
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
        
        # Выходной слой предсказывает только целевые классы (сервисы)
        self.fc2 = nn.Linear(128, num_classes)
    
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


class DAGNNSequential(nn.Module):
    """
    DAGNN + Sequential - Гибридная модель
    Объединяет графовые связи (DAGNN) с моделированием последовательности (GRU)
    
    Архитектура:
    1. DAGNN обрабатывает весь граф → node embeddings
    2. Извлекаем embeddings для всех узлов в последовательности
    3. GRU обрабатывает последовательность embeddings
    4. Финальное предсказание на основе последнего hidden state
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
        """
        Args:
            x: node features [num_nodes, in_channels]
            edge_index: graph edges [2, num_edges]
            sequences: node indices in sequences [batch_size, seq_len]
            lengths: real sequence lengths [batch_size]
            training: bool
        Returns:
            predictions [batch_size, out_channels]
        """
        # 1. DAGNN: получаем embeddings для всех узлов
        x = self.lin1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=training)
        
        node_embeddings = self.dagnn(x, edge_index, training=training)  # [num_nodes, hidden_channels]
        
        # 2. Извлекаем embeddings для узлов в последовательностях
        batch_size, seq_len = sequences.shape
        
        # Создаем padding embedding (нулевой вектор)
        padding_emb = torch.zeros(1, self.hidden_channels, device=node_embeddings.device)
        node_embeddings_with_padding = torch.cat([padding_emb, node_embeddings], dim=0)
        # Теперь индекс 0 = padding, индексы 1..N = реальные узлы
        
        # Flatten sequences для индексирования
        flat_sequences = sequences.view(-1)  # [batch_size * seq_len]
        
        # Получаем embeddings
        seq_embeddings = node_embeddings_with_padding[flat_sequences]  # [batch_size * seq_len, hidden_channels]
        seq_embeddings = seq_embeddings.view(batch_size, seq_len, self.hidden_channels)
        
        # 3. GRU обрабатывает последовательность
        # Pack padded sequence для эффективности
        packed = nn.utils.rnn.pack_padded_sequence(
            seq_embeddings, lengths.cpu(), batch_first=True, enforce_sorted=False
        )
        gru_out, hidden = self.gru(packed)
        
        # Берем последний hidden state
        last_hidden = hidden[-1]  # [batch_size, hidden_channels]
        
        # 4. Финальное предсказание
        out = self.bn2(last_hidden)
        out = F.relu(out)
        out = F.dropout(out, p=self.dropout, training=training)
        out = self.lin_out(out)
        
        return out


class DAGNNAttention(nn.Module):
    """
    DAGNN + Attention - Гибридная модель
    Использует attention mechanism для взвешивания узлов в последовательности
    
    Архитектура:
    1. DAGNN обрабатывает граф → node embeddings
    2. Multi-head attention по последовательности узлов
    3. Финальное предсказание
    """
    
    def __init__(self, in_channels: int, hidden_channels: int, out_channels: int,
                 K: int = 10, num_heads: int = 4, dropout: float = 0.4):
        super().__init__()
        self.dropout = dropout
        self.hidden_channels = hidden_channels
        
        # DAGNN для обработки графа
        self.lin1 = nn.Linear(in_channels, hidden_channels)
        self.bn1 = nn.BatchNorm1d(hidden_channels)
        
        self.dagnn = DAGNN(hidden_channels, K, dropout=dropout)
        
        # Multi-head attention
        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_channels,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        
        self.layer_norm = nn.LayerNorm(hidden_channels)
        
        # Output layers
        self.lin2 = nn.Linear(hidden_channels, hidden_channels // 2)
        self.bn2 = nn.BatchNorm1d(hidden_channels // 2)
        self.lin_out = nn.Linear(hidden_channels // 2, out_channels)
    
    def forward(self, x, edge_index, sequences, mask=None, training=True):
        """
        Args:
            x: node features [num_nodes, in_channels]
            edge_index: graph edges [2, num_edges]
            sequences: node indices in sequences [batch_size, seq_len]
            mask: attention mask [batch_size, seq_len] (optional)
            training: bool
        Returns:
            predictions [batch_size, out_channels]
        """
        # 1. DAGNN: получаем embeddings для всех узлов
        x = self.lin1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=training)
        
        node_embeddings = self.dagnn(x, edge_index, training=training)
        
        # 2. Извлекаем embeddings для узлов в последовательностях
        batch_size, seq_len = sequences.shape
        
        # Создаем padding embedding (нулевой вектор)
        padding_emb = torch.zeros(1, self.hidden_channels, device=node_embeddings.device)
        node_embeddings_with_padding = torch.cat([padding_emb, node_embeddings], dim=0)
        
        flat_sequences = sequences.view(-1)
        seq_embeddings = node_embeddings_with_padding[flat_sequences]
        seq_embeddings = seq_embeddings.view(batch_size, seq_len, self.hidden_channels)
        
        # 3. Self-attention по последовательности
        attn_out, attn_weights = self.attention(
            seq_embeddings, seq_embeddings, seq_embeddings,
            key_padding_mask=mask
        )
        
        # Residual connection + LayerNorm
        seq_embeddings = self.layer_norm(seq_embeddings + attn_out)
        
        # 4. Aggregation: берем последний элемент последовательности
        # Или можно использовать mean/max pooling
        last_output = seq_embeddings[:, -1, :]  # [batch_size, hidden_channels]
        
        # 5. Финальное предсказание
        out = self.lin2(last_output)
        out = self.bn2(out)
        out = F.relu(out)
        out = F.dropout(out, p=self.dropout, training=training)
        out = self.lin_out(out)
        
        return out


class MarkovChain:
    """
    Марковская цепь n-го порядка для последовательных рекомендаций
    
    Модель учитывает весь доступный контекст (последовательность):
    - Порядок определяется автоматически из максимальной длины последовательности в обучающих данных
    - Изучает вероятности переходов на основе всей последовательности
    - Предсказывает следующий элемент на основе полного контекста
    - Использует сглаживание Лапласа для обработки неизвестных переходов
    - Fallback к меньшим порядкам если точная последовательность не встречалась
    
    Преимущества марковской модели n-го порядка:
    - Интерпретируема
    - Быстро обучается
    - Учитывает весь доступный контекст из графа
    - Естественно подходит для последовательных данных
    """
    
    def __init__(self, num_states: int, smoothing: float = 1.0, max_order: int = None):
        """
        Args:
            num_states: Количество состояний (узлов в графе)
            smoothing: Параметр сглаживания Лапласа (по умолчанию 1.0)
            max_order: Максимальный порядок цепи (если None, определяется автоматически)
        """
        self.num_states = num_states
        self.smoothing = smoothing
        self.max_order = max_order
        self.order = None  # Будет установлен при обучении
        # Матрица переходов: transition_counts[context_tuple][target] = count
        self.transition_counts = {}
        self.context_totals = {}
        self.num_classes = None  # Будет установлено при обучении
        self.valid_targets = set()  # Множество валидных целевых классов (сервисов)
        
    def fit(self, sequences: torch.Tensor, targets: torch.Tensor, node_map: Dict, service_map: Dict):
        """
        Обучение на последовательностях
        
        Args:
            sequences: (batch_size, seq_len) - последовательности узлов (индексы в node_map)
            targets: (batch_size,) - целевые сервисы (индексы в service_map)
            node_map: Маппинг всех узлов
            service_map: Маппинг только сервисов (целевых классов)
        """
        logger.info(f"Training Markov Chain (n-th order) on {len(sequences)} sequences...")
        
        self.num_classes = len(service_map)
        self.valid_targets = set(service_map.values())
        
        # Определяем порядок цепи из максимальной длины последовательности
        if self.max_order is None:
            max_seq_length = 0
            for seq in sequences:
                seq_filtered = [s.item() for s in seq if s.item() != 0]
                max_seq_length = max(max_seq_length, len(seq_filtered))
            self.order = max_seq_length
        else:
            self.order = self.max_order
        
        logger.info(f"Markov Chain order: {self.order} (using full context)")
        
        # Подсчитываем переходы на основе полного контекста
        for seq, target in zip(sequences, targets):
            # Берем только не-padding элементы
            seq_filtered = tuple([s.item() for s in seq if s.item() != 0])
            
            if len(seq_filtered) == 0:
                continue
            
            # Контекст = вся последовательность (n-й порядок)
            context = seq_filtered
            
            # Целевой сервис
            target_state = target.item()
            
            # Обновляем счетчики переходов для полного контекста
            if context not in self.transition_counts:
                self.transition_counts[context] = {}
                self.context_totals[context] = 0
            
            if target_state not in self.transition_counts[context]:
                self.transition_counts[context][target_state] = 0
            
            self.transition_counts[context][target_state] += 1
            self.context_totals[context] += 1
        
        logger.info(f"Learned transitions from {len(self.transition_counts)} unique contexts")
        logger.info(f"Total transitions: {sum(self.context_totals.values())}")
        logger.info(f"Average context length: {np.mean([len(c) for c in self.transition_counts.keys()]):.2f}")
        
    def predict(self, sequences: torch.Tensor) -> Tuple[np.ndarray, np.ndarray]:
        """
        Предсказание для тестовых последовательностей с backoff стратегией
        
        Стратегия:
        1. Пытаемся найти точное совпадение полного контекста
        2. Если не нашли, используем backoff к меньшим порядкам (удаляем первый элемент)
        3. Если ничего не нашли, используем равномерное распределение
        
        Args:
            sequences: (batch_size, seq_len)
            
        Returns:
            predictions: (batch_size,) - предсказанные классы
            probabilities: (batch_size, num_classes) - вероятности для каждого класса
        """
        predictions = []
        probabilities = []
        
        for seq in sequences:
            # Берем все не-padding элементы
            seq_filtered = tuple([s.item() for s in seq if s.item() != 0])
            
            if len(seq_filtered) == 0:
                # Если последовательность пустая, предсказываем равномерное распределение
                probs = np.ones(self.num_classes) / self.num_classes
            else:
                # Пытаемся найти контекст, используя backoff стратегию
                context = seq_filtered
                found = False
                probs = np.zeros(self.num_classes)
                
                # Пробуем контексты от полного до минимального (последний элемент)
                while len(context) > 0:
                    if context in self.transition_counts:
                        # Нашли контекст! Вычисляем вероятности
                        total_count = self.context_totals[context]
                        
                        for target_idx in range(self.num_classes):
                            count = self.transition_counts[context].get(target_idx, 0)
                            # Сглаживание Лапласа
                            probs[target_idx] = (count + self.smoothing) / (total_count + self.smoothing * self.num_classes)
                        
                        found = True
                        break
                    
                    # Backoff: удаляем первый элемент контекста
                    context = context[1:]
                
                if not found:
                    # Если ничего не нашли, используем равномерное распределение
                    probs = np.ones(self.num_classes) / self.num_classes
                
                # Нормализуем (на случай ошибок округления)
                probs = probs / probs.sum()
            
            probabilities.append(probs)
            predictions.append(np.argmax(probs))
        
        return np.array(predictions), np.array(probabilities)
    
    def predict_proba(self, sequences: torch.Tensor) -> np.ndarray:
        """
        Получить только вероятности
        """
        _, probs = self.predict(sequences)
        return probs


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
                 num_layers: int = 2, dropout: float = 0.3, num_classes: int = None):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout
        
        # Если num_classes не задан, используем num_items (backward compatibility)
        if num_classes is None:
            num_classes = num_items
        
        # Embedding для узлов графа (все узлы, включая padding)
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
        
        # Выходной слой предсказывает только целевые классы (сервисы)
        self.fc2 = nn.Linear(hidden_size // 2, num_classes)
        
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

def load_dag_from_json(json_path: Path) -> Tuple[nx.DiGraph, List]:
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
    return dag, data


def extract_paths_from_compositions(data: List) -> List[List[str]]:
    """
    Извлекает РЕАЛЬНЫЕ пути из композиций (не искусственные через DFS)
    
    Args:
        data: исходные данные из compositionsDAG.json
    
    Returns:
        List of unique real paths from compositions
    """
    logger.info("Extracting REAL paths from compositions (not synthetic DFS paths)")
    
    all_paths = []
    
    for comp_idx, composition in enumerate(data):
        # Строим граф для этой конкретной композиции
        comp_graph = nx.DiGraph()
        id_to_mid = {}
        
        # Создаем маппинг для этой композиции
        for node in composition["nodes"]:
            node_id = str(node["id"])
            if "mid" in node:
                node_name = f"service_{node['mid']}"
            else:
                node_name = f"table_{node['id']}"
            id_to_mid[node_id] = node_name
        
        # Добавляем ребра из этой композиции
        for link in composition["links"]:
            source = str(link["source"])
            target = str(link["target"])
            if source in id_to_mid and target in id_to_mid:
                comp_graph.add_edge(id_to_mid[source], id_to_mid[target])
        
        # Извлекаем все простые пути в этой композиции
        start_nodes = [n for n in comp_graph.nodes() if comp_graph.in_degree(n) == 0]
        end_nodes = [n for n in comp_graph.nodes() if comp_graph.out_degree(n) == 0]
        
        # Извлекаем все пути от начальных к конечным узлам
        for start in start_nodes:
            for end in end_nodes:
                try:
                    for path in nx.all_simple_paths(comp_graph, start, end):
                        if len(path) > 1:
                            all_paths.append(path)
                except nx.NetworkXNoPath:
                    continue
    
    logger.info(f"Extracted {len(all_paths)} REAL paths from {len(data)} compositions")
    
    # Удаляем дубликаты
    unique_paths = []
    seen = set()
    for path in all_paths:
        path_tuple = tuple(path)
        if path_tuple not in seen:
            seen.add(path_tuple)
            unique_paths.append(path)
    
    logger.info(f"Unique paths: {len(unique_paths)}")
    
    return unique_paths


def build_graph_from_real_paths(paths: List[List[str]]) -> nx.DiGraph:
    """
    Строит граф на основе ТОЛЬКО реальных путей из композиций
    """
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


def extract_honest_graph_features(graph: nx.DiGraph, X_raw: List) -> np.ndarray:
    """
    Извлекает ЧЕСТНЫЕ графовые признаки БЕЗ использования target
    
    Признаки основаны ТОЛЬКО на контексте:
    - Характеристики последнего узла в контексте
    - Характеристики возможных следующих узлов (соседей)
    - Статистики по всему контексту
    
    НЕ используем: target node, расстояние до target, ребра к target
    """
    logger.info("Extracting HONEST graph features (without target leakage)...")
    
    # Вычисляем метрики графа один раз
    pagerank = nx.pagerank(graph)
    in_degree = dict(graph.in_degree())
    out_degree = dict(graph.out_degree())
    
    try:
        betweenness = nx.betweenness_centrality(graph)
    except:
        betweenness = {node: 0.0 for node in graph.nodes()}
    
    try:
        clustering = nx.clustering(graph.to_undirected())
    except:
        clustering = {node: 0.0 for node in graph.nodes()}
    
    # Нормализация
    max_in = max(in_degree.values()) if in_degree else 1
    max_out = max(out_degree.values()) if out_degree else 1
    
    graph_features = []
    
    for context in X_raw:
        features = []
        last_node = context[-1]
        
        # === Признаки ПОСЛЕДНЕГО узла (откуда переходим) ===
        features.append(in_degree.get(last_node, 0) / max_in)
        features.append(out_degree.get(last_node, 0) / max_out)
        features.append(pagerank.get(last_node, 0))
        features.append(betweenness.get(last_node, 0))
        features.append(clustering.get(last_node, 0))
        features.append(1 if last_node.startswith('service') else 0)
        
        # === Признаки КОНТЕКСТА ===
        features.append(len(context))  # Длина контекста
        
        # Средние значения по всему контексту
        context_pagerank = [pagerank.get(node, 0) for node in context]
        context_out_deg = [out_degree.get(node, 0) / max_out for node in context]
        
        features.append(np.mean(context_pagerank) if context_pagerank else 0)
        features.append(np.max(context_pagerank) if context_pagerank else 0)
        features.append(np.mean(context_out_deg) if context_out_deg else 0)
        
        # === Признаки ВОЗМОЖНЫХ следующих узлов (соседи последнего узла) ===
        successors = list(graph.successors(last_node))
        features.append(len(successors))  # Сколько вариантов для следующего шага
        
        if successors:
            # Характеристики соседей (агрегированные)
            succ_pagerank = [pagerank.get(s, 0) for s in successors]
            succ_in_deg = [in_degree.get(s, 0) / max_in for s in successors]
            
            features.append(np.mean(succ_pagerank))
            features.append(np.max(succ_pagerank))
            features.append(np.mean(succ_in_deg))
            features.append(np.max(succ_in_deg))
            
            # Сколько соседей являются services
            service_count = sum(1 for s in successors if s.startswith('service'))
            features.append(service_count / len(successors) if successors else 0)
        else:
            # Нет соседей - нулевые признаки
            features.extend([0, 0, 0, 0, 0])
        
        graph_features.append(features)
    
    graph_features_array = np.array(graph_features, dtype=np.float32)
    logger.info(f"Extracted honest graph features: shape {graph_features_array.shape}")
    
    return graph_features_array


def create_training_data(paths: List[List[str]]) -> Tuple[List, List]:
    """
    Создает обучающие примеры из путей
    
    Логика:
    - Для каждого пути используем ВСЕ переходы
    - Исключаем только переходы на таблицы (они стартовые, на них нельзя переходить)
    - Используем пути любой длины (включая длину 2)
    """
    X_raw = []
    y_raw = []

    for path in paths:
        # Используем все переходы от позиции 1 до конца (включая последний)
        for i in range(1, len(path)):
            context = tuple(path[:i])
            next_step = path[i]
            
            # Исключаем только переходы на таблицы (они стартовые)
            if next_step.startswith("service"):
                X_raw.append(context)
                y_raw.append(next_step)

    logger.info(f"Created {len(X_raw)} training samples from {len(paths)} paths")
    return X_raw, y_raw


def contexts_to_features(contexts_tensor, node_map, mlb):
    """
    Преобразует тензор контекстов (node indices) в feature vectors для RandomForest
    
    Args:
        contexts_tensor: torch.Tensor с индексами узлов
        node_map: dict отображение node_name -> node_idx
        mlb: MultiLabelBinarizer для кодирования
    
    Returns:
        np.ndarray: матрица признаков
    """
    # Обратное отображение: idx -> node_name
    idx_to_node = {idx: name for name, idx in node_map.items()}
    
    # Преобразуем contexts_tensor в списки имен узлов
    contexts_as_names = []
    for context_tensor in contexts_tensor:
        # Берем только не-padding элементы (предполагаем что padding = 0 или -1)
        context_names = []
        # Если context_tensor это скаляр (int), преобразуем в список
        if isinstance(context_tensor, (int, torch.Tensor)):
            if isinstance(context_tensor, torch.Tensor):
                idx_val = context_tensor.item()
            else:
                idx_val = context_tensor
            if idx_val > 0 and idx_val in idx_to_node:
                context_names.append(idx_to_node[idx_val])
        else:
            for idx in context_tensor.tolist():
                if idx > 0 and idx in idx_to_node:  # Пропускаем padding
                    context_names.append(idx_to_node[idx])
        contexts_as_names.append(tuple(context_names) if context_names else ('',))
    
    # Используем MultiLabelBinarizer для преобразования в features
    X_features = mlb.transform(contexts_as_names)
    return X_features


def prepare_pytorch_geometric_data(dag: nx.DiGraph, X_raw: List, y_raw: List, paths: List[List[str]]) -> Tuple[Data, torch.Tensor, torch.Tensor, Dict, Dict]:
    """
    Подготовка данных для PyTorch Geometric
    Граф строится из РЕАЛЬНЫХ путей
    
    ОПТИМАЛЬНАЯ КОНФИГУРАЦИЯ для малого датасета:
    - Направленный граф (сохраняет DAG семантику)
    - БЕЗ self-loops (меньше переобучение)
    - БЕЗ симметризации (сохраняет причинно-следственные связи)
    
    Returns:
        data_pyg: Graph data (направленный граф, 97 ребер)
        contexts: Context node indices (in node_map)
        targets: Target service indices (in service_map)
        node_map: Mapping all nodes -> indices
        service_map: Mapping only services -> indices (for predictions)
    """
    logger.info("Preparing PyTorch Geometric data from real paths...")
    
    # Строим граф на основе реальных путей
    path_graph = build_graph_from_real_paths(paths)
    
    node_list = list(path_graph.nodes)
    node_encoder = LabelEncoder()
    node_ids = node_encoder.fit_transform(node_list)
    node_map = {node: idx for node, idx in zip(node_list, node_ids)}

    # Направленный граф БЕЗ модификаций (оптимально для малого датасета)
    edge_index = torch.tensor([[node_map[u], node_map[v]] for u, v in path_graph.edges], dtype=torch.long).t()
    features = [[1, 0] if path_graph.nodes[n]['type'] == 'service' else [0, 1] for n in node_list]
    x = torch.tensor(features, dtype=torch.float)
    data_pyg = Data(x=x, edge_index=edge_index)
    
    logger.info(f"Graph: {edge_index.size(1)} directed edges (no self-loops, no symmetrization)")

    contexts = torch.tensor([node_map[context[-1]] for context in X_raw], dtype=torch.long)
    
    # Создаем отдельный маппинг ТОЛЬКО для сервисов (целевых классов)
    unique_services = sorted(set(y_raw))
    service_map = {service: idx for idx, service in enumerate(unique_services)}
    targets = torch.tensor([service_map[y] for y in y_raw], dtype=torch.long)
    
    logger.info(f"Created service_map: {len(service_map)} unique target services")
    logger.info(f"Services: {unique_services}")

    return data_pyg, contexts, targets, node_map, service_map


def prepare_gru4rec_data(X_raw: List, y_raw: List, node_map: Dict, service_map: Dict, max_seq_len: int = 10) -> Tuple:
    """
    Подготовка данных для GRU4Rec
    
    Args:
        X_raw: Список контекстов (туплы узлов)
        y_raw: Список целевых узлов (сервисов)
        node_map: Маппинг всех узлов в индексы (для входных последовательностей)
        service_map: Маппинг сервисов в индексы (для целевых значений)
        max_seq_len: Максимальная длина последовательности
    
    Returns:
        sequences: (num_samples, max_seq_len) - padded последовательности (индексы в node_map)
        lengths: (num_samples,) - реальные длины
        targets: (num_samples,) - целевые сервисы (индексы в service_map)
    """
    logger.info("Preparing GRU4Rec data")
    
    sequences = []
    lengths = []
    targets_list = []
    
    for context, target in zip(X_raw, y_raw):
        # Преобразуем контекст в индексы (используем node_map для всех узлов)
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
        # Целевой сервис используем из service_map (только сервисы)
        targets_list.append(service_map[target])
    
    sequences = torch.tensor(sequences, dtype=torch.long)
    lengths = torch.tensor(lengths, dtype=torch.long)
    targets_tensor = torch.tensor(targets_list, dtype=torch.long)
    
    logger.info(f"GRU4Rec data: {sequences.shape}, lengths: {lengths.shape}")
    logger.info(f"Targets use service_map: {len(service_map)} classes")
    
    return sequences, lengths, targets_tensor


class FocalLoss(nn.Module):
    """
    Focal Loss для борьбы с несбалансированными классами
    Уменьшает вес "легких" примеров и фокусируется на "сложных"
    """
    def __init__(self, alpha=None, gamma=2.0, reduction='mean'):
        super().__init__()
        self.alpha = alpha  # Веса классов
        self.gamma = gamma  # Фокусирующий параметр
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
    """
    Label smoothing для предотвращения переобучения
    Вместо one-hot [0, 1, 0] использует [0.05, 0.9, 0.05]
    """
    if num_classes is None:
        num_classes = pred.size(-1)
    
    confidence = 1.0 - smoothing
    smooth_value = smoothing / (num_classes - 1)
    
    one_hot = torch.zeros_like(pred).scatter_(1, target.unsqueeze(1), 1)
    smooth_one_hot = one_hot * confidence + (1 - one_hot) * smooth_value
    
    log_probs = F.log_softmax(pred, dim=-1)
    loss = -(smooth_one_hot * log_probs).sum(dim=-1).mean()
    
    return loss


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
    
    # WARNING для mode collapse
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


def train_model_with_focal_loss(model, data_pyg, contexts_train, targets_train, contexts_test, targets_test,
                                optimizer, scheduler, epochs, model_name, model_seed, 
                                class_weights=None, use_label_smoothing=True):
    """
    Улучшенная функция обучения с Focal Loss и Label Smoothing
    Борется с mode collapse и несбалансированными классами
    """
    
    torch.manual_seed(model_seed)
    np.random.seed(model_seed)
    
    # Создаем Focal Loss с весами классов
    if class_weights is not None:
        class_weights_tensor = torch.FloatTensor(class_weights)
        focal_criterion = FocalLoss(alpha=class_weights_tensor, gamma=2.0)
    else:
        focal_criterion = FocalLoss(gamma=2.0)
    
    best_loss = float('inf')
    patience_counter = 0
    patience = 50
    num_classes = None

    for epoch in tqdm(range(epochs), desc=f"{model_name} Training (Focal)"):
        model.train()
        optimizer.zero_grad()
        
        if hasattr(model, 'dagnn'):  # DAGNN
            out = model(data_pyg.x, data_pyg.edge_index, training=True)[contexts_train]
        else:  # GCN, GraphSAGE
            out = model(data_pyg, training=True)[contexts_train]
        
        # Получаем правильное количество классов из выхода модели (в первой эпохе)
        if num_classes is None:
            num_classes = out.size(-1)
        
        # Комбинированная loss: Focal Loss + Label Smoothing
        if use_label_smoothing:
            focal_loss = focal_criterion(out, targets_train)
            smooth_loss = label_smoothing_loss(out, targets_train, smoothing=0.1, num_classes=num_classes)
            loss = 0.7 * focal_loss + 0.3 * smooth_loss  # Взвешенная комбинация
        else:
            loss = focal_criterion(out, targets_train)
        
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
        
        # Temperature scaling для лучшей калибровки
        temperature = 1.5
        test_output = test_output / temperature
        
        preds = test_output.argmax(dim=1).numpy()
        proba = F.softmax(test_output, dim=1).numpy()

    return preds, proba


def train_sasrec(
    sequences_train, targets_train, sequences_test, targets_test,
    num_items, epochs, hidden_size=64, num_heads=2, num_blocks=2,
    dropout=0.3, lr=0.001, model_seed=42, num_classes=None
):
    """Обучение SASRec модели"""
    
    logger.info(f"Training SASRec (hidden={hidden_size}, heads={num_heads}, blocks={num_blocks}, dropout={dropout})...")
    
    torch.manual_seed(model_seed)
    np.random.seed(model_seed)
    
    # Если num_classes не задан, используем num_items
    if num_classes is None:
        num_classes = num_items
    
    model = SASRec(
        num_items=num_items + 1,
        num_classes=num_classes,
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


def train_dagnn_sequential(
    data_pyg, sequences_train, lengths_train, targets_train,
    sequences_test, lengths_test, targets_test,
    hidden_channels, epochs, K=10, num_gru_layers=2,
    dropout=0.4, lr=0.001, model_seed=42, num_service_classes=None
):
    """Обучение DAGNNSequential модели (DAGNN + GRU)"""
    
    logger.info(f"Training DAGNNSequential (hidden={hidden_channels}, K={K}, GRU_layers={num_gru_layers}, dropout={dropout})...")
    
    torch.manual_seed(model_seed)
    np.random.seed(model_seed)
    
    # Определяем количество классов для предсказания
    if num_service_classes is None:
        num_service_classes = data_pyg.x.size(0)  # Fallback: все узлы
    
    # Создаем модель
    model = DAGNNSequential(
        in_channels=2,
        hidden_channels=hidden_channels,
        out_channels=num_service_classes,  # Предсказываем только сервисы
        K=K,
        num_gru_layers=num_gru_layers,
        dropout=dropout
    )
    
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=20
    )
    
    best_loss = float('inf')
    patience_counter = 0
    patience = 50
    
    for epoch in tqdm(range(epochs), desc="DAGNNSequential Training"):
        model.train()
        optimizer.zero_grad()
        
        # Forward pass
        out = model(
            x=data_pyg.x,
            edge_index=data_pyg.edge_index,
            sequences=sequences_train,
            lengths=lengths_train,
            training=True
        )
        
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


def train_dagnn_attention(
    data_pyg, sequences_train, targets_train,
    sequences_test, targets_test,
    hidden_channels, epochs, K=10, num_heads=4,
    dropout=0.4, lr=0.001, model_seed=42, num_service_classes=None
):
    """Обучение DAGNNAttention модели (DAGNN + Attention)"""
    
    logger.info(f"Training DAGNNAttention (hidden={hidden_channels}, K={K}, heads={num_heads}, dropout={dropout})...")
    
    torch.manual_seed(model_seed)
    np.random.seed(model_seed)
    
    # Определяем количество классов для предсказания
    if num_service_classes is None:
        num_service_classes = data_pyg.x.size(0)  # Fallback: все узлы
    
    # Создаем модель
    model = DAGNNAttention(
        in_channels=2,
        hidden_channels=hidden_channels,
        out_channels=num_service_classes,  # Предсказываем только сервисы
        K=K,
        num_heads=num_heads,
        dropout=dropout
    )
    
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=20
    )
    
    best_loss = float('inf')
    patience_counter = 0
    patience = 50
    
    for epoch in tqdm(range(epochs), desc="DAGNNAttention Training"):
        model.train()
        optimizer.zero_grad()
        
        # Создаем маску для padding
        mask = (sequences_train == 0)
        
        # Forward pass
        out = model(
            x=data_pyg.x,
            edge_index=data_pyg.edge_index,
            sequences=sequences_train,
            mask=mask,
            training=True
        )
        
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
        mask_test = (sequences_test == 0)
        test_output = model(
            x=data_pyg.x,
            edge_index=data_pyg.edge_index,
            sequences=sequences_test,
            mask=mask_test,
            training=False
        )
        preds = test_output.argmax(dim=1).numpy()
        proba = F.softmax(test_output, dim=1).numpy()
    
    return preds, proba


def train_caser(
    sequences_train, targets_train, sequences_test, targets_test,
    num_items, epochs, embedding_dim=64, num_h_filters=16,
    num_v_filters=4, dropout=0.3, lr=0.001, model_seed=42, num_classes=None
):
    """Обучение Caser модели"""
    
    logger.info(f"Training Caser (embedding={embedding_dim}, h_filters={num_h_filters}, v_filters={num_v_filters}, dropout={dropout})...")
    
    torch.manual_seed(model_seed)
    np.random.seed(model_seed)
    
    # Если num_classes не задан, используем num_items
    if num_classes is None:
        num_classes = num_items
    
    model = Caser(
        num_items=num_items + 1,
        num_classes=num_classes,
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
    num_layers=2, dropout=0.4, lr=0.001, model_seed=42, num_classes=None
):
    """Обучение GRU4Rec модели"""
    
    logger.info(f"Training GRU4Rec (embedding={embedding_dim}, hidden={hidden_size}, layers={num_layers}, dropout={dropout})...")
    
    # Устанавливаем seed
    torch.manual_seed(model_seed)
    np.random.seed(model_seed)
    
    # Если num_classes не задан, используем num_items (backward compatibility)
    if num_classes is None:
        num_classes = num_items
    
    # Создаем модель
    model = GRU4Rec(
        num_items=num_items,  # Для эмбеддингов (все узлы + padding)
        num_classes=num_classes,  # Для предсказаний (только сервисы)
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


# ==================== Кросс-валидация ====================

def cross_validate_models(
    data_pyg, contexts, targets, sequences_all, lengths_all, 
    node_map, service_map, mlb, args, n_splits=5
):
    """
    Универсальная функция кросс-валидации для всех моделей
    
    Args:
        data_pyg: PyG graph data
        contexts: Context node indices
        targets: Target service indices
        sequences_all: Sequences for sequential models
        lengths_all: Sequence lengths
        node_map: Node to index mapping
        service_map: Service to index mapping
        mlb: MultiLabelBinarizer for feature encoding
        args: Command line arguments
        n_splits: Number of CV folds
    
    Returns:
        Dictionary with aggregated results for all models
    """
    logger.info(f"\n{'='*70}")
    logger.info(f"🔄 STARTING {n_splits}-FOLD CROSS-VALIDATION FOR ALL MODELS")
    logger.info(f"{'='*70}\n")
    
    # Инициализируем словарь для хранения результатов каждой модели по всем фолдам
    all_fold_results = {
        'Popularity': [],
        'Markov Chain': [],
        'Random Forest': [],
        'GCN': [],
        'DAGNN': [],
        'DAGNN-Improved (Focal)': [],
        'GraphSAGE': [],
        'GRU4Rec': [],
        'SASRec': [],
        'Caser': [],
        'DAGNNSequential (DAGNN+GRU)': [],
        'DAGNNAttention (DAGNN+Attn)': []
    }
    
    # StratifiedKFold для обеспечения одинакового распределения классов
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=args.random_seed)
    
    # Кросс-валидация
    fold_idx = 1
    for train_idx, test_idx in skf.split(contexts, targets):
        logger.info(f"\n{'='*70}")
        logger.info(f"📊 FOLD {fold_idx}/{n_splits}")
        logger.info(f"{'='*70}\n")
        
        # Split для графовых моделей
        contexts_train = contexts[train_idx]
        contexts_test = contexts[test_idx]
        targets_train = targets[train_idx]
        targets_test = targets[test_idx]
        
        # Split для sequential моделей
        sequences_train = sequences_all[train_idx]
        sequences_test = sequences_all[test_idx]
        lengths_train = lengths_all[train_idx]
        lengths_test = lengths_all[test_idx]
        
        logger.info(f"Fold {fold_idx}: Train samples = {len(train_idx)}, Test samples = {len(test_idx)}")
        logger.info(f"Train class distribution: {Counter(targets_train.numpy())}")
        logger.info(f"Test class distribution: {Counter(targets_test.numpy())}")
        
        # 1. Popularity baseline
        logger.info(f"\n[Fold {fold_idx}] Training Popularity baseline...")
        counter = Counter(targets_train.numpy())
        top_label = counter.most_common(1)[0][0]
        pop_preds = np.array([top_label] * len(targets_test))
        pop_proba = np.zeros((len(targets_test), len(service_map)))
        pop_proba[:, top_label] = 1
        fold_metrics = evaluate_model_with_ndcg(
            pop_preds, targets_test.numpy(), proba_preds=pop_proba, 
            name=f"Popularity (Fold {fold_idx})"
        )
        all_fold_results['Popularity'].append(fold_metrics)
        
        # 2. Markov Chain
        logger.info(f"\n[Fold {fold_idx}] Training Markov Chain...")
        markov = MarkovChain(num_states=len(node_map) + 1, smoothing=1.0)
        markov.fit(sequences_train, targets_train, node_map, service_map)
        markov_preds, markov_proba = markov.predict(sequences_test)
        fold_metrics = evaluate_model_with_ndcg(
            markov_preds, targets_test.numpy(), proba_preds=markov_proba, 
            name=f"Markov Chain (Fold {fold_idx})"
        )
        all_fold_results['Markov Chain'].append(fold_metrics)
        
        # 3. Random Forest
        logger.info(f"\n[Fold {fold_idx}] Training Random Forest...")
        # Преобразуем contexts в features для RandomForest
        X_train_rf = contexts_to_features(contexts_train, node_map, mlb)
        X_test_rf = contexts_to_features(contexts_test, node_map, mlb)
        
        np.random.seed(args.random_seed + fold_idx * 10)
        rf = RandomForestClassifier(
            n_estimators=200, 
            max_depth=15, 
            min_samples_split=5,
            random_state=args.random_seed + fold_idx * 10, 
            n_jobs=-1
        )
        rf.fit(X_train_rf, targets_train.numpy())
        rf_preds = rf.predict(X_test_rf)
        rf_proba = rf.predict_proba(X_test_rf)
        fold_metrics = evaluate_model_with_ndcg(
            rf_preds, targets_test.numpy(), proba_preds=rf_proba, 
            name=f"Random Forest (Fold {fold_idx})"
        )
        all_fold_results['Random Forest'].append(fold_metrics)
        
        # 4. GCN
        logger.info(f"\n[Fold {fold_idx}] Training GCN...")
        torch.manual_seed(args.random_seed + fold_idx * 10 + 1)
        gcn = GCNRecommender(
            in_channels=2, 
            hidden_channels=args.hidden_channels * 2,
            out_channels=len(service_map),
            dropout=0.3
        )
        opt_gcn = torch.optim.Adam(gcn.parameters(), lr=args.learning_rate * 1.5, weight_decay=5e-5)
        sched_gcn = torch.optim.lr_scheduler.ReduceLROnPlateau(opt_gcn, mode='min', factor=0.5, patience=20)
        
        gcn_preds, gcn_proba = train_model_generic(
            gcn, data_pyg, contexts_train, targets_train, contexts_test, targets_test,
            opt_gcn, sched_gcn, args.epochs, f"GCN", args.random_seed + fold_idx * 10 + 1
        )
        fold_metrics = evaluate_model_with_ndcg(
            gcn_preds, targets_test.numpy(), proba_preds=gcn_proba, 
            name=f"GCN (Fold {fold_idx})"
        )
        all_fold_results['GCN'].append(fold_metrics)
        
        # 5. DAGNN
        logger.info(f"\n[Fold {fold_idx}] Training DAGNN...")
        torch.manual_seed(args.random_seed + fold_idx * 10 + 2)
        dagnn = DAGNNRecommender(
            in_channels=2, 
            hidden_channels=args.hidden_channels,
            out_channels=len(service_map), 
            dropout=0.4
        )
        opt_dagnn = torch.optim.Adam(dagnn.parameters(), lr=args.learning_rate * 0.8, weight_decay=1e-4)
        sched_dagnn = torch.optim.lr_scheduler.ReduceLROnPlateau(opt_dagnn, mode='min', factor=0.5, patience=20)
        
        dagnn_preds, dagnn_proba = train_model_generic(
            dagnn, data_pyg, contexts_train, targets_train, contexts_test, targets_test,
            opt_dagnn, sched_dagnn, args.epochs, f"DAGNN", args.random_seed + fold_idx * 10 + 2
        )
        fold_metrics = evaluate_model_with_ndcg(
            dagnn_preds, targets_test.numpy(), proba_preds=dagnn_proba, 
            name=f"DAGNN (Fold {fold_idx})"
        )
        all_fold_results['DAGNN'].append(fold_metrics)
        
        # 6. DAGNN-Improved (с Focal Loss)
        logger.info(f"\n[Fold {fold_idx}] Training DAGNN-Improved (Focal Loss)...")
        
        # Веса классов для текущего фолда
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
        
        torch.manual_seed(args.random_seed + fold_idx * 10 + 20)
        dagnn_improved = DAGNNRecommender(
            in_channels=2, 
            hidden_channels=args.hidden_channels * 2,
            out_channels=len(service_map),
            K=15,
            dropout=0.5
        )
        opt_dagnn_improved = torch.optim.Adam(
            dagnn_improved.parameters(), 
            lr=args.learning_rate * 0.5,
            weight_decay=5e-4
        )
        sched_dagnn_improved = torch.optim.lr_scheduler.ReduceLROnPlateau(
            opt_dagnn_improved, mode='min', factor=0.5, patience=25
        )
        
        dagnn_improved_preds, dagnn_improved_proba = train_model_with_focal_loss(
            dagnn_improved, data_pyg, 
            contexts_train, targets_train, 
            contexts_test, targets_test,
            opt_dagnn_improved, sched_dagnn_improved, args.epochs, 
            f"DAGNN-Improved", args.random_seed + fold_idx * 10 + 20,
            class_weights=class_weights,
            use_label_smoothing=True
        )
        fold_metrics = evaluate_model_with_ndcg(
            dagnn_improved_preds, targets_test.numpy(), 
            proba_preds=dagnn_improved_proba, 
            name=f"DAGNN-Improved (Fold {fold_idx})"
        )
        all_fold_results['DAGNN-Improved (Focal)'].append(fold_metrics)
        
        # 7. GraphSAGE
        logger.info(f"\n[Fold {fold_idx}] Training GraphSAGE...")
        torch.manual_seed(args.random_seed + fold_idx * 10 + 3)
        sage = GraphSAGERecommender(
            in_channels=2, 
            hidden_channels=args.hidden_channels,
            out_channels=len(service_map), 
            dropout=0.4
        )
        opt_sage = torch.optim.Adam(sage.parameters(), lr=args.learning_rate, weight_decay=1e-4)
        sched_sage = torch.optim.lr_scheduler.ReduceLROnPlateau(opt_sage, mode='min', factor=0.5, patience=20)
        
        sage_preds, sage_proba = train_model_generic(
            sage, data_pyg, contexts_train, targets_train, contexts_test, targets_test,
            opt_sage, sched_sage, args.epochs, f"GraphSAGE", args.random_seed + fold_idx * 10 + 3
        )
        fold_metrics = evaluate_model_with_ndcg(
            sage_preds, targets_test.numpy(), proba_preds=sage_proba, 
            name=f"GraphSAGE (Fold {fold_idx})"
        )
        all_fold_results['GraphSAGE'].append(fold_metrics)
        
        # 8. GRU4Rec
        logger.info(f"\n[Fold {fold_idx}] Training GRU4Rec...")
        gru4rec_preds, gru4rec_proba = train_gru4rec(
            sequences_train, lengths_train, targets_train,
            sequences_test, lengths_test, targets_test,
            num_items=len(node_map) + 1,
            num_classes=len(service_map),
            epochs=args.epochs,
            embedding_dim=64,
            hidden_size=args.hidden_channels * 2,
            num_layers=2,
            dropout=args.dropout,
            lr=args.learning_rate,
            model_seed=args.random_seed + fold_idx * 10 + 4
        )
        fold_metrics = evaluate_model_with_ndcg(
            gru4rec_preds, targets_test.numpy(), proba_preds=gru4rec_proba, 
            name=f"GRU4Rec (Fold {fold_idx})"
        )
        all_fold_results['GRU4Rec'].append(fold_metrics)
        
        # 9. SASRec
        logger.info(f"\n[Fold {fold_idx}] Training SASRec...")
        sasrec_preds, sasrec_proba = train_sasrec(
            sequences_train, targets_train, sequences_test, targets_test,
            num_items=len(node_map) + 1,
            num_classes=len(service_map),
            epochs=args.epochs,
            hidden_size=args.hidden_channels,
            num_heads=2,
            num_blocks=2,
            dropout=args.dropout,
            lr=args.learning_rate,
            model_seed=args.random_seed + fold_idx * 10 + 5
        )
        fold_metrics = evaluate_model_with_ndcg(
            sasrec_preds, targets_test.numpy(), proba_preds=sasrec_proba, 
            name=f"SASRec (Fold {fold_idx})"
        )
        all_fold_results['SASRec'].append(fold_metrics)
        
        # 10. Caser
        logger.info(f"\n[Fold {fold_idx}] Training Caser...")
        caser_preds, caser_proba = train_caser(
            sequences_train, targets_train, sequences_test, targets_test,
            num_items=len(node_map) + 1,
            num_classes=len(service_map),
            epochs=args.epochs,
            embedding_dim=64,
            num_h_filters=16,
            num_v_filters=4,
            dropout=args.dropout,
            lr=args.learning_rate,
            model_seed=args.random_seed + fold_idx * 10 + 6
        )
        fold_metrics = evaluate_model_with_ndcg(
            caser_preds, targets_test.numpy(), proba_preds=caser_proba, 
            name=f"Caser (Fold {fold_idx})"
        )
        all_fold_results['Caser'].append(fold_metrics)
        
        # 11. DAGNNSequential
        logger.info(f"\n[Fold {fold_idx}] Training DAGNNSequential (DAGNN+GRU)...")
        dagnn_seq_preds, dagnn_seq_proba = train_dagnn_sequential(
            data_pyg,
            sequences_train, lengths_train, targets_train,
            sequences_test, lengths_test, targets_test,
            hidden_channels=args.hidden_channels,
            epochs=args.epochs,
            K=10,
            num_gru_layers=2,
            dropout=args.dropout,
            lr=args.learning_rate,
            model_seed=args.random_seed + fold_idx * 10 + 7,
            num_service_classes=len(service_map)
        )
        fold_metrics = evaluate_model_with_ndcg(
            dagnn_seq_preds, targets_test.numpy(), proba_preds=dagnn_seq_proba, 
            name=f"DAGNNSequential (Fold {fold_idx})"
        )
        all_fold_results['DAGNNSequential (DAGNN+GRU)'].append(fold_metrics)
        
        # 12. DAGNNAttention
        logger.info(f"\n[Fold {fold_idx}] Training DAGNNAttention (DAGNN+Attention)...")
        dagnn_att_preds, dagnn_att_proba = train_dagnn_attention(
            data_pyg,
            sequences_train, targets_train,
            sequences_test, targets_test,
            hidden_channels=args.hidden_channels,
            epochs=args.epochs,
            K=10,
            num_heads=4,
            dropout=args.dropout,
            lr=args.learning_rate,
            model_seed=args.random_seed + fold_idx * 10 + 8,
            num_service_classes=len(service_map)
        )
        fold_metrics = evaluate_model_with_ndcg(
            dagnn_att_preds, targets_test.numpy(), proba_preds=dagnn_att_proba, 
            name=f"DAGNNAttention (Fold {fold_idx})"
        )
        all_fold_results['DAGNNAttention (DAGNN+Attn)'].append(fold_metrics)
        
        fold_idx += 1
    
    # Агрегируем результаты по всем фолдам
    logger.info(f"\n{'='*70}")
    logger.info(f"📊 AGGREGATING RESULTS ACROSS ALL {n_splits} FOLDS")
    logger.info(f"{'='*70}\n")
    
    aggregated_results = {}
    for model_name, fold_metrics_list in all_fold_results.items():
        # Вычисляем среднее и стандартное отклонение для каждой метрики
        metrics_aggregated = {}
        metric_names = fold_metrics_list[0].keys()
        
        for metric_name in metric_names:
            values = [fm[metric_name] for fm in fold_metrics_list if fm[metric_name] is not None]
            if values:
                metrics_aggregated[f'{metric_name}_mean'] = np.mean(values)
                metrics_aggregated[f'{metric_name}_std'] = np.std(values)
            else:
                metrics_aggregated[f'{metric_name}_mean'] = None
                metrics_aggregated[f'{metric_name}_std'] = None
        
        aggregated_results[model_name] = metrics_aggregated
    
    return aggregated_results


def print_cv_results(aggregated_results):
    """
    Выводит результаты кросс-валидации в красивом формате
    """
    logger.info("\n" + "="*90)
    logger.info("🏆 ИТОГОВЫЕ РЕЗУЛЬТАТЫ КРОСС-ВАЛИДАЦИИ (Mean ± Std)")
    logger.info("="*90)
    
    # Сортируем модели по accuracy_mean
    sorted_results = sorted(
        aggregated_results.items(), 
        key=lambda x: x[1]['accuracy_mean'] if x[1]['accuracy_mean'] is not None else 0, 
        reverse=True
    )
    
    for rank, (model_name, metrics) in enumerate(sorted_results, 1):
        logger.info(f"\n{'='*90}")
        logger.info(f"#{rank} {model_name}")
        logger.info(f"{'='*90}")
        
        # Группируем метрики
        metric_groups = ['accuracy', 'f1', 'precision', 'recall', 'ndcg']
        
        for metric_base in metric_groups:
            mean_key = f'{metric_base}_mean'
            std_key = f'{metric_base}_std'
            
            if mean_key in metrics and metrics[mean_key] is not None:
                mean_val = metrics[mean_key]
                std_val = metrics[std_key] if std_key in metrics else 0
                logger.info(f"  {metric_base:12s}: {mean_val:.4f} ± {std_val:.4f}")
            elif mean_key in metrics:
                logger.info(f"  {metric_base:12s}: N/A")
    
    # Статистика различий
    logger.info("\n" + "="*90)
    logger.info("🔍 СТАТИСТИЧЕСКИЙ АНАЛИЗ")
    logger.info("="*90)
    
    accuracies = [m['accuracy_mean'] for m in aggregated_results.values() 
                  if m['accuracy_mean'] is not None]
    
    logger.info(f"Accuracy - Min:   {min(accuracies):.4f}")
    logger.info(f"Accuracy - Max:   {max(accuracies):.4f}")
    logger.info(f"Accuracy - Range: {max(accuracies) - min(accuracies):.4f}")
    logger.info(f"Accuracy - Mean:  {np.mean(accuracies):.4f}")
    logger.info(f"Accuracy - Std:   {np.std(accuracies):.4f}")
    
    # Лучшие модели
    logger.info(f"\n🥇 Best Model (Accuracy): {sorted_results[0][0]} "
                f"({sorted_results[0][1]['accuracy_mean']:.4f} ± {sorted_results[0][1]['accuracy_std']:.4f})")
    
    # Проверяем nDCG
    ndcg_results = [(name, m['ndcg_mean']) for name, m in aggregated_results.items() 
                    if m.get('ndcg_mean') is not None]
    if ndcg_results:
        best_ndcg = max(ndcg_results, key=lambda x: x[1])
        logger.info(f"🥇 Best Model (nDCG):     {best_ndcg[0]} ({best_ndcg[1]:.4f})")
    
    logger.info("="*90)


def main():
    parser = argparse.ArgumentParser(description="FINAL DAG-based Recommender with Fixed Results")
    parser.add_argument("--data", type=str, default="compositionsDAG.json")
    parser.add_argument("--test-size", type=float, default=0.3)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--hidden-channels", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument("--dropout", type=float, default=0.4)
    parser.add_argument("--random-seed", type=int, default=42)
    parser.add_argument("--use-cv", action="store_true", 
                        help="Use cross-validation instead of single train/test split")
    parser.add_argument("--cv-folds", type=int, default=5,
                        help="Number of folds for cross-validation (default: 5)")
    
    args = parser.parse_args()

    # Load data
    data_path = Path(args.data)
    if not data_path.exists():
        logger.error(f"Data file not found: {data_path}")
        sys.exit(1)

    dag, compositions_data = load_dag_from_json(data_path)
    paths = extract_paths_from_compositions(compositions_data)
    X_raw, y_raw = create_training_data(paths)
    
    # Построим граф из реальных путей для извлечения честных фичей
    path_graph = build_graph_from_real_paths(paths)
    
    # Извлекаем ЧЕСТНЫЕ графовые фичи (без утечки target)
    honest_graph_features = extract_honest_graph_features(path_graph, X_raw)

    # Vectorize
    logger.info("Vectorizing data...")
    mlb = MultiLabelBinarizer()
    X_base = mlb.fit_transform(X_raw)
    
    # Объединяем базовые признаки с честными графовыми
    logger.info(f"Combining base features {X_base.shape} with honest graph features {honest_graph_features.shape}")
    X_combined = np.hstack([X_base, honest_graph_features])
    logger.info(f"Combined feature matrix shape: {X_combined.shape}")
    
    le = LabelEncoder()
    y = le.fit_transform(y_raw)

    # Стратифицированный split (с проверкой возможности)
    y_counts = Counter(y)
    min_samples = min(y_counts.values())
    
    # Split для базовых признаков
    if min_samples >= 2:
        logger.info("Using STRATIFIED split to ensure balanced classes...")
        X_base_train, X_base_test, y_train, y_test = train_test_split(
            X_base, y, test_size=args.test_size, random_state=args.random_seed, stratify=y
        )
    else:
        logger.warning(f"Too few samples for stratification (min={min_samples}). Using random split.")
        X_base_train, X_base_test, y_train, y_test = train_test_split(
            X_base, y, test_size=args.test_size, random_state=args.random_seed
        )
    
    logger.info(f"Train samples: {len(X_base_train)}, Test samples: {len(X_base_test)}")
    logger.info(f"Train class distribution: {Counter(y_train)}")
    logger.info(f"Test class distribution: {Counter(y_test)}")

    # Prepare PyG data from real paths
    data_pyg, contexts, targets, node_map, service_map = prepare_pytorch_geometric_data(
        dag, X_raw, y_raw, paths
    )
    
    logger.info(f"\n{'='*60}")
    logger.info(f"📊 DATA SUMMARY:")
    logger.info(f"  Total nodes in graph: {len(node_map)}")
    logger.info(f"  Target service classes: {len(service_map)}")
    logger.info(f"  Models will predict {len(service_map)} classes (services only)")
    logger.info(f"  Graph edges: {data_pyg.edge_index.size(1)} (directed, no self-loops)")
    logger.info(f"{'='*60}\n")
    
    # Стратифицированный split для графовых моделей (с проверкой)
    target_counts_graph = Counter(targets.numpy())
    min_samples_graph = min(target_counts_graph.values())
    
    if min_samples_graph >= 2:
        contexts_train, contexts_test, targets_train, targets_test = train_test_split(
            contexts, targets, test_size=args.test_size, random_state=args.random_seed,
            stratify=targets.numpy()
        )
    else:
        logger.warning(f"Too few samples for stratification in graph data (min={min_samples_graph}). Using random split.")
        contexts_train, contexts_test, targets_train, targets_test = train_test_split(
            contexts, targets, test_size=args.test_size, random_state=args.random_seed
        )
    logger.info(f"Graph train samples: {len(contexts_train)}, test samples: {len(contexts_test)}")

    # Подготовка данных для Sequential моделей (нужны для обоих режимов)
    sequences_all, lengths_all, targets_gru = prepare_gru4rec_data(X_raw, y_raw, node_map, service_map, max_seq_len=10)
    
    # ВЫБОР РЕЖИМА: Кросс-валидация или простой train/test split
    if args.use_cv:
        logger.info(f"\n{'='*70}")
        logger.info(f"🔄 CROSS-VALIDATION MODE ENABLED ({args.cv_folds} folds)")
        logger.info(f"{'='*70}\n")
        
        # Запускаем кросс-валидацию для всех моделей
        aggregated_results = cross_validate_models(
            data_pyg=data_pyg,
            contexts=contexts,
            targets=targets,
            sequences_all=sequences_all,
            lengths_all=lengths_all,
            node_map=node_map,
            service_map=service_map,
            mlb=mlb,
            args=args,
            n_splits=args.cv_folds
        )
        
        # Выводим агрегированные результаты
        print_cv_results(aggregated_results)
        
        # Завершаем программу после кросс-валидации
        return
    
    # СТАНДАРТНЫЙ РЕЖИМ: простой train/test split
    logger.info(f"\n{'='*70}")
    logger.info(f"📊 STANDARD TRAIN/TEST SPLIT MODE")
    logger.info(f"{'='*70}\n")
    
    results = {}

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

    # Random Forest
    logger.info("Training Random Forest...")
    # Преобразуем contexts в features для RandomForest
    X_train_rf = contexts_to_features(contexts_train, node_map, mlb)
    X_test_rf = contexts_to_features(contexts_test, node_map, mlb)
    
    np.random.seed(args.random_seed)
    rf = RandomForestClassifier(
        n_estimators=200, 
        max_depth=15, 
        min_samples_split=5,
        random_state=args.random_seed, 
        n_jobs=-1
    )
    rf.fit(X_train_rf, targets_train.numpy())
    rf_preds = rf.predict(X_test_rf)
    rf_proba = rf.predict_proba(X_test_rf)
    results['Random Forest'] = evaluate_model_with_ndcg(
        rf_preds, targets_test.numpy(), proba_preds=rf_proba, name="Random Forest"
    )

    # GCN - с уникальным seed (улучшенные параметры)
    logger.info(f"Training GCN with seed={args.random_seed + 1}...")
    torch.manual_seed(args.random_seed + 1)
    gcn = GCNRecommender(
        in_channels=2, 
        hidden_channels=args.hidden_channels * 2,  # Больше capacity
        out_channels=len(service_map),  # Предсказываем только сервисы
        dropout=0.3  # Меньше dropout для лучшего обучения
    )
    opt_gcn = torch.optim.Adam(gcn.parameters(), lr=args.learning_rate * 1.5, weight_decay=5e-5)
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
        out_channels=len(service_map), dropout=0.4  # Предсказываем только сервисы
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

    # DAGNN IMPROVED - с Focal Loss и Label Smoothing
    logger.info("\n" + "="*70)
    logger.info(f"Training DAGNN IMPROVED (Focal Loss + Label Smoothing) seed={args.random_seed + 20}...")
    
    # Вычисляем веса классов для борьбы с несбалансированностью
    train_class_counts = Counter(targets_train.numpy())
    total_samples = sum(train_class_counts.values())
    num_all_classes = len(service_map)  # Всего сервисов (целевых классов)
    
    # Inverse frequency weights для ВСЕХ классов
    class_weights = []
    for i in range(num_all_classes):
        count = train_class_counts.get(i, 1)  # Если класса нет, вес = среднему
        if count > 0:
            weight = total_samples / (len(train_class_counts) * count)
        else:
            weight = 1.0  # Нейтральный вес для классов которых нет в train
        class_weights.append(weight)
    
    logger.info(f"Class weights (first 5): {[f'{w:.2f}' for w in class_weights[:5]]}")
    logger.info(f"Total service classes: {num_all_classes}, Present in train: {len(train_class_counts)}")
    logger.info(f"Using Focal Loss (gamma=2.0) + Label Smoothing (0.1)")
    
    torch.manual_seed(args.random_seed + 20)
    dagnn_improved = DAGNNRecommender(
        in_channels=2, 
        hidden_channels=args.hidden_channels * 2,  # Больше capacity
        out_channels=len(service_map),  # Предсказываем только сервисы
        K=15,  # Больше propagation steps
        dropout=0.5  # Больше dropout для регуляризации
    )
    opt_dagnn_improved = torch.optim.Adam(
        dagnn_improved.parameters(), 
        lr=args.learning_rate * 0.5,  # Меньше learning rate
        weight_decay=5e-4  # Больше weight decay
    )
    sched_dagnn_improved = torch.optim.lr_scheduler.ReduceLROnPlateau(
        opt_dagnn_improved, mode='min', factor=0.5, patience=25
    )
    
    dagnn_improved_preds, dagnn_improved_proba = train_model_with_focal_loss(
        dagnn_improved, data_pyg, 
        contexts_train, targets_train, 
        contexts_test, targets_test,
        opt_dagnn_improved, sched_dagnn_improved, args.epochs, 
        "DAGNN-Improved", args.random_seed + 20,
        class_weights=class_weights,
        use_label_smoothing=True
    )
    results['DAGNN-Improved (Focal)'] = evaluate_model_with_ndcg(
        dagnn_improved_preds, targets_test.numpy(), 
        proba_preds=dagnn_improved_proba, 
        name="DAGNN-Improved (Focal Loss + Label Smoothing)"
    )

    # GraphSAGE - с еще другим seed
    logger.info(f"Training GraphSAGE with seed={args.random_seed + 3}...")
    torch.manual_seed(args.random_seed + 3)
    sage = GraphSAGERecommender(
        in_channels=2, hidden_channels=args.hidden_channels,
        out_channels=len(service_map), dropout=0.4  # Предсказываем только сервисы
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
    
    # Split для Sequential моделей (данные уже подготовлены выше)
    targets_gru_counts = Counter(targets_gru.numpy())
    min_samples_gru = min(targets_gru_counts.values())
    
    if min_samples_gru >= 2:
        sequences_train, sequences_test, lengths_train, lengths_test, targets_gru_train, targets_gru_test = train_test_split(
            sequences_all, lengths_all, targets_gru,
            test_size=args.test_size, 
            random_state=args.random_seed,
            stratify=targets_gru.numpy()
        )
    else:
        logger.warning(f"Too few samples for stratification in Sequential data (min={min_samples_gru}). Using random split.")
        sequences_train, sequences_test, lengths_train, lengths_test, targets_gru_train, targets_gru_test = train_test_split(
            sequences_all, lengths_all, targets_gru,
            test_size=args.test_size, 
            random_state=args.random_seed
        )
    
    logger.info(f"Sequential models train: {len(sequences_train)}, test: {len(sequences_test)}")
    
    # Markov Chain - используем sequential данные
    logger.info(f"Training Markov Chain with seed={args.random_seed + 10}...")
    markov = MarkovChain(num_states=len(node_map) + 1, smoothing=1.0)
    markov.fit(sequences_train, targets_gru_train, node_map, service_map)
    markov_preds, markov_proba = markov.predict(sequences_test)
    results['Markov Chain'] = evaluate_model_with_ndcg(
        markov_preds, targets_gru_test.numpy(), proba_preds=markov_proba, name="Markov Chain"
    )
    
    # Обучение GRU4Rec
    gru4rec_preds, gru4rec_proba = train_gru4rec(
        sequences_train, lengths_train, targets_gru_train,
        sequences_test, lengths_test, targets_gru_test,
        num_items=len(node_map) + 1,  # +1 для эмбеддингов (включая padding_idx=0)
        num_classes=len(service_map),  # Предсказываем только сервисы
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
        num_items=len(node_map) + 1,  # +1 для эмбеддингов (включая padding_idx=0)
        num_classes=len(service_map),  # Предсказываем только сервисы
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
        num_items=len(node_map) + 1,  # +1 для эмбеддингов (включая padding_idx=0)
        num_classes=len(service_map),  # Предсказываем только сервисы
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

    # DAGNNSequential - Гибридная модель (DAGNN + GRU)
    logger.info(f"\nTraining DAGNNSequential (HYBRID: Graph + Sequence) with seed={args.random_seed + 7}...")
    dagnn_seq_preds, dagnn_seq_proba = train_dagnn_sequential(
        data_pyg,
        sequences_train, lengths_train, targets_gru_train,
        sequences_test, lengths_test, targets_gru_test,
        hidden_channels=args.hidden_channels,
        epochs=args.epochs,
        K=10,
        num_gru_layers=2,
        dropout=args.dropout,
        lr=args.learning_rate,
        model_seed=args.random_seed + 7,
        num_service_classes=len(service_map)  # Предсказываем только сервисы
    )
    results['DAGNNSequential (DAGNN+GRU)'] = evaluate_model_with_ndcg(
        dagnn_seq_preds, targets_gru_test.numpy(), proba_preds=dagnn_seq_proba, 
        name="DAGNNSequential (DAGNN+GRU)"
    )

    # DAGNNAttention - Гибридная модель (DAGNN + Attention)
    logger.info(f"\nTraining DAGNNAttention (HYBRID: Graph + Attention) with seed={args.random_seed + 8}...")
    dagnn_att_preds, dagnn_att_proba = train_dagnn_attention(
        data_pyg,
        sequences_train, targets_gru_train,
        sequences_test, targets_gru_test,
        hidden_channels=args.hidden_channels,
        epochs=args.epochs,
        K=10,
        num_heads=4,
        dropout=args.dropout,
        lr=args.learning_rate,
        model_seed=args.random_seed + 8,
        num_service_classes=len(service_map)  # Предсказываем только сервисы
    )
    results['DAGNNAttention (DAGNN+Attn)'] = evaluate_model_with_ndcg(
        dagnn_att_preds, targets_gru_test.numpy(), proba_preds=dagnn_att_proba, 
        name="DAGNNAttention (DAGNN+Attention)"
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

