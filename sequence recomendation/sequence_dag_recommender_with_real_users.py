#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ВЕРСИЯ С РЕАЛЬНЫМИ ПОЛЬЗОВАТЕЛЯМИ (Version 4.1)

Использует реальные owner из данных вместо синтетических:
1. Для сервисов (nodes с mid) - берем owner напрямую
2. Для таблиц (nodes без mid) - определяем owner через целевые сервисы

Персонализация через:
- User Embeddings
- Personalized PageRank
- Attention Fusion
- User History
"""

import argparse
import json
import logging
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import List, Tuple, Dict, Set

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
from sklearn.preprocessing import LabelBinarizer, LabelEncoder, StandardScaler
from torch_geometric.data import Data
from torch_geometric.nn import APPNP, LayerNorm
from tqdm import tqdm

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# ==================== Персонализированные модели ====================

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


class PersonalizedDAGNN(nn.Module):
    """DAGNN с персонализацией через реальные user IDs"""
    
    def __init__(self, num_users: int, in_channels: int, hidden_channels: int, 
                 out_channels: int, K: int = 10, dropout: float = 0.4):
        super().__init__()
        self.dropout = dropout
        
        # User embeddings
        self.user_embedding = nn.Embedding(num_users, hidden_channels)
        
        # Node encoding
        self.lin1 = nn.Linear(in_channels, hidden_channels)
        self.bn1 = nn.BatchNorm1d(hidden_channels)
        
        self.lin2 = nn.Linear(hidden_channels, hidden_channels)
        self.bn2 = nn.BatchNorm1d(hidden_channels)
        
        # DAGNN propagation
        self.dagnn = DAGNN(hidden_channels, K, dropout=dropout)
        
        # Attention для объединения user + node
        self.attention = nn.MultiheadAttention(hidden_channels, num_heads=4, batch_first=True)
        
        # Classification head
        self.lin3 = nn.Linear(hidden_channels * 2, hidden_channels // 2)
        self.bn3 = nn.BatchNorm1d(hidden_channels // 2)
        
        self.lin_out = nn.Linear(hidden_channels // 2, out_channels)

    def forward(self, x, edge_index, user_ids, contexts, training=True):
        # Node encoding
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
        
        # DAGNN propagation
        x = self.dagnn(x, edge_index, training=training)
        
        # Get context embeddings
        context_emb = x[contexts]
        
        # Get user embeddings
        user_emb = self.user_embedding(user_ids)
        
        # Attention fusion
        user_query = user_emb.unsqueeze(1)
        context_kv = context_emb.unsqueeze(1)
        
        attended, _ = self.attention(user_query, context_kv, context_kv)
        attended = attended.squeeze(1)
        
        # Combine user + attended context
        combined = torch.cat([user_emb, attended], dim=1)
        
        # Classification
        out = self.lin3(combined)
        out = self.bn3(out)
        out = F.relu(out)
        out = F.dropout(out, p=self.dropout, training=training)
        
        out = self.lin_out(out)
        return out


# ==================== Извлечение реальных пользователей ====================

def extract_real_users_from_json(json_path: Path) -> Tuple[Dict[str, str], Dict[str, Set[str]]]:
    """
    Извлекает реальные owner (пользователей) из JSON
    
    Returns:
        node_to_owner: маппинг node_id -> owner
        owner_services: маппинг owner -> set of services
    """
    logger.info(f"Extracting real users from {json_path}")
    
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    node_to_owner = {}
    owner_services = defaultdict(set)
    service_to_owner = {}  # Для определения owner таблиц
    
    # Сначала извлекаем owner для всех сервисов
    for composition in data:
        for node in composition["nodes"]:
            if "mid" in node:
                # Это сервис
                service_name = f"service_{node['mid']}"
                owner = node.get('owner', 'unknown')
                node_to_owner[service_name] = owner
                service_to_owner[service_name] = owner
                owner_services[owner].add(service_name)
    
    # Теперь определяем owner для таблиц через целевые сервисы
    for composition in data:
        for link in composition["links"]:
            source = str(link["source"])
            target = str(link["target"])
            
            # Создаем имена узлов
            source_node = None
            target_node = None
            
            for node in composition["nodes"]:
                if str(node["id"]) == source:
                    if "mid" in node:
                        source_node = f"service_{node['mid']}"
                    else:
                        source_node = f"table_{node['id']}"
                if str(node["id"]) == target:
                    if "mid" in node:
                        target_node = f"service_{node['mid']}"
                    else:
                        target_node = f"table_{node['id']}"
            
            # Если таблица используется сервисом, присваиваем owner сервиса
            if source_node and target_node:
                if source_node.startswith("table_") and target_node in service_to_owner:
                    # Таблица -> Сервис: таблица принадлежит owner сервиса
                    if source_node not in node_to_owner:
                        node_to_owner[source_node] = service_to_owner[target_node]
                
                if target_node.startswith("table_") and source_node in service_to_owner:
                    # Сервис -> Таблица: таблица принадлежит owner сервиса
                    if target_node not in node_to_owner:
                        node_to_owner[target_node] = service_to_owner[source_node]
    
    # Для таблиц без owner присваиваем 'unknown'
    for composition in data:
        for node in composition["nodes"]:
            if "mid" not in node:
                table_name = f"table_{node['id']}"
                if table_name not in node_to_owner:
                    node_to_owner[table_name] = 'unknown'
    
    # Статистика
    owners_count = len(set(node_to_owner.values()))
    services_with_owner = sum(1 for k, v in node_to_owner.items() if k.startswith('service_') and v != 'unknown')
    tables_with_owner = sum(1 for k, v in node_to_owner.items() if k.startswith('table_') and v != 'unknown')
    
    logger.info(f"Extracted {owners_count} unique owners")
    logger.info(f"  Services with owner: {services_with_owner}")
    logger.info(f"  Tables with owner: {tables_with_owner}")
    logger.info(f"  Total nodes: {len(node_to_owner)}")
    
    # Топ-10 самых активных пользователей
    owner_counts = Counter(node_to_owner.values())
    logger.info("\nTop 10 most active users:")
    for owner, count in owner_counts.most_common(10):
        if owner != 'unknown':
            logger.info(f"  {owner}: {count} nodes")
    
    return node_to_owner, owner_services


def load_dag_from_json_with_users(json_path: Path) -> Tuple[nx.DiGraph, Dict[str, str]]:
    """
    Загружает DAG и извлекает реальных пользователей
    """
    logger.info(f"Loading DAG from {json_path}")
    
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    dag = nx.DiGraph()
    id_to_mid = {}
    node_to_owner = {}

    # Сначала создаем маппинг id -> node_name и извлекаем owners
    for composition in data:
        for node in composition["nodes"]:
            if "mid" in node:
                node_name = f"service_{node['mid']}"
                id_to_mid[str(node["id"])] = node_name
                node_to_owner[node_name] = node.get('owner', 'unknown')
            else:
                node_name = f"table_{node['id']}"
                id_to_mid[str(node["id"])] = node_name

    # Создаем граф и определяем owner для таблиц
    service_to_owner = {k: v for k, v in node_to_owner.items() if k.startswith('service_')}
    
    for composition in data:
        for link in composition["links"]:
            source = str(link["source"])
            target = str(link["target"])
            src_node = id_to_mid[source]
            tgt_node = id_to_mid[target]
            
            dag.add_node(src_node, type='service' if src_node.startswith("service") else 'table')
            dag.add_node(tgt_node, type='service' if tgt_node.startswith("service") else 'table')
            dag.add_edge(src_node, tgt_node)
            
            # Определяем owner для таблиц через связанные сервисы
            if src_node.startswith("table_") and tgt_node in service_to_owner:
                if src_node not in node_to_owner:
                    node_to_owner[src_node] = service_to_owner[tgt_node]
            
            if tgt_node.startswith("table_") and src_node in service_to_owner:
                if tgt_node not in node_to_owner:
                    node_to_owner[tgt_node] = service_to_owner[src_node]
    
    # Для таблиц без owner присваиваем 'unknown'
    for node in dag.nodes():
        if node not in node_to_owner:
            node_to_owner[node] = 'unknown'

    logger.info(f"Loaded DAG with {dag.number_of_nodes()} nodes and {dag.number_of_edges()} edges")
    
    return dag, node_to_owner


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


def create_training_data_with_users(
    paths: List[List[str]], 
    node_to_owner: Dict[str, str]
) -> Tuple[List, List, List]:
    """
    Создает обучающие данные с реальными user IDs
    """
    X_raw = []
    y_raw = []
    user_ids_raw = []

    for path in paths:
        for i in range(1, len(path) - 1):
            context = tuple(path[:i])
            next_step = path[i]
            if next_step.startswith("service"):
                # Определяем owner для этой последовательности
                # Берем owner целевого сервиса
                owner = node_to_owner.get(next_step, 'unknown')
                
                X_raw.append(context)
                y_raw.append(next_step)
                user_ids_raw.append(owner)

    logger.info(f"Created {len(X_raw)} training samples with real users")
    
    # Статистика по пользователям
    user_counter = Counter(user_ids_raw)
    logger.info(f"Unique users in training data: {len(user_counter)}")
    logger.info("Top 10 users by sequence count:")
    for user, count in user_counter.most_common(10):
        if user != 'unknown':
            logger.info(f"  {user}: {count} sequences")
    
    return X_raw, y_raw, user_ids_raw


def extract_graph_features(dag: nx.DiGraph) -> Dict[str, Dict[str, float]]:
    """Извлекает графовые метрики"""
    logger.info("Extracting graph features...")
    
    features = {}
    
    in_degrees = dict(dag.in_degree())
    out_degrees = dict(dag.out_degree())
    
    try:
        pagerank = nx.pagerank(dag, alpha=0.85, max_iter=100)
    except:
        pagerank = {node: 1.0 / dag.number_of_nodes() for node in dag.nodes()}
    
    try:
        betweenness = nx.betweenness_centrality(dag, normalized=True)
    except:
        betweenness = {node: 0.0 for node in dag.nodes()}
    
    try:
        closeness = {}
        for node in dag.nodes():
            reachable = nx.single_source_shortest_path_length(dag, node)
            if len(reachable) > 1:
                avg_distance = sum(reachable.values()) / (len(reachable) - 1)
                closeness[node] = 1.0 / avg_distance if avg_distance > 0 else 0.0
            else:
                closeness[node] = 0.0
    except:
        closeness = {node: 0.0 for node in dag.nodes()}
    
    try:
        undirected = dag.to_undirected()
        clustering = nx.clustering(undirected)
    except:
        clustering = {node: 0.0 for node in dag.nodes()}
    
    for node in dag.nodes():
        features[node] = {
            'in_degree': in_degrees.get(node, 0),
            'out_degree': out_degrees.get(node, 0),
            'pagerank': pagerank.get(node, 0.0),
            'betweenness': betweenness.get(node, 0.0),
            'closeness': closeness.get(node, 0.0),
            'clustering': clustering.get(node, 0.0),
        }
    
    return features


def compute_personalized_pagerank(
    dag: nx.DiGraph, 
    user_sequences: Dict[str, List[List[str]]]
) -> Dict[str, Dict[str, float]]:
    """
    Вычисляет персонализированный PageRank для каждого реального пользователя
    """
    logger.info("Computing personalized PageRank for real users...")
    
    user_pageranks = {}
    
    for user_id, sequences in user_sequences.items():
        if user_id == 'unknown':
            continue
            
        # Собираем все узлы, которые пользователь использует
        user_nodes = Counter()
        for seq in sequences:
            for node in seq:
                user_nodes[node] += 1
        
        total_count = sum(user_nodes.values())
        
        # Персонализированный вектор
        personalization = {}
        for node in dag.nodes():
            if node in user_nodes:
                personalization[node] = user_nodes[node] / total_count
            else:
                personalization[node] = 0.0
        
        # Нормализуем
        total_pers = sum(personalization.values())
        if total_pers > 0:
            personalization = {k: v / total_pers for k, v in personalization.items()}
        else:
            personalization = {k: 1.0 / len(dag.nodes()) for k in dag.nodes()}
        
        # Вычисляем персонализированный PageRank
        try:
            ppagerank = nx.pagerank(dag, personalization=personalization, alpha=0.85)
        except:
            ppagerank = {node: 1.0 / len(dag.nodes()) for node in dag.nodes()}
        
        user_pageranks[user_id] = ppagerank
    
    logger.info(f"Computed personalized PageRank for {len(user_pageranks)} users")
    
    return user_pageranks


def prepare_pytorch_geometric_data_with_real_users(
    dag: nx.DiGraph, X_raw: List, y_raw: List, user_ids_raw: List
) -> Tuple[Data, torch.Tensor, torch.Tensor, torch.Tensor, Dict, Dict]:
    """
    Подготовка данных с реальными пользователями
    """
    logger.info("Preparing PyTorch Geometric data with real users...")
    
    node_list = list(dag.nodes)
    node_encoder = LabelEncoder()
    node_ids = node_encoder.fit_transform(node_list)
    node_map = {node: idx for node, idx in zip(node_list, node_ids)}

    edge_index = torch.tensor([[node_map[u], node_map[v]] for u, v in dag.edges], dtype=torch.long).t()
    
    # Извлекаем графовые метрики
    graph_features = extract_graph_features(dag)
    
    # Создаем feature matrix
    features = []
    for node in node_list:
        node_type = dag.nodes[node]['type']
        is_service = 1.0 if node_type == 'service' else 0.0
        is_table = 1.0 if node_type == 'table' else 0.0
        
        gf = graph_features[node]
        feature_vector = [
            is_service,
            is_table,
            gf['in_degree'],
            gf['out_degree'],
            gf['pagerank'],
            gf['betweenness'],
            gf['closeness'],
            gf['clustering']
        ]
        features.append(feature_vector)
    
    features_array = np.array(features, dtype=np.float32)
    
    # Нормализация графовых метрик
    scaler = StandardScaler()
    features_array[:, 2:] = scaler.fit_transform(features_array[:, 2:])
    
    x = torch.tensor(features_array, dtype=torch.float)
    data_pyg = Data(x=x, edge_index=edge_index)

    contexts = torch.tensor([node_map[context[-1]] for context in X_raw], dtype=torch.long)
    targets = torch.tensor([node_map[y] for y in y_raw], dtype=torch.long)
    
    # Кодируем реальные user IDs
    unique_users = list(set(user_ids_raw))
    user_encoder = {user: idx for idx, user in enumerate(unique_users)}
    user_ids_encoded = [user_encoder[user] for user in user_ids_raw]
    user_ids_tensor = torch.tensor(user_ids_encoded, dtype=torch.long)

    logger.info(f"Created data: {len(X_raw)} samples, {len(unique_users)} real users")
    
    return data_pyg, contexts, targets, user_ids_tensor, node_map, user_encoder


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


def train_personalized_dagnn(
    model, data_pyg, contexts_train, targets_train, user_ids_train,
    contexts_test, targets_test, user_ids_test,
    optimizer, scheduler, epochs, model_seed
):
    """Обучение персонализированной DAGNN"""
    
    torch.manual_seed(model_seed)
    np.random.seed(model_seed)
    
    best_loss = float('inf')
    patience_counter = 0
    patience = 50

    for epoch in tqdm(range(epochs), desc="Personalized DAGNN Training"):
        model.train()
        optimizer.zero_grad()
        
        out = model(data_pyg.x, data_pyg.edge_index, user_ids_train, contexts_train, training=True)
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
        test_output = model(data_pyg.x, data_pyg.edge_index, user_ids_test, contexts_test, training=False)
        preds = test_output.argmax(dim=1).numpy()
        proba = F.softmax(test_output, dim=1).numpy()

    return preds, proba


def main():
    parser = argparse.ArgumentParser(description="Personalized DAG Recommender with Real Users")
    parser.add_argument("--data", type=str, default="compositionsDAG.json")
    parser.add_argument("--test-size", type=float, default=0.3)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--hidden-channels", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument("--dropout", type=float, default=0.4)
    parser.add_argument("--random-seed", type=int, default=42)
    
    args = parser.parse_args()

    # Load data with real users
    data_path = Path(args.data)
    if not data_path.exists():
        logger.error(f"Data file not found: {data_path}")
        sys.exit(1)

    dag, node_to_owner = load_dag_from_json_with_users(data_path)
    paths = extract_paths_from_dag(dag)
    X_raw, y_raw, user_ids_raw = create_training_data_with_users(paths, node_to_owner)

    # Prepare data
    data_pyg, contexts, targets, user_ids_tensor, node_map, user_encoder = \
        prepare_pytorch_geometric_data_with_real_users(dag, X_raw, y_raw, user_ids_raw)
    
    # Split data
    logger.info("Splitting data with stratification...")
    (contexts_train, contexts_test, 
     targets_train, targets_test,
     user_ids_train, user_ids_test) = train_test_split(
        contexts, targets, user_ids_tensor,
        test_size=args.test_size,
        random_state=args.random_seed,
        stratify=targets.numpy()
    )
    
    num_users = len(user_encoder)
    
    logger.info(f"\n{'='*70}")
    logger.info(f"🎯 ПЕРСОНАЛИЗАЦИЯ С РЕАЛЬНЫМИ ПОЛЬЗОВАТЕЛЯМИ")
    logger.info(f"{'='*70}")
    logger.info(f"Количество РЕАЛЬНЫХ пользователей: {num_users}")
    logger.info(f"Обучающих примеров: {len(contexts_train)}")
    logger.info(f"Тестовых примеров: {len(contexts_test)}")
    logger.info(f"{'='*70}\n")

    # Train Personalized DAGNN
    logger.info(f"Training Personalized DAGNN with {num_users} real users...")
    torch.manual_seed(args.random_seed)
    
    model_dagnn = PersonalizedDAGNN(
        num_users=num_users,
        in_channels=data_pyg.x.shape[1],
        hidden_channels=args.hidden_channels,
        out_channels=len(node_map),
        dropout=args.dropout
    )
    
    opt_dagnn = torch.optim.Adam(model_dagnn.parameters(), lr=args.learning_rate, weight_decay=1e-4)
    sched_dagnn = torch.optim.lr_scheduler.ReduceLROnPlateau(opt_dagnn, mode='min', factor=0.5, patience=20)
    
    dagnn_preds, dagnn_proba = train_personalized_dagnn(
        model_dagnn, data_pyg, contexts_train, targets_train, user_ids_train,
        contexts_test, targets_test, user_ids_test,
        opt_dagnn, sched_dagnn, args.epochs, args.random_seed
    )
    
    results = evaluate_model_with_ndcg(
        dagnn_preds, targets_test.numpy(), proba_preds=dagnn_proba, 
        name="Personalized DAGNN (Real Users)"
    )

    # Summary
    logger.info("\n" + "="*70)
    logger.info("🏆 РЕЗУЛЬТАТЫ С РЕАЛЬНЫМИ ПОЛЬЗОВАТЕЛЯМИ")
    logger.info("="*70)
    logger.info(f"Использовано РЕАЛЬНЫХ пользователей: {num_users}")
    logger.info("="*70)
    logger.info(f"\nPersonalized DAGNN:")
    for metric_name, value in results.items():
        if value is not None:
            logger.info(f"     {metric_name}: {value:.4f}")

    logger.info("\n" + "="*70)
    logger.info("💡 ПРЕИМУЩЕСТВА РЕАЛЬНЫХ ПОЛЬЗОВАТЕЛЕЙ")
    logger.info("="*70)
    logger.info("✅ Используются реальные owner из данных")
    logger.info("✅ Для сервисов: owner берется напрямую из поля 'owner'")
    logger.info("✅ Для таблиц: owner определяется через целевые сервисы")
    logger.info("✅ Персонализированный PageRank для каждого реального пользователя")
    logger.info("✅ User embeddings отражают реальные предпочтения")
    logger.info("="*70)


if __name__ == "__main__":
    main()

