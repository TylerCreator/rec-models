#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Directed DAG Sequence Models
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Скрипт сравнивает несколько моделей, которые продолжают последовательность
в Directed Acyclic Graph, строго учитывая направленную структуру:

1. Popularity baseline
2. DirectedDAGNN  (APPNP-style propagation c направленными весами)
3. DeepDAG (2022) (depth-aware attention)
4. DAG-GNN (Yu et al., 2019 адаптация) с обучаемыми весами на рёбрах
5. GRU4Rec (маскирует выходы по направлению графа)

Usage (пример):
    python directed_dag_models.py --data compositionsDAG.json --epochs 150
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

import networkx as nx
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, ndcg_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from torch_geometric.data import Data

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("directed_dag_models")

# ---------------------------------------------------------------------------
# Data utilities (match sequence_dag_recommender_final.py)
# ---------------------------------------------------------------------------


def load_dag_from_json(json_path: Path) -> Tuple[nx.DiGraph, List]:
    logger.info(f"Loading DAG from {json_path}")
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    dag = nx.DiGraph()
    for composition in data:
        id_to_mid = {}
        for node in composition["nodes"]:
            node_id = str(node["id"])
            if "mid" in node:
                id_to_mid[node_id] = f"service_{node['mid']}"
            else:
                id_to_mid[node_id] = f"table_{node['id']}"

        for link in composition["links"]:
            source = str(link["source"])
            target = str(link["target"])
            if source not in id_to_mid or target not in id_to_mid:
                continue
            src_node = id_to_mid[source]
            tgt_node = id_to_mid[target]
            dag.add_node(src_node, type='service' if src_node.startswith("service") else 'table')
            dag.add_node(tgt_node, type='service' if tgt_node.startswith("service") else 'table')
            dag.add_edge(src_node, tgt_node)

    logger.info(f"Loaded DAG with {dag.number_of_nodes()} nodes and {dag.number_of_edges()} edges")
    return dag, data


def extract_paths_from_compositions(data: List[dict]) -> List[Tuple[List[str], int]]:
    logger.info("Extracting REAL paths from compositions (not synthetic DFS paths)")
    all_paths = []

    for comp_idx, entry in enumerate(data):
        composition = entry["composition"] if "composition" in entry else entry
        comp_graph = nx.DiGraph()
        id_to_mid = {}
        for node in composition["nodes"]:
            node_id = str(node["id"])
            if "mid" in node:
                node_name = f"service_{node['mid']}"
            else:
                node_name = f"table_{node['id']}"
            id_to_mid[node_id] = node_name

        for link in composition["links"]:
            source = str(link["source"])
            target = str(link["target"])
            if source in id_to_mid and target in id_to_mid:
                comp_graph.add_edge(id_to_mid[source], id_to_mid[target])

        start_nodes = [n for n in comp_graph.nodes() if comp_graph.in_degree(n) == 0]
        end_nodes = [n for n in comp_graph.nodes() if comp_graph.out_degree(n) == 0]
        for start in start_nodes:
            for end in end_nodes:
                try:
                    for path in nx.all_simple_paths(comp_graph, start, end):
                        if len(path) > 1:
                            all_paths.append((path, comp_idx))
                except nx.NetworkXNoPath:
                    continue

    logger.info(f"Extracted {len(all_paths)} REAL paths from {len(data)} compositions")
    return all_paths


def create_training_pairs(paths_with_idx: List[Tuple[List[str], int]]) -> Tuple[List[Tuple[str, ...]], List[str], List[int]]:
    X, y, comp_ids = [], [], []
    for path, comp_idx in paths_with_idx:
        for idx in range(1, len(path)):
            context = tuple(path[:idx])
            target = path[idx]
            if target.startswith("service"):
                X.append(context)
                y.append(target)
                comp_ids.append(comp_idx)
    return X, y, comp_ids


def build_graph(paths: List[List[str]]) -> nx.DiGraph:
    """Build graph with edge weights based on transition frequency in compositions."""
    g = nx.DiGraph()
    edge_counts = defaultdict(int)
    
    # Count how many times each edge appears in paths
    for path in paths:
        for i in range(len(path) - 1):
            edge = (path[i], path[i + 1])
            edge_counts[edge] += 1
    
    # Add edges with their frequency weights
    for (u, v), count in edge_counts.items():
        g.add_edge(u, v, weight=count)
    
    logger.info(f"Built graph with {g.number_of_nodes()} nodes, {g.number_of_edges()} edges")
    logger.info(f"Edge weight range: {min(edge_counts.values())} - {max(edge_counts.values())}")
    
    return g


def build_composition_graphs(compositions: List[dict]) -> Tuple[List[Data], List[Dict[str, int]]]:
    graphs = []
    node_maps = []
    for entry in compositions:
        composition = entry["composition"] if "composition" in entry else entry
        id_to_mid = {}
        for node in composition["nodes"]:
            node_id = str(node["id"])
            if "mid" in node:
                id_to_mid[node_id] = f"service_{node['mid']}"
            else:
                id_to_mid[node_id] = f"table_{node['id']}"
        node_names = list(dict.fromkeys(id_to_mid.values()))
        node_map = {name: idx for idx, name in enumerate(node_names)}
        features = torch.zeros((len(node_names), 2), dtype=torch.float32)
        for name, idx in node_map.items():
            if name.startswith("service"):
                features[idx, 0] = 1.0
            else:
                features[idx, 1] = 1.0
        edges = []
        for link in composition["links"]:
            source = str(link["source"])
            target = str(link["target"])
            if source in id_to_mid and target in id_to_mid:
                src_name = id_to_mid[source]
                tgt_name = id_to_mid[target]
                edges.append([node_map[src_name], node_map[tgt_name]])
        edge_index = torch.tensor(edges, dtype=torch.long).t() if edges else torch.empty((2, 0), dtype=torch.long)
        graphs.append(Data(x=features, edge_index=edge_index))
        node_maps.append(node_map)
    return graphs, node_maps


def prepare_pyg(graph: nx.DiGraph, nodes: List[str]) -> Tuple[Data, Dict[str, int]]:
    encoder = LabelEncoder()
    node_ids = encoder.fit_transform(nodes)
    node_map = {node: idx for node, idx in zip(nodes, node_ids)}
    
    # Extract edges and their weights
    edges = []
    edge_weights = []
    for u, v, data in graph.edges(data=True):
        edges.append([node_map[u], node_map[v]])
        edge_weights.append(data.get('weight', 1.0))  # Default weight is 1.0
    
    edge_index = torch.tensor(edges, dtype=torch.long).t()
    edge_weight = torch.tensor(edge_weights, dtype=torch.float32)
    
    features = torch.zeros((len(nodes), 2), dtype=torch.float32)
    for node, idx in node_map.items():
        if node.startswith("service"):
            features[idx, 0] = 1.0
        else:
            features[idx, 1] = 1.0
    
    data = Data(x=features, edge_index=edge_index, edge_weight=edge_weight)
    return data, node_map


def prepare_sequences(contexts: List[Tuple[str, ...]], targets: List[str],
                      node_map: Dict[str, int], service_map: Dict[str, int],
                      max_len: int = 10) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    sequences, lengths, labels = [], [], []
    for ctx, target in zip(contexts, targets):
        idxs = [node_map[n] + 1 for n in ctx]  # padding=0
        if len(idxs) >= max_len:
            idxs = idxs[-max_len:]
        else:
            idxs = [0] * (max_len - len(idxs)) + idxs
        sequences.append(idxs)
        lengths.append(min(len(ctx), max_len))
        labels.append(service_map[target])
    return (torch.tensor(sequences, dtype=torch.long),
            torch.tensor(lengths, dtype=torch.long),
            torch.tensor(labels, dtype=torch.long))


def split_data(contexts: List[Tuple[str, ...]], targets: List[str], comp_indices: List[int],
               test_size: float, seed: int):
    lb = LabelEncoder().fit(targets)
    y_enc = lb.transform(targets)
    counts = Counter(y_enc)
    min_count = min(counts.values())
    stratify = y_enc if min_count >= 2 else None
    if stratify is None:
        logger.warning("Too few samples for stratified split (min=%d). Using random split.", min_count)
    ctx_train, ctx_test, y_train, y_test, comp_train, comp_test = train_test_split(
        contexts, targets, comp_indices, test_size=test_size, random_state=seed, stratify=stratify
    )
    return ctx_train, ctx_test, y_train, y_test, comp_train, comp_test


# ---------------------------------------------------------------------------
# Loss Functions (from original GRU4Rec)
# ---------------------------------------------------------------------------


class BPRLoss(nn.Module):
    """
    Bayesian Personalized Ranking loss (BPR-max from GRU4Rec).
    Maximizes the difference between positive and negative item scores.
    """
    def forward(self, pos_scores, neg_scores):
        """
        Args:
            pos_scores: Scores for positive items (batch,)
            neg_scores: Scores for negative items (batch, n_neg)
        """
        # BPR-max: maximize difference for hardest negatives
        diff = pos_scores.unsqueeze(1) - neg_scores  # (batch, n_neg)
        loss = -torch.log(torch.sigmoid(diff) + 1e-24).mean()
        return loss


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------


class DirectedAPPNPPropagation(nn.Module):
    def __init__(self, K: int, alpha: float = 0.1):
        super().__init__()
        self.K = K
        self.alpha = alpha

    def forward(self, x, edge_index, edge_weight=None):
        h0 = x
        h = x
        row, col = edge_index
        num_nodes = x.size(0)
        
        # Use edge weights for normalization if provided
        if edge_weight is not None:
            # Compute sum of outgoing edge weights for each node
            weight_sum = torch.zeros(num_nodes, dtype=torch.float32, device=x.device)
            weight_sum.index_add_(0, row, edge_weight)
            weight_sum = weight_sum.clamp(min=1.0)
            # Normalize by weight sum (like weighted out-degree)
            edge_norm = edge_weight / weight_sum[row]
        else:
            # Fall back to degree-based normalization
            deg = torch.bincount(row, minlength=num_nodes).float().clamp(min=1.0).to(x.device)
            edge_norm = 1.0 / deg[row]
        
        for _ in range(self.K):
            messages = h[row] * edge_norm.unsqueeze(-1)
            agg = torch.zeros_like(h).index_add(0, col, messages)
            h = (1 - self.alpha) * agg + self.alpha * h0
        return h


class DirectedDAGNN(nn.Module):
    def __init__(self, in_channels: int, hidden: int, out_channels: int, K: int = 10, dropout: float = 0.4):
        super().__init__()
        self.dropout = dropout
        
        # First encoding layer
        self.lin1 = nn.Linear(in_channels, hidden)
        self.bn1 = nn.BatchNorm1d(hidden)
        
        # Second layer with residual connection (like DAGNNRecommender)
        self.lin2 = nn.Linear(hidden, hidden)
        self.bn2 = nn.BatchNorm1d(hidden)
        
        # Propagation
        self.prop = DirectedAPPNPPropagation(K=K, alpha=0.1)
        self.att = nn.Parameter(torch.ones(K + 1))
        
        # Output head
        self.head = nn.Sequential(
            nn.Linear(hidden, hidden // 2),
            nn.BatchNorm1d(hidden // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden // 2, out_channels)
        )

    def forward(self, x, edge_index, edge_weight=None):
        # First layer
        x = self.lin1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        
        # Residual block (like in DAGNNRecommender)
        identity = x
        x = self.lin2(x)
        x = self.bn2(x)
        x = F.relu(x)
        x = x + identity  # Residual connection
        x = F.dropout(x, p=self.dropout, training=self.training)
        
        # Propagation with attention (using edge weights)
        xs = [x]
        h = x
        for _ in range(self.att.size(0) - 1):
            h = self.prop(h, edge_index, edge_weight)
            h = F.dropout(h, p=self.dropout, training=self.training)
            xs.append(h)
        stacked = torch.stack(xs, dim=-1)
        weights = F.softmax(self.att, dim=0)
        fused = (stacked * weights.view(1, 1, -1)).sum(dim=-1)
        
        return self.head(fused)


class DeepDAGBlock(nn.Module):
    def __init__(self, hidden: int, heads: int = 4, dropout: float = 0.3):
        super().__init__()
        assert hidden % heads == 0
        self.gat = torch.nn.MultiheadAttention(hidden, heads, dropout=dropout, batch_first=True)
        self.norm1 = nn.LayerNorm(hidden)
        self.ffn = nn.Sequential(
            nn.Linear(hidden, hidden * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden * 2, hidden)
        )
        self.norm2 = nn.LayerNorm(hidden)

    def forward(self, x, attn_mask=None):
        # Use attention mask to restrict attention to graph edges only
        h, _ = self.gat(x, x, x, attn_mask=attn_mask, need_weights=False)
        x = self.norm1(x + h)
        h2 = self.ffn(x)
        return self.norm2(x + h2)


class DeepDAGEncoder(nn.Module):
    def __init__(self, in_channels: int, hidden: int, depth_emb: int = 16,
                 num_layers: int = 3, heads: int = 4, dropout: float = 0.3):
        super().__init__()
        self.depth_encoder = nn.Sequential(
            nn.Linear(1, depth_emb),
            nn.GELU(),
            nn.Linear(depth_emb, depth_emb),
        )
        self.input_proj = nn.Linear(in_channels + depth_emb, hidden)
        self.blocks = nn.ModuleList([DeepDAGBlock(hidden, heads, dropout) for _ in range(num_layers)])

    def forward(self, x, depth, attn_mask=None):
        depth_feat = self.depth_encoder(depth.unsqueeze(-1))
        h = self.input_proj(torch.cat([x, depth_feat], dim=-1))
        h = h.unsqueeze(0)  # treat nodes as sequence (1, N, H)
        for block in self.blocks:
            h = block(h, attn_mask)
        return h.squeeze(0)


class DeepDAGRecommender(nn.Module):
    def __init__(self, in_channels: int, hidden: int, out_channels: int,
                 num_layers: int = 3, heads: int = 4, dropout: float = 0.3):
        super().__init__()
        self.encoder = DeepDAGEncoder(in_channels, hidden, num_layers=num_layers, heads=heads, dropout=dropout)
        self.head = nn.Sequential(
            nn.Linear(hidden, hidden // 2),
            nn.LayerNorm(hidden // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden // 2, out_channels)
        )

    def create_attention_mask(self, edge_index, num_nodes, device):
        """
        Create attention mask from edge_index.
        Mask allows attention only along graph edges (and self-loops).
        Returns mask of shape (num_nodes, num_nodes) where:
        - 0.0 = allowed attention
        - -inf = blocked attention
        """
        # Initialize mask: block all connections
        mask = torch.full((num_nodes, num_nodes), float('-inf'), device=device)
        
        # Allow self-attention (each node can attend to itself)
        mask.fill_diagonal_(0.0)
        
        # Allow attention along edges (src -> dst)
        if edge_index.numel() > 0:
            src, dst = edge_index
            mask[dst, src] = 0.0  # dst can attend to src
        
        return mask

    def forward(self, x, edge_index):
        depth = compute_normalized_depth(edge_index, x.size(0), x.device)
        # Create attention mask to restrict attention to graph structure
        attn_mask = self.create_attention_mask(edge_index, x.size(0), x.device)
        h = self.encoder(x, depth, attn_mask)
        return self.head(h)


def compute_normalized_depth(edge_index: torch.Tensor, num_nodes: int, device: torch.device) -> torch.Tensor:
    graph = [[] for _ in range(num_nodes)]
    indeg = [0] * num_nodes
    row, col = edge_index
    for u, v in zip(row.tolist(), col.tolist()):
        graph[u].append(v)
        indeg[v] += 1
    depth = [0] * num_nodes
    queue = [i for i in range(num_nodes) if indeg[i] == 0]
    for node in queue:
        for child in graph[node]:
            depth[child] = max(depth[child], depth[node] + 1)
            indeg[child] -= 1
            if indeg[child] == 0:
                queue.append(child)
    depth_tensor = torch.tensor(depth, dtype=torch.float32, device=device)
    max_depth = depth_tensor.max().clamp(min=1.0)
    return depth_tensor / max_depth


class DAGGNNLayer(nn.Module):
    """
    Реализация DAG-GNN (Yu et al., 2019) с фиксированной структурой графа:
    - отдельные матрицы для сообщений от родителей и детей;
    - аддитивный self-term и LayerNorm.
    """

    def __init__(self, hidden: int):
        super().__init__()
        self.parent_weight = nn.Parameter(torch.randn(hidden, hidden) * 0.02)
        self.child_weight = nn.Parameter(torch.randn(hidden, hidden) * 0.02)
        self.self_lin = nn.Linear(hidden, hidden)
        self.norm = nn.LayerNorm(hidden)
        self.act = nn.GELU()

    def forward(self, h, edge_index, rev_edge_index, dropout: float, training: bool):
        parent_msg = self.aggregate(h, edge_index, self.parent_weight)
        child_msg = self.aggregate(h, rev_edge_index, self.child_weight)
        out = self.self_lin(h) + parent_msg + child_msg
        out = self.act(out)
        out = F.dropout(out, p=dropout, training=training)
        return self.norm(h + out)

    @staticmethod
    def aggregate(h, edge_index, weight):
        if edge_index.numel() == 0:
            return torch.zeros_like(h)
        src, dst = edge_index
        transformed = h[src] @ weight
        agg = torch.zeros_like(h)
        agg.index_add_(0, dst, transformed)
        deg = torch.bincount(dst, minlength=h.size(0)).clamp(min=1).unsqueeze(-1).to(h.device)
        return agg / deg


class DAGGNNRecommender(nn.Module):
    def __init__(self, in_channels: int, hidden: int, out_channels: int, edge_index: torch.Tensor,
                 num_layers: int = 3, dropout: float = 0.3):
        super().__init__()
        self.input_proj = nn.Linear(in_channels, hidden)
        self.layers = nn.ModuleList([DAGGNNLayer(hidden) for _ in range(num_layers)])
        self.dropout = dropout
        self.edge_index = edge_index
        self.rev_edge_index = edge_index.flip(0)
        self.head = nn.Sequential(
            nn.Linear(hidden, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, out_channels)
        )

    def forward(self, x, edge_index=None):
        if edge_index is None:
            edge_index = self.edge_index
            rev_edge_index = self.rev_edge_index
        else:
            rev_edge_index = edge_index.flip(0)
        h = self.input_proj(x)
        for layer in self.layers:
            h = layer(h, edge_index, rev_edge_index, self.dropout, self.training)
        return self.head(h)


class GRU4Rec(nn.Module):
    """
    GRU4Rec with techniques from the original ICLR 2016 paper.
    Enhanced with:
    - Separate dropout for embeddings and hidden layers
    - Support for BPR-max and TOP1-max losses
    - DAG structure awareness through output masking
    """
    def __init__(self, num_nodes: int, num_services: int, embedding_dim: int = 64, hidden: int = 128, num_layers: int = 2,
                 dropout_embed: float = 0.25, dropout_hidden: float = 0.4, 
                 dag_successors: Dict[int, List[int]] = None,
                 dag_successor_nodes: Dict[int, List[int]] = None):
        super().__init__()
        self.embedding = nn.Embedding(num_nodes + 1, embedding_dim, padding_idx=0)
        self.dropout_embed = dropout_embed
        self.dropout_hidden = dropout_hidden
        
        # GRU with dropout only between layers (not on output)
        self.gru = nn.GRU(embedding_dim, hidden, num_layers=num_layers, 
                         batch_first=True, dropout=dropout_hidden if num_layers > 1 else 0)
        
        self.fc = nn.Linear(hidden, num_services)
        self.dag_successors = dag_successors or {}
        self.num_services = num_services

    def forward(self, sequences, lengths, last_nodes=None, compute_scores=False):
        """
        Args:
            sequences: (batch, seq_len) node indices
            lengths: (batch,) sequence lengths
            last_nodes: (batch,) last node indices for DAG masking
            compute_scores: If True, return raw scores instead of logits (for BPR/TOP1)
        """
        # Embedding with separate dropout
        emb = self.embedding(sequences)
        emb = F.dropout(emb, p=self.dropout_embed, training=self.training)
        
        # GRU processing
        gru_out, _ = self.gru(emb)
        
        # Additional dropout on GRU output
        gru_out = F.dropout(gru_out, p=self.dropout_hidden, training=self.training)
        
        # Get last hidden state
        last_hidden = gru_out[torch.arange(gru_out.size(0)), lengths - 1]
        
        # Compute logits/scores
        logits = self.fc(last_hidden)
        
        # DAG structure masking
        if last_nodes is not None and not compute_scores:
            mask = torch.zeros_like(logits) - 1e9
            for idx, node in enumerate(last_nodes.tolist()):
                succ = self.dag_successors.get(node, [])
                if succ:
                    mask[idx, succ] = 0.0
            logits = logits + mask
        
        return logits


class PerDAGGRU(nn.Module):
    def __init__(self, graph_in_channels: int, graph_hidden: int, seq_hidden: int,
                 out_channels: int, max_len: int = 10, num_layers: int = 1, dropout: float = 0.3):
        super().__init__()
        self.graph_encoder = DeepDAGEncoder(
            in_channels=graph_in_channels,
            hidden=graph_hidden,
            depth_emb=graph_hidden // 8,
            num_layers=3,
            heads=4,
            dropout=dropout
        )
        self.max_len = max_len
        self.gru = nn.GRU(graph_hidden, seq_hidden, num_layers=num_layers, batch_first=True, dropout=dropout)
        self.head = nn.Linear(seq_hidden, out_channels)

    def encode_graph(self, data: Data):
        depth = compute_normalized_depth(data.edge_index, data.x.size(0), data.x.device)
        return self.graph_encoder(data.x, depth)

    def build_sequence_tensor(self, contexts: List[Tuple[str, ...]], node_map: Dict[str, int],
                              embeddings: torch.Tensor):
        sequences = []
        lengths = []
        kept_indices = []
        hidden_dim = embeddings.size(1)
        zero_vec = torch.zeros(hidden_dim, dtype=embeddings.dtype, device=embeddings.device)
        for idx, ctx in enumerate(contexts):
            emb_list = []
            valid = True
            for node in ctx:
                if node not in node_map:
                    valid = False
                    break
                emb_list.append(embeddings[node_map[node]])
            if not valid or not emb_list:
                continue
            emb_list = emb_list[-self.max_len:]
            lengths.append(len(emb_list))
            if len(emb_list) < self.max_len:
                pad = [zero_vec] * (self.max_len - len(emb_list))
                emb_list = pad + emb_list
            sequences.append(torch.stack(emb_list, dim=0))
            kept_indices.append(idx)
        if not sequences:
            return None, None, None
        seq_tensor = torch.stack(sequences, dim=0)
        len_tensor = torch.tensor(lengths, dtype=torch.long, device=embeddings.device)
        return seq_tensor, len_tensor, kept_indices

    def forward(self, seq_batch, lengths):
        gru_out, _ = self.gru(seq_batch)
        last_hidden = gru_out[torch.arange(gru_out.size(0)), lengths - 1]
        return self.head(last_hidden)


# ---------------------------------------------------------------------------
# Training helpers
# ---------------------------------------------------------------------------


def train_graph_model(model, data_pyg, train_idx, test_idx, targets_train, targets_test,
                      optimizer, epochs: int, name: str):
    import time
    criterion = nn.CrossEntropyLoss()
    logger.info("Training %s ...", name)
    
    # Check if model supports edge_weight (DirectedDAGNN does, DAGGNNRecommender doesn't)
    edge_weight = data_pyg.edge_weight if hasattr(data_pyg, 'edge_weight') else None
    use_edge_weight = edge_weight is not None and isinstance(model, DirectedDAGNN)
    
    for epoch in range(epochs):
        epoch_start = time.time()
        model.train()
        optimizer.zero_grad()
        if use_edge_weight:
            logits = model(data_pyg.x, data_pyg.edge_index, edge_weight)[train_idx]
        else:
            logits = model(data_pyg.x, data_pyg.edge_index)[train_idx]
        loss = criterion(logits, targets_train)
        loss.backward()
        optimizer.step()
        epoch_time = time.time() - epoch_start
        if (epoch + 1) % 10 == 0:
            logger.info("%s Epoch %d/%d loss=%.4f time=%.2fs", name, epoch + 1, epochs, loss.item(), epoch_time)
    model.eval()
    with torch.no_grad():
        if use_edge_weight:
            logits = model(data_pyg.x, data_pyg.edge_index, edge_weight)[test_idx]
        else:
            logits = model(data_pyg.x, data_pyg.edge_index)[test_idx]
        preds = logits.argmax(dim=1)
        probs = F.softmax(logits, dim=1)
    return compute_metrics(preds.numpy(), targets_test.numpy(), probs.numpy(), name)


def train_deepdag_per_composition(model, comp_graphs, comp_node_maps,
                                  contexts_train, contexts_test,
                                  comp_indices_train, comp_indices_test,
                                  targets_train_indices, targets_test_indices,
                                  optimizer, epochs: int, name: str):
    criterion = nn.CrossEntropyLoss()
    comp_to_train = defaultdict(list)
    for idx, comp_idx in enumerate(comp_indices_train):
        comp_to_train[comp_idx].append(idx)

    # Pre-compute depth for all compositions to avoid recomputing every forward pass
    logger.info("Pre-computing depths for %d compositions...", len(comp_graphs))
    comp_depths = []
    for data in comp_graphs:
        depth = compute_normalized_depth(data.edge_index, data.x.size(0), data.x.device)
        comp_depths.append(depth)

    logger.info("Training %s (per-composition) with %d compositions...", name, len(comp_to_train))
    import time
    for epoch in range(epochs):
        epoch_start = time.time()
        model.train()
        total_loss = 0.0
        total_samples = 0
        for comp_idx, sample_indices in comp_to_train.items():
            data = comp_graphs[comp_idx]
            # Use pre-computed depth instead of recomputing
            depth = comp_depths[comp_idx]
            # Forward pass with cached depth
            h = model.encoder(data.x, depth)
            logits = model.head(h)
            
            node_ids, label_ids = [], []
            node_map = comp_node_maps[comp_idx]
            for sample_idx in sample_indices:
                node_name = contexts_train[sample_idx][-1]
                if node_name not in node_map:
                    continue
                node_ids.append(node_map[node_name])
                label_ids.append(targets_train_indices[sample_idx])
            if not node_ids:
                continue
            node_tensor = torch.tensor(node_ids, dtype=torch.long)
            label_tensor = torch.tensor(label_ids, dtype=torch.long)
            batch_logits = logits[node_tensor]
            loss = criterion(batch_logits, label_tensor)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * len(node_ids)
            total_samples += len(node_ids)
        if total_samples == 0:
            logger.warning("%s training skipped (no samples).", name)
            break
        epoch_time = time.time() - epoch_start
        # Log every 10 epochs instead of every 50 to show progress
        if (epoch + 1) % 10 == 0:
            logger.info("%s Epoch %d/%d loss=%.4f time=%.2fs", name, epoch + 1, epochs, total_loss / total_samples, epoch_time)

    model.eval()
    preds_list, probs_list, labels_list = [], [], []
    with torch.no_grad():
        for idx, comp_idx in enumerate(comp_indices_test):
            data = comp_graphs[comp_idx]
            # Use pre-computed depth for evaluation as well
            depth = comp_depths[comp_idx]
            h = model.encoder(data.x, depth)
            logits = model.head(h)
            
            node_map = comp_node_maps[comp_idx]
            node_name = contexts_test[idx][-1]
            if node_name not in node_map:
                continue
            node_id = node_map[node_name]
            logit = logits[node_id].unsqueeze(0)
            prob = F.softmax(logit, dim=1)
            preds_list.append(logit.argmax(dim=1).cpu().numpy())
            probs_list.append(prob.cpu().numpy())
            labels_list.append(targets_test_indices[idx])

    if not preds_list:
        raise RuntimeError(f"No valid test samples for {name}")

    preds = np.concatenate(preds_list, axis=0)
    probs = np.concatenate(probs_list, axis=0)
    labels = np.array(labels_list)
    metrics = compute_metrics(preds, labels, probs, name)
    return metrics


def sample_negatives(targets, num_classes, n_sample, sample_alpha=0.75, item_popularity=None):
    """
    Sample negative items using popularity-based sampling (like original GRU4Rec).
    
    Args:
        targets: Positive target indices (batch,)
        num_classes: Total number of classes
        n_sample: Number of negative samples per positive
        sample_alpha: Sampling temperature (0=uniform, 1=popularity)
        item_popularity: Popularity counts for each class (num_classes,)
    
    Returns:
        Negative sample indices (batch, n_sample)
    """
    batch_size = targets.size(0)
    
    if item_popularity is None or sample_alpha == 0:
        # Uniform sampling
        neg_samples = torch.randint(0, num_classes, (batch_size, n_sample), device=targets.device)
    else:
        # Popularity-based sampling: prob ~ popularity^sample_alpha
        probs = item_popularity.float() ** sample_alpha
        probs = probs / probs.sum()
        neg_samples = torch.multinomial(probs, batch_size * n_sample, replacement=True)
        neg_samples = neg_samples.view(batch_size, n_sample)
    
    return neg_samples


def train_gru_model(model: GRU4Rec, seq_train, len_train, seq_test, len_test,
                    targets_train, targets_test, last_nodes_train, last_nodes_test,
                    optimizer, epochs: int, loss_type='ce', n_sample=0, sample_alpha=0.75):
    """
    Train GRU4Rec with original techniques.
    
    Args:
        loss_type: 'ce' (cross-entropy) or 'bpr' (BPR-max)
        n_sample: Number of negative samples (0 = use only in-batch negatives)
        sample_alpha: Sampling exponent for popularity-based sampling
    """
    import time
    
    # Initialize loss function
    if loss_type == 'bpr':
        criterion = BPRLoss()
        logger.info("Using BPR-max loss with %d negative samples", n_sample if n_sample > 0 else "in-batch")
    else:
        criterion = nn.CrossEntropyLoss()
        logger.info("Using Cross-Entropy loss")
    
    # Compute item popularity for sampling
    item_popularity = None
    if n_sample > 0 and sample_alpha > 0:
        item_popularity = torch.bincount(targets_train, minlength=model.num_services)
        logger.info("Using popularity-based sampling with alpha=%.2f", sample_alpha)
    
    logger.info("Training GRU4Rec ...")
    for epoch in range(epochs):
        epoch_start = time.time()
        model.train()
        optimizer.zero_grad()
        
        if loss_type == 'bpr' and n_sample > 0:
            # BPR loss with negative sampling
            logits = model(seq_train, len_train, last_nodes_train, compute_scores=True)
            
            # Get positive scores
            pos_scores = logits[torch.arange(logits.size(0)), targets_train]
            
            # Sample negatives
            neg_indices = sample_negatives(targets_train, model.num_services, n_sample, 
                                          sample_alpha, item_popularity)
            neg_scores = logits.gather(1, neg_indices)
            
            loss = criterion(pos_scores, neg_scores)
        else:
            # Cross-entropy loss (standard)
            logits = model(seq_train, len_train, last_nodes_train)
            loss = criterion(logits, targets_train)
        
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        epoch_time = time.time() - epoch_start
        
        if (epoch + 1) % 10 == 0:
            logger.info("GRU4Rec Epoch %d/%d loss=%.4f time=%.2fs", epoch + 1, epochs, loss.item(), epoch_time)
    
    # Evaluation
    model.eval()
    with torch.no_grad():
        logits = model(seq_test, len_test, last_nodes_test)
        preds = logits.argmax(dim=1)
        probs = F.softmax(logits, dim=1)
    return compute_metrics(preds.numpy(), targets_test.numpy(), probs.numpy(), "GRU4Rec")


def train_per_dag_gru(model: PerDAGGRU, comp_graphs, comp_node_maps,
                      train_samples_by_comp, test_samples_by_comp,
                      optimizer, epochs: int, name: str, device: torch.device):
    import time
    criterion = nn.CrossEntropyLoss()
    logger.info("Training %s with %d compositions...", name, len(train_samples_by_comp))
    model.to(device)

    for epoch in range(epochs):
        epoch_start = time.time()
        model.train()
        total_loss = 0.0
        total_count = 0
        for comp_idx, samples in train_samples_by_comp.items():
            if not samples:
                continue
            data = comp_graphs[comp_idx]
            embeddings = model.encode_graph(data).to(device)
            contexts = [ctx for ctx, _ in samples]
            targets = torch.tensor([t for _, t in samples], dtype=torch.long)
            seqs, lens, kept = model.build_sequence_tensor(contexts, comp_node_maps[comp_idx], embeddings)
            if seqs is None:
                continue
            seqs = seqs.to(device)
            lens = lens.to(device)
            target_tensor = targets[kept].to(device)
            optimizer.zero_grad()
            logits = model(seqs, lens)
            loss = criterion(logits, target_tensor)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += loss.item() * target_tensor.size(0)
            total_count += target_tensor.size(0)
        if total_count == 0:
            logger.warning("No samples for %s training.", name)
            break
        epoch_time = time.time() - epoch_start
        if (epoch + 1) % 10 == 0:
            logger.info("%s Epoch %d/%d loss=%.4f time=%.2fs", name, epoch + 1, epochs, total_loss / total_count, epoch_time)

    model.eval()
    preds_list, probs_list, labels_list = [], [], []
    with torch.no_grad():
        for comp_idx, samples in test_samples_by_comp.items():
            if not samples:
                continue
            data = comp_graphs[comp_idx]
            embeddings = model.encode_graph(data).to(device)
            contexts = [ctx for ctx, _ in samples]
            targets = torch.tensor([t for _, t in samples], dtype=torch.long)
            seqs, lens, kept = model.build_sequence_tensor(contexts, comp_node_maps[comp_idx], embeddings)
            if seqs is None:
                continue
            logits = model(seqs.to(device), lens.to(device))
            probs = F.softmax(logits, dim=1).cpu().numpy()
            preds = logits.argmax(dim=1).cpu().numpy()
            labels = targets[kept].numpy()
            preds_list.append(preds)
            probs_list.append(probs)
            labels_list.append(labels)

    if not preds_list:
        raise RuntimeError(f"No valid test samples for {name}")

    preds = np.concatenate(preds_list, axis=0)
    probs = np.concatenate(probs_list, axis=0)
    labels = np.concatenate(labels_list, axis=0)
    return compute_metrics(preds, labels, probs, name)


def compute_metrics(preds, labels, probs, name: str) -> Dict[str, float]:
    metrics = {
        "accuracy": accuracy_score(labels, preds),
        "f1": f1_score(labels, preds, average="macro", zero_division=0),
        "precision": precision_score(labels, preds, average="macro", zero_division=0),
        "recall": recall_score(labels, preds, average="macro", zero_division=0),
        "ndcg": ndcg_score(np.eye(probs.shape[1])[labels], probs),
    }
    logger.info("%s metrics: %s", name, metrics)
    return metrics


def popularity_baseline(targets_train, targets_test, num_classes, name="Popularity"):
    counter = Counter(targets_train.numpy())
    top_label = counter.most_common(1)[0][0]
    preds = np.full_like(targets_test.numpy(), top_label)
    probs = np.zeros((len(preds), num_classes))
    probs[:, top_label] = 1.0
    return compute_metrics(preds, targets_test.numpy(), probs, name)


# ---------------------------------------------------------------------------
# Main experiment runner
# ---------------------------------------------------------------------------


def main(args):
    dag, compositions = load_dag_from_json(Path(args.data))
    paths_with_idx = extract_paths_from_compositions(compositions)
    contexts, targets, comp_indices = create_training_pairs(paths_with_idx)
    
    logger.info(f"Total training pairs created: {len(contexts)}")
    
    ctx_train, ctx_test, y_train, y_test, comp_train_idx, comp_test_idx = split_data(
        contexts, targets, comp_indices, args.test_size, args.seed
    )
    
    logger.info(f"Train set size: {len(ctx_train)} samples ({len(ctx_train)/len(contexts)*100:.1f}%)")
    logger.info(f"Test set size: {len(ctx_test)} samples ({len(ctx_test)/len(contexts)*100:.1f}%)")

    paths_only = [p for p, _ in paths_with_idx]
    graph = build_graph(paths_only)
    nodes = list(graph.nodes())
    data_pyg, node_map = prepare_pyg(graph, nodes)
    comp_graphs, comp_node_maps = build_composition_graphs(compositions)

    services = sorted({y for y in targets})
    service_map = {svc: idx for idx, svc in enumerate(services)}
    logger.info(f"Number of unique services (classes): {len(services)}")
    
    targets_tensor = torch.tensor([service_map[y] for y in targets], dtype=torch.long)

    train_idx = torch.tensor([node_map[ctx[-1]] for ctx in ctx_train], dtype=torch.long)
    test_idx = torch.tensor([node_map[ctx[-1]] for ctx in ctx_test], dtype=torch.long)
    targets_train = torch.tensor([service_map[y] for y in y_train], dtype=torch.long)
    targets_test = torch.tensor([service_map[y] for y in y_test], dtype=torch.long)

    successors = defaultdict(list)
    successor_nodes = defaultdict(list)
    for u, v in graph.edges():
        if v.startswith("service"):
            successors[node_map[u]].append(service_map[v])
        successor_nodes[node_map[u]].append(node_map[v])

    seq_all, len_all, labels_all = prepare_sequences(contexts, targets, node_map, service_map, max_len=args.max_len)
    label_counts = Counter(labels_all.numpy())
    min_seq_count = min(label_counts.values())
    stratify_seq = labels_all.numpy() if min_seq_count >= 2 else None
    if stratify_seq is None:
        logger.warning("Too few samples for stratified sequential split (min=%d). Using random split.", min_seq_count)
    seq_train, seq_test, len_train, len_test, lab_train, lab_test = train_test_split(
        seq_all, len_all, labels_all,
        test_size=args.test_size,
        random_state=args.seed,
        stratify=stratify_seq
    )
    last_nodes_train = torch.tensor([node_map[ctx[-1]] for ctx in ctx_train], dtype=torch.long)
    last_nodes_test = torch.tensor([node_map[ctx[-1]] for ctx in ctx_test], dtype=torch.long)

    train_samples_by_comp = defaultdict(list)
    test_samples_by_comp = defaultdict(list)
    train_target_indices = [service_map[y] for y in y_train]
    test_target_indices = [service_map[y] for y in y_test]
    for ctx, comp_idx, target_idx in zip(ctx_train, comp_train_idx, train_target_indices):
        train_samples_by_comp[comp_idx].append((ctx, target_idx))
    for ctx, comp_idx, target_idx in zip(ctx_test, comp_test_idx, test_target_indices):
        test_samples_by_comp[comp_idx].append((ctx, target_idx))

    results = {}
    results["Popularity"] = popularity_baseline(targets_train, targets_test, len(service_map))

    directed_dagnn = DirectedDAGNN(in_channels=2, hidden=args.hidden, out_channels=len(service_map),
                                   K=args.K, dropout=args.dropout)
    opt_dagnn = torch.optim.Adam(directed_dagnn.parameters(), lr=args.lr)
    results["DirectedDAGNN"] = train_graph_model(
        directed_dagnn, data_pyg, train_idx, test_idx, targets_train, targets_test, opt_dagnn, args.epochs, "DirectedDAGNN"
    )

    # DeepDAG2022 - now using global graph like other GNN models
    deepdag = DeepDAGRecommender(in_channels=2, hidden=args.hidden * 2, out_channels=len(service_map),
                                 num_layers=3, heads=4, dropout=args.dropout)
    opt_deepdag = torch.optim.Adam(deepdag.parameters(), lr=args.lr * 0.8)
    results["DeepDAG2022"] = train_graph_model(
        deepdag, data_pyg, train_idx, test_idx, targets_train, targets_test, opt_deepdag, args.epochs, "DeepDAG2022"
    )

    daggnn = DAGGNNRecommender(in_channels=2, hidden=args.hidden, out_channels=len(service_map),
                               edge_index=data_pyg.edge_index, num_layers=3, dropout=args.dropout)
    opt_daggnn = torch.optim.Adam(daggnn.parameters(), lr=args.lr * 0.5)
    results["DAG-GNN"] = train_graph_model(
        daggnn, data_pyg, train_idx, test_idx, targets_train, targets_test, opt_daggnn, args.epochs, "DAG-GNN"
    )

    # GRU4Rec with original techniques
    gru_model = GRU4Rec(
        num_nodes=len(node_map), 
        num_services=len(service_map),
        embedding_dim=64, 
        hidden=args.hidden * 2, 
        num_layers=2,
        dropout_embed=args.dropout_embed,
        dropout_hidden=args.dropout_hidden,
        dag_successors=successors,
        dag_successor_nodes=successor_nodes
    )
    opt_gru = torch.optim.Adam(gru_model.parameters(), lr=args.lr)
    results["GRU4Rec"] = train_gru_model(
        gru_model, seq_train, len_train, seq_test, len_test, lab_train, lab_test,
        last_nodes_train, last_nodes_test, opt_gru, args.epochs,
        loss_type=args.loss, n_sample=args.n_sample, sample_alpha=args.sample_alpha
    )
    
    # PerDAG-GRU - COMMENTED OUT (slow)
    # per_dag_gru = PerDAGGRU(
    #     graph_in_channels=2,
    #     graph_hidden=args.hidden * 2,
    #     seq_hidden=args.hidden * 2,
    #     out_channels=len(service_map),
    #     max_len=args.max_len,
    #     num_layers=2,
    #     dropout=args.dropout
    # )
    # opt_per_dag = torch.optim.Adam(per_dag_gru.parameters(), lr=args.lr * 0.8)
    # results["PerDAG-GRU"] = train_per_dag_gru(
    #     per_dag_gru,
    #     comp_graphs,
    #     comp_node_maps,
    #     train_samples_by_comp,
    #     test_samples_by_comp,
    #     opt_per_dag,
    #     args.epochs,
    #     "PerDAG-GRU",
    #     torch.device("cpu")
    # )

    print("\n=== SUMMARY ===")
    for name, metrics in results.items():
        print(f"{name:15s} | acc={metrics['accuracy']:.4f} | ndcg={metrics['ndcg']:.4f} | f1={metrics['f1']:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Directed DAG sequence models comparison")
    parser.add_argument("--data", type=str, default="compositionsDAG.json")
    parser.add_argument("--epochs", type=int, default=150)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--hidden", type=int, default=64)
    parser.add_argument("--dropout", type=float, default=0.4)
    parser.add_argument("--test-size", type=float, default=0.3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-len", type=int, default=10)
    parser.add_argument("--K", type=int, default=10, help="Propagation steps for DirectedDAGNN")
    
    # GRU4Rec original techniques
    parser.add_argument("--loss", type=str, default="ce", choices=["ce", "bpr"],
                       help="Loss function: ce (cross-entropy), bpr (BPR-max)")
    parser.add_argument("--n-sample", type=int, default=0,
                       help="Number of negative samples (0 = use only in-batch negatives)")
    parser.add_argument("--sample-alpha", type=float, default=0.75,
                       help="Sampling exponent for popularity-based sampling (0=uniform, 1=popularity)")
    parser.add_argument("--dropout-embed", type=float, default=0.25,
                       help="Dropout rate for embeddings")
    parser.add_argument("--dropout-hidden", type=float, default=0.4,
                       help="Dropout rate for hidden layers")
    
    args = parser.parse_args()
    main(args)


