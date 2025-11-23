#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
BTR-style Gated Graph Neural Network for DAG workflows.

Implementation is aligned with the Bioinformatics Tool Recommendation (BTR)
approach described in:
    - BTR: A Bioinformatics Tool Recommendation System (ResearchGate, 2023)
    - https://github.com/ryangreenj/bioinformatics_tool_recommendation

The code adapts the original idea to the compositionsDAG.json dataset: each
path extracted from the directed compositions graph is converted into a
partial graph with a masked node, and a gated message-passing network
predicts the next service.
"""
from __future__ import annotations

import argparse
import json
import logging
import math
import random
from collections import Counter, defaultdict
from copy import deepcopy
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import networkx as nx
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import ndcg_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import torch_geometric
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from tqdm.auto import tqdm

log = logging.getLogger("btr_workflow")


# ---------------------------------------------------------------------------
# Data utilities (adapted from directed_dag_models.py)
# ---------------------------------------------------------------------------


def load_compositions(json_path: Path) -> List[dict]:
    with json_path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    log.info("Loaded %d compositions from %s", len(data), json_path)
    return data


def extract_paths_from_compositions(data: List[dict]) -> List[Tuple[List[str], int]]:
    log.info("Extracting simple paths from %d compositions", len(data))
    all_paths: List[Tuple[List[str], int]] = []

    for comp_idx, entry in enumerate(data):
        composition = entry["composition"] if "composition" in entry else entry
        comp_graph = nx.DiGraph()
        id_to_mid: Dict[str, str] = {}
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

    log.info("Extracted %d paths", len(all_paths))
    return all_paths


def create_training_pairs(
    paths_with_idx: List[Tuple[List[str], int]]
) -> Tuple[List[Tuple[str, ...]], List[str], List[int]]:
    contexts: List[Tuple[str, ...]] = []
    targets: List[str] = []
    comp_ids: List[int] = []
    for path, comp_idx in paths_with_idx:
        for idx in range(1, len(path)):
            context = tuple(path[:idx])
            target = path[idx]
            if target.startswith("service"):
                contexts.append(context)
                targets.append(target)
                comp_ids.append(comp_idx)
    return contexts, targets, comp_ids


def split_contexts(
    contexts: Sequence[Tuple[str, ...]],
    targets: Sequence[str],
    test_size: float,
    seed: int,
) -> Tuple[List[Tuple[str, ...]], List[Tuple[str, ...]], List[str], List[str]]:
    lb = LabelEncoder().fit(targets)
    y_enc = lb.transform(targets)
    counts = Counter(y_enc)
    stratify = y_enc if counts and min(counts.values()) >= 2 else None
    if stratify is None:
        log.warning(
            "Too few samples for stratified split (min=%d). Falling back to random split.",
            min(counts.values()) if counts else 0,
        )
    ctx_train, ctx_test, y_train, y_test = train_test_split(
        contexts,
        targets,
        test_size=test_size,
        random_state=seed,
        stratify=stratify,
    )
    return list(ctx_train), list(ctx_test), list(y_train), list(y_test)


def safe_stratify_labels(labels: Sequence[str], split_name: str):
    counts = Counter(labels)
    if not counts:
        return None
    min_count = min(counts.values())
    if min_count < 2:
        log.warning(
            "Skipping stratified %s split: минимум в классе = %d (<2).",
            split_name,
            min_count,
        )
        return None
    return labels


def format_node_name(node: dict) -> str:
    if "mid" in node:
        return f"service_{node['mid']}"
    return f"table_{node['id']}"


def safe_int(value) -> int:
    try:
        return int(str(value))
    except (TypeError, ValueError):
        return 0


def build_node_feature_map(
    paths: Sequence[Sequence[str]], compositions: Sequence[dict]
) -> Tuple[Dict[str, int], Dict[str, np.ndarray]]:
    node_counts: Counter = Counter()
    node_meta: Dict[str, dict] = {}
    max_mid = 1
    max_table_id = 1

    for entry in compositions:
        composition = entry["composition"] if "composition" in entry else entry
        for node in composition["nodes"]:
            name = format_node_name(node)
            node_counts[name] += 1
            node_meta.setdefault(name, node)
            if name.startswith("service") and "mid" in node:
                max_mid = max(max_mid, safe_int(node["mid"]))
            elif "id" in node:
                max_table_id = max(max_table_id, safe_int(node["id"]))

    deg_in: Counter = Counter()
    deg_out: Counter = Counter()
    for path in paths:
        for idx in range(len(path) - 1):
            deg_out[path[idx]] += 1
            deg_in[path[idx + 1]] += 1

    max_in = max(deg_in.values()) if deg_in else 1
    max_out = max(deg_out.values()) if deg_out else 1
    max_count = max(node_counts.values()) if node_counts else 1

    def build_feature(name: str) -> np.ndarray:
        meta = node_meta.get(name, {})
        is_service = 1.0 if name.startswith("service") else 0.0
        is_table = 1.0 - is_service
        indeg = deg_in.get(name, 0) / max_in
        outdeg = deg_out.get(name, 0) / max_out
        freq = math.log1p(node_counts.get(name, 0)) / math.log1p(max_count)
        if is_service:
            norm_id = safe_int(meta.get("mid", 0)) / max_mid
        else:
            norm_id = safe_int(meta.get("id", 0)) / max_table_id
        return np.array([is_service, is_table, indeg, outdeg, freq, norm_id], dtype=np.float32)

    node_features = {name: build_feature(name) for name in node_counts}
    tool_name_to_id = {name: idx for idx, name in enumerate(sorted(node_counts.keys()))}
    log.info(
        "Prepared %d unique nodes (%d services, %d tables)",
        len(tool_name_to_id),
        sum(1 for n in tool_name_to_id if n.startswith("service")),
        sum(1 for n in tool_name_to_id if n.startswith("table")),
    )
    return tool_name_to_id, node_features


def build_graph_dataset(
    contexts: Sequence[Tuple[str, ...]],
    targets: Sequence[str],
    tool_name_to_id: Dict[str, int],
    node_features: Dict[str, np.ndarray],
    feature_dim: int,
    mask_token_id: int,
) -> List[Data]:
    dataset: List[Data] = []
    zero_feat = np.zeros(feature_dim, dtype=np.float32)

    for ctx, target in zip(contexts, targets):
        if target not in tool_name_to_id:
            continue

        rows = []
        senders: List[int] = []
        receivers: List[int] = []

        for node_name in ctx:
            if node_name not in tool_name_to_id:
                continue
            feats = node_features.get(node_name)
            if feats is None:
                continue
            tool_idx = float(tool_name_to_id[node_name])
            rows.append(
                np.concatenate(
                    (np.array([tool_idx], dtype=np.float32), feats),
                    axis=0,
                )
            )

        if not rows:
            continue

        for idx in range(len(rows) - 1):
            senders.append(idx)
            receivers.append(idx + 1)

        last_idx = len(rows) - 1
        mask_features = np.concatenate(
            (np.array([float(mask_token_id)], dtype=np.float32), zero_feat),
            axis=0,
        )
        rows.append(mask_features)
        senders.append(last_idx)
        receivers.append(len(rows) - 1)

        rows_np = np.stack(rows, axis=0)
        x_tensor = torch.from_numpy(rows_np).float()
        if senders:
            edge_index = torch.tensor([senders, receivers], dtype=torch.long)
        else:
            edge_index = torch.empty((2, 0), dtype=torch.long)

        y_tensor = torch.tensor(tool_name_to_id[target], dtype=torch.long)
        dataset.append(Data(x=x_tensor, edge_index=edge_index, y=y_tensor))

    return dataset


# ---------------------------------------------------------------------------
# Model definition (directly adapted from BTR reference code)
# ---------------------------------------------------------------------------


class GatedGraphConv(torch_geometric.nn.conv.MessagePassing):
    def __init__(self, in_channels: int, out_channels: int, aggr: str = "add", **kwargs):
        super().__init__(aggr=aggr, **kwargs)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.gru = nn.GRUCell(in_channels, out_channels, bias=False)

    def forward(self, x, edge_index):
        message = self.propagate(edge_index, x=x, size=None)
        return self.gru(message, x)

    def message(self, x_j):
        return x_j

    def message_and_aggregate(self, adj_t, x):
        return torch.matmul(adj_t, x, reduce=self.aggr)


class GatedGNN(nn.Module):
    def __init__(self, config: Dict[str, float]):
        super().__init__()
        self.hidden_channels = config["hidden_channels"]
        self.num_tools = config["num_tools"]
        self.emb_dropout = config["emb_dropout"]
        self.dropout = config["dropout"]

        if config["model_type"] == "graph":
            self.num_tools += 1  # masked node

        self.combined_channels = self.hidden_channels + config["description_size"]
        self.embedding = nn.Embedding(self.num_tools, self.hidden_channels)
        self.graph = GatedGraphConv(self.combined_channels, self.combined_channels)
        self.dropout_one = nn.Dropout(self.dropout)

        self.linear_one = nn.Linear(self.combined_channels, self.combined_channels, bias=False)
        self.linear_two = nn.Linear(self.combined_channels, self.combined_channels, bias=True)
        self.q = nn.Linear(self.combined_channels, self.combined_channels, bias=True)

        self.linear_transform = nn.Linear(self.combined_channels * 2, self.combined_channels, bias=False)
        self.compress = nn.Linear(self.combined_channels, self.hidden_channels, bias=False)

    def forward(self, data: Data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        emb = self.get_embedding(x)
        h = self.graph(emb, edge_index)
        h = self.dropout_one(h)
        w_l, w_g = self.get_workflow_reps(h, batch)
        w = torch.cat([w_l, w_g], dim=1)
        w = self.linear_transform(w)
        w = self.compress(w)
        logits = torch.matmul(self.embedding.weight, w.T).T
        return logits

    def get_embedding(self, x):
        emb = self.embedding(x[:, 0].long())
        emb = emb.permute(1, 0)
        if emb.dim() == 2:
            emb = nn.functional.dropout(
                emb,
                p=self.emb_dropout,
                training=self.training,
            )
        else:
            emb = nn.functional.dropout2d(
                emb,
                p=self.emb_dropout,
                training=self.training,
            )
        emb = emb.permute(1, 0)
        emb = torch.cat([emb, x[:, 1:]], dim=1)
        return emb

    def get_workflow_reps(self, h, batch):
        split_sections = list(torch.bincount(batch).cpu())
        h_split = torch.split(h, split_sections, dim=0)
        w_l = torch.stack([h_split[i][-1] for i in range(len(h_split))])

        w_g_r = torch.cat([h_split[i][-1].repeat(len(h_split[i]), 1) for i in range(len(h_split))])
        q1 = self.linear_one(w_g_r)
        q2 = self.linear_two(h)
        alpha = self.q(torch.sigmoid(q1 + q2))
        a = alpha * h

        a_split = torch.split(a, split_sections, dim=0)
        w_g = torch.stack([torch.sum(a_split[i], dim=0) for i in range(len(a_split))])
        return w_l, w_g


# ---------------------------------------------------------------------------
# Training / evaluation helpers
# ---------------------------------------------------------------------------


def get_metrics(logits, data: Data, config: Dict, all_metrics: bool = True, by_length=None):
    pred = torch.argmax(logits, dim=1)
    acc_by_item = pred == data.y
    acc = torch.sum(acc_by_item).item() / len(data.y)
    top_k = 0.0
    mrr_k = 0.0
    ndcg = 0.0

    if all_metrics:
        k = min(config["top_k"], logits.size(1))
        mrr_limit = min(config["mrr_k"], logits.size(1))
        ndcg_limit = min(config["ndcg_k"], logits.size(1))

        graph_lengths = torch.bincount(data.batch)
        if config["model_type"] == "graph":
            graph_lengths = graph_lengths - 1

        top_k_preds = torch.topk(logits, k=k, dim=1)[1]
        top_k_by_item = torch.sum(top_k_preds == data.y.view(-1, 1), dim=1)
        top_k = torch.sum(top_k_by_item).item() / len(data.y)

        mrr_top_k_preds = torch.topk(logits, k=mrr_limit, dim=1)[1]
        mrr_top_k_by_item = torch.sum(mrr_top_k_preds == data.y.view(-1, 1), dim=1)
        correct_index = torch.where(mrr_top_k_preds == data.y.view(-1, 1))
        mrr_k_by_item = torch.zeros(len(data.y), dtype=torch.float, device=config["device"])
        if correct_index[0].numel() > 0:
            mrr_k_by_item[correct_index[0]] = 1.0 / (correct_index[1].float() + 1.0)
        mrr_k_by_item = torch.where(
            mrr_top_k_by_item == 0,
            mrr_top_k_by_item.float(),
            mrr_k_by_item,
        )
        mrr_k = torch.sum(mrr_k_by_item).item() / len(data.y)

        y_true = nn.functional.one_hot(data.y, num_classes=logits.size(1)).cpu().numpy()
        y_score = logits.detach().cpu().numpy()
        ndcg = float(ndcg_score(y_true, y_score, k=ndcg_limit))

        if by_length is None:
            by_length = {}
        for i in range(len(graph_lengths)):
            curr_len = graph_lengths[i].item()
            if curr_len not in by_length:
                by_length[curr_len] = {"acc": [], "top_k": [], "mrr_k": [], "ndcg": []}
            by_length[curr_len]["acc"].append(acc_by_item[i].item())
            by_length[curr_len]["top_k"].append(top_k_by_item[i].item())
            by_length[curr_len]["mrr_k"].append(mrr_k_by_item[i].item())
            by_length[curr_len]["ndcg"].append(ndcg)

    return acc, top_k, mrr_k, ndcg, by_length


def evaluate_model(model, loader, config, all_metrics: bool = True, use_tqdm: bool = False):
    model.eval()
    total_acc = 0.0
    total_top_k = 0.0
    total_mrr_k = 0.0
    total_ndcg = 0.0
    by_length = {}

    iterator = tqdm(loader, desc="Eval", leave=False) if use_tqdm else loader
    with torch.no_grad():
        for data in iterator:
            data = data.to(config["device"])
            logits = model(data)
            acc, top_k, mrr_k, ndcg_val, by_length = get_metrics(
                logits, data, config, all_metrics, by_length
            )
            total_acc += acc
            total_top_k += top_k
            total_mrr_k += mrr_k
            total_ndcg += ndcg_val

    if all_metrics:
        for key in list(by_length.keys()):
            for metric, values in by_length[key].items():
                by_length[key][metric] = float(np.mean(values)) if values else 0.0
        size = max(len(loader), 1)
        return (
            total_acc / size,
            total_top_k / size,
            total_mrr_k / size,
            total_ndcg / size,
            by_length,
        )
    size = max(len(loader), 1)
    return total_acc / size


def train_btr_model(train_loader, val_loader, config, epochs: int):
    model = GatedGNN(config).to(config["device"])
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config["learning_rate"],
        weight_decay=config["weight_decay"],
    )
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer,
        step_size=config["step_size"],
        gamma=config["lr_gamma"],
    )
    criterion = nn.CrossEntropyLoss()

    best_val = 0.0
    best_state = None
    history = []

    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0
        iterator = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}", leave=False)
        for data in iterator:
            data = data.to(config["device"])
            optimizer.zero_grad()
            logits = model(data)
            loss = criterion(logits, data.y)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            epoch_loss += loss.item()
            iterator.set_postfix(loss=f"{loss.item():.4f}")

        scheduler.step()
        val_acc = evaluate_model(model, val_loader, config, all_metrics=False)
        history.append({"epoch": epoch + 1, "loss": epoch_loss / max(len(train_loader), 1), "val_acc": val_acc})
        log.info("Epoch %d | train_loss=%.4f | val_acc=%.4f", epoch + 1, history[-1]["loss"], val_acc)

        if val_acc >= best_val:
            best_val = val_acc
            best_state = deepcopy(model.state_dict())

    if best_state is not None:
        model.load_state_dict(best_state)
    return model, history


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------


def main(args: argparse.Namespace):
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    compositions = load_compositions(Path(args.data))
    paths_with_idx = extract_paths_from_compositions(compositions)
    if not paths_with_idx:
        raise RuntimeError("Нет путей в данных – проверьте файл compositonsDAG.json")

    contexts, targets, _ = create_training_pairs(paths_with_idx)
    if not contexts:
        raise RuntimeError("Не удалось построить обучающие пары (нет сервисов в качестве таргета).")

    ctx_train_full, ctx_test, y_train_full, y_test = split_contexts(
        contexts, targets, args.test_size, args.seed
    )
    ctx_train, ctx_val, y_train, y_val = train_test_split(
        ctx_train_full,
        y_train_full,
        test_size=args.val_size,
        random_state=args.seed,
        stratify=safe_stratify_labels(y_train_full, "val"),
    )

    node_map, node_features = build_node_feature_map(
        [path for path, _ in paths_with_idx],
        compositions,
    )
    if not node_map:
        raise RuntimeError("Не удалось построить признаки узлов.")

    feature_dim = len(next(iter(node_features.values())))
    mask_token_id = len(node_map)

    train_graphs = build_graph_dataset(ctx_train, y_train, node_map, node_features, feature_dim, mask_token_id)
    val_graphs = build_graph_dataset(ctx_val, y_val, node_map, node_features, feature_dim, mask_token_id)
    test_graphs = build_graph_dataset(ctx_test, y_test, node_map, node_features, feature_dim, mask_token_id)

    if not train_graphs or not val_graphs or not test_graphs:
        raise RuntimeError("Одна из выборок пуста – уменьшите test/val size или соберите больше данных.")

    log.info(
        "Dataset sizes | train=%d | val=%d | test=%d",
        len(train_graphs),
        len(val_graphs),
        len(test_graphs),
    )

    train_loader = DataLoader(
        train_graphs,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.batch_workers,
    )
    val_loader = DataLoader(
        val_graphs,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.batch_workers,
    )
    test_loader = DataLoader(
        test_graphs,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.batch_workers,
    )

    device = torch.device("cuda" if torch.cuda.is_available() and not args.force_cpu else "cpu")
    config = {
        "hidden_channels": args.hidden,
        "num_tools": len(node_map),
        "description_size": feature_dim,
        "emb_dropout": args.emb_dropout,
        "dropout": args.dropout,
        "model_type": "graph",
        "top_k": args.top_k,
        "mrr_k": args.mrr_k,
        "ndcg_k": args.ndcg_k,
        "device": device,
        "learning_rate": args.lr,
        "weight_decay": args.weight_decay,
        "step_size": args.step_size,
        "lr_gamma": args.lr_gamma,
    }

    log.info("Training on %s", device)
    model, history = train_btr_model(train_loader, val_loader, config, args.epochs)

    test_acc, test_top_k, test_mrr, test_ndcg, by_length = evaluate_model(
        model,
        test_loader,
        config,
        all_metrics=True,
        use_tqdm=False,
    )

    log.info("=== TEST RESULTS ===")
    log.info(
        "Accuracy: %.4f | nDCG@%d: %.4f | Top-%d: %.4f | MRR@%d: %.4f",
        test_acc,
        args.ndcg_k,
        test_ndcg,
        args.top_k,
        test_top_k,
        args.mrr_k,
        test_mrr,
    )
    log.info("Per-length breakdown: %s", {k: {m: round(v, 4) for m, v in val.items()} for k, val in by_length.items()})

    if args.print_history:
        for entry in history:
            log.info("Hist | epoch=%d | loss=%.4f | val_acc=%.4f", entry["epoch"], entry["loss"], entry["val_acc"])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="BTR-style DAG workflow recommender")
    parser.add_argument("--data", type=str, default="sequence recomendation/compositionsDAG.json", help="Путь до compositionsDAG.json")
    parser.add_argument("--epochs", type=int, default=150)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--hidden", type=int, default=128)
    parser.add_argument("--emb-dropout", type=float, default=0.2)
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--step-size", type=int, default=30)
    parser.add_argument("--lr-gamma", type=float, default=0.7)
    parser.add_argument("--test-size", type=float, default=0.25)
    parser.add_argument("--val-size", type=float, default=0.15)
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--mrr-k", type=int, default=5)
    parser.add_argument("--ndcg-k", type=int, default=5)
    parser.add_argument("--batch-workers", type=int, default=0, help="Сохранено для совместимости (не используется).")
    parser.add_argument("--force-cpu", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--print-history", action="store_true")

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )

    main(parser.parse_args())

