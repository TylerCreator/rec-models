#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Transformers4Rec-based recommender for continuing sequences in a DAG.

Основные шаги:
1. Загружаем композиции из JSON и извлекаем реальные пути.
2. Формируем обучающую выборку (контекст -> следующий сервис).
3. Кодируем последовательности в фиксированную длину.
4. Обучаем трансформер из Transformers4Rec на задачу Next-Item Prediction.

Пример запуска:
    python transformers4rec_recommender.py --data compositionsDAG.json --epochs 50
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from transformers4rec.config.schema import ColumnSchema, Schema, Tags
from transformers4rec.config.transformer import TransformerConfig
from transformers4rec.torch import TabularSequenceFeatures, NextItemPredictionTask
from transformers4rec.torch.block import SequentialBlock, TransformerBlock

from directed_dag_models import (
    build_graph,
    create_training_pairs,
    extract_paths_from_compositions,
    load_dag_from_json,
    prepare_pyg,
    split_data,
    DeepDAGRecommender,
    train_graph_model,
)

logger = logging.getLogger("transformers4rec_recommender")


# -----------------------------------------------------------------------------
# Dataset utilities
# -----------------------------------------------------------------------------


def build_node_index(contexts: List[Tuple[str, ...]], targets: List[str]) -> Dict[str, int]:
    nodes = set()
    for ctx in contexts:
        nodes.update(ctx)
    nodes.update(targets)
    node_map = {node: idx + 1 for idx, node in enumerate(sorted(nodes))}
    logger.info("Built node index with %d unique nodes", len(node_map))
    return node_map


def encode_sequences(
    contexts: List[Tuple[str, ...]],
    node_map: Dict[str, int],
    max_len: int,
) -> Tuple[np.ndarray, np.ndarray]:
    sequences = []
    lengths = []
    for ctx in contexts:
        ids = [node_map[node] for node in ctx if node in node_map]
        if not ids:
            lengths.append(0)
            sequences.append([0] * max_len)
            continue
        lengths.append(min(len(ids), max_len))
        ids = ids[-max_len:]
        if len(ids) < max_len:
            ids = [0] * (max_len - len(ids)) + ids
        sequences.append(ids)
    return np.array(sequences, dtype=np.int64), np.array(lengths, dtype=np.int64)


def build_service_map(targets: List[str]) -> Dict[str, int]:
    services = sorted(set(targets))
    service_map = {svc: idx for idx, svc in enumerate(services)}
    logger.info("Number of unique services: %d", len(service_map))
    return service_map


class SequenceDataset(Dataset):
    def __init__(self, sequences: np.ndarray, lengths: np.ndarray, targets: np.ndarray):
        self.sequences = torch.from_numpy(sequences)
        self.lengths = torch.from_numpy(lengths)
        self.targets = torch.from_numpy(targets)

    def __len__(self):
        return self.sequences.size(0)

    def __getitem__(self, idx):
        return {
            "item_id": self.sequences[idx],
            "item_id__seq_length": self.lengths[idx],
            "item_id__target": self.targets[idx],
        }


# -----------------------------------------------------------------------------
# Model builder
# -----------------------------------------------------------------------------


def build_t4rec_model(
    num_items: int,
    max_len: int,
    hidden_dim: int,
    num_heads: int,
    num_layers: int,
    dropout: float,
):
    schema = Schema(
        [
            ColumnSchema(
                name="item_id",
                tags=[Tags.ITEM_ID, Tags.CATEGORICAL, Tags.SEQUENCE],
                properties={
                    "value_count": {"min": 1, "max": max_len},
                    "domain": {"min": 0, "max": num_items},
                },
            )
        ]
    )

    sequence_features = TabularSequenceFeatures.from_schema(
        schema,
        max_sequence_length=max_len,
        continuous_projection=None,
        aggregation="concat",
        embedding_dim_default=hidden_dim,
    )

    transformer_config = TransformerConfig.build(
        d_model=hidden_dim,
        n_head=num_heads,
        dropout=dropout,
        ff_dim=hidden_dim * 4,
        activation="gelu",
        num_layers=num_layers,
    )

    body = SequentialBlock(
        sequence_features,
        TransformerBlock(transformer_config, masking="clm"),
    )

    task = NextItemPredictionTask(
        target_dim=num_items,
        weight_tying=True,
        hf_format=False,
        metrics=[],  # Будем вычислять вручную
    )

    return task.to_torch_model(body)


# -----------------------------------------------------------------------------
# Training & evaluation
# -----------------------------------------------------------------------------


def move_batch_to_device(batch: Dict[str, torch.Tensor], device: torch.device) -> Dict[str, torch.Tensor]:
    return {k: v.to(device) for k, v in batch.items()}


def train_epoch(model, dataloader, optimizer, device: torch.device):
    model.train()
    total_loss = 0.0
    total_steps = 0
    for batch in dataloader:
        batch = move_batch_to_device(batch, device)
        outputs = model(batch)
        loss = outputs["loss"]
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        optimizer.zero_grad()
        total_loss += loss.item()
        total_steps += 1
    return total_loss / max(1, total_steps)


def evaluate(model, dataloader, device: torch.device, num_classes: int) -> Dict[str, float]:
    model.eval()
    preds_list = []
    probs_list = []
    labels_list = []
    with torch.no_grad():
        for batch in dataloader:
            batch = move_batch_to_device(batch, device)
            outputs = model(batch)
            logits = outputs["logits"]
            probs = F.softmax(logits, dim=-1)
            preds = probs.argmax(dim=-1)
            preds_list.append(preds.cpu().numpy())
            probs_list.append(probs.cpu().numpy())
            labels_list.append(batch["item_id__target"].cpu().numpy())

    preds = np.concatenate(preds_list, axis=0)
    probs = np.concatenate(probs_list, axis=0)
    labels = np.concatenate(labels_list, axis=0)

    accuracy = float((preds == labels).mean())
    f1 = float(
        (preds == labels).sum() / len(labels)
    )  # грубая оценка, точные метрики вычислим ниже

    # Точные метрики (те же, что в directed_dag_models)
    from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, ndcg_score

    metrics = {
        "accuracy": accuracy_score(labels, preds),
        "f1": f1_score(labels, preds, average="macro", zero_division=0),
        "precision": precision_score(labels, preds, average="macro", zero_division=0),
        "recall": recall_score(labels, preds, average="macro", zero_division=0),
        "ndcg": ndcg_score(
            np.eye(num_classes)[labels],
            probs,
            k=min(10, num_classes),
        ),
    }
    return metrics


def train_transformers4rec(
    train_dataset: SequenceDataset,
    test_dataset: SequenceDataset,
    num_items: int,
    max_len: int,
    hidden_dim: int,
    num_heads: int,
    num_layers: int,
    dropout: float,
    lr: float,
    batch_size: int,
    epochs: int,
    device: torch.device,
) -> Dict[str, float]:
    model = build_t4rec_model(
        num_items=num_items,
        max_len=max_len,
        hidden_dim=hidden_dim,
        num_heads=num_heads,
        num_layers=num_layers,
        dropout=dropout,
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, drop_last=False)

    for epoch in range(epochs):
        loss = train_epoch(model, train_loader, optimizer, device)
        if (epoch + 1) % 10 == 0:
            logger.info("Epoch %d/%d — train loss: %.4f", epoch + 1, epochs, loss)

    return evaluate(model, test_loader, device, num_items)


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------


def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    logger.info("Using device: %s", device)

    _, compositions = load_dag_from_json(Path(args.data))
    paths_with_idx = extract_paths_from_compositions(compositions)
    contexts, targets, comp_indices = create_training_pairs(paths_with_idx)

    ctx_train, ctx_test, y_train, y_test, comp_train_idx, comp_test_idx = split_data(
        contexts, targets, comp_indices, args.test_size, args.seed
    )

    node_map = build_node_index(contexts, targets)
    service_map = build_service_map(targets)

    seq_train_np, len_train_np = encode_sequences(ctx_train, node_map, args.max_len)
    seq_test_np, len_test_np = encode_sequences(ctx_test, node_map, args.max_len)

    y_train_np = np.array([service_map[y] for y in y_train], dtype=np.int64)
    y_test_np = np.array([service_map[y] for y in y_test], dtype=np.int64)

    train_dataset = SequenceDataset(seq_train_np, len_train_np, y_train_np)
    test_dataset = SequenceDataset(seq_test_np, len_test_np, y_test_np)

    t4rec_metrics = train_transformers4rec(
        train_dataset=train_dataset,
        test_dataset=test_dataset,
        num_items=len(service_map),
        max_len=args.max_len,
        hidden_dim=args.hidden,
        num_heads=args.num_heads,
        num_layers=args.num_layers,
        dropout=args.dropout,
        lr=args.lr,
        batch_size=args.batch_size,
        epochs=args.epochs,
        device=device,
    )

    print("\n=== Transformers4Rec SUMMARY ===")
    print(
        f"acc={t4rec_metrics['accuracy']:.4f} | "
        f"ndcg={t4rec_metrics['ndcg']:.4f} | "
        f"f1={t4rec_metrics['f1']:.4f} | "
        f"precision={t4rec_metrics['precision']:.4f} | "
        f"recall={t4rec_metrics['recall']:.4f}"
    )

    if args.compare_deepdag:
        logger.info("Training DeepDAG2022 baseline for comparison...")
        train_comp_set = set(comp_train_idx)
        train_paths_only = [path for path, comp_idx in paths_with_idx if comp_idx in train_comp_set]
        all_nodes = sorted({node for path, _ in paths_with_idx for node in path})
        graph = build_graph(train_paths_only)
        graph.add_nodes_from(all_nodes)
        nodes_sorted = sorted(graph.nodes())
        data_pyg, node_map = prepare_pyg(graph, nodes_sorted)

        train_idx = torch.tensor([node_map[ctx[-1]] for ctx in ctx_train], dtype=torch.long)
        test_idx = torch.tensor([node_map[ctx[-1]] for ctx in ctx_test], dtype=torch.long)
        targets_train = torch.tensor([service_map[y] for y in y_train], dtype=torch.long)
        targets_test = torch.tensor([service_map[y] for y in y_test], dtype=torch.long)

        deepdag = DeepDAGRecommender(
            in_channels=2,
            hidden=args.deep_hidden,
            out_channels=len(service_map),
            num_layers=args.deep_layers,
            heads=args.deep_heads,
            dropout=args.dropout,
        )
        opt_deepdag = torch.optim.Adam(deepdag.parameters(), lr=args.lr * 0.8)
        deep_metrics = train_graph_model(
            deepdag,
            data_pyg,
            train_idx,
            test_idx,
            targets_train,
            targets_test,
            opt_deepdag,
            args.epochs,
            "DeepDAG2022",
        )
        print("\n=== DeepDAG2022 SUMMARY ===")
        print(
            f"acc={deep_metrics['accuracy']:.4f} | "
            f"ndcg={deep_metrics['ndcg']:.4f} | "
            f"f1={deep_metrics['f1']:.4f} | "
            f"precision={deep_metrics['precision']:.4f} | "
            f"recall={deep_metrics['recall']:.4f}"
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Transformers4Rec DAG sequence recommender")
    parser.add_argument("--data", type=str, default="compositionsDAG.json")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--max-len", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--hidden", type=int, default=128)
    parser.add_argument("--num-heads", type=int, default=4)
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--deep-hidden", type=int, default=128)
    parser.add_argument("--deep-heads", type=int, default=4)
    parser.add_argument("--deep-layers", type=int, default=3)
    parser.add_argument("--compare-deepdag", action="store_true", help="Train DeepDAG2022 baseline for comparison")
    parser.add_argument("--test-size", type=float, default=0.3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--cpu", action="store_true", help="Force CPU even if CUDA is available")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    main(args)

