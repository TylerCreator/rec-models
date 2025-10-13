import json
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, ndcg_score
from collections import Counter
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder, LabelBinarizer, MultiLabelBinarizer
from sklearn.neighbors import NearestNeighbors
from torch_geometric.nn import GCNConv, GATConv, APPNP

# ==================== Подготовка данных ====================
def prepare_sequences(calls_df, context_size=3):
    calls_df = calls_df.sort_values(['owner', 'start_time'])
    user_groups = calls_df.groupby('owner')
    X, y, user_ids = [], [], []
    for user, group in user_groups:
        mids = group['mid'].tolist()
        for i in range(context_size, len(mids)):
            context = mids[i-context_size:i]
            target = mids[i]
            X.append(context)
            y.append(target)
            user_ids.append(user)
    return X, y, user_ids

def split_by_time(calls_df, context_size=3, test_ratio=0.3):
    calls_df = calls_df.sort_values(['owner', 'start_time'])
    train_rows, test_rows = [], []
    for user, group in calls_df.groupby('owner'):
        n = len(group)
        if n <= context_size:
            continue
        split = int(n * (1 - test_ratio))
        if split <= context_size:
            continue
        train_rows.append(group.iloc[:split])
        test_rows.append(group.iloc[split:])
    train_df = pd.concat(train_rows)
    test_df = pd.concat(test_rows)
    X_train, y_train, user_train = prepare_sequences(train_df, context_size)
    X_test, y_test, user_test = prepare_sequences(test_df, context_size)
    return X_train, y_train, user_train, X_test, y_test, user_test

# ==================== Метрики ====================
def evaluate_topk_model(preds, true_labels, proba_preds, name="Model", k=15):
    hits = 0
    precisions = []
    recalls = []
    ndcgs = []
    for i in range(len(true_labels)):
        topk = np.argpartition(proba_preds[i], -k)[-k:]
        topk = topk[np.argsort(proba_preds[i][topk])[::-1]]
        if true_labels[i] in topk:
            hits += 1
            precisions.append(1.0 / k)
            recalls.append(1.0)
            rank = np.where(topk == true_labels[i])[0][0]
            ndcgs.append(1.0 / np.log2(rank + 2))
        else:
            precisions.append(0.0)
            recalls.append(0.0)
            ndcgs.append(0.0)
    print(f"\n📊 {name} Top-{k} Metrics")
    print(f"Precision@{k}: {np.mean(precisions):.4f}")
    print(f"Recall@{k}:    {np.mean(recalls):.4f}")
    print(f"nDCG@{k}:      {np.mean(ndcgs):.4f}")

def evaluate_model(preds, true_labels, proba_preds=None, name="Model", label_binarizer=None):
    acc = accuracy_score(true_labels, preds)
    f1 = f1_score(true_labels, preds, average='macro')
    precision = precision_score(true_labels, preds, average='macro', zero_division=0)
    recall = recall_score(true_labels, preds, average='macro', zero_division=0)
    print(f"\n📊 {name} Metrics")
    print(f"Accuracy:  {acc:.4f}")
    print(f"F1-score:  {f1:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    if proba_preds is not None:
        try:
            n_classes = proba_preds.shape[1]
            lb = LabelBinarizer()
            lb.fit(range(n_classes))
            true_bin = lb.transform(true_labels)
            if true_bin.ndim == 1:
                true_bin = np.eye(n_classes)[true_labels]
            ndcg = ndcg_score(true_bin, proba_preds)
            print(f"nDCG:      {ndcg:.4f}")
        except Exception as e:
            print(f"nDCG:      ❌ Error calculating nDCG: {e}")
    else:
        print("nDCG:      Not available (no probabilities)")

# ====== PHCF-BPR (заглушка, для примера, можно заменить на реальную реализацию) ======
class PHCFBPRRecommender:
    def __init__(self, num_users, num_services):
        self.num_users = num_users
        self.num_services = num_services
        self.user_emb = nn.Embedding(num_users, 32)
        self.service_emb = nn.Embedding(num_services, 32)
        self.trained = False
    def fit(self, user_idx, service_idx):
        # Простейшее обучение: просто инициализация (заглушка)
        self.trained = True
    def predict_scores(self, user_idx):
        # Возвращает скор для всех сервисов для каждого пользователя
        user_vecs = self.user_emb(user_idx)
        all_service_vecs = self.service_emb.weight  # [num_services, emb_dim]
        scores = torch.matmul(user_vecs, all_service_vecs.t())  # [batch, num_services]
        return scores.detach().cpu().numpy()

# ====== KNN+Popular ======
def knn_popular_scores(X_train_bow, user_test_idx, num_services):
    # KNN по bag-of-services
    knn = NearestNeighbors(n_neighbors=5, metric='cosine')
    knn.fit(X_train_bow)
    knn_scores = knn.kneighbors(X_train_bow[user_test_idx], return_distance=False)
    # Popular
    popular_scores = X_train_bow.sum(axis=0)
    # Для каждого пользователя: если KNN > 0, брать KNN, иначе popular * min_knn
    scores = []
    for user_idx, idxs in enumerate(knn_scores):
        knn_vec = X_train_bow[idxs].mean(axis=0)
        min_knn = knn_vec[knn_vec > 0].min() if np.any(knn_vec > 0) else 1.0
        user_scores = np.zeros(num_services)
        for s in range(num_services):
            if knn_vec[s] > 0:
                user_scores[s] = knn_vec[s]
            else:
                user_scores[s] = popular_scores[s] * min_knn
        scores.append(user_scores)
    return np.array(scores)

# ====== Гибридные модели ======
class HybridGNNKNNPopular:
    def __init__(self, gnn_model, X_train_bow, user_test_idx, num_services, alpha=0.5):
        self.gnn_model = gnn_model
        self.X_train_bow = X_train_bow
        self.user_test_idx = user_test_idx
        self.num_services = num_services
        self.alpha = alpha
    def predict(self, user_test_idx, X_test_idx, edge_index):
        with torch.no_grad():
            gnn_logits = self.gnn_model(user_test_idx, X_test_idx, edge_index)
            gnn_scores = F.softmax(gnn_logits, dim=1).cpu().numpy()
        knn_pop_scores = knn_popular_scores(self.X_train_bow, user_test_idx.cpu().numpy(), self.num_services)
        # Усреднение скорингов
        hybrid_scores = self.alpha * gnn_scores + (1 - self.alpha) * knn_pop_scores
        return hybrid_scores

class HybridGNNPHCFBPR:
    def __init__(self, gnn_model, phcf_model, alpha=0.5):
        self.gnn_model = gnn_model
        self.phcf_model = phcf_model
        self.alpha = alpha
    def predict(self, user_test_idx, X_test_idx, edge_index):
        with torch.no_grad():
            gnn_logits = self.gnn_model(user_test_idx, X_test_idx, edge_index)
            gnn_scores = F.softmax(gnn_logits, dim=1).cpu().numpy()
        phcf_scores = self.phcf_model.predict_scores(user_test_idx)
        # Взвешенное объединение скорингов
        hybrid_scores = self.alpha * gnn_scores + (1 - self.alpha) * phcf_scores
        return hybrid_scores

# ... (main: обучение GNN, PHCFBPR, подготовка KNN+Popular, сравнение всех моделей, фильтрация новых сервисов, top-15 метрики) ... 

class GCNRecommender(nn.Module):
    def __init__(self, num_users, num_services, emb_dim=32, context_size=3):
        super().__init__()
        self.user_emb = nn.Embedding(num_users, emb_dim)
        self.service_emb = nn.Embedding(num_services, emb_dim)
        self.gcn1 = GCNConv(emb_dim, emb_dim)
        self.gcn2 = GCNConv(emb_dim, emb_dim)
        self.fc = nn.Linear(emb_dim * (context_size + 1), num_services)
    def forward(self, user_idx, context_idx, edge_index):
        user_vec = self.user_emb(user_idx)
        all_service_emb = self.service_emb.weight  # [num_services, emb_dim]
        gnn_out = F.relu(self.gcn1(all_service_emb, edge_index))
        gnn_out = self.gcn2(gnn_out, edge_index)
        context_vecs = gnn_out[context_idx]  # [batch, context_size, emb_dim]
        context_vec = context_vecs.view(context_vecs.size(0), -1)
        x = torch.cat([user_vec, context_vec], dim=1)
        out = self.fc(x)
        return out

class GATRecommender(nn.Module):
    def __init__(self, num_users, num_services, emb_dim=32, context_size=3, heads=2):
        super().__init__()
        self.user_emb = nn.Embedding(num_users, emb_dim)
        self.service_emb = nn.Embedding(num_services, emb_dim)
        self.gat1 = GATConv(emb_dim, emb_dim, heads=heads)
        self.gat2 = GATConv(emb_dim*heads, emb_dim, heads=1)
        self.fc = nn.Linear(emb_dim * (context_size + 1), num_services)
    def forward(self, user_idx, context_idx, edge_index):
        user_vec = self.user_emb(user_idx)
        all_service_emb = self.service_emb.weight
        gnn_out = F.elu(self.gat1(all_service_emb, edge_index))
        gnn_out = self.gat2(gnn_out, edge_index)
        context_vecs = gnn_out[context_idx]
        context_vec = context_vecs.view(context_vecs.size(0), -1)
        x = torch.cat([user_vec, context_vec], dim=1)
        out = self.fc(x)
        return out

class DAGNN(nn.Module):
    def __init__(self, in_channels: int, K: int):
        super().__init__()
        self.propagation = APPNP(K=K, alpha=0)
        self.att = nn.Parameter(torch.Tensor(K + 1))
        self.reset_parameters()
    def reset_parameters(self):
        self.propagation.reset_parameters()
        nn.init.zeros_(self.att)
    def forward(self, x, edge_index):
        xs = [x]
        edge_weight = torch.ones(edge_index.size(1), dtype=torch.float32, device=edge_index.device)
        for _ in range(self.propagation.K):
            x = self.propagation.propagate(edge_index, x=x, edge_weight=edge_weight)
            xs.append(x)
        out = torch.stack(xs, dim=-1)
        out = (out * self.att.view(1, 1, -1)).sum(dim=-1)
        return out

class DAGNNRecommender(nn.Module):
    def __init__(self, num_users, num_services, emb_dim=32, context_size=3, K=5):
        super().__init__()
        self.user_emb = nn.Embedding(num_users, emb_dim)
        self.service_emb = nn.Embedding(num_services, emb_dim)
        self.dagnn = DAGNN(emb_dim, K)
        self.fc = nn.Linear(emb_dim * (context_size + 1), num_services)
    def forward(self, user_idx, context_idx, edge_index):
        user_vec = self.user_emb(user_idx)
        all_service_emb = self.service_emb.weight
        gnn_out = self.dagnn(all_service_emb, edge_index)
        context_vecs = gnn_out[context_idx]
        context_vec = context_vecs.view(context_vecs.size(0), -1)
        x = torch.cat([user_vec, context_vec], dim=1)
        out = self.fc(x)
        return out

class HybridSeqRecommender(nn.Module):
    def __init__(self, num_users, num_services, emb_dim=32, context_size=3):
        super().__init__()
        self.user_emb = nn.Embedding(num_users, emb_dim)
        self.service_emb = nn.Embedding(num_services, emb_dim)
        self.fc = nn.Linear(emb_dim * (context_size + 1), num_services)
    def forward(self, user_idx, context_idx):
        user_vec = self.user_emb(user_idx)
        context_vecs = self.service_emb(context_idx)
        context_vec = context_vecs.view(context_vecs.size(0), -1)
        x = torch.cat([user_vec, context_vec], dim=1)
        out = self.fc(x)
        return out

if __name__ == "__main__":
    # Подготовка данных (как в предыдущих версиях)
    calls_df = pd.read_csv("calls.csv", sep=';')
    context_size = 3
    X_train, y_train, user_train, X_test, y_test, user_test = split_by_time(calls_df, context_size=context_size, test_ratio=0.3)
    user_encoder = LabelEncoder()
    service_encoder = LabelEncoder()
    all_users = user_encoder.fit_transform(list(set(user_train) | set(user_test)))
    all_services = service_encoder.fit_transform(list(set(y_train) | set(y_test) | {s for ctx in X_train+X_test for s in ctx}))
    num_users = len(user_encoder.classes_)
    num_services = len(service_encoder.classes_)
    user_train_idx = torch.tensor(user_encoder.transform(user_train), dtype=torch.long)
    user_test_idx = torch.tensor(user_encoder.transform(user_test), dtype=torch.long)
    X_train_idx = torch.tensor([[service_encoder.transform([s])[0] for s in ctx] for ctx in X_train], dtype=torch.long)
    X_test_idx = torch.tensor([[service_encoder.transform([s])[0] for s in ctx] for ctx in X_test], dtype=torch.long)
    y_train_idx = torch.tensor(service_encoder.transform(y_train), dtype=torch.long)
    y_test_idx = torch.tensor(service_encoder.transform(y_test), dtype=torch.long)

    # Граф для GNN
    edge_set = set()
    for ctx, target in zip(X_train, y_train):
        for i in range(1, len(ctx)):
            edge_set.add((service_encoder.transform([ctx[i-1]])[0], service_encoder.transform([ctx[i]])[0]))
        edge_set.add((service_encoder.transform([ctx[-1]])[0], service_encoder.transform([target])[0]))
    edge_index = torch.tensor(list(edge_set), dtype=torch.long).t().contiguous() if edge_set else torch.empty((2,0), dtype=torch.long)

    # Обучение GNN (например, GAT)
    gnn = GATRecommender(num_users, num_services, emb_dim=32, context_size=context_size)
    gnn_opt = torch.optim.Adam(gnn.parameters(), lr=0.01)
    n_epochs = 10
    batch_size = 256
    gnn.train()
    for epoch in range(n_epochs):
        perm = torch.randperm(len(X_train_idx))
        for i in range(0, len(X_train_idx), batch_size):
            idx = perm[i:i+batch_size]
            batch_user = user_train_idx[idx]
            batch_context = X_train_idx[idx]
            batch_target = y_train_idx[idx]
            gnn_opt.zero_grad()
            out = gnn(batch_user, batch_context, edge_index)
            loss = F.cross_entropy(out, batch_target)
            loss.backward()
            gnn_opt.step()

    # PHCF-BPR (заглушка)
    phcf = PHCFBPRRecommender(num_users, num_services)
    phcf.fit(user_train_idx, y_train_idx)

    # KNN+Popular
    mlb = MultiLabelBinarizer(classes=service_encoder.classes_)
    X_train_bow = mlb.fit_transform(X_train)

    # Гибридные модели
    hybrid_gnn_knn_pop = HybridGNNKNNPopular(gnn, X_train_bow, user_test_idx, num_services, alpha=0.5)
    hybrid_gnn_phcf = HybridGNNPHCFBPR(gnn, phcf, alpha=0.5)

    # Оценка всех моделей
    def filter_and_eval_scores(scores, user_test_idx, X_test_idx, y_test_idx, user_train_idx, X_train_idx, name, service_encoder=None, N_examples=5):
        filtered_preds = []
        filtered_true = []
        filtered_proba = []
        examples = []
        total = len(user_test_idx)
        filtered = 0
        for i in range(len(user_test_idx)):
            user = user_test_idx[i].item()
            context = X_test_idx[i].cpu().numpy().tolist()
            true_service = y_test_idx[i].item()
            user_train_mask = (user_train_idx == user)
            used_services = set(X_train_idx[user_train_mask].cpu().numpy().flatten().tolist())
            used_services.update(context)
            if true_service in used_services:
                continue
            proba_i = scores[i].copy()
            for s in used_services:
                if 0 <= s < proba_i.shape[0]:
                    proba_i[s] = 0
            if proba_i.sum() == 0:
                continue
            pred = proba_i.argmax()
            filtered_preds.append(pred)
            filtered_true.append(true_service)
            filtered_proba.append(proba_i)
            filtered += 1
            # Примеры рекомендаций
            if len(examples) < N_examples and service_encoder is not None:
                top5 = np.argsort(proba_i)[-5:][::-1]
                context_labels = service_encoder.inverse_transform(context)
                top5_labels = service_encoder.inverse_transform(top5)
                true_label = service_encoder.inverse_transform([true_service])[0]
                examples.append({
                    'user': user,
                    'context': context_labels,
                    'true_service': true_label,
                    'top5_recs': top5_labels
                })
        if filtered_preds:
            print(f"\nДоля тестовых примеров с новым сервисом: {filtered}/{total} = {filtered/total:.3f}")
            evaluate_model(np.array(filtered_preds), np.array(filtered_true), proba_preds=np.array(filtered_proba), name=name)
            evaluate_topk_model(np.array(filtered_preds), np.array(filtered_true), np.array(filtered_proba), name=name, k=15)
            if examples:
                print(f"\nПримеры рекомендаций для {name}:")
                for ex in examples:
                    print(f"User: {ex['user']}, Context: {list(ex['context'])}, True: {ex['true_service']}, Top-5: {list(ex['top5_recs'])}")
        else:
            print(f"Нет ни одного тестового примера с новым сервисом для пользователя! ({name})")

    # GNN
    with torch.no_grad():
        gnn_logits = gnn(user_test_idx, X_test_idx, edge_index)
        gnn_scores = F.softmax(gnn_logits, dim=1).cpu().numpy()
    filter_and_eval_scores(gnn_scores, user_test_idx, X_test_idx, y_test_idx, user_train_idx, X_train_idx, "GAT (new service only)")

    # Hybrid GNN+KNN+Popular
    hybrid_scores = hybrid_gnn_knn_pop.predict(user_test_idx, X_test_idx, edge_index)
    filter_and_eval_scores(hybrid_scores, user_test_idx, X_test_idx, y_test_idx, user_train_idx, X_train_idx, "Hybrid GNN+KNN+Popular")

    # Hybrid GNN+PHCF-BPR
    hybrid_scores = hybrid_gnn_phcf.predict(user_test_idx, X_test_idx, edge_index)
    filter_and_eval_scores(hybrid_scores, user_test_idx, X_test_idx, y_test_idx, user_train_idx, X_train_idx, "Hybrid GNN+PHCF-BPR") 