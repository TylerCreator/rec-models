# 🎯 Персонализация для последовательных рекомендаций

## 💡 Что такое персонализация?

**Без персонализации (v3.0, v3.1):** Все пользователи получают одинаковые рекомендации для одной и той же последовательности

**С персонализацией (v4.1):** Рекомендации учитывают:
- ✅ **Реальные owner** из поля `owner` в данных
- ✅ История конкретного пользователя
- ✅ Предпочтения пользователя (через user embeddings)
- ✅ Персонализированный PageRank для каждого пользователя

---

## 🏆 Реализовано в версии 4.1 - Реальные пользователи

### Извлечение owner из данных

#### Для сервисов (nodes с `mid`):
```json
{
  "id": 3985,
  "mid": 308,
  "owner": "50f7a1d80d58140037000006",  ← Берем напрямую
  ...
}
```

#### Для таблиц (nodes без `mid`):
```python
# Таблица не имеет owner напрямую
table_node = {"id": "1002132"}

# Определяем owner через целевой сервис:
# table_1002132 -> service_308 (owner: 50f7a1d80d58140037000006)
# => table_1002132 наследует owner от service_308
```

**Результат в v4.1:**
- ✅ User 1 (50f7a1d80d58140037000006): 80 последовательностей
- ✅ User 2 (360): 27 последовательностей
- ✅ Accuracy: 0.7576 (+47% vs базовой!)

---

## 🔍 Источники персонализации для DAG-рекомендаций

### 1. **Real Owner IDs** (используется в v4.1)
```python
# Реальные owner из данных
user_features = {
    'user_id': '50f7a1d80d58140037000006',  # Из поля owner
    'sequences_count': 80,                    # Количество последовательностей
    'unique_services': 15,                    # Уникальных сервисов
    'avg_path_length': 4.5                    # Средняя длина пути
}
```

### 2. **Историческое поведение**
```python
# История последовательностей конкретного пользователя
user_history = [
    ['service_1', 'service_2', 'service_5'],  # прошлая сессия 1
    ['service_1', 'service_3', 'service_5'],  # прошлая сессия 2
    ['service_2', 'service_4', 'service_6'],  # прошлая сессия 3
]
```

### 3. **Частота использования сервисов**
```python
# Какие сервисы пользователь использует чаще
user_service_frequency = {
    'service_1': 0.8,  # использует в 80% сессий
    'service_2': 0.5,  # использует в 50% сессий
    'service_5': 0.3,  # использует в 30% сессий
}
```

### 4. **Временные паттерны**
```python
# Когда пользователь обычно использует сервисы
temporal_context = {
    'hour': 14,           # 14:00
    'day_of_week': 2,     # вторник
    'is_weekend': False,
    'is_business_hours': True
}
```

### 5. **Контекст сессии**
```python
# Информация о текущей сессии
session_context = {
    'session_length': 5,      # длина текущей последовательности
    'time_since_start': 120,  # секунд с начала
    'error_count': 0,         # количество ошибок
    'latency_avg': 45.3       # средняя задержка
}
```

---

## 🛠️ Методы персонализации

### Метод 1: **User Embeddings** (рекомендуется) ⭐

**Идея:** Создать векторное представление для каждого пользователя

```python
class PersonalizedDAGNN(nn.Module):
    def __init__(self, num_users, num_items, embedding_dim=64):
        super().__init__()
        
        # User embeddings
        self.user_embedding = nn.Embedding(num_users, embedding_dim)
        
        # Node features + graph features
        self.node_encoder = nn.Linear(8, embedding_dim)
        
        # GNN слой
        self.gnn = GCNConv(embedding_dim, embedding_dim)
        
        # Attention для комбинации user + node
        self.attention = nn.MultiheadAttention(embedding_dim, num_heads=4)
        
        # Classifier
        self.classifier = nn.Linear(embedding_dim * 2, num_items)
    
    def forward(self, node_features, edge_index, user_ids, contexts):
        # Получаем user embeddings
        user_emb = self.user_embedding(user_ids)  # [batch, embedding_dim]
        
        # Получаем node embeddings
        node_emb = self.node_encoder(node_features)
        node_emb = self.gnn(node_emb, edge_index)
        context_emb = node_emb[contexts]  # [batch, embedding_dim]
        
        # Комбинируем user и context через attention
        combined, _ = self.attention(
            user_emb.unsqueeze(0),
            context_emb.unsqueeze(0),
            context_emb.unsqueeze(0)
        )
        combined = combined.squeeze(0)
        
        # Concatenate и predict
        final = torch.cat([user_emb, combined], dim=1)
        output = self.classifier(final)
        return output
```

**Преимущества:**
- ✅ Автоматически учит паттерны пользователя
- ✅ Масштабируется на большое количество пользователей
- ✅ Работает даже для новых последовательностей

---

### Метод 2: **User Features** (простой) 🚀

**Идея:** Добавить пользовательские признаки к входу модели

```python
# Текущие признаки узла (8)
node_features = [is_service, is_table, in_degree, out_degree, 
                pagerank, betweenness, closeness, clustering]

# Добавляем user features (5)
user_features = [
    user_service_frequency,  # как часто пользователь использует этот сервис
    user_avg_path_length,    # средняя длина путей пользователя
    user_session_count,      # количество сессий пользователя
    is_premium_user,         # премиум пользователь или нет
    user_error_rate          # частота ошибок пользователя
]

# Итого: 13 признаков
combined_features = node_features + user_features
```

**Преимущества:**
- ✅ Простая реализация
- ✅ Интерпретируемость
- ✅ Не требует много данных

---

### Метод 3: **Personalized PageRank** 🎯

**Идея:** Использовать персонализированный PageRank для каждого пользователя

```python
def personalized_pagerank_features(dag, user_history):
    """
    Вычисляем персонализированный PageRank на основе истории пользователя
    """
    # Узлы, которые пользователь часто использует
    user_preferred_nodes = set()
    for path in user_history:
        user_preferred_nodes.update(path)
    
    # Персонализированный вектор (больше вес для предпочитаемых узлов)
    personalization = {}
    for node in dag.nodes():
        if node in user_preferred_nodes:
            personalization[node] = 1.0 / len(user_preferred_nodes)
        else:
            personalization[node] = 0.0
    
    # Вычисляем персонализированный PageRank
    ppagerank = nx.pagerank(dag, personalization=personalization)
    
    return ppagerank
```

**Преимущества:**
- ✅ Использует структуру графа
- ✅ Учитывает предпочтения пользователя
- ✅ Основан на известном алгоритме

---

### Метод 4: **User-specific RNN Hidden State** 🔄

**Идея:** Инициализировать RNN (GRU4Rec) с пользовательским hidden state

```python
class PersonalizedGRU4Rec(nn.Module):
    def __init__(self, num_users, num_items, embedding_dim=64, hidden_size=128):
        super().__init__()
        
        # User-specific initial hidden state
        self.user_hidden_init = nn.Embedding(num_users, hidden_size)
        
        # Item embedding
        self.item_embedding = nn.Embedding(num_items, embedding_dim)
        
        # GRU
        self.gru = nn.GRU(embedding_dim, hidden_size, batch_first=True)
        
        # Classifier
        self.fc = nn.Linear(hidden_size, num_items)
    
    def forward(self, sequences, user_ids):
        batch_size = sequences.size(0)
        
        # Получаем user-specific initial hidden state
        h0 = self.user_hidden_init(user_ids).unsqueeze(0)  # [1, batch, hidden]
        
        # Embeddings
        embedded = self.item_embedding(sequences)
        
        # GRU с персонализированным начальным состоянием
        output, hidden = self.gru(embedded, h0)
        
        # Prediction
        last_hidden = hidden[-1]
        predictions = self.fc(last_hidden)
        return predictions
```

**Преимущества:**
- ✅ Каждый пользователь имеет свое "стартовое состояние"
- ✅ Естественная интеграция с RNN
- ✅ Capture long-term preferences

---

### Метод 5: **Collaborative Filtering + Sequential** 🤝

**Идея:** Комбинировать CF с последовательными рекомендациями

```python
class HybridPersonalizedRecommender(nn.Module):
    def __init__(self, num_users, num_items, embedding_dim=64):
        super().__init__()
        
        # Collaborative Filtering компонент
        self.user_embedding = nn.Embedding(num_users, embedding_dim)
        self.item_embedding = nn.Embedding(num_items, embedding_dim)
        
        # Sequential компонент (GRU/Transformer)
        self.sequential_model = GRU4Rec(num_items, embedding_dim)
        
        # Fusion layer
        self.fusion = nn.Linear(embedding_dim * 2, num_items)
    
    def forward(self, sequences, user_ids, target_items):
        # CF score
        user_emb = self.user_embedding(user_ids)
        item_emb = self.item_embedding(target_items)
        cf_score = (user_emb * item_emb).sum(dim=1)
        
        # Sequential score
        seq_output = self.sequential_model(sequences)
        
        # Combine
        combined = torch.cat([user_emb, seq_output], dim=1)
        final_score = self.fusion(combined)
        
        return final_score
```

**Преимущества:**
- ✅ Использует и CF, и sequential информацию
- ✅ Работает для cold-start пользователей
- ✅ Robust к sparse data

---

## 📊 Сравнение методов

| Метод | Сложность | Данные | Качество | Интерпретируемость |
|-------|-----------|--------|----------|-------------------|
| User Embeddings | Средняя | Много | ⭐⭐⭐⭐⭐ | ⭐⭐ |
| User Features | Низкая | Мало | ⭐⭐⭐ | ⭐⭐⭐⭐⭐ |
| Personalized PageRank | Средняя | Средне | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ |
| User-RNN Hidden | Средняя | Много | ⭐⭐⭐⭐ | ⭐⭐ |
| Hybrid CF+Seq | Высокая | Много | ⭐⭐⭐⭐⭐ | ⭐⭐ |

---

## 🚀 Рекомендуемый подход для DAG-рекомендаций

### Шаг 1: Начните с **User Features** (простой)

```python
# Минимальная персонализация
user_features = {
    'user_service_frequency': compute_frequency(user_history),
    'user_avg_path_length': np.mean([len(path) for path in user_history]),
    'user_session_count': len(user_history)
}
```

### Шаг 2: Добавьте **Personalized PageRank**

```python
ppagerank = personalized_pagerank_features(dag, user_history)
node_features.append(ppagerank[node])
```

### Шаг 3: Если есть много данных, используйте **User Embeddings**

```python
model = PersonalizedDAGNN(
    num_users=1000,
    num_items=50,
    embedding_dim=64
)
```

---

## 📈 Ожидаемые улучшения

### Без персонализации:
```
DAGNN Accuracy: 0.5152
```

### С персонализацией (ожидаемо):
```
Personalized DAGNN Accuracy: 0.60-0.75 (+15-25%)
```

**Почему?**
- ✅ Модель знает предпочтения пользователя
- ✅ Учитывает исторические паттерны
- ✅ Адаптируется под конкретные use cases

---

## 🎯 Примеры использования

### Use Case 1: Разные типы приложений

```python
# Mobile приложения чаще используют легкие сервисы
mobile_user = {
    'app_type': 'mobile',
    'preferred_services': ['service_1', 'service_2', 'service_5']
}

# Enterprise приложения используют тяжелые сервисы
enterprise_user = {
    'app_type': 'enterprise',
    'preferred_services': ['service_10', 'service_15', 'service_20']
}

# Модель будет рекомендовать разные следующие шаги
```

### Use Case 2: Региональные различия

```python
# Европейские пользователи предпочитают GDPR-compliant сервисы
europe_user = {
    'region': 'europe',
    'gdpr_preference': True,
    'preferred_services': ['gdpr_service_1', 'gdpr_service_2']
}

# Азиатские пользователи предпочитают локальные сервисы
asia_user = {
    'region': 'asia',
    'preferred_services': ['local_service_3', 'local_service_4']
}
```

### Use Case 3: Уровень опыта пользователя

```python
# Новички используют простые последовательности
beginner_user = {
    'experience_level': 'beginner',
    'avg_path_length': 3,  # короткие пути
    'error_rate': 0.2      # высокий error rate
}

# Эксперты используют сложные последовательности
expert_user = {
    'experience_level': 'expert',
    'avg_path_length': 10,  # длинные пути
    'error_rate': 0.02      # низкий error rate
}
```

---

## 💻 Готовый код будет в файле:

`sequence_dag_recommender_personalized.py` - полная реализация с персонализацией

---

## 📚 Дополнительные материалы

### Статьи:
1. **"Session-based Recommendations with Recurrent Neural Networks"** (2016)
2. **"Personalized Top-N Sequential Recommendation via Convolutional Sequence Embedding"** (2018)
3. **"Self-Attentive Sequential Recommendation"** (2018)

### Подходы из индустрии:
- **YouTube:** User embeddings + Sequential RNN
- **Amazon:** Session-based + User history
- **Netflix:** Hybrid CF + Sequential Transformers

---

**Дата:** 2025-10-24  
**Версия:** 4.0 Personalization Concept

