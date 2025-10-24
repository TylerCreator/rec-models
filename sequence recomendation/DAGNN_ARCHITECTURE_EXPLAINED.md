# 🏗️ Архитектура DAGNN: Детальное описание

## 📋 Содержание
1. [Подготовка данных](#1-подготовка-данных)
2. [Входные параметры модели](#2-входные-параметры-модели)
3. [Encoder Block](#3-encoder-block)
4. [Residual Connection](#4-residual-connection)
5. [DAGNN Propagation](#5-dagnn-propagation)
6. [Classifier](#6-classifier)
7. [Полный Forward Pass](#7-полный-forward-pass)

---

## 1. Подготовка данных

### Исходные данные

**Файл:** `compositionsDAG.json`
```json
{
  "id": "comp_001",
  "composition": {
    "nodes": [
      {"id": "n1", "service": "table_1002132"},
      {"id": "n2", "service": "service_308"},
      {"id": "n3", "service": "service_309"}
    ],
    "links": [
      {"source": "n1", "target": "n2"},
      {"source": "n2", "target": "n3"}
    ]
  }
}
```

**Что извлекаем:**
- 943 композиции → 106 уникальных путей
- Пример пути: `['table_1002132', 'service_308', 'service_309']`

---

### Шаг 1.1: Построение графа из реальных путей

**Функция:** `build_graph_from_real_paths(paths)`

**Что делает:**
```python
path_graph = nx.DiGraph()  # Создаем направленный граф

for path in paths:  # Для каждого уникального пути
    for i in range(len(path) - 1):
        source = path[i]      # Например: 'table_1002132'
        target = path[i + 1]  # Например: 'service_308'
        
        # Определяем тип узла
        source_type = 'service' if source.startswith('service') else 'table'
        target_type = 'service' if target.startswith('service') else 'table'
        
        # Добавляем в граф
        path_graph.add_node(source, type=source_type)
        path_graph.add_node(target, type=target_type)
        path_graph.add_edge(source, target)
```

**Результат:**
```
Граф:
  - 50 узлов (15 сервисов + 35 таблиц)
  - 97 направленных ребер
  - Связи: table → service, service → service
```

**Зачем нужен граф:**
- Представляет структуру зависимостей между узлами
- DAGNN будет распространять информацию по этим связям
- Граф кодирует, какие узлы влияют друг на друга

---

### Шаг 1.2: Создание обучающих примеров

**Функция:** `create_training_data(paths)`

**Что делает:**
```python
X_raw = []  # Контексты
y_raw = []  # Целевые значения

for path in paths:
    # path = ['table_1002132', 'service_308', 'service_309']
    
    for i in range(1, len(path)):
        context = tuple(path[:i])  # ('table_1002132',) или ('table_1002132', 'service_308')
        next_step = path[i]         # 'service_308' или 'service_309'
        
        # Берем только переходы на сервисы (они - целевые классы)
        if next_step.startswith("service"):
            X_raw.append(context)
            y_raw.append(next_step)
```

**Пример данных:**

| № | Контекст (X_raw) | Целевой сервис (y_raw) | Пояснение |
|---|------------------|------------------------|-----------|
| 0 | `('table_1002132',)` | `'service_308'` | После таблицы → первый сервис |
| 1 | `('table_1002132', 'service_308')` | `'service_309'` | После table+service → следующий сервис |

**Результат:**
- 125 обучающих примеров
- X_raw: списки контекстов (переменной длины)
- y_raw: названия целевых сервисов

**Зачем:**
- Задача: по контексту предсказать следующий сервис
- Это supervised learning - нужны пары (вход, правильный ответ)

---

### Шаг 1.3: Подготовка данных для PyTorch Geometric

**Функция:** `prepare_pytorch_geometric_data(dag, X_raw, y_raw, paths)`

#### 1.3.1: Маппинг узлов → индексы

```python
node_list = list(path_graph.nodes)
# ['table_1002132', 'table_1002133', ..., 'service_308', 'service_309', ...]

node_encoder = LabelEncoder()
node_ids = node_encoder.fit_transform(node_list)
# [0, 1, 2, ..., 48, 49]

node_map = {node: idx for node, idx in zip(node_list, node_ids)}
# {
#   'table_1002132': 0,
#   'table_1002133': 1,
#   ...
#   'service_308': 35,
#   'service_309': 36,
#   ...
# }
```

**Зачем нужен node_map:**
- Нейросети работают с числами, не со строками
- Каждому узлу назначается уникальный индекс 0-49
- Используется для edge_index и контекстов

#### 1.3.2: Создание edge_index

```python
edge_index = torch.tensor(
    [[node_map[u], node_map[v]] for u, v in path_graph.edges],
    dtype=torch.long
).t()

# Пример:
# path_graph.edges: [('table_1002132', 'service_308'), ('service_308', 'service_309'), ...]
# 
# edge_index после преобразования:
# tensor([[0, 35, 36, ...],    ← источники (source nodes)
#         [35, 36, 37, ...]])  ← назначения (target nodes)
```

**Формат edge_index: [2, num_edges]**
```
edge_index[0] = [0,  35, 36, ...]  ← индексы узлов-источников
edge_index[1] = [35, 36, 37, ...]  ← индексы узлов-назначений

Означает ребра:
  0 → 35  (table_1002132 → service_308)
  35 → 36 (service_308 → service_309)
  36 → 37 (service_309 → service_312)
  ...
```

**Зачем нужен edge_index:**
- **Главный параметр для GNN!**
- Определяет структуру графа
- DAGNN использует его для propagation (распространения информации)
- Говорит: "кто с кем связан и в каком направлении"

#### 1.3.3: Создание признаков узлов (x)

```python
features = [
    [1, 0] if path_graph.nodes[n]['type'] == 'service' 
    else [0, 1] 
    for n in node_list
]

x = torch.tensor(features, dtype=torch.float)
# Результат: tensor([
#   [0, 1],  ← table_1002132
#   [0, 1],  ← table_1002133
#   ...
#   [1, 0],  ← service_308
#   [1, 0],  ← service_309
#   ...
# ])
# Размер: [50, 2]
```

**Признаки узлов:**
- `[1, 0]` = сервис (is_service=1, is_table=0)
- `[0, 1]` = таблица (is_service=0, is_table=1)

**Зачем только 2 признака:**
- На малом датасете простые признаки работают лучше
- Больше признаков → переобучение
- DAGNN сам извлечет сложные паттерны через propagation

#### 1.3.4: Создание PyTorch Geometric Data

```python
data_pyg = Data(x=x, edge_index=edge_index)

# data_pyg содержит:
#   - x: [50, 2] - признаки всех 50 узлов
#   - edge_index: [2, 97] - 97 направленных ребер
```

**Зачем Data object:**
- Стандартный формат PyTorch Geometric
- Удобно передавать в GNN модели
- Автоматическая обработка батчей

#### 1.3.5: Подготовка контекстов и целей

```python
# Контексты - индексы последних узлов в контекстах
contexts = torch.tensor([node_map[context[-1]] for context in X_raw], dtype=torch.long)
# contexts = [0, 35, 36, ...]  - индексы узлов в node_map
# Размер: [125]

# Целевые классы - создаем отдельный маппинг ТОЛЬКО для сервисов
unique_services = sorted(set(y_raw))
# ['service_306', 'service_307', ..., 'service_397'] - 15 сервисов

service_map = {service: idx for idx, service in enumerate(unique_services)}
# {
#   'service_306': 0,
#   'service_307': 1,
#   'service_308': 2,
#   ...
#   'service_397': 14
# }

targets = torch.tensor([service_map[y] for y in y_raw], dtype=torch.long)
# targets = [2, 3, 2, ...]  - индексы классов в service_map
# Размер: [125]
```

**Важно:**
- `contexts` использует `node_map` (индексы 0-49 для всех узлов)
- `targets` использует `service_map` (индексы 0-14 для сервисов)
- Это РАЗНЫЕ пространства индексов!

**Зачем два маппинга:**
- `node_map`: для работы с графом (все 50 узлов)
- `service_map`: для предсказаний (только 15 целевых классов)

---

### Итог подготовки данных

**Подготовлены:**

1. **data_pyg** - граф с признаками:
   - `x`: [50, 2] - признаки всех узлов
   - `edge_index`: [2, 97] - структура графа

2. **contexts** - [125] - индексы последних узлов контекстов в node_map

3. **targets** - [125] - индексы целевых сервисов в service_map

4. **node_map** - 50 элементов (все узлы → индексы 0-49)

5. **service_map** - 15 элементов (сервисы → индексы 0-14)

**Split на train/test:**
```python
# 87 train / 38 test
contexts_train, contexts_test, targets_train, targets_test = train_test_split(
    contexts, targets, test_size=0.3, random_state=42
)
```

---

## 2. Входные параметры модели

### Архитектура DAGNNRecommender

```python
class DAGNNRecommender(nn.Module):
    def __init__(self, 
                 in_channels: int,      # Размерность входных признаков узлов
                 hidden_channels: int,  # Размерность скрытых слоев
                 out_channels: int,     # Количество классов для предсказания
                 K: int = 10,           # Количество propagation hops
                 dropout: float = 0.4   # Вероятность dropout
    ):
```

### Конкретные значения для нашей задачи:

```python
model = DAGNNRecommender(
    in_channels=2,        # 2 признака узла: [is_service, is_table]
    hidden_channels=64,   # Размер эмбеддингов узлов
    out_channels=15,      # 15 классов (сервисов) для предсказания
    K=10,                 # 10 hops распространения информации
    dropout=0.4           # 40% dropout для регуляризации
)
```

### Объяснение параметров:

**in_channels = 2:**
- Каждый узел описывается 2 признаками: `[is_service, is_table]`
- Простые бинарные признаки работают лучше на малом датасете
- Входной слой должен принять эти 2 признака

**hidden_channels = 64:**
- Размерность промежуточных представлений (эмбеддингов)
- Все узлы будут представлены векторами размерности 64
- Достаточно для кодирования паттернов, но не избыточно

**out_channels = 15:**
- Количество выходных нейронов = количество целевых классов
- 15 сервисов, которые можем предсказать
- Каждый нейрон = оценка для одного сервиса

**K = 10:**
- Количество шагов распространения (propagation hops)
- На каждом шаге информация распространяется на соседей
- 10 hops позволяет информации дойти далеко по графу

**dropout = 0.4:**
- При обучении случайно отключаем 40% нейронов
- Предотвращает переобучение
- Улучшает генерализацию

---

### Данные на входе forward pass

```python
def forward(self, x, edge_index, training=True):
    # x: [50, 2] - признаки всех узлов графа
    # edge_index: [2, 97] - структура графа
    # training: True/False - режим обучения/inference
```

**x: Node features [50, 2]**
```python
x = tensor([
    [0, 1],  # узел 0 (table_1002132): is_table
    [0, 1],  # узел 1 (table_1002133): is_table
    ...
    [1, 0],  # узел 35 (service_308): is_service
    [1, 0],  # узел 36 (service_309): is_service
    ...
])
```

**edge_index: Graph structure [2, 97]**
```python
edge_index = tensor([
    [0,  35, 36, 1,  35, ...],  # источники
    [35, 36, 37, 35, 40, ...]   # назначения
])

# Означает ребра:
# 0 → 35:  table_1002132 → service_308
# 35 → 36: service_308 → service_309
# 36 → 37: service_309 → service_312
# ...
```

**training: режим работы**
- `True`: обучение (dropout активен, BatchNorm в train mode)
- `False`: inference (dropout выключен, BatchNorm в eval mode)

---

## 3. Encoder Block

### Назначение
Преобразовать простые признаки узлов (2 фичи) в богатые эмбеддинги (64 dims)

### Архитектура

```python
# Слой 1: Первое преобразование
self.lin1 = nn.Linear(in_channels, hidden_channels)     # 2 → 64
self.bn1 = nn.BatchNorm1d(hidden_channels)              # Нормализация
```

### Forward pass в Encoder

```python
# ШАГ 1: Линейное преобразование
x = self.lin1(x)
# Вход:  [50, 2]
# Выход: [50, 64]
# Формула: x_out = x_in @ W + b
# W: [2, 64] - веса, b: [64] - bias

# ШАГ 2: Batch Normalization
x = self.bn1(x)
# Нормализация по батчу для стабильности обучения
# Формула: x_norm = (x - mean) / sqrt(var + eps)

# ШАГ 3: Активация ReLU
x = F.relu(x)
# Формула: relu(x) = max(0, x)
# Добавляет нелинейность

# ШАГ 4: Dropout (только при training=True)
x = F.dropout(x, p=self.dropout, training=training)
# Случайно обнуляет 40% элементов
# Предотвращает переобучение
```

**Результат после Encoder:**
```python
x: [50, 64]
# Каждый узел теперь представлен вектором размерности 64
# Вместо [is_service, is_table] имеем богатое представление
```

**Зачем Encoder:**
- Увеличивает размерность (2 → 64)
- Создает "пространство эмбеддингов" для узлов
- Позволяет модели учиться сложным паттернам

---

## 4. Residual Connection

### Назначение
Добавить "shortcut" для градиентов и стабилизировать обучение

### Архитектура

```python
self.lin2 = nn.Linear(hidden_channels, hidden_channels)  # 64 → 64
self.bn2 = nn.BatchNorm1d(hidden_channels)
```

### Forward pass с Residual

```python
# ШАГ 1: Сохраняем вход (identity)
identity = x  # [50, 64] - запоминаем

# ШАГ 2: Второе линейное преобразование
x = self.lin2(x)
# [50, 64] → [50, 64]

# ШАГ 3: Batch Normalization
x = self.bn2(x)

# ШАГ 4: Активация
x = F.relu(x)

# ШАГ 5: Residual connection (КЛЮЧЕВОЙ МОМЕНТ!)
x = x + identity
# Формула: output = F(x) + x
# где F(x) = relu(BN(Linear(x)))
```

**Визуализация Residual:**
```
      x [50, 64]
       ↓         \
    Linear        \
       ↓           \  (skip connection)
   BatchNorm       \
       ↓            \
     ReLU            \
       ↓              ↓
       ┴──────────────┴ (сложение)
             ↓
    x + identity [50, 64]
```

**Зачем нужна residual connection:**

1. **Градиенты течут лучше:**
   ```
   При backpropagation градиент может "обойти" слой
   ∂Loss/∂x_in = ∂Loss/∂x_out * (∂F/∂x + 1)
                                        ↑
                                 от skip connection
   ```

2. **Стабильность обучения:**
   - Если F(x) ≈ 0, то x ≈ identity (сохраняется вход)
   - Модель учится постепенным изменениям, не резким

3. **Лучшая оптимизация:**
   - Помогает обучать глубокие сети
   - Избегает "vanishing gradients"

**ШАГ 6: Dropout**
```python
x = F.dropout(x, p=self.dropout, training=training)
# Выход: [50, 64]
```

**Результат после Residual Block:**
```python
x: [50, 64]
# Улучшенные эмбеддинги с учетом residual обучения
```

---

## 5. DAGNN Propagation

### Что это такое

**DAGNN = Directed Acyclic Graph Neural Network**

Основан на **APPNP** (Approximate Personalized Propagation of Neural Predictions)

### Назначение
Распространить информацию между узлами по структуре графа

### Архитектура DAGNN

```python
class DAGNN(nn.Module):
    def __init__(self, in_channels: int, K: int, dropout: float = 0.5):
        super().__init__()
        self.propagation = APPNP(K=K, alpha=0.1)
        self.att = nn.Parameter(torch.Tensor(K + 1))  # Веса для разных hops
        self.dropout = dropout
```

**Параметры:**
- `K = 10`: количество propagation hops
- `alpha = 0.1`: teleport probability (вероятность "вернуться" к исходному состоянию)
- `att`: [11] - обучаемые веса внимания для каждого hop

### Формула APPNP Propagation

На каждом шаге k:

```
H^(k+1) = (1 - α) · A · H^(k) + α · H^(0)
```

**Где:**
- `H^(k)` - представления узлов на шаге k
- `H^(0)` - исходные представления (от encoder)
- `A` - нормализованная матрица смежности
- `α = 0.1` - teleport probability

**Интуиция:**
- **(1 - α) · A · H^(k)**: агрегируем от соседей (90%)
- **α · H^(0)**: "помним" исходное представление (10%)
- Баланс между информацией от соседей и собственными признаками

### Forward pass в DAGNN

```python
def forward(self, x, edge_index, training=True):
    # Вход:
    #   x: [50, 64] - эмбеддинги от encoder
    #   edge_index: [2, 97] - структура графа
    
    xs = [x]  # Список для хранения представлений на каждом hop
    
    # Веса ребер (все равны 1.0)
    edge_weight = torch.ones(edge_index.size(1), dtype=torch.float32)
    # [97] - вес для каждого из 97 ребер
    
    # ОСНОВНОЙ ЦИКЛ: K = 10 hops
    for hop in range(self.propagation.K):  # hop = 0, 1, 2, ..., 9
        
        # PROPAGATION STEP
        x = self.propagation.propagate(edge_index, x=x, edge_weight=edge_weight)
        # Формула: x_new[i] = Σ(x[j] * edge_weight[j→i]) для всех j соседей i
        
        # Dropout (только при обучении)
        if training:
            x = F.dropout(x, p=self.dropout, training=training)
        
        # Сохраняем результат этого hop
        xs.append(x)
    
    # Теперь xs содержит K+1 представлений:
    # xs[0] = H^(0) - исходное
    # xs[1] = H^(1) - после 1 hop
    # xs[2] = H^(2) - после 2 hops
    # ...
    # xs[10] = H^(10) - после 10 hops
```

### Детальный разбор одного hop

**Пример для узла service_308 (индекс 35):**

```python
# До propagation:
x[35] = [0.5, 0.2, 0.8, ..., 0.3]  # 64-мерный вектор

# Находим соседей по edge_index:
# edge_index[:, edge_index[1] == 35] → входящие ребра в узел 35
# Пусть соседи: [0, 34] (table_1002132 и service_307)

# Агрегация:
x_new[35] = (1 - 0.1) * (x[0] + x[34]) / 2 + 0.1 * x_original[35]
          = 0.9 * mean(neighbors) + 0.1 * original
          
# После propagation:
x[35] = [0.52, 0.25, 0.75, ..., 0.28]  # Обновленный вектор
```

**Что происходит:**
1. Узел агрегирует информацию от входящих соседей
2. 90% - от соседей, 10% - сохраняет свое исходное значение
3. Информация "течет" по направлению ребер

### Attention для разных hops

После K hops имеем K+1 представлений. Нужно их объединить:

```python
# Складываем все представления в один тензор
out = torch.stack(xs, dim=-1)
# Размер: [50, 64, 11]
#         узлы × dims × hops

# Вычисляем веса внимания для каждого hop
att_weights = F.softmax(self.att, dim=0)
# self.att: [11] - обучаемые параметры
# att_weights: [11] - нормализованные веса (сумма = 1)
# Пример: [0.05, 0.08, 0.12, 0.15, 0.20, 0.15, 0.10, 0.08, 0.04, 0.02, 0.01]

# Взвешенная сумма по hops
out = (out * att_weights.view(1, 1, -1)).sum(dim=-1)
# Формула: out[node, dim] = Σ(xs[k][node, dim] * att_weights[k]) for k in 0..K
# Размер: [50, 64]
```

**Зачем attention для hops:**
- Разные hops содержат разную информацию
- Hop 0: локальная информация узла
- Hop 1-2: информация от прямых соседей  
- Hop 3-5: информация от дальних узлов
- Hop 6-10: глобальная информация графа
- **Модель САМА учится**, какие hops важнее!

**Пример весов внимания:**
```
att_weights = [0.05, 0.08, 0.12, 0.15, 0.20, 0.15, 0.10, 0.08, 0.04, 0.02, 0.01]
               ↑     ↑     ↑     ↑     ↑     ↑     ↑     ↑     ↑     ↑     ↑
              hop0  hop1  hop2  hop3  hop4  hop5  hop6  hop7  hop8  hop9  hop10

Максимальный вес на hop 4 (0.20) → информация с расстояния 4 шага наиболее важна
```

**Результат DAGNN Propagation:**
```python
x: [50, 64]
# Эмбеддинги узлов с учетом ВСЕГО графа
# Каждый узел "знает" о своих соседях (через propagation)
```

---

### Подробная формула APPNP

**Инициализация:**
```
H^(0) = x  # Исходные эмбеддинги от encoder
```

**Итерация (для каждого hop k = 1, 2, ..., K):**
```
H^(k) = (1 - α) · D^(-1/2) · A · D^(-1/2) · H^(k-1) + α · H^(0)
```

**Где:**
- `A`: матрица смежности (из edge_index)
- `D`: диагональная матрица степеней узлов
- `α = 0.1`: teleport probability
- `D^(-1/2) · A · D^(-1/2)`: нормализованная матрица (симметричная нормализация)

**В коде (упрощенно):**
```python
H_0 = x  # Сохраняем исходное
H = x

for k in range(K):
    # Propagate: агрегация от соседей
    H = propagate(edge_index, H)  # Матричное умножение с нормализацией
    
    # Teleport: добавляем исходное представление
    H = (1 - alpha) * H + alpha * H_0
    # H = 0.9 * (информация от соседей) + 0.1 * (исходная информация)
```

**Пример распространения (service_308):**

```
Hop 0: [0.5, 0.2, 0.8, ...]  ← исходный эмбеддинг

Hop 1: Агрегируем от table_1002132
       [0.52, 0.25, 0.75, ...] = 0.9 * (эмбеддинг table) + 0.1 * hop0

Hop 2: Агрегируем от соседей hop 1
       [0.48, 0.30, 0.70, ...] = 0.9 * (соседи) + 0.1 * hop0

...

Hop 10: Информация от всего графа
        [0.55, 0.28, 0.72, ...]
```

---

## 6. Classifier

### Назначение
Преобразовать эмбеддинги узлов в предсказания классов

### Архитектура

```python
# Уменьшение размерности
self.lin3 = nn.Linear(hidden_channels, hidden_channels // 2)  # 64 → 32
self.bn3 = nn.BatchNorm1d(hidden_channels // 2)

# Выходной слой
self.lin_out = nn.Linear(hidden_channels // 2, out_channels)  # 32 → 15
```

### Forward pass в Classifier

```python
# ПОСЛЕ DAGNN имеем эмбеддинги всех 50 узлов: [50, 64]
x = self.dagnn(x, edge_index, training=training)

# ШАГ 1: Промежуточный слой
x = self.lin3(x)
# [50, 64] → [50, 32]
# Уменьшаем размерность для финального предсказания

# ШАГ 2: Batch Normalization
x = self.bn3(x)

# ШАГ 3: Активация
x = F.relu(x)

# ШАГ 4: Dropout
x = F.dropout(x, p=self.dropout, training=training)

# ШАГ 5: Выходной слой (ФИНАЛЬНЫЕ ЛОГИТЫ)
x = self.lin_out(x)
# [50, 32] → [50, 15]
# Для каждого из 50 узлов - 15 логитов (оценки для каждого сервиса)
```

**Результат Classifier:**
```python
x: [50, 15]
# Для каждого узла: 15 логитов (оценок) для 15 возможных сервисов

# Пример для узла 35 (service_308):
x[35] = [-2.1, -1.8, 3.5, 2.1, -0.5, ..., -1.0]
         ↑     ↑     ↑    ↑
       s_306 s_307 s_308 s_309
```

**Зачем многоступенчатый classifier:**
- `64 → 32`: промежуточное сжатие
- `32 → 15`: финальная проекция
- BatchNorm + ReLU между слоями: нелинейность и стабильность
- Dropout: регуляризация

---

## 7. Полный Forward Pass

### Шаг за шагом с конкретным примером

**Задача:**
```
Контекст: ('table_1002132', 'service_308')
Нужно предсказать: следующий сервис
```

### ШАГ 0: Подготовка входных данных

```python
# Граф (все 50 узлов)
x = tensor([
    [0, 1],  # узел 0: table_1002132
    ...
    [1, 0],  # узел 35: service_308
    ...
])  # [50, 2]

edge_index = tensor([
    [0, 35, ...],  # источники
    [35, 36, ...]  # назначения
])  # [2, 97]

# Контекст (индекс последнего узла)
context_id = 35  # service_308 в node_map
```

---

### ШАГ 1: ENCODER

```python
# 1.1: Linear projection
x = self.lin1(x)  # [50, 2] → [50, 64]

# Конкретные значения (пример):
x[0] = [0.3, 0.5, 0.2, ..., 0.8]    # table_1002132
x[35] = [0.6, 0.4, 0.7, ..., 0.5]   # service_308

# 1.2: BatchNorm + ReLU + Dropout
x = F.dropout(F.relu(self.bn1(x)), p=0.4, training=True)
# [50, 64]
```

**Что произошло:**
- Простые признаки [0,1] или [1,0] превратились в богатые векторы размерности 64
- Каждый узел теперь имеет "начальный эмбеддинг"

---

### ШАГ 2: RESIDUAL BLOCK

```python
# 2.1: Сохраняем identity
identity = x  # [50, 64]

# 2.2: Второе преобразование
x = self.lin2(x)  # [50, 64] → [50, 64]
x = self.bn2(x)
x = F.relu(x)

# 2.3: RESIDUAL CONNECTION
x = x + identity  # Ключевой момент!

# 2.4: Dropout
x = F.dropout(x, p=0.4, training=True)
# [50, 64]
```

**Что произошло:**
- Эмбеддинги "улучшены" через дополнительный слой
- Skip connection сохранил важную информацию

---

### ШАГ 3: DAGNN PROPAGATION (10 hops)

```python
# Исходное состояние
H_0 = x  # [50, 64]
xs = [H_0]  # Список всех состояний

# HOP 1
H_1 = self.propagation.propagate(edge_index, x=H_0, edge_weight=edge_weight)
# Формула: H_1 = (1-0.1) * A*H_0 + 0.1*H_0 = 0.9*A*H_0 + 0.1*H_0

# Пример для service_308 (узел 35):
# Входящие соседи: узел 0 (table_1002132)
H_1[35] = 0.9 * H_0[0] + 0.1 * H_0[35]
        = 0.9 * [0.3, 0.5, ...] + 0.1 * [0.6, 0.4, ...]
        = [0.33, 0.49, ...]

xs.append(H_1)  # Сохраняем

# HOP 2
H_2 = self.propagation.propagate(edge_index, x=H_1, edge_weight=edge_weight)
# Теперь service_308 получает информацию от соседей его соседей

xs.append(H_2)

# ... продолжаем до HOP 10 ...

# HOP 10
H_10 = self.propagation.propagate(edge_index, x=H_9, edge_weight=edge_weight)
xs.append(H_10)
```

**Что происходит на каждом hop:**

```
Hop 0: [50, 64] - исходные эмбеддинги

Hop 1: [50, 64] - узлы знают о прямых соседях
  service_308 ← table_1002132
  service_309 ← service_308

Hop 2: [50, 64] - узлы знают о соседях соседей
  service_309 ← service_308 ← table_1002132
  
Hop 3-10: [50, 64] - узлы знают о все более дальних узлах
  Информация распространяется по всему графу
```

**Формула propagate (детально):**

Для узла i:
```
h_new[i] = (1/sqrt(deg[i])) * Σ (h[j] / sqrt(deg[j])) 
                               j∈N(i)

где:
  N(i) - множество входящих соседей узла i
  deg[i] - степень узла i
  h[j] - эмбеддинг соседа j
```

С teleport:
```
h_final[i] = (1 - α) * h_new[i] + α * h_0[i]
           = 0.9 * h_new[i] + 0.1 * h_0[i]
```

### Attention aggregation

```python
# После 10 hops имеем 11 представлений
out = torch.stack(xs, dim=-1)  # [50, 64, 11]

# Веса attention (обучаемые)
att_weights = F.softmax(self.att, dim=0)
# [11] - веса для каждого hop
# Пример: [0.08, 0.09, 0.11, 0.13, 0.15, 0.14, 0.12, 0.09, 0.05, 0.03, 0.01]

# Взвешенная сумма
out = (out * att_weights.view(1, 1, -1)).sum(dim=-1)
# [50, 64]

# Для узла i и размерности d:
out[i, d] = Σ(xs[k][i, d] * att_weights[k]) for k=0..10
```

**Пример для service_308:**
```python
# Эмбеддинг по размерности 0:
xs[0][35, 0] = 0.6   (hop 0, исходный)
xs[1][35, 0] = 0.55  (hop 1, с соседями)
xs[2][35, 0] = 0.52  (hop 2)
...
xs[10][35, 0] = 0.48 (hop 10)

# Attention weights:
att = [0.08, 0.09, 0.11, ..., 0.01]

# Итоговое значение:
out[35, 0] = 0.6*0.08 + 0.55*0.09 + 0.52*0.11 + ... + 0.48*0.01
           = 0.534  # Взвешенная комбинация всех hops
```

**Результат DAGNN Propagation:**
```python
x: [50, 64]
# Каждый узел теперь содержит:
# - Свою собственную информацию
# - Информацию от соседей (1-2 hops)
# - Информацию от дальних узлов (3-10 hops)
# - Глобальную структуру графа
```

**Зачем DAGNN Propagation:**
- **Главная фишка модели!**
- Узлы обмениваются информацией через граф
- service_308 "узнает" о table_1002132 и service_309
- Модель понимает контекст и структуру зависимостей

---

### ШАГ 4: CLASSIFIER

```python
# Вход: эмбеддинги всех 50 узлов после DAGNN
x: [50, 64]

# 4.1: Промежуточный слой
x = self.lin3(x)  # [50, 64] → [50, 32]
x = self.bn3(x)
x = F.relu(x)
x = F.dropout(x, p=0.4, training=True)

# 4.2: Выходной слой
x = self.lin_out(x)  # [50, 32] → [50, 15]
```

**Результат:**
```python
x: [50, 15]
# Для каждого узла - 15 логитов (оценок) для 15 сервисов

# Пример для service_308 (узел 35):
x[35] = [-2.1, -1.8, 3.5, 2.1, -0.5, -1.2, -0.8, ...]
         ↑     ↑     ↑    ↑
       s_306 s_307 s_308 s_309
       
# Высокий логит (3.5) для service_308
# Средний логит (2.1) для service_309
# Низкие логиты для остальных
```

---

### ШАГ 5: Выбор предсказания для контекста

```python
# В обучении/inference мы не используем ВСЕ 50 узлов
# Берем только узлы из контекстов!

# Контексты train (индексы последних узлов)
contexts_train = [0, 35, 36, ...]  # [87]

# Берем логиты только для контекстных узлов
out = x[contexts_train]  # [87, 15]

# Пример:
# Контекст 0: ('table_1002132', 'service_308')
# context_id = 35 (service_308)
# out[0] = x[35] = [-2.1, -1.8, 3.5, 2.1, ...]
```

**Зачем берем только контекстные узлы:**
- У нас 87 обучающих примеров
- Каждый пример заканчивается определенным узлом (контекст)
- Предсказание делаем "от имени" этого узла
- Не нужны предсказания для всех 50 узлов, только для 87 контекстных

---

### ШАГ 6: Обучение (Training)

```python
# Логиты для train примеров
out = model(data_pyg.x, data_pyg.edge_index, training=True)[contexts_train]
# [87, 15]

# Целевые классы
targets_train = [2, 3, 2, 8, ...]  # [87] - индексы в service_map

# Loss function
loss = F.cross_entropy(out, targets_train)
```

**Что делает Cross Entropy:**

```python
# Для каждого примера:
# 1. Softmax к логитам
probs = softmax(out[i])
# [-2.1, -1.8, 3.5, 2.1, ...] → [0.01, 0.02, 0.68, 0.19, ...]

# 2. Берем вероятность правильного класса
true_class = targets_train[i]  # Например, 3 (service_309)
p_true = probs[true_class]      # 0.19

# 3. Вычисляем loss
loss_i = -log(p_true) = -log(0.19) = 1.66

# 4. Усредняем по всем 87 примерам
loss = mean(loss_i for all i)
```

**Backpropagation:**
```python
loss.backward()  # Вычисляем градиенты
optimizer.step()  # Обновляем веса
```

---

### ШАГ 7: Inference (Предсказание)

```python
model.eval()  # Переключаем в режим оценки

with torch.no_grad():  # Без вычисления градиентов
    # Forward pass
    test_output = model(data_pyg.x, data_pyg.edge_index, training=False)
    # [50, 15] - логиты для всех узлов
    
    # Берем только test контексты
    test_output = test_output[contexts_test]
    # [38, 15] - логиты для 38 тестовых примеров
    
    # Вероятности
    probs = F.softmax(test_output, dim=1)
    # [38, 15] - вероятности для каждого сервиса
    
    # Предсказания
    preds = test_output.argmax(dim=1)
    # [38] - индексы классов с максимальной вероятностью
```

**Конкретный пример:**

```python
# Тестовый пример 0:
# Контекст: ('table_1002132', 'service_308')
# context_id = 35

# Логиты
logits = test_output[0] = [-2.1, -1.8, 3.5, 2.1, -0.5, ...]

# Вероятности после softmax
probs = [0.01, 0.02, 0.68, 0.19, 0.01, ...]
         ↑     ↑     ↑     ↑
       s_306 s_307 s_308 s_309
       (0)   (1)   (2)   (3)

# Предсказание
pred = argmax(probs) = 2  # индекс в service_map

# Расшифровка
predicted_service = service_map_inverse[2] = 'service_308'

# Правильный ответ
true_service = 'service_309' (класс 3)

# Результат: ❌ Неправильно (mode collapse!)
```

---

## 8. Полная схема данных и их применение

### 8.1: Данные в процессе работы

```
┌─────────────────────────────────────────────────────────┐
│  ПОДГОТОВКА ДАННЫХ                                      │
├─────────────────────────────────────────────────────────┤
│                                                          │
│  Композиции (943) → Уникальные пути (106)              │
│         ↓                                                │
│  Обучающие примеры (125):                               │
│    X_raw: [('table_X',), ('table_X', 'service_308'), ...]│
│    y_raw: ['service_308', 'service_309', ...]           │
│         ↓                                                │
│  Граф (NetworkX):                                        │
│    - 50 узлов с атрибутами (type: service/table)       │
│    - 97 направленных ребер                              │
│         ↓                                                │
│  PyTorch Geometric Data:                                 │
│    - x: [50, 2] - признаки узлов                       │
│    - edge_index: [2, 97] - структура графа             │
│         ↓                                                │
│  Маппинги:                                               │
│    - node_map: {узел → индекс 0-49}                    │
│    - service_map: {сервис → индекс 0-14}               │
│         ↓                                                │
│  Индексы:                                                │
│    - contexts: [125] - индексы узлов в node_map        │
│    - targets: [125] - индексы сервисов в service_map   │
│         ↓                                                │
│  Train/Test Split:                                       │
│    - 87 train / 38 test                                 │
│                                                          │
└─────────────────────────────────────────────────────────┘
         ↓
┌─────────────────────────────────────────────────────────┐
│  FORWARD PASS В МОДЕЛИ                                  │
├─────────────────────────────────────────────────────────┤
│                                                          │
│  INPUT:                                                  │
│    x: [50, 2] - признаки всех узлов                    │
│    edge_index: [2, 97] - структура графа               │
│         ↓                                                │
│  ┌────────────────────────────┐                         │
│  │  ENCODER BLOCK             │                         │
│  │  Linear(2 → 64)            │ ← in_channels          │
│  │  + BatchNorm + ReLU        │                         │
│  └────────────────────────────┘                         │
│         ↓                                                │
│    [50, 64]                                             │
│         ↓                                                │
│  ┌────────────────────────────┐                         │
│  │  RESIDUAL BLOCK            │                         │
│  │  Linear(64 → 64)           │ ← hidden_channels      │
│  │  + BatchNorm + ReLU        │                         │
│  │  + Skip Connection         │ ← RESIDUAL!            │
│  └────────────────────────────┘                         │
│         ↓                                                │
│    [50, 64]                                             │
│         ↓                                                │
│  ┌────────────────────────────┐                         │
│  │  DAGNN PROPAGATION         │                         │
│  │  K=10 hops                 │ ← K                    │
│  │  α=0.1 teleport            │                         │
│  │  + Attention weights       │ ← обучаемые веса       │
│  └────────────────────────────┘                         │
│         ↓                                                │
│    [50, 64] с графовой информацией                     │
│         ↓                                                │
│  ┌────────────────────────────┐                         │
│  │  CLASSIFIER                │                         │
│  │  Linear(64 → 32)           │                         │
│  │  + BatchNorm + ReLU        │                         │
│  │  Linear(32 → 15)           │ ← out_channels         │
│  └────────────────────────────┘                         │
│         ↓                                                │
│  OUTPUT:                                                 │
│    [50, 15] - логиты для всех узлов                    │
│         ↓                                                │
│  Выбор контекстных узлов:                               │
│    out[contexts_train] → [87, 15]                       │
│                                                          │
└─────────────────────────────────────────────────────────┘
         ↓
┌─────────────────────────────────────────────────────────┐
│  ОБУЧЕНИЕ / ПРЕДСКАЗАНИЕ                                │
├─────────────────────────────────────────────────────────┤
│                                                          │
│  Training:                                               │
│    loss = CrossEntropy(out, targets_train)              │
│    loss.backward() → обновление весов                   │
│                                                          │
│  Inference:                                              │
│    probs = softmax(out) → [87, 15]                     │
│    preds = argmax(probs) → [87]                        │
│    predicted_services = [service_map_inverse[p] for p]  │
│                                                          │
└─────────────────────────────────────────────────────────┘
```

---

## 9. Применение данных по этапам

### Таблица использования данных

| Данные | Где создаются | Где используются | Зачем |
|--------|---------------|------------------|-------|
| **x** [50, 2] | prepare_pytorch_geometric_data | Вход модели | Исходные признаки узлов |
| **edge_index** [2, 97] | prepare_pytorch_geometric_data | DAGNN propagation | Структура графа |
| **node_map** | prepare_pytorch_geometric_data | Создание contexts | Маппинг узлов → индексы |
| **service_map** | prepare_pytorch_geometric_data | Создание targets, финальное декодирование | Маппинг сервисов → классы |
| **contexts** [125] | prepare_pytorch_geometric_data | Выбор узлов для предсказания | Индексы последних узлов контекстов |
| **targets** [125] | prepare_pytorch_geometric_data | Loss function | Правильные ответы |
| **H^(0)** [50, 64] | Encoder | DAGNN (исходное состояние) | Начальные эмбеддинги |
| **H^(k)** [50, 64] | DAGNN (каждый hop) | Attention aggregation | Эмбеддинги после k hops |
| **att_weights** [11] | DAGNN (обучаемые) | Взвешенная сумма hops | Веса важности для каждого hop |
| **out** [87, 15] | Classifier → contexts_train | CrossEntropyLoss | Логиты для обучения |

---

## 10. Математические формулы (полные)

### 10.1: Encoder

```
H^(0) = ReLU(BN(x · W_1 + b_1))

где:
  x: [50, 2] - входные признаки
  W_1: [2, 64] - веса линейного слоя
  b_1: [64] - bias
  BN: Batch Normalization
  H^(0): [50, 64] - начальные эмбеддинги
```

### 10.2: Residual Block

```
H^(res) = ReLU(BN(H^(0) · W_2 + b_2)) + H^(0)

где:
  W_2: [64, 64] - веса
  + H^(0) - skip connection (residual)
```

### 10.3: DAGNN Propagation (APPNP)

**Для каждого hop k = 1, 2, ..., K:**

```
H^(k) = (1 - α) · D^(-1/2) · A · D^(-1/2) · H^(k-1) + α · H^(0)

где:
  A[i,j] = 1, если есть ребро i → j, иначе 0
  D[i,i] = степень узла i (количество входящих + исходящих ребер)
  α = 0.1 - teleport probability
  H^(0) - исходные эмбеддинги (от encoder + residual)
```

**Для узла i:**
```
h_i^(k) = (1 - α) · Σ(h_j^(k-1) / sqrt(deg_i * deg_j)) + α · h_i^(0)
                    j∈N_in(i)

где:
  N_in(i) - входящие соседи узла i
  deg_i - степень узла i
```

**Attention aggregation:**
```
H^(final) = Σ(w_k · H^(k)) для k = 0, 1, ..., K

где:
  w_k = softmax(att)[k] - обучаемые веса attention
  att: [K+1] - обучаемые параметры
```

### 10.4: Classifier

```
logits = Linear_out(ReLU(BN(Linear_3(H^(final)))))

Детально:
  z_1 = H^(final) · W_3 + b_3      # [50, 64] → [50, 32]
  z_2 = ReLU(BN(z_1))              # Активация + нормализация
  logits = z_2 · W_out + b_out     # [50, 32] → [50, 15]

где:
  W_3: [64, 32]
  W_out: [32, 15]
```

### 10.5: Loss Function

```
L = CrossEntropy(logits[contexts], targets)
  = -(1/N) Σ log(softmax(logits[i])[targets[i]])
  
где:
  N = 87 - количество train примеров
  logits[i]: [15] - логиты для примера i
  targets[i] - правильный класс (0-14)
```

---

## 11. Конкретный пример: полный проход

### Входные данные

```python
# Обучающий пример:
context = ('table_1002132', 'service_308')
target = 'service_309'

# После маппинга:
context_node_id = 35  # service_308 в node_map
target_class_id = 3   # service_309 в service_map
```

### Проход через модель

```
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
ENCODER
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Вход: x[35] = [1, 0]  (service_308 - это сервис)
      ↓ Linear(2 → 64)
      [0.6, 0.4, 0.7, 0.3, ..., 0.5]  (64 dims)
      ↓ BatchNorm + ReLU + Dropout
H^(0)[35] = [0.62, 0.38, 0.72, ..., 0.48]

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
RESIDUAL BLOCK
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

      ↓ Linear(64 → 64) + BN + ReLU
F(x) = [0.65, 0.42, 0.68, ..., 0.52]
      ↓ + identity
H^(res)[35] = [0.63, 0.40, 0.70, ..., 0.50]

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
DAGNN PROPAGATION
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Hop 0: [0.63, 0.40, 0.70, ..., 0.50]

Hop 1: Агрегирует от table_1002132 (узел 0)
       [0.58, 0.45, 0.65, ..., 0.48]
       = 0.9*(эмб table) + 0.1*(hop 0)

Hop 2: Агрегирует от соседей hop 1
       [0.60, 0.42, 0.68, ..., 0.52]

Hop 3: Информация распространяется дальше
       [0.59, 0.44, 0.66, ..., 0.51]

...

Hop 10: Информация от всего графа
        [0.61, 0.43, 0.67, ..., 0.49]

Attention aggregation:
  att_weights = [0.08, 0.09, 0.11, 0.13, 0.15, ...]
  H^(final)[35] = weighted_sum(all hops)
                = [0.605, 0.425, 0.675, ..., 0.495]

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
CLASSIFIER
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

      ↓ Linear(64 → 32) + BN + ReLU
      [0.55, 0.38, 0.62, ..., 0.44]  (32 dims)
      ↓ Linear(32 → 15)
logits[35] = [-2.1, -1.8, 3.5, 2.1, -0.5, -1.2, ...]
              ↑     ↑     ↑    ↑
            s_306 s_307 s_308 s_309
            (0)   (1)   (2)   (3)

      ↓ Softmax
probs[35] = [0.01, 0.02, 0.68, 0.19, 0.01, ...]
            
      ↓ Argmax
prediction = 2 → service_308

Правильный ответ: 3 (service_309)
Результат: ❌ Неправильно
```

---

## 12. Почему каждый компонент важен

### Encoder (Linear 2→64)
**Зачем:** Увеличить размерность для кодирования сложных паттернов
**Без него:** Модель работала бы с 2-мерными векторами (недостаточно capacity)

### Residual Connection
**Зачем:** Стабилизировать обучение и помочь градиентам
**Без него:** Градиенты могут "затухать", сложнее обучать глубокую сеть

### DAGNN Propagation
**Зачем:** **КЛЮЧЕВОЙ КОМПОНЕНТ!** Распространить информацию по графу
**Без него:** Узлы "не знают" о соседях, только о своих признаках

### Attention для hops
**Зачем:** Объединить информацию с разных расстояний
**Без него:** Использовали бы только последний hop (теряем промежуточную информацию)

### Classifier
**Зачем:** Преобразовать эмбеддинги в предсказания классов
**Без него:** Имели бы только векторы, а не вероятности классов

---

## 13. Параметры модели

### Количество параметров DAGNN-Improved:

```python
# Encoder
lin1: 2 × 64 + 64 = 192
bn1: 64 × 2 = 128

# Residual
lin2: 64 × 64 + 64 = 4160
bn2: 64 × 2 = 128

# DAGNN
att: 11 (веса для hops)
APPNP: нет обучаемых параметров (только нормализация)

# Classifier
lin3: 64 × 32 + 32 = 2080
bn3: 32 × 2 = 64
lin_out: 32 × 15 + 15 = 495

ИТОГО ≈ 7,312 параметров
```

**Сравнение:**
- GRU4Rec (50 классов): ≈ 50,000 параметров
- DAGNN (15 классов): ≈ 7,300 параметров
- **DAGNN в 7 раз меньше!** → меньше переобучение

---

## 14. Оптимальные параметры (найдены экспериментально)

### DAGNN-Improved (лучшая конфигурация)

```python
model = DAGNNRecommender(
    in_channels=2,
    hidden_channels=128,      # Больше чем базовый DAGNN (64)
    out_channels=15,
    K=15,                     # Больше hops (vs 10)
    dropout=0.5               # Больше dropout (vs 0.4)
)

optimizer = Adam(
    model.parameters(),
    lr=0.0005,                # Меньше lr (vs 0.001)
    weight_decay=5e-4         # Больше weight decay (vs 1e-4)
)

# Loss: Focal Loss (70%) + Label Smoothing (30%)
focal_loss = FocalLoss(gamma=2.0, alpha=class_weights)
smooth_loss = LabelSmoothingLoss(smoothing=0.1)
loss = 0.7 * focal_loss + 0.3 * smooth_loss

epochs = 200
```

**Результат: nDCG = 0.6814** 🏆

---

## 15. Выводы

### Ключевые компоненты DAGNN:

1. **edge_index** - самый важный параметр
   - Определяет структуру графа
   - Используется в propagation
   - Без него DAGNN не может работать

2. **Propagation** - ядро модели
   - Распространяет информацию по графу
   - K hops → информация от дальних узлов
   - Teleport → сохранение исходной информации

3. **Attention** - умное объединение
   - Модель сама учится, какие hops важны
   - Разные расстояния дают разную информацию

4. **Residual** - стабильность
   - Помогает обучению
   - Сохраняет важную информацию

### Почему DAGNN работает на DAG:

- ✅ Направленная propagation (по направлению ребер)
- ✅ Учитывает структуру зависимостей
- ✅ Многошаговое распространение (K hops)
- ✅ Teleport предотвращает "размазывание" информации

### Оптимальная конфигурация для малого датасета:

- ✅ 15 выходных классов (только сервисы)
- ✅ Направленный граф БЕЗ модификаций (97 ребер)
- ✅ Простые признаки (2 фичи)
- ✅ Residual connections
- ✅ Focal Loss для борьбы с несбалансированностью

---

*Файл создан: 2025-10-24*  
*Версия модели: DAGNN v4.2 (Optimized)*

