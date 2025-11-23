# DA-GCN: Directed Acyclic Graph Convolutional Network

## Обзор

**DA-GCN (Directed Acyclic Graph Convolutional Network)** — это современная архитектура нейронной сети для рекомендаций с учетом множественных типов поведения пользователей, представленная в статье "Multi-Behavior Recommendation with Personalized Directed Acyclic Behavior Graphs" (Zhu et al., ACM TOIS 2024).

## Основная идея

DA-GCN расширяет концепцию монотонных цепочек поведения до **персонализированных направленных ациклических графов поведения (P-DABG)**, которые моделируют:
- Персонализированные паттерны взаимодействия пользователей
- Внутренние свойства элементов
- Зависимости между различными типами поведения

## Архитектура

### 1. Directed Edge Encoder (Кодировщик направленных рёбер)

Каждое направленное ребро кодируется с использованием GCN-подобного подхода:

```python
h_src = transform_src(h[source_node])  # Преобразование исходного узла
h_tgt = transform_tgt(h[target_node])  # Преобразование целевого узла
edge_embedding = h_src + h_tgt         # Комбинация представлений
```

**Особенности:**
- Отдельные веса для исходных и целевых узлов
- Учет направленности связей в графе
- Взвешивание рёбер по частоте появления в композициях

### 2. Attentive Aggregation (Внимательная агрегация)

Модель использует механизм многоголового внимания для агрегации информации от предшественников:

```python
#Self-attention на агрегированных признаках
attn_out = MultiheadAttention(node_features, edge_aggregated, edge_aggregated)
h = LayerNorm(h + attn_out)
```

**Преимущества:**
- Автоматическое взвешивание вклада предшествующих узлов
- Адаптивная агрегация в зависимости от контекста
- Обучаемые веса внимания

### 3. Multi-Layer Propagation (Многослойное распространение)

Модель использует несколько слоёв DA-GCN для постепенного уточнения представлений:

```
Input → Projection → DA-GCN Layer 1 → DA-GCN Layer 2 → DA-GCN Layer 3 → Output
```

Каждый слой включает:
- Кодирование направленных рёбер
- Внимательную агрегацию
- Feed-forward сеть с GELU активацией
- Layer Normalization и residual connections

### 4. Layer-wise Attention (Послойное внимание)

Финальное представление комбинирует выходы всех слоёв с обучаемыми весами:

```python
# Объединение представлений из всех слоёв
layer_outputs = [h0, h1, h2, h3]  # Начальное + 3 слоя DA-GCN
attention_weights = softmax(attention_network(concat(layer_outputs)))
final_repr = weighted_sum(layer_outputs, attention_weights)
```

## Ключевые компоненты реализации

### DAGCNLayer

Основной строительный блок модели:

```python
class DAGCNLayer(nn.Module):
    def __init__(self, hidden, num_heads=4, dropout=0.3):
        # Трансформации для направленных рёбер
        self.edge_src_transform = nn.Linear(hidden, hidden)
        self.edge_tgt_transform = nn.Linear(hidden, hidden)
        
        # Multi-head attention для агрегации
        self.attention = nn.MultiheadAttention(hidden, num_heads, dropout)
        
        # Feed-forward сеть
        self.ffn = nn.Sequential(
            nn.Linear(hidden, hidden * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden * 2, hidden)
        )
        
        # Нормализация
        self.norm1 = nn.LayerNorm(hidden)
        self.norm2 = nn.LayerNorm(hidden)
```

### DAGCNRecommender

Полная модель рекомендаций:

```python
class DAGCNRecommender(nn.Module):
    def __init__(self, in_channels, hidden, out_channels, 
                 num_layers=3, num_heads=4, dropout=0.3):
        # Входная проекция
        self.input_proj = nn.Sequential(
            nn.Linear(in_channels, hidden),
            nn.LayerNorm(hidden),
            nn.GELU(),
            nn.Dropout(dropout)
        )
        
        # Стек DA-GCN слоёв
        self.dagcn_layers = nn.ModuleList([
            DAGCNLayer(hidden, num_heads, dropout)
            for _ in range(num_layers)
        ])
        
        # Послойное внимание
        self.layer_attention = nn.Sequential(...)
        
        # Выходной классификатор
        self.output_head = nn.Sequential(...)
```

## Особенности адаптации для sequence recommendation

В нашей реализации DA-GCN адаптирован для задачи рекомендации последовательностей в DAG:

1. **Входные признаки узлов**: One-hot кодирование типа узла (service/table)
2. **Взвешивание рёбер**: Использование частоты появления рёбер в композициях
3. **Node-level классификация**: Предсказание следующего сервиса на основе контекста

## Результаты сравнения

На датасете compositionsDAG.json (150 эпох):

| Модель          | Accuracy | NDCG@10 | F1-Score | Особенности                    |
|-----------------|----------|---------|----------|--------------------------------|
| Popularity      | 0.4787   | 0.6106  | 0.0540   | Baseline                       |
| DirectedDAGNN   | 0.5213   | 0.7566  | 0.1805   | APPNP-style propagation        |
| **DA-GCN**      | **0.5115** | **0.7530** | **0.1138** | **Attentive edge encoding**    |
| DeepDAG2022     | 0.5213   | 0.7571  | 0.1805   | Depth-aware attention          |
| DAG-GNN         | 0.5213   | 0.7571  | 0.1805   | Learnable edge weights         |
| DAGNN2021       | 0.5213   | 0.7571  | 0.1805   | Topological processing         |
| GRU4Rec         | 0.5344   | 0.7674  | 0.2241   | Sequential RNN                 |

## Анализ производительности

### Сильные стороны DA-GCN:

1. **Направленное кодирование рёбер**: Явное моделирование направленности связей в графе
2. **Attentive aggregation**: Адаптивная агрегация информации от предшественников
3. **Multi-layer refinement**: Постепенное уточнение представлений через несколько слоёв
4. **Layer-wise attention**: Комбинирование информации с разных уровней абстракции

### Особенности поведения:

1. **Конкурентоспособный NDCG**: DA-GCN показывает NDCG@10 = 0.7530, что близко к лучшим GNN моделям (0.7571)
2. **Средняя точность**: Accuracy = 0.5115, что на уровне других GNN моделей
3. **Более низкий F1**: F1 = 0.1138, что может указывать на менее сбалансированные предсказания

### Время обучения:

- **~0.01s на эпоху** (150 эпох = ~1.2 секунды)
- Быстрее чем DAGNN2021 (~0.03s на эпоху)
- Сопоставимо с DirectedDAGNN и DeepDAG2022
- Значительно быстрее чем GRU4Rec (~0.08s на эпоху)

## Гиперпараметры

Рекомендуемые настройки:

```python
model = DAGCNRecommender(
    in_channels=2,        # Размерность входных признаков
    hidden=64,            # Размерность скрытых представлений
    out_channels=15,      # Количество классов (сервисов)
    num_layers=3,         # Количество слоёв DA-GCN
    num_heads=4,          # Количество голов внимания
    dropout=0.4           # Dropout rate
)

optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
```

## Применимость

DA-GCN особенно эффективен для задач, где:

1. **Направленность связей критична**: Важен порядок и направление взаимодействий
2. **Множественные типы поведения**: Модель изначально разработана для multi-behavior recommendation
3. **Персонализация**: Нужно учитывать индивидуальные паттерны пользователей
4. **Иерархические зависимости**: Присутствуют сложные иерархические отношения

## Ссылки

- **Статья**: Zhu, X., Lin, F., Zhao, Z., Xu, T., Zhao, X., Yin, Z., Li, X., & Chen, E. (2024). "Multi-Behavior Recommendation with Personalized Directed Acyclic Behavior Graphs". ACM Transactions on Information Systems.
- **Репозиторий**: https://github.com/Richard5White/da-gcn5
- **Конференция**: ACM TOIS 2024

## Использование в directed_dag_models.py

```bash
# Базовый запуск
python directed_dag_models.py --data compositionsDAG.json --epochs 150

# Кастомизация гиперпараметров
python directed_dag_models.py \
    --data compositionsDAG.json \
    --epochs 150 \
    --hidden 64 \
    --dropout 0.4 \
    --lr 0.001
```

Модель автоматически включена в сравнение и выводится как "DA-GCN" в итоговой таблице результатов.

## Заключение

DA-GCN представляет собой современный подход к моделированию направленных графов с использованием:
- Направленного кодирования рёбер
- Механизма внимания для агрегации
- Многослойного уточнения представлений
- Послойного внимания для комбинирования информации

Модель показывает конкурентоспособные результаты на задаче sequence recommendation в DAG и может быть особенно полезна для задач с множественными типами поведения и сложными иерархическими зависимостями.

