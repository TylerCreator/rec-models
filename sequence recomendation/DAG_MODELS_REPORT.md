## Directed Graph-Based Sequence Models

### 1. DirectedDAGNN
**Данные.** Из `compositionsDAG.json` извлекаются все реальные простые пути каждой композиции (источник → сток). Каждая пара *(контекст, следующий сервис)* становится образцом. Узлы кодируются как `service_*` или `table_*`, на глобальном графе каждому узлу соответствует вектор `[is_service, is_table]`.

**Архитектура.**
- `Linear(2 → H)` + BatchNorm + ReLU.
- Многократная направленная пропагация (DirectedAPPNP): сообщения идут только по рёбрам `u→v`, веса фиксированы как `1/out_degree(u)`, после каждого шага применяется dropout, а hop-wise attention учится выбирать релевантную глубину.
- `Linear(H → H/2)` + BatchNorm + ReLU + Dropout.
- `Linear(H/2 → |services|)`.

**Обучение.** Для каждого образца берём логиты строки, соответствующей последнему узлу контекста. Loss — cross-entropy, оптимизатор Adam, 50 эпох. Модель оценивается по accuracy, macro-F1, precision, recall, nDCG.

### 2. DeepDAG (2022-like, per-DAG)
**Данные.** Для каждой композиции строится собственный PyTorch Geometric граф (узлы+рёбра). Контексты привязаны к конкретному DAG; train/test split хранит индекс композиции.

**Архитектура.**
- Depth-aware encoder: линейная проекция `[features || depth_embed] → H`, далее стек self-attention блоков (MultiheadAttention + FFN).
- Hop-level attention (усреднение выходов блоков с softmax).
- Классификатор `Linear(H → H/2 → |services|)` с LayerNorm+Dropout.

**Обучение.** Перебираем композиции, для каждой считаем loss только по узлам, присутствующим в контекстах этой композиции. Это повторяет схему оригинального DeepDAG: модель “адаптируется” к конкретному DAG. На тесте аналогично прогоняем каждый граф и берём logits нужных узлов.

### 3. DAG-GNN (Yu et al., 2019, адаптировано)
**Данные.** Тот же глобальный граф, что и для DirectedDAGNN, структура рёбер фиксирована. Никаких learnable mask’ов на рёбрах.

**Архитектура.**
- `Linear(2 → H)`.
- Несколько блоков:
  ```
  parent_msg = Σ_{parent} h_parent W_parent / out_deg(parent)
  child_msg  = Σ_{child}  h_child  W_child  / out_deg(child)
  self_msg   = h W_self
  h ← LayerNorm( h + Dropout( GELU(self_msg + parent_msg + child_msg) ) )
  ```
- Голова `Linear(H → H) → GELU → Dropout → Linear(H → |services|)`.

**Обучение/оценка.** Идентичны DirectedDAGNN, но forward учитывает как родителей, так и детей, что даёт более богатый контекст при жёстко фиксированной структуре.

### 4. GRU4Rec-DAG (Branch-aware)
**Данные.** Последовательности формируются из путей. Вход — индексы узлов (padding слева). Дополнительно заранее сохраняются:
- `successors[node]` — допустимые сервисы-дети (для маски на выходе).
- `successor_nodes[node]` — все дети (service/table) для ветвевой агрегации.

**Архитектура.**
- Embedding `(num_nodes+1, d)`.
- Branch aggregator: на каждом timestep вычисляется средний embedding всех детей текущего узла.
- GRU получает конкатенацию `[node_emb_t || branch_emb_t]`, что позволяет ему “видеть” ветвления DAG.
- Выходной `Linear(hidden → |services|)` + маска, запрещающая предсказывать сервисы вне исходящих рёбер.

**Обучение.** Cross-entropy поверх masked logits, grad clipping. На тесте — те же последовательности + маска, метрики как у графовых моделей.

### Формирование train/test
1. `extract_paths_from_compositions` собирает все пути `(path, composition_id)`.
2. `create_training_pairs` → `(contexts, targets, composition_id_of_sample)`.
3. `train_test_split` делит примеры на обучающие и тестовые. Для sequential моделей параллельно генерируются padded последовательности, длины и `last_node_idx`.
4. Графовые модели используют индексы узлов в глобальном графе; DeepDAG — per-composition графы; GRU4Rec-DAG — sequences + branch info.

### Размерности (по умолчанию)
- Hidden размер GNN: 64.
- Embedding узлов для GRU: 64; вход GRU = 128 (конкатенация).
- GRU hidden = 128, num_layers = 2.
- Кол-во классов = число уникальных сервисов в данных (≈15).


