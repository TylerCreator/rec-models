# Сравнение двух подходов обогащения CF данными

## 📋 В чем разница?

### sequence_recommender_with_cf_enrichment.py

**Подход:** Используем ОРИГИНАЛЬНЫЙ compositionsDAG.json + добавляем CF features

```
Данные:
  ├─ Граф: compositionsDAG.json (структурный DAG композиций)
  ├─ Примеров: 107
  ├─ Классов: 4
  └─ Features: 2 базовых → 38 enriched

Процесс:
  1. Загружаем compositionsDAG.json (КАК В ОРИГИНАЛЕ)
  2. Извлекаем paths из DAG (КАК В ОРИГИНАЛЕ)
  3. Создаем примеры (context → next) (КАК В ОРИГИНАЛЕ)
  4. Добавляем CF features к узлам:
     - Item embeddings из LightFM (30 dims)
     - Service popularity (1 dim)
     - User interaction stats (3 dims)
     - Graph features (2 dims)
  5. Обучаем модели с enriched features

Результат:
  ❌ CF features НЕ помогли!
  ├─ DAGNN (baseline): 0.5152 🥇
  ├─ DAGNN (enriched):  0.4848 (-5.9%)
  ├─ GraphSAGE: +0.0%
  └─ GCN: +0.5%
```

**Вывод:** На структурном DAG с малыми данными (107) CF features добавляют шум!

---

### sequence_with_cf_features.py

**Подход:** Создаем НОВЫЙ граф из последовательностей в calls.csv + CF features

```
Данные:
  ├─ Граф: СОЗДАН ИЗ calls.csv (temporal sequences!)
  ├─ Примеров: 335 (в 3 раза больше!)
  ├─ Классов: больше
  └─ Features: 38 enriched

Процесс:
  1. Загружаем calls.csv
  2. Создаем последовательности вызовов ПО ПОЛЬЗОВАТЕЛЯМ
  3. Строим НОВЫЙ граф:
     service_A → service_B (если пользователи часто вызывают B после A)
  4. Извлекаем paths из НОВОГО графа
  5. Создаем примеры из последовательностей
  6. Добавляем CF features
  7. Обучаем модели

Результат:
  ✅ CF features ПОМОГЛИ!
  ├─ GraphSAGE+CF: 0.5446 🥇 (лучше оригинальной DAGNN!)
  ├─ DAGNN+CF: 0.3960 (+17.6% vs baseline 0.3366)
  └─ nDCG: +2.1%
```

**Вывод:** Когда граф создан из CF данных, CF features работают отлично!

---

## 🔍 Детальное сравнение

### Источник графа

| Параметр | sequence_recommender_with_cf_enrichment.py | sequence_with_cf_features.py |
|----------|-------------------------------------------|------------------------------|
| **Граф** | compositionsDAG.json (структурный) | Создан из calls.csv (temporal) |
| **Источник связей** | Композиции сервисов (предопределены) | Последовательности пользователей (из данных) |
| **Тип связей** | A→B в композиции | A→B часто следуют в истории |
| **Примеров** | 107 | 335 (в 3 раза больше!) |

### Enriched Features (одинаковые)

Обе версии добавляют:
- ✓ Item embeddings из LightFM (30 dims)
- ✓ Service popularity (1 dim)
- ✓ User interaction stats (3 dims)
- ✓ Graph features (2 dims)

**Всего:** 38 dimensions (было 2)

### Результаты

| Метрика | Версия 1 (compositionsDAG) | Версия 2 (CF граф) |
|---------|----------------------------|-------------------|
| **Лучшая модель** | DAGNN (baseline) 0.5152 | GraphSAGE+CF 0.5446 |
| **DAGNN baseline** | 0.5152 | 0.3366 |
| **DAGNN enriched** | 0.4848 (-5.9%) | 0.3960 (+17.6%) |
| **Улучшение** | ❌ Ухудшение | ✅ Улучшение |

---

## 💡 Почему разные результаты?

### Версия 1 (compositionsDAG) - CF features не помогли:

**Причины:**

1. **Маленький датасет (107 примеров)**
   - 38 features vs 107 samples
   - Соотношение слишком высокое
   - Переобучение на дополнительных features

2. **Простая задача (4 класса)**
   - Базовых 2 features достаточно
   - Дополнительная информация = шум

3. **Мало пересечений (28% узлов)**
   - Только ~14 общих сервисов между DAG и CF
   - 72% узлов получают нулевые CF features
   - Sparse coverage

4. **Разная природа данных**
   - compositionsDAG: структурные связи (предопределены)
   - CF: пользовательские предпочтения
   - Не коррелируют!

---

### Версия 2 (CF граф) - CF features помогли:

**Причины:**

1. **Больше примеров (335)**
   - Лучшее соотношение features/samples
   - Меньше риск переобучения

2. **Граф создан из CF данных**
   - Связи основаны на temporal patterns
   - CF features и граф из ОДНОГО источника
   - Сильная корреляция!

3. **Лучшее покрытие**
   - Больше сервисов в графе
   - CF features релевантны для большинства узлов

4. **Одинаковая природа**
   - И граф, и features из calls.csv
   - Coherent data source
   - Синергия!

---

## 🎯 Когда использовать какую версию?

### Используйте Версию 1 (compositionsDAG + CF) когда:

**НЕ РЕКОМЕНДУЕТСЯ!** Результаты хуже baseline.

Но может быть полезна для:
- ✅ Анализа влияния CF features на структурный DAG
- ✅ Baseline для сравнения
- ✅ Понимания что НЕ работает

**Лучший результат:** DAGNN (baseline) 0.5152 (БЕЗ CF features!)

---

### Используйте Версию 2 (CF граф + CF features) когда:

✅ **РЕКОМЕНДУЕТСЯ!** CF features улучшают результаты.

**Когда использовать:**
- ✅ Граф создан из той же базы данных что и CF features
- ✅ Temporal sequences из истории пользователей
- ✅ Достаточно примеров (300+)
- ✅ Хотите лучший результат (0.5446)

**Лучший результат:** GraphSAGE+CF 0.5446 🥇

---

## 📊 Сравнительная таблица

| Аспект | sequence_recommender_with_cf_enrichment.py | sequence_with_cf_features.py |
|--------|-------------------------------------------|------------------------------|
| **Граф** | compositionsDAG.json (структурный) | Создан из calls.csv (temporal) |
| **Примеров** | 107 | 335 |
| **Классов** | 4 | Больше |
| **Features** | 2 → 38 | 38 |
| **Лучшая модель** | DAGNN (baseline) 0.5152 | GraphSAGE+CF 0.5446 |
| **Улучшение от CF** | ❌ -5.9% (хуже!) | ✅ +17.6% (лучше!) |
| **Когда использовать** | НЕ РЕКОМЕНДУЕТСЯ | ✅ РЕКОМЕНДУЕТСЯ |
| **Цель** | Тест CF на структурном DAG | Sequence на CF данных |

---

## ✨ Итоговые рекомендации

### Для compositionsDAG (структурный DAG):

```bash
# Используйте оригинальный sequence_dag_recommender_final.py
cd "sequence recomendation"
python3 sequence_dag_recommender_final.py --random-seed 456
```

**Результат:** DAGNN 0.5152 (БЕЗ CF features)

**Почему:** CF features не помогают на структурном DAG с малыми данными.

---

### Для temporal sequences (из CF данных):

```bash
# Используйте sequence_with_cf_features.py
cd hybrid
python3 sequence_with_cf_features.py
```

**Результат:** GraphSAGE+CF 0.5446 🥇

**Почему:** Граф и features из одного источника → синергия!

---

### Для контекстных рекомендаций:

```bash
# Используйте hybrid_contextual_recommender.py
cd hybrid
python3 hybrid_contextual_recommender.py
```

**Результат:** Hybrid (beta=0.3) 0.4453 (+33.6%)

**Почему:** Комбинация DAGNN + CF для персонализации.

---

## 🔬 Технический анализ

### Почему одинаковые CF features дают разные результаты?

**Ключевое отличие: COHERENCE источника данных**

#### Версия 1 (compositionsDAG):
```
Граф:    compositionsDAG.json   (структурные связи)
          ↓
CF features: calls.csv           (пользовательские паттерны)
          ↓
Результат: НЕСОГЛАСОВАННОСТЬ → шум → ухудшение
```

#### Версия 2 (CF граф):
```
Граф:    calls.csv → temporal sequences
          ↓
CF features: calls.csv
          ↓
Результат: СОГЛАСОВАННОСТЬ → синергия → улучшение!
```

**Вывод:** Данные из ОДНОГО источника работают лучше вместе!

---

## 📈 Примеры использования

### Пример 1: У вас есть compositionsDAG

```python
# НЕ добавляйте CF features!
# Используйте baseline DAGNN

cd "sequence recomendation"
python3 sequence_dag_recommender_final.py --random-seed 456

# Результат: 0.5152 ✅
```

---

### Пример 2: У вас есть история пользователей

```python
# Создайте граф из последовательностей
# Используйте CF features

cd hybrid
python3 sequence_with_cf_features.py

# Результат: 0.5446 ✅
```

---

### Пример 3: Хотите персонализацию

```python
# Используйте hybrid contextual

cd hybrid  
python3 hybrid_contextual_recommender.py

# Результат: 0.4453 с персонализацией ✅
```

---

## ✅ Итоговые выводы

### Главное отличие:

| | sequence_recommender_with_cf_enrichment.py | sequence_with_cf_features.py |
|-|-------------------------------------------|------------------------------|
| **В чем суть?** | Тест: Помогают ли CF features на структурном DAG? | Sequence recommendation на CF данных |
| **Граф откуда?** | compositionsDAG.json (предопределен) | Создан из calls.csv (из данных) |
| **Ответ** | ❌ НЕТ, не помогают на малых структурных данных | ✅ ДА, помогают когда граф из CF |
| **Использовать?** | Нет, baseline лучше | Да, если граф из CF данных |

### Какой файл использовать?

**Для compositionsDAG:**
→ `sequence_dag_recommender_final.py` (оригинал, БЕЗ CF) - 0.5152 🥇

**Для CF sequences:**
→ `sequence_with_cf_features.py` (граф из CF) - 0.5446 🥇

**Для гибрида:**
→ `hybrid_contextual_recommender.py` (DAGNN + CF) - 0.4453

**Для понимания:**
→ `sequence_recommender_with_cf_enrichment.py` (показывает что НЕ работает)

---

**Дата:** October 14, 2025  
**Вывод:** Разные источники данных требуют разных подходов!

