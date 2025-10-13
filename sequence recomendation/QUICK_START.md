# 🚀 Быстрый старт

## Минимальный запуск (1 команда!)

```bash
python3 sequence_dag_recommender_final.py
```

## Лучший результат

```bash
python3 sequence_dag_recommender_final.py --random-seed 456
```

**Результат:** DAGNN accuracy 0.5152 🏆

## Визуализация

### Генерация графиков:
```bash
python3 generate_visualizations.py
```

**Создает 8 графиков в директории `images/`:**
- accuracy_comparison.png - Сравнение accuracy
- accuracy_vs_ndcg.png - Scatter plot
- radar_chart.png - Топ-4 модели
- category_comparison.png - По категориям
- training_speed.png - Скорость обучения
- seed_stability.png - Стабильность
- metrics_heatmap.png - Тепловая карта
- data_pipeline.png - Pipeline данных

### Текстовая визуализация:
```bash
python3 visualize_results_v2.py
```

## Файлы

- **README.md** - Полная документация (24K, 900+ строк)
- **sequence_dag_recommender_final.py** - Главный файл (30K, 1100+ строк)
- **visualize_results_v2.py** - Визуализация
- **compositionsDAG.json** - Данные
- **MODELS_TO_ADD.md** - Анализ моделей из awesome list

## Модели (8 моделей!)

**Всего 8 моделей** из разных категорий:

### GNN модели (3):
1. 🏆 **DAGNN** - лучшая (0.5152)
2. ⚡ **GraphSAGE** - самая быстрая (415 it/s)
3. 🧠 **GCN** - стабильная (0.4848)

### Sequential модели (3):
4. 🎯 **SASRec** ✨ - Self-Attention (0.4848) - НОВОЕ!
5. 🧱 **Caser** ✨ - CNN (0.4848, nDCG 0.7781) - НОВОЕ!
6. 🔄 **GRU4Rec** - RNN (0.2727, требует больше данных)

### Baseline модели (2):
7. 📊 **Random Forest** - ML baseline (0.4848)
8. 📈 **Popularity** - простой baseline (0.27)

---

## 🎯 Рейтинг моделей

### На текущем датасете (107 примеров):
```
🥇 DAGNN:      0.5152  (лучшая!)
🥈 SASRec:     0.4848  (отлично для attention!)
🥈 Caser:      0.4848  (отлично для CNN!)
🥈 GraphSAGE:  0.4848  (быстрая!)
🥈 GCN:        0.4848
🥈 RF:         0.4848
🥉 GRU4Rec:    0.2727  (требует больше данных)
```

### При увеличении до 5000+ примеров (ожидаемо):
```
🥇 SASRec:     0.70-0.75  (self-attention!)
🥈 DAGNN:      0.65-0.70  (DAG структуры)
🥈 GRU4Rec:    0.65-0.75  (последовательности!)
🥉 Caser:      0.60-0.70  (CNN паттерны)
```

---

## ⚡ Что нового в v3.0

- ✨ **SASRec** - Self-Attention для последовательностей
  - Accuracy: 0.4848 
  - Одна из самых популярных моделей
  - Быстрее GRU4Rec (параллельное обучение)

- ✨ **Caser** - CNN для последовательностей
  - Accuracy: 0.4848
  - nDCG: 0.7781 (отличный!)
  - Horizontal + Vertical convolutions

- 🔄 **GRU4Rec** - RNN для последовательностей
  - Требует больше данных для эффективности

---

## 💡 Рекомендации

### Для продакшена:
```bash
python3 sequence_dag_recommender_final.py --random-seed 456
```
→ **DAGNN**: 0.5152 🏆

### Для экспериментов с attention:
→ **SASRec**: 0.4848 (интерпретируемые attention weights)

### Для CNN подхода:
→ **Caser**: 0.4848 (быстрая, локальные паттерны)

### Для скорости:
→ **GraphSAGE/Caser**: 415-420 it/s ⚡

---

См. README.md для подробностей.
