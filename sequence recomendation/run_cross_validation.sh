#!/bin/bash
# Скрипт для запуска кросс-валидации всех моделей

echo "=================================================="
echo "Запуск кросс-валидации для всех моделей"
echo "=================================================="
echo ""

# Параметры по умолчанию
DATA_FILE="${1:-compositionsDAG.json}"
CV_FOLDS="${2:-5}"
EPOCHS="${3:-200}"

echo "Параметры:"
echo "  Файл данных: $DATA_FILE"
echo "  Количество фолдов: $CV_FOLDS"
echo "  Количество эпох: $EPOCHS"
echo ""
echo "Начало обучения..."
echo ""

# Запуск с кросс-валидацией
python sequence_dag_recommender_final.py \
    --data "$DATA_FILE" \
    --use-cv \
    --cv-folds "$CV_FOLDS" \
    --epochs "$EPOCHS" \
    --hidden-channels 64 \
    --learning-rate 0.001 \
    --dropout 0.4 \
    --random-seed 42

echo ""
echo "=================================================="
echo "Кросс-валидация завершена!"
echo "=================================================="

