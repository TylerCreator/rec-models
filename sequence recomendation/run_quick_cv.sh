#!/bin/bash
# Быстрый запуск кросс-валидации для тестирования (3 фолда, 100 эпох)

echo "=================================================="
echo "Быстрая кросс-валидация (3 фолда, 100 эпох)"
echo "=================================================="
echo ""

DATA_FILE="${1:-compositionsDAG.json}"

echo "Файл данных: $DATA_FILE"
echo ""
echo "Начало обучения..."
echo ""

python sequence_dag_recommender_final.py \
    --data "$DATA_FILE" \
    --use-cv \
    --cv-folds 3 \
    --epochs 100 \
    --hidden-channels 64 \
    --learning-rate 0.001 \
    --dropout 0.4 \
    --random-seed 42

echo ""
echo "=================================================="
echo "Быстрая кросс-валидация завершена!"
echo "=================================================="

