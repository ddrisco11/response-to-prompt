#!/bin/bash
set -eu
cd "$(dirname "$0")"
mkdir -p logs
export PYTHONUNBUFFERED=1
py=${TRAIN_PYTHON:-python}
"$py" -m mlx_lm lora --config train.yaml --model .assets/base --data data --adapter-path adapters > logs/train.log 2>&1
"$py" evaluate.py > logs/evaluate.log 2>&1
"$py" publish.py > logs/publish.log 2>&1
