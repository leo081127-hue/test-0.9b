#!/usr/bin/env bash
# CPU smoke test:不需要 GPU、不需要網路。
# 用本地隨機初始化的小模型(tools/tiny_model)把整條管線跑一遍,驗證程式邏輯。
set -euo pipefail
cd "$(dirname "$0")/.."

PY="${PY:-python}"

echo "==> [1/6] demo 資料"
$PY data/generate_demo_data.py --seasons 3 --games-per-season 300 --out data/demo/matches_smoke.csv

echo "==> [2/6] 建立 SFT / DPO 資料"
$PY data/build_dataset.py --matches data/demo/matches_smoke.csv --out data/out_smoke --max-pairs 200

echo "==> [3/6] 建立 tiny 模型(本地,不下載)"
$PY tools/smoke_tiny_model.py --train-jsonl data/out_smoke/train.jsonl --out tools/tiny_model

echo "==> [4/6] SFT smoke"
$PY train/sft.py --model-id tools/tiny_model --dtype float32 \
  --train-jsonl data/out_smoke/train.jsonl --adapter-dir output/smoke_sft \
  --max-steps 15 --batch-size 2 --max-len 1024 --logging-steps 5

echo "==> [5/6] DPO smoke"
$PY train/dpo.py --model-id tools/tiny_model --dtype float32 \
  --adapter-dir output/smoke_sft --pairs data/out_smoke/dpo_pairs.jsonl \
  --output-dir output/smoke_dpo --max-steps 8 --batch-size 2

echo "==> [6/6] 評估 + 推論 smoke"
$PY eval/evaluate.py --model-id tools/tiny_model --dtype float32 \
  --adapter-dir output/smoke_dpo --test-jsonl data/out_smoke/test.jsonl \
  --matches-csv data/demo/matches_smoke.csv --report output/smoke_report.json \
  --limit 4 --no-model
$PY eval/evaluate.py --model-id tools/tiny_model --dtype float32 \
  --adapter-dir output/smoke_dpo --test-jsonl data/out_smoke/test.jsonl \
  --matches-csv data/demo/matches_smoke.csv --report output/smoke_report_llm.json --limit 3
$PY infer/predict.py --model-id tools/tiny_model --dtype float32 \
  --adapter-dir output/smoke_dpo --csv data/demo/matches_smoke.csv --row 899

echo "SMOKE OK"
