#!/usr/bin/env bash
# 一整條流水線:demo 資料 → SFT → DPO → 評估。
# 在「有 GPU 的機器」上執行。環境變數:
#   MODEL_ID    基座模型(預設 IFM/K2-Horizon-0.9B)
#   EXTRA_ARGS  傳給各训练腳本的額外參數(K2-Horizon 需要 "--trust-remote-code")
#   MATCHES     真實資料 csv(預設用合成 demo)
#   PY          python 路徑(預設 python)
set -euo pipefail
cd "$(dirname "$0")/.."

MODEL_ID="${MODEL_ID:-IFM/K2-Horizon-0.9B}"
EXTRA_ARGS="${EXTRA_ARGS:---trust-remote-code}"
MATCHES="${MATCHES:-data/demo/matches.csv}"
PY="${PY:-python}"

echo "==> [1/5] demo 資料(有真實資料時設定 MATCHES=你的csv,或直接先產生再覆蓋)"
if [ ! -f "$MATCHES" ]; then
  $PY data/generate_demo_data.py --out "$MATCHES"
fi

echo "==> [2/5] 建立 SFT / DPO 資料"
$PY data/build_dataset.py --matches "$MATCHES" --out data/out

echo "==> [3/5] SFT (LoRA)"
$PY train/sft.py $EXTRA_ARGS \
  --model-id "$MODEL_ID" \
  --train-jsonl data/out/train.jsonl --val-jsonl data/out/val.jsonl \
  --adapter-dir output/sft --epochs 3 --batch-size 4 --grad-accum 4

echo "==> [4/5] DPO"
$PY train/dpo.py $EXTRA_ARGS \
  --model-id "$MODEL_ID" \
  --adapter-dir output/sft --pairs data/out/dpo_pairs.jsonl \
  --output-dir output/dpo --epochs 1 --lr 5e-5

echo "==> [5/5] 評估"
$PY eval/evaluate.py $EXTRA_ARGS \
  --model-id "$MODEL_ID" --adapter-dir output/dpo \
  --test-jsonl data/out/test.jsonl --matches-csv "$MATCHES" \
  --report output/report.json --plot output/calibration.png

echo "done. report: output/report.json"
