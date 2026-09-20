"""階段 3(選用、進階):GRPO — 以「Brier 分數 + 格式分數」為 reward 的 RL。

為什麼放這裡:K2-Horizon 系列的訓練流程本身就大量使用 GRPO(RL 專家模型 + merge)。
體育預測是一個有「可驗證答案」的任務(勝負/盤口最終都有結果),因此可以用
on-policy RL 直接對準「預測是否準」,而不是只模仿 teacher。

reward 設計(單樣本):
  - 格式正確(能解析出機率)         → +1.0
  - 勝率 Brier:1 - (p - y)^2        → 0..1
  - 預測方向正確(主/客)            → +0.5
  - 讓分覆蓋 Brier(權重 0.3)       → 0..0.3

⚠️ 注意:
  1. 需要 trl: pip install trl
  2. TRL 的 GRPOTrainer API 在版本之間變動很快;此腳本以 0.11+ 風格撰寫,
     若與你所裝版本不兼容,請對照該版文件的 sample(範例)微調引數。
  3. 建議先跑完 SFT(+DPO)再上 GRPO,否則小模型的 on-policy 探索會很不穩。
  4. VRAM:0.9B + GRPO(num_generations=8)在 24GB(3090/4090)可用;8GB 請調小
     num_generations / max_completion_length 並開啟 --lora。

範例:
  python train/grpo.py --model-id IFM/K2-Horizon-0.9B --trust-remote-code \
      --train-jsonl data/out/train.jsonl --adapter-dir output/sft --output-dir output/grpo \
      --num-generations 8 --batch-size 2 --max-completion 512
"""
from __future__ import annotations

import argparse
import json
import os
import sys

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

try:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
except ImportError as e:  # pragma: no cover
    sys.exit(f"缺少依賴({e});先 pip install -r requirements.txt")

try:
    from trl import GRPOConfig, GRPOTrainer
except ImportError:
    sys.exit("缺少 trl: pip install trl 後重試(GRPO 為選用階段)")

from common import parse_response
from train.sft import DTYPES, load_causal_lm


def make_reward():
    """TRL reward function:接收 completions(字串)與 dataset 的額外欄位。"""

    def reward(completions, outcome_home, cover_home, **_kw):
        out = []
        for text, y, c in zip(completions, outcome_home, cover_home):
            s = 0.0
            p = parse_response(text if isinstance(text, str) else str(text))
            if p["format_ok"] and p["p_home"] is not None:
                s += 1.0
                s += 1.0 - (p["p_home"] - float(y)) ** 2
                pred = p["pred"] or ("主隊" if p["p_home"] >= 0.5 else "客隊")
                s += 0.5 if pred == ("主隊" if y == 1 else "客隊") else 0.0
                if p["cover_home"] is not None:
                    s += 0.3 * (1.0 - (p["cover_home"] - float(c)) ** 2)
            out.append(s)
        return out

    return reward


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model-id", default="IFM/K2-Horizon-0.9B")
    ap.add_argument("--trust-remote-code", action="store_true")
    ap.add_argument("--dtype", default="bfloat16", choices=sorted(DTYPES))
    ap.add_argument("--train-jsonl", default="data/out/train.jsonl")
    ap.add_argument("--adapter-dir", default=None, help="從 SFT adapter 繼續(建議)")
    ap.add_argument("--output-dir", default="output/grpo")
    ap.add_argument("--batch-size", type=int, default=2)
    ap.add_argument("--num-generations", type=int, default=8)
    ap.add_argument("--max-completion", type=int, default=512)
    ap.add_argument("--lr", type=float, default=1e-6)
    ap.add_argument("--beta", type=float, default=0.0, help="KL 係數;0=不限制(可試 0.01)")
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--max-steps", type=int, default=-1)
    ap.add_argument("--lora", action="store_true", help="用 LoRA 跑 GRPO(省 VRAM,預設建議開啟)")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    # ---- 資料:GRPO 只需要 prompt + 答案欄位(reward 用)----
    rows = []
    with open(args.train_jsonl, encoding="utf-8") as f:
        for line in f:
            d = json.loads(line)
            rows.append({
                "prompt": d["prompt"],
                "outcome_home": int(d["outcome_home"]),
                "cover_home": int(d["cover_home"]),
            })
    print(f"GRPO rows: {len(rows)}")

    tok = AutoTokenizer.from_pretrained(args.model_id, trust_remote_code=args.trust_remote_code)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    model = load_causal_lm(args.model_id, DTYPES[args.dtype], args.trust_remote_code)
    if args.adapter_dir:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, args.adapter_dir)
    if args.lora:
        from peft import LoraConfig, TaskType, get_peft_model
        model = get_peft_model(model, LoraConfig(
            r=16, lora_alpha=32, bias="none", task_type=TaskType.CAUSAL_LM))
    if torch.cuda.is_available():
        model = model.cuda()

    cfg = GRPOConfig(
        output_dir=args.output_dir,
        learning_rate=args.lr,
        per_device_train_batch_size=args.batch_size,
        num_generations=args.num_generations,
        max_completion_length=args.max_completion,
        beta=args.beta,
        num_train_epochs=args.epochs,
        max_steps=args.max_steps,
        logging_steps=10,
        report_to=[],
        seed=args.seed,
        bf16=torch.cuda.is_available(),
    )
    trainer = GRPOTrainer(
        model=model,
        args=cfg,
        reward_funcs=[make_reward()],
        train_dataset=rows,
        processing_class=tok,
    )
    trainer.train()
    os.makedirs(args.output_dir, exist_ok=True)
    trainer.save_model(args.output_dir)
    print(f"GRPO adapter saved -> {args.output_dir}")


if __name__ == "__main__":
    main()
