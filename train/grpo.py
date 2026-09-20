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
  1. 需要 trl(pip install trl);本腳本以 TRL 1.x 的 GRPOTrainer API 撰寫與驗證。
  2. 建議先跑完 SFT(+DPO)再上 GRPO,否則小模型的 on-policy 探索會很不穩。
  3. VRAM:0.9B + GRPO(num_generations=8)在 24GB(3090/4090)可用;8GB 請調小
     --num-generations / --max-completion。

範例:
  python train/grpo.py --model-id IFM/K2-Horizon-0.9B --trust-remote-code \
      --train-jsonl data/out/train.jsonl --adapter-dir output/sft --output-dir output/grpo \
      --num-generations 8 --batch-size 2 --max-completion 512

RL 準備度審計(不訓練;RL 前後各跑一次比較,確認 RL 真的把 reward 推上去):
  python train/grpo.py --model-id IFM/K2-Horizon-0.9B --trust-remote-code \
      --train-jsonl data/out/train.jsonl --adapter-dir output/sft \
      --reward-audit --audit-n 50 --audit-out output/reward_audit.json

CPU smoke test(tiny 模型;注意 TRL 要求 batch-size 能被 num-generations 整除):
  python train/grpo.py --model-id tools/tiny_model --dtype float32 \
      --train-jsonl data/out_smoke/train.jsonl --output-dir output/smoke_grpo \
      --lora --num-generations 2 --batch-size 2 --max-completion 96 --max-steps 2
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
    from datasets import Dataset as HFDataset
    from trl import GRPOConfig, GRPOTrainer
except ImportError:
    sys.exit("缺少 trl / datasets: pip install trl datasets 後重試(GRPO 為選用階段)")

from common import parse_response_any
from train.sft import DTYPES, load_causal_lm

# 預設 reward 權重(RLCD 式:對準「校準的機率」,不是對準文字偏好)
DEFAULT_WEIGHTS = {"format": 1.0, "brier": 1.0, "direction": 0.5, "cover": 0.3}


def reward_components_raw(text, y_home, y_cover=None):
    """單個 completion 的「未乘權重」reward 分項(dict)。

    format:    1.0  解析出完整機率行(prose 或 typed JSON 皆可)
    brier:     1 - (p_home - y)^2
    direction: 1.0  預測主/客方向正確
    cover:     1 - (cover_home - c)^2   (沒有 cover 資訊時 0)
    權重由 make_reward(weights) / run_audit 套用。
    """
    if isinstance(text, (bytes, bytearray)):
        text = text.decode("utf-8", errors="ignore")
    p = parse_response_any(str(text))
    comp = {"format": 0.0, "brier": 0.0, "direction": 0.0, "cover": 0.0}
    if p["format_ok"] and p["p_home"] is not None:
        comp["format"] = 1.0
        comp["brier"] = 1.0 - (p["p_home"] - float(y_home)) ** 2
        pred = p["pred"] or ("主隊" if p["p_home"] >= 0.5 else "客隊")
        comp["direction"] = 1.0 if pred == ("主隊" if float(y_home) == 1 else "客隊") else 0.0
        if p["cover_home"] is not None and y_cover is not None:
            comp["cover"] = 1.0 - (p["cover_home"] - float(y_cover)) ** 2
    return comp


def reward_components(text, y_home, y_cover=None):
    """reward_components_raw × 預設權重(舊 API,行為與過去版本一致)。"""
    c = reward_components_raw(text, y_home, y_cover)
    return {k: (c[k] if k in ("format", "brier") else DEFAULT_WEIGHTS[k] * c[k])
            for k in c}


def reward_components_soccer_raw(text, y):
    """足球 1X2 單個 completion 的未乘權重分項(0=主勝 1=和 2=客勝)。

    format:  1.0   brier: 1 - Σ(p_i - y_i)²   direction: 1.0(argmax 對)
    """
    if isinstance(text, (bytes, bytearray)):
        text = text.decode("utf-8", errors="ignore")
    p = parse_response_any(str(text), "soccer")
    comp = {"format": 0.0, "brier": 0.0, "direction": 0.0, "cover": 0.0}
    if p["format_ok"] and p["p_home"] is not None:
        yv = [0.0, 0.0, 0.0]
        yv[int(y)] = 1.0
        comp["format"] = 1.0
        comp["brier"] = 1.0 - (p["p_home"] - yv[0]) ** 2 - (p["p_draw"] - yv[1]) ** 2 \
            - (p["p_away"] - yv[2]) ** 2
        from common import OUTCOME_LABELS
        comp["direction"] = 1.0 if p["pred"] == OUTCOME_LABELS[int(y)] else 0.0
    return comp


def reward_components_soccer(text, y):
    """reward_components_soccer_raw × 預設權重(舊 API)。"""
    c = reward_components_soccer_raw(text, y)
    return {k: (c[k] if k in ("format", "brier") else DEFAULT_WEIGHTS[k] * c[k])
            for k in c}


def make_reward(weights: dict = None):
    """TRL 1.x reward function(支援 basketball 二結果與 soccer 1X2;prose 與 typed 皆可)。

    TRL 以關鍵字呼叫:reward_func(prompts=..., completions=..., completion_ids=...,
    **dataset 額外欄位)→ 回傳 list[float](每個 completion 一個分數)。
    weights 可調(RLCD 風格:例如把 brier 權重調高,更用力壓校準)。
    """
    w = dict(DEFAULT_WEIGHTS, **(weights or {}))

    def _score(comp):
        return (w["format"] * comp["format"] + w["brier"] * comp["brier"]
                + w["direction"] * comp["direction"] + w["cover"] * comp["cover"])

    def reward(prompts=None, completions=None, completion_ids=None,
               outcome_home=None, cover_home=None, outcome=None, sport=None, **_kw):
        if completions is None:
            return None  # TRL 會以 NaN 處理並警告
        is_soccer = any(s == "soccer" for s in (sport if isinstance(sport, list) else [])) \
            or (sport == "soccer") or (outcome is not None and outcome_home is None)
        if is_soccer:
            if outcome is None:
                return None
            ys = outcome if isinstance(outcome, list) else [outcome] * len(completions)
            out = []
            for text, y in zip(completions, ys):
                out.append(_score(reward_components_soccer_raw(text, float(y))))
            return out
        if outcome_home is None:
            return None
        out = []
        ys = outcome_home if isinstance(outcome_home, list) else [outcome_home]
        cs = cover_home if isinstance(cover_home, list) else ([cover_home] * len(ys))
        for text, y, c in zip(completions, ys, cs):
            out.append(_score(reward_components_raw(text, float(y),
                                                     float(c) if c is not None else None)))
        return out

    return reward


def run_audit(model, tok, rows, n=20, max_new=128, weights: dict = None):
    """RL 準備度審計:對 rows 取前 n 個 prompt,greedy 產生 completion,
    以 reward_components 逐項打分 → 各分項平均值 dict(reward_mean 用相同權重)。

    用途:上 GRPO 前看「格式率/Brier/方向/覆蓋」各差多少;RL 前後各跑一次
    比較,確認 RL 真的把 reward 推上去(而不是只改格式)。
    """
    w = dict(DEFAULT_WEIGHTS, **(weights or {}))
    rows = rows[:n]
    # prompt 可能是 chat message list(train.jsonl 原格式)或純字串
    convos = [r["prompt"] if isinstance(r["prompt"], list)
              else [{"role": "user", "content": r["prompt"]}] for r in rows]
    inputs = tok.apply_chat_template(
        convos, tokenize=True, add_generation_prompt=True, return_tensors="pt",
        padding=True, max_length=512, truncation=True)
    model.eval()
    was_training = model.training
    prev_side = tok.padding_side
    tok.padding_side = "left"  # 生成必須從 prompt 末尾開始,不能從 PAD 開始
    try:
        with torch.no_grad():
            ids = model.generate(**inputs, max_new_tokens=max_new, do_sample=False,
                                 pad_token_id=tok.pad_token_id or tok.eos_token_id)
    finally:
        tok.padding_side = prev_side
    model.train(was_training)
    seq_len = inputs["input_ids"].shape[1]
    texts = [tok.decode(ids[j][seq_len:], skip_special_tokens=True) for j in range(len(rows))]
    tot = {"format": 0.0, "brier": 0.0, "direction": 0.0, "cover": 0.0}
    sport = "soccer" if rows and rows[0].get("sport") == "soccer" else "basketball"
    for text, r in zip(texts, rows):
        if sport == "soccer":
            comp = reward_components_soccer_raw(text, float(r["outcome"]))
        else:
            comp = reward_components_raw(text, float(r["outcome_home"]),
                                         r.get("cover_home"))
        for k in tot:
            tot[k] += comp[k]
    k = len(rows)
    return {
        "n": k,
        "sport": sport,
        "weights": w,
        "reward_mean": (w["format"] * tot["format"] + w["brier"] * tot["brier"]
                        + w["direction"] * tot["direction"] + w["cover"] * tot["cover"]) / k,
        "format_rate": tot["format"] / k,
        "brier_mean": tot["brier"] / k,
        "direction_mean": tot["direction"] / k,
        "cover_mean": tot["cover"] / k,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model-id", default="IFM/K2-Horizon-0.9B")
    ap.add_argument("--trust-remote-code", action="store_true")
    ap.add_argument("--dtype", default="bfloat16", choices=sorted(DTYPES))
    ap.add_argument("--train-jsonl", default="data/out/train.jsonl")
    ap.add_argument("--adapter-dir", default=None, help="從 SFT/DPO adapter 繼續(建議)")
    ap.add_argument("--output-dir", default="output/grpo")
    ap.add_argument("--batch-size", type=int, default=2)
    ap.add_argument("--num-generations", type=int, default=8)
    ap.add_argument("--max-completion", type=int, default=512)
    ap.add_argument("--lr", type=float, default=1e-6)
    ap.add_argument("--beta", type=float, default=0.0, help="KL 係數;0=不限制(可試 0.01)")
    ap.add_argument("--temperature", type=float, default=1.0)
    # reward 權重(RLCD 風格:調高 brier = 更用力壓校準)
    ap.add_argument("--w-format", type=float, default=DEFAULT_WEIGHTS["format"])
    ap.add_argument("--w-brier", type=float, default=DEFAULT_WEIGHTS["brier"])
    ap.add_argument("--w-direction", type=float, default=DEFAULT_WEIGHTS["direction"])
    ap.add_argument("--w-cover", type=float, default=DEFAULT_WEIGHTS["cover"])
    ap.add_argument("--logging-steps", type=int, default=5)
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--max-steps", type=int, default=-1)
    ap.add_argument("--lora", action="store_true", help="無 adapter 時:用新 LoRA 跑 GRPO(省 VRAM)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--reward-audit", action="store_true",
                    help="不訓練:對 --train-jsonl 前 N 個 prompt 跑 RL 準備度審計(reward 分項)")
    ap.add_argument("--audit-n", type=int, default=20)
    ap.add_argument("--audit-max-new", type=int, default=128)
    ap.add_argument("--audit-out", default=None, help="審計結果 JSON 輸出路徑")
    args = ap.parse_args()
    weights = {"format": args.w_format, "brier": args.w_brier,
               "direction": args.w_direction, "cover": args.w_cover}

    # ---- 資料:GRPO 只需要 prompt + 答案欄位(reward 用)----
    rows = []
    with open(args.train_jsonl, encoding="utf-8") as f:
        for line in f:
            d = json.loads(line)
            row = {"prompt": d["prompt"]}
            if d.get("sport") == "soccer" or "outcome" in d:
                row["sport"] = "soccer"
                row["outcome"] = int(d["outcome"])
            else:
                row["outcome_home"] = int(d["outcome_home"])
                row["cover_home"] = int(d["cover_home"])
            rows.append(row)
    if args.num_generations > 1 and len(rows) % args.num_generations:
        pad = args.num_generations - len(rows) % args.num_generations
        rows += rows[:pad]
        print(f"GRPO: rows pad 至能被 num_generations 整除 → {len(rows)} "
              f"(TRL 要求 batch-size 與 num_generations 整除)")
    print(f"GRPO rows: {len(rows)}")

    tok = AutoTokenizer.from_pretrained(args.model_id, trust_remote_code=args.trust_remote_code)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    model = load_causal_lm(args.model_id, DTYPES[args.dtype], args.trust_remote_code)
    if args.adapter_dir:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, args.adapter_dir, is_trainable=True)
    elif args.lora:
        from peft import LoraConfig, TaskType, get_peft_model
        model = get_peft_model(model, LoraConfig(
            r=16, lora_alpha=32, bias="none", task_type=TaskType.CAUSAL_LM))
    if torch.cuda.is_available():
        model = model.cuda()

    if args.reward_audit:
        summary = run_audit(model, tok, rows, n=args.audit_n, max_new=args.audit_max_new,
                            weights=weights)
        print(json.dumps(summary, indent=2, ensure_ascii=False))
        if args.audit_out:
            d = os.path.dirname(os.path.abspath(args.audit_out))
            os.makedirs(d, exist_ok=True)
            with open(args.audit_out, "w", encoding="utf-8") as f:
                json.dump(summary, f, indent=2, ensure_ascii=False)
            print(f"reward audit -> {args.audit_out}")
        return

    cfg = GRPOConfig(
        output_dir=args.output_dir,
        learning_rate=args.lr,
        per_device_train_batch_size=args.batch_size,
        num_generations=args.num_generations,
        max_completion_length=args.max_completion,
        temperature=args.temperature,
        beta=args.beta,
        num_train_epochs=args.epochs,
        max_steps=args.max_steps,
        logging_steps=args.logging_steps,
        save_steps=args.max_steps if args.max_steps > 0 else 100,
        report_to=[],
        seed=args.seed,
        bf16=torch.cuda.is_available(),
    )
    trainer = GRPOTrainer(
        model=model,
        args=cfg,
        reward_funcs=[make_reward(weights)],
        train_dataset=HFDataset.from_list(rows),
        processing_class=tok,
    )
    trainer.train()
    os.makedirs(args.output_dir, exist_ok=True)
    trainer.save_model(args.output_dir)
    print(f"GRPO adapter saved -> {args.output_dir}")


if __name__ == "__main__":
    main()
