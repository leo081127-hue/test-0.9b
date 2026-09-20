"""階段 2(選用):DPO — 用 data/out/dpo_pairs.jsonl 做偏好微調。

policy = base + SFT adapter(LoRA 可訓練);reference = 同一 base(凍結、關閉 adapter 層)。
DPO loss = -logsigmoid(beta * ((logπ(c|x) - logπ_ref(c|x)) - (logπ(r|x) - logπ_ref(r|x))))

GPU 機器範例:
  python train/dpo.py \
      --model-id IFM/K2-Horizon-0.9B --trust-remote-code \
      --adapter-dir output/sft --pairs data/out/dpo_pairs.jsonl \
      --output-dir output/dpo --epochs 1 --lr 5e-5 --beta 0.1
"""
from __future__ import annotations

import argparse
import json
import os
import sys

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

from train.sft import DTYPES, load_causal_lm


def completion_logprob(model, tok, prompt_text: str, completion_text: str, max_len: int) -> torch.Tensor:
    """completion 段 token log-prob 的總和(對可訓練模型保留 grad)。"""
    p_ids = tok(prompt_text, add_special_tokens=False)["input_ids"]
    c_ids = tok(completion_text, add_special_tokens=False)["input_ids"]
    ids = (p_ids + c_ids)[:max_len]
    input_ids = torch.tensor([ids])
    att = torch.ones_like(input_ids)
    out = model(input_ids=input_ids, attention_mask=att)
    logp = torch.log_softmax(out.logits[0], dim=-1)          # [L, V]
    tgt = input_ids[0, 1:]                                   # [L-1]
    lp = logp[:-1].gather(1, tgt.unsqueeze(-1)).squeeze(-1)  # lp[i] = log P(tok[i+1] | <=i)
    start = max(len(p_ids) - 1, 0)
    if start >= len(lp):
        return lp.new_zeros(())
    mask = torch.zeros_like(lp, dtype=torch.bool)
    mask[start:] = True
    return lp[mask].sum()


def ref_completion_logprob(model, tok, prompt_text, completion_text, max_len):
    """reference:同一模型但關閉 adapter(省一份 VRAM)。"""
    model.disable_adapter_layers()
    try:
        with torch.no_grad():
            return completion_logprob(model, tok, prompt_text, completion_text, max_len)
    finally:
        model.enable_adapter_layers()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model-id", default="IFM/K2-Horizon-0.9B")
    ap.add_argument("--trust-remote-code", action="store_true")
    ap.add_argument("--dtype", default="bfloat16", choices=sorted(DTYPES))
    ap.add_argument("--adapter-dir", default="output/sft", help="SFT 後的 adapter 目錄")
    ap.add_argument("--pairs", default="data/out/dpo_pairs.jsonl")
    ap.add_argument("--output-dir", default="output/dpo")
    ap.add_argument("--epochs", type=float, default=1)
    ap.add_argument("--max-steps", type=int, default=-1)
    ap.add_argument("--batch-size", type=int, default=2)
    ap.add_argument("--lr", type=float, default=5e-5)
    ap.add_argument("--beta", type=float, default=0.1)
    ap.add_argument("--max-len", type=int, default=1024)
    ap.add_argument("--logging-steps", type=int, default=10)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    tok = AutoTokenizer.from_pretrained(args.model_id, trust_remote_code=args.trust_remote_code)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    base = load_causal_lm(args.model_id, DTYPES[args.dtype], args.trust_remote_code)
    policy = PeftModel.from_pretrained(base, args.adapter_dir, is_trainable=True)
    if torch.cuda.is_available():
        policy = policy.cuda()
    policy.train()

    pairs = []
    with open(args.pairs, encoding="utf-8") as f:
        for line in f:
            pairs.append(json.loads(line))
    print(f"DPO pairs: {len(pairs)}")

    opt = torch.optim.AdamW(
        (p for p in policy.parameters() if p.requires_grad), lr=args.lr, weight_decay=0.0
    )

    steps = 0
    total = 0 if args.max_steps <= 0 else args.max_steps
    for epoch in range(int(args.epochs)):
        gen = torch.Generator().manual_seed(args.seed + epoch)
        idx = torch.randperm(len(pairs), generator=gen).tolist()
        for i in range(0, len(idx), args.batch_size):
            batch = [pairs[j] for j in idx[i:i + args.batch_size]]
            losses, chosen_wins = [], 0
            for pair in batch:
                pc = completion_logprob(policy, tok, pair["prompt"], pair["chosen"], args.max_len)
                pr = completion_logprob(policy, tok, pair["prompt"], pair["rejected"], args.max_len)
                rc = ref_completion_logprob(policy, tok, pair["prompt"], pair["chosen"], args.max_len)
                rr = ref_completion_logprob(policy, tok, pair["prompt"], pair["rejected"], args.max_len)
                chosen_logratio = pc - rc
                rejected_logratio = pr - rr
                losses.append(-F.logsigmoid(args.beta * (chosen_logratio - rejected_logratio)))
                chosen_wins += float((pc - rc) > (pr - rr))
            loss = torch.stack(losses).mean()
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
            opt.step()

            steps += 1
            if steps % args.logging_steps == 0:
                acc = chosen_wins / len(batch)
                print(f"step {steps}  loss {loss.item():.4f}  chosen>ref_acc {acc:.2f}")
            if total and steps >= total:
                break
        if total and steps >= total:
            break

    os.makedirs(args.output_dir, exist_ok=True)
    policy.save_pretrained(args.output_dir)
    with open(os.path.join(args.output_dir, "metadata.json"), "w", encoding="utf-8") as f:
        json.dump({"model_id": args.model_id, "stage": "dpo", "beta": args.beta,
                   "base_adapter": args.adapter_dir}, f, ensure_ascii=False, indent=2)
    print(f"DPO adapter saved -> {args.output_dir}")


if __name__ == "__main__":
    main()
