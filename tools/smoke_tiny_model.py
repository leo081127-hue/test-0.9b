"""建立一個超小的 Llama(隨機初始化)+ 本地訓練的 BPE tokenizer。

用途:
  1. 本沙盒 / CI 的 CPU smoke test——不依賴網路、不需 GPU,驗證整條
     資料→SFT→DPO→評估→推論 的程式管線。
  2. 注意:它只是「格式正確的小模型」,沒有真實語意,訓練後輸出仍可能是亂碼,
     不能用來評估預測品質;真實訓練請用 IFM/K2-Horizon-0.9B 或 Qwen3-1.7B 等。

用法:
  python tools/smoke_tiny_model.py --train-jsonl data/out/train.jsonl --out tools/tiny_model
"""
from __future__ import annotations

import argparse
import json
import os

from tokenizers import Tokenizer, models as tmodels, pre_tokenizers, trainers as ttrainers, decoders
from transformers import LlamaConfig, LlamaForCausalLM, PreTrainedTokenizerFast

CHAT_TEMPLATE = (
    "{% for m in messages %}"
    "{% if m['role'] == 'system' %}{{ '<|system|> ' + m['content'] + '\n' }}"
    "{% elif m['role'] == 'user' %}{{ '<|user|> ' + m['content'] + '\n' }}"
    "{% else %}{{ '<|assistant|> ' + m['content'] + '\n' }}"
    "{% endif %}{% endfor %}"
    "{% if add_generation_prompt %}<|assistant|>{% endif %}"
)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--train-jsonl", default="data/out/train.jsonl")
    ap.add_argument("--val-jsonl", default="data/out/val.jsonl")
    ap.add_argument("--out", default="tools/tiny_model")
    ap.add_argument("--hidden", type=int, default=512)
    ap.add_argument("--layers", type=int, default=8)
    ap.add_argument("--vocab", type=int, default=4000)
    args = ap.parse_args()

    texts: list[str] = []
    for p in (args.train_jsonl, args.val_jsonl):
        if not os.path.exists(p):
            continue
        with open(p, encoding="utf-8") as f:
            for line in f:
                d = json.loads(line)
                texts.append(d["prompt"][0]["content"])
                texts.append(d["response"])
    print(f"BPE corpus: {len(texts)} documents")

    tok = Tokenizer(tmodels.BPE(unk_token="<unk>"))
    tok.pre_tokenizer = pre_tokenizers.ByteLevel()
    # 關鍵:ByteLevel BPE 的 decode 必須還原 byte-level 字元 → UTF-8,
    # 否則 decode 回亂碼,任何「模型輸出解析」(評估/推論/GRPO reward)都會失敗。
    tok.decoder = decoders.ByteLevel()
    trainer = ttrainers.BpeTrainer(
        vocab_size=args.vocab,
        special_tokens=["<unk>", "<pad>", "<bos>", "<eos>",
                        "<|user|>", "<|assistant|>", "<|system|>"],
    )
    tok.train_from_iterator(texts, trainer)
    ftok = PreTrainedTokenizerFast(
        tokenizer_object=tok,
        pad_token="<pad>", bos_token="<bos>", eos_token="<eos>",
        model_max_length=2048, chat_template=CHAT_TEMPLATE,
    )

    cfg = LlamaConfig(
        vocab_size=len(ftok),
        hidden_size=args.hidden,
        intermediate_size=args.hidden * 2,
        num_hidden_layers=args.layers,
        num_attention_heads=args.hidden // 64,
        num_key_value_heads=max(2, args.hidden // 128),
        max_position_embeddings=2048,
        rms_norm_eps=1e-5,
        tie_word_embeddings=True,
    )
    model = LlamaForCausalLM(cfg)
    os.makedirs(args.out, exist_ok=True)
    ftok.save_pretrained(args.out)
    model.save_pretrained(args.out)
    n_params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"tiny model -> {args.out}  ({n_params:.1f}M params, vocab={len(ftok)})")


if __name__ == "__main__":
    main()
