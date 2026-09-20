"""階段 1:LoRA SFT — 用 data/out/train.jsonl 微調基座模型。

prompt 部分 loss 設為 -100(只學 assistant 的回應),使用 HF Trainer + PEFT。

GPU 機器範例(有 GPU 的機器):
  python train/sft.py \
      --model-id IFM/K2-Horizon-0.9B --trust-remote-code \
      --train-jsonl data/out/train.jsonl --val-jsonl data/out/val.jsonl \
      --adapter-dir output/sft --epochs 3 --batch-size 4 --grad-accum 4

CPU smoke test(小模型,見 tools/smoke_tiny_model.py):
  python train/sft.py --model-id tools/tiny_model --dtype float32 \
      --train-jsonl data/out/train.jsonl --adapter-dir output/sft \
      --max-steps 20 --batch-size 2 --max-len 1024
"""
from __future__ import annotations

import argparse
import json
import os

import torch
from torch.utils.data import Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    Trainer,
    TrainingArguments,
)
from peft import LoraConfig, TaskType, get_peft_model

DTYPES = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}

# 通用 LoRA target 名字(找不到的會被丟掉)
CAND_TARGETS = (
    "q_proj", "k_proj", "v_proj", "o_proj",
    "gate_proj", "up_proj", "down_proj",
    "in_proj", "out_proj", "qkv_proj", "proj",
)


def detect_targets(model) -> list[str] | None:
    hit = set()
    for name, mod in model.named_modules():
        if isinstance(mod, torch.nn.Linear):
            last = name.split(".")[-1]
            if last in CAND_TARGETS:
                hit.add(last)
    return sorted(hit) if hit else None


def load_causal_lm(model_id: str, dtype: torch.dtype, trust_remote_code: bool):
    """兼容新版 transformers 的 dtype/torch_dtype 參數名差異。"""
    common = dict(trust_remote_code=trust_remote_code, low_cpu_mem_usage=True)
    try:
        return AutoModelForCausalLM.from_pretrained(model_id, dtype=dtype, **common)
    except TypeError:
        return AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=dtype, **common)


class SFTDataset(Dataset):
    """把 {prompt, response} 轉成 (input_ids, labels);labels 在 prompt 段設 -100。

    前提:chat template 滿足「prompt 是完整對話的前綴」(主流模型都滿足)。
    """

    def __init__(self, path: str, tokenizer, max_len: int):
        self.data: list[dict] = []
        with open(path, encoding="utf-8") as f:
            for line in f:
                self.data.append(json.loads(line))
        self.tok = tokenizer
        self.max_len = max_len
        self.pad_id = tokenizer.pad_token_id

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, i: int) -> dict:
        item = self.data[i]
        msgs = item["prompt"]
        prompt_text = self.tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        full_text = self.tok.apply_chat_template(
            msgs + [{"role": "assistant", "content": item["response"]}], tokenize=False
        )
        enc = self.tok(full_text, truncation=True, max_length=self.max_len)
        plen = self.tok(prompt_text, truncation=True, max_length=self.max_len)["input_ids"]
        labels = list(enc["input_ids"])
        n_prompt = min(len(plen), len(labels))
        labels[:n_prompt] = [-100] * n_prompt
        if self.pad_id is not None:
            for j in range(len(labels)):
                if labels[j] == self.pad_id:
                    labels[j] = -100
        return {
            "input_ids": enc["input_ids"],
            "attention_mask": enc["attention_mask"],
            "labels": labels,
        }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model-id", default="IFM/K2-Horizon-0.9B",
                    help="HF id 或本地目錄(如 tools/tiny_model)")
    ap.add_argument("--trust-remote-code", action="store_true",
                    help="K2-Horizon 等自帶 custom code 的模型需要")
    ap.add_argument("--dtype", default="bfloat16", choices=sorted(DTYPES))
    ap.add_argument("--train-jsonl", default="data/out/train.jsonl")
    ap.add_argument("--val-jsonl", default="data/out/val.jsonl")
    ap.add_argument("--adapter-dir", default="output/sft")
    ap.add_argument("--epochs", type=float, default=3)
    ap.add_argument("--max-steps", type=int, default=-1, help=">0 時覆蓋 epochs( smoke test 用)")
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--grad-accum", type=int, default=4)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--max-len", type=int, default=1024)
    ap.add_argument("--lora-r", type=int, default=16)
    ap.add_argument("--lora-alpha", type=int, default=32)
    ap.add_argument("--lora-dropout", type=float, default=0.05)
    ap.add_argument("--lora-targets", default="auto",
                    help="逗號分隔;auto=自動偵測 Q/K/V/O/MLP projections")
    ap.add_argument("--logging-steps", type=int, default=10)
    ap.add_argument("--eval-steps", type=int, default=100)
    ap.add_argument("--save-steps", type=int, default=500)
    ap.add_argument("--warmup-steps", type=int, default=50)
    ap.add_argument("--grad-ckpt", action="store_true", help="省 VRAM,速度約 -20%%")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    cuda = torch.cuda.is_available()
    if not cuda and args.dtype != "float32":
        print("[warn] 無 CUDA,建議 --dtype float32(CPU smoke)")

    print(f"loading tokenizer + model: {args.model_id}")
    tok = AutoTokenizer.from_pretrained(args.model_id, trust_remote_code=args.trust_remote_code)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    model = load_causal_lm(args.model_id, DTYPES[args.dtype], args.trust_remote_code)

    targets = (
        [t for t in args.lora_targets.split(",") if t]
        if args.lora_targets != "auto" else detect_targets(model)
    )
    print(f"LoRA targets: {targets or 'all-linear'}")
    lora_cfg = LoraConfig(
        r=args.lora_r, lora_alpha=args.lora_alpha, lora_dropout=args.lora_dropout,
        bias="none", task_type=TaskType.CAUSAL_LM,
        target_modules=targets if targets else "all-linear",
    )
    model = get_peft_model(model, lora_cfg)
    model.print_trainable_parameters()

    train_ds = SFTDataset(args.train_jsonl, tok, args.max_len)
    val_ds = SFTDataset(args.val_jsonl, tok, args.max_len) if args.val_jsonl and os.path.exists(args.val_jsonl) else None
    print(f"train={len(train_ds)}" + (f"  val={len(val_ds)}" if val_ds else ""))

    collator = DataCollatorForSeq2Seq(tokenizer=tok, padding=True, label_pad_token_id=-100)
    targs = TrainingArguments(
        output_dir=args.adapter_dir,
        num_train_epochs=args.epochs,
        max_steps=args.max_steps,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        weight_decay=0.01,
        warmup_steps=args.warmup_steps,
        lr_scheduler_type="cosine",
        logging_steps=args.logging_steps,
        eval_strategy="steps" if val_ds else "no",
        eval_steps=args.eval_steps,
        save_strategy="steps",
        save_steps=args.save_steps,
        save_total_limit=2,
        bf16=cuda,
        fp16=False,
        gradient_checkpointing=args.grad_ckpt,
        report_to=[],
        seed=args.seed,
        dataloader_num_workers=0,
        remove_unused_columns=False,
    )
    trainer = Trainer(
        model=model, args=targs,
        train_dataset=train_ds, eval_dataset=val_ds,
        data_collator=collator,
    )
    trainer.train()
    trainer.save_model(args.adapter_dir)
    with open(os.path.join(args.adapter_dir, "metadata.json"), "w", encoding="utf-8") as f:
        json.dump({"model_id": args.model_id, "dtype": args.dtype,
                   "lora": {"r": args.lora_r, "alpha": args.lora_alpha,
                            "targets": targets or "all-linear"}}, f, ensure_ascii=False, indent=2)
    print(f"adapter saved -> {args.adapter_dir}")


if __name__ == "__main__":
    main()
